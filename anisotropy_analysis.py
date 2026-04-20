"""
anisotropy_utils.py
===================
Drop-in replacements for est_g_at_sigma / est_g_at_t that additionally
return (g_iso, g_sem, anisotropy_ratio), plus a publication-quality
figure generator for Experiments III-1 and V-1.

Usage
-----
# In stats_sd3.py  — replace est_g_at_sigma with est_g_aniso_sd3
# In stats_sd15.py — replace est_g_at_t    with est_g_aniso_sd15

# Standalone figure (after running stats):
python anisotropy_utils.py \
    --flux   path/to/flux_stats.npz \
    --sd3    path/to/sd3_stats.npz  \
    --sd15   path/to/sd15_stats.npz \
    --out    figures/anisotropy.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# SD3 / CFG  —  adds semantic direction probe to existing est_g_at_sigma
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def est_g_aniso_sd3(
    tr,
    sigma: float,
    x0_list,
    bank,                     # PromptBankSD3
    h: int,
    w: int,
    d: int,
    cfg_scale: float,
    n_probes: int = 4,
    delta: float = 0.005,
) -> tuple[float, float, float]:
    """
    Returns (g_iso, g_sem, anisotropy_ratio) for SD3 / CFG models.

    g_iso = (1/d) Tr(∇_x v̂_CFG)          isotropic FD-Hutchinson
    g_sem = (1/d) u^T ∇_x v̂_CFG u        along CFG semantic direction
            u = (v_cond – v_null) / ‖·‖

    Forward budget per sample: 2 + n_probes + 2 = n_probes + 4
    """
    from .stats_sd3 import single_vel   # adjust import to your package layout

    trace_iso = 0.0
    trace_sem = 0.0
    count_iso = 0
    count_sem = 0

    for xi in x0_list:
        xi = xi.unsqueeze(0) if xi.dim() == 3 else xi
        enc_cond, pool_cond = bank.sample(1)
        enc_null, pool_null = bank.null(1)
        noise = torch.randn_like(xi)
        xs    = (1.0 - sigma) * xi + sigma * noise

        # baseline velocities
        v0_null = single_vel(tr, xs, sigma, enc_null, pool_null, h, w)
        v0_cond = single_vel(tr, xs, sigma, enc_cond, pool_cond, h, w)
        v0_hat  = (1.0 - cfg_scale) * v0_null + cfg_scale * v0_cond

        # semantic direction u
        u_raw = (cfg_scale * (v0_cond.float() - v0_null.float()))
        u_norm = u_raw.norm()
        if u_norm < 1e-8:
            u_unit = torch.randn_like(u_raw)
            u_unit = u_unit / (u_unit.norm() + 1e-8)
        else:
            u_unit = u_raw / u_norm

        # isotropic probes
        for _ in range(n_probes):
            z      = torch.randn_like(xs)
            xs_p   = xs + delta * z
            vp_null = single_vel(tr, xs_p, sigma, enc_null, pool_null, h, w)
            vp_cond = single_vel(tr, xs_p, sigma, enc_cond, pool_cond, h, w)
            vp_hat  = (1.0 - cfg_scale) * vp_null + cfg_scale * vp_cond
            trace_iso += float(
                ((vp_hat.float() - v0_hat.float()) * z.float()).sum()
            ) / delta
            count_iso += 1
            if xi.device.type == "cuda":
                torch.cuda.empty_cache()

        # semantic directional probe
        xs_u    = xs + delta * u_unit
        vp_null_u = single_vel(tr, xs_u, sigma, enc_null, pool_null, h, w)
        vp_cond_u = single_vel(tr, xs_u, sigma, enc_cond, pool_cond, h, w)
        vp_hat_u  = (1.0 - cfg_scale) * vp_null_u + cfg_scale * vp_cond_u
        dv        = (vp_hat_u.float() - v0_hat.float()) / delta
        trace_sem += float((dv * u_unit).sum())
        count_sem += 1
        if xi.device.type == "cuda":
            torch.cuda.empty_cache()

    g_iso  = trace_iso / (count_iso * d)
    g_sem  = trace_sem / (count_sem * d)
    ratio  = g_sem / (abs(g_iso) + 1e-8)
    return g_iso, g_sem, ratio


# ──────────────────────────────────────────────────────────────────────────────
# SD1.5 / VP-SDE + CFG  —  adds semantic direction probe to est_g_at_t
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def est_g_aniso_sd15(
    unet,
    t_idx: int,
    alpha_t: float,
    sigma_t: float,
    d: int,
    x0_list,
    bank,                     # PromptBank (sd15)
    cfg_scale: float,
    n_probes: int = 4,
    delta: float = 0.01,
) -> tuple[float, float, float]:
    """
    Returns (g_iso, g_sem, anisotropy_ratio) for SD1.5 / VP-SDE + CFG.

    g_iso = (1/d) Tr(∇_x ε_θ^CFG)
    g_sem = (1/d) u^T ∇_x ε_θ^CFG u
            u = CFG-amplified noise direction = w*(ε_cond-ε_null)/‖·‖
    """
    def _cfg(x, enc_null, enc_cond):
        e_null = unet(x, torch.full((x.shape[0],), t_idx,
                                    device=x.device, dtype=torch.long),
                      encoder_hidden_states=enc_null, return_dict=False)[0]
        e_cond = unet(x, torch.full((x.shape[0],), t_idx,
                                    device=x.device, dtype=torch.long),
                      encoder_hidden_states=enc_cond, return_dict=False)[0]
        return e_null + cfg_scale * (e_cond - e_null)

    trace_iso = 0.0
    trace_sem = 0.0
    count_iso = 0
    count_sem = 0

    for xi in x0_list:
        xi       = xi.unsqueeze(0)
        enc_cond = bank.sample(1)
        enc_null = bank.null(1)
        eps_n    = torch.randn_like(xi)
        x_t      = alpha_t * xi + sigma_t * eps_n

        v0_null = unet(x_t, torch.full((1,), t_idx, device=x_t.device,
                                        dtype=torch.long),
                       encoder_hidden_states=enc_null, return_dict=False)[0]
        v0_cond = unet(x_t, torch.full((1,), t_idx, device=x_t.device,
                                        dtype=torch.long),
                       encoder_hidden_states=enc_cond, return_dict=False)[0]
        v0_cfg  = v0_null + cfg_scale * (v0_cond - v0_null)

        # semantic direction: CFG amplification direction
        u_raw  = cfg_scale * (v0_cond.float() - v0_null.float())
        u_norm = u_raw.norm()
        if u_norm < 1e-8:
            u_unit = torch.randn_like(u_raw)
            u_unit = u_unit / (u_unit.norm() + 1e-8)
        else:
            u_unit = u_raw / u_norm

        # isotropic probes
        for _ in range(n_probes):
            z      = torch.randn_like(x_t)
            x_t_p  = x_t + delta * z
            vp_cfg = _cfg(x_t_p, enc_null, enc_cond)
            trace_iso += float(
                ((vp_cfg.float() - v0_cfg.float()) * z.float()).sum()
            ) / delta
            count_iso += 1
            if xi.device.type == "cuda":
                torch.cuda.empty_cache()

        # semantic probe
        x_t_u  = x_t + delta * u_unit
        vp_cfg_u = _cfg(x_t_u, enc_null, enc_cond)
        dv     = (vp_cfg_u.float() - v0_cfg.float()) / delta
        trace_sem += float((dv * u_unit).sum())
        count_sem += 1
        if xi.device.type == "cuda":
            torch.cuda.empty_cache()

    g_iso  = trace_iso / (count_iso * d)
    g_sem  = trace_sem / (count_sem * d)
    ratio  = g_sem / (abs(g_iso) + 1e-8)
    return g_iso, g_sem, ratio


# ──────────────────────────────────────────────────────────────────────────────
# npz key names (consistent across all stats scripts)
# ──────────────────────────────────────────────────────────────────────────────
#
# Add to save_dict in each stats script:
#   g_iso_values   = g_iso_arr.astype(np.float32)
#   g_sem_values   = g_sem_arr.astype(np.float32)
#   g_ratio_values = g_ratio_arr.astype(np.float32)
#
# t_grid / sigma_grid key: use existing "lambda_grid" or "t_grid" per script


# ──────────────────────────────────────────────────────────────────────────────
# Publication-quality figure
# ──────────────────────────────────────────────────────────────────────────────

def _load_aniso(npz_path: str) -> dict | None:
    """Load anisotropy arrays from a stats npz, return None if keys missing."""
    data = np.load(npz_path)
    grid_key = "lambda_grid" if "lambda_grid" in data else "t_grid"
    needed = ["g_iso_values", "g_sem_values", "g_ratio_values"]
    if not all(k in data for k in needed):
        return None
    return {
        "grid":  data[grid_key].astype(np.float64),
        "g_iso": data["g_iso_values"].astype(np.float64),
        "g_sem": data["g_sem_values"].astype(np.float64),
        "ratio": data["g_ratio_values"].astype(np.float64),
    }


def plot_anisotropy(
    model_npzs: dict[str, str],   # {"FLUX-dev": "path.npz", "SD3.5": ..., "SD v1.5": ...}
    out_path: str,
    sigma_space: bool = True,      # True  → x-axis is σ (FM models)
                                   # False → x-axis is λ_sched (VP models)
    figsize: tuple = (14, 9),
    dpi: int = 300,
) -> None:
    """
    Two-panel publication figure:
      Left  : g_iso and g_sem curves per model
      Right : anisotropy_ratio = g_sem/g_iso per model

    Saves to out_path (.pdf or .png).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.signal import savgol_filter

    # ── aesthetics ─────────────────────────────────────────────────────────
    PALETTE = {
        "FLUX-dev":  "#2d6a9f",
        "SD3.5":     "#e05c00",
        "SD v1.5":   "#2a7d4f",
    }
    LS_ISO = "-"
    LS_SEM = "--"
    LW     = 2.0

    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.labelsize":   12,
        "axes.titlesize":   12,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_g, ax_r = axes

    def _smooth(y, win=7):
        if len(y) < win:
            return y
        w = win if win % 2 == 1 else win + 1
        w = min(w, len(y) // 2 * 2 - 1)
        return savgol_filter(y, w, 2) if w >= 3 else y

    for label, path in model_npzs.items():
        d = _load_aniso(path)
        if d is None:
            print(f"  [warn] {label}: g_iso/g_sem not found in {path}, skipping.")
            continue

        grid  = d["grid"]
        g_iso = _smooth(d["g_iso"])
        g_sem = _smooth(d["g_sem"])
        ratio = _smooth(d["ratio"])
        col   = PALETTE.get(label, "#555555")

        ax_g.plot(grid, g_iso, color=col, ls=LS_ISO, lw=LW,
                  label=f"{label} $g_{{\\mathrm{{iso}}}}$")
        ax_g.plot(grid, g_sem, color=col, ls=LS_SEM, lw=LW,
                  label=f"{label} $g_{{\\mathrm{{sem}}}}$")
        ax_r.plot(grid, ratio, color=col, lw=LW, label=label)

    # ── left panel ─────────────────────────────────────────────────────────
    ax_g.axhline(0, color="gray", lw=0.7, ls=":")
    if sigma_space:
        ax_g.set_xlabel(r"Noise level $\sigma$")
        ax_g.invert_xaxis()
    else:
        ax_g.set_xlabel(r"Log-SNR $\lambda_{\mathrm{sched}}$")
    ax_g.set_ylabel(r"Scalar Jacobian proxy $g$")
    ax_g.set_title(r"Isotropic $g_{\mathrm{iso}}$ vs Semantic $g_{\mathrm{sem}}$")
    ax_g.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_g.grid(True, alpha=0.25, linewidth=0.5)

    # custom legend: one entry per model for iso/sem linestyle
    handles_g, labels_g = ax_g.get_legend_handles_labels()
    ax_g.legend(handles_g, labels_g, ncol=1,
                frameon=False, loc="upper left")

    # ── right panel ────────────────────────────────────────────────────────
    ax_r.axhline(1.0, color="gray", lw=1.0, ls="--",
                 label="Ratio = 1  (isotropic)")
    if sigma_space:
        ax_r.set_xlabel(r"Noise level $\sigma$")
        ax_r.invert_xaxis()
    else:
        ax_r.set_xlabel(r"Log-SNR $\lambda_{\mathrm{sched}}$")
    ax_r.set_ylabel(r"Anisotropy ratio $g_{\mathrm{sem}} / g_{\mathrm{iso}}$")
    ax_r.set_title(r"Anisotropy Ratio across Models")
    ax_r.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax_r.grid(True, alpha=0.25, linewidth=0.5)
    ax_r.legend(frameon=False, loc="upper left")

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[anisotropy] figure saved → {out_path}")


def plot_g_iso_sem_separate(
    model_npzs: dict[str, str],
    out_path: str,
    figsize: tuple = (18, 5),
    dpi: int = 300,
) -> None:
    """
    Three-panel figure for Experiment III-1 (distilled vs CFG comparison):
      Panel 1: g_iso across models
      Panel 2: g_sem across models
      Panel 3: ratio across models
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    PALETTE = {
        "FLUX-dev":  "#2d6a9f",
        "SD3.5":     "#e05c00",
        "SD v1.5":   "#2a7d4f",
    }
    LW = 2.0

    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.labelsize":    12,
        "axes.titlesize":    12,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_iso, ax_sem, ax_rat = axes
    titles = [
        r"(a) Isotropic $g_{\mathrm{iso}}$",
        r"(b) Semantic $g_{\mathrm{sem}}$",
        r"(c) Anisotropy ratio",
    ]
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.axhline(0, color="gray", lw=0.6, ls=":")

    def _smooth(y):
        if len(y) < 7: return y
        return savgol_filter(y, min(7, len(y)//2*2-1), 2)

    for label, path in model_npzs.items():
        d = _load_aniso(path)
        if d is None:
            continue
        grid  = d["grid"]
        col   = PALETTE.get(label, "#555555")
        ax_iso.plot(grid, _smooth(d["g_iso"]), color=col, lw=LW, label=label)
        ax_sem.plot(grid, _smooth(d["g_sem"]), color=col, lw=LW, label=label)
        ax_rat.plot(grid, _smooth(d["ratio"]), color=col, lw=LW, label=label)

    ax_rat.axhline(1.0, color="gray", lw=1.0, ls="--", label="= 1")

    # shared x-label heuristic: use σ if grid ∈ [0,1], else λ
    for ax in axes:
        if all(0 <= np.load(list(model_npzs.values())[0])["t_grid" if "t_grid" in
               np.load(list(model_npzs.values())[0]) else "lambda_grid"] <= 1):
            ax.set_xlabel(r"$\sigma$")
            ax.invert_xaxis()
        else:
            ax.set_xlabel(r"$\lambda_{\mathrm{sched}}$")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Value")
    axes[2].set_ylabel(r"$g_{\mathrm{sem}} / g_{\mathrm{iso}}$")

    # annotation: shade "distilled" vs "CFG" regions in ratio panel
    ax_rat.axhline(1.0, color="gray", lw=0.8, ls="--")

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[anisotropy] 3-panel figure saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Integration snippet  (paste into each stats script)
# ──────────────────────────────────────────────────────────────────────────────

INTEGRATION_NOTE = """
# ── In your stats loop, replace the g estimation call with: ──────────────────

# For SD3 / stats_sd3.py:
#   from anisotropy_utils import est_g_aniso_sd3
#   g_iso_arr   = np.zeros(len(sg))
#   g_sem_arr   = np.zeros(len(sg))
#   g_ratio_arr = np.zeros(len(sg))
#   for k, s in enumerate(sg):
#       g_iso_arr[k], g_sem_arr[k], g_ratio_arr[k] = est_g_aniso_sd3(
#           tr, s, x0_g, bank, h, w, d, cfg_scale,
#           n_probes=args.n_hutchinson, delta=args.g_delta)

# For SD1.5 / stats_sd15.py:
#   from anisotropy_utils import est_g_aniso_sd15
#   g_iso_arr   = np.zeros(n_lam)
#   g_sem_arr   = np.zeros(n_lam)
#   g_ratio_arr = np.zeros(n_lam)
#   for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
#       g_iso_arr[k], g_sem_arr[k], g_ratio_arr[k] = est_g_aniso_sd15(
#           unet, t_idx, float(alphas[t_idx]), float(sigmas[t_idx]),
#           d, x0_g, bank, args.cfg_scale,
#           n_probes=args.n_hutchinson, delta=args.g_delta)

# ── Then add to save_dict: ───────────────────────────────────────────────────
#   save["g_iso_values"]   = g_iso_arr.astype(np.float32)
#   save["g_sem_values"]   = g_sem_arr.astype(np.float32)
#   save["g_ratio_values"] = g_ratio_arr.astype(np.float32)
#   # keep g_values = g_iso_arr for backwards compatibility with born_schedule.py
#   save["g_values"]       = g_iso_arr.astype(np.float32)
"""


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Plot anisotropy figures for BornSchedule paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--flux",  default=None, help="FLUX-dev stats npz")
    p.add_argument("--sd3",   default=None, help="SD3.5 stats npz")
    p.add_argument("--sd15",  default=None, help="SD v1.5 stats npz")
    p.add_argument("--out",   default="figures/anisotropy.pdf")
    p.add_argument("--three_panel", action="store_true",
                   help="Use three-panel layout (Exp III-1) instead of two-panel (Exp V-1)")
    p.add_argument("--sigma_space", action="store_true",
                   help="x-axis is σ (FM models); default is λ_sched (VP models)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    npzs = {}
    if args.flux:  npzs["FLUX-dev"] = args.flux
    if args.sd3:   npzs["SD3.5"]    = args.sd3
    if args.sd15:  npzs["SD v1.5"]  = args.sd15

    if not npzs:
        print("No npz files provided.  Use --flux / --sd3 / --sd15.")
        raise SystemExit(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.three_panel:
        plot_g_iso_sem_separate(npzs, args.out, figsize=(18, 5))
    else:
        plot_anisotropy(npzs, args.out, sigma_space=args.sigma_space)

    print(INTEGRATION_NOTE)