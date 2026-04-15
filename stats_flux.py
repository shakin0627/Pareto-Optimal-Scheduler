import os
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
import os

load_dotenv()  

HF_TOKEN = os.getenv("HF_TOKEN")

"""
stats_flux.py  —  Offline BornSchedule statistics for FLUX flow-matching models.

Coordinate: σ ∈ [0, 1]   (σ=1 : pure noise,  σ=0 : clean data)
Linear FM interpolation:  x_σ = (1–σ)·x₀ + σ·ε
FM target velocity:       v*  = ε – x₀   (constant along true trajectory → ∇_x v* = 0)

Estimates:
  σ²_η(σ)    velocity-prediction error variance  E‖v_θ – v*‖²/d
  σ²_v̇(σ)    velocity time-derivative variance   (discretisation-error proxy)
  g(σ)        scalar Jacobian proxy (1/d)Tr(∇_x v_θ)  [optional, --no_g]
  ρ̂_η(s)     cross-step velocity-error correlation
  ρ̂_v̇(s)    cross-step velocity-acceleration correlation  [optional, --no_vdot_corr]

Output .npz keys mirror stats_cifar10.py so born_schedule.py needs minimal changes:
  lambda_grid / lambda_min / lambda_max   ← aliases for t_grid / sigma_min / sigma_max
  sigma2_values                           ← alias for sigma2_eta
  sigma2_gpp_values                       ← alias for sigma2_vdot
  g_values                                ← zeros if --no_g

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
  python stats_flux.py \\
      --model black-forest-labs/FLUX.1-dev \\
      --latent_size 64 --n_sigma 50 --n_samples 128 \\
      --n_ell_samples 64 --batch 1 --no_g
"""

from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_flux(model_name: str, device: torch.device, dtype=torch.bfloat16):
    from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    print(f"  [flux] loading transformer from {model_name} …")
    tr = FluxTransformer2DModel.from_pretrained(
        model_name, subfolder="transformer",
        torch_dtype=dtype, token=HF_TOKEN,
    ).to(device).eval()
    for m in tr.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    print(f"  [flux] loading scheduler …")
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler", token=HF_TOKEN,
    )
    return tr, sched


# ══════════════════════════════════════════════════════════════════════════════
# Latent packing / unpacking   (FLUX packs 2×2 spatial patches → C·4 channels)
# ══════════════════════════════════════════════════════════════════════════════

def pack_latents(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, 16, H, W) → (B, H//2 · W//2, 64)"""
    B = x.shape[0]
    x = x.view(B, 16, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (h // 2) * (w // 2), 64)


def unpack_latents(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, H//2 · W//2, 64) → (B, 16, H, W)"""
    B = x.shape[0]
    x = x.view(B, h // 2, w // 2, 16, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 16, h, w)


# ══════════════════════════════════════════════════════════════════════════════
# Null conditioning + position IDs
# ══════════════════════════════════════════════════════════════════════════════

def null_cond(B: int, txt_seq: int, device, dtype):
    """Zero T5 (4096-dim) + CLIP (768-dim) embeddings for unconditional stats."""
    enc  = torch.zeros(B, txt_seq, 4096, device=device, dtype=dtype)
    pool = torch.zeros(B, 768,     device=device, dtype=dtype)
    return enc, pool


def make_ids(B: int, h: int, w: int, txt_seq: int, device, dtype):
    """
    img_ids : 2-D spatial positions for FLUX RoPE (rows, cols in packed space)
    txt_ids : all-zero for null conditioning
    """
    hp, wp = h // 2, w // 2
    img = torch.zeros(hp, wp, 3, device=device, dtype=dtype)
    img[..., 1] += torch.arange(hp, device=device, dtype=dtype)[:, None]
    img[..., 2] += torch.arange(wp, device=device, dtype=dtype)[None, :]
    img = img.reshape(1, hp * wp, 3).expand(B, -1, -1)
    txt = torch.zeros(1, txt_seq, 3, device=device, dtype=dtype).expand(B, -1, -1)
    return img.contiguous(), txt.contiguous()


# ══════════════════════════════════════════════════════════════════════════════
# Velocity prediction wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _needs_guidance(tr) -> bool:
    return bool(getattr(tr.config, "guidance_embeds", False))


def flux_vel(
    tr,
    x_t: torch.Tensor,         # (B, 16, H, W)  unpacked latent
    sigma: float,               # ∈ [0, 1],  1 = pure noise
    enc: torch.Tensor,          # (B, txt_seq, 4096)
    pool: torch.Tensor,         # (B, 768)
    img_ids: torch.Tensor,      # (B, seq_img, 3)
    txt_ids: torch.Tensor,      # (B, txt_seq, 3)
    h: int, w: int,
    guidance_val: float = 3.5,
) -> torch.Tensor:
    """Returns velocity prediction (B, 16, H, W)."""
    B   = x_t.shape[0]
    dt  = x_t.dtype
    dev = x_t.device
    xp  = pack_latents(x_t, h, w)
    t   = torch.full((B,), sigma, device=dev, dtype=dt)
    g   = (torch.full((B,), guidance_val, device=dev, dtype=dt)
           if _needs_guidance(tr) else None)
    vp  = tr(hidden_states=xp, timestep=t,
              encoder_hidden_states=enc, pooled_projections=pool,
              img_ids=img_ids, txt_ids=txt_ids,
              guidance=g, return_dict=False)[0]
    return unpack_latents(vp, h, w)


# ══════════════════════════════════════════════════════════════════════════════
# σ²_η(σ)  —  velocity prediction error variance
# E‖ v_θ(x_σ, σ) – (ε – x₀) ‖² / d
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_eta(tr, sigma, x0, enc, pool, img_ids, txt_ids,
                   h, w, d, guidance_val=3.5) -> float:
    B     = x0.shape[0]
    noise = torch.randn_like(x0)
    xs    = (1.0 - sigma) * x0 + sigma * noise   # linear FM interpolant
    vstar = noise - x0                             # FM constant-velocity target
    vpred = flux_vel(tr, xs, sigma, enc, pool, img_ids, txt_ids, h, w, guidance_val)
    err   = vpred.float() - vstar.float()
    return float((err ** 2).sum()) / (B * d)


# ══════════════════════════════════════════════════════════════════════════════
# σ²_v̇(σ)  —  velocity time-derivative variance  (discretisation-error proxy)
#
# Path-coherent central-difference in σ (same x₀ and ε across lo/mid/hi)
# D(σ) ≈ E‖ (v_hi – 2v_mid + v_lo) / dσ² ‖² / d
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_vdot(tr, sigma, x0, enc, pool, img_ids, txt_ids,
                    h, w, d, dsigma=0.025, guidance_val=3.5) -> float:
    s_lo = max(0.0, sigma - dsigma)
    s_hi = min(1.0, sigma + dsigma)
    ds   = (s_hi - s_lo) / 2.0
    if ds < 1e-6:
        return 0.0
    noise = torch.randn_like(x0)
    xlo   = (1 - s_lo)   * x0 + s_lo   * noise
    xmid  = (1 - sigma)  * x0 + sigma  * noise
    xhi   = (1 - s_hi)   * x0 + s_hi   * noise
    vlo   = flux_vel(tr, xlo,  s_lo,  enc, pool, img_ids, txt_ids, h, w, guidance_val)
    vmid  = flux_vel(tr, xmid, sigma, enc, pool, img_ids, txt_ids, h, w, guidance_val)
    vhi   = flux_vel(tr, xhi,  s_hi,  enc, pool, img_ids, txt_ids, h, w, guidance_val)
    vddot = (vhi.float() - 2.0 * vmid.float() + vlo.float()) / (ds ** 2)
    return float((vddot ** 2).sum()) / (x0.shape[0] * d)


# ══════════════════════════════════════════════════════════════════════════════
# g(σ)  —  scalar Jacobian proxy  (FD-Hutchinson, pure forward passes)
#
# (1/d) Tr(∇_x v_θ)  ≈  (1/n) Σ_k  z_k^T [v(x+δz_k, σ) – v(x, σ)] / (δ·d)
# No autograd backward needed → safe for bfloat16 12 B model on 48 GB GPU.
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_g_at_sigma(tr, sigma, x0, enc1, pool1, img1, txt1,
                   h, w, d, n_probes=4, delta=0.005, guidance_val=3.5) -> float:
    """enc1/pool1/img1/txt1 must have batch size 1."""
    trace = 0.0; count = 0
    for xi in x0:
        xi    = xi.unsqueeze(0)
        noise = torch.randn_like(xi)
        xs    = (1 - sigma) * xi + sigma * noise
        v0    = flux_vel(tr, xs, sigma, enc1, pool1, img1, txt1, h, w, guidance_val)
        for _ in range(n_probes):
            z  = torch.randn_like(xs)
            vp = flux_vel(tr, xs + delta * z, sigma, enc1, pool1, img1, txt1,
                          h, w, guidance_val)
            # z^T J z  (single Rademacher probe → unbiased Tr(J))
            trace += float(((vp.float() - v0.float()) * z.float()).sum()) / delta
            count += 1
            if xs.device.type == "cuda":
                torch.cuda.empty_cache()
    return trace / (count * d)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-step velocity-error correlation  ρ̂_η(s)
# Independent ε per σ point eliminates shared-noise floor artefact.
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_eta(tr, sigma_grid, x0, enc1, pool1, img1, txt1,
                h, w, d, guidance_val=3.5):
    """
    Returns: C (raw covariance), rho (normalised),
             delta_sigmas (upper-tri pairs), rho_pairs
    """
    M   = len(sigma_grid)
    C   = np.zeros((M, M), dtype=np.float64)
    n   = 0
    for xi in x0:
        xi   = xi.unsqueeze(0)
        errs = []
        for s in sigma_grid:
            noise = torch.randn_like(xi)
            xs    = (1 - s) * xi + s * noise
            vp    = flux_vel(tr, xs, s, enc1, pool1, img1, txt1, h, w, guidance_val)
            e     = (vp.float() - (noise - xi).float()).squeeze(0).flatten().cpu().double()
            errs.append(e)
        for i in range(M):
            for j in range(i, M):
                dot = (errs[i] * errs[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n += 1
        if n % 16 == 0:
            print(f"    ρ̂_η: {n}/{len(x0)} samples")
    C  /= max(n, 1)
    diag = np.diag(C).clip(1e-30)
    rho  = np.clip(C / np.sqrt(np.outer(diag, diag)), 1e-8, 1.0)
    dl, rp = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl.append(abs(sigma_grid[i] - sigma_grid[j]))
            rp.append(float(rho[i, j]))
    return C, rho, np.array(dl), np.array(rp)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-step velocity-acceleration correlation  ρ̂_v̇(s)
# Analogous to g'' correlation in stats_cifar10.py.
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_vdot(tr, sigma_grid, x0, enc1, pool1, img1, txt1,
                 h, w, d, dsigma=0.025, guidance_val=3.5):
    M  = len(sigma_grid)
    C  = np.zeros((M, M), dtype=np.float64)
    n  = 0
    for xi in x0:
        xi    = xi.unsqueeze(0)
        vdots = []
        for s in sigma_grid:
            s_lo = max(0.0, s - dsigma)
            s_hi = min(1.0, s + dsigma)
            ds   = (s_hi - s_lo) / 2.0
            if ds < 1e-6:
                vdots.append(None)
                continue
            noise = torch.randn_like(xi)
            xlo   = (1 - s_lo) * xi + s_lo * noise
            xmid  = (1 - s)    * xi + s    * noise
            xhi   = (1 - s_hi) * xi + s_hi * noise
            vlo   = flux_vel(tr, xlo,  s_lo, enc1, pool1, img1, txt1, h, w, guidance_val)
            vmid  = flux_vel(tr, xmid, s,    enc1, pool1, img1, txt1, h, w, guidance_val)
            vhi   = flux_vel(tr, xhi,  s_hi, enc1, pool1, img1, txt1, h, w, guidance_val)
            vd    = (vhi.float() - 2*vmid.float() + vlo.float()) / (ds**2)
            vdots.append(vd.squeeze(0).flatten().cpu().double())
        for i in range(M):
            for j in range(i, M):
                if vdots[i] is None or vdots[j] is None:
                    continue
                dot = (vdots[i] * vdots[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n += 1
        if n % 8 == 0:
            print(f"    ρ̂_v̇: {n}/{len(x0)} samples")
    C  /= max(n, 1)
    diag = np.diag(C).clip(1e-30)
    rho  = np.clip(C / np.sqrt(np.outer(diag, diag)), -1.0, 1.0)
    dl, rp = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl.append(abs(sigma_grid[i] - sigma_grid[j]))
            rp.append(float(rho[i, j]))
    return C, rho, np.array(dl), np.array(rp)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions (model-agnostic, mirrored from stats_cifar10.py)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_curve(dl, rho, n_bins=40, anchor_at_one=False):
    valid = dl > 0
    if valid.sum() < 5:
        return np.linspace(0, 1, 5), np.ones(5)
    edges = np.unique(np.percentile(dl[valid], np.linspace(0, 100, n_bins + 1)))
    sl, vl = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (dl >= lo) & (dl < hi)
        if mask.sum() < 3:
            continue
        sl.append(float(np.median(dl[mask])))
        vl.append(float(np.median(rho[mask])))
    if not sl:
        return np.array([0.0, dl.max()]), np.array([1.0, 0.5])
    sa, va = np.array(sl), np.array(vl)
    if anchor_at_one:
        sa = np.concatenate([[0.0], sa])
        va = np.concatenate([[1.0], va])
    else:
        sa = np.concatenate([[0.0], sa])
        va = np.concatenate([[va[0]], va])
    log_v = np.log(np.clip(va, 1e-10, 1.0))
    for k in range(1, len(log_v)):
        if log_v[k] > log_v[k - 1]:
            log_v[k] = log_v[k - 1]
    va    = np.exp(log_v)
    pchip = PchipInterpolator(sa, va, extrapolate=False)
    so    = np.linspace(sa[0], sa[-1], 200)
    vo    = np.clip(pchip(so), va[-1], va[0])
    return so, vo


def analyse_plateau(rho_s, rho_vals, label=""):
    tail  = max(1, int(0.80 * len(rho_vals)))
    ri    = float(np.clip(np.median(rho_vals[tail:]), 0.0, 1 - 1e-6))
    rr    = np.clip(rho_vals - ri, 0.0, None)
    rn    = rr / max(rr[0], 1e-8)
    cross = np.where(rn <= 1.0 / np.e)[0]
    ell   = float(rho_s[cross[0]]) if len(cross) else float(rho_s[-1])
    sm    = float(np.trapezoid(rr, rho_s))
    fr    = ri / max(float(rho_vals[0]), 1e-8)
    print(f"\n  ── Plateau [{label}] ────────────────────────────────")
    print(f"     ρ(0)={rho_vals[0]:.4f}  ρ_∞={ri:.4f}  rank-1 frac={fr:.3f}")
    print(f"     ℓ_res={ell:.4f}  ∫ρ_res ds={sm:.4f}")
    verdict = "STRONG rank-1" if fr > 0.3 else "WEAK rank-1"
    print(f"     → {verdict}")
    return dict(rho_infty=ri, frac_rank1=fr, ell_res=ell,
                spectral_mass_res=sm, verdict=verdict)


def verify_stationarity(rho_s, rho_vals, sigma_grid, nfe_list=(4, 8, 16, 28, 50)):
    ri    = float(np.median(rho_vals[-max(5, len(rho_vals) // 5):]))
    rr    = np.clip((rho_vals - ri) / max(1 - ri, 1e-8), 0, 1)
    cross = np.where(rr <= 1.0 / np.e)[0]
    ell   = float(rho_s[cross[0]]) if len(cross) else float(rho_s[-1])
    print(f"\n  ρ_∞={ri:.4f}  ell_corr={ell:.4f}")
    print(f"  σ range = [{sigma_grid[0]:.3f}, {sigma_grid[-1]:.3f}]  (Δσ_total=1.0)")
    print(f"\n  {'NFE':>5}  {'h_typ':>8}  {'ell/h':>8}  {'banded?':>10}")
    for nfe in nfe_list:
        h  = 1.0 / nfe
        r  = ell / h
        ok = "✓ yes" if r < 0.5 else ("~ marginal" if r < 1.5 else "✗ no")
        print(f"  {nfe:>5}  {h:>8.4f}  {r:>8.3f}  {ok:>10}")
    return ri, ell


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _visualize(
    sigma_grid, g_vals, s2_eta, s2_vdot,
    dl_eta, rp_eta, rho_s_eta, rho_eta,
    dl_vdot, rp_vdot, rho_s_vd, rho_vd,
    plateau_eta, plateau_vdot,
    seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
    model_name, out_path,
):
    seg_colors = ["#e05c00", "#2d6a9f", "#2a7d4f", "#9b3fa0"]
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Row 0 ── g(σ)
    ax = fig.add_subplot(gs[0, 0])
    if g_vals is not None and g_vals.any():
        ax.plot(sigma_grid, g_vals, color="#2d6a9f", lw=2)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
    else:
        ax.text(0.5, 0.5, "g(σ) not estimated\n(use --no_g to skip)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("σ"); ax.set_ylabel("g(σ)")
    ax.set_title("Scalar Jacobian Proxy  g(σ)\n[≈0 for well-trained FM]")
    ax.grid(True, alpha=0.3)

    # Row 0 ── σ²_η and σ²_v̇
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(sigma_grid, s2_eta,  color="#b5451b", lw=2,   label="σ²_η (vel. error)")
    ax.semilogy(sigma_grid, s2_vdot, color="#9b3fa0", lw=2, ls="--", label="σ²_v̇ (disc. proxy)")
    ax.set_xlabel("σ"); ax.set_title("Error Variances σ²_η & σ²_v̇")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # Row 0 ── ρ̂_η global
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(dl_eta, rp_eta, s=4, alpha=0.2, color="#999", rasterized=True)
    ax.plot(rho_s_eta, rho_eta, color="#2d6a9f", lw=2.5, label="global ρ̂_η")
    if plateau_eta:
        ax.axhline(plateau_eta["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                   label=f"ρ_∞={plateau_eta['rho_infty']:.3f}")
    ax.set_xlabel("|Δσ|"); ax.set_ylabel("ρ̂"); ax.set_title("Velocity-Error  ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    # Row 1 ── Stationarity CV
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_common, cv_s, color="#555", lw=2)
    ax.axhline(0.15, color="#2a7d4f", lw=1.2, ls="--", label="0.15 stationary")
    ax.axhline(0.35, color="#d12",    lw=1.2, ls="--", label="0.35 non-stat.")
    vc = ("#d12"    if "STRONGLY" in stat_verdict else
          "#e08000" if "MILDLY"   in stat_verdict else "#2a7d4f")
    ax.set_title(f"Stationarity CV  mean={mean_cv:.3f}\n{stat_verdict}",
                 color=vc, fontsize=9)
    ax.set_xlabel("s"); ax.set_ylabel("CV")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Row 1 ── Per-segment ρ̂_η
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rho_s_eta, rho_eta, color="k", lw=2, ls="--", alpha=0.5, label="global")
    for k, (rs, rv) in enumerate(seg_curves):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        ax.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                label=f"σ̄∈[{lo:.2f},{hi:.2f})")
    ax.set_xlabel("|Δσ|"); ax.set_title("Per-Segment  ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    # Row 1 ── Cost integrand proxy: σ²_η(σ)·(1–σ)²
    # (1-σ)² approximates Γ²_j when g(σ)≈0, so this is ∝ per-step schedule cost)
    ax = fig.add_subplot(gs[1, 2])
    cost_proxy = s2_eta * (1.0 - sigma_grid) ** 2
    ax.plot(sigma_grid, cost_proxy, color="#2a7d4f", lw=2)
    ax.set_xlabel("σ")
    ax.set_title("Schedule Cost Proxy  σ²_η(σ)·(1–σ)²\n[optimal schedule equalises this]")
    ax.grid(True, alpha=0.3)

    # Row 2 ── ρ̂_v̇
    ax = fig.add_subplot(gs[2, 0])
    if len(dl_vdot) > 0 and len(rho_s_vd) > 1:
        ax.scatter(dl_vdot, rp_vdot, s=4, alpha=0.2, color="#999", rasterized=True)
        ax.plot(rho_s_vd, rho_vd, color="#9b3fa0", lw=2.5, label="ρ̂_v̇")
        if plateau_vdot:
            ax.axhline(plateau_vdot["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                       label=f"ρ_∞={plateau_vdot['rho_infty']:.3f}")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "v̇ correlation\nnot estimated\n(--no_vdot_corr)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("|Δσ|"); ax.set_ylabel("ρ̂_v̇")
    ax.set_title("Vel-Accel  ρ̂_v̇(s)\n(plateau → D_j has rank-1 component)")
    ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)

    # Row 2 ── Rank-1 plateau comparison
    ax = fig.add_subplot(gs[2, 1])
    labels = ["η-error ρ_∞"]
    vals   = [plateau_eta["rho_infty"] if plateau_eta else 0.0]
    if plateau_vdot:
        labels.append("v̇-accel ρ_∞")
        vals.append(plateau_vdot["rho_infty"])
    bars = ax.bar(labels, vals, color=["#2d6a9f", "#9b3fa0"][: len(labels)], width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.3, color="#e05c00", lw=1.2, ls="--", label="0.3 rank-1 threshold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("ρ_∞")
    ax.set_title("Rank-1 Plateau Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    # Row 2 ── Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    pe = plateau_eta  or {}
    pv = plateau_vdot or {}
    txt = (
        f"Model: {model_name}\n{'─'*35}\n"
        f"σ grid: [{sigma_grid[0]:.3f}, {sigma_grid[-1]:.3f}]  N={len(sigma_grid)}\n\n"
        f"Velocity-error kernel:\n"
        f"  ρ_∞       = {pe.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 f. = {pe.get('frac_rank1', float('nan')):.3f}\n"
        f"  ℓ_res     = {pe.get('ell_res', float('nan')):.4f}\n"
        f"  ∫ρ_res ds = {pe.get('spectral_mass_res', float('nan')):.4f}\n\n"
        + (
        f"v̇-accel kernel:\n"
        f"  ρ_∞       = {pv.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 f. = {pv.get('frac_rank1', float('nan')):.3f}\n"
        f"  → {pv.get('verdict','N/A')}\n\n"
        if plateau_vdot else
        "v̇ correlation: not estimated\n\n"
        )
        + f"Stationarity: {stat_verdict}\n"
          f"  mean CV = {mean_cv:.3f}\n"
    )
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaa"))
    ax.set_title("Summary", fontsize=10)

    fig.suptitle(f"BornSchedule Statistics (FM/FLUX) — {model_name}", fontsize=13, y=1.005)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main estimation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_estimation(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16
    print(f"[stats_flux] device={device}  model={args.model}")

    # ── Load ────────────────────────────────────────────────────────────────
    tr, _sched = load_flux(args.model, device, dtype)
    h, w   = args.latent_size, args.latent_size
    d      = 16 * h * w
    txt_seq = args.txt_seq
    guidance_val = args.guidance
    print(f"  latent: 16×{h}×{w}  d={d:,}  txt_seq={txt_seq}")
    print(f"  guidance_embeds: {_needs_guidance(tr)}  guidance_val: {guidance_val}")

    # ── σ grid: denser at low-σ (near-clean) and high-σ (near-noise) ────────
    # Both endpoints are where velocity errors tend to spike for FM models.
    n_lo = args.n_sigma // 3
    n_hi = args.n_sigma - n_lo
    sg   = np.concatenate([
        np.linspace(0.01, 0.35, n_lo, endpoint=False),
        np.linspace(0.35, 0.99, n_hi),
    ])
    print(f"  σ grid: {len(sg)} points  [{sg[0]:.3f}, {sg[-1]:.3f}]")

    # ── Random latent "data" (Gaussian approximation of latent distribution) ─
    print(f"  generating {args.n_samples} random latents …")
    x0_all = torch.randn(args.n_samples, 16, h, w, device=device, dtype=dtype)

    # ── Single-sample conditioning (reused for per-sample loops) ────────────
    enc1, pool1 = null_cond(1, txt_seq, device, dtype)
    img1, txt1  = make_ids(1, h, w, txt_seq, device, dtype)

    # ── σ²_η(σ) ─────────────────────────────────────────────────────────────
    print(f"\n[stats_flux] σ²_η at {len(sg)} σ points …")
    s2_eta = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb = x0_all[start : start + args.batch]
            B  = xb.shape[0]
            ec, pc = null_cond(B, txt_seq, device, dtype)
            ii, ti = make_ids(B, h, w, txt_seq, device, dtype)
            acc += est_sigma2_eta(tr, s, xb, ec, pc, ii, ti, h, w, d, guidance_val)
            nb += 1
        s2_eta[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_η  {k+1}/{len(sg)}  σ={s:.3f}  σ²={s2_eta[k]:.4e}")

    # ── σ²_v̇(σ) ──────────────────────────────────────────────────────────────
    print(f"\n[stats_flux] σ²_v̇ at {len(sg)} σ points …")
    s2_vdot = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb = x0_all[start : start + args.batch]
            B  = xb.shape[0]
            ec, pc = null_cond(B, txt_seq, device, dtype)
            ii, ti = make_ids(B, h, w, txt_seq, device, dtype)
            acc += est_sigma2_vdot(tr, s, xb, ec, pc, ii, ti, h, w, d,
                                   dsigma=args.dsigma, guidance_val=guidance_val)
            nb += 1
        s2_vdot[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_v̇  {k+1}/{len(sg)}  σ={s:.3f}  σ²={s2_vdot[k]:.4e}")

    # Smooth σ²_v̇ (central-difference amplifies noise; same Savitzky-Golay as cifar10)
    wl = min(7, (len(sg) // 4) * 2 + 1)
    log_vd       = savgol_filter(np.log(np.clip(s2_vdot, 1e-30, None)), wl, 2)
    s2_vdot_sm   = np.exp(log_vd)

    # ── g(σ) ─────────────────────────────────────────────────────────────────
    g_vals = np.zeros(len(sg), dtype=np.float32)    # zero = "not computed"
    if not args.no_g:
        n_g = min(args.n_g_samples, args.n_samples)
        print(f"\n[stats_flux] g(σ) [FD-Hutchinson, n_probes={args.n_hutchinson},"
              f" n_g_samples={n_g}] …")
        x0_g = x0_all[:n_g]
        for k, s in enumerate(sg):
            g_vals[k] = est_g_at_sigma(tr, s, x0_g, enc1, pool1, img1, txt1,
                                        h, w, d,
                                        n_probes=args.n_hutchinson,
                                        delta=args.g_delta,
                                        guidance_val=guidance_val)
            if (k + 1) % max(1, len(sg) // 8) == 0:
                print(f"  g  {k+1}/{len(sg)}  σ={s:.3f}  g={g_vals[k]:.5f}")

    # ── ρ̂_η(s) — velocity-error correlation ─────────────────────────────────
    n_ell = min(args.n_ell_samples, args.n_samples)
    print(f"\n[stats_flux] ρ̂_η  ({n_ell} samples) …")
    x0_ell = x0_all[:n_ell]
    _, rho_mat, dl_eta, rp_eta = est_rho_eta(
        tr, sg, x0_ell, enc1, pool1, img1, txt1, h, w, d, guidance_val)
    rho_s_eta, rho_eta = _extract_rho_curve(dl_eta, rp_eta, anchor_at_one=True)
    plateau_eta = analyse_plateau(rho_s_eta, rho_eta, "vel-error η")

    # ── ρ̂_v̇(s) — velocity-acceleration correlation ───────────────────────────
    dl_vdot = rp_vdot = np.array([])
    rho_s_vd = rho_vd = np.array([0.0, 1.0])
    plateau_vdot = None
    if not args.no_vdot_corr:
        print(f"\n[stats_flux] ρ̂_v̇  ({n_ell} samples, 3 NFE per σ point) …")
        _, _, dl_vdot, rp_vdot = est_rho_vdot(
            tr, sg, x0_ell, enc1, pool1, img1, txt1, h, w, d,
            dsigma=args.dsigma, guidance_val=guidance_val)
        if len(dl_vdot) > 5:
            rho_s_vd, rho_vd = _extract_rho_curve(dl_vdot, rp_vdot, anchor_at_one=False)
            plateau_vdot = analyse_plateau(rho_s_vd, rho_vd, "v̇-accel")

    # ── Stationarity ─────────────────────────────────────────────────────────
    M = len(sg)
    lbar_all = [0.5 * (sg[i] + sg[j])
                for i in range(M) for j in range(i + 1, M)]
    lbar_arr = np.array(lbar_all)

    n_segs = 3
    seg_bounds = np.linspace(0.0, 1.0, n_segs + 1)
    seg_curves = []
    print(f"\n[stats_flux] stationarity ({n_segs} σ̄-segments) …")
    for k in range(n_segs):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        mask   = (lbar_arr >= lo) & (lbar_arr < hi)
        if mask.sum() < 10:
            seg_curves.append(_extract_rho_curve(dl_eta, rp_eta))
        else:
            rs, rv = _extract_rho_curve(dl_eta[mask], rp_eta[mask], anchor_at_one=True)
            seg_curves.append((rs, rv))
            print(f"  σ̄∈[{lo:.2f},{hi:.2f})  {mask.sum()} pairs  "
                  f"ρ̂(0)={rv[0]:.3f}  ρ̂(end)={rv[-1]:.4f}")

    s_common = rho_s_eta[1:]
    seg_mats = []
    for rs, rv in seg_curves:
        pchip = PchipInterpolator(rs, rv, extrapolate=False)
        vi    = np.where(np.isnan(pchip(s_common)), rv[-1], pchip(s_common))
        seg_mats.append(np.clip(vi, 0.0, 1.0))
    seg_mat = np.array(seg_mats)
    cv_s    = seg_mat.std(0) / (seg_mat.mean(0) + 1e-8)
    valid_s = rho_eta[1:] > 0.1
    mean_cv = float(cv_s[valid_s].mean()) if valid_s.any() else float("nan")

    if   np.isnan(mean_cv):  stat_verdict = "UNKNOWN"
    elif mean_cv < 0.15:     stat_verdict = "STATIONARY"
    elif mean_cv < 0.35:     stat_verdict = "MILDLY NON-STATIONARY"
    else:                    stat_verdict = "STRONGLY NON-STATIONARY"
    print(f"  mean CV={mean_cv:.3f}  →  {stat_verdict}")

    verify_stationarity(rho_s_eta, rho_eta, sg,
                        nfe_list=(4, 8, 16, 28, 50))

    # ── Save .npz ────────────────────────────────────────────────────────────
    safe = args.model.replace("/", "--").replace(":", "--")
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz  = out / f"{safe}.npz"

    save = dict(
        # ── FM-native keys ────────────────────────────────────────────────
        t_grid            = sg.astype(np.float32),
        sigma_min         = np.float32(sg[0]),
        sigma_max         = np.float32(sg[-1]),
        sigma2_eta        = s2_eta.astype(np.float32),
        sigma2_vdot       = s2_vdot_sm.astype(np.float32),
        g_values          = g_vals.astype(np.float32),
        rho_s             = rho_s_eta.astype(np.float32),
        rho_values        = rho_eta.astype(np.float32),
        # ── born_schedule.py compatibility aliases ────────────────────────
        # (born_schedule.py reads lambda_grid / sigma2_values / sigma2_gpp_values)
        lambda_grid       = sg.astype(np.float32),
        lambda_min        = np.float32(sg[0]),
        lambda_max        = np.float32(sg[-1]),
        sigma2_values     = s2_eta.astype(np.float32),
        sigma2_gpp_values = s2_vdot_sm.astype(np.float32),
    )
    if plateau_vdot is not None:
        save["rho_s_gpp"]      = rho_s_vd.astype(np.float32)
        save["rho_values_gpp"] = rho_vd.astype(np.float32)

    np.savez(str(npz), **save)
    print(f"\n[stats_flux] saved → {npz}")

    rhomat_path = out / f"{safe}_rhomat.npy"
    np.save(str(rhomat_path), rho_mat.astype(np.float32))
    print(f"  rho_mat → {rhomat_path}")

    # ── Visualise ─────────────────────────────────────────────────────────
    vis = npz.with_suffix(".png")
    _visualize(
        sg, g_vals if g_vals.any() else None, s2_eta, s2_vdot_sm,
        dl_eta, rp_eta, rho_s_eta, rho_eta,
        dl_vdot, rp_vdot, rho_s_vd, rho_vd,
        plateau_eta, plateau_vdot,
        seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
        args.model, vis,
    )

    print("\n══ Summary ═══════════════════════════════════════════════════")
    print(f"  model          : {args.model}")
    print(f"  d              : {d:,}   (latent {16}×{h}×{w})")
    print(f"  σ grid         : [{sg[0]:.3f}, {sg[-1]:.3f}]  N={len(sg)}")
    print(f"  σ²_η range     : [{s2_eta.min():.3e}, {s2_eta.max():.3e}]")
    print(f"  σ²_v̇ range     : [{s2_vdot_sm.min():.3e}, {s2_vdot_sm.max():.3e}]")
    print(f"  ρ_∞(η)         : {plateau_eta['rho_infty']:.4f}  "
          f"(rank-1 frac {plateau_eta['frac_rank1']:.2f})")
    if plateau_vdot:
        print(f"  ρ_∞(v̇)        : {plateau_vdot['rho_infty']:.4f}  "
              f"(rank-1 frac {plateau_vdot['frac_rank1']:.2f})")
    print(f"  stationarity   : {stat_verdict}  (CV={mean_cv:.3f})")
    print(f"  output         : {npz}")
    print("═══════════════════════════════════════════════════════════════\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BornSchedule offline statistics for FLUX flow-matching models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       default="black-forest-labs/FLUX.1-dev",
                   help="HuggingFace model ID (dev or schnell)")
    p.add_argument("--output_dir",  default=str(Path.home() / ".cache" / "opt_schedule"))
    p.add_argument("--latent_size", type=int, default=64,
                   help="Latent spatial size H=W (64→512px, 128→1024px)")
    p.add_argument("--txt_seq",     type=int, default=256,
                   help="T5 token sequence length for null conditioning")
    p.add_argument("--n_sigma",     type=int, default=50,
                   help="Number of σ grid points")
    p.add_argument("--n_samples",   type=int, default=128,
                   help="Random latent samples for σ²_η, σ²_v̇")
    p.add_argument("--n_ell_samples", type=int, default=64,
                   help="Samples for ρ̂ estimation (each costs M forward passes)")
    p.add_argument("--n_g_samples",   type=int, default=32,
                   help="Samples for g(σ) FD-Hutchinson estimation")
    p.add_argument("--n_hutchinson",  type=int, default=4,
                   help="FD-Hutchinson probes per sample for g(σ)")
    p.add_argument("--g_delta",       type=float, default=0.005,
                   help="FD perturbation magnitude for g(σ)")
    p.add_argument("--dsigma",        type=float, default=0.025,
                   help="Central-difference step dσ for σ²_v̇ and ρ̂_v̇")
    p.add_argument("--guidance",      type=float, default=3.5,
                   help="Guidance scale for FLUX.1-dev (ignored by schnell)")
    p.add_argument("--batch",         type=int, default=1,
                   help="Batch size for σ²_η / σ²_v̇ (≥2 needs more VRAM)")
    p.add_argument("--device",        default=None,
                   help="e.g. cuda:0  (set CUDA_DEVICE_ORDER=PCI_BUS_ID first)")
    p.add_argument("--no_g",          action="store_true",
                   help="Skip g(σ) estimation (recommended: 12B model, Tr(J)≈0 for FM)")
    p.add_argument("--no_vdot_corr",  action="store_true",
                   help="Skip v̇ cross-step correlation (saves 3× model calls per sample)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_estimation(args)