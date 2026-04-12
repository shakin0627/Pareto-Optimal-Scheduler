import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
import os

load_dotenv()  

HF_TOKEN = os.getenv("HF_TOKEN")
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator

# ─────────────────────────────────────────────────────────────────────────────
# Noise-schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def alphas_from_scheduler(model_name: str, device: torch.device):
    sched = DDPMScheduler.from_pretrained(model_name, subfolder=None)
    if hasattr(sched, "alphas_cumprod"):
        acp = sched.alphas_cumprod.to(device).double()
    else:
        betas = sched.betas.to(device).double()
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
    alpha = acp.sqrt()
    sigma = (1.0 - acp).sqrt()
    lambdas = torch.log(alpha / sigma.clamp(min=1e-12))
    return acp, alpha, sigma, lambdas


def lambda_to_t(lambda_val: float, lambdas_dense: torch.Tensor) -> int:
    idx = (lambdas_dense - lambda_val).abs().argmin().item()
    return int(idx)


# ─────────────────────────────────────────────────────────────────────────────
# UNet loading  (pixel-space, unconditional — google/ddpm-cifar10-32 style)
# ─────────────────────────────────────────────────────────────────────────────

def load_unet(model_name: str, device: torch.device, latent: bool):
    if latent:
        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet",
            torch_dtype=torch.float32,
        ).to(device)
    else:
        unet = UNet2DModel.from_pretrained(
            model_name, torch_dtype=torch.float32,
        ).to(device)
    unet.eval()
    for m in unet.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return unet


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass abstraction
# ─────────────────────────────────────────────────────────────────────────────

def unet_eps(unet, x_t: torch.Tensor, t: torch.Tensor, latent: bool) -> torch.Tensor:
    if latent:
        B = x_t.shape[0]
        cross_attn_dim = unet.config.cross_attention_dim
        null_emb = torch.zeros(
            (B, 77, cross_attn_dim),
            device=x_t.device, dtype=x_t.dtype
        )
        out = unet(x_t, t, encoder_hidden_states=null_emb, return_dict=False)
        return out[0]
    else:
        # pixel-space unconditional (e.g. google/ddpm-cifar10-32)
        out = unet(x_t, t)
        return out.sample if hasattr(out, "sample") else out


# ─────────────────────────────────────────────────────────────────────────────
# g(λ) — Hutchinson trace estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_g_at_t(
    unet, t_idx, alpha_t, sigma_t, d, x0_batch, n_probes, latent, device,
) -> float:
    t_tensor = torch.full((1,), t_idx, device=device, dtype=torch.long)
    trace_acc = 0.0
    count = 0

    for xi in x0_batch:
        xi = xi.unsqueeze(0)
        for _ in range(n_probes):
            eps = torch.randn_like(xi)
            x_t = alpha_t * xi + sigma_t * eps
            v = torch.randint(0, 2, x_t.shape, device=device, dtype=x_t.dtype) * 2 - 1
            x_t_req = x_t.detach().requires_grad_(True)
            noise_pred = unet_eps(unet, x_t_req, t_tensor, latent)
            JTv = torch.autograd.grad(
                (noise_pred * v).sum(), x_t_req,
                create_graph=False, retain_graph=False,
            )[0]
            vJv = (JTv * v).sum().item()
            trace_acc += vJv
            count += 1
            del x_t_req, noise_pred, JTv
            torch.cuda.empty_cache() if device.type == "cuda" else None

    return trace_acc / (count * d)


# ─────────────────────────────────────────────────────────────────────────────
# σ²_η(λ) — score approximation error variance
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma2_at_t(
    unet, t_idx, alpha_t, sigma_t, d, x0_batch, latent, device,
) -> float:
    B = x0_batch.shape[0]
    t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
    eps = torch.randn_like(x0_batch)
    x_t = alpha_t * x0_batch + sigma_t * eps
    noise_pred = unet_eps(unet, x_t, t_tensor, latent)
    mse = F.mse_loss(noise_pred, eps, reduction="sum").item()
    return mse / (B * d)


# ─────────────────────────────────────────────────────────────────────────────
# σ²_{g''}(λ) — discretization error variance
# (1/d) E[||∂²_λ [e^{-λ} ε_θ]||²]
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma2_gpp_at_t(
    unet, t_idx, lambdas_dense, alphas, sigmas, d,
    x0_batch, latent, device, delta_t=2,
) -> float:
    """
    σ²_g''(λ) = (1/d) E[||∂²_λ [e^{-λ} ε_θ]||²]

    Three-point central difference on g(λ) = e^{-λ}ε_θ.
    Same eps across lo/mid/hi — path-coherent derivative estimate.
    """
    B = x0_batch.shape[0]
    t_lo  = max(0, t_idx - delta_t)
    t_hi  = min(len(lambdas_dense) - 1, t_idx + delta_t)
    t_mid = t_idx

    lam_lo  = float(lambdas_dense[t_lo])
    lam_hi  = float(lambdas_dense[t_hi])
    lam_mid = float(lambdas_dense[t_mid])
    h = (lam_hi - lam_lo) / 2

    if abs(h) < 1e-8 or t_lo == t_hi:
        return 0.0

    t_lo_t  = torch.full((B,), t_lo,  device=device, dtype=torch.long)
    t_mid_t = torch.full((B,), t_mid, device=device, dtype=torch.long)
    t_hi_t  = torch.full((B,), t_hi,  device=device, dtype=torch.long)

    eps = torch.randn_like(x0_batch)
    x_lo  = float(alphas[t_lo])  * x0_batch + float(sigmas[t_lo])  * eps
    x_mid = float(alphas[t_mid]) * x0_batch + float(sigmas[t_mid]) * eps
    x_hi  = float(alphas[t_hi])  * x0_batch + float(sigmas[t_hi])  * eps

    g_lo  = np.exp(-lam_lo)  * unet_eps(unet, x_lo,  t_lo_t,  latent)
    g_mid = np.exp(-lam_mid) * unet_eps(unet, x_mid, t_mid_t, latent)
    g_hi  = np.exp(-lam_hi)  * unet_eps(unet, x_hi,  t_hi_t,  latent)

    g_pp = (g_hi - 2.0 * g_mid + g_lo) / (h ** 2)
    return (g_pp ** 2).sum().item() / (B * d)


# ─────────────────────────────────────────────────────────────────────────────
# g''(λ) cross-step correlation  ρ_gpp(|Δλ|)
#
# Motivation: if D_j = Γ_j² · K_gpp · h^6 has the same rank-1 structure as
# the score-error kernel, we can factor out a "discretization rank-1 term"
# analogous to ρ_∞(Σ A_j Γ_j)² in the current cost functional.
#
# ρ_gpp(|λ_i − λ_j|) = E[<g''(λ_i), g''(λ_j)>] / sqrt(σ²_gpp(λ_i) · σ²_gpp(λ_j))
#
# Estimation uses the SAME eps across λ (path-coherent) so that the
# cross-step inner product is unbiased under the denoising objective.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_gpp_correlation(
    unet,
    t_indices: np.ndarray,
    lambdas_dense: torch.Tensor,
    alphas: torch.Tensor,
    sigmas: torch.Tensor,
    d: int,
    x0_batch: torch.Tensor,
    latent: bool,
    device: torch.device,
    delta_t: int = 2,
) -> tuple:
    """
    Returns:
        gpp_corr_mat  : (M, M) normalized correlation matrix of g'' vectors
        delta_lams_gpp: pairwise |Δλ| values (upper triangle)
        rho_gpp_pairs : corresponding correlation values
    """
    M = len(t_indices)
    B = x0_batch.shape[0]

    # Accumulate cross-step dot products:  C[i,j] = (1/d) E[<g''(λ_i), g''(λ_j)>]
    C = np.zeros((M, M), dtype=np.float64)
    n_batches = 0

    for xi in x0_batch:
        xi = xi.unsqueeze(0)  # [1, C, H, W]

        # Compute g'' vectors at each λ for this sample
        gpp_vecs = []
        for m, t_idx in enumerate(t_indices):
            t_lo  = max(0, t_idx - delta_t)
            t_hi  = min(len(lambdas_dense) - 1, t_idx + delta_t)
            t_mid = t_idx

            lam_lo  = float(lambdas_dense[t_lo])
            lam_hi  = float(lambdas_dense[t_hi])
            lam_mid = float(lambdas_dense[t_mid])
            h_fd    = (lam_hi - lam_lo) / 2.0

            if abs(h_fd) < 1e-8 or t_lo == t_hi:
                gpp_vecs.append(None)
                continue

            t_lo_t  = torch.full((1,), t_lo,  device=device, dtype=torch.long)
            t_mid_t = torch.full((1,), t_mid, device=device, dtype=torch.long)
            t_hi_t  = torch.full((1,), t_hi,  device=device, dtype=torch.long)

            # Same eps as path-coherent — cross-step correlation is meaningful
            eps = torch.randn_like(xi)
            x_lo  = float(alphas[t_lo])  * xi + float(sigmas[t_lo])  * eps
            x_mid = float(alphas[t_mid]) * xi + float(sigmas[t_mid]) * eps
            x_hi  = float(alphas[t_hi])  * xi + float(sigmas[t_hi])  * eps

            g_lo  = np.exp(-lam_lo)  * unet_eps(unet, x_lo,  t_lo_t,  latent)
            g_mid = np.exp(-lam_mid) * unet_eps(unet, x_mid, t_mid_t, latent)
            g_hi  = np.exp(-lam_hi)  * unet_eps(unet, x_hi,  t_hi_t,  latent)

            gpp = (g_hi - 2.0 * g_mid + g_lo) / (h_fd ** 2)
            gpp_vecs.append(gpp.squeeze(0).flatten().cpu().double())

        # Accumulate cross-products
        for i in range(M):
            for j in range(i, M):
                if gpp_vecs[i] is None or gpp_vecs[j] is None:
                    continue
                dot = (gpp_vecs[i] * gpp_vecs[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n_batches += 1

    C /= max(n_batches, 1)

    # Normalize to correlation
    diag = np.diag(C).clip(min=1e-30)
    gpp_corr_mat = C / np.sqrt(np.outer(diag, diag))
    gpp_corr_mat = np.clip(gpp_corr_mat, -1.0, 1.0)

    # Extract pairwise (|Δλ|, ρ_gpp) for curve fitting
    lambdas_grid_np = lambdas_dense.cpu().numpy()[t_indices]
    delta_lams_gpp, rho_gpp_pairs = [], []
    for i in range(M):
        for j in range(i + 1, M):
            delta_lams_gpp.append(abs(lambdas_grid_np[i] - lambdas_grid_np[j]))
            rho_gpp_pairs.append(float(gpp_corr_mat[i, j]))

    return (gpp_corr_mat,
            np.array(delta_lams_gpp, dtype=np.float64),
            np.array(rho_gpp_pairs,  dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Score-error correlation matrix estimation
# (independent ε per λ — eliminates shared-ε floor artefact)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_ell(
    unet,
    t_indices: np.ndarray,
    alphas: torch.Tensor,
    sigmas: torch.Tensor,
    d: int,
    x0_batch: torch.Tensor,
    latent: bool,
    device: torch.device,
    min_delta_lambda: float = 0.05,
) -> tuple:
    """
    Estimate K_ξ(λ_i, λ_j) = (1/d) E[<ξ(λ_i), ξ(λ_j)>].

    Independent ε per λ eliminates the shared-ε bias term:
        E[e_i · e_j] = K_ξ(i,j)  (no floor artefact)
    because E[ε*(λ,x_t) − ε] = 0 by DSM and ε_i ⊥ ε_j.
    """
    M = len(t_indices)
    B = x0_batch.shape[0]

    C = np.zeros((M, M), dtype=np.float64)
    n_batches = 0

    for xi in x0_batch:
        xi = xi.unsqueeze(0)

        errors = []
        for m, t_idx in enumerate(t_indices):
            alpha_t  = float(alphas[t_idx])
            sigma_t  = float(sigmas[t_idx])
            t_tensor = torch.full((1,), t_idx, device=device, dtype=torch.long)

            eps_m = torch.randn_like(xi)          # fresh independent noise
            x_t   = alpha_t * xi + sigma_t * eps_m
            noise_pred = unet_eps(unet, x_t, t_tensor, latent)

            e = (noise_pred - eps_m).squeeze(0).flatten().cpu().double()
            errors.append(e)

        for i in range(M):
            for j in range(i, M):
                dot = (errors[i] * errors[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n_batches += 1

    C /= n_batches

    diag = np.diag(C).clip(min=1e-30)
    rho  = C / np.sqrt(np.outer(diag, diag))
    rho  = np.clip(rho, 1e-8, 1.0)

    delta_lams, log_corrs = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl = abs(t_indices[i] - t_indices[j])
            delta_lams.append(dl)
            log_corrs.append(np.log(rho[i, j]))

    return C, rho, np.array(delta_lams, dtype=np.float64), np.array(log_corrs, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# ρ curve extraction  (shared by score-error and g'' correlations)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rho_curve(dl, rho, n_bins=40, anchor_at_one=True):
    """
    Bin (|Δλ|, ρ) pairs by percentile, take median per bin,
    enforce log-monotone decay, PCHIP interpolate.

    anchor_at_one: if True, prepend (0, 1.0) — correct for score-error corr.
                   if False, prepend (0, first-bin median) — for g'' corr
                   where ρ_gpp(0) is not necessarily 1.
    """
    valid = dl > 0
    if valid.sum() < 5:
        s = np.linspace(0, 1, 5)
        return s, np.ones(5)

    edges = np.unique(np.percentile(dl[valid], np.linspace(0, 100, n_bins + 1)))
    s_list, v_list = [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (dl >= lo) & (dl < hi)
        if mask.sum() < 3:
            continue
        s_list.append(float(np.median(dl[mask])))
        v_list.append(float(np.median(rho[mask])))

    if len(s_list) == 0:
        return np.array([0.0, dl.max()]), np.array([1.0, 0.5])

    s_arr = np.array(s_list)
    v_arr = np.array(v_list)

    # Anchor
    if anchor_at_one:
        s_arr = np.concatenate([[0.0], s_arr])
        v_arr = np.concatenate([[1.0], v_arr])
    else:
        s_arr = np.concatenate([[0.0], s_arr])
        v_arr = np.concatenate([[v_arr[0]], v_arr])   # don't force 1.0

    # Enforce log-monotone (isotonic in log space)
    log_v = np.log(np.clip(v_arr, 1e-10, 1.0))
    for k in range(1, len(log_v)):
        if log_v[k] > log_v[k - 1]:
            log_v[k] = log_v[k - 1]
    v_arr = np.exp(log_v)

    pchip = PchipInterpolator(s_arr, v_arr, extrapolate=False)
    s_out = np.linspace(s_arr[0], s_arr[-1], 200)
    v_out = pchip(s_out)
    v_out = np.clip(v_out, v_arr[-1], v_arr[0])
    return s_out, v_out


# ─────────────────────────────────────────────────────────────────────────────
# Plateau / rank-1 eigenvalue analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_plateau(rho_s, rho_vals, label="score-error"):
    """
    Quantify the rank-1 (plateau) component of the kernel.

    The kernel K_ξ(s) = ρ(s) decomposes as:
        ρ(s) = ρ_∞  +  ρ_res(s)
    where ρ_∞ = plateau level and ρ_res decays to 0.

    'Eigenvalue' interpretation:
      If K acts on functions over [λ_min, λ_max] with uniform measure L,
        λ_rank1  ≈ ρ_∞ · L       (rank-1 eigenvalue)
        λ_res    ≈ ∫ ρ_res(s) ds  (residual spectral mass)
      Fraction of variance in rank-1 mode = ρ_∞ / ρ(0).

    Returns dict with key metrics.
    """
    tail_start  = max(1, int(0.80 * len(rho_vals)))
    rho_infty   = float(np.median(rho_vals[tail_start:]))
    rho_infty   = np.clip(rho_infty, 0.0, 1.0 - 1e-6)

    rho_res     = np.clip(rho_vals - rho_infty, 0.0, None)
    rho0        = float(rho_vals[0])
    frac_rank1  = rho_infty / max(rho0, 1e-8)

    # Correlation length of residual (1/e crossing)
    rho_res_norm = rho_res / max(rho_res[0], 1e-8)
    cross = np.where(rho_res_norm <= 1.0 / np.e)[0]
    ell_res = float(rho_s[cross[0]]) if len(cross) > 0 else float(rho_s[-1])

    # Spectral mass of residual: ∫ ρ_res(s) ds
    ds = rho_s[1] - rho_s[0]
    spectral_mass_res = float(np.trapezoid(rho_res, rho_s))

    print(f"\n  ── Plateau analysis [{label}] ──────────────────────────")
    print(f"     ρ(0)             = {rho0:.4f}")
    print(f"     ρ_∞  (plateau)   = {rho_infty:.4f}")
    print(f"     rank-1 fraction  = {frac_rank1:.3f}   ({frac_rank1*100:.1f}% of variance)")
    print(f"     ℓ_res  (1/e of ρ_res) = {ell_res:.4f}")
    print(f"     ∫ρ_res ds        = {spectral_mass_res:.4f}")

    verdict = ("STRONG rank-1 structure — low-rank decomposition likely profitable"
               if frac_rank1 > 0.3 else
               "WEAK rank-1 structure — full kernel needed")
    print(f"     → {verdict}")

    return dict(
        rho_infty=rho_infty,
        frac_rank1=frac_rank1,
        ell_res=ell_res,
        spectral_mass_res=spectral_mass_res,
        verdict=verdict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_banded_assumption(rho_s, rho_vals, lambda_min, lambda_max,
                              nfe_list=(5, 10, 20)):
    rho_infty = float(np.median(rho_vals[-max(5, len(rho_vals)//5):]))
    print(f"  ρ_∞ (plateau)  = {rho_infty:.4f}")

    rho_res = (rho_vals - rho_infty) / max(1.0 - rho_infty, 1e-8)
    rho_res = np.clip(rho_res, 0, 1)

    target = 1.0 / np.e
    cross  = np.where(rho_res <= target)[0]
    if len(cross) == 0:
        ell_corr = rho_s[-1]
        print(f"  ρ_res never reaches 1/e — ell_corr > {rho_s[-1]:.3f}")
    else:
        ell_corr = float(rho_s[cross[0]])
        print(f"  ell_corr (1/e decay of ρ_res) = {ell_corr:.4f}")

    lambda_range = lambda_max - lambda_min
    print(f"  λ range = {lambda_range:.3f}")
    print()
    print(f"  {'NFE':>5}  {'h_typical':>10}  {'ell/h':>8}  {'banded?':>10}")
    for nfe in nfe_list:
        h_typ = lambda_range / nfe
        ratio = ell_corr / h_typ
        ok = "✓ yes" if ratio < 0.5 else ("~ marginal" if ratio < 1.5 else "✗ no")
        print(f"  {nfe:>5}  {h_typ:>10.4f}  {ratio:>8.3f}  {ok:>10}")

    return rho_infty, ell_corr


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _visualize_full(
    lambdas_grid, g_vals, sigma2_vals, sigma2_gp_vals,
    delta_lams, lbar_arr, rho_pairs, log_corrs,
    rho_s, rho_vals, rho_mat,
    seg_bounds, seg_rho_curves, cv_per_s, s_common,
    mean_cv, stationarity_verdict,
    # g'' correlation data (may be None if not estimated)
    rho_s_gpp, rho_vals_gpp, plateau_score, plateau_gpp,
    model_name, out_path,
):
    seg_colors = ["#e05c00", "#2d6a9f", "#2a7d4f", "#9b3fa0"]
    n_segs     = len(seg_rho_curves)
    has_gpp    = (rho_s_gpp is not None)

    # 3 rows × 3 cols  (last row for gpp curve + plateau summary)
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Row 0 ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lambdas_grid, g_vals, color="#2d6a9f", lw=2)
    ax1.axhline(0, color="gray", lw=0.7, ls="--")
    ax1.set_xlabel("λ"); ax1.set_ylabel("g(λ)")
    ax1.set_title("Scalar Jacobian Proxy  g(λ)")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(lambdas_grid, sigma2_vals,    color="#b5451b", lw=2, label="σ²_η")
    ax2.semilogy(lambdas_grid, sigma2_gp_vals, color="#9b3fa0", lw=2, ls="--", label="σ²_g''")
    ax2.set_xlabel("λ"); ax2.set_ylabel("[log scale]")
    ax2.set_title("Score Error σ²_η  &  Disc Error σ²_g''")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(delta_lams, rho_pairs, s=4, alpha=0.25, color="#999", rasterized=True)
    ax3.plot(rho_s, rho_vals, color="#2d6a9f", lw=2.5, label="global ρ̂(s)")
    if plateau_score is not None:
        ax3.axhline(plateau_score["rho_infty"], color="#e05c00", lw=1.5,
                    ls="--", label=f"ρ_∞={plateau_score['rho_infty']:.3f}")
    ax3.set_xlabel("|Δλ|"); ax3.set_ylabel("ρ̂")
    ax3.set_title("Score-Error  ρ̂(s)  (independent ε)")
    ax3.legend(fontsize=8); ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    # ── Row 1 ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(s_common, cv_per_s, color="#555", lw=2)
    ax4.axhline(0.15, color="#2a7d4f", lw=1.2, ls="--", label="0.15 (stationary)")
    ax4.axhline(0.35, color="#d12",    lw=1.2, ls="--", label="0.35 (non-stat.)")
    ax4.fill_between(s_common, 0, cv_per_s, where=(cv_per_s >= 0.35),
                     alpha=0.15, color="#d12")
    ax4.fill_between(s_common, 0, cv_per_s,
                     where=(cv_per_s >= 0.15) & (cv_per_s < 0.35), alpha=0.12, color="#e08000")
    ax4.set_xlabel("s"); ax4.set_ylabel("CV")
    verdict_color = "#d12" if "STRONGLY" in stationarity_verdict else \
                    ("#e08000" if "MILDLY" in stationarity_verdict else "#2a7d4f")
    ax4.set_title(f"Stationarity CV(s)\nmean={mean_cv:.3f} → {stationarity_verdict}",
                  color=verdict_color, fontsize=9)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rho_s, rho_vals, color="k", lw=2, ls="--", label="global", alpha=0.6)
    for k, (rs, rv) in enumerate(seg_rho_curves):
        lo, hi = seg_bounds[k], seg_bounds[k+1]
        ax5.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                 label=f"λ̄∈[{lo:.1f},{hi:.1f})")
    ax5.set_xlabel("|Δλ|"); ax5.set_ylabel("ρ̂")
    ax5.set_title("Per-Segment ρ̂(s) vs Global")
    ax5.legend(fontsize=8); ax5.set_ylim(-0.05, 1.05); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(lambdas_grid, np.abs(g_vals) * sigma2_vals, color="#2a7d4f", lw=2)
    ax6.set_xlabel("λ"); ax6.set_ylabel("|g(λ)| · σ²_η(λ)")
    ax6.set_title("Cost Integrand  |g| · σ²_η")
    ax6.grid(True, alpha=0.3)

    # ── Row 2 : g'' correlation + plateau summary ─────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    if has_gpp:
        ax7.scatter([], [], s=0)   # blank (raw pairs too noisy to show usefully)
        ax7.plot(rho_s_gpp, rho_vals_gpp, color="#9b3fa0", lw=2.5, label="ρ̂_g''(s)")
        if plateau_gpp is not None:
            ax7.axhline(plateau_gpp["rho_infty"], color="#e05c00", lw=1.5,
                        ls="--", label=f"ρ_∞={plateau_gpp['rho_infty']:.3f}")
        ax7.set_xlabel("|Δλ|"); ax7.set_ylabel("ρ̂_g''")
        ax7.set_title("g'' Cross-Step Correlation  ρ̂_g''(s)\n"
                      "(plateau → D_j has rank-1 component)")
        ax7.legend(fontsize=8); ax7.set_ylim(-0.15, 1.05)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.axis("off")
        ax7.text(0.5, 0.5, "g'' correlation\nnot estimated\n(--no_gpp_corr)",
                 ha="center", va="center", transform=ax7.transAxes, fontsize=11)

    # Comparison of ρ_∞ for score vs g''
    ax8 = fig.add_subplot(gs[2, 1])
    labels, vals = ["score-error ρ_∞"], [plateau_score["rho_infty"] if plateau_score else 0]
    if plateau_gpp:
        labels.append("g'' ρ_∞"); vals.append(plateau_gpp["rho_infty"])
    bars = ax8.bar(labels, vals, color=["#2d6a9f", "#9b3fa0"][:len(labels)], width=0.4)
    for bar, val in zip(bars, vals):
        ax8.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax8.axhline(0.3, color="#e05c00", lw=1.2, ls="--", label="0.3 rank-1 threshold")
    ax8.set_ylim(0, 1.05)
    ax8.set_ylabel("ρ_∞"); ax8.set_title("Rank-1 Plateau Comparison")
    ax8.legend(fontsize=8); ax8.grid(True, alpha=0.2, axis="y")

    # Summary text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    ps = plateau_score or {}
    pg = plateau_gpp   or {}
    txt = (
        f"Model: {model_name}\n"
        f"{'─'*35}\n"
        f"Score-error kernel:\n"
        f"  ρ_∞            = {ps.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 frac.   = {ps.get('frac_rank1', float('nan')):.3f}\n"
        f"  ℓ_res (1/e)    = {ps.get('ell_res', float('nan')):.4f}\n"
        f"  ∫ρ_res ds      = {ps.get('spectral_mass_res', float('nan')):.4f}\n\n"
        + (
        f"g'' kernel:\n"
        f"  ρ_∞_gpp        = {pg.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 frac.   = {pg.get('frac_rank1', float('nan')):.3f}\n"
        f"  → {pg.get('verdict','N/A')[:40]}\n\n"
        if has_gpp else
        "g'' correlation: not estimated\n\n"
        ) +
        f"Stationarity: {stationarity_verdict}\n"
        f"  mean CV = {mean_cv:.3f}\n"
    )
    ax9.text(0.04, 0.97, txt, transform=ax9.transAxes, fontsize=8.5,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaa"))
    ax9.set_title("Summary", fontsize=10)

    fig.suptitle(f"BornSchedule Model Statistics — {model_name}", fontsize=13, y=1.005)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main estimation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_estimation(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[stats] device = {device}")
    print(f"[stats] model  = {args.model}")

    print("[stats] loading noise schedule …")
    try:
        acp, alphas, sigmas, lambdas_dense = alphas_from_scheduler(args.model, device)
    except Exception:
        from diffusers import DDPMScheduler as _S
        sched = _S.from_pretrained(args.model, subfolder="scheduler")
        acp   = sched.alphas_cumprod.to(device).double()
        alphas = acp.sqrt()
        sigmas = (1.0 - acp).sqrt()
        lambdas_dense = torch.log(alphas / sigmas.clamp(min=1e-12))

    T = len(lambdas_dense)
    lam_all    = lambdas_dense.cpu().numpy()
    lambda_min = float(lambdas_dense[-1])
    lambda_max = float(lambdas_dense[1])
    print(f"  λ range: [{lambda_min:.3f}, {lambda_max:.3f}]  (T={T})")

    n_lam    = args.n_lambda
    n_dense  = n_lam // 3
    n_coarse = n_lam - n_dense
    lam_split  = lambda_min + 0.80 * (lambda_max - lambda_min)
    lam_coarse = np.linspace(lambda_min, lam_split, n_coarse, endpoint=False)
    lam_dense  = np.linspace(lam_split,  lambda_max, n_dense)
    lam_uniform = np.concatenate([lam_coarse, lam_dense])

    t_indices   = np.array([
        int((lambdas_dense - lam).abs().argmin().item())
        for lam in lam_uniform
    ])
    lambdas_grid = lam_all[t_indices]

    print("[stats] loading UNet …")
    unet     = load_unet(args.model, device, latent=args.latent)
    unet_cfg = unet.config

    in_ch     = int(unet_cfg.get("in_channels", 3))
    sample_sz = int(unet_cfg.get("sample_size", 32))
    if hasattr(sample_sz, "__len__"):
        sample_sz = sample_sz[0]
    d = in_ch * sample_sz * sample_sz
    print(f"  UNet input: {in_ch} × {sample_sz} × {sample_sz}  →  d = {d:,}")

    x0_shape = (args.n_samples, in_ch, sample_sz, sample_sz)
    print(f"[stats] generating {args.n_samples} Gaussian x_0 samples  shape={x0_shape}")
    x0_all = torch.randn(x0_shape, device=device)

    # ── g(λ) ────────────────────────────────────────────────────────────
    print(f"[stats] estimating g(λ) at {n_lam} λ points …")
    g_vals = np.zeros(n_lam)
    for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
        alpha_t = float(alphas[t_idx])
        sigma_t = float(sigmas[t_idx])
        g_acc = 0.0; n_sub = 0
        for start in range(0, args.n_samples, args.batch):
            x0_sub = x0_all[start:start + args.batch]
            g_acc += estimate_g_at_t(
                unet, t_idx, alpha_t, sigma_t, d,
                x0_sub, args.n_hutchinson, args.latent, device,
            )
            n_sub += 1
        g_vals[k] = g_acc / n_sub
        if (k + 1) % max(1, n_lam // 10) == 0 or k == 0:
            print(f"  g: {k+1}/{n_lam}  λ={lam:.3f}  g={g_vals[k]:.4f}")

    # ── σ²_η(λ) ─────────────────────────────────────────────────────────
    print(f"[stats] estimating σ²_η(λ) at {n_lam} λ points …")
    sigma2_vals = np.zeros(n_lam)
    with torch.no_grad():
        for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
            alpha_t = float(alphas[t_idx])
            sigma_t = float(sigmas[t_idx])
            acc = 0.0; n_sub = 0
            for start in range(0, args.n_samples, args.batch):
                x0_sub = x0_all[start:start + args.batch]
                acc += estimate_sigma2_at_t(
                    unet, t_idx, alpha_t, sigma_t, d,
                    x0_sub, args.latent, device,
                )
                n_sub += 1
            sigma2_vals[k] = acc / n_sub
            if (k + 1) % max(1, n_lam // 10) == 0 or k == 0:
                print(f"  σ²_η: {k+1}/{n_lam}  λ={lam:.3f}  σ²={sigma2_vals[k]:.4e}")

    # ── σ²_g''(λ) ────────────────────────────────────────────────────────
    print(f"[stats] estimating σ²_g''(λ) at {n_lam} λ points …")
    sigma2_gp_vals = np.zeros(n_lam)
    with torch.no_grad():
        for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
            acc = 0.0; n_sub = 0
            for start in range(0, args.n_samples, args.batch):
                x0_sub = x0_all[start:start + args.batch]
                acc += estimate_sigma2_gpp_at_t(
                    unet, t_idx, lambdas_dense, alphas, sigmas, d,
                    x0_sub, args.latent, device,
                )
                n_sub += 1
            sigma2_gp_vals[k] = acc / n_sub
            if (k+1) % max(1, n_lam//10) == 0:
                print(f"  σ²_g'': {k+1}/{n_lam}  λ={lam:.3f}  σ²={sigma2_gp_vals[k]:.4e}")

    from scipy.signal import savgol_filter
    log_gp = np.log(np.clip(sigma2_gp_vals, 1e-30, None))
    log_gp_smooth = savgol_filter(
        log_gp, window_length=min(7, n_lam//4*2+1), polyorder=2
    )
    sigma2_gp_vals_smooth = np.exp(log_gp_smooth)

    # ── ρ̂(s) score-error (independent ε) ────────────────────────────────
    n_ell = min(args.n_ell_samples, args.n_samples)
    print(f"[stats] estimating score-error ρ̂(s) with {n_ell} samples …")
    x0_ell = x0_all[:n_ell]
    C_mat, rho_mat, _, _ = estimate_ell(
        unet, t_indices, alphas, sigmas, d,
        x0_ell, args.latent, device,
    )

    dl_list, lbar_list, rho_list = [], [], []
    for i in range(n_lam):
        for j in range(i + 1, n_lam):
            dl_list.append(abs(lambdas_grid[i] - lambdas_grid[j]))
            lbar_list.append(0.5 * (lambdas_grid[i] + lambdas_grid[j]))
            rho_list.append(float(rho_mat[i, j]))
    delta_lams = np.array(dl_list)
    lbar_arr   = np.array(lbar_list)
    rho_pairs  = np.array(rho_list)
    log_corrs  = np.log(np.clip(rho_pairs, 1e-10, 1.0))

    rho_s, rho_vals = _extract_rho_curve(delta_lams, rho_pairs, anchor_at_one=True)
    print(f"\n  Score ρ̂: s∈[{rho_s[0]:.3f},{rho_s[-1]:.3f}]  "
          f"ρ̂∈[{rho_vals[-1]:.4f},{rho_vals[0]:.4f}]")
    print(f"  floor check: min ρ̂={rho_vals[-1]:.4f}  "
          f"({'floor remains' if rho_vals[-1] > 0.15 else 'decays to ~0'})")

    # ── Plateau / rank-1 analysis — score error ───────────────────────────
    plateau_score = analyse_plateau(rho_s, rho_vals, label="score-error")

    # ── g'' cross-step correlation ρ̂_gpp(s) ─────────────────────────────
    rho_s_gpp = rho_vals_gpp = None
    plateau_gpp = None
    if not args.no_gpp_corr:
        print(f"\n[stats] estimating g'' cross-step correlation with {n_ell} samples …")
        gpp_corr_mat, delta_lams_gpp, rho_gpp_pairs = estimate_gpp_correlation(
            unet, t_indices, lambdas_dense, alphas, sigmas, d,
            x0_ell, args.latent, device,
        )
        # Use anchor_at_one=False: ρ_gpp(0) is not necessarily 1 for second deriv.
        rho_s_gpp, rho_vals_gpp = _extract_rho_curve(
            delta_lams_gpp, rho_gpp_pairs, anchor_at_one=False
        )
        print(f"  g'' ρ̂: s∈[{rho_s_gpp[0]:.3f},{rho_s_gpp[-1]:.3f}]  "
              f"ρ̂∈[{rho_vals_gpp[-1]:.4f},{rho_vals_gpp[0]:.4f}]")
        plateau_gpp = analyse_plateau(rho_s_gpp, rho_vals_gpp, label="g''")
    else:
        print("[stats] Skipping g'' correlation (--no_gpp_corr set).")

    # ── Stationarity ─────────────────────────────────────────────────────
    n_segs    = 3
    seg_bounds = np.linspace(lbar_arr.min(), lbar_arr.max(), n_segs + 1)
    seg_rho_curves = []
    print(f"\n[stats] stationarity assessment ({n_segs} segments) …")
    for k in range(n_segs):
        lo, hi    = seg_bounds[k], seg_bounds[k + 1]
        seg_mask  = (lbar_arr >= lo) & (lbar_arr < hi)
        n_pairs   = int(seg_mask.sum())
        if n_pairs < 10:
            seg_rho_curves.append(_extract_rho_curve(delta_lams, rho_pairs))
        else:
            rs, rv = _extract_rho_curve(
                delta_lams[seg_mask], rho_pairs[seg_mask], anchor_at_one=True
            )
            seg_rho_curves.append((rs, rv))
            print(f"  λ̄∈[{lo:.2f},{hi:.2f})  {n_pairs} pairs  "
                  f"ρ̂(0)={rv[0]:.3f}  ρ̂(max)={rv[-1]:.4f}")

    s_common = rho_s[1:]
    seg_mats = []
    for (rs, rv) in seg_rho_curves:
        seg_pchip = PchipInterpolator(rs, rv, extrapolate=False)
        v_interp  = seg_pchip(s_common)
        v_interp  = np.where(np.isnan(v_interp), rv[-1], v_interp)
        v_interp  = np.clip(v_interp, 0.0, 1.0)
        seg_mats.append(v_interp)
    seg_mat   = np.array(seg_mats)
    seg_mean_s = seg_mat.mean(axis=0)
    seg_std_s  = seg_mat.std(axis=0)
    cv_per_s   = seg_std_s / (seg_mean_s + 1e-8)

    valid_s = rho_vals[1:] > 0.1
    mean_cv = float(cv_per_s[valid_s].mean()) if valid_s.any() else float("nan")

    if np.isnan(mean_cv):
        stationarity_verdict = "UNKNOWN"
        use_segmented = False
    elif mean_cv < 0.15:
        stationarity_verdict = "STATIONARY"
        use_segmented = False
    elif mean_cv < 0.35:
        stationarity_verdict = "MILDLY NON-STATIONARY"
        use_segmented = False
    else:
        stationarity_verdict = "STRONGLY NON-STATIONARY"
        use_segmented = True

    print(f"\n  mean CV = {mean_cv:.3f}  →  {stationarity_verdict}")

    verify_banded_assumption(rho_s, rho_vals, lambda_min, lambda_max,
                             nfe_list=(5, 10, 20, 50))

    # ── Save ──────────────────────────────────────────────────────────────
    safe_name = args.model.replace("/", "--").replace(":", "--")
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"{safe_name}.npz"

    rhomat_path = out_dir / f"{safe_name}_rhomat.npy"
    np.save(str(rhomat_path), rho_mat.astype(np.float32))
    print(f"  rho_mat saved → {rhomat_path}")

    save_dict = dict(
        lambda_grid       = lambdas_grid.astype(np.float32),
        g_values          = g_vals.astype(np.float32),
        sigma2_values     = sigma2_vals.astype(np.float32),
        sigma2_gpp_values = sigma2_gp_vals_smooth.astype(np.float32),  # key name used by born_schedule.py
        lambda_min        = np.float32(lambda_min),
        lambda_max        = np.float32(lambda_max),
        rho_s             = rho_s.astype(np.float32),
        rho_values        = rho_vals.astype(np.float32),
    )
    if use_segmented or getattr(args, "force_segmented", False):
        save_dict["seg_lambda_bounds"] = seg_bounds.astype(np.float32)
        for k, (rs, rv) in enumerate(seg_rho_curves):
            save_dict[f"rho_s_seg{k}"]      = rs.astype(np.float32)
            save_dict[f"rho_values_seg{k}"] = rv.astype(np.float32)
    # Save g'' correlation curve if estimated (for future low-rank D_j)
    if rho_s_gpp is not None:
        save_dict["rho_s_gpp"]      = rho_s_gpp.astype(np.float32)
        save_dict["rho_values_gpp"] = rho_vals_gpp.astype(np.float32)

    np.savez(str(out_path), **save_dict)
    print(f"[stats] saved → {out_path}")

    vis_path = out_path.with_suffix(".png")
    _visualize_full(
        lambdas_grid, g_vals, sigma2_vals, sigma2_gp_vals_smooth,
        delta_lams, lbar_arr, rho_pairs, log_corrs,
        rho_s, rho_vals, rho_mat,
        seg_bounds, seg_rho_curves, cv_per_s, s_common,
        mean_cv, stationarity_verdict,
        rho_s_gpp, rho_vals_gpp, plateau_score, plateau_gpp,
        args.model, vis_path,
    )

    print("\n══ Summary ═════════════════════════════════════════════════")
    print(f"  Model           : {args.model}")
    print(f"  d               : {d:,}")
    print(f"  λ range         : [{lambda_min:.3f}, {lambda_max:.3f}]")
    print(f"  g(λ)            : [{g_vals.min():.4f}, {g_vals.max():.4f}]")
    print(f"  σ²_η(λ)         : [{sigma2_vals.min():.2e}, {sigma2_vals.max():.2e}]")
    print(f"  σ²_g''(λ)       : [{sigma2_gp_vals_smooth.min():.2e}, {sigma2_gp_vals_smooth.max():.2e}]")
    print(f"  score ρ_∞       : {plateau_score['rho_infty']:.4f}  "
          f"(rank-1 frac {plateau_score['frac_rank1']:.2f})")
    if plateau_gpp:
        print(f"  g'' ρ_∞         : {plateau_gpp['rho_infty']:.4f}  "
              f"(rank-1 frac {plateau_gpp['frac_rank1']:.2f})")
        print(f"  g'' verdict     : {plateau_gpp['verdict']}")
    print(f"  Stationarity    : {stationarity_verdict}  (CV={mean_cv:.3f})")
    print(f"  Output          : {out_path}")
    print("════════════════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate BornSchedule statistics (g, σ²_η, σ²_g'', ρ̂) from a diffusers UNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    required=True,
                   help="e.g. google/ddpm-cifar10-32")
    p.add_argument("--output_dir",
                   default=str(Path.home() / ".cache" / "opt_schedule"))
    p.add_argument("--latent",   action="store_true",
                   help="Set for latent diffusion models (SD 1.5 etc.); "
                        "omit for pixel-space models (CIFAR-10 etc.)")
    p.add_argument("--n_lambda",      type=int, default=60)
    p.add_argument("--n_samples",     type=int, default=256)
    p.add_argument("--n_ell_samples", type=int, default=256)
    p.add_argument("--n_hutchinson",  type=int, default=8)
    p.add_argument("--batch",         type=int, default=4)
    p.add_argument("--device",        default=None)
    p.add_argument("--force_segmented", action="store_true")
    p.add_argument("--no_gpp_corr",   action="store_true",
                   help="Skip g'' cross-step correlation estimation (faster)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_estimation(args)