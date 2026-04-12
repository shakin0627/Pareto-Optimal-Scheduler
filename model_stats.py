from __future__ import annotations

import os
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
# UNet loading
# ─────────────────────────────────────────────────────────────────────────────

def load_unet(model_name: str, device: torch.device, latent: bool = True):
    """
    SD 1.5
    """
    if not latent:
        raise ValueError("SD1.5 是 latent model，必须设置 latent=True")

    print(f"[stats] Loading StableDiffusionPipeline for {model_name} ...")
    
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        safety_checker=None,        
        requires_safety_checker=False,
    )

    unet = pipe.unet.to(device)
    unet.eval()


    for m in unet.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    print(f"[stats] UNet loaded successfully | parameters: {sum(p.numel() for p in unet.parameters()):,}")

    del pipe.vae
    del pipe.text_encoder
    del pipe.tokenizer
    del pipe.scheduler
    if hasattr(pipe, "safety_checker"):
        del pipe.safety_checker
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return unet

# def load_unet(model_name: str, device: torch.device, latent: bool):
#     if latent:
#         unet = UNet2DConditionModel.from_pretrained(
#             model_name, subfolder="unet",
#             torch_dtype=torch.float32,
#         ).to(device)
#     else:
#         unet = UNet2DModel.from_pretrained(
#             model_name, torch_dtype=torch.float32,
#         ).to(device)
#     unet.eval()
#     for m in unet.modules():
#         if isinstance(m, torch.nn.Dropout):
#             m.p = 0.0
#     return unet

# ─────────────────────────────────────────────────────────────────────────────
# Forward pass abstraction
# ─────────────────────────────────────────────────────────────────────────────
 
# def unet_eps(unet, x_t: torch.Tensor, t: torch.Tensor, latent: bool) -> torch.Tensor:
#     if latent:
#         B = x_t.shape[0]
#         dim = unet.config.cross_attention_dim
#         null_emb = torch.zeros(B, 77, dim, device=x_t.device, dtype=x_t.dtype)
#         out = unet(x_t, t, encoder_hidden_states=null_emb)
#     else:
#         out = unet(x_t, t)
#     return out.sample

def unet_eps(unet, x_t: torch.Tensor, t: torch.Tensor, latent: bool) -> torch.Tensor:
    """
    调用 UNet 返回 noise prediction (ε_θ)。
    - 如果是 latent 模型（如 SD1.5），自动传入 null text embedding（无条件）。
    - 如果是 pixel-based 无条件模型（如 CIFAR），直接调用。
    """
    if latent:
        # SD1.5 / SDXL 等 latent diffusion 模型需要 encoder_hidden_states
        B = x_t.shape[0]
        # 获取 cross_attention_dim（SD1.5 是 768）
        cross_attn_dim = unet.config.cross_attention_dim
        
        # null embedding（相当于 unconditional）
        null_emb = torch.zeros(
            (B, 77, cross_attn_dim),   # 77 是 CLIP tokenizer 的最大长度
            device=x_t.device,
            dtype=x_t.dtype
        )
        
        # encoder_hidden_states
        out = unet(
            x_t,
            t,
            encoder_hidden_states=null_emb,
            return_dict=False          # 返回 tuple，[0] 是 sample
        )
        return out[0]                  # .sample 等价于 out[0]
    else:
        # 普通无条件 UNet（如 google/ddpm-cifar10-32）
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
            # noise_pred = unet_eps(unet, x_t_req, t_tensor, latent)
            noise_pred = unet_eps(unet, x_t_req, t_tensor, latent=True)
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
# σ²_{g'}(λ) — discretization error variance
# Estimates (1/d) E[||∂_λ [e^{-λ} ε_θ]||²]
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma2_gpp_at_t(
    unet, t_idx, lambdas_dense, alphas, sigmas, d,
    x0_batch, latent, device, delta_t=2,
) -> float:
    """
    σ²_g''(λ) = (1/d) E[||∂²_λ [e^{-λ} ε_θ]||²]
    
    Three-point central difference on g(λ) = e^{-λ}ε_θ.
    SAME eps across lo/mid/hi — estimates derivative along
    the diffusion path, not across independent samples.
    """
    B = x0_batch.shape[0]
    t_lo  = max(0, t_idx - delta_t)
    t_hi  = min(len(lambdas_dense) - 1, t_idx + delta_t)
    t_mid = t_idx

    lam_lo  = float(lambdas_dense[t_lo])
    lam_hi  = float(lambdas_dense[t_hi])
    lam_mid = float(lambdas_dense[t_mid])
    h = (lam_hi - lam_lo) / 2   # single spacing

    if abs(h) < 1e-8 or t_lo == t_hi:
        return 0.0

    t_lo_t  = torch.full((B,), t_lo,  device=device, dtype=torch.long)
    t_mid_t = torch.full((B,), t_mid, device=device, dtype=torch.long)
    t_hi_t  = torch.full((B,), t_hi,  device=device, dtype=torch.long)

    # Shared eps: path-coherent — same x0 perturbed along the same noise direction
    eps = torch.randn_like(x0_batch)
    x_lo  = float(alphas[t_lo])  * x0_batch + float(sigmas[t_lo])  * eps
    x_mid = float(alphas[t_mid]) * x0_batch + float(sigmas[t_mid]) * eps
    x_hi  = float(alphas[t_hi])  * x0_batch + float(sigmas[t_hi])  * eps

    g_lo  = np.exp(-lam_lo)  * unet_eps(unet, x_lo,  t_lo_t,  latent)
    g_mid = np.exp(-lam_mid) * unet_eps(unet, x_mid, t_mid_t, latent)
    g_hi  = np.exp(-lam_hi)  * unet_eps(unet, x_hi,  t_hi_t,  latent)

    # Central second difference: (g(λ+h) - 2g(λ) + g(λ-h)) / h²
    g_pp = (g_hi - 2.0 * g_mid + g_lo) / (h ** 2)
    return (g_pp ** 2).sum().item() / (B * d)

# ─────────────────────────────────────────────────────────────────────────────
# Score-error correlation matrix estimation
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate K_ξ(λ_i, λ_j) = (1/d) E[<ξ(λ_i), ξ(λ_j)>].

    KEY CHANGE vs previous version:
    Each λ gets its own independent noise sample ε_m ~ N(0,I).
    This eliminates the shared-ε pollution term:

        Old (shared ε):
            e_i · e_j = (ξ_i + ε*_i − ε)(ξ_j + ε*_j − ε)
            E[e_i·e_j] = K_ξ(i,j) + E[(ε*_i−ε)(ε*_j−ε)]   ← floor artefact

        New (independent ε_i ⊥ ε_j):
            E[e_i·e_j] = K_ξ(i,j) + E[ε*_i−ε_i]·E[ε*_j−ε_j] = K_ξ(i,j)

    because E[ε*(λ,x_t) − ε] = 0 by the denoising score matching objective,
    and ε_i, ε_j are independent of each other.

    The diagonal C[i,i] still equals σ²_η(λ_i) (same as estimate_sigma2_at_t)
    since with a single sample the independence argument is vacuous.
    """
    M = len(t_indices)
    B = x0_batch.shape[0]

    C = np.zeros((M, M), dtype=np.float64)
    n_batches = 0

    for xi in x0_batch:
        xi = xi.unsqueeze(0)   # [1, C, H, W]

        # Independent ε per λ — critical for unbiased K_ξ estimation
        errors = []
        for m, t_idx in enumerate(t_indices):
            alpha_t = float(alphas[t_idx])
            sigma_t = float(sigmas[t_idx])
            t_tensor = torch.full((1,), t_idx, device=device, dtype=torch.long)

            eps_m = torch.randn_like(xi)          # fresh independent noise
            x_t = alpha_t * xi + sigma_t * eps_m
            noise_pred = unet_eps(unet, x_t, t_tensor, latent)

            # residual e_m = ε_θ(x_t, t) − ε_m ≈ ξ(λ_m) + (ε*(λ_m) − ε_m)
            # cross-λ expectation of the noise term vanishes by independence
            e = (noise_pred - eps_m).squeeze(0).flatten().cpu().double()
            errors.append(e)

        for i in range(M):
            for j in range(i, M):
                dot = (errors[i] * errors[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n_batches += 1

    C /= n_batches

    # Normalize to correlation
    diag = np.diag(C).clip(min=1e-30)
    rho = C / np.sqrt(np.outer(diag, diag))
    rho = np.clip(rho, 1e-8, 1.0)

    delta_lams = []
    log_corrs  = []
    for i in range(M):
        for j in range(i + 1, M):
            dl = abs(t_indices[i] - t_indices[j])
            delta_lams.append(dl)
            log_corrs.append(np.log(rho[i, j]))

    return C, rho, np.array(delta_lams, dtype=np.float64), np.array(log_corrs, dtype=np.float64)
 
# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
 
def _visualize_full(
    lambdas_grid, g_vals, sigma2_vals, sigma2_gp_vals,
    delta_lams, lbar_arr, rho_pairs, log_corrs,
    rho_s, rho_vals, rho_mat,
    seg_bounds, seg_rho_curves, cv_per_s, s_common,
    mean_cv, stationarity_verdict,
    model_name, out_path,
):
    seg_colors = ["#e05c00", "#2d6a9f", "#2a7d4f", "#9b3fa0"]
    n_segs = len(seg_rho_curves)
 
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)
 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lambdas_grid, g_vals, color="#2d6a9f", lw=2)
    ax1.axhline(0, color="gray", lw=0.7, ls="--")
    ax1.set_xlabel("λ"); ax1.set_ylabel("g(λ)")
    ax1.set_title("Scalar Jacobian Proxy  g(λ)")
    ax1.grid(True, alpha=0.3)
 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(lambdas_grid, sigma2_vals, color="#b5451b", lw=2, label="σ²_η")
    ax2.semilogy(lambdas_grid, sigma2_gp_vals, color="#9b3fa0", lw=2, ls="--", label="σ²_g'")
    ax2.set_xlabel("λ"); ax2.set_ylabel("[log scale]")
    ax2.set_title("Score Error σ²_η  &  Disc Error σ²_g'")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")
 
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(delta_lams, rho_pairs, s=4, alpha=0.25, color="#999",
                rasterized=True, label="raw pairs")
    ax3.plot(rho_s, rho_vals, color="#2d6a9f", lw=2.5, label="global ρ̂(s)")
    ax3.set_xlabel("|Δλ|"); ax3.set_ylabel("ρ̂")
    ax3.set_title("Global Non-Parametric  ρ̂(s)\n(independent ε — floor should vanish)")
    ax3.legend(fontsize=8); ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
 
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(s_common, cv_per_s, color="#555", lw=2, label="CV(s)")
    ax4.axhline(0.15, color="#2a7d4f", lw=1.2, ls="--", label="stat. threshold (0.15)")
    ax4.axhline(0.35, color="#d12",    lw=1.2, ls="--", label="non-stat. threshold (0.35)")
    ax4.fill_between(s_common, 0, cv_per_s,
                     where=(cv_per_s >= 0.35), alpha=0.15, color="#d12")
    ax4.fill_between(s_common, 0, cv_per_s,
                     where=(cv_per_s >= 0.15) & (cv_per_s < 0.35), alpha=0.12, color="#e08000")
    ax4.set_xlabel("s = |Δλ|"); ax4.set_ylabel("CV across segments")
    verdict_color = "#d12" if "STRONGLY" in stationarity_verdict else \
                    ("#e08000" if "MILDLY" in stationarity_verdict else "#2a7d4f")
    ax4.set_title(f"Stationarity: CV(s) across λ̄ segments\n"
                  f"mean CV={mean_cv:.3f}  →  {stationarity_verdict}",
                  color=verdict_color, fontsize=9)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)
 
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(rho_s, rho_vals, color="k", lw=2, ls="--", label="global", alpha=0.6)
    for k, (rs, rv) in enumerate(seg_rho_curves):
        lo, hi = seg_bounds[k], seg_bounds[k+1]
        ax5.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                 label=f"λ̄∈[{lo:.1f},{hi:.1f})")
    ax5.set_xlabel("|Δλ|"); ax5.set_ylabel("ρ̂")
    ax5.set_title("Per-Segment ρ̂(s) vs Global")
    ax5.legend(fontsize=8); ax5.set_ylim(-0.05, 1.05); ax5.grid(True, alpha=0.3)
 
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(lambdas_grid, np.abs(g_vals) * sigma2_vals, color="#2a7d4f", lw=2)
    ax6.set_xlabel("λ"); ax6.set_ylabel("|g(λ)| · σ²_η(λ)")
    ax6.set_title("Cost Integrand  |g| · σ²_η\n(peak → densest steps needed here)")
    ax6.grid(True, alpha=0.3)
 
    ax7 = fig.add_subplot(gs[1, 2])
    M = len(lambdas_grid)
    for offset in [1, 2, 4, 8, 16]:
        if offset >= M:
            continue
        y_diag = [rho_mat[i, i + offset] for i in range(M - offset)]
        x_diag = [0.5*(lambdas_grid[i]+lambdas_grid[i+offset]) for i in range(M - offset)]
        dl_mean = float(np.mean([abs(lambdas_grid[i]-lambdas_grid[i+offset])
                                 for i in range(M-offset)]))
        ax7.plot(x_diag, y_diag, lw=1.5, label=f"+{offset} (|Δλ|≈{dl_mean:.2f})")
    ax7.axhline(0, color="gray", lw=0.7, ls="--")
    ax7.set_xlabel("λ̄  (position)"); ax7.set_ylabel("ρ̂(λᵢ, λᵢ₊ₖ)")
    ax7.set_title("Diagonal Slices of ρ̂\n(flat → stationary; varying → not)")
    ax7.legend(fontsize=7, ncol=2); ax7.set_ylim(-0.1, 1.05); ax7.grid(True, alpha=0.3)
 
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
    txt = (
        f"Model: {model_name}\n"
        f"{'─'*35}\n"
        f"g(λ)   ∈ [{g_vals.min():.3f}, {g_vals.max():.3f}]\n"
        f"σ²_η   ∈ [{sigma2_vals.min():.2e}, {sigma2_vals.max():.2e}]\n"
        f"σ²_g'  ∈ [{sigma2_gp_vals.min():.2e}, {sigma2_gp_vals.max():.2e}]\n"
        f"λ      ∈ [{lambdas_grid[0]:.2f}, {lambdas_grid[-1]:.2f}]\n\n"
        f"ρ̂ estimation: INDEPENDENT ε per λ\n"
        f"  (eliminates shared-ε floor artefact)\n\n"
        f"Stationarity test:\n"
        f"  mean CV(s) = {mean_cv:.3f}\n"
        f"  → {stationarity_verdict}\n\n"
        f"ρ̂ curve: {len(rho_s)} points\n"
        f"  s  ∈ [{rho_s[0]:.2f}, {rho_s[-1]:.2f}]\n"
        f"  ρ̂  ∈ [{rho_vals[-1]:.4f}, {rho_vals[0]:.4f}]\n\n"
        f"Kernel mode:\n"
        f"  {'segmented ('+str(n_segs)+' segs)' if mean_cv >= 0.35 else 'global non-parametric'}"
    )
    ax8.text(0.05, 0.97, txt, transform=ax8.transAxes, fontsize=8.5,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaa"))
    ax8.set_title("Summary", fontsize=10)
 
    fig.suptitle(f"BornSchedule Model Statistics — {model_name}", fontsize=13, y=1.01)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {out_path}")

def verify_banded_assumption(rho_s, rho_vals, lambda_min, lambda_max,
                              nfe_list=(5, 10, 20)):
    """
    Check whether K_xi^res decays within one step for given NFE budgets.
    rho_infty estimated from plateau of rho curve.
    """
    # Estimate plateau (rho_infty) from tail
    rho_infty = float(np.median(rho_vals[-max(5, len(rho_vals)//5):]))
    print(f"  ρ_∞ (plateau)  = {rho_infty:.4f}")

    # Residual: rho_res(s) = rho(s) - rho_infty, normalised so rho_res(0)=1
    rho_res = (rho_vals - rho_infty) / max(1.0 - rho_infty, 1e-8)
    rho_res = np.clip(rho_res, 0, 1)

    # Correlation length: s where rho_res first drops to 1/e
    target = 1.0 / np.e
    cross = np.where(rho_res <= target)[0]
    if len(cross) == 0:
        ell_corr = rho_s[-1]
        print(f"  ρ_res never reaches 1/e — ell_corr > {rho_s[-1]:.3f}")
    else:
        ell_corr = float(rho_s[cross[0]])
        print(f"  ell_corr (1/e decay of ρ_res) = {ell_corr:.4f}")

    lambda_range = lambda_max - lambda_min
    print(f"  λ range = {lambda_range:.3f}")
    print()
    print(f"  {'NFE':>5}  {'N_steps':>8}  {'h_typical':>10}  {'ell/h':>8}  {'banded?':>10}")
    for nfe in nfe_list:
        n_steps = nfe // 2          # DPM-Solver-2: NFE = 2 * steps
        h_typ = lambda_range / n_steps
        ratio = ell_corr / h_typ
        ok = "✓ yes" if ratio < 0.5 else ("~ marginal" if ratio < 1.5 else "✗ no")
        print(f"  {nfe:>5}  {n_steps:>8}  {h_typ:>10.4f}  {ratio:>8.3f}  {ok:>10}")

    return rho_infty, ell_corr

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
    lam_all = lambdas_dense.cpu().numpy()
    lambda_min = float(lambdas_dense[-1])
    lambda_max = float(lambdas_dense[1])
    print(f"  λ range: [{lambda_min:.3f}, {lambda_max:.3f}]  (T={T})")
 
    n_lam = args.n_lambda
    n_dense   = n_lam // 3
    n_coarse  = n_lam - n_dense
    lam_split = lambda_min + 0.80 * (lambda_max - lambda_min)
    lam_coarse = np.linspace(lambda_min, lam_split, n_coarse, endpoint=False)
    lam_dense  = np.linspace(lam_split,  lambda_max, n_dense)
    lam_uniform = np.concatenate([lam_coarse, lam_dense])

    t_indices = np.array([
        int((lambdas_dense - lam).abs().argmin().item())
        for lam in lam_uniform
    ])
    lambdas_grid = lam_all[t_indices]
 
    print("[stats] loading UNet …")
    # unet = load_unet(args.model, device, latent=args.latent)
    unet = load_unet(args.model, device, latent=True)
    unet_cfg = unet.config
 
    in_ch = int(unet_cfg.get("in_channels", 3))
    sample_sz = int(unet_cfg.get("sample_size", 32))
    if hasattr(sample_sz, "__len__"):
        sample_sz = sample_sz[0]
    d = in_ch * sample_sz * sample_sz
    print(f"  UNet input: {in_ch} × {sample_sz} × {sample_sz}  →  d = {d:,}")
 
    x0_shape = (args.n_samples, in_ch, sample_sz, sample_sz)
    print(f"[stats] generating {args.n_samples} Gaussian x_0 samples  shape={x0_shape}")
    x0_all = torch.randn(x0_shape, device=device)
 
    # ── Estimate g(λ) ────────────────────────────────────────────────────
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
 
    # ── Estimate σ²_η(λ) ─────────────────────────────────────────────────
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

    # ── Estimate σ²_{g'}(λ) for D_j ──────────────────────────────────────
    print(f"[stats] estimating σ²_g'(λ) at {n_lam} λ points …")
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
                print(f"  σ²_g': {k+1}/{n_lam}  λ={lam:.3f}  σ²_g'={sigma2_gp_vals[k]:.4e}")

    from scipy.signal import savgol_filter
    log_gp = np.log(np.clip(sigma2_gp_vals, 1e-30, None))
    log_gp_smooth = savgol_filter(log_gp, window_length=min(7, n_lam//4*2+1), polyorder=2)
    sigma2_gp_vals_smooth = np.exp(log_gp_smooth)

    # ── Estimate ρ̂(s) with independent ε ────────────────────────────────
    n_ell = min(args.n_ell_samples, args.n_samples)
    print(f"[stats] estimating ρ̂(s) using {n_ell} samples × {n_lam} λ points …")
    print(f"  [ρ̂] Using INDEPENDENT ε per λ to eliminate shared-ε floor artefact.")
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
 
    def _extract_rho_curve(dl, rho, n_bins=40):
        valid = dl > 0
        if valid.sum() < 5:
            s = np.linspace(0, 1, 5)
            return s, np.ones(5)
        edges = np.unique(np.percentile(dl[valid], np.linspace(0, 100, n_bins + 1)))
        s_list, v_list = [0.0], [1.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = valid & (dl >= lo) & (dl < hi) & (rho > 0)
            if mask.sum() < 3:
                continue
            s_list.append(float(np.median(dl[mask])))
            v_list.append(float(np.median(rho[mask])))
        s_arr = np.array(s_list)
        v_arr = np.array(v_list)
        log_v = np.log(np.clip(v_arr, 1e-10, 1.0))
        for k in range(1, len(log_v)):
            if log_v[k] > log_v[k - 1]:
                log_v[k] = log_v[k - 1]
        v_arr = np.exp(log_v)
        pchip = PchipInterpolator(s_arr, v_arr, extrapolate=False)
        s_out = np.linspace(s_arr[0], s_arr[-1], 200)
        v_out = pchip(s_out)
        v_out = np.clip(v_out, v_arr[-1], 1.0)
        v_out[0] = 1.0
        return s_out, v_out

    rho_s, rho_vals = _extract_rho_curve(delta_lams, rho_pairs)
    print(f"\n  Global ρ̂: {len(rho_s)} points  "
          f"s∈[{rho_s[0]:.3f},{rho_s[-1]:.3f}]  "
          f"ρ̂∈[{rho_vals[-1]:.4f},{rho_vals[0]:.4f}]")
    print(f"  ρ̂ floor check: min ρ̂ = {rho_vals[-1]:.4f}  "
          f"({'floor remains — true long-range corr?' if rho_vals[-1] > 0.15 else 'floor gone — banded assumption ok'})")

    n_segs = 3
    seg_bounds = np.linspace(lbar_arr.min(), lbar_arr.max(), n_segs + 1)
    seg_rho_curves = []
    print(f"\n[stats] stationarity assessment ({n_segs} λ̄-segments) …")
    for k in range(n_segs):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        seg_mask = (lbar_arr >= lo) & (lbar_arr < hi)
        lab = f"λ̄∈[{lo:.2f},{hi:.2f})"
        n_pairs = int(seg_mask.sum())
        if n_pairs < 10:
            print(f"  {lab:<28}  {n_pairs:>5} pairs  → too few, using global fallback")
            seg_rho_curves.append(_extract_rho_curve(delta_lams, rho_pairs))
        else:
            rs, rv = _extract_rho_curve(delta_lams[seg_mask], rho_pairs[seg_mask])
            seg_rho_curves.append((rs, rv))
            print(f"  {lab:<28}  {n_pairs:>5} pairs  "
                  f"ρ̂(s→0)={rv[0]:.3f}  ρ̂(s_max)={rv[-1]:.4f}")

    s_common = rho_s[1:]
    seg_mats = []
    for (rs, rv) in seg_rho_curves:
        seg_pchip = PchipInterpolator(rs, rv, extrapolate=False)
        v_interp  = seg_pchip(s_common)
        v_interp  = np.where(np.isnan(v_interp), rv[-1], v_interp)
        v_interp  = np.clip(v_interp, 0.0, 1.0)
        seg_mats.append(v_interp)
    seg_mat = np.array(seg_mats)

    seg_mean_s = seg_mat.mean(axis=0)
    seg_std_s  = seg_mat.std(axis=0)
    cv_per_s   = seg_std_s / (seg_mean_s + 1e-8)

    valid_s  = rho_vals[1:] > 0.1
    mean_cv  = float(cv_per_s[valid_s].mean()) if valid_s.any() else float("nan")

    if np.isnan(mean_cv):
        stationarity_verdict = "UNKNOWN (insufficient data)"
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

    print(f"\n  Curve CV (mean across s, ρ̂>0.1) = {mean_cv:.3f}")
    print(f"  Verdict: {stationarity_verdict}")

    rho_infty, ell_corr = verify_banded_assumption(
        rho_s,
        rho_vals,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        nfe_list=(5, 10, 20, 50)
    )

    # ── Save ──────────────────────────────────────────────────────────────
    safe_name = args.model.replace("/", "--").replace(":", "--")
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"{safe_name}.npz"

    rhomat_path = out_dir / f"{safe_name}_rhomat.npy"
    np.save(str(rhomat_path), rho_mat.astype(np.float32))
    print(f"  rho_mat saved → {rhomat_path}")

    save_dict = dict(
        lambda_grid      = lambdas_grid.astype(np.float32),
        g_values         = g_vals.astype(np.float32),
        sigma2_values    = sigma2_vals.astype(np.float32),
        sigma2_gp_values = sigma2_gp_vals_smooth.astype(np.float32),
        lambda_min       = np.float32(lambda_min),
        lambda_max       = np.float32(lambda_max),
        rho_s            = rho_s.astype(np.float32),
        rho_values       = rho_vals.astype(np.float32),
    )
    if use_segmented or getattr(args, "force_segmented", False):
        save_dict["seg_lambda_bounds"] = seg_bounds.astype(np.float32)
        for k, (rs, rv) in enumerate(seg_rho_curves):
            save_dict[f"rho_s_seg{k}"]      = rs.astype(np.float32)
            save_dict[f"rho_values_seg{k}"] = rv.astype(np.float32)
    np.savez(str(out_path), **save_dict)
    print(f"[stats] saved → {out_path}")

    vis_path = out_path.with_suffix(".png")
    _visualize_full(
        lambdas_grid, g_vals, sigma2_vals, sigma2_gp_vals_smooth,
        delta_lams, lbar_arr, rho_pairs, log_corrs,
        rho_s, rho_vals, rho_mat,
        seg_bounds, seg_rho_curves, cv_per_s, s_common,
        mean_cv, stationarity_verdict,
        args.model, vis_path,
    )

    print("\n══ Model Statistics Summary ════════════════════════════════")
    print(f"  Model       : {args.model}")
    print(f"  d           : {d:,}")
    print(f"  λ range     : [{lambda_min:.3f}, {lambda_max:.3f}]")
    print(f"  g(λ)        : mean={g_vals.mean():.4f}  [{g_vals.min():.4f}, {g_vals.max():.4f}]")
    print(f"  σ²_η(λ)     : mean={sigma2_vals.mean():.4e}  [{sigma2_vals.min():.4e}, {sigma2_vals.max():.4e}]")
    print(f"  σ²_g'(λ)    : mean={sigma2_gp_vals_smooth.mean():.4e}  [{sigma2_gp_vals_smooth.min():.4e}, {sigma2_gp_vals_smooth.max():.4e}]")
    print(f"  ρ̂ floor     : {rho_vals[-1]:.4f}  ({'WARN: floor remains' if rho_vals[-1]>0.15 else 'OK: decays to ~0'})")
    print(f"  Stationarity: {stationarity_verdict}  (curve CV={mean_cv:.3f})")
    print(f"  Output      : {out_path}")
    print("════════════════════════════════════════════════════════════\n")

    return dict(
        lambda_grid=lambdas_grid, g_values=g_vals,
        sigma2_values=sigma2_vals, sigma2_gp_values=sigma2_gp_vals_smooth,
        lambda_min=lambda_min, lambda_max=lambda_max,
        rho_s=rho_s, rho_values=rho_vals,
        stationarity_verdict=stationarity_verdict, mean_cv=mean_cv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate BornSchedule statistics (g, σ²_η, σ²_g', ρ̂) from a diffusers UNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    required=True)
    p.add_argument("--output_dir",
                   default=str(Path.home() / ".cache" / "opt_schedule"))
    p.add_argument("--latent",   action="store_true")
    p.add_argument("--n_lambda", type=int, default=60)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--n_ell_samples", type=int, default=128)
    p.add_argument("--n_hutchinson", type=int, default=8)
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--device",   default=None)
    p.add_argument("--force_segmented", action="store_true")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    run_estimation(args)