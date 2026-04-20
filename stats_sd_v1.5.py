"""
stats_sd15.py  —  BornSchedule statistics for Stable Diffusion v1.5.

VP-SDE + CFG + latent UNet.  Coordinate: λ = log(α/σ) ∈ [λ_min, λ_max].

Usage:
  python stats_sd15.py \\
      --model runwayml/stable-diffusion-v1-5 \\
      --cfg_scale 7.5 \\
      --n_lambda 60 --n_samples 256 --n_ell_samples 128 \\
      --output_dir /media/ssd_horse/keying/Pareto-Optimal-Scheduler/stats_sd_v1.5
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# VP-SDE helpers  
# ══════════════════════════════════════════════════════════════════════════════

def alphas_from_scheduler(model_name: str, device: torch.device):
    from diffusers import DDPMScheduler
    sched = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    if hasattr(sched, "alphas_cumprod"):
        acp = sched.alphas_cumprod.to(device).double()
    else:
        betas   = sched.betas.to(device).double()
        acp     = torch.cumprod(1.0 - betas, dim=0)
    alpha   = acp.sqrt()
    sigma   = (1.0 - acp).sqrt()
    lambdas = torch.log(alpha / sigma.clamp(min=1e-12))
    return acp, alpha, sigma, lambdas


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(model_name: str, device: torch.device, dtype):
    from diffusers import StableDiffusionPipeline
    print(f"  [sd15] loading pipeline from {model_name} …")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=dtype, token=HF_TOKEN,
        safety_checker=None,
    ).to(device)
    unet = pipe.unet
    unet.eval()
    for m in unet.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    params = sum(p.numel() for p in unet.parameters()) / 1e9
    print(f"  [sd15] UNet {params:.1f}B params")
    return unet, pipe


# ══════════════════════════════════════════════════════════════════════════════
# Text encoding  (CLIP via diffusers encode_prompt)
# ══════════════════════════════════════════════════════════════════════════════

def encode_prompts(pipe, prompts: list, device, dtype, batch_size: int = 8) -> torch.Tensor:
    """Returns (N, 77, 768) text embeddings on CPU."""
    all_embs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        with torch.no_grad():
            embs = pipe.encode_prompt(
                batch,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )[0]
        all_embs.append(embs.cpu().to(dtype))
        if start % max(batch_size, 16) == 0:
            print(f"    encoded {start + len(batch)}/{len(prompts)} prompts")
    return torch.cat(all_embs, dim=0)   # (N, 77, 768)


def encode_null(pipe, device, dtype) -> torch.Tensor:
    """Encode empty string "" — the correct CFG null conditioning for SD v1.5."""
    print("  [sd15] encoding null conditioning (\"\") …")
    with torch.no_grad():
        emb = pipe.encode_prompt(
            [""],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
    return emb.cpu().to(dtype)   # (1, 77, 768)


def load_coco_captions(n: int = 300, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    try:
        from datasets import load_dataset
        ds    = load_dataset("phiyodr/coco2017", split="validation", token=HF_TOKEN)
        capts = [c for row in ds for c in row["captions"]][:n * 5]
        idx   = rng.choice(len(capts), size=min(n, len(capts)), replace=False)
        caps  = [capts[i] for i in idx]
        print(f"  [prompts] COCO2017: {len(caps)} captions")
        return caps
    except Exception as e:
        print(f"  [prompts] COCO failed ({e}), using fallback …")
        return ["a photo of a dog", "a city street at night",
                "a mountain landscape", "a bowl of fruit"] * (n // 4 + 1)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt bank
# ══════════════════════════════════════════════════════════════════════════════

class PromptBank:
    """Holds pre-encoded CLIP embeddings + null embedding."""
    def __init__(self, cond_embs: torch.Tensor, null_emb: torch.Tensor,
                 device, dtype, seed: int = 0):
        self.cond    = cond_embs   # (N, 77, 768) on CPU
        self._null   = null_emb    # (1, 77, 768) on CPU
        self.N       = cond_embs.shape[0]
        self.device  = device
        self.dtype   = dtype
        self.rng     = np.random.default_rng(seed)

    def sample(self, B: int) -> torch.Tensor:
        idx = self.rng.integers(0, self.N, size=B)
        return self.cond[idx].to(device=self.device, dtype=self.dtype)

    def null(self, B: int) -> torch.Tensor:
        return self._null.expand(B, -1, -1).to(device=self.device, dtype=self.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# UNet forward  (VP-SDE: integer timestep index)
# ══════════════════════════════════════════════════════════════════════════════

def unet_eps(unet, x_t: torch.Tensor, t_idx: int,
             enc: torch.Tensor) -> torch.Tensor:
    """Single forward pass, returns ε_θ(x_t, t, c)."""
    B      = x_t.shape[0]
    t_tens = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)
    out    = unet(x_t, t_tens, encoder_hidden_states=enc, return_dict=False)
    return out[0]


def cfg_eps(unet, x_t: torch.Tensor, t_idx: int,
            enc_null: torch.Tensor, enc_cond: torch.Tensor,
            cfg_scale: float) -> torch.Tensor:
    """CFG-combined noise prediction: ε_null + w*(ε_cond - ε_null)."""
    e_null = unet_eps(unet, x_t, t_idx, enc_null)
    e_cond = unet_eps(unet, x_t, t_idx, enc_cond)
    return e_null + cfg_scale * (e_cond - e_null)


# ══════════════════════════════════════════════════════════════════════════════
# Statistics estimators  (all in λ / VP-SDE convention)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_eta(unet, t_idx, alpha_t, sigma_t, d,
                   x0_batch, bank: PromptBank, cfg_scale) -> float:
    """σ²_η(λ) = (1/d) E[||ε_θ^CFG - ε||²]"""
    B        = x0_batch.shape[0]
    enc_cond = bank.sample(B)
    enc_null = bank.null(B)
    eps      = torch.randn_like(x0_batch)
    x_t      = alpha_t * x0_batch + sigma_t * eps
    eps_hat  = cfg_eps(unet, x_t, t_idx, enc_null, enc_cond, cfg_scale)
    err      = eps_hat.float() - eps.float()
    return float((err ** 2).sum()) / (B * d)


@torch.no_grad()
def est_sigma2_gpp(unet, t_idx, lambdas_dense, alphas, sigmas, d,
                   x0_batch, bank: PromptBank, cfg_scale,
                   delta_t: int = 2) -> float:
    """
    σ²_g''(λ) = (1/d) E[||∂²_λ [e^{-λ} ε_θ^CFG]||²]
    Three-point central FD, same ε across lo/mid/hi (path-coherent).
    """
    T     = len(lambdas_dense)
    t_lo  = max(0, t_idx - delta_t)
    t_hi  = min(T - 1, t_idx + delta_t)
    t_mid = t_idx
    lam_lo  = float(lambdas_dense[t_lo])
    lam_hi  = float(lambdas_dense[t_hi])
    lam_mid = float(lambdas_dense[t_mid])
    h_fd    = (lam_hi - lam_lo) / 2.0
    if abs(h_fd) < 1e-8 or t_lo == t_hi:
        return 0.0

    B        = x0_batch.shape[0]
    enc_cond = bank.sample(B)
    enc_null = bank.null(B)
    eps      = torch.randn_like(x0_batch)

    x_lo  = float(alphas[t_lo])  * x0_batch + float(sigmas[t_lo])  * eps
    x_mid = float(alphas[t_mid]) * x0_batch + float(sigmas[t_mid]) * eps
    x_hi  = float(alphas[t_hi])  * x0_batch + float(sigmas[t_hi])  * eps

    g_lo  = np.exp(-lam_lo)  * cfg_eps(unet, x_lo,  t_lo,  enc_null, enc_cond, cfg_scale).float()
    g_mid = np.exp(-lam_mid) * cfg_eps(unet, x_mid, t_mid, enc_null, enc_cond, cfg_scale).float()
    g_hi  = np.exp(-lam_hi)  * cfg_eps(unet, x_hi,  t_hi,  enc_null, enc_cond, cfg_scale).float()

    gpp = (g_hi - 2.0 * g_mid + g_lo) / (h_fd ** 2)
    return float((gpp ** 2).sum()) / (B * d)


def est_g_at_t(unet, t_idx, alpha_t, sigma_t, d,
               x0_list, bank: PromptBank, cfg_scale,
               n_probes: int = 4, delta: float = 0.01) -> float:
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

@torch.no_grad()
def est_score_corr(unet, t_indices, lambdas_dense, alphas, sigmas, d,
                   x0_batch, bank: PromptBank, cfg_scale) -> tuple:
    """
    K_ξ(λ_i, λ_j) = (1/d) E[<η(λ_i), η(λ_j)>]
    Independent ε per λ — eliminates shared-ε floor artefact.
    """
    M = len(t_indices)
    B = x0_batch.shape[0]
    C = np.zeros((M, M), dtype=np.float64)
    n = 0

    for xi in x0_batch:
        xi   = xi.unsqueeze(0)
        errs = []
        enc_cond = bank.sample(1)
        enc_null = bank.null(1)
        for t_idx in t_indices:  
            eps_m    = torch.randn_like(xi)
            x_t      = float(alphas[t_idx]) * xi + float(sigmas[t_idx]) * eps_m
            eps_hat  = cfg_eps(unet, x_t, t_idx, enc_null, enc_cond, cfg_scale)
            e = (eps_hat.float() - eps_m.float()).squeeze(0).flatten().cpu().double()
            errs.append(e)
        for i in range(M):
            for j in range(i, M):
                dot = (errs[i] * errs[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n += 1
        if n % 16 == 0:
            print(f"    score corr: {n}/{B} samples")

    C /= max(n, 1)
    diag = np.diag(C).clip(1e-30)
    rho  = np.clip(C / np.sqrt(np.outer(diag, diag)), -1.0, 1.0)

    lams   = lambdas_dense.cpu().numpy()[t_indices]
    dl, rp = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl.append(abs(lams[i] - lams[j]))
            rp.append(float(rho[i, j]))
    eigvals = np.linalg.eigvalsh(C)
    eigvals_sorted = np.sort(eigvals)[::-1]
    total = np.sum(np.abs(eigvals_sorted)) + 1e-12

    print("\n Score error Spectrum summary:")
    print(f"  λ1 / sum(|λ|) = {eigvals_sorted[0] / total:.4f}")
    print(f"  λ2 / λ1       = {eigvals_sorted[1] / eigvals_sorted[0]:.4f}")
    print(f"  λ3 / λ1       = {eigvals_sorted[2] / eigvals_sorted[0]:.4f}")
    return C, rho, np.array(dl), np.array(rp)


@torch.no_grad()
def est_gpp_corr(unet, t_indices, lambdas_dense, alphas, sigmas, d,
                 x0_batch, bank: PromptBank, cfg_scale,
                 delta_t: int = 2) -> tuple:
    """g'' cross-step correlation — same structure as estimate_model_stats.py."""
    M  = len(t_indices)
    B  = x0_batch.shape[0]
    C  = np.zeros((M, M), dtype=np.float64)
    T  = len(lambdas_dense)
    n  = 0

    for xi in x0_batch:
        xi      = xi.unsqueeze(0)
        gpp_vecs = []
        enc_cond = bank.sample(1)
        enc_null = bank.null(1)
        for t_idx in t_indices:
            t_lo  = max(0, t_idx - delta_t)
            t_hi  = min(T - 1, t_idx + delta_t)
            t_mid = t_idx
            lam_lo  = float(lambdas_dense[t_lo])
            lam_hi  = float(lambdas_dense[t_hi])
            lam_mid = float(lambdas_dense[t_mid])
            h_fd    = (lam_hi - lam_lo) / 2.0
            if abs(h_fd) < 1e-8 or t_lo == t_hi:
                gpp_vecs.append(None); continue

            
            eps_m    = torch.randn_like(xi)
            x_lo  = float(alphas[t_lo])  * xi + float(sigmas[t_lo])  * eps_m
            x_mid = float(alphas[t_mid]) * xi + float(sigmas[t_mid]) * eps_m
            x_hi  = float(alphas[t_hi])  * xi + float(sigmas[t_hi])  * eps_m

            g_lo  = np.exp(-lam_lo)  * cfg_eps(unet, x_lo,  t_lo,  enc_null, enc_cond, cfg_scale).float()
            g_mid = np.exp(-lam_mid) * cfg_eps(unet, x_mid, t_mid, enc_null, enc_cond, cfg_scale).float()
            g_hi  = np.exp(-lam_hi)  * cfg_eps(unet, x_hi,  t_hi,  enc_null, enc_cond, cfg_scale).float()
            gpp   = (g_hi - 2.0 * g_mid + g_lo) / (h_fd ** 2)
            gpp_vecs.append(gpp.squeeze(0).flatten().cpu().double())

        for i in range(M):
            for j in range(i, M):
                if gpp_vecs[i] is None or gpp_vecs[j] is None: continue
                dot = (gpp_vecs[i] * gpp_vecs[j]).sum().item() / d
                C[i, j] += dot; C[j, i] += dot
        n += 1
        if n % 8 == 0:
            print(f"    g'' corr: {n}/{B} samples")

    C /= max(n, 1)
    diag = np.diag(C).clip(1e-30)
    rho  = np.clip(C / np.sqrt(np.outer(diag, diag)), -1.0, 1.0)
    lams = lambdas_dense.cpu().numpy()[t_indices]
    dl, rp = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl.append(abs(lams[i] - lams[j]))
            rp.append(float(rho[i, j]))
    return C, rho, np.array(dl), np.array(rp)


# ══════════════════════════════════════════════════════════════════════════════
# ρ curve extraction  (shared helper)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_curve(dl, rp, n_bins=40, anchor_at_one=True):
    valid = dl > 0
    if valid.sum() < 5:
        return np.linspace(0, 1, 5), np.ones(5)
    edges = np.unique(np.percentile(dl[valid], np.linspace(0, 100, n_bins + 1)))
    sl, vl = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (dl >= lo) & (dl < hi)
        if mask.sum() < 3: continue
        sl.append(float(np.median(dl[mask])))
        vl.append(float(np.median(rp[mask])))
    if not sl:
        return np.array([0.0, dl.max()]), np.array([1.0, 0.5])
    sa, va = np.array(sl), np.array(vl)
    sa = np.concatenate([[0.0], sa])
    va = np.concatenate([[1.0 if anchor_at_one else va[0]], va])
    log_v = np.log(np.clip(va, 1e-10, 1.0))
    for k in range(1, len(log_v)):
        if log_v[k] > log_v[k - 1]:
            log_v[k] = log_v[k - 1]
    va    = np.exp(log_v)
    pchip = PchipInterpolator(sa, va, extrapolate=False)
    so    = np.linspace(sa[0], sa[-1], 200)
    return so, np.clip(pchip(so), va[-1], va[0])


# ══════════════════════════════════════════════════════════════════════════════
# Plateau / stationarity analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_plateau(rho_s, rho_vals, label=""):
    """ρ_∞, ell_res, rank-1 fraction — identical logic to estimate_model_stats.py."""
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.maximum(np.abs(rho_vals[:-1]), 0.05))
    MIN_FLAT = max(5, len(rho_vals) // 10)
    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i:i + MIN_FLAT] < 0.02):
            plateau_start = i; break
    if plateau_start is None:
        warnings.warn(f"[{label}] No plateau — extend rho_s range.")
        plateau_start = int(0.5 * len(rho_vals))

    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i; break

    plateau_vals = rho_vals[plateau_start:plateau_end]
    rho_infty    = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))
    cv           = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(f"[{label}] Plateau CV={cv:.3f} — ρ_∞ unreliable.")

    rho_res  = np.maximum(rho_vals - rho_infty, 0.0)
    frac_r1  = rho_infty / max(float(rho_vals[0]), 1e-8)
    rho_norm = rho_res / max(rho_res[0], 1e-8)
    cross    = np.where(rho_norm <= 1.0 / np.e)[0]
    ell_res  = float(rho_s[cross[0]]) if len(cross) else float(rho_s[-1])
    spec_mass = float(np.trapezoid(rho_res, rho_s))

    print(f"\n  ── Plateau [{label}] ────────────────────")
    print(f"     ρ_∞={rho_infty:.4f}  rank-1 frac={frac_r1:.3f}  ℓ_res={ell_res:.4f}")
    verdict = "STRONG rank-1" if frac_r1 > 0.3 else "WEAK rank-1"
    print(f"     → {verdict}")
    return dict(rho_infty=rho_infty, frac_rank1=frac_r1,
                ell_res=ell_res, spectral_mass_res=spec_mass, verdict=verdict)


def check_epsilon_validity(lambdas_grid, g_vals, nfe_list=(10, 20, 50)):
    """
    Check whether |ε_k| << 1 (Born validity) for each NFE.
    ε_k ≈ -α_s · I_full · g(λ_s) · φ_k
    In the high-noise limit: |ε| ≈ e^{h/2}(1-e^{-h}) · g
    Flags NFE settings that risk propagation-chain collapse.
    """
    import math
    span  = lambdas_grid[-1] - lambdas_grid[0]
    g_eff = float(np.percentile(np.abs(g_vals[g_vals != 0]), 90)) \
            if np.any(g_vals != 0) else 0.0
    print(f"\n  ── Born validity check (λ span={span:.2f}, p90|g|={g_eff:.3f}) ──")
    print(f"  {'NFE':>5}  {'h_typ':>8}  {'|ε|_est':>9}  {'safe?':>10}")
    for nfe in nfe_list:
        h    = span / nfe
        val  = math.exp(h / 2) * (1 - math.exp(-h)) * g_eff
        safe = "✓ |ε|<<1" if val < 0.5 else ("~ marginal" if val < 1.0 else "✗ chain risk")
        print(f"  {nfe:>5}  {h:>8.4f}  {val:>9.4f}  {safe:>10}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════
def _visualize(lambdas_grid, g_vals, g_sem_vals, g_ratio, s2_eta, s2_gpp,
               rho_s, rho_vals, rho_s_gpp, rho_vals_gpp,
               plateau_score, plateau_gpp,
               seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
               model_name, cfg_scale, out_path):
    fig = plt.figure(figsize=(18, 12))
    gs  = plt.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
    seg_colors = ["#e05c00", "#2d6a9f", "#2a7d4f", "#9b3fa0"]

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lambdas_grid, g_vals, color="#2d6a9f", lw=2, label="g_iso")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("λ"); ax.set_ylabel("g(λ)")
    ax.set_title(f"Scalar Jacobian Proxy  g(λ)\n[CFG w={cfg_scale}]")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(lambdas_grid, s2_eta,  color="#b5451b", lw=2, label="σ²_η (CFG err)")
    ax.semilogy(lambdas_grid, s2_gpp,  color="#9b3fa0", lw=2, ls="--", label="σ²_g''")
    ax.set_xlabel("λ"); ax.set_title("Error Variances  σ²_η & σ²_g''")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(rho_s, rho_vals, color="#2d6a9f", lw=2.5, label="ρ̂_η(s)")
    if plateau_score:
        ax.axhline(plateau_score["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                   label=f"ρ_∞={plateau_score['rho_infty']:.3f}")
    ax.set_xlabel("|Δλ|"); ax.set_ylabel("ρ̂")
    ax.set_title("Score-Error Correlation ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_common, cv_s, color="#555", lw=2)
    ax.axhline(0.15, color="#2a7d4f", lw=1.2, ls="--", label="0.15 stationary")
    ax.axhline(0.35, color="#d12",    lw=1.2, ls="--", label="0.35 non-stat.")
    vc = "#d12" if "STRONGLY" in stat_verdict else ("#e08000" if "MILDLY" in stat_verdict else "#2a7d4f")
    ax.set_title(f"Stationarity CV  mean={mean_cv:.3f}\n{stat_verdict}", color=vc, fontsize=9)
    ax.set_xlabel("s"); ax.set_ylabel("CV")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rho_s, rho_vals, color="k", lw=2, ls="--", alpha=0.5, label="global")
    for k, (rs, rv) in enumerate(seg_curves):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        ax.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                label=f"λ̄∈[{lo:.1f},{hi:.1f})")
    ax.set_xlabel("|Δλ|"); ax.set_title("Per-Segment ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)

    # ── gs[1,2]: anisotropy (replaces cost integrand) ──────────────────────
    ax = fig.add_subplot(gs[1, 2])
    has_g = np.any(g_vals != 0)
    if has_g:
        ax2 = ax.twinx()
        ax.plot(lambdas_grid, g_vals,     color="#2d6a9f", lw=2,   label="g_iso")
        ax.plot(lambdas_grid, g_sem_vals, color="#b5451b", lw=2,   label="g_sem", ls="--")
        ax2.plot(lambdas_grid, g_ratio,   color="#2a7d4f", lw=1.5, label="ratio", ls=":")
        ax2.axhline(1.0, color="#2a7d4f", lw=0.8, ls="--", alpha=0.5)
        ax2.set_ylabel("g_sem / g_iso", color="#2a7d4f", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#2a7d4f")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "g not estimated\n(--no_g)", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("λ"); ax.set_title("Anisotropy  g_iso vs g_sem")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    if rho_s_gpp is not None and len(rho_s_gpp) > 1:
        ax.plot(rho_s_gpp, rho_vals_gpp, color="#9b3fa0", lw=2.5, label="ρ̂_g''(s)")
        if plateau_gpp:
            ax.axhline(plateau_gpp["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                       label=f"ρ_∞={plateau_gpp['rho_infty']:.3f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "g'' corr\nnot estimated", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("|Δλ|"); ax.set_title("g'' Correlation ρ̂_g''(s)")
    ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    labels = ["score-error ρ_∞"]; vals = [plateau_score["rho_infty"] if plateau_score else 0]
    if plateau_gpp:
        labels.append("g'' ρ_∞"); vals.append(plateau_gpp["rho_infty"])
    bars = ax.bar(labels, vals, color=["#2d6a9f", "#9b3fa0"][:len(labels)], width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    ax.axhline(0.3, color="#e05c00", lw=1.2, ls="--", label="0.3 rank-1 threshold")
    ax.set_ylim(0, 1.05); ax.set_title("Rank-1 Plateau Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ps = plateau_score or {}; pg = plateau_gpp or {}
    txt = (
        f"Model: {model_name}\nCFG scale: {cfg_scale}\n{'─'*35}\n"
        f"Score-error kernel:\n"
        f"  ρ_∞       = {ps.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 f. = {ps.get('frac_rank1', float('nan')):.3f}\n"
        f"  ℓ_res     = {ps.get('ell_res', float('nan')):.4f}\n\n"
        + (f"g'' kernel:\n  ρ_∞ = {pg.get('rho_infty', float('nan')):.4f}\n"
           f"  → {pg.get('verdict','N/A')}\n\n" if plateau_gpp else "g'' corr: not estimated\n\n")
        + f"Stationarity: {stat_verdict}  CV={mean_cv:.3f}\n"
    )
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            family="monospace", bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaa"))
    ax.set_title("Summary")

    fig.suptitle(f"BornSchedule Statistics (SD v1.5 VP-SDE+CFG) — {model_name}  w={cfg_scale}",
                 fontsize=13, y=1.005)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_estimation(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype  = torch.float32
    print(f"[stats_sd15] device={device}  model={args.model}  cfg={args.cfg_scale}")

    # ── Noise schedule ────────────────────────────────────────────────────────
    print("[stats_sd15] loading noise schedule …")
    _, alphas, sigmas, lambdas_dense = alphas_from_scheduler(args.model, device)
    T          = len(lambdas_dense)
    lam_all    = lambdas_dense.cpu().numpy()
    lambda_min = float(lambdas_dense[-1])
    lambda_max = float(lambdas_dense[1])   # t=0 is pure noise end in diffusers
    print(f"  λ ∈ [{lambda_min:.3f}, {lambda_max:.3f}]  T={T}")

    # λ grid: denser in high-σ²_η region (low λ)
    n_dense  = args.n_lambda // 3
    n_coarse = args.n_lambda - n_dense
    lam_split  = lambda_min + 0.25 * (lambda_max - lambda_min)
    lam_dense  = np.linspace(lambda_min, lam_split, n_dense,  endpoint=False)
    lam_coarse = np.linspace(lam_split,  lambda_max, n_coarse)
    lam_grid   = np.concatenate([lam_dense, lam_coarse])

    t_indices    = np.array([int((lambdas_dense - lam).abs().argmin().item())
                             for lam in lam_grid])
    lambdas_grid = lam_all[t_indices]

    # ── Load model ────────────────────────────────────────────────────────────
    unet, pipe = load_pipeline(args.model, device, dtype)

    in_ch     = int(unet.config.get("in_channels", 4))
    sample_sz = int(unet.config.get("sample_size", 64))
    if hasattr(sample_sz, "__len__"): sample_sz = sample_sz[0]
    d = in_ch * sample_sz * sample_sz
    print(f"  UNet latent: {in_ch}×{sample_sz}×{sample_sz}  d={d:,}")

    # ── Text embeddings ───────────────────────────────────────────────────────
    captions = load_coco_captions(n=args.n_prompts, seed=args.seed)
    cond_embs = encode_prompts(pipe, captions, device, dtype, batch_size=args.encode_batch)
    null_emb  = encode_null(pipe, device, dtype)

    bank = PromptBank(cond_embs, null_emb, device, dtype, seed=args.seed)

    # Offload text encoder to save VRAM
    for attr in ("text_encoder", "tokenizer", "vae", "safety_checker"):
        obj = getattr(pipe, attr, None)
        if obj is not None and hasattr(obj, "cpu"):
            obj.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Random latents ────────────────────────────────────────────────────────
    print(f"[stats_sd15] generating {args.n_samples} latents …")
    x0_all = torch.randn(args.n_samples, in_ch, sample_sz, sample_sz,
                         device=device, dtype=dtype)

    n_lam = len(lambdas_grid)

    # ── σ²_η(λ) ──────────────────────────────────────────────────────────────
    print(f"\n[stats_sd15] σ²_η at {n_lam} λ points …")
    s2_eta = np.zeros(n_lam)
    for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb = x0_all[start:start + args.batch]
            acc += est_sigma2_eta(unet, t_idx,
                                  float(alphas[t_idx]), float(sigmas[t_idx]),
                                  d, xb, bank, args.cfg_scale)
            nb += 1
        s2_eta[k] = acc / nb
        if (k + 1) % max(1, n_lam // 8) == 0:
            print(f"  σ²_η  {k+1}/{n_lam}  λ={lam:.3f}  val={s2_eta[k]:.4e}")

    # ── σ²_g''(λ) ────────────────────────────────────────────────────────────
    print(f"\n[stats_sd15] σ²_g'' at {n_lam} λ points …")
    s2_gpp = np.zeros(n_lam)
    for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb = x0_all[start:start + args.batch]
            acc += est_sigma2_gpp(unet, t_idx, lambdas_dense, alphas, sigmas,
                                  d, xb, bank, args.cfg_scale, delta_t=args.delta_t)
            nb += 1
        s2_gpp[k] = acc / nb
        if (k + 1) % max(1, n_lam // 8) == 0:
            print(f"  σ²_g''  {k+1}/{n_lam}  λ={lam:.3f}  val={s2_gpp[k]:.4e}")
    wl        = min(7, (n_lam // 4) * 2 + 1)
    s2_gpp_sm = np.exp(savgol_filter(np.log(np.clip(s2_gpp, 1e-30, None)), wl, 2))

    # ── g(λ) ─────────────────────────────────────────────────────────────────
    g_iso_vals = np.zeros(n_lam)
    g_sem_vals = np.zeros(n_lam)
    g_ratio    = np.zeros(n_lam)
    if not args.no_g:
        n_g    = min(args.n_g_samples, args.n_samples)
        x0_g   = [x0_all[i] for i in range(n_g)]
        print(f"\n[stats_sd15] g(λ)  n_g={n_g}  n_probes={args.n_hutchinson} …")
        for k, (t_idx, lam) in enumerate(zip(t_indices, lambdas_grid)):
            g_iso_vals[k], g_sem_vals[k], g_ratio[k] = est_g_at_t(unet, t_idx,
                                    float(alphas[t_idx]), float(sigmas[t_idx]),
                                    d, x0_g, bank, args.cfg_scale,
                                    n_probes=args.n_hutchinson, delta=args.g_delta)
            if (k + 1) % max(1, n_lam // 8) == 0:
                print(f"  g  {k+1}/{n_lam}  λ={lam:.3f}  g={g_iso_vals[k]:.5f}")

    # ── Born validity check ───────────────────────────────────────────────────
    check_epsilon_validity(lambdas_grid, g_iso_vals, nfe_list=(10, 20, 50))

    # ── Score-error correlation ───────────────────────────────────────────────
    n_ell  = min(args.n_ell_samples, args.n_samples)
    x0_ell = x0_all[:n_ell]
    print(f"\n[stats_sd15] score-error corr  ({n_ell} samples) …")
    _, _, dl_score, rp_score = est_score_corr(
        unet, t_indices, lambdas_dense, alphas, sigmas,
        d, x0_ell, bank, args.cfg_scale)
    rho_s, rho_vals = _extract_rho_curve(dl_score, rp_score, anchor_at_one=True)
    plateau_score   = analyse_plateau(rho_s, rho_vals, "score-error")

    # ── g'' correlation ───────────────────────────────────────────────────────
    rho_s_gpp = rho_vals_gpp = None
    plateau_gpp = None
    if not args.no_gpp_corr:
        print(f"\n[stats_sd15] g'' corr  ({n_ell} samples) …")
        _, _, dl_gpp, rp_gpp = est_gpp_corr(
            unet, t_indices, lambdas_dense, alphas, sigmas,
            d, x0_ell, bank, args.cfg_scale, delta_t=args.delta_t)
        rho_s_gpp, rho_vals_gpp = _extract_rho_curve(dl_gpp, rp_gpp, anchor_at_one=False)
        plateau_gpp = analyse_plateau(rho_s_gpp, rho_vals_gpp, "g''")

    # ── Stationarity ──────────────────────────────────────────────────────────
    n_segs     = 3
    seg_bounds = np.linspace(lambdas_grid[0], lambdas_grid[-1], n_segs + 1)
    lbar_all   = np.array([0.5 * (lambdas_grid[i] + lambdas_grid[j])
                           for i in range(n_lam) for j in range(i + 1, n_lam)])
    seg_curves = []
    print(f"\n[stats_sd15] stationarity ({n_segs} segments) …")
    for k in range(n_segs):
        lo, hi   = seg_bounds[k], seg_bounds[k + 1]
        mask     = (lbar_all >= lo) & (lbar_all < hi)
        dl_seg   = dl_score[mask[:len(dl_score)]] if mask.sum() > 10 else dl_score
        rp_seg   = rp_score[mask[:len(rp_score)]] if mask.sum() > 10 else rp_score
        rs, rv   = _extract_rho_curve(dl_seg, rp_seg, anchor_at_one=True)
        seg_curves.append((rs, rv))
        print(f"  λ̄∈[{lo:.2f},{hi:.2f})  ρ̂(0)={rv[0]:.3f}  ρ̂(end)={rv[-1]:.4f}")

    s_common = rho_s[1:]
    seg_mats = []
    for rs, rv in seg_curves:
        pchip = PchipInterpolator(rs, rv, extrapolate=False)
        vi    = np.where(np.isnan(pchip(s_common)), rv[-1], pchip(s_common))
        seg_mats.append(np.clip(vi, 0.0, 1.0))
    seg_mat  = np.array(seg_mats)
    cv_s     = seg_mat.std(0) / (seg_mat.mean(0) + 1e-8)
    valid_s  = rho_vals[1:] > 0.1
    mean_cv  = float(cv_s[valid_s].mean()) if valid_s.any() else float("nan")

    if   np.isnan(mean_cv):  stat_verdict = "UNKNOWN"
    elif mean_cv < 0.15:     stat_verdict = "STATIONARY"
    elif mean_cv < 0.35:     stat_verdict = "MILDLY NON-STATIONARY"
    else:                    stat_verdict = "STRONGLY NON-STATIONARY"
    print(f"  mean CV={mean_cv:.3f}  →  {stat_verdict}")

    # ── Save ──────────────────────────────────────────────────────────────────
    safe = args.model.replace("/", "--").replace(":", "--")
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz  = out / f"{safe}_cfg{args.cfg_scale}.npz"

    save = dict(
        lambda_grid       = lambdas_grid.astype(np.float32),
        g_values          = g_iso_vals.astype(np.float32),
        sigma2_values     = s2_eta.astype(np.float32),
        sigma2_gpp_values = s2_gpp_sm.astype(np.float32),
        lambda_min        = np.float32(lambda_min),
        lambda_max        = np.float32(lambda_max),
        rho_s             = rho_s.astype(np.float32),
        rho_values        = rho_vals.astype(np.float32),
        cfg_scale         = np.float32(args.cfg_scale),
    )
    save.update(dict(
        g_iso_values = g_iso_vals.astype(np.float32),
        g_sem_values = g_sem_vals.astype(np.float32),
        g_ratio      = g_ratio.astype(np.float32),
    ))
    if rho_s_gpp is not None:
        save["rho_s_gpp"]      = rho_s_gpp.astype(np.float32)
        save["rho_values_gpp"] = rho_vals_gpp.astype(np.float32)

    np.savez(str(npz), **save)
    print(f"\n[stats_sd15] saved → {npz}")

    _visualize(lambdas_grid, g_iso_vals, g_sem_vals, g_ratio, s2_eta, s2_gpp_sm,
               rho_s, rho_vals, rho_s_gpp, rho_vals_gpp,
               plateau_score, plateau_gpp,
               seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
               args.model, args.cfg_scale, npz.with_suffix(".png"))

    print("\n══ Summary ══════════════════════════════════════════════════")
    print(f"  model       : {args.model}")
    print(f"  CFG scale   : {args.cfg_scale}")
    print(f"  d           : {d:,}  (latent {in_ch}×{sample_sz}×{sample_sz})")
    print(f"  λ range     : [{lambda_min:.3f}, {lambda_max:.3f}]")
    print(f"  σ²_η range  : [{s2_eta.min():.2e}, {s2_eta.max():.2e}]")
    print(f"  score ρ_∞   : {plateau_score['rho_infty']:.4f}")
    if plateau_gpp:
        print(f"  g'' ρ_∞     : {plateau_gpp['rho_infty']:.4f}")
    print(f"  stationarity: {stat_verdict}  (CV={mean_cv:.3f})")
    print(f"  output      : {npz}")
    print("═════════════════════════════════════════════════════════════\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BornSchedule statistics for SD v1.5 (VP-SDE + CFG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",        default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir",   default=str(Path.home() / ".cache" / "opt_schedule"))
    p.add_argument("--cfg_scale",    type=float, default=7.5)
    p.add_argument("--n_lambda",     type=int, default=60)
    p.add_argument("--n_samples",    type=int, default=256)
    p.add_argument("--n_ell_samples",type=int, default=128)
    p.add_argument("--n_g_samples",  type=int, default=32)
    p.add_argument("--n_hutchinson", type=int, default=4)
    p.add_argument("--n_prompts",    type=int, default=300)
    p.add_argument("--encode_batch", type=int, default=8)
    p.add_argument("--batch",        type=int, default=4)
    p.add_argument("--g_delta",      type=float, default=0.01)
    p.add_argument("--delta_t",      type=int, default=2,
                   help="timestep offset for g'' FD (larger = smoother but less local)")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       default=None)
    p.add_argument("--no_g",         action="store_true")
    p.add_argument("--no_gpp_corr",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_estimation(args)