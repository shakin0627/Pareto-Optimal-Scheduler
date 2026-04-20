"""
stats_sd3.py  —  Offline BornSchedule statistics for SD3 / SD3.5 (rectified flow, CFG).

Coordinate: σ ∈ [0, 1]   (σ=1 : pure noise,  σ=0 : clean data)
Linear FM interpolation:  x_σ = (1–σ)·x₀ + σ·ε
FM target velocity:       v*  = ε – x₀

SD3 specifics vs. FLUX / BAGEL
────────────────────────────────

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
  python stats_sd3.py \\
      --model stabilityai/stable-diffusion-3.5-large \\
      --latent_size 64 --n_sigma 50 --n_samples 128 \\
      --n_ell_samples 64 --cfg_scale 4.5 --n_prompts 300

  # For SD3 Medium (2B):
  python stats_sd3.py --model stabilityai/stable-diffusion-3-medium-diffusers
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_sd3_pipeline(model_name: str, device: torch.device, dtype):
    from diffusers import StableDiffusion3Pipeline
    print(f"  [sd3] loading pipeline from {model_name} …")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name, torch_dtype=dtype, token=HF_TOKEN,
    ).to(device)
    tr = pipe.transformer
    tr.eval()
    for m in tr.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    print(f"  [sd3] loaded  transformer params: "
          f"{sum(p.numel() for p in tr.parameters()) / 1e9:.1f}B")
    return tr, pipe


# ══════════════════════════════════════════════════════════════════════════════
# Text encoding  (SD3: CLIP-L + CLIP-G + T5-XXL via diffusers encode_prompt)
# ══════════════════════════════════════════════════════════════════════════════

def _call_encode_prompt(pipe, prompts: list, device, dtype, max_seq: int):
    """
    Thin wrapper around StableDiffusion3Pipeline.encode_prompt.

    SD3 encode_prompt returns a 4-tuple when called with default args:
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds)

    Returns (te, pe) on the same device, in the requested dtype.
    """
    with torch.no_grad():
        out = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            # do NOT pass negative_prompt here so we always get positive embeds
        )
    # Unpack — diffusers SD3 always returns 4 items
    if isinstance(out, (tuple, list)) and len(out) == 4:
        te, _, pe, _ = out
    elif isinstance(out, (tuple, list)) and len(out) == 2:
        te, pe = out
    else:
        te = out.prompt_embeds
        pe = out.pooled_prompt_embeds
    return te.to(dtype=dtype), pe.to(dtype=dtype)


def encode_prompts_sd3(
    pipe,
    prompts:    list,
    device:     torch.device,
    dtype:      torch.dtype,
    batch_size: int = 4,
    max_seq:    int = 256,
) -> tuple:
    """
    Encode a list of prompts.  Returns (enc_all, pool_all) on CPU.
        enc_all  : (N, seq, 4096)
        pool_all : (N, 2048)
    """
    enc_list, pool_list = [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        te, pe = _call_encode_prompt(pipe, batch, device, dtype, max_seq)
        enc_list.append(te.cpu())
        pool_list.append(pe.cpu())
        if start % max(batch_size, 16) == 0:
            print(f"    encoded {start + len(batch)}/{len(prompts)} prompts")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    enc_all  = torch.cat(enc_list,  dim=0)
    pool_all = torch.cat(pool_list, dim=0)
    print(f"  [prompts] enc {tuple(enc_all.shape)}  pool {tuple(pool_all.shape)}")
    return enc_all, pool_all


def encode_null_sd3(pipe, device, dtype, max_seq: int = 256) -> tuple:
    """
    Encode the empty string "" as the unconditional (null) conditioning.

    IMPORTANT: SD3 was trained with CFG using encoded-"" as the negative
    prompt, not zero embeddings.  Using zeros would bias all statistics
    because zero lies off the training-data manifold.
    """
    print("  [sd3] encoding null conditioning (\"\") …")
    te, pe = _call_encode_prompt(pipe, [""], device, dtype, max_seq)
    # te: (1, seq, 4096),  pe: (1, 2048)
    return te.cpu(), pe.cpu()


# ══════════════════════════════════════════════════════════════════════════════
# Prompt bank  (SD3-specific: holds both cond embeddings AND cached null)
# ══════════════════════════════════════════════════════════════════════════════

class PromptBankSD3:
    """
    Stores pre-encoded COCO prompt embeddings + the null ("") embedding.

    Calling .sample(B) draws B random cond embeddings.
    Calling .null(B)   returns the null embedding tiled to batch size B.
    """
    def __init__(
        self,
        enc_all:   torch.Tensor,   # (N, seq, 4096) on CPU
        pool_all:  torch.Tensor,   # (N, 2048) on CPU
        null_enc:  torch.Tensor,   # (1, seq, 4096) on CPU
        null_pool: torch.Tensor,   # (1, 2048) on CPU
        device:    torch.device,
        dtype:     torch.dtype,
        seed:      int = 0,
    ):
        self.enc    = enc_all
        self.pool   = pool_all
        self._null_enc  = null_enc    # (1, seq, 4096)
        self._null_pool = null_pool   # (1, 2048)
        self.N      = enc_all.shape[0]
        self.txt_seq  = enc_all.shape[1]
        self.enc_dim  = enc_all.shape[2]
        self.pool_dim = pool_all.shape[1]
        self.device = device
        self.dtype  = dtype
        self.rng    = np.random.default_rng(seed)

    def sample(self, B: int) -> tuple:
        """Return B randomly sampled cond embeddings on device."""
        idx  = self.rng.integers(0, self.N, size=B)
        enc  = self.enc[idx].to(device=self.device,  dtype=self.dtype)
        pool = self.pool[idx].to(device=self.device, dtype=self.dtype)
        return enc, pool

    def null(self, B: int) -> tuple:
        """Return null ("") embedding tiled to batch size B on device."""
        enc  = self._null_enc.expand(B, -1, -1).to(device=self.device, dtype=self.dtype)
        pool = self._null_pool.expand(B, -1).to(device=self.device,    dtype=self.dtype)
        return enc, pool


# ══════════════════════════════════════════════════════════════════════════════
# Prompt loading (COCO / Flickr30k)   — unchanged from stats_bagel.py
# ══════════════════════════════════════════════════════════════════════════════

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
        print(f"  [prompts] COCO failed ({e}), trying Flickr30k …")
    try:
        from datasets import load_dataset
        ds    = load_dataset("nlphuji/flickr30k", split="test", token=HF_TOKEN)
        capts = [c for row in ds for c in row["caption"]][:n * 5]
        idx   = rng.choice(len(capts), size=min(n, len(capts)), replace=False)
        caps  = [capts[i] for i in idx]
        print(f"  [prompts] Flickr30k: {len(caps)} captions")
        return caps
    except Exception as e:
        print(f"  [prompts] Flickr30k failed ({e}), using empty-string fallback.")
        return [""] * min(n, 64)


# ══════════════════════════════════════════════════════════════════════════════
# Single-step velocity  (SD3Transformer2DModel forward)
# ══════════════════════════════════════════════════════════════════════════════

def single_vel(
    tr,
    x_t:   torch.Tensor,   # (B, 16, H, W)
    sigma: float,           # ∈ [0, 1],  1 = pure noise
    enc:   torch.Tensor,    # (B, seq, 4096)
    pool:  torch.Tensor,    # (B, 2048)
    h:     int,
    w:     int,
    **_kw,
) -> torch.Tensor:
    """
    One forward pass of SD3Transformer2DModel.

    Timestep convention: SD3 uses σ ∈ [0, 1] directly, consistent with
    FlowMatchEulerDiscreteScheduler.  No ×1000 scaling.

    If you observe garbage outputs, check whether the model you loaded
    expects timesteps in [0, 1000] and uncomment the scaling line below.
    """
    B   = x_t.shape[0]
    dev = x_t.device
    dt  = x_t.dtype
    t   = torch.full((B,), sigma, device=dev, dtype=dt)
    # t = t * 1000.0   # ← uncomment if model expects [0, 1000] range

    out = tr(
        hidden_states         = x_t,
        encoder_hidden_states = enc,
        pooled_projections    = pool,
        timestep              = t,
        return_dict           = False,
    )
    return out[0]   # (B, 16, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# CFG combination  —  model-agnostic, identical to stats_bagel.py
# ══════════════════════════════════════════════════════════════════════════════

def cfg_vel(
    tr,
    x_t:       torch.Tensor,
    sigma:     float,
    enc_null:  torch.Tensor,
    pool_null: torch.Tensor,
    enc_cond:  torch.Tensor,
    pool_cond: torch.Tensor,
    h:         int,
    w:         int,
    cfg_scale: float,
) -> torch.Tensor:
    """v̂ = (1–w)·v_null  +  w·v_cond"""
    v_null = single_vel(tr, x_t, sigma, enc_null, pool_null, h, w)
    v_cond = single_vel(tr, x_t, sigma, enc_cond, pool_cond, h, w)
    return (1.0 - cfg_scale) * v_null + cfg_scale * v_cond


# ══════════════════════════════════════════════════════════════════════════════
# Statistics estimation  —  all from η̂ = v̂_CFG − v*
# Identical logic to stats_bagel.py; only PromptBank type changed.
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_eta(
    tr, sigma, x0, bank: PromptBankSD3, h, w, d, cfg_scale,
) -> float:
    B                   = x0.shape[0]
    enc_cond, pool_cond = bank.sample(B)
    enc_null, pool_null = bank.null(B)
    noise  = torch.randn_like(x0)
    xs     = (1.0 - sigma) * x0 + sigma * noise
    vstar  = noise - x0
    v_hat  = cfg_vel(tr, xs, sigma, enc_null, pool_null, enc_cond, pool_cond,
                     h, w, cfg_scale)
    err    = v_hat.float() - vstar.float()
    return float((err ** 2).sum()) / (B * d)


@torch.no_grad()
def est_sigma2_vdot_fd1(
    tr, sigma, x0, bank: PromptBankSD3, h, w, d, cfg_scale, dsigma=0.025,
) -> float:
    """4 forward passes: (null, cond) × (lo, hi).  Same (x₀,ε,c) at both ends."""
    s_lo = max(0.0, sigma - dsigma)
    s_hi = min(1.0, sigma + dsigma)
    ds   = (s_hi - s_lo) / 2.0
    if ds < 1e-6:
        return 0.0
    B                   = x0.shape[0]
    enc_cond, pool_cond = bank.sample(B)
    enc_null, pool_null = bank.null(B)
    noise   = torch.randn_like(x0)
    xlo     = (1.0 - s_lo) * x0 + s_lo * noise
    xhi     = (1.0 - s_hi) * x0 + s_hi * noise
    vhat_lo = cfg_vel(tr, xlo, s_lo, enc_null, pool_null, enc_cond, pool_cond, h, w, cfg_scale)
    vhat_hi = cfg_vel(tr, xhi, s_hi, enc_null, pool_null, enc_cond, pool_cond, h, w, cfg_scale)
    vdot    = (vhat_hi.float() - vhat_lo.float()) / (2.0 * ds)
    return float((vdot ** 2).sum()) / (B * d)


@torch.no_grad()
def est_g_at_sigma(
    tr, sigma, x0_list, bank: PromptBankSD3, h, w, d, cfg_scale,
    n_probes=4, delta=0.005,
) -> float:
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

@torch.no_grad()
def est_rho_eta(
    tr, sigma_grid, x0_list, bank: PromptBankSD3, h, w, d, cfg_scale,
):
    """
    Cross-step correlation of η̂.
    Independent (ε, c) per σ point — eliminates shared-noise floor artefact.
    """
    M = len(sigma_grid)
    C = np.zeros((M, M), dtype=np.float64)
    n = 0
    for xi in x0_list:
        xi   = xi.unsqueeze(0) if xi.dim() == 3 else xi
        errs = []
        enc_cond, pool_cond = bank.sample(1)
        enc_null, pool_null = bank.null(1)
        for s in sigma_grid:
            noise  = torch.randn_like(xi)
            xs     = (1.0 - s) * xi + s * noise
            vstar  = noise - xi
            v_hat  = cfg_vel(tr, xs, s, enc_null, pool_null, enc_cond, pool_cond, h, w, cfg_scale)
            e      = (v_hat.float() - vstar.float()).squeeze(0).flatten().cpu().double()
            errs.append(e)
        for i in range(M):
            for j in range(i, M):
                dot = (errs[i] * errs[j]).sum().item() / d
                C[i, j] += dot;  C[j, i] += dot
        n += 1
        if n % 16 == 0:
            print(f"    ρ̂_η̂: {n}/{len(x0_list)} samples")
    C   /= max(n, 1)
    diag = np.diag(C).clip(1e-30)
    rho  = np.clip(C / np.sqrt(np.outer(diag, diag)), 1e-8, 1.0)
    dl, rp = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl.append(abs(sigma_grid[i] - sigma_grid[j]))
            rp.append(float(rho[i, j]))
    return C, rho, np.array(dl), np.array(rp)


@torch.no_grad()
def est_rho_vdot_fd1(
    tr, sigma_grid, x0_list, bank: PromptBankSD3, h, w, d, cfg_scale, dsigma=0.025,
):
    """Cross-step correlation of v̇.  Path-coherent; 4 forwards per (sample, σ)."""
    M = len(sigma_grid)
    C = np.zeros((M, M), dtype=np.float64)
    n = 0
    for xi in x0_list:
        xi    = xi.unsqueeze(0) if xi.dim() == 3 else xi
        vdots = []
        enc_cond, pool_cond = bank.sample(1)
        enc_null, pool_null = bank.null(1)
        for s in sigma_grid:
            s_lo = max(0.0, s - dsigma);  s_hi = min(1.0, s + dsigma)
            ds   = (s_hi - s_lo) / 2.0
            if ds < 1e-6:
                vdots.append(None);  continue
            noise    = torch.randn_like(xi)
            xlo      = (1.0 - s_lo) * xi + s_lo * noise
            xhi      = (1.0 - s_hi) * xi + s_hi * noise
            vhat_lo  = cfg_vel(tr, xlo, s_lo, enc_null, pool_null, enc_cond, pool_cond, h, w, cfg_scale)
            vhat_hi  = cfg_vel(tr, xhi, s_hi, enc_null, pool_null, enc_cond, pool_cond, h, w, cfg_scale)
            vd = (vhat_hi.float() - vhat_lo.float()) / (2.0 * ds)
            vdots.append(vd.squeeze(0).flatten().cpu().double())
        for i in range(M):
            for j in range(i, M):
                if vdots[i] is None or vdots[j] is None:
                    continue
                dot = (vdots[i] * vdots[j]).sum().item() / d
                C[i, j] += dot;  C[j, i] += dot
        n += 1
        if n % 8 == 0:
            print(f"    ρ̂_v̇_fd1: {n}/{len(x0_list)} samples")
    C   /= max(n, 1)
    diag = np.diag(C).clip(1e-30)
    rho  = np.clip(C / np.sqrt(np.outer(diag, diag)), -1.0, 1.0)
    dl, rp = [], []
    for i in range(M):
        for j in range(i + 1, M):
            dl.append(abs(sigma_grid[i] - sigma_grid[j]))
            rp.append(float(rho[i, j]))
    return C, rho, np.array(dl), np.array(rp)


# ══════════════════════════════════════════════════════════════════════════════
# Stats helpers  —  identical to stats_bagel.py
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_infty(rho_s: np.ndarray, rho_vals: np.ndarray) -> float:
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.abs(rho_vals[:-1]) + 1e-8)
    MIN_FLAT = max(5, len(rho_vals) // 10)
    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i : i + MIN_FLAT] < 0.02):
            plateau_start = i;  break
    if plateau_start is None:
        warnings.warn("[stats_sd3] No plateau found — extend rho_s range.")
        plateau_start = len(rho_vals) // 2
    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i;  break
    plateau_vals = rho_vals[plateau_start:plateau_end]
    ri = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))
    cv = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(f"[stats_sd3] Plateau CV={cv:.3f} — ρ_∞ estimate unreliable.")
    return ri


def _extract_rho_curve_viz(dl, rp, n_bins=40, anchor_at_one=False):
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
        vl.append(float(np.median(rp[mask])))
    if not sl:
        return np.array([0.0, dl.max()]), np.array([1.0, 0.5])
    sa, va = np.array(sl), np.array(vl)
    if anchor_at_one:
        sa = np.concatenate([[0.0], sa]);  va = np.concatenate([[1.0], va])
    else:
        sa = np.concatenate([[0.0], sa]);  va = np.concatenate([[va[0]], va])
    log_v = np.log(np.clip(va, 1e-10, 1.0))
    for k in range(1, len(log_v)):
        if log_v[k] > log_v[k - 1]:
            log_v[k] = log_v[k - 1]
    va    = np.exp(log_v)
    pchip = PchipInterpolator(sa, va, extrapolate=False)
    so    = np.linspace(sa[0], sa[-1], 200)
    return so, np.clip(pchip(so), va[-1], va[0])


def analyse_plateau(rho_s, rho_vals, label=""):
    ri   = _extract_rho_infty(rho_s, rho_vals)
    rr   = np.clip(rho_vals - ri, 0.0, None)
    rn   = rr / max(rr[0], 1e-8)
    cross = np.where(rn <= 1.0 / np.e)[0]
    ell  = float(rho_s[cross[0]]) if len(cross) else float(rho_s[-1])
    sm   = float(np.trapezoid(rr, rho_s))
    fr   = ri / max(float(rho_vals[0]), 1e-8)
    print(f"\n  ── Plateau [{label}] ────────────────────────────────")
    print(f"     ρ(0)={rho_vals[0]:.4f}  ρ_∞={ri:.4f}  rank-1 frac={fr:.3f}")
    print(f"     ℓ_res={ell:.4f}  ∫ρ_res ds={sm:.4f}")
    verdict = "STRONG rank-1" if fr > 0.3 else "WEAK rank-1"
    print(f"     → {verdict}")
    return dict(rho_infty=ri, frac_rank1=fr, ell_res=ell,
                spectral_mass_res=sm, verdict=verdict)


def verify_stationarity(rho_s, rho_vals, sigma_grid, nfe_list=(4, 8, 16, 28, 50)):
    ri    = _extract_rho_infty(rho_s, rho_vals)
    rr    = np.clip((rho_vals - ri) / max(1 - ri, 1e-8), 0, 1)
    cross = np.where(rr <= 1.0 / np.e)[0]
    ell   = float(rho_s[cross[0]]) if len(cross) else float(rho_s[-1])
    print(f"\n  ρ_∞={ri:.4f}  ell_corr={ell:.4f}")
    print(f"  σ range = [{sigma_grid[0]:.3f}, {sigma_grid[-1]:.3f}]")
    print(f"\n  {'NFE':>5}  {'h_typ':>8}  {'ell/h':>8}  {'banded?':>12}")
    for nfe in nfe_list:
        h  = 1.0 / nfe
        r  = ell / h
        ok = "✓ yes" if r < 0.5 else ("~ marginal" if r < 1.5 else "✗ no")
        print(f"  {nfe:>5}  {h:>8.4f}  {r:>8.3f}  {ok:>12}")
    return ri, ell


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation  —  identical to stats_bagel.py (label strings updated)
# ══════════════════════════════════════════════════════════════════════════════

def _visualize(
    sigma_grid, g_vals, s2_eta, s2_vdot,
    dl_eta, rp_eta, rho_s_eta, rho_eta,
    dl_vdot, rp_vdot, rho_s_vd, rho_vd,
    plateau_eta, plateau_vdot,
    seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
    model_name, cfg_scale, out_path,
):
    seg_colors = ["#e05c00", "#2d6a9f", "#2a7d4f", "#9b3fa0"]
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    if g_vals is not None and g_vals.any():
        ax.plot(sigma_grid, g_vals, color="#2d6a9f", lw=2)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
    else:
        ax.text(0.5, 0.5, "g(σ) not estimated\n(--no_g)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("σ"); ax.set_ylabel("g(σ)")
    ax.set_title(f"Scalar Jacobian Proxy  g(σ)\n[CFG w={cfg_scale}]")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(sigma_grid, s2_eta,  color="#b5451b", lw=2,   label="σ²_η̂ (CFG err)")
    ax.semilogy(sigma_grid, s2_vdot, color="#9b3fa0", lw=2, ls="--", label="σ²_v̇_fd1")
    ax.set_xlabel("σ"); ax.set_title("CFG Error Variances  σ²_η̂ & σ²_v̇_fd1")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(dl_eta, rp_eta, s=4, alpha=0.2, color="#999", rasterized=True)
    ax.plot(rho_s_eta, rho_eta, color="#2d6a9f", lw=2.5, label="global ρ̂_η̂")
    if plateau_eta:
        ax.axhline(plateau_eta["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                   label=f"ρ_∞={plateau_eta['rho_infty']:.3f}")
    ax.set_xlabel("|Δσ|"); ax.set_ylabel("ρ̂")
    ax.set_title("CFG error ρ̂_η̂(s)  [K× auto-captured]")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_common, cv_s, color="#555", lw=2)
    ax.axhline(0.15, color="#2a7d4f", lw=1.2, ls="--", label="0.15 stationary")
    ax.axhline(0.35, color="#d12",    lw=1.2, ls="--", label="0.35 non-stat.")
    vc = ("#d12" if "STRONGLY" in stat_verdict else
          "#e08000" if "MILDLY"   in stat_verdict else "#2a7d4f")
    ax.set_title(f"Stationarity CV  mean={mean_cv:.3f}\n{stat_verdict}",
                 color=vc, fontsize=9)
    ax.set_xlabel("s"); ax.set_ylabel("CV")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rho_s_eta, rho_eta, color="k", lw=2, ls="--", alpha=0.5, label="global")
    for k, (rs, rv) in enumerate(seg_curves):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        ax.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                label=f"σ̄∈[{lo:.2f},{hi:.2f})")
    ax.set_xlabel("|Δσ|"); ax.set_title("Per-Segment  ρ̂_η̂(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    cost_proxy = s2_eta * (1.0 - sigma_grid) ** 2
    ax.plot(sigma_grid, cost_proxy, color="#2a7d4f", lw=2)
    ax.set_xlabel("σ")
    ax.set_title("Schedule Cost Proxy  σ²_η̂·(1–σ)²")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    if len(dl_vdot) > 0 and len(rho_s_vd) > 1:
        ax.scatter(dl_vdot, rp_vdot, s=4, alpha=0.2, color="#999", rasterized=True)
        ax.plot(rho_s_vd, rho_vd, color="#9b3fa0", lw=2.5, label="ρ̂_v̇_fd1")
        if plateau_vdot:
            ax.axhline(plateau_vdot["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                       label=f"ρ_∞={plateau_vdot['rho_infty']:.3f}")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "v̇ corr not estimated\n(--no_vdot_corr)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("|Δσ|"); ax.set_title("Disc-error  ρ̂_v̇_fd1(s)")
    ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    labels = ["η̂-error ρ_∞"];  vals = [plateau_eta["rho_infty"] if plateau_eta else 0.0]
    if plateau_vdot:
        labels.append("v̇ ρ_∞");  vals.append(plateau_vdot["rho_infty"])
    bars = ax.bar(labels, vals, color=["#2d6a9f", "#9b3fa0"][: len(labels)], width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.3, color="#e05c00", lw=1.2, ls="--", label="0.3 rank-1 threshold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("ρ_∞")
    ax.set_title("Rank-1 Plateau (CFG-combined η̂)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    pe  = plateau_eta  or {}
    pv  = plateau_vdot or {}
    txt = (
        f"Model: {model_name}\nCFG scale w={cfg_scale}\n{'─'*35}\n"
        f"Null cond: encoded \"\" (not zeros)\n\n"
        f"CFG error kernel ρ̂_η̂  [K× included]:\n"
        f"  ρ_∞       = {pe.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 f. = {pe.get('frac_rank1', float('nan')):.3f}\n"
        f"  ℓ_res     = {pe.get('ell_res', float('nan')):.4f}\n"
        f"  ∫ρ_res ds = {pe.get('spectral_mass_res', float('nan')):.4f}\n\n"
        + (
            f"Disc-error kernel ρ̂_v̇_fd1:\n"
            f"  ρ_∞       = {pv.get('rho_infty', float('nan')):.4f}\n"
            f"  rank-1 f. = {pv.get('frac_rank1', float('nan')):.3f}\n"
            f"  → {pv.get('verdict','N/A')}\n\n"
            if plateau_vdot else "v̇ correlation: not estimated\n\n"
        )
        + f"Stationarity: {stat_verdict}  (CV={mean_cv:.3f})\n"
    )
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaa"))
    ax.set_title("Summary", fontsize=10)

    fig.suptitle(
        f"BornSchedule Statistics (SD3 rectified flow) — {model_name}  w={cfg_scale}",
        fontsize=13, y=1.005,
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_estimation(args):
    device    = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype     = torch.bfloat16
    cfg_scale = args.cfg_scale
    C         = 16           # SD3 VAE latent channels
    h = w     = args.latent_size
    d         = C * h * w
    print(f"[stats_sd3] device={device}  model={args.model}  cfg_scale={cfg_scale}")
    print(f"  latent: {C}×{h}×{w}  d={d:,}")

    # σ grid  (denser near endpoints where velocity errors spike)
    n_lo = args.n_sigma // 3
    n_hi = args.n_sigma - n_lo
    sg   = np.concatenate([
        np.linspace(0.01, 0.35, n_lo, endpoint=False),
        np.linspace(0.35, 0.99, n_hi),
    ])
    print(f"  σ grid: {len(sg)} points  [{sg[0]:.3f}, {sg[-1]:.3f}]")

    # ── Load model ────────────────────────────────────────────────────────────
    tr, pipe = load_sd3_pipeline(args.model, device, dtype)

    # ── Encode null ("") conditioning ONCE ───────────────────────────────────
    null_enc_cpu, null_pool_cpu = encode_null_sd3(pipe, device, dtype, args.txt_seq)
    actual_txt_seq  = null_enc_cpu.shape[1]
    actual_enc_dim  = null_enc_cpu.shape[2]
    actual_pool_dim = null_pool_cpu.shape[1]
    print(f"  null cond shape: enc {tuple(null_enc_cpu.shape)}  "
          f"pool {tuple(null_pool_cpu.shape)}")

    # ── Encode COCO prompts ───────────────────────────────────────────────────
    captions = load_coco_captions(n=args.n_prompts, seed=args.seed)
    enc_all, pool_all = encode_prompts_sd3(
        pipe, captions, device, dtype,
        batch_size=args.encode_batch, max_seq=actual_txt_seq,
    )

    bank = PromptBankSD3(
        enc_all, pool_all, null_enc_cpu, null_pool_cpu,
        device, dtype, seed=args.seed,
    )

    # Offload text encoders to save VRAM
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3",
                 "tokenizer", "tokenizer_2", "tokenizer_3"):
        obj = getattr(pipe, attr, None)
        if obj is not None and hasattr(obj, "cpu"):
            obj.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Random latents ────────────────────────────────────────────────────────
    print(f"  generating {args.n_samples} random latents …")
    x0_all = torch.randn(args.n_samples, C, h, w, device=device, dtype=dtype)

    # ── σ²_η̂ ──────────────────────────────────────────────────────────────────
    print(f"\n[stats_sd3] σ²_η̂ at {len(sg)} σ points …")
    s2_eta = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0;  nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb   = x0_all[start : start + args.batch]
            acc += est_sigma2_eta(tr, s, xb, bank, h, w, d, cfg_scale)
            nb  += 1
        s2_eta[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_η̂  {k+1}/{len(sg)}  σ={s:.3f}  val={s2_eta[k]:.4e}")

    # ── σ²_v̇_fd1 ──────────────────────────────────────────────────────────────
    print(f"\n[stats_sd3] σ²_v̇_fd1 at {len(sg)} σ points …")
    s2_vdot = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0;  nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb   = x0_all[start : start + args.batch]
            acc += est_sigma2_vdot_fd1(tr, s, xb, bank, h, w, d, cfg_scale,
                                        dsigma=args.dsigma)
            nb  += 1
        s2_vdot[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_v̇_fd1  {k+1}/{len(sg)}  σ={s:.3f}  val={s2_vdot[k]:.4e}")

    wl         = min(7, (len(sg) // 4) * 2 + 1)
    log_vd     = savgol_filter(np.log(np.clip(s2_vdot, 1e-30, None)), wl, 2)
    s2_vdot_sm = np.exp(log_vd)

    # ── g(σ) ──────────────────────────────────────────────────────────────────
    g_vals = np.zeros(len(sg), dtype=np.float32)
    if not args.no_g:
        n_g  = min(args.n_g_samples, args.n_samples)
        x0_g = [x0_all[i] for i in range(n_g)]
        print(f"\n[stats_sd3] g(σ)  n_g={n_g}  n_probes={args.n_hutchinson} …")
        for k, s in enumerate(sg):
            g_vals[k], g_sem, ratio = est_g_at_sigma(
                tr, s, x0_g, bank, h, w, d, cfg_scale,
                n_probes=args.n_hutchinson, delta=args.g_delta,
            )
            if (k + 1) % max(1, len(sg) // 8) == 0:
                print(f"  g  {k+1}/{len(sg)}  σ={s:.3f}  g={g_vals[k]:.5f}")

    # ── ρ̂_η̂ ───────────────────────────────────────────────────────────────────
    n_ell  = min(args.n_ell_samples, args.n_samples)
    x0_ell = [x0_all[i] for i in range(n_ell)]
    print(f"\n[stats_sd3] ρ̂_η̂  ({n_ell} samples) …")
    _, rho_mat_eta, dl_eta, rp_eta = est_rho_eta(
        tr, sg, x0_ell, bank, h, w, d, cfg_scale)
    rho_s_eta, rho_eta = _extract_rho_curve_viz(dl_eta, rp_eta, anchor_at_one=True)
    plateau_eta = analyse_plateau(rho_s_eta, rho_eta, "CFG error η̂")

    # ── ρ̂_v̇_fd1 ───────────────────────────────────────────────────────────────
    dl_vdot = rp_vdot = np.array([])
    rho_s_vd = rho_vd = np.array([0.0, 1.0])
    plateau_vdot = None;  rho_mat_vdot = None
    if not args.no_vdot_corr:
        print(f"\n[stats_sd3] ρ̂_v̇_fd1  ({n_ell} samples) …")
        _, rho_mat_vdot, dl_vdot, rp_vdot = est_rho_vdot_fd1(
            tr, sg, x0_ell, bank, h, w, d, cfg_scale, dsigma=args.dsigma)
        if len(dl_vdot) > 5:
            rho_s_vd, rho_vd = _extract_rho_curve_viz(dl_vdot, rp_vdot)
            plateau_vdot = analyse_plateau(rho_s_vd, rho_vd, "disc-error v̇_fd1")

    # ── Stationarity ──────────────────────────────────────────────────────────
    M          = len(sg)
    lbar_all   = np.array([0.5*(sg[i]+sg[j]) for i in range(M) for j in range(i+1,M)])
    n_segs     = 3
    seg_bounds = np.linspace(0.0, 1.0, n_segs + 1)
    seg_curves = []
    print(f"\n[stats_sd3] stationarity ({n_segs} σ̄-segments) …")
    for k in range(n_segs):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        mask   = (lbar_all >= lo) & (lbar_all < hi)
        if mask.sum() < 10:
            seg_curves.append(_extract_rho_curve_viz(dl_eta, rp_eta))
        else:
            rs, rv = _extract_rho_curve_viz(dl_eta[mask], rp_eta[mask], anchor_at_one=True)
            seg_curves.append((rs, rv))
            print(f"  σ̄∈[{lo:.2f},{hi:.2f})  {mask.sum()} pairs  "
                  f"ρ̂(0)={rv[0]:.3f}  ρ̂(end)={rv[-1]:.4f}")

    s_common = rho_s_eta[1:]
    seg_mats = []
    for rs, rv in seg_curves:
        pchip = PchipInterpolator(rs, rv, extrapolate=False)
        vi    = np.where(np.isnan(pchip(s_common)), rv[-1], pchip(s_common))
        seg_mats.append(np.clip(vi, 0.0, 1.0))
    seg_mat  = np.array(seg_mats)
    cv_s     = seg_mat.std(0) / (seg_mat.mean(0) + 1e-8)
    valid_s  = rho_eta[1:] > 0.1
    mean_cv  = float(cv_s[valid_s].mean()) if valid_s.any() else float("nan")

    if   np.isnan(mean_cv):  stat_verdict = "UNKNOWN"
    elif mean_cv < 0.15:     stat_verdict = "STATIONARY"
    elif mean_cv < 0.35:     stat_verdict = "MILDLY NON-STATIONARY"
    else:                    stat_verdict = "STRONGLY NON-STATIONARY"
    print(f"  mean CV={mean_cv:.3f}  →  {stat_verdict}")
    verify_stationarity(rho_s_eta, rho_eta, sg, nfe_list=(4, 8, 16, 28, 50))

    # ── Save .npz ─────────────────────────────────────────────────────────────
    safe = args.model.replace("/", "--").replace(":", "--")
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz  = out / f"{safe}_cfg{cfg_scale}.npz"

    save = dict(
        t_grid              = sg.astype(np.float32),
        sigma_min           = np.float32(sg[0]),
        sigma_max           = np.float32(sg[-1]),
        sigma2_eta          = s2_eta.astype(np.float32),
        sigma2_vdot_fd1     = s2_vdot_sm.astype(np.float32),
        g_values            = g_vals.astype(np.float32),
        rho_s               = rho_s_eta.astype(np.float32),
        rho_values          = rho_eta.astype(np.float32),
        cfg_scale           = np.float32(cfg_scale),
        # born_schedule.py compatibility aliases
        lambda_grid         = sg.astype(np.float32),
        lambda_min          = np.float32(sg[0]),
        lambda_max          = np.float32(sg[-1]),
        sigma2_values       = s2_eta.astype(np.float32),
        sigma2_gpp_values   = s2_vdot_sm.astype(np.float32),
    )
    if plateau_vdot is not None:
        save["rho_s_gpp"]      = rho_s_vd.astype(np.float32)
        save["rho_values_gpp"] = rho_vd.astype(np.float32)

    np.savez(str(npz), **save)
    print(f"\n[stats_sd3] saved → {npz}")
    np.save(str(out / f"{safe}_cfg{cfg_scale}_rhomat_eta.npy"),
            rho_mat_eta.astype(np.float32))
    if rho_mat_vdot is not None:
        np.save(str(out / f"{safe}_cfg{cfg_scale}_rhomat_vdot.npy"),
                rho_mat_vdot.astype(np.float32))

    # ── Visualise ─────────────────────────────────────────────────────────────
    _visualize(
        sg, g_vals if g_vals.any() else None,
        s2_eta, s2_vdot_sm,
        dl_eta, rp_eta, rho_s_eta, rho_eta,
        dl_vdot, rp_vdot, rho_s_vd, rho_vd,
        plateau_eta, plateau_vdot,
        seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
        args.model, cfg_scale, npz.with_suffix(".png"),
    )

    print("\n══ Summary ═══════════════════════════════════════════════════")
    print(f"  model           : {args.model}")
    print(f"  CFG scale       : {cfg_scale}")
    print(f"  null cond       : encoded \"\" string (NOT zeros)")
    print(f"  d               : {d:,}   (latent {C}×{h}×{w})")
    print(f"  prompts         : {len(captions)} COCO captions")
    print(f"  σ grid          : [{sg[0]:.3f}, {sg[-1]:.3f}]  N={len(sg)}")
    print(f"  σ²_η̂ range      : [{s2_eta.min():.3e}, {s2_eta.max():.3e}]")
    print(f"  σ²_v̇_fd1 range  : [{s2_vdot_sm.min():.3e}, {s2_vdot_sm.max():.3e}]")
    print(f"  ρ_∞(η̂)          : {plateau_eta['rho_infty']:.4f}  "
          f"(rank-1 frac {plateau_eta['frac_rank1']:.2f})")
    if plateau_vdot:
        print(f"  ρ_∞(v̇_fd1)     : {plateau_vdot['rho_infty']:.4f}  "
              f"(rank-1 frac {plateau_vdot['frac_rank1']:.2f})")
    print(f"  stationarity    : {stat_verdict}  (CV={mean_cv:.3f})")
    print(f"  output          : {npz}")
    print("═══════════════════════════════════════════════════════════════\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BornSchedule CFG statistics for SD3 / SD3.5 rectified flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          default="stabilityai/stable-diffusion-3.5-large",
                   help="HF model ID  (also works: stable-diffusion-3-medium-diffusers)")
    p.add_argument("--output_dir",     default="/media/ssd_horse/keying/stats_sd3")
    p.add_argument("--latent_size",    type=int, default=64,
                   help="Latent H=W  (64 → 512 px,  128 → 1024 px)")
    p.add_argument("--txt_seq",        type=int, default=256,
                   help="T5 sequence length  (overridden by actual encode output)")
    p.add_argument("--n_sigma",        type=int, default=50)
    p.add_argument("--n_samples",      type=int, default=128,
                   help="Latent samples for σ²_η̂, σ²_v̇")
    p.add_argument("--n_ell_samples",  type=int, default=64,
                   help="Samples for ρ̂ estimation")
    p.add_argument("--n_g_samples",    type=int, default=32)
    p.add_argument("--n_hutchinson",   type=int, default=4,
                   help="FD-Hutchinson probes per sample for g(σ)")
    p.add_argument("--g_delta",        type=float, default=0.005)
    p.add_argument("--dsigma",         type=float, default=0.025)
    p.add_argument("--cfg_scale",      type=float, default=4.5,
                   help="CFG guidance scale w  (SD3.5-large default: 4.5)")
    p.add_argument("--batch",          type=int, default=1)
    p.add_argument("--n_prompts",      type=int, default=300,
                   help="Number of COCO captions  (200–500 recommended)")
    p.add_argument("--encode_batch",   type=int, default=4,
                   help="Batch size for text encoding  (SD3.5-large: keep ≤4)")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--device",         default=None)
    p.add_argument("--no_g",           action="store_true")
    p.add_argument("--no_vdot_corr",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_estimation(args)