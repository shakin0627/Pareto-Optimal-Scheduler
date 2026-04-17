"""
stats_flux.py  —  Offline BornSchedule statistics for FLUX flow-matching models.

Coordinate: σ ∈ [0, 1]   (σ=1 : pure noise,  σ=0 : clean data)
Linear FM interpolation:  x_σ = (1–σ)·x₀ + σ·ε
FM target velocity:       v*  = ε – x₀

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
  python stats_flux.py \\
      --model black-forest-labs/FLUX.1-dev \\
      --latent_size 64 --n_sigma 50 --n_samples 128 \\
      --n_ell_samples 64 --batch 1 --no_g \\
      --n_prompts 300
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
# COCO / Flickr30k prompt loading + FLUX text encoding
# ══════════════════════════════════════════════════════════════════════════════

def load_coco_captions(n: int = 300, seed: int = 0) -> list[str]:
    """
    Load up to `n` captions from COCO 2017 validation via HF datasets.
    Falls back to Flickr30k if COCO is unavailable.
    Returns a Python list of strings.
    """
    rng = np.random.default_rng(seed)

    # ── try COCO 2017 ────────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset("phiyodr/coco2017", split="validation",
                          token=HF_TOKEN)
        # column is "captions" (list of 5 strings per image)
        capts = []
        for row in ds:
            for c in row["captions"]:
                capts.append(c)
            if len(capts) >= n * 5:
                break
        idx  = rng.choice(len(capts), size=min(n, len(capts)), replace=False)
        caps = [capts[i] for i in idx]
        print(f"  [prompts] COCO2017: {len(caps)} captions")
        return caps
    except Exception as e:
        print(f"  [prompts] COCO failed ({e}), trying Flickr30k …")

    # ── fallback: Flickr30k ──────────────────────────────────────────────────
    try:
        from datasets import load_dataset
        ds   = load_dataset("nlphuji/flickr30k", split="test",
                            token=HF_TOKEN)
        capts = []
        for row in ds:
            for c in row["caption"]:
                capts.append(c)
            if len(capts) >= n * 5:
                break
        idx  = rng.choice(len(capts), size=min(n, len(capts)), replace=False)
        caps = [capts[i] for i in idx]
        print(f"  [prompts] Flickr30k: {len(caps)} captions")
        return caps
    except Exception as e:
        print(f"  [prompts] Flickr30k failed ({e}), using empty-string fallback.")
        return [""] * min(n, 64)


def encode_prompts_flux(
    pipe,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a list of prompts with FLUX's T5 + CLIP encoders.
    Returns:
        enc_all  (N, txt_seq, 4096)  – T5 hidden states
        pool_all (N, 768)            – CLIP pooled embedding
    Both are on CPU (moved there immediately to save VRAM).
    """
    enc_list  = []
    pool_list = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        # pipe.encode_prompt returns (prompt_embeds, pooled_prompt_embeds, ...)
        with torch.no_grad():
            out = pipe.encode_prompt(
                prompt=batch,
                prompt_2=batch,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=getattr(pipe, "tokenizer_max_length", 256),
            )
        # diffusers FluxPipeline.encode_prompt returns (te, pe, ...) or a tuple
        # depending on version; handle both
        if isinstance(out, (tuple, list)):
            te, pe = out[0], out[1]
        else:
            te = out.prompt_embeds
            pe = out.pooled_prompt_embeds
        enc_list.append(te.cpu())
        pool_list.append(pe.cpu())
        if start % max(batch_size, 32) == 0:
            print(f"    encoded {start+len(batch)}/{len(prompts)} prompts")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    enc_all  = torch.cat(enc_list,  dim=0)   # (N, seq, 4096)
    pool_all = torch.cat(pool_list, dim=0)   # (N, 768)
    print(f"  [prompts] encoded: enc {tuple(enc_all.shape)}  pool {tuple(pool_all.shape)}")
    return enc_all, pool_all


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


def load_flux_pipeline(model_name: str, device: torch.device, dtype=torch.bfloat16):
    """Load full FluxPipeline (needed for text encoding)."""
    from diffusers import FluxPipeline
    print(f"  [flux] loading full pipeline for text encoding …")
    pipe = FluxPipeline.from_pretrained(
        model_name, torch_dtype=dtype, token=HF_TOKEN,
    )
    pipe.to(device)
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# Latent packing / unpacking
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
# Position IDs
# ══════════════════════════════════════════════════════════════════════════════

def make_ids(B: int, h: int, w: int, txt_seq: int, device, dtype):
    hp, wp = h // 2, w // 2
    img = torch.zeros(hp, wp, 3, device=device, dtype=dtype)
    img[..., 1] += torch.arange(hp, device=device, dtype=dtype)[:, None]
    img[..., 2] += torch.arange(wp, device=device, dtype=dtype)[None, :]
    img = img.reshape(1, hp * wp, 3).expand(B, -1, -1)
    txt = torch.zeros(1, txt_seq, 3, device=device, dtype=dtype).expand(B, -1, -1)
    return img.contiguous(), txt.contiguous()


# ══════════════════════════════════════════════════════════════════════════════
# Conditioning helpers
# ══════════════════════════════════════════════════════════════════════════════

def null_cond(B: int, txt_seq: int, device, dtype):
    """Zero embeddings — kept as fast fallback for single-sample inner loops."""
    enc  = torch.zeros(B, txt_seq, 4096, device=device, dtype=dtype)
    pool = torch.zeros(B, 768,     device=device, dtype=dtype)
    return enc, pool


class PromptBank:
    """
    Thin wrapper around pre-encoded COCO embeddings.
    Randomly samples B prompts for each model call so statistics
    are averaged over the caption distribution.
    """
    def __init__(
        self,
        enc_all:  torch.Tensor,   # (N, seq, 4096) on CPU
        pool_all: torch.Tensor,   # (N, 768) on CPU
        device: torch.device,
        dtype:  torch.dtype,
        seed:   int = 0,
    ):
        self.enc   = enc_all
        self.pool  = pool_all
        self.N     = enc_all.shape[0]
        self.txt_seq = enc_all.shape[1]
        self.device  = device
        self.dtype   = dtype
        self.rng     = np.random.default_rng(seed)

    def sample(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (enc, pool) on device, shape (B, seq, 4096) / (B, 768)."""
        idx = self.rng.integers(0, self.N, size=B)
        enc  = self.enc[idx].to(device=self.device, dtype=self.dtype)
        pool = self.pool[idx].to(device=self.device, dtype=self.dtype)
        return enc, pool

    def get_one(self, i: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conditioning for a single fixed index (for per-sample loops)."""
        enc  = self.enc[i : i + 1].to(device=self.device, dtype=self.dtype)
        pool = self.pool[i : i + 1].to(device=self.device, dtype=self.dtype)
        return enc, pool


# ══════════════════════════════════════════════════════════════════════════════
# Velocity prediction wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _needs_guidance(tr) -> bool:
    return bool(getattr(tr.config, "guidance_embeds", False))


def flux_vel(
    tr,
    x_t: torch.Tensor,
    sigma: float,
    enc: torch.Tensor,
    pool: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    h: int, w: int,
    guidance_val: float = 3.5,
) -> torch.Tensor:
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
# σ²_η(σ)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_eta(
    tr, sigma, x0, prompt_bank: PromptBank,
    h, w, d, guidance_val=3.5,
) -> float:
    B     = x0.shape[0]
    enc, pool = prompt_bank.sample(B)
    img_ids, txt_ids = make_ids(B, h, w, enc.shape[1], x0.device, x0.dtype)
    noise = torch.randn_like(x0)
    xs    = (1.0 - sigma) * x0 + sigma * noise
    vstar = noise - x0
    vpred = flux_vel(tr, xs, sigma, enc, pool, img_ids, txt_ids, h, w, guidance_val)
    err   = vpred.float() - vstar.float()
    return float((err ** 2).sum()) / (B * d)


# ══════════════════════════════════════════════════════════════════════════════
# σ²_v̇(σ)  — saved as "sigma2_vdot_fd1"
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_vdot_fd1(
    tr, sigma, x0, prompt_bank: PromptBank,
    h, w, d, dsigma=0.025, guidance_val=3.5,
) -> float:
    """
    D(σ) ≈ E‖(v_θ(x_{σ+dσ},σ+dσ) − v_θ(x_{σ−dσ},σ−dσ)) / (2·dσ)‖² / d
    Path-coherent: x_lo/x_hi share the same (x₀, ε).
    """
    s_lo = max(0.0, sigma - dsigma)
    s_hi = min(1.0, sigma + dsigma)
    ds   = (s_hi - s_lo) / 2.0
    if ds < 1e-6:
        return 0.0

    B = x0.shape[0]
    enc, pool = prompt_bank.sample(B)
    img_ids, txt_ids = make_ids(B, h, w, enc.shape[1], x0.device, x0.dtype)

    noise = torch.randn_like(x0)
    xlo   = (1 - s_lo) * x0 + s_lo * noise
    xhi   = (1 - s_hi) * x0 + s_hi * noise

    vlo   = flux_vel(tr, xlo, s_lo, enc, pool, img_ids, txt_ids, h, w, guidance_val)
    vhi   = flux_vel(tr, xhi, s_hi, enc, pool, img_ids, txt_ids, h, w, guidance_val)

    vdot  = (vhi.float() - vlo.float()) / (2.0 * ds)
    return float((vdot ** 2).sum()) / (B * d)


# ══════════════════════════════════════════════════════════════════════════════
# g(σ)  —  scalar Jacobian proxy  (FD-Hutchinson)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_g_at_sigma(
    tr, sigma, x0_list, prompt_bank: PromptBank,
    h, w, d, n_probes=4, delta=0.005, guidance_val=3.5,
) -> float:
    """x0_list: iterable of (1, 16, H, W) tensors."""
    trace = 0.0; count = 0
    for xi in x0_list:
        xi  = xi.unsqueeze(0) if xi.dim() == 3 else xi
        enc, pool = prompt_bank.sample(1)
        img_ids, txt_ids = make_ids(1, h, w, enc.shape[1], xi.device, xi.dtype)
        noise = torch.randn_like(xi)
        xs    = (1 - sigma) * xi + sigma * noise
        v0    = flux_vel(tr, xs, sigma, enc, pool, img_ids, txt_ids, h, w, guidance_val)
        for _ in range(n_probes):
            z  = torch.randn_like(xs)
            vp = flux_vel(tr, xs + delta * z, sigma, enc, pool,
                          img_ids, txt_ids, h, w, guidance_val)
            trace += float(((vp.float() - v0.float()) * z.float()).sum()) / delta
            count += 1
            if xi.device.type == "cuda":
                torch.cuda.empty_cache()
    return trace / (count * d)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-step velocity-error correlation  ρ̂_η(s)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_eta(
    tr, sigma_grid, x0_list, prompt_bank: PromptBank,
    h, w, d, guidance_val=3.5,
):
    """
    Independent ε per σ point eliminates shared-noise floor artefact.
    Returns: C (covariance M×M), rho (normalised),
             delta_sigmas (upper-tri pairs), rho_pairs.
    """
    M   = len(sigma_grid)
    C   = np.zeros((M, M), dtype=np.float64)
    n   = 0

    for xi in x0_list:
        xi   = xi.unsqueeze(0) if xi.dim() == 3 else xi
        errs = []
        for s in sigma_grid:
            enc, pool = prompt_bank.sample(1)
            img_ids, txt_ids = make_ids(1, h, w, enc.shape[1], xi.device, xi.dtype)
            noise = torch.randn_like(xi)
            xs    = (1 - s) * xi + s * noise
            vp    = flux_vel(tr, xs, s, enc, pool, img_ids, txt_ids, h, w, guidance_val)
            e     = (vp.float() - (noise - xi).float()).squeeze(0).flatten().cpu().double()
            errs.append(e)
        for i in range(M):
            for j in range(i, M):
                dot = (errs[i] * errs[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot
        n += 1
        if n % 16 == 0:
            print(f"    ρ̂_η: {n}/{len(x0_list)} samples")

    C   /= max(n, 1)
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
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_vdot_fd1(
    tr, sigma_grid, x0_list, prompt_bank: PromptBank,
    h, w, d, dsigma=0.025, guidance_val=3.5,
):
    """
    Path-coherent central-difference: same (x₀, ε) for lo/hi.
    Returns: C, rho, delta_sigmas, rho_pairs.
    """
    M  = len(sigma_grid)
    C  = np.zeros((M, M), dtype=np.float64)
    n  = 0

    for xi in x0_list:
        xi    = xi.unsqueeze(0) if xi.dim() == 3 else xi
        vdots = []

        for s in sigma_grid:
            s_lo = max(0.0, s - dsigma)
            s_hi = min(1.0, s + dsigma)
            ds   = (s_hi - s_lo) / 2.0
            if ds < 1e-6:
                vdots.append(None)
                continue

            enc, pool = prompt_bank.sample(1)
            img_ids, txt_ids = make_ids(1, h, w, enc.shape[1], xi.device, xi.dtype)
            noise = torch.randn_like(xi)
            xlo   = (1 - s_lo) * xi + s_lo * noise
            xhi   = (1 - s_hi) * xi + s_hi * noise

            vlo   = flux_vel(tr, xlo, s_lo, enc, pool, img_ids, txt_ids, h, w, guidance_val)
            vhi   = flux_vel(tr, xhi, s_hi, enc, pool, img_ids, txt_ids, h, w, guidance_val)

            vd    = (vhi.float() - vlo.float()) / (2.0 * ds)
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
            print(f"    ρ̂_v̇ (fd1): {n}/{len(x0_list)} samples")

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
# Stats helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_infty(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
) -> float:
    """
    Detect the long-lag plateau of ρ(s) and return ρ_∞.

    Algorithm
    ---------
    1. Slide a window of size MIN_FLAT; find the first index where all
       relative increments |Δρ/ρ| < 2 %.
    2. Extend the plateau window until ρ drops more than 5 % below its
       value at the plateau start.
    3. Return the median over the plateau window.
    4. Emit a UserWarning if the plateau CV > 5 % (unreliable estimate).
    """
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.abs(rho_vals[:-1]) + 1e-8)
    MIN_FLAT = max(5, len(rho_vals) // 10)

    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i : i + MIN_FLAT] < 0.02):
            plateau_start = i
            break
    if plateau_start is None:
        warnings.warn(
            "[stats_flux] No plateau found — extend rho_s range.",
            UserWarning,
        )
        plateau_start = len(rho_vals) // 2

    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i
            break

    plateau_vals = rho_vals[plateau_start:plateau_end]
    ri = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))

    cv = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(
            f"[stats_flux] Plateau CV={cv:.3f} — ρ_∞ estimate unreliable. "
            f"Plateau indices [{plateau_start}, {plateau_end}), "
            f"s=[{rho_s[plateau_start]:.3f}, "
            f"{rho_s[min(plateau_end, len(rho_s)-1)]:.3f}].",
            UserWarning,
        )
    return ri


def _extract_rho_curve_viz(
    dl: np.ndarray,
    rp: np.ndarray,
    n_bins: int = 40,
    anchor_at_one: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth the scatter (dl, rp) → (s, ρ) curve for visualisation only.
    Uses percentile binning + Pchip interpolation with monotone-clamp.
    """
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


def analyse_plateau(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    label: str = "",
) -> dict:
    """
    Full plateau analysis for one correlation curve.
    Uses _extract_rho_infty for the scalar ρ_∞ estimate.
    """
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

    # Row 0 — g(σ)
    ax = fig.add_subplot(gs[0, 0])
    if g_vals is not None and g_vals.any():
        ax.plot(sigma_grid, g_vals, color="#2d6a9f", lw=2)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
    else:
        ax.text(0.5, 0.5, "g(σ) not estimated\n(--no_g)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("σ"); ax.set_ylabel("g(σ)")
    ax.set_title("Scalar Jacobian Proxy  g(σ)"); ax.grid(True, alpha=0.3)

    # Row 0 — variances
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(sigma_grid, s2_eta,  color="#b5451b", lw=2,   label="σ²_η (vel-err)")
    ax.semilogy(sigma_grid, s2_vdot, color="#9b3fa0", lw=2, ls="--", label="σ²_v̇_fd1 (disc)")
    ax.set_xlabel("σ"); ax.set_title("Error Variances σ²_η & σ²_v̇_fd1")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # Row 0 — ρ̂_η global
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(dl_eta, rp_eta, s=4, alpha=0.2, color="#999", rasterized=True)
    ax.plot(rho_s_eta, rho_eta, color="#2d6a9f", lw=2.5, label="global ρ̂_η")
    if plateau_eta:
        ax.axhline(plateau_eta["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                   label=f"ρ_∞={plateau_eta['rho_infty']:.3f}")
    ax.set_xlabel("|Δσ|"); ax.set_ylabel("ρ̂"); ax.set_title("Score-approx error  ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    # Row 1 — Stationarity CV
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_common, cv_s, color="#555", lw=2)
    ax.axhline(0.15, color="#2a7d4f", lw=1.2, ls="--", label="0.15 stationary")
    ax.axhline(0.35, color="#d12",    lw=1.2, ls="--", label="0.35 non-stat.")
    vc = ("#d12" if "STRONGLY" in stat_verdict else
          "#e08000" if "MILDLY" in stat_verdict else "#2a7d4f")
    ax.set_title(f"Stationarity CV  mean={mean_cv:.3f}\n{stat_verdict}",
                 color=vc, fontsize=9)
    ax.set_xlabel("s"); ax.set_ylabel("CV")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Row 1 — Per-segment ρ̂_η
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rho_s_eta, rho_eta, color="k", lw=2, ls="--", alpha=0.5, label="global")
    for k, (rs, rv) in enumerate(seg_curves):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        ax.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                label=f"σ̄∈[{lo:.2f},{hi:.2f})")
    ax.set_xlabel("|Δσ|"); ax.set_title("Per-Segment  ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    # Row 1 — Cost proxy
    ax = fig.add_subplot(gs[1, 2])
    cost_proxy = s2_eta * (1.0 - sigma_grid) ** 2
    ax.plot(sigma_grid, cost_proxy, color="#2a7d4f", lw=2)
    ax.set_xlabel("σ")
    ax.set_title("Schedule Cost Proxy  σ²_η·(1–σ)²")
    ax.grid(True, alpha=0.3)

    # Row 2 — ρ̂_v̇ (discretisation-error correlation)
    ax = fig.add_subplot(gs[2, 0])
    if len(dl_vdot) > 0 and len(rho_s_vd) > 1:
        ax.scatter(dl_vdot, rp_vdot, s=4, alpha=0.2, color="#999", rasterized=True)
        ax.plot(rho_s_vd, rho_vd, color="#9b3fa0", lw=2.5, label="ρ̂_v̇_fd1")
        if plateau_vdot:
            ax.axhline(plateau_vdot["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                       label=f"ρ_∞={plateau_vdot['rho_infty']:.3f}")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "v̇ correlation\nnot estimated\n(--no_vdot_corr)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("|Δσ|"); ax.set_ylabel("ρ̂_v̇_fd1")
    ax.set_title("Disc-error  ρ̂_v̇_fd1(s)")
    ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)

    # Row 2 — Rank-1 comparison
    ax = fig.add_subplot(gs[2, 1])
    labels = ["score-approx ρ_∞"]
    vals   = [plateau_eta["rho_infty"] if plateau_eta else 0.0]
    if plateau_vdot:
        labels.append("disc-error ρ_∞")
        vals.append(plateau_vdot["rho_infty"])
    bars = ax.bar(labels, vals, color=["#2d6a9f", "#9b3fa0"][: len(labels)], width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.3, color="#e05c00", lw=1.2, ls="--", label="0.3 rank-1 threshold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("ρ_∞")
    ax.set_title("Rank-1 Plateau Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    # Row 2 — Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    pe  = plateau_eta  or {}
    pv  = plateau_vdot or {}
    txt = (
        f"Model: {model_name}\n{'─'*35}\n"
        f"Score-approx error kernel (ρ̂_η):\n"
        f"  ρ_∞       = {pe.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 f. = {pe.get('frac_rank1', float('nan')):.3f}\n"
        f"  ℓ_res     = {pe.get('ell_res', float('nan')):.4f}\n"
        f"  ∫ρ_res ds = {pe.get('spectral_mass_res', float('nan')):.4f}\n\n"
        + (
        f"Disc-error kernel (ρ̂_v̇_fd1):\n"
        f"  ρ_∞       = {pv.get('rho_infty', float('nan')):.4f}\n"
        f"  rank-1 f. = {pv.get('frac_rank1', float('nan')):.3f}\n"
        f"  → {pv.get('verdict','N/A')}\n\n"
        if plateau_vdot else
        "Disc-error kernel: not estimated\n\n"
        )
        + f"Stationarity: {stat_verdict}\n"
          f"  mean CV = {mean_cv:.3f}\n"
    )
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#aaa"))
    ax.set_title("Summary", fontsize=10)

    fig.suptitle(f"BornSchedule Statistics (FM/FLUX) — {model_name}",
                 fontsize=13, y=1.005)
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

    # ── σ grid ───────────────────────────────────────────────────────────────
    n_lo = args.n_sigma // 3
    n_hi = args.n_sigma - n_lo
    sg   = np.concatenate([
        np.linspace(0.01, 0.35, n_lo, endpoint=False),
        np.linspace(0.35, 0.99, n_hi),
    ])
    print(f"  σ grid: {len(sg)} points  [{sg[0]:.3f}, {sg[-1]:.3f}]")

    h, w    = args.latent_size, args.latent_size
    d       = 16 * h * w
    txt_seq = args.txt_seq
    guidance_val = args.guidance
    print(f"  latent: 16×{h}×{w}  d={d:,}  txt_seq={txt_seq}")

    # ── Load full pipeline (needed for text encoding) ─────────────────────────
    pipe = load_flux_pipeline(args.model, device, dtype)
    tr   = pipe.transformer
    for m in tr.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    tr.eval()
    print(f"  guidance_embeds: {_needs_guidance(tr)}  guidance_val: {guidance_val}")

    # ── Encode COCO prompts ──────────────────────────────────────────────────
    captions = load_coco_captions(n=args.n_prompts, seed=args.seed)
    enc_all, pool_all = encode_prompts_flux(
        pipe, captions, device, dtype, batch_size=args.encode_batch,
    )
    # txt_seq might differ from the actual encoded length; update
    actual_txt_seq = enc_all.shape[1]
    if actual_txt_seq != txt_seq:
        print(f"  [prompts] actual txt_seq={actual_txt_seq} "
              f"(overrides --txt_seq={txt_seq})")
        txt_seq = actual_txt_seq

    prompt_bank = PromptBank(enc_all, pool_all, device, dtype, seed=args.seed)

    # Offload text encoders to save VRAM (transformer is all we need now)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.cpu()
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Random latent "data" ─────────────────────────────────────────────────
    print(f"  generating {args.n_samples} random latents …")
    x0_all = torch.randn(args.n_samples, 16, h, w, device=device, dtype=dtype)

    # ── σ²_η(σ) ──────────────────────────────────────────────────────────────
    print(f"\n[stats_flux] σ²_η at {len(sg)} σ points …")
    s2_eta = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb  = x0_all[start : start + args.batch]
            acc += est_sigma2_eta(tr, s, xb, prompt_bank, h, w, d, guidance_val)
            nb  += 1
        s2_eta[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_η  {k+1}/{len(sg)}  σ={s:.3f}  val={s2_eta[k]:.4e}")

    # ── σ²_v̇_fd1(σ) ───────────────────────────────────────────────────────────
    print(f"\n[stats_flux] σ²_v̇_fd1 at {len(sg)} σ points …")
    s2_vdot = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb  = x0_all[start : start + args.batch]
            acc += est_sigma2_vdot_fd1(tr, s, xb, prompt_bank, h, w, d,
                                        dsigma=args.dsigma,
                                        guidance_val=guidance_val)
            nb  += 1
        s2_vdot[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_v̇_fd1  {k+1}/{len(sg)}  σ={s:.3f}  val={s2_vdot[k]:.4e}")

    wl           = min(7, (len(sg) // 4) * 2 + 1)
    log_vd       = savgol_filter(np.log(np.clip(s2_vdot, 1e-30, None)), wl, 2)
    s2_vdot_sm   = np.exp(log_vd)

    # ── g(σ) ─────────────────────────────────────────────────────────────────
    g_vals = np.zeros(len(sg), dtype=np.float32)
    if not args.no_g:
        n_g   = min(args.n_g_samples, args.n_samples)
        x0_g  = [x0_all[i] for i in range(n_g)]
        print(f"\n[stats_flux] g(σ)  n_g_samples={n_g}  n_probes={args.n_hutchinson} …")
        for k, s in enumerate(sg):
            g_vals[k] = est_g_at_sigma(
                tr, s, x0_g, prompt_bank, h, w, d,
                n_probes=args.n_hutchinson, delta=args.g_delta,
                guidance_val=guidance_val,
            )
            if (k + 1) % max(1, len(sg) // 8) == 0:
                print(f"  g  {k+1}/{len(sg)}  σ={s:.3f}  g={g_vals[k]:.5f}")

    # ── ρ̂_η — score-approximation-error correlation ──────────────────────────
    n_ell  = min(args.n_ell_samples, args.n_samples)
    x0_ell = [x0_all[i] for i in range(n_ell)]
    print(f"\n[stats_flux] ρ̂_η  ({n_ell} samples) …")
    _, rho_mat_eta, dl_eta, rp_eta = est_rho_eta(
        tr, sg, x0_ell, prompt_bank, h, w, d, guidance_val)
    rho_s_eta, rho_eta = _extract_rho_curve_viz(dl_eta, rp_eta, anchor_at_one=True)
    plateau_eta = analyse_plateau(rho_s_eta, rho_eta, "score-approx error η")

    # ── ρ̂_v̇_fd1 — discretisation-error correlation ───────────────────────────
    dl_vdot = rp_vdot = np.array([])
    rho_s_vd = rho_vd = np.array([0.0, 1.0])
    plateau_vdot = None
    rho_mat_vdot = None

    if not args.no_vdot_corr:
        print(f"\n[stats_flux] ρ̂_v̇_fd1  ({n_ell} samples) …")
        _, rho_mat_vdot, dl_vdot, rp_vdot = est_rho_vdot_fd1(
            tr, sg, x0_ell, prompt_bank, h, w, d,
            dsigma=args.dsigma, guidance_val=guidance_val)
        if len(dl_vdot) > 5:
            rho_s_vd, rho_vd = _extract_rho_curve_viz(
                dl_vdot, rp_vdot, anchor_at_one=False)
            plateau_vdot = analyse_plateau(rho_s_vd, rho_vd, "disc-error v̇_fd1")

    # ── Stationarity ─────────────────────────────────────────────────────────
    M = len(sg)
    lbar_all = np.array([
        0.5 * (sg[i] + sg[j])
        for i in range(M) for j in range(i + 1, M)
    ])
    n_segs    = 3
    seg_bounds = np.linspace(0.0, 1.0, n_segs + 1)
    seg_curves = []
    print(f"\n[stats_flux] stationarity ({n_segs} σ̄-segments) …")
    for k in range(n_segs):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        mask   = (lbar_all >= lo) & (lbar_all < hi)
        if mask.sum() < 10:
            seg_curves.append(_extract_rho_curve_viz(dl_eta, rp_eta))
        else:
            rs, rv = _extract_rho_curve_viz(
                dl_eta[mask], rp_eta[mask], anchor_at_one=True)
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
    npz  = out / f"{safe}.npz"

    save = dict(
        # ── FM-native keys ────────────────────────────────────────────────
        t_grid              = sg.astype(np.float32),
        sigma_min           = np.float32(sg[0]),
        sigma_max           = np.float32(sg[-1]),
        sigma2_eta          = s2_eta.astype(np.float32),
        sigma2_vdot_fd1     = s2_vdot_sm.astype(np.float32),  
        g_values            = g_vals.astype(np.float32),
        rho_s               = rho_s_eta.astype(np.float32),
        rho_values          = rho_eta.astype(np.float32), 
    )
    if plateau_vdot is not None:
        save["rho_s_gpp"]      = rho_s_vd.astype(np.float32)
        save["rho_values_gpp"] = rho_vd.astype(np.float32)

    np.savez(str(npz), **save)
    print(f"\n[stats_flux] saved → {npz}")
    print(f"  Keys: {list(save.keys())}")

    # Save raw correlation matrices
    np.save(str(out / f"{safe}_rhomat_eta.npy"),
            rho_mat_eta.astype(np.float32))
    if rho_mat_vdot is not None:
        np.save(str(out / f"{safe}_rhomat_vdot_fd1.npy"),
                rho_mat_vdot.astype(np.float32))

    # ── Visualise ─────────────────────────────────────────────────────────────
    vis = npz.with_suffix(".png")
    _visualize(
        sg, g_vals if g_vals.any() else None,
        s2_eta, s2_vdot_sm,
        dl_eta, rp_eta, rho_s_eta, rho_eta,
        dl_vdot, rp_vdot, rho_s_vd, rho_vd,
        plateau_eta, plateau_vdot,
        seg_bounds, seg_curves, cv_s, s_common, mean_cv, stat_verdict,
        args.model, vis,
    )

    print("\n══ Summary ═══════════════════════════════════════════════════")
    print(f"  model           : {args.model}")
    print(f"  d               : {d:,}   (latent 16×{h}×{w})")
    print(f"  prompts         : {len(captions)} COCO captions")
    print(f"  σ grid          : [{sg[0]:.3f}, {sg[-1]:.3f}]  N={len(sg)}")
    print(f"  σ²_η range      : [{s2_eta.min():.3e}, {s2_eta.max():.3e}]")
    print(f"  σ²_v̇_fd1 range  : [{s2_vdot_sm.min():.3e}, {s2_vdot_sm.max():.3e}]")
    print(f"  ρ_∞(η)          : {plateau_eta['rho_infty']:.4f}  "
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
        description="BornSchedule offline statistics for FLUX flow-matching models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",        default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--output_dir",   default="/media/ssd_horse/keying/stats_flux")
    p.add_argument("--latent_size",  type=int, default=64,
                   help="Latent H=W (64→512px, 128→1024px)")
    p.add_argument("--txt_seq",      type=int, default=256,
                   help="T5 sequence length (overridden by actual encode output)")
    p.add_argument("--n_sigma",      type=int, default=50)
    p.add_argument("--n_samples",    type=int, default=128,
                   help="Latent samples for σ²_η, σ²_v̇")
    p.add_argument("--n_ell_samples",type=int, default=64,
                   help="Samples for ρ̂ estimation")
    p.add_argument("--n_g_samples",  type=int, default=32)
    p.add_argument("--n_hutchinson", type=int, default=4,
                   help="FD-Hutchinson probes per sample for g(σ)")
    p.add_argument("--g_delta",      type=float, default=0.005)
    p.add_argument("--dsigma",       type=float, default=0.025,
                   help="Central-difference step for σ²_v̇_fd1 and ρ̂_v̇_fd1")
    p.add_argument("--guidance",     type=float, default=3.5)
    p.add_argument("--batch",        type=int, default=1,
                   help="Batch size for variance estimation")
    p.add_argument("--n_prompts",    type=int, default=300,
                   help="Number of COCO captions to load (200–500 recommended)")
    p.add_argument("--encode_batch", type=int, default=8,
                   help="Batch size for text encoding")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       default=None)
    p.add_argument("--no_g",         action="store_true",
                   help="Skip g(σ) estimation")
    p.add_argument("--no_vdot_corr", action="store_true",
                   help="Skip v̇_fd1 cross-step correlation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_estimation(args)