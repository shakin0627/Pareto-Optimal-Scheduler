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

import io
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
# [F4] Actual Flux sigma range helper
# ══════════════════════════════════════════════════════════════════════════════

def get_flux_sigma_range(pipe, latent_size: int, nfe: int = 50) -> tuple[float, float]:
    from diffusers.pipelines.flux.modular_pipeline_flux_utils import (
        _get_initial_timesteps_and_optionals,
    )
    # latent_size=64 → 512px
    pixel_size = latent_size * 8

    _, _, sigmas, _ = _get_initial_timesteps_and_optionals(
        transformer        = pipe.transformer,
        scheduler          = pipe.scheduler,
        batch_size         = 1,
        height             = pixel_size,
        width              = pixel_size,
        vae_scale_factor   = pipe.vae_scale_factor,
        num_inference_steps= nfe,
        guidance_scale     = 3.5,
        sigmas             = None,
        device             = "cpu",
    )

    # sigmas    = pipe.scheduler.sigmas
    sigma_max = float(sigmas.max().item())
    sigma_min = float(sigmas[sigmas > 0].min().item())
    print(f"  Flux actual σ ∈ [{sigma_min:.5f}, {sigma_max:.5f}]  "
          f"(pixel={pixel_size}, nfe={nfe})")
    return sigma_min, sigma_max


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
        ds = load_dataset("phiyodr/coco2017", split="validation", token=HF_TOKEN)
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
        ds   = load_dataset("nlphuji/flickr30k", split="test", token=HF_TOKEN)
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
    Both are on CPU.
    """
    enc_list  = []
    pool_list = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        with torch.no_grad():
            out = pipe.encode_prompt(
                prompt=batch,
                prompt_2=batch,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=getattr(pipe, "tokenizer_max_length", 256),
            )
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
# [F3] Real VAE latent encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_images_to_latents(
    pipe,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_size: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """
    Encode real COCO images through the Flux VAE to get x0 samples from
    the true data distribution. Falls back to randn if COCO / VAE unavailable.

    Returns (N, 16, latent_size, latent_size) on device.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    pixel_size = latent_size * 8   # Flux VAE: 8× spatial downscale
    print(f"  [F3] encoding {n_samples} real COCO images via VAE "
          f"(pixel_size={pixel_size}) …")

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "phiyodr/coco2017", split="validation",
            token=HF_TOKEN, streaming=True,
        )
    except Exception as e:
        print(f"  [F3] COCO unavailable ({e}) — falling back to randn latents")
        return torch.randn(n_samples, 16, latent_size, latent_size,
                           device=device, dtype=dtype)

    pipe.vae = pipe.vae.to(device)
    latents_list: list[torch.Tensor] = []

    for i, row in enumerate(ds):
        if len(latents_list) >= n_samples:
            break
        try:
            img = row["image"]
            # Streaming datasets sometimes wrap images as dicts
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"]))
            elif not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            # Square centre-crop then resize
            s   = min(img.size)
            img = TF.center_crop(img, s)
            img = img.resize((pixel_size, pixel_size), Image.LANCZOS)
            x   = TF.to_tensor(img).unsqueeze(0).to(device=device, dtype=dtype)
            x   = (x - 0.5) * 2.0   # [0,1] → [-1,1]

            with torch.no_grad():
                lat = pipe.vae.encode(x).latent_dist.sample()
                # Flux VAE normalisation
                lat = (lat - pipe.vae.config.shift_factor) \
                      * pipe.vae.config.scaling_factor

            latents_list.append(lat.squeeze(0).cpu())
        except Exception:
            continue

        if (i + 1) % 50 == 0:
            print(f"    VAE encoded {len(latents_list)}/{n_samples}")

    if len(latents_list) < n_samples:
        n_pad = n_samples - len(latents_list)
        print(f"  [F3] only got {len(latents_list)} images, "
              f"padding {n_pad} with randn")
        for _ in range(n_pad):
            latents_list.append(
                torch.randn(16, latent_size, latent_size)
            )

    x0_all = torch.stack(latents_list[:n_samples]).to(device=device, dtype=dtype)
    print(f"  [F3] x0 latents: {tuple(x0_all.shape)}  "
          f"mean={x0_all.mean():.3f}  std={x0_all.std():.3f}")

    # Offload VAE to free VRAM
    pipe.vae.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return x0_all


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_flux_pipeline(model_name: str, device: torch.device, dtype=torch.bfloat16):
    """Load full FluxPipeline (needed for text encoding + VAE)."""
    from diffusers import FluxPipeline
    print(f"  [flux] loading full pipeline for text encoding + VAE …")
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

def _needs_guidance(tr) -> bool:
    return bool(getattr(tr.config, "guidance_embeds", False))


class PromptBank:
    """
    Thin wrapper around pre-encoded COCO embeddings.
    Randomly samples B prompts for each model call so statistics
    are averaged over the caption distribution.
    """
    def __init__(
        self,
        enc_all:  torch.Tensor,
        pool_all: torch.Tensor,
        device: torch.device,
        dtype:  torch.dtype,
        seed:   int = 0,
    ):
        self.enc     = enc_all
        self.pool    = pool_all
        self.N       = enc_all.shape[0]
        self.txt_seq = enc_all.shape[1]
        self.device  = device
        self.dtype   = dtype
        self.rng     = np.random.default_rng(seed)

    def sample(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx  = self.rng.integers(0, self.N, size=B)
        enc  = self.enc[idx].to(device=self.device, dtype=self.dtype)
        pool = self.pool[idx].to(device=self.device, dtype=self.dtype)
        return enc, pool


# ══════════════════════════════════════════════════════════════════════════════
# Velocity prediction wrapper
# ══════════════════════════════════════════════════════════════════════════════

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
# σ²_v̇(σ)  [F2: adaptive half-step]
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_vdot_fd1(
    tr, sigma, x0, prompt_bank: PromptBank,
    h, w, d, dsigma=0.025, guidance_val=3.5,
) -> float:
    """
    D(σ) ≈ E‖(v_θ(x_{σ+h},σ+h) − v_θ(x_{σ−h},σ−h)) / (2h)‖² / d

    [F2] half-step h = min(dsigma, σ−ε, 1−σ−ε) so the difference quotient
    is always a symmetric central difference, even near σ=0 / σ=1.
    Path-coherent: x_lo/x_hi share the same (x₀, ε).
    """
    # [F2] adaptive half-step
    half = min(dsigma, sigma - 1e-4, 1.0 - sigma - 1e-4)
    if half < 1e-5:
        return 0.0

    s_lo = sigma - half
    s_hi = sigma + half

    B = x0.shape[0]
    enc, pool = prompt_bank.sample(B)
    img_ids, txt_ids = make_ids(B, h, w, enc.shape[1], x0.device, x0.dtype)

    noise = torch.randn_like(x0)
    xlo   = (1 - s_lo) * x0 + s_lo * noise
    xhi   = (1 - s_hi) * x0 + s_hi * noise

    vlo   = flux_vel(tr, xlo, s_lo, enc, pool, img_ids, txt_ids, h, w, guidance_val)
    vhi   = flux_vel(tr, xhi, s_hi, enc, pool, img_ids, txt_ids, h, w, guidance_val)

    vdot  = (vhi.float() - vlo.float()) / (2.0 * half)
    return float((vdot ** 2).sum()) / (B * d)


# ══════════════════════════════════════════════════════════════════════════════
# g(σ)  —  scalar Jacobian proxy  (FD-Hutchinson)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_g_at_sigma(
    tr, sigma, x0_list, prompt_bank: PromptBank,
    h, w, d, n_probes=4, delta=0.005, guidance_val=3.5,
) -> tuple[float, float, float]:
    """
    Returns (g_iso, g_sem, anisotropy_ratio).
 
    g_iso  = (1/d) Tr(∇_x v̂)   via isotropic FD-Hutchinson  [original]
    g_sem  = (1/d) u^T ∇_x v̂ u  along CFG semantic direction u [new]
             u(x,σ) = (v_text – v_null) / ‖v_text – v_null‖_F
 
    anisotropy_ratio = g_sem / (g_iso + ε)
 
    If g_sem ≫ g_iso  →  Jacobian is highly anisotropic in the semantic
    direction; isotropic Hutchinson under-weights semantic modes.
 
    Forward-pass budget per sample:
        baseline   : 1 (v_text, already used for g_iso probes)
        null branch: 1  (v_null, for u construction)
        g_iso probes: n_probes (random z, text branch only)
        g_sem probe : 2  (text + null perturbed along u_norm)
        total      : n_probes + 4
    """
    trace_iso = 0.0
    trace_sem = 0.0
    count     = 0
 
    # pre-build null (zero) embeddings shape; reused across samples
    # (actual null enc/pool built per-sample from enc shape)
    for xi in x0_list:
        xi  = xi.unsqueeze(0) if xi.dim() == 3 else xi
        enc_text, pool_text = prompt_bank.sample(1)
        img_ids, txt_ids = make_ids(
            1, h, w, enc_text.shape[1], xi.device, xi.dtype)
 
        # null conditioning: zero embeddings  (only for u-direction reference)
        enc_null  = torch.zeros_like(enc_text)
        pool_null = torch.zeros_like(pool_text)
 
        noise = torch.randn_like(xi)
        xs    = (1 - sigma) * xi + sigma * noise
 
        # ── baseline velocities (2 fwds) ─────────────────────────────────────
        v0_text = flux_vel(
            tr, xs, sigma, enc_text, pool_text,
            img_ids, txt_ids, h, w, guidance_val)
        v0_null = flux_vel(
            tr, xs, sigma, enc_null, pool_null,
            img_ids, txt_ids, h, w, guidance_val)
 
        # semantic direction  u ∈ R^d
        u_raw  = v0_text.float() - v0_null.float()          # (1,C,H,W)
        u_norm_val = u_raw.norm()
        if u_norm_val < 1e-8:
            # degenerate: text ≈ null  →  skip semantic probe this sample
            u_unit = torch.randn_like(u_raw)
            u_unit = u_unit / (u_unit.norm() + 1e-8)
        else:
            u_unit = u_raw / u_norm_val                      # unit vector
 
        # ── isotropic Hutchinson probes  (n_probes fwds) ─────────────────────
        for _ in range(n_probes):
            z  = torch.randn_like(xs)
            vp = flux_vel(
                tr, xs + delta * z, sigma,
                enc_text, pool_text, img_ids, txt_ids, h, w, guidance_val)
            trace_iso += float(
                ((vp.float() - v0_text.float()) * z.float()).sum()
            ) / delta
            count += 1
            if xi.device.type == "cuda":
                torch.cuda.empty_cache()
 
        # ── directional probe along u  (2 fwds: text + null) ─────────────────
        xs_u   = xs + delta * u_unit
        vp_text_u = flux_vel(
            tr, xs_u, sigma, enc_text, pool_text,
            img_ids, txt_ids, h, w, guidance_val)
        vp_null_u = flux_vel(
            tr, xs_u, sigma, enc_null, pool_null,
            img_ids, txt_ids, h, w, guidance_val)
 
        # CFG-combined perturbed velocity (guidance direction consistent)
        # For guidance-distilled FLUX: v̂ = single-forward with text+guidance
        # We approximate the Jacobian along u using both branches so the
        # direction probe is self-consistent with how u was constructed.
        dv_text = (vp_text_u.float() - v0_text.float()) / delta   # ∇v_text · u
        dv_null = (vp_null_u.float() - v0_null.float()) / delta   # ∇v_null · u
 
        # Rayleigh quotient  u^T J u  for each branch, averaged
        trace_sem += float((dv_text * u_unit).sum())               # text branch
        # (null branch gives the "background" Jacobian component)
        # record net semantic Jacobian as text-branch directional derivative
        # (the quantity that enters the cost function is the text-cond predictor)
 
        if xi.device.type == "cuda":
            torch.cuda.empty_cache()
 
    n_samples = len(x0_list)
    g_iso  = trace_iso / (count * d)          # count = n_samples * n_probes
    g_sem  = trace_sem / (n_samples * d)      # one directional probe per sample
    ratio  = g_sem / (abs(g_iso) + 1e-8)
    return g_iso, g_sem, ratio


# ══════════════════════════════════════════════════════════════════════════════
# Cross-step velocity-error correlation  ρ̂_η(s)
# [F1] prompt sampled OUTSIDE σ-loop
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_eta(
    tr, sigma_grid, x0_list, prompt_bank: PromptBank,
    h, w, d, guidance_val=3.5,
):
    """
    [F1] enc/pool are drawn once per x0 sample (outside the σ-loop).
    Independent ε per σ point eliminates shared-noise floor artefact.
    Returns: C (covariance M×M), rho (normalised),
             delta_sigmas (upper-tri pairs), rho_pairs.
    """
    M = len(sigma_grid)
    C = np.zeros((M, M), dtype=np.float64)
    n = 0

    for xi in x0_list:
        xi = xi.unsqueeze(0) if xi.dim() == 3 else xi

        # [F1] sample prompt ONCE per x0, held fixed across all σ
        enc, pool = prompt_bank.sample(1)
        img_ids, txt_ids = make_ids(1, h, w, enc.shape[1], xi.device, xi.dtype)

        errs = []
        for s in sigma_grid:
            noise = torch.randn_like(xi)          # ε independent per σ ✓
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
# [F1] prompt outside σ-loop  [F2] adaptive half-step
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_vdot_fd1(
    tr, sigma_grid, x0_list, prompt_bank: PromptBank,
    h, w, d, dsigma=0.025, guidance_val=3.5,
):
    """
    Path-coherent central-difference: same (x₀, ε) for lo/hi.
    [F1] enc/pool sampled once per x0 outside the σ-loop.
    [F2] adaptive half-step near boundaries.
    Returns: C, rho, delta_sigmas, rho_pairs.
    """
    M  = len(sigma_grid)
    C  = np.zeros((M, M), dtype=np.float64)
    n  = 0

    for xi in x0_list:
        xi = xi.unsqueeze(0) if xi.dim() == 3 else xi

        # [F1] sample prompt ONCE per x0
        enc, pool = prompt_bank.sample(1)
        img_ids, txt_ids = make_ids(1, h, w, enc.shape[1], xi.device, xi.dtype)

        vdots = []
        for s in sigma_grid:
            # [F2] adaptive half-step
            half = min(dsigma, s - 1e-4, 1.0 - s - 1e-4)
            if half < 1e-5:
                vdots.append(None)
                continue

            s_lo = s - half
            s_hi = s + half

            noise = torch.randn_like(xi)
            xlo   = (1 - s_lo) * xi + s_lo * noise
            xhi   = (1 - s_hi) * xi + s_hi * noise

            vlo   = flux_vel(tr, xlo, s_lo, enc, pool, img_ids, txt_ids, h, w, guidance_val)
            vhi   = flux_vel(tr, xhi, s_hi, enc, pool, img_ids, txt_ids, h, w, guidance_val)

            vd    = (vhi.float() - vlo.float()) / (2.0 * half)
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
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.abs(rho_vals[:-1]) + 1e-8)
    MIN_FLAT = max(5, len(rho_vals) // 10)

    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i : i + MIN_FLAT] < 0.02):
            plateau_start = i
            break
    if plateau_start is None:
        warnings.warn("[stats_flux] No plateau found — extend rho_s range.", UserWarning)
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

    ax = fig.add_subplot(gs[0, 0])
    if g_vals is not None and g_vals.any():
        ax.plot(sigma_grid, g_vals, color="#2d6a9f", lw=2)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
    else:
        ax.text(0.5, 0.5, "g(σ) not estimated\n(--no_g)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("σ"); ax.set_ylabel("g(σ)")
    ax.set_title("Scalar Jacobian Proxy  g(σ)"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(sigma_grid, s2_eta,  color="#b5451b", lw=2,   label="σ²_η (vel-err)")
    ax.semilogy(sigma_grid, s2_vdot, color="#9b3fa0", lw=2, ls="--", label="σ²_v̇_fd1 (disc)")
    ax.set_xlabel("σ"); ax.set_title("Error Variances σ²_η & σ²_v̇_fd1")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(dl_eta, rp_eta, s=4, alpha=0.2, color="#999", rasterized=True)
    ax.plot(rho_s_eta, rho_eta, color="#2d6a9f", lw=2.5, label="global ρ̂_η")
    if plateau_eta:
        ax.axhline(plateau_eta["rho_infty"], color="#e05c00", lw=1.5, ls="--",
                   label=f"ρ_∞={plateau_eta['rho_infty']:.3f}")
    ax.set_xlabel("|Δσ|"); ax.set_ylabel("ρ̂"); ax.set_title("Score-approx error  ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

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

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rho_s_eta, rho_eta, color="k", lw=2, ls="--", alpha=0.5, label="global")
    for k, (rs, rv) in enumerate(seg_curves):
        lo, hi = seg_bounds[k], seg_bounds[k + 1]
        ax.plot(rs, rv, color=seg_colors[k % len(seg_colors)], lw=2,
                label=f"σ̄∈[{lo:.2f},{hi:.2f})")
    ax.set_xlabel("|Δσ|"); ax.set_title("Per-Segment  ρ̂_η(s)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

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

    # ── Load full pipeline ────────────────────────────────────────────────────
    pipe = load_flux_pipeline(args.model, device, dtype)
    
    # ── [F4] Query actual Flux sigma range ───────────────────────────────────
    actual_sigma_min, actual_sigma_max = get_flux_sigma_range(pipe, args.latent_size, nfe=50)

    # ── σ grid: use actual scheduler range ───────────────────────────────────
    # Keep finer spacing near low-σ end where costs vary rapidly
    sg_min = max(actual_sigma_min, 1e-3)
    sg_max = min(actual_sigma_max, 1.0 - 1e-3)
    n_lo   = args.n_sigma // 3
    n_hi   = args.n_sigma - n_lo
    sg     = np.concatenate([
        np.linspace(sg_min,  0.35, n_lo, endpoint=False),
        np.linspace(0.35,    sg_max, n_hi),
    ])
    print(f"  σ grid: {len(sg)} points  [{sg[0]:.4f}, {sg[-1]:.4f}]  "
          f"(aligned to Flux scheduler range)")

    h, w    = args.latent_size, args.latent_size
    d       = 16 * h * w
    txt_seq = args.txt_seq
    guidance_val = args.guidance
    print(f"  latent: 16×{h}×{w}  d={d:,}  txt_seq={txt_seq}")

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
    actual_txt_seq = enc_all.shape[1]
    if actual_txt_seq != txt_seq:
        print(f"  [prompts] actual txt_seq={actual_txt_seq} "
              f"(overrides --txt_seq={txt_seq})")
        txt_seq = actual_txt_seq

    prompt_bank = PromptBank(enc_all, pool_all, device, dtype, seed=args.seed)

    # Offload text encoders
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.cpu()
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── [F3] Real VAE latents ─────────────────────────────────────────────────
    x0_all = encode_images_to_latents(
        pipe, args.n_samples, device, dtype,
        latent_size=h, seed=args.seed,
    )
    # VAE already offloaded inside encode_images_to_latents

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

    # ── σ²_v̇_fd1(σ)  [F2] ─────────────────────────────────────────────────────
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

    wl         = min(7, (len(sg) // 4) * 2 + 1)
    log_vd     = savgol_filter(np.log(np.clip(s2_vdot, 1e-30, None)), wl, 2)
    s2_vdot_sm = np.exp(log_vd)

    # ── g(σ) ─────────────────────────────────────────────────────────────────
    g_vals = np.zeros(len(sg), dtype=np.float32)
    if not args.no_g:
        n_g  = min(args.n_g_samples, args.n_samples)
        x0_g = [x0_all[i] for i in range(n_g)]
        print(f"\n[stats_flux] g(σ)  n_g_samples={n_g}  n_probes={args.n_hutchinson} …")
        for k, s in enumerate(sg):
            g_vals[k], g_sem, ratio = est_g_at_sigma(
                tr, s, x0_g, prompt_bank, h, w, d,
                n_probes=args.n_hutchinson,
                delta=args.g_delta,
                guidance_val=guidance_val,
            )

            if (k + 1) % max(1, len(sg) // 8) == 0:
                print(f"  g  {k+1}/{len(sg)}  σ={s:.3f}  "
                    f"g_iso={g_vals[k]:.5f}  "
                    f"g_sem={g_sem:.5f}  "
                    f"ratio={ratio:.3f}")

    # ── ρ̂_η  [F1] ─────────────────────────────────────────────────────────────
    n_ell  = min(args.n_ell_samples, args.n_samples)
    x0_ell = [x0_all[i] for i in range(n_ell)]
    print(f"\n[stats_flux] ρ̂_η  ({n_ell} samples) …")
    _, rho_mat_eta, dl_eta, rp_eta = est_rho_eta(
        tr, sg, x0_ell, prompt_bank, h, w, d, guidance_val)
    rho_s_eta, rho_eta = _extract_rho_curve_viz(dl_eta, rp_eta, anchor_at_one=True)
    plateau_eta = analyse_plateau(rho_s_eta, rho_eta, "score-approx error η")

    # ── ρ̂_v̇_fd1  [F1, F2] ────────────────────────────────────────────────────
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

    # ── Save .npz  [F4: use actual scheduler sigma range] ─────────────────────
    safe = args.model.replace("/", "--").replace(":", "--")
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz  = out / f"{safe}.npz"

    save = dict(
        # FM-native keys
        t_grid              = sg.astype(np.float32),
        # [F4] actual Flux scheduler boundaries, not hand-tuned grid endpoints
        sigma_min           = np.float32(actual_sigma_min),
        sigma_max           = np.float32(actual_sigma_max),
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
    print(f"  σ grid          : [{sg[0]:.4f}, {sg[-1]:.4f}]  N={len(sg)}")
    print(f"  σ_min/max (sched): [{actual_sigma_min:.5f}, {actual_sigma_max:.5f}]")
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