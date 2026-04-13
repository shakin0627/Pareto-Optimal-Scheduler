"""
patch_vdot_fd1.py  —  Recompute v̇ statistics using FIRST-ORDER central difference.

理论依据:
  D_j = E‖dv_θ/dt‖²/d
  dv_θ/dt ≈ [v_θ(x_{σ+dσ}, σ+dσ) − v_θ(x_{σ−dσ}, σ−dσ)] / (2·dσ)

原 stats_flux.py 用的是 (v_hi − 2v_mid + v_lo)/dσ² (二阶差分, 加速度),
本脚本修正为一阶中心差分。

输出: 在原 .npz 旁边创建备份，然后 patch 进以下新 key：
  sigma2_vdot_fd1      : σ²_v̇(σ) 一阶修正值 (平滑后)
  sigma2_gpp_values_fd1: 同上 (born_schedule.py 兼容别名)
  rho_s_vdot_fd1       : ρ̂_v̇ 插值 x 轴
  rho_values_vdot_fd1  : ρ̂_v̇ 插值 y 轴

所有原始 key 完全保留，不删除不覆盖。

用法:
  python patch_vdot_fd1.py \\
      --npz ~/.cache/opt_schedule/black-forest-labs--FLUX.1-dev.npz \\
      --model black-forest-labs/FLUX.1-dev \\
      --n_samples 128 --n_ell_samples 64 --batch 1 \\
      --dsigma 0.025
"""

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"]    = "1"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.signal     import savgol_filter


# ══════════════════════════════════════════════════════════════════════════════
# Model loading  (identical to stats_flux.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_flux(model_name, device, dtype=torch.bfloat16):
    from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    print(f"  [flux] loading transformer …")
    tr = FluxTransformer2DModel.from_pretrained(
        model_name, subfolder="transformer",
        torch_dtype=dtype, token=HF_TOKEN,
    ).to(device).eval()
    for m in tr.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler", token=HF_TOKEN,
    )
    return tr, sched


def pack_latents(x, h, w):
    B = x.shape[0]
    x = x.view(B, 16, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (h // 2) * (w // 2), 64)

def unpack_latents(x, h, w):
    B = x.shape[0]
    x = x.view(B, h // 2, w // 2, 16, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 16, h, w)

def null_cond(B, txt_seq, device, dtype):
    enc  = torch.zeros(B, txt_seq, 4096, device=device, dtype=dtype)
    pool = torch.zeros(B, 768,     device=device, dtype=dtype)
    return enc, pool

def make_ids(B, h, w, txt_seq, device, dtype):
    hp, wp = h // 2, w // 2
    img = torch.zeros(hp, wp, 3, device=device, dtype=dtype)
    img[..., 1] += torch.arange(hp, device=device, dtype=dtype)[:, None]
    img[..., 2] += torch.arange(wp, device=device, dtype=dtype)[None, :]
    img = img.reshape(1, hp * wp, 3).expand(B, -1, -1)
    txt = torch.zeros(1, txt_seq, 3, device=device, dtype=dtype).expand(B, -1, -1)
    return img.contiguous(), txt.contiguous()

def _needs_guidance(tr):
    return bool(getattr(tr.config, "guidance_embeds", False))

def flux_vel(tr, x_t, sigma, enc, pool, img_ids, txt_ids, h, w,
             guidance_val=3.5):
    B  = x_t.shape[0]
    dt = x_t.dtype
    dev = x_t.device
    xp = pack_latents(x_t, h, w)
    t  = torch.full((B,), sigma, device=dev, dtype=dt)
    g  = (torch.full((B,), guidance_val, device=dev, dtype=dt)
          if _needs_guidance(tr) else None)
    vp = tr(hidden_states=xp, timestep=t,
             encoder_hidden_states=enc, pooled_projections=pool,
             img_ids=img_ids, txt_ids=txt_ids,
             guidance=g, return_dict=False)[0]
    return unpack_latents(vp, h, w)


# ══════════════════════════════════════════════════════════════════════════════
# σ²_v̇(σ) — 一阶中心差分  dv_θ/dt ≈ (v_hi − v_lo) / (2·dσ)
#
# 注: x_lo/x_hi 路径一致 (同一 x₀ 和 ε), 消除随机噪声偏差
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_sigma2_vdot_fd1(tr, sigma, x0, enc, pool, img_ids, txt_ids,
                         h, w, d, dsigma=0.025, guidance_val=3.5) -> float:
    """
    D(σ) = E‖dv_θ/dt‖²/d
         ≈ E‖(v_θ(x_{σ+dσ}, σ+dσ) − v_θ(x_{σ−dσ}, σ−dσ)) / (2·dσ)‖² / d
    路径相干: x_lo/x_hi 共享同一 (x₀, ε)。
    """
    s_lo = max(0.0, sigma - dsigma)
    s_hi = min(1.0, sigma + dsigma)
    ds   = (s_hi - s_lo) / 2.0          # effective half-step
    if ds < 1e-6:
        return 0.0

    noise = torch.randn_like(x0)
    xlo   = (1 - s_lo) * x0 + s_lo * noise
    xhi   = (1 - s_hi) * x0 + s_hi * noise

    vlo   = flux_vel(tr, xlo, s_lo, enc, pool, img_ids, txt_ids, h, w, guidance_val)
    vhi   = flux_vel(tr, xhi, s_hi, enc, pool, img_ids, txt_ids, h, w, guidance_val)

    # first-order central difference  (NOT second-order)
    vdot  = (vhi.float() - vlo.float()) / (2.0 * ds)
    return float((vdot ** 2).sum()) / (x0.shape[0] * d)


# ══════════════════════════════════════════════════════════════════════════════
# ρ̂_v̇(s) — 一阶导数的跨步相关
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def est_rho_vdot_fd1(tr, sigma_grid, x0, enc1, pool1, img1, txt1,
                      h, w, d, dsigma=0.025, guidance_val=3.5):
    """
    对每个样本, 在 sigma_grid 上计算 dv_θ/dt (一阶差分),
    然后估计跨 σ 点对的归一化相关 ρ̂_v̇(|Δσ|)。
    """
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

            noise = torch.randn_like(xi)         # path-coherent: same (x₀, ε) for lo/hi
            xlo   = (1 - s_lo) * xi + s_lo * noise
            xhi   = (1 - s_hi) * xi + s_hi * noise

            vlo   = flux_vel(tr, xlo, s_lo, enc1, pool1, img1, txt1, h, w, guidance_val)
            vhi   = flux_vel(tr, xhi, s_hi, enc1, pool1, img1, txt1, h, w, guidance_val)

            vd    = (vhi.float() - vlo.float()) / (2.0 * ds)   # first-order
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
            print(f"    ρ̂_v̇ (fd1): {n}/{len(x0)} samples")

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
# Utility: 相关曲线提取  (完全复制自 stats_flux.py)
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
    sa = np.concatenate([[0.0], sa])
    va = np.concatenate([[va[0]] if not anchor_at_one else [1.0], va])
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
    return dict(rho_infty=ri, frac_rank1=fr, ell_res=ell,
                spectral_mass_res=sm,
                verdict="STRONG rank-1" if fr > 0.3 else "WEAK rank-1")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # ── 备份原始 npz ──────────────────────────────────────────────────────────
    backup = npz_path.with_suffix(".bak.npz")
    shutil.copy2(npz_path, backup)
    print(f"[patch] backed up → {backup}")

    # ── 读取原始 npz ──────────────────────────────────────────────────────────
    orig = dict(np.load(str(npz_path), allow_pickle=False))
    sg   = orig["t_grid"].astype(np.float64)   # σ grid from original run
    print(f"[patch] σ grid: {len(sg)} points  [{sg[0]:.3f}, {sg[-1]:.3f}]")

    # ── 设备 & 模型 ───────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype  = torch.bfloat16
    print(f"[patch] device={device}  model={args.model}")

    tr, _ = load_flux(args.model, device, dtype)
    h, w  = args.latent_size, args.latent_size
    d     = 16 * h * w
    txt_seq = args.txt_seq
    guidance_val = args.guidance

    # ── 随机 latent ───────────────────────────────────────────────────────────
    print(f"[patch] generating {args.n_samples} random latents …")
    x0_all = torch.randn(args.n_samples, 16, h, w, device=device, dtype=dtype)

    enc1, pool1 = null_cond(1, txt_seq, device, dtype)
    img1, txt1  = make_ids(1, h, w, txt_seq, device, dtype)

    # ── σ²_v̇ (一阶 FD) ────────────────────────────────────────────────────────
    print(f"\n[patch] σ²_v̇ (fd1) at {len(sg)} σ points …")
    s2_vdot_fd1 = np.zeros(len(sg))
    for k, s in enumerate(sg):
        acc = 0.0; nb = 0
        for start in range(0, args.n_samples, args.batch):
            xb = x0_all[start : start + args.batch]
            B  = xb.shape[0]
            ec, pc = null_cond(B, txt_seq, device, dtype)
            ii, ti = make_ids(B, h, w, txt_seq, device, dtype)
            acc += est_sigma2_vdot_fd1(tr, s, xb, ec, pc, ii, ti, h, w, d,
                                        dsigma=args.dsigma,
                                        guidance_val=guidance_val)
            nb += 1
        s2_vdot_fd1[k] = acc / nb
        if (k + 1) % max(1, len(sg) // 8) == 0:
            print(f"  σ²_v̇(fd1)  {k+1}/{len(sg)}  σ={s:.3f}  val={s2_vdot_fd1[k]:.4e}")

    # Savitzky-Golay 平滑 (同 stats_flux.py)
    wl = min(7, (len(sg) // 4) * 2 + 1)
    log_vd         = savgol_filter(np.log(np.clip(s2_vdot_fd1, 1e-30, None)), wl, 2)
    s2_vdot_fd1_sm = np.exp(log_vd)

    # ── ρ̂_v̇ (一阶 FD) ────────────────────────────────────────────────────────
    n_ell = min(args.n_ell_samples, args.n_samples)
    print(f"\n[patch] ρ̂_v̇ (fd1, {n_ell} samples) …")
    x0_ell = x0_all[:n_ell]
    _, rho_mat_vd, dl_vd, rp_vd = est_rho_vdot_fd1(
        tr, sg, x0_ell, enc1, pool1, img1, txt1,
        h, w, d, dsigma=args.dsigma, guidance_val=guidance_val)

    rho_s_vd, rho_vd = _extract_rho_curve(dl_vd, rp_vd, anchor_at_one=False)
    plateau_vd = analyse_plateau(rho_s_vd, rho_vd, "v̇-fd1")

    # ── Patch npz — 仅添加新 key，不删除/覆盖旧 key ───────────────────────────
    orig["sigma2_vdot_fd1"]       = s2_vdot_fd1_sm.astype(np.float32)
    orig["sigma2_gpp_values_fd1"] = s2_vdot_fd1_sm.astype(np.float32)  # born_schedule 兼容别名
    orig["rho_s_vdot_fd1"]        = rho_s_vd.astype(np.float32)
    orig["rho_values_vdot_fd1"]   = rho_vd.astype(np.float32)

    np.savez(str(npz_path), **orig)
    print(f"\n[patch] patched → {npz_path}")
    print(f"  new keys: sigma2_vdot_fd1, sigma2_gpp_values_fd1, "
          f"rho_s_vdot_fd1, rho_values_vdot_fd1")

    # 同时保存一阶 rho matrix
    rhomat_path = npz_path.parent / (npz_path.stem + "_rhomat_vdot_fd1.npy")
    np.save(str(rhomat_path), rho_mat_vd.astype(np.float32))
    print(f"  rho_mat (v̇ fd1) → {rhomat_path}")

    # ── 简要汇总 ──────────────────────────────────────────────────────────────
    print("\n══ Patch Summary ══════════════════════════════════════════")
    print(f"  σ²_v̇(fd1) range : [{s2_vdot_fd1_sm.min():.3e}, {s2_vdot_fd1_sm.max():.3e}]")
    print(f"  ρ_∞(v̇ fd1)     : {plateau_vd['rho_infty']:.4f}  "
          f"(rank-1 frac {plateau_vd['frac_rank1']:.2f})")
    print(f"  verdict          : {plateau_vd['verdict']}")
    print(f"  npz              : {npz_path}")
    print(f"  backup           : {backup}")
    print("═══════════════════════════════════════════════════════════\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Patch existing BornSchedule npz with first-order v̇ statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz",   required=True,
                   help="Path to existing .npz file produced by stats_flux.py")
    p.add_argument("--model", default="black-forest-labs/FLUX.1-dev",
                   help="HuggingFace model ID (must match the original run)")
    p.add_argument("--latent_size", type=int, default=64)
    p.add_argument("--txt_seq",     type=int, default=256)
    p.add_argument("--n_samples",   type=int, default=128,
                   help="Samples for σ²_v̇ estimation")
    p.add_argument("--n_ell_samples", type=int, default=64,
                   help="Samples for ρ̂_v̇ estimation")
    p.add_argument("--batch",         type=int, default=1)
    p.add_argument("--dsigma",        type=float, default=0.025,
                   help="Half-step for central difference dσ")
    p.add_argument("--guidance",      type=float, default=3.5)
    p.add_argument("--device",        default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())