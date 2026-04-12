import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
from pathlib import Path
from typing import Optional, Tuple
 
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
from diffusers import FluxPipeline

# ─────────────────────────────────────────────────────────────
# Time / flow
# ─────────────────────────────────────────────────────────────

def make_time_grid(n_t, t_min=1e-3, t_max=0.999):
    return np.linspace(t_min, t_max, n_t)


def flow_forward(x0, eps, t):
    return (1.0 - t) * x0 + t * eps


def true_velocity(x0, eps):
    return eps - x0


# ─────────────────────────────────────────────────────────────
# Flux model
# ─────────────────────────────────────────────────────────────

from diffusers import FluxPipeline


def load_model(model_name, device):
    pipe = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)

    pipe.transformer.eval()
    pipe.vae.eval()

    cfg = dict(
        in_channels=pipe.transformer.config.in_channels,
        sample_size=pipe.transformer.config.sample_size,
        vae_scale_factor=8,
    )

    return pipe, cfg


def model_velocity(pipe, x, t):
    """
    Correct Flux forward:
    hidden_states: [B, C, H, W]
    timestep: int 0-1000
    """
    B = x.shape[0]

    timesteps = torch.full(
        (B,),
        int(t * 1000),
        device=x.device,
        dtype=torch.long,
    )

    out = pipe.transformer(
        hidden_states=x,
        timestep=timesteps,
        encoder_hidden_states=None,
        pooled_projections=None,
        return_dict=True,
    )

    return out.sample


# ─────────────────────────────────────────────────────────────
# Hutchinson estimator
# ─────────────────────────────────────────────────────────────

def estimate_g_at_t(pipe, t_val, d, x0_batch, n_probes, device):
    acc = 0.0
    count = 0

    for xi in x0_batch:
        xi = xi.unsqueeze(0)

        eps = torch.randn_like(xi)
        x_t = flow_forward(xi, eps, t_val)

        for _ in range(n_probes):
            v_rnd = torch.randint(
                0, 2, x_t.shape, device=device, dtype=x_t.dtype
            ) * 2 - 1

            x_req = x_t.requires_grad_(True)
            v = model_velocity(pipe, x_req, t_val)

            proj = (v * v_rnd).sum()
            grad = torch.autograd.grad(proj, x_req)[0]

            acc += (grad * v_rnd).sum().item()
            count += 1

    return acc / (count * d)


# ─────────────────────────────────────────────────────────────
# σ² error
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma2(pipe, t_val, d, x0_batch, device):
    eps = torch.randn_like(x0_batch)
    x_t = flow_forward(x0_batch, eps, t_val)

    v_star = true_velocity(x0_batch, eps)
    v_pred = model_velocity(pipe, x_t, t_val)

    mse = ((v_pred - v_star) ** 2).sum().item()
    return mse / (x0_batch.shape[0] * d)


# ─────────────────────────────────────────────────────────────
# D(t) finite difference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_D(pipe, t_val, d, x0_batch, device, delta=0.02):
    t2 = min(t_val + delta, 0.999)

    eps = torch.randn_like(x0_batch)

    x1 = flow_forward(x0_batch, eps, t_val)
    x2 = flow_forward(x0_batch, eps, t2)

    v1 = model_velocity(pipe, x1, t_val)
    v2 = model_velocity(pipe, x2, t2)

    dv = (v2 - v1) / (t2 - t_val)

    return (dv ** 2).sum().item() / (x0_batch.shape[0] * d)


# ─────────────────────────────────────────────────────────────
# correlation kernel
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_kernel(pipe, t_grid, x0_batch, d, device):
    M = len(t_grid)
    C = np.zeros((M, M))

    for xi in x0_batch:
        xi = xi.unsqueeze(0)

        eps = torch.randn_like(xi)

        v_list = []

        for t in t_grid:
            x_t = flow_forward(xi, eps, t)
            v = model_velocity(pipe, x_t, t)
            v_list.append(v.flatten().cpu())

        for i in range(M):
            for j in range(i, M):
                dot = (v_list[i] * v_list[j]).sum().item() / d
                C[i, j] += dot
                C[j, i] += dot

    C /= len(x0_batch)

    diag = np.diag(C) + 1e-12
    corr = C / np.sqrt(np.outer(diag, diag))

    return corr


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def run(args):

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("[1] loading Flux ...")
    pipe, cfg = load_model(args.model, device)

    in_ch = cfg["in_channels"]
    H = W = cfg["sample_size"]
    d = in_ch * H * W

    print(f"[info] d = {d}")

    print("[2] sampling x0 ...")
    x0 = torch.randn(args.n_samples, in_ch, H, W, device=device)

    t_grid = make_time_grid(args.n_t, args.t_min, args.t_max)

    g_vals = []
    sigma_vals = []
    D_vals = []

    print("[3] computing statistics ...")

    for t in t_grid:

        g = estimate_g_at_t(
            pipe, t, d,
            x0[:args.batch],
            args.n_hutchinson,
            device
        )

        s2 = estimate_sigma2(pipe, t, d, x0[:args.batch], device)
        D = estimate_D(pipe, t, d, x0[:args.batch], device)

        g_vals.append(g)
        sigma_vals.append(s2)
        D_vals.append(D)

        print(f"t={t:.3f} g={g:.4f} sigma={s2:.4e} D={D:.4e}")

    g_vals = np.array(g_vals)
    sigma_vals = np.array(sigma_vals)
    D_vals = np.array(D_vals)

    print("[4] kernel estimation ...")
    corr = estimate_kernel(
        pipe, t_grid,
        x0[:args.n_ell_samples],
        d,
        device
    )

    # ── save ─────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)

    np.savez(
        out / "flux_stats.npz",
        t_grid=t_grid,
        g=g_vals,
        sigma=sigma_vals,
        D=D_vals,
        corr=corr,
    )

    # ── plot ─────────────────────────────
    plt.figure()
    plt.plot(t_grid, g_vals)
    plt.title("g(t)")
    plt.savefig(out / "g.png")

    plt.figure()
    plt.semilogy(t_grid, sigma_vals)
    plt.title("sigma^2(t)")
    plt.savefig(out / "sigma.png")

    plt.figure()
    plt.semilogy(t_grid, D_vals)
    plt.title("D(t)")
    plt.savefig(out / "D.png")

    print("[done] saved to", out)


# ─────────────────────────────────────────────────────────────

def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", default="./flux_stats")

    p.add_argument("--n_t", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--n_hutchinson", type=int, default=2)
    p.add_argument("--n_ell_samples", type=int, default=8)

    p.add_argument("--t_min", type=float, default=1e-3)
    p.add_argument("--t_max", type=float, default=0.999)

    p.add_argument("--device", default=None)

    return p.parse_args()


if __name__ == "__main__":
    run(parse())