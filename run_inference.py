"""
BornSchedule inference on google/ddpm-cifar10-32
Usage:
    python run_inference.py --steps 20 --batch 16 --seed 42
"""
import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import torch
import numpy as np
from pathlib import Path

from dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from scheduling_born import OptimalSchedule

# ── diffusers (only used to load the checkpoint) ──────────────────────────────
from diffusers import DDPMPipeline


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10(device):
    """
    Returns (unet, ns):
        unet – epsilon-prediction UNet on `device`
        ns   – NoiseScheduleVP built from the checkpoint's alphas_cumprod
    """
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    unet = pipe.unet.eval().to(device)
    scheduler = pipe.scheduler  # DDPMScheduler, only needed for betas

    # NoiseScheduleVP wants alphas_cumprod (= ᾱ_n in DDPM notation)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    ns = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod)

    return unet, ns


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DPM-Solver wrapper
# ─────────────────────────────────────────────────────────────────────────────

def make_dpm_solver(unet, ns, device):
    """
    Wraps the UNet and builds a DPM_Solver instance.
    """
    def raw_unet(x, t_input, **kwargs):
        # ensure t_input is on same device as x
        t_input = t_input.to(x.device)
        return unet(x, t_input).sample

    model_fn = model_wrapper(
        raw_unet,
        ns,
        model_type="noise",        # epsilon-prediction
        guidance_type="uncond",
    )

    dpm_solver = DPM_Solver(
        model_fn,
        ns,
        algorithm_type="dpmsolver",   # DPM-Solver (epsilon prediction form)
    )
    return dpm_solver


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BornSchedule inference loop
# ─────────────────────────────────────────────────────────────────────────────

def born_sample(dpm_solver, x_T, born, steps, device):
    """
    Run BornSchedule-optimised singlestep DPM-Solver-2.
    """
    assert steps % 2 == 0, "steps must be even for pure DPM-Solver-2 sampling"

    ns = dpm_solver.noise_schedule
    K  = steps // 2

    # ── optimise ──────────────────────────────────────────────────────────────
    born._optimise(K)
    lambdas_opt = born.get_lambdas()
    
    lam_tensor = torch.tensor(lambdas_opt, dtype=torch.float32, device=device)
    t_outer    = ns.inverse_lambda(lam_tensor)  # shape (K+1,), t decreasing

    x = x_T
    with torch.no_grad():
        for k in range(K):
            s = t_outer[k    ].reshape(1)
            t = t_outer[k + 1].reshape(1)

            # Inner time grid: [s, midpoint, t] with logSNR-uniform spacing
            t_inner   = dpm_solver.get_time_steps(
                            "logSNR", s.item(), t.item(), N=2, device=device
                        ).to(device)
            lam_inner = ns.marginal_lambda(t_inner)
            h         = lam_inner[-1] - lam_inner[0]
            r1        = float((lam_inner[1] - lam_inner[0]) / h)

            x = dpm_solver.singlestep_dpm_solver_update(
                x, s, t,
                order=2,
                solver_type="dpmsolver",
                r1=r1,
            )
    return x, lambdas_opt


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Baseline: original DPM-Solver-2 with uniform logSNR spacing
# ─────────────────────────────────────────────────────────────────────────────

def baseline_sample(dpm_solver, x_T, steps, ns):
    """Uniform-logSNR singlestep DPM-Solver-2."""
    with torch.no_grad():
        x = dpm_solver.sample(
            x_T,
            steps=steps,
            t_start=ns.T,
            t_end=1.0 / ns.total_N,
            order=2,
            skip_type="logSNR",
            method="singlestep",
        )
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Post-processing & save
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(x):
    """x: (B,3,H,W) in [-1,1] → list of PIL images"""
    from torchvision.utils import make_grid
    from PIL import Image
    grid = make_grid(x.clamp(-1, 1).cpu(), nrow=4, normalize=True, value_range=(-1, 1))
    arr  = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=10, help="Total NFE (must be even)")
    parser.add_argument("--batch",  type=int, default=16, help="Number of images to sample")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--outdir", type=str, default="samples")
    parser.add_argument("--no_baseline", action="store_true", help="Skip baseline sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── deterministic seeds
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ── load model
    print("Loading model…")
    unet, ns = load_cifar10(device)
    dpm_solver = make_dpm_solver(unet, ns, device)

    # ── BornSchedule
    born = OptimalSchedule(
        "google/ddpm-cifar10-32",
        r1=0.5,
        max_iter=26000,
        lr=1e-2,
        tol=1e-7,
        verbose=True,
    )

    # ── sample
    x_T = torch.randn(args.batch, 3, 32, 32, device=device)
    print(f"\n── BornSchedule ({args.steps} NFE) ──")
    x_born, lambdas_opt = born_sample(dpm_solver, x_T.clone(), born, args.steps, device)

    if not args.no_baseline:
        print(f"\n── Baseline logSNR ({args.steps} NFE) ──")
        x_base = baseline_sample(dpm_solver, x_T.clone(), args.steps, ns)

    # ── diagnostics
    lambdas_opt = born.get_lambdas()
    if lambdas_opt is not None:
        print(f"\nOptimal λ nodes: {lambdas_opt}")
        residuals = born.equidistribution_residuals()
        print(f"V·Γ² residuals (should be ≈ uniform at KKT): {residuals}")
        print(f"Cost:  {born.cost_at_schedule():.6e}")

        # Uniform-logSNR cost for comparison
        K = args.steps // 2
        lam_uniform = np.linspace(born.lambda_min, born.lambda_max, K + 1)
        
        from scheduling_born import _cost_functional
        cost_uniform = _cost_functional(
            lam_uniform,
            g_fn          = born.g_fn,
            sigma2_fn     = born.sigma2_fn,
            sigma2_gpp_fn = born.sigma2_gpp_fn,
            phi_res_fn    = born._phi_res_fn,
            rho_infty     = born.rho_infty,
            r1            = born.r1,
        )
        print(f"Uniform cost:  {cost_uniform:.6e}")
        print(f"Cost ratio (uniform / born): {cost_uniform / born.cost_at_schedule():.4f}x")

    # ── save
    os.makedirs(args.outdir, exist_ok=True)
    tensor_to_pil(x_born).save(f"{args.outdir}/born_step{args.steps}.png")
    print(f"Saved → {args.outdir}/born_step{args.steps}.png")

    if not args.no_baseline:
        tensor_to_pil(x_base).save(f"{args.outdir}/baseline_step{args.steps}.png")
        print(f"Saved → {args.outdir}/baseline_step{args.steps}.png")

    from proxy_validation import run_proxy_validation
    lambdas_opt = born.get_lambdas()
    K = args.steps // 2
    run_proxy_validation(born, lambdas_opt, dpm_solver, x_T, K, ns, device, args.outdir)

    

if __name__ == "__main__":
    main()