"""
BornSchedule vs uniform-logSNR FID comparison on google/ddpm-cifar10-32

Usage:
    python benchmark.py --nfe 10 --n_images 10000 --batch 16 --seed 42
    python benchmark.py --nfe 5 10 20 --n_images 10000 --batch 64
"""

import os, gc, shutil, argparse
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid

from dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from opt_schedule import OptimalSchedule, _cost_functional
from diffusers import DDPMPipeline

import lpips
from prdc import compute_prdc

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def free_gpu(*tensors_or_modules):
    """Move modules to CPU, delete tensors, empty cache."""
    for obj in tensors_or_modules:
        try:
            obj.cpu()
        except AttributeError:
            pass
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1]  →  (B,H,W,3) uint8"""
    x = x.clamp(-1, 1).cpu().float()
    x = (x + 1) / 2 * 255
    return x.permute(0, 2, 3, 1).numpy().astype(np.uint8)


def save_batch_to_dir(imgs_uint8: np.ndarray, out_dir: Path, start_idx: int):
    """Save (B,H,W,3) uint8 array as individual PNG files."""
    for i, img in enumerate(imgs_uint8):
        Image.fromarray(img).save(out_dir / f"{start_idx + i:06d}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10(device):
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    unet = pipe.unet.eval().to(device)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    ns = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod)
    # Keep scheduler on CPU — we only needed alphas_cumprod
    del pipe.scheduler
    return unet, ns


def make_dpm_solver(unet, ns):
    def raw_unet(x, t_input, **kwargs):
        return unet(x, t_input.to(x.device)).sample

    model_fn = model_wrapper(
        raw_unet, ns,
        model_type="noise",
        guidance_type="uncond",
    )
    return DPM_Solver(model_fn, ns, algorithm_type="dpmsolver")


# ─────────────────────────────────────────────────────────────────────────────
# Samplers
# ─────────────────────────────────────────────────────────────────────────────

def born_sample_batch(dpm_solver, x_T, t_outer, device):
    """
    Single-step DPM-Solver-2 with pre-computed outer time grid t_outer (K+1,).
    x_T: (B,3,H,W)  t_outer: decreasing tensor on device
    """
    ns = dpm_solver.noise_schedule
    K  = len(t_outer) - 1
    x  = x_T
    with torch.no_grad():
        for k in range(K):
            s = t_outer[k    ].reshape(1)
            t = t_outer[k + 1].reshape(1)
            t_inner = dpm_solver.get_time_steps(
                "logSNR", s.item(), t.item(), N=2, device=device
            )
            lam_inner = ns.marginal_lambda(t_inner)
            h  = lam_inner[-1] - lam_inner[0]
            r1 = float((lam_inner[1] - lam_inner[0]) / h)
            x  = dpm_solver.singlestep_dpm_solver_update(
                x, s, t, order=2, solver_type="dpmsolver", r1=r1,
            )
    return x


def baseline_sample_batch(dpm_solver, x_T, steps, ns):
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
# Generate-to-disk  (VRAM freed after every batch)
# ─────────────────────────────────────────────────────────────────────────────

def generate_to_disk(
    dpm_solver, ns, out_dir: Path,
    mode: str,          # "born" | "baseline"
    steps: int,
    n_images: int,
    batch_size: int,
    device: str,
    born: OptimalSchedule = None,   # required when mode=="born"
    seed: int = 42,
):
    """
    Write n_images PNGs to out_dir one batch at a time.
    No intermediate tensors linger outside the inner loop.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute optimised schedule once (only for born)
    t_outer = None
    if mode == "born":
        assert born is not None
        K = steps // 2
        lambdas_opt = born._optimise(K)
        born._lambdas = lambdas_opt
        lam_tensor = torch.tensor(lambdas_opt, dtype=torch.float32, device=device)
        t_outer    = ns.inverse_lambda(lam_tensor)  # (K+1,) decreasing

    saved = 0
    rng   = torch.Generator(device=device)
    while saved < n_images:
        rng.manual_seed(seed + saved)          # reproducible, non-overlapping
        this_batch = min(batch_size, n_images - saved)
        x_T = torch.randn(this_batch, 3, 32, 32, device=device, generator=rng)

        if mode == "born":
            x = born_sample_batch(dpm_solver, x_T, t_outer, device)
        else:
            x = baseline_sample_batch(dpm_solver, x_T, steps, ns)

        save_batch_to_dir(tensor_to_uint8(x), out_dir, saved)
        saved += this_batch

        # Free batch tensors immediately
        del x, x_T
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  [{mode}] saved {saved}/{n_images}", end="\r", flush=True)

    print(f"  [{mode}] done — {saved} images in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# FID  (runs after GPU is fully freed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid(img_dir: Path):
    """
    FID against CIFAR-10 train using torch-fidelity.
    Downloads CIFAR-10 on first run via torchvision (本地缓存).
    """
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError("Run: pip install torch-fidelity")

    metrics = torch_fidelity.calculate_metrics(
        input1=str(img_dir),
        input2="cifar10-train",   
        fid=True,
        isc=False,
        kid=False,
        cuda=torch.cuda.is_available(),
        batch_size=256,
        datasets_root=".fid_cache",   
        datasets_download=True,
        verbose=False,
    )
    return metrics["frechet_inception_distance"]
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe",      type=int,   nargs="+", default=[20],
                        help="NFE values to benchmark (must be even)")
    parser.add_argument("--n_images", type=int,   default=10000)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--outdir",   type=str,   default="fid_out")
    parser.add_argument("--keep_imgs", action="store_true",
                        help="Keep generated PNGs after FID (default: delete)")
    # BornSchedule optimiser knobs
    parser.add_argument("--max_iter", type=int,   default=4000)
    parser.add_argument("--lr",       type=float, default=1e-2)
    parser.add_argument("--tol",      type=float, default=1e-7)
    args = parser.parse_args()

    for nfe in args.nfe:
        assert nfe % 2 == 0, f"NFE must be even, got {nfe}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load model once ───────────────────────────────────────────────────────
    print("Loading model…")
    unet, ns = load_cifar10(device)
    dpm_solver = make_dpm_solver(unet, ns)

    results = {}   # {nfe: {"born": fid_val, "baseline": fid_val}}

    for nfe in args.nfe:
        print(f"\n{'='*60}")
        print(f"  NFE = {nfe}  ({nfe//2} outer steps, r1=0.5)")
        print(f"{'='*60}")
        results[nfe] = {}

        base_dir = Path(args.outdir) / f"nfe{nfe}"

        # ── BornSchedule ──────────────────────────────────────────────────────
        born = OptimalSchedule(
            "google/ddpm-cifar10-32",
            r1=0.5,
            max_iter=args.max_iter,
            lr=args.lr,
            tol=args.tol,
            verbose=True,
        )
        born_dir = base_dir / "born"
        print(f"\n[born] Generating {args.n_images} images → {born_dir}")
        generate_to_disk(
            dpm_solver, ns, born_dir,
            mode="born", steps=nfe,
            n_images=args.n_images, batch_size=args.batch,
            device=device, born=born, seed=args.seed,
        )

        # Print schedule diagnostics
        lambdas_opt = born.get_lambdas()
        if lambdas_opt is not None:
            residuals = born.equidistribution_residuals()
            cost_born = born.cost_at_schedule()
            K = nfe // 2
            lam_uniform = np.linspace(born.lambda_min, born.lambda_max, K + 1)
            cost_unif = _cost_functional(
                lam_uniform,
                g_fn          = born.g_fn,
                sigma2_fn     = born.sigma2_fn,
                sigma2_gpp_fn = born.sigma2_gpp_fn,
                ell_gpp       = born.ell_gpp,
                phi_res_fn    = born._phi_res_fn,
                rho_infty     = born.rho_infty,
                r1            = born.r1,
            )
            # print(f"  ell_gpp:          {born.ell_gpp:.4f}")
            print(f"  λ nodes:          {np.round(lambdas_opt, 4)}")
            print(f"  V·Γ² residuals:   {np.round(residuals, 4)}")
            print(f"  Born cost:        {cost_born:.4e}")
            print(f"  Uniform cost:     {cost_unif:.4e}")
            print(f"  Ratio (unif/born):{cost_unif/cost_born:.4f}x")

        # ── Baseline ─────────────────────────────────────────────────────────
        base_dir_imgs = base_dir / "baseline"
        print(f"\n[baseline] Generating {args.n_images} images → {base_dir_imgs}")
        generate_to_disk(
            dpm_solver, ns, base_dir_imgs,
            mode="baseline", steps=nfe,
            n_images=args.n_images, batch_size=args.batch,
            device=device, seed=args.seed,
        )

        # ── Unload model before FID ───────────────────────────────────────────
        # Move UNet to CPU and clear cache so Inception can use full VRAM
        print("\n[FID] Offloading UNet to CPU before Inception…")
        unet.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Compute FID ───────────────────────────────────────────────────────
        print("[FID] Computing BornSchedule FID…")
        fid_born = compute_fid(born_dir)
        results[nfe]["born"] = fid_born
        print(f"  Born FID:     {fid_born:.2f}")

        print("[FID] Computing Baseline FID…")
        fid_base = compute_fid(base_dir_imgs)
        results[nfe]["baseline"] = fid_base
        print(f"  Baseline FID: {fid_base:.2f}")

        # ── Reload UNet for next NFE ──────────────────────────────────────────
        if nfe != args.nfe[-1]:
            print("[reload] Moving UNet back to GPU…")
            unet.to(device)

        # ── Clean up images ───────────────────────────────────────────────────
        if not args.keep_imgs:
            shutil.rmtree(base_dir)
            print(f"[cleanup] Removed {base_dir}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FID Summary  ({args.n_images} images, CIFAR-10 train)")
    print(f"{'='*60}")
    print(f"{'NFE':>6}  {'Born FID':>12}  {'Baseline FID':>12}  {'Δ FID':>10}")
    print("-" * 50)
    for nfe in args.nfe:
        b = results[nfe]["born"]
        u = results[nfe]["baseline"]
        print(f"{nfe:>6}  {b:>12.2f}  {u:>12.2f}  {u-b:>+10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()