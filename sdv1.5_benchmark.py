import argparse
import numpy as np
import torch
from diffusers import DPMSolverSinglestepScheduler, StableDiffusionPipeline
from PIL import Image
from pathlib import Path

MODEL_ID = "runwayml/stable-diffusion-v1-5"


# ─────────────────────────────────────────────
# scheduler
# ─────────────────────────────────────────────

def make_scheduler():
    return DPMSolverSinglestepScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
        algorithm_type="dpmsolver",
        solver_order=2,
        solver_type="midpoint",
        lower_order_final=True,
        final_sigmas_type="sigma_min",
    )


def inject_sigmas(scheduler, sigmas_target: np.ndarray, device=None):
    """
    统一 sigma 注入（保证 fairness）
    """
    N = len(sigmas_target)

    all_sigmas = ((1 - scheduler.alphas_cumprod) /
                  scheduler.alphas_cumprod) ** 0.5
    all_sigmas = all_sigmas.detach().cpu().numpy()
    log_sigmas = np.log(all_sigmas)

    t_indices = np.array([
        scheduler._sigma_to_t(s, log_sigmas) for s in sigmas_target
    ]).round().astype(np.int64)

    t_indices = np.clip(
        t_indices, 0, scheduler.config.num_train_timesteps - 1
    )

    scheduler.set_timesteps(
        timesteps=t_indices.tolist(),
        device=device
    )

    sigma_min = float(all_sigmas[-1])
    sigmas_full = np.concatenate([sigmas_target, [sigma_min]]).astype(np.float32)

    scheduler.sigmas = torch.from_numpy(sigmas_full)
    if device is not None:
        scheduler.sigmas = scheduler.sigmas.to(device)

    return scheduler


# ─────────────────────────────────────────────
# sigma schedules
# ─────────────────────────────────────────────

def born_sigmas(lambdas_opt: np.ndarray) -> np.ndarray:
    sigmas = 1.0 / np.sqrt(1.0 + np.exp(2.0 * lambdas_opt))
    return sigmas[::-1][:-1].copy()


def logsnr_sigmas(N: int, scheduler) -> np.ndarray:
    alpha_t = scheduler.alphas_cumprod.detach().cpu().numpy()
    sigma_t = ((1 - scheduler.alphas_cumprod) /
               scheduler.alphas_cumprod).detach().cpu().numpy()

    lam_all = np.log(alpha_t / np.clip(sigma_t, 1e-12, None))
    lambda_min, lambda_max = lam_all[-1], lam_all[1]

    lambdas = np.linspace(lambda_max, lambda_min, N + 1)[:-1]
    return 1.0 / np.sqrt(1.0 + np.exp(2.0 * lambdas))


def karras_sigmas(N: int, scheduler, rho: float = 7.0) -> np.ndarray:
    all_sigmas = ((1 - scheduler.alphas_cumprod) /
                  scheduler.alphas_cumprod) ** 0.5
    all_sigmas = all_sigmas.detach().cpu().numpy()

    sigma_min = float(all_sigmas[-1])
    sigma_max = float(all_sigmas[1])

    ramp = np.linspace(0, 1, N)
    return (sigma_max ** (1 / rho) +
            ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


AYS_SIGMAS_SD15 = {
    10: [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152],
    20: [14.615, 9.292, 6.475, 4.972, 3.861, 3.085, 2.479, 2.006, 1.623, 1.316,
         1.062, 0.862, 0.699, 0.568, 0.457, 0.367, 0.295, 0.234, 0.175, 0.117],
}


def ays_sigmas(N: int) -> np.ndarray:
    if N not in AYS_SIGMAS_SD15:
        raise ValueError(f"AYS only supports {list(AYS_SIGMAS_SD15.keys())}")
    return np.array(AYS_SIGMAS_SD15[N], dtype=np.float64)


# ─────────────────────────────────────────────
# pipeline run
# ─────────────────────────────────────────────

def run(pipe, prompt, sigmas_target, N, device):
    scheduler = make_scheduler()
    scheduler = inject_sigmas(scheduler, sigmas_target, device=device)
    pipe.scheduler = scheduler

    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=N,
            guidance_scale=7.5,
        ).images[0]

    return image


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, default="out.png")

    p.add_argument("--method", type=str,
                   choices=["born", "logsnr", "karras", "ays"],
                   default="karras")

    p.add_argument("--N", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")

    # optional for born schedule (placeholder)
    p.add_argument("--lambda_file", type=str, default=None)

    return p.parse_args()


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    device = args.device

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    ).to(device)

    scheduler = make_scheduler()

    # ── build sigma schedule ─────────────────────
    if args.method == "karras":
        sigmas = karras_sigmas(args.N, scheduler)

    elif args.method == "logsnr":
        sigmas = logsnr_sigmas(args.N, scheduler)

    elif args.method == "ays":
        sigmas = ays_sigmas(args.N)

    elif args.method == "born":
        if args.lambda_file is None:
            raise ValueError("--lambda_file required for born")
        lambdas = np.load(args.lambda_file)
        sigmas = born_sigmas(lambdas)

    else:
        raise ValueError("unknown method")

    print(f"[INFO] method={args.method}, N={args.N}")
    print(f"[INFO] sigmas = {sigmas}")

    # ── run ──────────────────────────────────────
    img = run(pipe, args.prompt, sigmas, args.N, device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)

    print(f"[DONE] saved to {args.output}")


if __name__ == "__main__":
    main()