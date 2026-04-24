"""
hpsv2_eval.py  —  HPSv2 benchmark evaluation for BornSchedule vs baselines.

Schedules compared:
  born         — BornSchedule (optimised)
  logSNR       — uniform in log-SNR
  uniform      — linear in σ
  beta         — Beta(0.6, 0.6) quantile (SD3/FLUX training distribution)
  cosine       — cosine alpha init (no optimisation)
  flux_default — pipe.scheduler as-is, zero intervention

Usage:
    CUDA_VISIBLE_DEVICES=0 python hpsv2_eval.py \
        --npz ~/.cache/opt_schedule/black-forest-labs--FLUX_1-dev.npz \
        --nfe 8 --out_dir ./hps_out

    # Quick smoke-test:
    python hpsv2_eval.py --npz ... --nfe 8 --out_dir ./hps_out --n_prompts 50

    # Score only (images already generated):
    python hpsv2_eval.py --npz ... --nfe 8 --out_dir ./hps_out --score_only
"""

import os, copy, types, argparse
os.environ["HF_HUB_DISABLE_XET"] = "1"

from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()

import hpsv2
from diffusers import FluxPipeline
from flux_schedule import optimize_schedule, _alpha_to_sigmas, _cosine_alpha_init


# ════════════════════════════════════════════════════════════════════════════
# Schedule definitions
# ════════════════════════════════════════════════════════════════════════════

def logsnr_sigmas(nfe, sigma_max, sigma_min):
    """Uniform in log-SNR.  σ=1/(1+e^λ) is decreasing in λ,
    so λ must increase to make σ decrease: linspace(lam_min→lam_max)."""
    lam_max = np.log((1 - sigma_min) / sigma_min)
    lam_min = np.log((1 - sigma_max) / sigma_max)
    lams    = np.linspace(lam_min, lam_max, nfe + 1)
    return 1.0 / (1.0 + np.exp(lams))


def uniform_sigmas(nfe, sigma_max, sigma_min):
    return np.linspace(sigma_max, sigma_min, nfe + 1)


def beta_sigmas(nfe, sigma_max, sigma_min, alpha=0.6, beta=0.6):
    from scipy.stats import beta as beta_dist
    q = np.linspace(1.0, 0.0, nfe + 1)
    s = beta_dist.ppf(q, alpha, beta)
    return sigma_min + (sigma_max - sigma_min) * s


def cosine_sigmas(nfe, sigma_max, sigma_min):
    """Cosine alpha init converted to sigmas — no optimisation."""
    alpha = _cosine_alpha_init(nfe, sigma_max, sigma_min)
    return _alpha_to_sigmas(alpha, sigma_max, sigma_min)


# ════════════════════════════════════════════════════════════════════════════
# Scheduler injection
# ════════════════════════════════════════════════════════════════════════════

def make_locked_scheduler(base_scheduler, sigmas_np):
    sched    = copy.deepcopy(base_scheduler)
    sigmas_t = torch.tensor(sigmas_np, dtype=torch.float32)
    if sigmas_t[-1].item() != 0.0:
        sigmas_t = torch.cat([sigmas_t, sigmas_t.new_zeros(1)])
    timesteps_t = (sigmas_t[:-1] * 1000.0).to(torch.float32)
    n = len(timesteps_t)

    def _locked(self, num_inference_steps=None, device=None,
                sigmas=None, mu=None, **kw):
        dev = device or "cpu"
        self.sigmas              = sigmas_t.to(dev)
        self.timesteps           = timesteps_t.to(dev)
        self.num_inference_steps = n
        self._step_index         = None
        self._begin_index        = None

    sched.set_timesteps = types.MethodType(_locked, sched)
    return sched


# ════════════════════════════════════════════════════════════════════════════
# Generation
# ════════════════════════════════════════════════════════════════════════════

def generate(pipe, prompt, nfe, locked_scheduler=None,
             seed=42, height=512, width=512, guidance_scale=3.5):
    """
    locked_scheduler=None  → flux_default: use pipe.scheduler as-is, no hook.
    locked_scheduler=<obj> → swap in custom scheduler for this call only.
    """
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    kw  = dict(
        prompt=prompt,
        num_inference_steps=nfe,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        generator=gen,
        output_type="pil",
    )
    if locked_scheduler is not None:
        orig = pipe.scheduler
        pipe.scheduler = locked_scheduler
        out = pipe(**kw).images[0]
        pipe.scheduler = orig
    else:
        # flux_default: pipe completely untouched
        out = pipe(**kw).images[0]
    return out


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--npz",          required=True)
    p.add_argument("--model",        default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--nfe",          type=int,   default=8)
    p.add_argument("--out_dir",      default="./hps_out")
    p.add_argument("--n_prompts",    type=int,   default=None,
                   help="Max prompts per style (None = all ~800)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--height",       type=int,   default=512)
    p.add_argument("--width",        type=int,   default=512)
    p.add_argument("--guidance",     type=float, default=3.5)
    p.add_argument("--hps_version",  default="v2.1", choices=["v2.0", "v2.1"])
    p.add_argument("--score_only",   action="store_true",
                   help="Skip generation, just run hpsv2.evaluate()")
    p.add_argument("--device",       default=None)
    # optimizer knobs
    p.add_argument("--w_rank1",      type=float, default=1.0)
    p.add_argument("--w_vres",       type=float, default=1.0)
    p.add_argument("--w_disc",       type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Read npz sigma range ─────────────────────────────────────────────────
    data      = np.load(args.npz)
    sigma_max = float(data["sigma_max"])
    sigma_min = float(data["sigma_min"])
    print(f"[hpsv2_eval] σ range: [{sigma_min:.4f}, {sigma_max:.4f}]")

    # ── Build schedules ──────────────────────────────────────────────────────
    # sigmas=None signals flux_default (no hook at all)
    print(f"[hpsv2_eval] optimising BornSchedule (NFE={args.nfe}) …")
    sigmas_born = optimize_schedule(
        args.npz, args.nfe,
        sigma_max_override=sigma_max,
        sigma_min_override=sigma_min,
        n_steps=2000, n_restarts=3, verbose=False,
        w_rank1=args.w_rank1, w_vres=args.w_vres, w_disc=args.w_disc,
    )
    print(f"  born:   {np.round(sigmas_born, 3).tolist()}")

    sigmas_map = {
        "born":         sigmas_born,
        "logSNR":       logsnr_sigmas(args.nfe, sigma_max, sigma_min),
        "uniform":      uniform_sigmas(args.nfe, sigma_max, sigma_min),
        "beta":         beta_sigmas(args.nfe, sigma_max, sigma_min),
        "cosine":       cosine_sigmas(args.nfe, sigma_max, sigma_min),
        "flux_default": None,
    }

    for name, s in sigmas_map.items():
        if s is not None:
            print(f"  {name:14s}: {np.round(s, 3).tolist()}")
        else:
            print(f"  {name:14s}: (pipe.scheduler, no hook)")

    # ── Generation phase ─────────────────────────────────────────────────────
    if not args.score_only:
        device   = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        HF_TOKEN = os.getenv("HF_TOKEN")

        print(f"\n[hpsv2_eval] loading {args.model} …")
        pipe = FluxPipeline.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, token=HF_TOKEN,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        base_sched = pipe.scheduler

        gen_kw = dict(nfe=args.nfe, seed=args.seed,
                      height=args.height, width=args.width,
                      guidance_scale=args.guidance)

        all_prompts = hpsv2.benchmark_prompts('all')

        for sched_name, sigmas in sigmas_map.items():
            locked = make_locked_scheduler(base_sched, sigmas) if sigmas is not None else None
            print(f"\n[hpsv2_eval] generating — schedule: {sched_name}")

            for style, style_prompts in all_prompts.items():
                if args.n_prompts is not None:
                    style_prompts = style_prompts[: args.n_prompts]

                img_dir = out / sched_name / style
                img_dir.mkdir(parents=True, exist_ok=True)

                n_existing = sum(1 for _ in img_dir.glob("*.jpg"))
                if n_existing >= len(style_prompts):
                    print(f"  {style}: {n_existing} images already exist, skipping")
                    continue

                print(f"  {style}: {len(style_prompts)} prompts …", flush=True)
                for idx, prompt in enumerate(style_prompts):
                    save_path = img_dir / f"{idx:05d}.jpg"
                    if save_path.exists():
                        continue
                    img = generate(pipe, prompt, locked_scheduler=locked, **gen_kw)
                    img.save(str(save_path), quality=95)

                print(f"  {style}: done")

    # ── Scoring phase ────────────────────────────────────────────────────────
    print(f"\n[hpsv2_eval] scoring with HPS {args.hps_version} …\n")

    for sched_name in sigmas_map:
        img_path = out / sched_name
        if not img_path.exists():
            print(f"  {sched_name}: directory not found, skipping\n")
            continue
        print(f"=== {sched_name} ===")
        hpsv2.evaluate(str(img_path), hps_version=args.hps_version)
        print()

    print(f"[hpsv2_eval] done — images in {out}")


if __name__ == "__main__":
    main()
