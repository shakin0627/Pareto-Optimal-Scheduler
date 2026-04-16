#!/usr/bin/env python3
"""
-----------
  # one-time setup
  pip install diffusers transformers accelerate pyiqa open-clip-torch \
              pillow tqdm numpy torch torchvision

  # run (background, immune to SSH drop)
  nohup python flux_benchmark.py \
        --stats flux_stats.npz \
        --model "black-forest-labs/FLUX.1-dev" \
        --nfe_list 50 25 10 \
        --n_prompts 200 \
        --out_dir /media/ssd_horse/keying/eval_results \
        --device cuda \
  > eval_flux.log 2>&1 &

  tail -f eval_flux.log
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(ckpt_path: Path) -> dict:
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return {}


def save_progress(ckpt_path: Path, progress: dict):
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Schedule builders
# ─────────────────────────────────────────────────────────────────────────────

def uniform_sigma_schedule(nfe: int, sigma_max=0.97, sigma_min=0.02) -> np.ndarray:
    """Linear interpolation baseline."""
    return np.linspace(sigma_max, sigma_min, nfe + 1)


def born_sigma_schedule(npz_path: str, nfe: int,
                        sigma_max=0.97, sigma_min=0.02,
                        n_steps=1000, lr=1e-3, lr_decay=0.995,
                        n_restarts=2) -> np.ndarray:
    """Call flux_schedule.optimize_schedule (from the BornSchedule codebase)."""
    try:
        import flux_schedule as fs
    except ImportError:
        raise ImportError(
            "flux_schedule.py not found in PYTHONPATH.\n"
            "Place flux_schedule.py in the same directory or set PYTHONPATH."
        )
    return fs.optimize_schedule(
        npz_path=npz_path, nfe=nfe,
        sigma_max=sigma_max, sigma_min=sigma_min,
        n_steps=n_steps, lr=lr, lr_decay=lr_decay,
        n_restarts=n_restarts, verbose=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BornSchedule Flux evaluation (T2I-CompBench + CLIPIQA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stats",     default="/media/ssd_horse/keying/opt_schedule_out/black-forest-labs--FLUX.1-dev.npz",
                   help=".npz from stats_flux.py (required for BornSchedule)")
    p.add_argument("--model",     default="black-forest-labs/FLUX.1-dev",
                   help="HuggingFace Flux model name")
    p.add_argument("--nfe_list",  nargs="+", type=int, default=[50, 25, 10],
                   help="NFE values to evaluate")
    p.add_argument("--seeds",     nargs="+", type=int, default=[42],
                   help="Random seeds (one image per prompt per seed)")
    p.add_argument("--height",    type=int, default=512)
    p.add_argument("--width",     type=int, default=512)
    p.add_argument("--out_dir",   default="eval_results")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--born_n_steps",    type=int,   default=1000)
    p.add_argument("--born_lr",         type=float, default=1e-3)
    p.add_argument("--born_lr_decay",   type=float, default=0.995)
    p.add_argument("--born_restarts",   type=int,   default=2)
    p.add_argument("--sigma_max",       type=float, default=0.97)
    p.add_argument("--sigma_min",       type=float, default=0.02)
    p.add_argument("--guidance_scale",  type=float, default=3.5)
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip generation; only (re-)run scoring on existing images")
    p.add_argument("--baseline_tag",  default=None,
                   help="Tag used as 1× baseline for speedup (default: uniform_nfe<max>)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "progress.json"

    # ── Build schedule dict ────────────────────────────────────────────────
    schedules: dict[str, np.ndarray] = {}
    for nfe in args.nfe_list:
        # Uniform baseline
        uni = uniform_sigma_schedule(nfe, args.sigma_max, args.sigma_min)
        schedules[f"uniform_nfe{nfe}"] = uni
        # BornSchedule (skip if no stats file)
        if args.stats:
            print(f"\n  Optimising BornSchedule NFE={nfe} …")
            born = born_sigma_schedule(
                args.stats, nfe,
                sigma_max=args.sigma_max, sigma_min=args.sigma_min,
                n_steps=args.born_n_steps, lr=args.born_lr,
                lr_decay=args.born_lr_decay, n_restarts=args.born_restarts,
            )
            schedules[f"born_nfe{nfe}"] = born
            np.save(str(out_dir / f"born_nfe{nfe}.npy"), born)
        else:
            print(f"  [warn] No --stats provided; skipping BornSchedule NFE={nfe}.")

    # Save schedules for inspection
    sched_txt = out_dir / "schedules.txt"
    with open(sched_txt, "w") as f:
        for tag, sg in schedules.items():
            f.write(f"{tag}:\n  {np.round(sg,5).tolist()}\n\n")
    print(f"  Schedules saved → {sched_txt}")

if __name__ == "__main__":
    main() 