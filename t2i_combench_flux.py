"""
T2I-CompBench Generation Pipeline for BornSchedule
====================================================
Generates images for all dataset configs × NFE values × schedules.
Evaluation is left as TODO stubs.

"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Schedule loading
# ─────────────────────────────────────────────────────────────────────────────

def load_schedules(schedules_dir: Path) -> dict[str, np.ndarray]:
    """
    Returns {tag: sigma_array} for all born_nfe* .npy files found,
    plus matching uniform_nfe* schedules derived from the same [σ_max, σ_min].

    sigma_array is a *descending* float32 array of length NFE+1
    (i.e. sigma_array[0] = σ_max, sigma_array[-1] = σ_min > 0).
    """
    schedules: dict[str, np.ndarray] = {}
    npy_files = sorted(schedules_dir.glob("born_nfe*.npy"))

    if not npy_files:
        raise FileNotFoundError(
            f"No born_nfe*.npy files found in {schedules_dir}.\n"
            "Expected filenames like born_nfe5.npy, born_nfe10.npy, …"
        )

    print(f"\n[schedules] Loading from {schedules_dir}")
    for npy_path in npy_files:
        stem = npy_path.stem                            # e.g. "born_nfe10"
        nfe  = int(stem.replace("born_nfe", ""))
        sigmas = np.load(str(npy_path)).astype(np.float32)

        assert sigmas.ndim == 1 and len(sigmas) == nfe + 1, (
            f"{npy_path}: expected shape ({nfe+1},), got {sigmas.shape}"
        )
        assert sigmas[0] >= sigmas[-1], (
            f"{npy_path}: sigma array must be descending"
        )

        schedules[f"born_nfe{nfe}"] = sigmas
        print(f"  born_nfe{nfe}  : {nfe} steps, "
              f"σ ∈ [{sigmas[-1]:.4f}, {sigmas[0]:.4f}]")

        # Matching uniform schedule over the same range
        sigma_max, sigma_min = float(sigmas[0]), float(sigmas[-1])
        uni = np.linspace(sigma_max, sigma_min, nfe + 1, dtype=np.float32)
        schedules[f"uniform_nfe{nfe}"] = uni
        print(f"  uniform_nfe{nfe}: {nfe} steps, "
              f"σ ∈ [{uni[-1]:.4f}, {uni[0]:.4f}]  (derived)")

    return schedules


# ─────────────────────────────────────────────────────────────────────────────
# Locked scheduler
# ─────────────────────────────────────────────────────────────────────────────

def make_locked_scheduler(base_scheduler, custom_sigmas_np: np.ndarray):
    """
    Deep-copies base_scheduler and monkey-patches set_timesteps so that
    no matter what the pipeline passes internally, our custom sigmas are used.

    custom_sigmas_np : descending float32 array of length NFE+1,
                       WITHOUT a trailing 0.
    """
    sched = copy.deepcopy(base_scheduler)

    # FlowMatchEulerDiscreteScheduler expects sigmas to end with 0
    sigmas_t = torch.tensor(custom_sigmas_np, dtype=torch.float32)
    if sigmas_t[-1].item() != 0.0:
        sigmas_t = torch.cat([sigmas_t, sigmas_t.new_zeros(1)])

    # Flux convention: timesteps = sigmas[:-1] * 1000
    timesteps_t = (sigmas_t[:-1] * 1000.0).to(torch.float32)
    n = len(timesteps_t)

    def _locked_set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        **kwargs,
    ):
        dev = device if device is not None else "cpu"
        self.sigmas              = sigmas_t.to(dev)
        self.timesteps           = timesteps_t.to(dev)
        self.num_inference_steps = n
        self._step_index         = None
        self._begin_index        = None

    sched.set_timesteps = types.MethodType(_locked_set_timesteps, sched)
    return sched


# ─────────────────────────────────────────────────────────────────────────────
# Progress checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_progress(path: Path, prog: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=2)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def free_gpu(*objs):
    for o in objs:
        try:
            if hasattr(o, "cpu"):
                o.cpu()
        except Exception:
            pass
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_mb  = torch.cuda.mem_get_info()[0] / 1024**2
        total_mb = torch.cuda.mem_get_info()[1] / 1024**2
        print(f"  [GPU] {free_mb:.0f}/{total_mb:.0f} MiB free")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_t2i_compbench() -> dict[str, list[dict]]:
    """
    Returns {config_name: [{"prompt": str, "idx": int}, …]} for all
    configs in sayakpaul/t2i-compbench (val split).
    """
    from datasets import get_dataset_config_names, load_dataset

    dataset_name = "sayakpaul/t2i-compbench"
    configs = get_dataset_config_names(dataset_name)

    print(f"\n[dataset] Loading {dataset_name}")
    all_prompts: dict[str, list[dict]] = {}
    for cfg in configs:
        ds = load_dataset(dataset_name, cfg, split="val")
        # HF datasets may use "text" or "prompt" column
        text_col = "text" if "text" in ds.column_names else "prompt"
        prompts  = [
            {"prompt": row[text_col], "idx": i}
            for i, row in enumerate(ds)
        ]
        all_prompts[cfg] = prompts
        print(f"  {cfg}: {len(prompts)} prompts")

    return all_prompts


# ─────────────────────────────────────────────────────────────────────────────
# Single-run generation  (one schedule tag × one dataset config)
# ─────────────────────────────────────────────────────────────────────────────

def generate_one_run(
    *,
    pipe,
    orig_scheduler,
    tag: str,
    sigmas: np.ndarray,
    prompts: list[dict],
    seeds: list[int],
    img_dir: Path,
    ckpt_path: Path,
    height: int,
    width: int,
    guidance_scale: float,
):
    """
    Generate all images for one (tag, cfg) combination.
    Skips already-completed (idx, seed) pairs via progress checkpoint.
    """
    nfe      = len(sigmas) - 1
    progress = load_progress(ckpt_path)

    todo = [
        (item["idx"], seed)
        for item in prompts
        for seed in seeds
        if f"{item['idx']}/{seed}" not in progress
    ]

    if not todo:
        print(f"    [skip] {tag} – all {len(prompts)*len(seeds)} images exist")
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"    {len(todo)} images to generate …")

    locked = make_locked_scheduler(orig_scheduler, sigmas)
    pipe.scheduler = locked

    prompt_by_idx = {item["idx"]: item["prompt"] for item in prompts}

    with tqdm(total=len(todo), desc=f"  {tag}", ncols=90, leave=False) as pbar:
        for idx, seed in todo:
            img_path = img_dir / f"p{idx:04d}_s{seed}.png"
            gen      = torch.Generator(device="cpu").manual_seed(seed)

            t0  = time.perf_counter()
            out = pipe(
                prompt=prompt_by_idx[idx],
                num_inference_steps=nfe,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=gen,
                output_type="pil",
            )
            latency = time.perf_counter() - t0

            out.images[0].save(str(img_path))
            progress[f"{idx}/{seed}"] = {
                "path": str(img_path),
                "latency_s": round(latency, 3),
            }
            save_progress(ckpt_path, progress)

            del out, gen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def phase_generate(
    *,
    model_name: str,
    schedules: dict[str, np.ndarray],
    all_prompts: dict[str, list[dict]],
    seeds: list[int],
    out_dir: Path,
    height: int,
    width: int,
    guidance_scale: float,
    only_tags: list[str] | None = None,
):
    """
    Outer loop: for each (tag, cfg) pair, call generate_one_run.

    only_tags: if given, restrict to these schedule tags
               (e.g. ["born_nfe10"] to regenerate only one NFE).
    """
    tags = [t for t in schedules if (only_tags is None or t in only_tags)]
    tags = sorted(tags)   # deterministic order

    total_runs = len(tags) * len(all_prompts)
    print(f"\n[generate] {len(tags)} schedules × "
          f"{len(all_prompts)} configs = {total_runs} runs")

    # ── Load model once ──────────────────────────────────────────────────────
    from diffusers import FluxPipeline

    print(f"\n[generate] Loading pipeline: {model_name} …")
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    orig_scheduler = copy.deepcopy(pipe.scheduler)
    print("[generate] Pipeline loaded.\n")

    try:
        run_idx = 0
        for tag in tags:
            sigmas = schedules[tag]
            nfe    = len(sigmas) - 1

            for cfg, prompts in all_prompts.items():
                run_idx += 1
                folder   = f"{tag}_{cfg}"          # e.g. born_nfe10_color
                img_dir  = out_dir / folder
                ckpt     = out_dir / f"progress_{folder}.json"

                print(f"[{run_idx}/{total_runs}] {folder}  "
                      f"({len(prompts)} prompts × {len(seeds)} seeds)")

                generate_one_run(
                    pipe=pipe,
                    orig_scheduler=orig_scheduler,
                    tag=tag,
                    sigmas=sigmas,
                    prompts=prompts,
                    seeds=seeds,
                    img_dir=img_dir,
                    ckpt_path=ckpt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                )

    finally:
        print("\n[generate] Unloading pipeline …")
        free_gpu(pipe)
        del pipe

    print("\n[generate] ✓ All done.")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation stubs  (TODO)
# ─────────────────────────────────────────────────────────────────────────────

def phase_evaluate(
    *,
    out_dir: Path,
    all_prompts: dict[str, list[dict]],
    schedules: dict[str, np.ndarray],
    seeds: list[int],
):
    """
    TODO: run T2I-CompBench evaluators (BLIP-VQA, UniDet, CLIP, etc.)
          and aggregate scores per (tag, cfg).

    Expected output:
        out_dir/
            eval_results.json   ← {tag_cfg: {metric: score}}
            eval_summary.csv    ← human-readable table
    """
    raise NotImplementedError("Evaluation not yet implemented.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BornSchedule × T2I-CompBench generation pipeline"
    )
    p.add_argument(
        "--schedules_dir", type=Path, required=True,
        help="Directory containing born_nfe*.npy files"
    )
    p.add_argument(
        "--model_name", type=str, default="black-forest-labs/FLUX.1-dev",
        help="HuggingFace model id for the Flux pipeline"
    )
    p.add_argument(
        "--out_dir", type=Path, default=Path("./t2i_outputs"),
        help="Root directory for generated images and checkpoints"
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Random seeds (one image per prompt per seed)"
    )
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width",  type=int, default=512)
    p.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Classifier-free guidance scale for Flux"
    )
    p.add_argument(
        "--only_tags", type=str, nargs="*", default=None,
        help="Restrict generation to specific schedule tags "
             "(e.g. born_nfe10 uniform_nfe10). "
             "Default: all tags derived from schedules_dir."
    )
    p.add_argument(
        "--skip_generate", action="store_true",
        help="Skip generation phase (useful to re-run eval only)"
    )
    p.add_argument(
        "--run_eval", action="store_true",
        help="Run evaluation phase after generation (not yet implemented)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load schedules ────────────────────────────────────────────────────
    schedules = load_schedules(args.schedules_dir)

    # ── 2. Load dataset ──────────────────────────────────────────────────────
    all_prompts = load_t2i_compbench()

    # ── 3. Generate ──────────────────────────────────────────────────────────
    if not args.skip_generate:
        phase_generate(
            model_name=args.model_name,
            schedules=schedules,
            all_prompts=all_prompts,
            seeds=args.seeds,
            out_dir=args.out_dir,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            only_tags=args.only_tags,
        )
    else:
        print("[main] --skip_generate set, skipping generation.")

    # ── 4. Evaluate (TODO) ───────────────────────────────────────────────────
    if args.run_eval:
        phase_evaluate(
            out_dir=args.out_dir,
            all_prompts=all_prompts,
            schedules=schedules,
            seeds=args.seeds,
        )
    else:
        print("[main] Evaluation skipped (pass --run_eval to enable).")


if __name__ == "__main__":
    main()
