"""
trajectory_viz_flux.py
===========
Visual comparison of Born / Uniform / Hybrid schedules on Flux,
with per-step denoising trajectory saved as image strips.

Schedules generated per prompt:
  born_nfeN       – our schedule
  uniform_nfeN    – linearly spaced over same [σ_max, σ_min]
  hybrid_ub_nfeN  – Uniform (high-noise) → Born  (low-noise)  (switch at σ*)
  hybrid_bu_nfeN  – Born   (high-noise) → Uniform(low-noise)  (switch at σ*)

Output per (prompt, schedule):
  out_dir/{schedule_tag}/
      p{i:02d}_final.png        – final image
      p{i:02d}_steps.png        – horizontal strip: step 0 … step N-1 → final
      p{i:02d}_schedule.png     – σ trajectory plot (matplotlib)

Usage:
  python trajectory_viz_flux.py \\
      --schedules_dir /media/ssd_horse/keying/eval_results \\
      --model black-forest-labs/FLUX.1-dev \\
      --nfe 10 \\
      --switch_sigma 0.45 \\
      --out_dir ./viz_output \\
      --seed 42
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import types
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Representative prompts: one per T2I-CompBench category
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    # (category, label, prompt)
    ("SO",   "single-obj",          "a yellow banana on a white plate"),
    ("TO",   "two-obj",             "a tall building and a short tree"),
    ("CT",   "counting",            "three cars parked on the street"),
    ("CL",   "color-in-context",    "a pink cupcake on a white plate"),
    ("ATTR", "multi-attribute",     "a large round watermelon"),
    ("PO",   "spatial relation",    "a car parked behind a building"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_born(schedules_dir: Path, nfe: int) -> np.ndarray:
    """Load born_nfe{nfe}.npy → descending float32 array of length nfe+1."""
    p = schedules_dir / f"born_nfe{nfe}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Expected {p}")
    s = np.load(str(p)).astype(np.float32)
    assert len(s) == nfe + 1, f"Expected {nfe+1} values, got {len(s)}"
    return s


def make_uniform(born: np.ndarray) -> np.ndarray:
    """Linearly spaced schedule matching [σ_max, σ_min] of born."""
    return np.linspace(born[0], born[-1], len(born), dtype=np.float32)


def make_hybrid(
    born: np.ndarray,
    uniform: np.ndarray,
    switch_sigma: float,
    mode: str,          # "ub" (uniform→born) or "bu" (born→uniform)
) -> np.ndarray:
    """
    Splice two schedules at the step where σ first drops below switch_sigma.

    switch_sigma is compared against the *from* sigma of each step, i.e.
    sigmas[i] (the noise level at the START of step i).

    mode "ub": high-noise steps use uniform spacing, low-noise steps use born.
    mode "bu": high-noise steps use born spacing, low-noise steps use uniform.

    The splice is guaranteed monotonically decreasing because both arrays
    share the same endpoint values and we cut at the same index.
    """
    nfe = len(born) - 1
    # Find first index i where born[i] <= switch_sigma (step i starts below σ*)
    switch_idx = nfe   # fallback: never switch
    for i, s in enumerate(born):
        if s <= switch_sigma:
            switch_idx = i
            break

    if mode == "ub":
        hyb = np.concatenate([uniform[:switch_idx], born[switch_idx:]])
    else:
        hyb = np.concatenate([born[:switch_idx], uniform[switch_idx:]])

    return hyb.astype(np.float32)


def build_all_schedules(
    schedules_dir: Path,
    nfe: int,
    switch_sigma: float,
) -> dict[str, np.ndarray]:
    born    = load_born(schedules_dir, nfe)
    uniform = make_uniform(born)
    return {
        f"born_nfe{nfe}":       born,
        f"uniform_nfe{nfe}":    uniform,
        f"hybrid_ub_nfe{nfe}":  make_hybrid(born, uniform, switch_sigma, "ub"),
        f"hybrid_bu_nfe{nfe}":  make_hybrid(born, uniform, switch_sigma, "bu"),
    }


def find_switch_step(sigmas: np.ndarray, switch_sigma: float) -> int | None:
    """Return step index where σ first drops below switch_sigma, or None."""
    for i, s in enumerate(sigmas[:-1]):    # sigmas[:-1] are step start values
        if s <= switch_sigma:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Locked scheduler
# ─────────────────────────────────────────────────────────────────────────────

def make_locked_scheduler(base_scheduler, sigmas_np: np.ndarray):
    """
    Deep-copy base_scheduler and monkey-patch set_timesteps so the pipeline
    can never overwrite our sigmas regardless of what args it passes.
    """
    sched = copy.deepcopy(base_scheduler)

    sigmas_t = torch.tensor(sigmas_np, dtype=torch.float32)
    if sigmas_t[-1].item() != 0.0:
        sigmas_t = torch.cat([sigmas_t, sigmas_t.new_zeros(1)])

    timesteps_t = (sigmas_t[:-1] * 1000.0).to(torch.float32)
    n = len(timesteps_t)

    def _locked(self, num_inference_steps=None, device=None,
                 sigmas=None, mu=None, **kwargs):
        dev = device or "cpu"
        self.sigmas              = sigmas_t.to(dev)
        self.timesteps           = timesteps_t.to(dev)
        self.num_inference_steps = n
        self._step_index         = None
        self._begin_index        = None

    sched.set_timesteps = types.MethodType(_locked, sched)
    return sched


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate-step decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode_latent(pipe, latent: torch.Tensor, height: int, width: int) -> Image.Image:
    """
    Decode a single Flux latent tensor (on CPU) to a PIL image.
    Flux stores latents as [1, C, h//8, w//8] after vae-encode.
    """
    lat = latent.to(dtype=pipe.vae.dtype, device=pipe.vae.device)
    # Flux packs latents; we must unpack before decoding
    # pipe._unpack_latents is available in diffusers >= 0.28
    if hasattr(pipe, "_unpack_latents"):
        lat_up = pipe._unpack_latents(lat, height, width, pipe.vae_scale_factor)
    else:
        lat_up = lat

    lat_up = (lat_up / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.no_grad():
        img_tensor = pipe.vae.decode(lat_up).sample
    img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img_np * 255).round().astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Image strip / annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_img(img: Image.Image, text: str, font_size: int = 16) -> Image.Image:
    """Add a text label below an image."""
    pad   = font_size + 6
    new   = Image.new("RGB", (img.width, img.height + pad), (30, 30, 30))
    new.paste(img, (0, 0))
    draw  = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                   font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, img.height + 2), text, fill=(220, 220, 220), font=font)
    return new


def make_step_strip(
    step_images: list[Image.Image],
    thumb_w: int = 128,
) -> Image.Image:
    """
    Horizontal strip of per-step thumbnails.
    Each thumbnail is labelled "step i  σ=x.xx".
    """
    thumbs = [img.resize(
        (thumb_w, int(img.height * thumb_w / img.width)),
        Image.LANCZOS,
    ) for img in step_images]
    h    = max(t.height for t in thumbs)
    strip = Image.new("RGB", (len(thumbs) * thumb_w + (len(thumbs)-1)*4, h),
                      (20, 20, 20))
    x = 0
    for t in thumbs:
        strip.paste(t, (x, 0))
        x += thumb_w + 4
    return strip


def make_schedule_plot(
    schedules: dict[str, np.ndarray],
    switch_sigma: float,
    save_path: Path,
):
    """σ-vs-step plot for all four schedules."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available, skipping schedule plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    styles = {
        "born":       dict(color="#E74C3C", lw=2.2, ls="-",  marker="o", ms=5),
        "uniform":    dict(color="#3498DB", lw=2.2, ls="--", marker="s", ms=5),
        "hybrid_ub":  dict(color="#27AE60", lw=2.0, ls="-.", marker="^", ms=5),
        "hybrid_bu":  dict(color="#F39C12", lw=2.0, ls=":",  marker="D", ms=5),
    }
    for tag, sigmas in schedules.items():
        key = tag.split("_nfe")[0].replace("born", "born").replace(
              "uniform", "uniform").replace("hybrid_ub", "hybrid_ub").replace(
              "hybrid_bu", "hybrid_bu")
        style = styles.get(key, {})
        steps = np.arange(len(sigmas))
        ax.plot(steps, sigmas, label=tag, **style)

    ax.axhline(switch_sigma, color="gray", lw=1.2, ls="--",
               label=f"switch σ* = {switch_sigma:.2f}")
    ax.set_xlabel("Step index", fontsize=11)
    ax.set_ylabel("σ (noise level)", fontsize=11)
    ax.set_title("Schedule comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"  [plot] Schedule plot → {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Single image + trajectory generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_trajectory(
    pipe,
    orig_scheduler,
    prompt: str,
    sigmas: np.ndarray,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
) -> tuple[Image.Image, list[Image.Image]]:
    """
    Run one denoising pass.  Returns (final_image, [step_0_img, step_1_img, …]).

    Per-step images are decoded by hooking into callback_on_step_end, which
    fires after each denoiser call with the *current latent estimate* in
    callback_kwargs["latents"].  These are decoded with the VAE so they show
    the model's best guess at each noise level.
    """
    nfe     = len(sigmas) - 1
    pipe.scheduler = make_locked_scheduler(orig_scheduler, sigmas)
    gen     = torch.Generator(device="cpu").manual_seed(seed)

    # Collect latents at each step
    step_latents: list[torch.Tensor] = []

    def _step_callback(p, step_idx, timestep, cb_kwargs):
        step_latents.append(cb_kwargs["latents"].clone().cpu())
        return cb_kwargs

    out = pipe(
        prompt=prompt,
        num_inference_steps=nfe,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen,
        output_type="pil",
        callback_on_step_end=_step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    final_img = out.images[0]

    # Decode all captured latents
    step_images: list[Image.Image] = []
    for lat in step_latents:
        try:
            step_images.append(decode_latent(pipe, lat, height, width))
        except Exception as e:
            print(f"    [warn] latent decode failed at one step: {e}")
            # Fallback: gray placeholder
            step_images.append(Image.new("RGB", (height, width), (128, 128, 128)))

    return final_img, step_images


# ─────────────────────────────────────────────────────────────────────────────
# Annotated step strip with σ labels
# ─────────────────────────────────────────────────────────────────────────────

def build_annotated_strip(
    step_images: list[Image.Image],
    sigmas: np.ndarray,
    final_image: Image.Image,
    switch_sigma: float,
    thumb_w: int = 128,
    font_size: int = 13,
) -> Image.Image:
    """
    Horizontal strip: [step0 | step1 | … | step_N-1 | ➜ | final]
    Each thumbnail labeled "step i / σ=x.xx".
    Switch boundary marked with a colored border (green=uniform region,
    red=born region for hybrid; both same color for pure schedules).
    """
    all_imgs  = step_images + [final_image]
    all_sigma = list(sigmas[:-1]) + [sigmas[-1]]   # start-of-step σ values
    all_labels = [f"step {i}\nσ={s:.3f}" for i, s in enumerate(sigmas[:-1])]
    all_labels += [f"final\nσ={sigmas[-1]:.3f}"]

    pad_label = font_size * 2 + 8
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    thumb_h = int(all_imgs[0].height * thumb_w / all_imgs[0].width)
    tile_h  = thumb_h + pad_label

    # separator width between step tiles
    sep_w   = 6
    n       = len(all_imgs)
    total_w = n * thumb_w + (n - 1) * sep_w + sep_w * 2  # +2 for final separator

    canvas = Image.new("RGB", (total_w, tile_h), (22, 22, 22))
    draw   = ImageDraw.Draw(canvas)

    x = 0
    for i, (img, lbl, s_val) in enumerate(zip(all_imgs, all_labels, all_sigma)):
        is_final = (i == len(all_imgs) - 1)

        # Border color: mark switch region
        if s_val <= switch_sigma:
            border_col = (231, 76, 60)   # red → born territory
        else:
            border_col = (52, 152, 219)  # blue → uniform territory
        if is_final:
            border_col = (255, 255, 255)

        t = img.resize((thumb_w, thumb_h), Image.LANCZOS)

        # Draw border
        bordered = Image.new("RGB", (thumb_w + 4, thumb_h + 4), border_col)
        bordered.paste(t, (2, 2))

        canvas.paste(bordered, (x, 0))
        draw.text((x + 2, thumb_h + 6), lbl, fill=(210, 210, 210), font=font)

        x += thumb_w + 4 + sep_w

    # Legend bar
    draw.rectangle([0, tile_h - 4, total_w, tile_h], fill=(50, 50, 50))

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Born/Uniform/Hybrid schedule visual comparison with denoising trajectories",
    )
    pa.add_argument("--schedules_dir", required=True,
                    help="Dir containing born_nfe*.npy files")
    pa.add_argument("--model",  default="black-forest-labs/FLUX.1-dev")
    pa.add_argument("--nfe",    type=int, default=10,
                    help="Number of function evaluations")
    pa.add_argument("--switch_sigma", type=float, default=0.45,
                    help="σ threshold at which hybrid schedules switch strategies")
    pa.add_argument("--out_dir", default="./viz_output")
    pa.add_argument("--seed",   type=int, default=42)
    pa.add_argument("--height", type=int, default=512)
    pa.add_argument("--width",  type=int, default=512)
    pa.add_argument("--guidance_scale", type=float, default=3.5)
    pa.add_argument("--thumb_w", type=int, default=128,
                    help="Thumbnail width in step strips (px)")
    pa.add_argument("--skip_trajectory", action="store_true",
                    help="Skip per-step decoding (faster; only saves final image)")
    args = pa.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sched_dir = Path(args.schedules_dir)

    # ── 1. Build schedules ───────────────────────────────────────────────────
    print(f"\n[schedules] NFE={args.nfe}, switch σ*={args.switch_sigma}")
    schedules = build_all_schedules(sched_dir, args.nfe, args.switch_sigma)

    for tag, sigs in schedules.items():
        sw = find_switch_step(sigs, args.switch_sigma)
        sw_str = f" (switch @ step {sw})" if sw is not None else " (no switch)"
        print(f"  {tag:<26} σ ∈ [{sigs[-1]:.4f}, {sigs[0]:.4f}]{sw_str}")

    # Schedule comparison plot (shared across prompts)
    plot_path = out_dir / f"schedules_nfe{args.nfe}.png"
    make_schedule_plot(schedules, args.switch_sigma, plot_path)

    # ── 2. Load pipeline ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Loading {args.model} …")
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    orig_scheduler = copy.deepcopy(pipe.scheduler)
    print("[pipeline] Loaded.\n")

    # ── 3. Generate ──────────────────────────────────────────────────────────
    n_total = len(PROMPTS) * len(schedules)
    run_idx = 0

    for cat, lbl, prompt in PROMPTS:
        prompt_slug = cat.lower() + "_" + lbl.replace(" ", "_").replace("-", "_")
        print(f"\n{'─'*60}")
        print(f"[prompt] [{cat}] {prompt}")

        for tag, sigmas in schedules.items():
            run_idx += 1
            tag_dir = out_dir / tag
            tag_dir.mkdir(exist_ok=True)

            final_path = tag_dir / f"{prompt_slug}_final.png"
            strip_path = tag_dir / f"{prompt_slug}_steps.png"

            print(f"  [{run_idx}/{n_total}] {tag} ", end="", flush=True)

            if final_path.exists() and (args.skip_trajectory or strip_path.exists()):
                print("(cached, skipping)")
                continue

            if args.skip_trajectory:
                # Fast path: just generate final image
                pipe.scheduler = make_locked_scheduler(orig_scheduler, sigmas)
                gen = torch.Generator(device="cpu").manual_seed(args.seed)
                out = pipe(
                    prompt=prompt,
                    num_inference_steps=len(sigmas) - 1,
                    guidance_scale=args.guidance_scale,
                    height=args.height, width=args.width,
                    generator=gen,
                    output_type="pil",
                )
                final_img = out.images[0]
                step_images = []
                del out, gen
            else:
                final_img, step_images = generate_with_trajectory(
                    pipe=pipe,
                    orig_scheduler=orig_scheduler,
                    prompt=prompt,
                    sigmas=sigmas,
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                )

            # Save final image
            final_img.save(str(final_path))

            # Save step strip
            if step_images:
                strip = build_annotated_strip(
                    step_images=step_images,
                    sigmas=sigmas,
                    final_image=final_img,
                    switch_sigma=args.switch_sigma,
                    thumb_w=args.thumb_w,
                )
                strip.save(str(strip_path))
                print(f"→ {len(step_images)} steps saved")
            else:
                print("→ (no trajectory)")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── 4. Summary grid ──────────────────────────────────────────────────────
    print("\n[summary] Building overview grid …")
    _build_summary_grid(out_dir, schedules, args.nfe)

    print(f"\nDone. Results in {out_dir}/")
    print(f"  schedules_nfe{args.nfe}.png   ← σ trajectory comparison")
    print(f"  summary_nfe{args.nfe}.png     ← final images grid (prompts × schedules)")
    for tag in schedules:
        print(f"  {tag}/  ← final + step-strip PNGs")


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid: rows = prompts, columns = schedules
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary_grid(out_dir: Path, schedules: dict, nfe: int):
    """
    Grid image: rows = prompts, columns = schedules.
    Reads already-saved final PNGs; skips missing ones gracefully.
    """
    cell_w, cell_h = 256, 256
    label_h = 22
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    tags    = list(schedules.keys())
    n_cols  = len(tags)
    n_rows  = len(PROMPTS)

    col_label_w = 24   # narrow left gutter (rotated text not supported easily)

    grid_w  = col_label_w + n_cols * (cell_w + 2)
    grid_h  = label_h + n_rows * (cell_h + label_h + 2)
    grid    = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))
    draw    = ImageDraw.Draw(grid)

    # Column headers (schedule tags)
    for j, tag in enumerate(tags):
        x = col_label_w + j * (cell_w + 2) + 4
        draw.text((x, 2), tag.replace(f"_nfe{nfe}", ""), fill=(200, 200, 200), font=font)

    for i, (cat, lbl, _prompt) in enumerate(PROMPTS):
        slug = cat.lower() + "_" + lbl.replace(" ", "_").replace("-", "_")
        row_y = label_h + i * (cell_h + label_h + 2)

        # Row label
        draw.text((2, row_y + cell_h // 2), cat, fill=(150, 200, 150), font=font)

        for j, tag in enumerate(tags):
            img_path = out_dir / tag / f"{slug}_final.png"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB").resize(
                    (cell_w, cell_h), Image.LANCZOS)
            else:
                img = Image.new("RGB", (cell_w, cell_h), (60, 30, 30))
                d = ImageDraw.Draw(img)
                d.text((10, cell_h // 2 - 8), "missing", fill=(200, 100, 100), font=font)

            x = col_label_w + j * (cell_w + 2)
            grid.paste(img, (x, row_y))

            # Prompt caption below cell
            draw.text(
                (x + 2, row_y + cell_h + 2),
                lbl[:30],
                fill=(160, 160, 160),
                font=font,
            )

    save_path = out_dir / f"summary_nfe{nfe}.png"
    grid.save(str(save_path))
    print(f"  → {save_path.name}  ({grid_w}×{grid_h})")


if __name__ == "__main__":
    main()