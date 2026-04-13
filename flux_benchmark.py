#!/usr/bin/env python3
"""
eval_flux_born.py
=================
Evaluate BornSchedule-optimised Flux sigma-schedules with:
  • T2I-CompBench metrics  SO / TO / CT / CL / ATTR / PO / Overall
    (BLIP-VQA + UniDet object-detection based, via local eval helpers)
  • CLIPIQA  (pyiqa no-reference IQA)
  • Speedup & Latency  (wall-clock, relative to 50-step uniform baseline)

Tested NFE settings: 50 (baseline), 25, 10.

SSH-safe design
---------------
  • Every generated image is written to disk immediately under out_dir/
  • A per-run JSON checkpoint  <out_dir>/progress.json  tracks which
    (schedule_tag, prompt_idx) pairs are already done.
  • On restart the script skips completed pairs and continues.

Quick start
-----------
  # one-time setup
  pip install diffusers transformers accelerate pyiqa open-clip-torch \
              pillow tqdm numpy torch torchvision

  # run (background, immune to SSH drop)
  nohup python eval_flux_born.py \
        --stats flux_stats.npz \
        --model "black-forest-labs/FLUX.1-schnell" \
        --nfe_list 50 25 10 \
        --n_prompts 200 \
        --out_dir eval_results \
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
#  T2I-CompBench prompt sets (representative subset)
#  Full benchmark: https://github.com/Karine-Huang/T2I-CompBench
#  Format: {"prompt": str, "category": str, "attribute": str}
# ─────────────────────────────────────────────────────────────────────────────

T2I_PROMPTS = [
    # ── Single-Object (SO) ────────────────────────────────────────────────
    {"prompt": "a red apple on a wooden table",          "cat": "SO", "attr": "color"},
    {"prompt": "a small blue bird on a branch",          "cat": "SO", "attr": "color"},
    {"prompt": "a large green frog on a lily pad",       "cat": "SO", "attr": "size"},
    {"prompt": "a tiny orange kitten in a basket",       "cat": "SO", "attr": "size"},
    {"prompt": "a yellow banana on a white plate",       "cat": "SO", "attr": "color"},
    {"prompt": "a purple butterfly on a flower",         "cat": "SO", "attr": "color"},
    {"prompt": "a brown bear in the forest",             "cat": "SO", "attr": "color"},
    {"prompt": "a shiny silver car on the road",         "cat": "SO", "attr": "texture"},
    {"prompt": "a fluffy white cat sleeping",            "cat": "SO", "attr": "texture"},
    {"prompt": "a tall pine tree in the snow",           "cat": "SO", "attr": "size"},
    {"prompt": "a pink flamingo standing in water",      "cat": "SO", "attr": "color"},
    {"prompt": "a striped zebra on the savanna",         "cat": "SO", "attr": "texture"},
    {"prompt": "a giant panda eating bamboo",            "cat": "SO", "attr": "size"},
    {"prompt": "a golden retriever playing fetch",       "cat": "SO", "attr": "color"},
    {"prompt": "a transparent glass vase with flowers",  "cat": "SO", "attr": "texture"},
    # ── Two-Object (TO) ───────────────────────────────────────────────────
    {"prompt": "a red car and a blue bicycle",           "cat": "TO", "attr": "color"},
    {"prompt": "a large dog and a small cat",            "cat": "TO", "attr": "size"},
    {"prompt": "a wooden chair and a metal table",       "cat": "TO", "attr": "texture"},
    {"prompt": "a yellow sun and white clouds",          "cat": "TO", "attr": "color"},
    {"prompt": "a black horse and a white fence",        "cat": "TO", "attr": "color"},
    {"prompt": "a big elephant and a tiny bird",         "cat": "TO", "attr": "size"},
    {"prompt": "a glass bottle and a ceramic mug",       "cat": "TO", "attr": "texture"},
    {"prompt": "a red rose and a blue vase",             "cat": "TO", "attr": "color"},
    {"prompt": "a tall building and a short tree",       "cat": "TO", "attr": "size"},
    {"prompt": "a rough stone wall and a smooth floor",  "cat": "TO", "attr": "texture"},
    {"prompt": "a green apple and an orange",            "cat": "TO", "attr": "color"},
    {"prompt": "a large pizza and a small salad",        "cat": "TO", "attr": "size"},
    {"prompt": "a fluffy pillow and a wooden chair",     "cat": "TO", "attr": "texture"},
    {"prompt": "a brown dog and a gray cat",             "cat": "TO", "attr": "color"},
    {"prompt": "a huge mountain and a tiny cabin",       "cat": "TO", "attr": "size"},
    # ── Count (CT) ────────────────────────────────────────────────────────
    {"prompt": "three red apples on a table",            "cat": "CT", "attr": "count"},
    {"prompt": "two dogs playing in the park",           "cat": "CT", "attr": "count"},
    {"prompt": "five colorful balloons in the sky",      "cat": "CT", "attr": "count"},
    {"prompt": "four cats sitting on a couch",           "cat": "CT", "attr": "count"},
    {"prompt": "six birds on a power line",              "cat": "CT", "attr": "count"},
    {"prompt": "two children running in a field",        "cat": "CT", "attr": "count"},
    {"prompt": "three cars parked on the street",        "cat": "CT", "attr": "count"},
    {"prompt": "four flowers in a vase",                 "cat": "CT", "attr": "count"},
    {"prompt": "two bicycles leaning against a wall",    "cat": "CT", "attr": "count"},
    {"prompt": "five fish in a fish tank",               "cat": "CT", "attr": "count"},
    {"prompt": "three books on a shelf",                 "cat": "CT", "attr": "count"},
    {"prompt": "two planes in the sky",                  "cat": "CT", "attr": "count"},
    {"prompt": "four candles on a birthday cake",        "cat": "CT", "attr": "count"},
    {"prompt": "six stars in the night sky",             "cat": "CT", "attr": "count"},
    {"prompt": "three horses in a meadow",               "cat": "CT", "attr": "count"},
    # ── Color (CL) ────────────────────────────────────────────────────────
    {"prompt": "a blue dress on a mannequin",            "cat": "CL", "attr": "color"},
    {"prompt": "a red umbrella in the rain",             "cat": "CL", "attr": "color"},
    {"prompt": "a green parrot on a perch",              "cat": "CL", "attr": "color"},
    {"prompt": "a yellow school bus on the road",        "cat": "CL", "attr": "color"},
    {"prompt": "a pink cupcake on a white plate",        "cat": "CL", "attr": "color"},
    {"prompt": "a black cat on a red sofa",              "cat": "CL", "attr": "color"},
    {"prompt": "a white swan on a blue lake",            "cat": "CL", "attr": "color"},
    {"prompt": "an orange tiger in the jungle",          "cat": "CL", "attr": "color"},
    {"prompt": "a purple grape on a wooden table",       "cat": "CL", "attr": "color"},
    {"prompt": "a gray elephant in the savanna",         "cat": "CL", "attr": "color"},
    {"prompt": "a brown wooden cabin in the snow",       "cat": "CL", "attr": "color"},
    {"prompt": "a silver robot in a factory",            "cat": "CL", "attr": "color"},
    {"prompt": "a turquoise ocean with white waves",     "cat": "CL", "attr": "color"},
    {"prompt": "a gold crown on a velvet pillow",        "cat": "CL", "attr": "color"},
    {"prompt": "a crimson sunset over the mountains",    "cat": "CL", "attr": "color"},
    # ── Attribute Binding (ATTR) ──────────────────────────────────────────
    {"prompt": "a shiny red sports car",                 "cat": "ATTR", "attr": "color-material"},
    {"prompt": "a fluffy white polar bear",              "cat": "ATTR", "attr": "texture-color"},
    {"prompt": "a large round watermelon",               "cat": "ATTR", "attr": "size-shape"},
    {"prompt": "a small square brown box",               "cat": "ATTR", "attr": "size-shape"},
    {"prompt": "a tall silver metallic skyscraper",      "cat": "ATTR", "attr": "size-material"},
    {"prompt": "a smooth black marble floor",            "cat": "ATTR", "attr": "texture-color"},
    {"prompt": "a tiny delicate pink flower",            "cat": "ATTR", "attr": "size-color"},
    {"prompt": "a rough brown stone wall",               "cat": "ATTR", "attr": "texture-color"},
    {"prompt": "a bright yellow rubber duck",            "cat": "ATTR", "attr": "color-material"},
    {"prompt": "a heavy dark iron gate",                 "cat": "ATTR", "attr": "texture-color"},
    {"prompt": "a transparent thin glass cup",           "cat": "ATTR", "attr": "texture-size"},
    {"prompt": "a thick green velvet curtain",           "cat": "ATTR", "attr": "texture-color"},
    {"prompt": "a warm golden wooden floor",             "cat": "ATTR", "attr": "color-material"},
    {"prompt": "a soft pink cotton blanket",             "cat": "ATTR", "attr": "texture-color"},
    {"prompt": "a cold grey concrete building",          "cat": "ATTR", "attr": "color-material"},
    # ── Positional / Spatial (PO) ─────────────────────────────────────────
    {"prompt": "a cat sitting on top of a table",        "cat": "PO", "attr": "spatial"},
    {"prompt": "a bird flying above the clouds",         "cat": "PO", "attr": "spatial"},
    {"prompt": "a dog running beside a bicycle",         "cat": "PO", "attr": "spatial"},
    {"prompt": "a lamp next to a bookshelf",             "cat": "PO", "attr": "spatial"},
    {"prompt": "a child standing in front of a door",   "cat": "PO", "attr": "spatial"},
    {"prompt": "a boat under a bridge",                  "cat": "PO", "attr": "spatial"},
    {"prompt": "a flower pot on a window ledge",         "cat": "PO", "attr": "spatial"},
    {"prompt": "a car parked behind a building",         "cat": "PO", "attr": "spatial"},
    {"prompt": "a flag on top of a mountain",            "cat": "PO", "attr": "spatial"},
    {"prompt": "a fish swimming below the surface",      "cat": "PO", "attr": "spatial"},
    {"prompt": "a hat hanging above a coat rack",        "cat": "PO", "attr": "spatial"},
    {"prompt": "a shadow beneath a tree",                "cat": "PO", "attr": "spatial"},
    {"prompt": "a kite above a green hill",              "cat": "PO", "attr": "spatial"},
    {"prompt": "an umbrella beside a park bench",        "cat": "PO", "attr": "spatial"},
    {"prompt": "a mirror opposite a window",             "cat": "PO", "attr": "spatial"},
]

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
#  Flux pipeline wrapper with custom σ injection
# ─────────────────────────────────────────────────────────────────────────────

class FluxCustomSchedule:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        from diffusers import FluxPipeline
        print(f"  Loading Flux pipeline: {model_name} …")
        self.pipe = FluxPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.dtype  = dtype
        print("  Pipeline loaded.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        sigmas: np.ndarray,
        seed: int = 42,
        height: int = 512,
        width:  int = 512,
        guidance_scale: float = 3.5,
    ) -> tuple[Image.Image, float]:
        """
        Returns (PIL image, latency_seconds).
        Injects custom sigmas into FlowMatchEulerDiscreteScheduler.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler
        import copy

        nfe = len(sigmas) - 1

        # Clone the scheduler and inject custom sigmas (timesteps in σ-space)
        sched = copy.deepcopy(self.pipe.scheduler)
        ts = torch.tensor(sigmas, dtype=torch.float32)
        sched.timesteps = ts
        sched.sigmas    = ts

        gen = torch.Generator(device=self.device).manual_seed(seed)

        t0 = time.perf_counter()
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=nfe,
            guidance_scale=guidance_scale,
            height=height, width=width,
            generator=gen,
            scheduler=sched,     # type: ignore[arg-type]
            output_type="pil",
        )
        latency = time.perf_counter() - t0
        return out.images[0], latency


# ─────────────────────────────────────────────────────────────────────────────
#  CLIPIQA scorer
# ─────────────────────────────────────────────────────────────────────────────

class CLIPIQAScorer:
    def __init__(self, device: str = "cuda"):
        try:
            import pyiqa
        except ImportError:
            raise ImportError("pip install pyiqa")
        self.metric = pyiqa.create_metric("clipiqa", device=device)
        self.device = device

    def score(self, image: Image.Image) -> float:
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.metric(t))


# ─────────────────────────────────────────────────────────────────────────────
#  BLIP-VQA T2I-CompBench-style scorer
# ─────────────────────────────────────────────────────────────────────────────
#
#  For each prompt category we build a yes/no VQA question and ask BLIP.
#  P(yes) = alignment score for that sample.  Mean over category = category score.
#
#  This mirrors the approach in T2I-CompBench (Huang et al., NeurIPS 2023).
# ─────────────────────────────────────────────────────────────────────────────

class VQAScorer:
    def __init__(self, device: str = "cuda"):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        print("  Loading BLIP-VQA …")
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.device = device
        print("  BLIP-VQA loaded.")

    def _p_yes(self, image: Image.Image, question: str) -> float:
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0]           # (vocab_size,)
        yes_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        no_id  = self.processor.tokenizer.convert_tokens_to_ids("no")
        probs  = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)
        return float(probs[0])

    def score_item(self, image: Image.Image, entry: dict) -> dict:
        """
        Return per-category alignment scores ∈ [0,1] for one image+prompt.
        """
        cat    = entry["cat"]
        prompt = entry["prompt"]
        scores = {}

        if cat == "SO":
            # Check the main object and colour/size attribute
            q = f"Does the image show {prompt}? Answer yes or no."
            scores["SO"] = self._p_yes(image, q)

        elif cat == "TO":
            parts  = prompt.split(" and ", 1)
            q1 = f"Is there {parts[0].strip()} in the image? Answer yes or no."
            q2 = f"Is there {parts[1].strip()} in the image? Answer yes or no." if len(parts) > 1 else q1
            scores["TO"] = (self._p_yes(image, q1) + self._p_yes(image, q2)) / 2.0

        elif cat == "CT":
            q = f"Does the image show exactly what is described: {prompt}? Answer yes or no."
            scores["CT"] = self._p_yes(image, q)

        elif cat == "CL":
            # Extract colour word (simple heuristic)
            color_words = ["red","blue","green","yellow","purple","pink","brown",
                           "black","white","orange","gray","grey","silver","gold",
                           "turquoise","crimson","golden"]
            detected = [c for c in color_words if c in prompt.lower()]
            if detected:
                q = f"Is the main object {detected[0]}? Answer yes or no."
            else:
                q = f"Does the image match the description: {prompt}? Answer yes or no."
            scores["CL"] = self._p_yes(image, q)

        elif cat == "ATTR":
            q = f"Does the image correctly show {prompt}? Answer yes or no."
            scores["ATTR"] = self._p_yes(image, q)

        elif cat == "PO":
            q = f"Is the spatial arrangement correct: {prompt}? Answer yes or no."
            scores["PO"] = self._p_yes(image, q)

        return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Image generation loop  (SSH-safe)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    pipe_wrapper, prompts, schedules_dict,
    out_dir: Path, progress: dict, ckpt_path: Path,
    seeds: list, height: int, width: int,
):
    """
    schedules_dict: {tag: sigmas_array}
    progress:       {f"{tag}/{idx}/{seed}": {"path": str, "latency": float}}
    """
    for tag, sigmas in schedules_dict.items():
        tag_dir = out_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        for idx, entry in enumerate(prompts):
            for seed in seeds:
                key = f"{tag}/{idx}/{seed}"
                if key in progress:
                    continue   # already done — skip
                img, lat = pipe_wrapper.generate(
                    entry["prompt"], sigmas, seed=seed,
                    height=height, width=width)
                img_path = tag_dir / f"p{idx:04d}_s{seed}.png"
                img.save(str(img_path))
                progress[key] = {"path": str(img_path), "latency": lat}
                save_progress(ckpt_path, progress)

    print("  [gen] All images generated.")


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring loop
# ─────────────────────────────────────────────────────────────────────────────

def score_all(
    clipiqa: CLIPIQAScorer,
    vqa:     VQAScorer,
    prompts, schedules_dict: dict,
    progress: dict, ckpt_path: Path,
) -> dict:
    """
    Returns results dict:
      {tag: {"SO": [...], "TO": [...], "CT": [...], "CL": [...],
             "ATTR": [...], "PO": [...], "clipiqa": [...], "latency": [...]} }
    """
    results = {tag: {"SO":[],"TO":[],"CT":[],"CL":[],"ATTR":[],"PO":[],
                     "clipiqa":[],"latency":[]}
               for tag in schedules_dict}

    for tag in schedules_dict:
        print(f"\n  Scoring  [{tag}] …")
        for idx, entry in enumerate(tqdm(prompts, desc=tag, ncols=80)):
            for key, meta in progress.items():
                if not key.startswith(f"{tag}/{idx}/"):
                    continue
                img_path = Path(meta["path"])
                if not img_path.exists():
                    continue
                img = Image.open(img_path).convert("RGB")

                # CLIPIQA
                cq = clipiqa.score(img)
                results[tag]["clipiqa"].append(cq)
                results[tag]["latency"].append(meta["latency"])

                # VQA
                vqa_scores = vqa.score_item(img, entry)
                for cat, sc in vqa_scores.items():
                    results[tag][cat].append(sc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate & print table
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: dict, baseline_tag: str) -> dict:
    cats = ["SO","TO","CT","CL","ATTR","PO"]
    table = {}
    baseline_lat = np.mean(results[baseline_tag]["latency"]) if baseline_tag in results else 1.0

    for tag, r in results.items():
        row = {}
        for c in cats:
            vals = r.get(c, [])
            row[c] = float(np.mean(vals)) if vals else float("nan")
        row["Overall"]  = float(np.nanmean([row[c] for c in cats]))
        row["CLIPIQA"]  = float(np.mean(r["clipiqa"])) if r["clipiqa"] else float("nan")
        row["Lat(s)"]   = float(np.mean(r["latency"])) if r["latency"] else float("nan")
        row["Spd"]      = baseline_lat / row["Lat(s)"] if row["Lat(s)"] > 0 else float("nan")
        table[tag] = row
    return table


def print_table(table: dict):
    cols = ["SO","TO","CT","CL","ATTR","PO","Overall","CLIPIQA","Spd↑","Lat(s)↓"]
    header = f"{'Schedule':<28}" + "".join(f"{c:>10}" for c in cols)
    print("\n" + "─"*len(header))
    print(header)
    print("─"*len(header))
    for tag, row in table.items():
        cells = [
            row.get("SO",float("nan")), row.get("TO",float("nan")),
            row.get("CT",float("nan")), row.get("CL",float("nan")),
            row.get("ATTR",float("nan")), row.get("PO",float("nan")),
            row.get("Overall",float("nan")), row.get("CLIPIQA",float("nan")),
            row.get("Spd",float("nan")), row.get("Lat(s)",float("nan")),
        ]
        line = f"{tag:<28}" + "".join(
            f"{(v if not np.isnan(v) else 0.0):>10.4f}" for v in cells)
        print(line)
    print("─"*len(header) + "\n")


def save_table_csv(table: dict, path: Path):
    cols = ["SO","TO","CT","CL","ATTR","PO","Overall","CLIPIQA","Spd","Lat(s)"]
    with open(path, "w") as f:
        f.write("Schedule," + ",".join(cols) + "\n")
        for tag, row in table.items():
            vals = [str(round(row.get(c, float("nan")), 6)) for c in cols]
            f.write(tag + "," + ",".join(vals) + "\n")
    print(f"  [csv] → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: latex table
# ─────────────────────────────────────────────────────────────────────────────

def save_latex(table: dict, path: Path):
    cols = ["SO","TO","CT","CL","ATTR","PO","Overall","CLIPIQA","Spd↑","Lat(s)↓"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "r"*len(cols) + r"}",
        r"\toprule",
        "Schedule & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for tag, row in table.items():
        vals = []
        for c in ["SO","TO","CT","CL","ATTR","PO","Overall","CLIPIQA","Spd","Lat(s)"]:
            v = row.get(c, float("nan"))
            vals.append(f"{v:.4f}" if not np.isnan(v) else "--")
        lines.append(tag.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule",r"\end{tabular}}",
              r"\caption{BornSchedule vs Uniform: T2I-CompBench + CLIPIQA on Flux.}",
              r"\label{tab:born_flux}",r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [latex] → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BornSchedule Flux evaluation (T2I-CompBench + CLIPIQA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stats",     default=None,
                   help=".npz from stats_flux.py (required for BornSchedule)")
    p.add_argument("--model",     default="black-forest-labs/FLUX.1-schnell",
                   help="HuggingFace Flux model name")
    p.add_argument("--nfe_list",  nargs="+", type=int, default=[50, 25, 10],
                   help="NFE values to evaluate")
    p.add_argument("--n_prompts", type=int, default=len(T2I_PROMPTS),
                   help="How many prompts to use (max=%d)" % len(T2I_PROMPTS))
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

    prompts = T2I_PROMPTS[:args.n_prompts]
    print(f"[eval_flux_born]  NFE={args.nfe_list}  prompts={len(prompts)}  "
          f"seeds={args.seeds}  model={args.model}")

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

    # ── Load progress checkpoint ───────────────────────────────────────────
    progress = load_progress(ckpt_path)
    print(f"  Checkpoint: {len(progress)} items already done.")

    # ── Generation (skipped if --eval_only) ───────────────────────────────
    if not args.eval_only:
        pipe_w = FluxCustomSchedule(args.model, device=args.device)
        print("\n  Starting image generation …")

        total = len(schedules) * len(prompts) * len(args.seeds)
        done  = sum(1 for k in progress if any(
            k.startswith(f"{tag}/") for tag in schedules))
        print(f"  Total={total}  Already={done}  Remaining={total-done}")

        with tqdm(total=total-done, desc="Generating", ncols=90) as pbar:
            for tag, sigmas in schedules.items():
                tag_dir = out_dir / tag
                tag_dir.mkdir(parents=True, exist_ok=True)
                for idx, entry in enumerate(prompts):
                    for seed in args.seeds:
                        key = f"{tag}/{idx}/{seed}"
                        if key in progress:
                            continue
                        img, lat = pipe_w.generate(
                            entry["prompt"], sigmas, seed=seed,
                            height=args.height, width=args.width,
                            guidance_scale=args.guidance_scale,
                        )
                        img_path = tag_dir / f"p{idx:04d}_s{seed}.png"
                        img.save(str(img_path))
                        progress[key] = {"path": str(img_path), "latency": lat}
                        save_progress(ckpt_path, progress)
                        pbar.update(1)

        del pipe_w
        torch.cuda.empty_cache()
        print("  [gen] Done.")
    else:
        print("  --eval_only: skipping generation.")

    # ── Scoring ───────────────────────────────────────────────────────────
    print("\n  Starting scoring …")
    clipiqa_scorer = CLIPIQAScorer(device=args.device)
    vqa_scorer     = VQAScorer(device=args.device)
    results = score_all(clipiqa_scorer, vqa_scorer,
                        prompts, schedules, progress, ckpt_path)

    # ── Aggregate & display ───────────────────────────────────────────────
    baseline_tag = args.baseline_tag or f"uniform_nfe{max(args.nfe_list)}"
    table = aggregate(results, baseline_tag=baseline_tag)
    print_table(table)

    # Save outputs
    save_table_csv(table, out_dir / "results.csv")
    save_latex(table, out_dir / "results_table.tex")
    with open(out_dir / "results_full.json", "w") as f:
        json.dump(
            {tag: {k: (v if not isinstance(v, list) else [float(x) for x in v])
                   for k, v in r.items()}
             for tag, r in results.items()}, f, indent=2)
    print(f"  [eval_flux_born] All done.  Outputs → {out_dir}/")


if __name__ == "__main__":
    main()