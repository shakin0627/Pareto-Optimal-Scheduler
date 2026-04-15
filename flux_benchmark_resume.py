"""
flux_benchmark_resume.py
========================
nohup python flux_benchmark_resume.py \
      --schedules_dir /media/ssd_horse/keying/eval_results \
      --model black-forest-labs/FLUX.1-dev \
      --out_dir /media/ssd_horse/keying/eval_results \
      --device cuda \
      --n_prompts 200 \
> /media/ssd_horse/keying/flux_eval2.log 2>&1 &

python flux_benchmark_resume.py \
      --schedules_dir /media/ssd_horse/keying/eval_results \
      --out_dir /media/ssd_horse/keying/eval_results \
      --device cuda \
      --eval_only
"""

import argparse, gc, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

T2I_PROMPTS = [
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
    {"prompt": "a cat sitting on top of a table",        "cat": "PO", "attr": "spatial"},
    {"prompt": "a bird flying above the clouds",         "cat": "PO", "attr": "spatial"},
    {"prompt": "a dog running beside a bicycle",         "cat": "PO", "attr": "spatial"},
    {"prompt": "a lamp next to a bookshelf",             "cat": "PO", "attr": "spatial"},
    {"prompt": "a child standing in front of a door",    "cat": "PO", "attr": "spatial"},
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
# Schedule loading
# ─────────────────────────────────────────────────────────────────────────────

def load_schedules(schedules_dir: Path) -> dict[str, np.ndarray]:
    """
    return {tag: sigma_array}，sigma_array decreasing
    """
    schedules = {}
    npy_files = sorted(schedules_dir.glob("born_nfe*.npy"))
    if not npy_files:
        raise FileNotFoundError(
            f"No born_nfe*.npy found in {schedules_dir}.\n"
            "Make sure --schedules_dir points to the directory with those files."
        )

    for npy_path in npy_files:
        # e.g. born_nfe10.npy → nfe=10
        stem = npy_path.stem          # "born_nfe10"
        nfe  = int(stem.replace("born_nfe", ""))
        sigmas = np.load(str(npy_path))
        schedules[f"born_nfe{nfe}"] = sigmas
        print(f"  Loaded born_nfe{nfe}: {len(sigmas)-1} steps, "
              f"σ ∈ [{sigmas[-1]:.4f}, {sigmas[0]:.4f}]")

        sigma_max, sigma_min = float(sigmas[0]), float(sigmas[-1])
        uni = np.linspace(sigma_max, sigma_min, nfe + 1)
        schedules[f"uniform_nfe{nfe}"] = uni

    return schedules


# ─────────────────────────────────────────────────────────────────────────────
# Progress checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_progress(path: Path, prog: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(prog, f)
    tmp.replace(path)


def free_gpu(*objs):
    for o in objs:
        try:
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
        print(f"  [GPU] freed → {free_mb:.0f}/{total_mb:.0f} MiB free")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 generation
# ─────────────────────────────────────────────────────────────────────────────

def phase_generate(
    model_name: str,
    schedules: dict[str, np.ndarray],
    prompts: list[dict],
    seeds: list[int],
    out_dir: Path,
    progress: dict,
    ckpt_path: Path,
    device: str,
    height: int, width: int,
    guidance_scale: float,
):  
    # successful hack?
    # 内部重新调用set_timesteps覆盖
    total    = len(schedules) * len(prompts) * len(seeds)
    done_keys = {k for k in progress}
    remaining = [
        (tag, idx, seed)
        for tag in schedules
        for idx  in range(len(prompts))
        for seed in seeds
        if f"{tag}/{idx}/{seed}" not in done_keys
    ]
    if not remaining:
        print("  [gen] All images already generated, skipping.")
        return

    print(f"  [gen] {len(remaining)}/{total} images to generate …")

    import copy
    from diffusers import FluxPipeline

    print(f"  [gen] Loading Flux: {model_name} (sequential_cpu_offload) …")
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    print("  [gen] Flux loaded (sequential offload).")

    try:
        with tqdm(total=len(remaining), desc="Generating", ncols=90) as pbar:
            for tag, idx, seed in remaining:
                sigmas  = schedules[tag]
                entry   = prompts[idx]
                nfe     = len(sigmas) - 1
                key     = f"{tag}/{idx}/{seed}"
                img_dir = out_dir / tag
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"p{idx:04d}_s{seed}.png"

                # sigma schedule
                import copy as _copy
                sched = _copy.deepcopy(pipe.scheduler)
                ts    = torch.tensor(sigmas, dtype=torch.float32)
                sched.timesteps = ts
                sched.sigmas    = ts
                pipe.scheduler  = sched

                gen = torch.Generator(device="cpu").manual_seed(seed)  # sequential_offload 要求 cpu generator
                t0  = time.perf_counter()
                out = pipe(
                    prompt=entry["prompt"],
                    num_inference_steps=nfe,
                    guidance_scale=guidance_scale,
                    height=height, width=width,
                    generator=gen,
                    output_type="pil",
                )
                latency = time.perf_counter() - t0
                img     = out.images[0]

                img.save(str(img_path))
                progress[key] = {"path": str(img_path), "latency": latency}
                save_progress(ckpt_path, progress)

                del out, img, gen, sched, ts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.update(1)

    finally:
        print("\n  [gen] Unloading Flux from GPU …")
        free_gpu(pipe)
        del pipe

class CLIPIQAScorer:
    def __init__(self, device):
        import pyiqa
        self.metric = pyiqa.create_metric("clipiqa", device=device)
        self.device = device

    def score(self, image: Image.Image) -> float:
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.metric(t))


class VQAScorer:
    def __init__(self, device):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        print("  [score] Loading BLIP-VQA …")
        self.proc  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.device = device
        print("  [score] BLIP-VQA loaded.")

    def _p_yes(self, image: Image.Image, question: str) -> float:
        inp = self.proc(image, question, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        logits = out.logits[0]
        yes_id = self.proc.tokenizer.convert_tokens_to_ids("yes")
        no_id  = self.proc.tokenizer.convert_tokens_to_ids("no")
        probs  = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)
        return float(probs[0])

    def score_item(self, image: Image.Image, entry: dict) -> dict:
        cat, prompt = entry["cat"], entry["prompt"]
        if cat == "SO":
            return {"SO": self._p_yes(image,
                f"Does the image show {prompt}? Answer yes or no.")}
        elif cat == "TO":
            parts = prompt.split(" and ", 1)
            q1 = f"Is there {parts[0].strip()} in the image? Answer yes or no."
            q2 = (f"Is there {parts[1].strip()} in the image? Answer yes or no."
                  if len(parts) > 1 else q1)
            return {"TO": (self._p_yes(image, q1) + self._p_yes(image, q2)) / 2}
        elif cat == "CT":
            return {"CT": self._p_yes(image,
                f"Does the image show exactly what is described: {prompt}? Answer yes or no.")}
        elif cat == "CL":
            colors = ["red","blue","green","yellow","purple","pink","brown",
                      "black","white","orange","gray","grey","silver","gold",
                      "turquoise","crimson","golden"]
            found  = [c for c in colors if c in prompt.lower()]
            q = (f"Is the main object {found[0]}? Answer yes or no."
                 if found else
                 f"Does the image match the description: {prompt}? Answer yes or no.")
            return {"CL": self._p_yes(image, q)}
        elif cat == "ATTR":
            return {"ATTR": self._p_yes(image,
                f"Does the image correctly show {prompt}? Answer yes or no.")}
        elif cat == "PO":
            return {"PO": self._p_yes(image,
                f"Is the spatial arrangement correct: {prompt}? Answer yes or no.")}
        return {}


def phase_score(
    schedules: dict[str, np.ndarray],
    prompts: list[dict],
    seeds: list[int],
    progress: dict,
    device: str,
) -> dict:
    """
    return {tag: {cat: [scores], "clipiqa": [...], "latency": [...]}}
    """
    clipiqa = CLIPIQAScorer(device)
    vqa     = VQAScorer(device)

    results = {
        tag: {"SO":[],"TO":[],"CT":[],"CL":[],"ATTR":[],"PO":[],
              "clipiqa":[],"latency":[]}
        for tag in schedules
    }

    for tag in schedules:
        print(f"\n  [score] {tag} …")
        for idx, entry in enumerate(tqdm(prompts, desc=tag, ncols=80)):
            for seed in seeds:
                key  = f"{tag}/{idx}/{seed}"
                meta = progress.get(key)
                if meta is None:
                    continue
                img_path = Path(meta["path"])
                if not img_path.exists():
                    print(f"  [warn] missing: {img_path}")
                    continue

                img = Image.open(img_path).convert("RGB")

                results[tag]["clipiqa"].append(clipiqa.score(img))
                results[tag]["latency"].append(meta["latency"])

                for cat, sc in vqa.score_item(img, entry).items():
                    results[tag][cat].append(sc)

    free_gpu(clipiqa.metric, vqa.model)
    return results

def aggregate(results: dict, baseline_tag: str) -> dict:
    cats = ["SO","TO","CT","CL","ATTR","PO"]
    base_lat = np.mean(results.get(baseline_tag, {}).get("latency", [1.0]) or [1.0])
    table = {}
    for tag, r in results.items():
        row = {c: float(np.mean(r[c])) if r[c] else float("nan") for c in cats}
        row["Overall"] = float(np.nanmean([row[c] for c in cats]))
        row["CLIPIQA"] = float(np.mean(r["clipiqa"])) if r["clipiqa"] else float("nan")
        lat = float(np.mean(r["latency"])) if r["latency"] else float("nan")
        row["Lat(s)"]  = lat
        row["Spd"]     = base_lat / lat if lat > 0 else float("nan")
        table[tag]     = row
    return table


def print_table(table: dict):
    cols = ["SO","TO","CT","CL","ATTR","PO","Overall","CLIPIQA","Spd","Lat(s)"]
    hdr  = f"{'Schedule':<28}" + "".join(f"{c:>9}" for c in cols)
    sep  = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for tag, row in sorted(table.items()):
        vals = [row.get(c, float("nan")) for c in cols]
        line = f"{tag:<28}" + "".join(
            f"{v:>9.4f}" if not np.isnan(v) else f"{'nan':>9}" for v in vals)
        print(line)
    print(sep + "\n")


def save_results(table: dict, out_dir: Path):
    cols = ["SO","TO","CT","CL","ATTR","PO","Overall","CLIPIQA","Spd","Lat(s)"]
    # CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w") as f:
        f.write("Schedule," + ",".join(cols) + "\n")
        for tag, row in table.items():
            f.write(tag + "," + ",".join(
                str(round(row.get(c, float("nan")), 6)) for c in cols) + "\n")
    print(f"  [out] CSV  → {csv_path}")

    # LaTeX
    tex_path = out_dir / "results_table.tex"
    lines = [
        r"\begin{table}[t]\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "r"*len(cols) + r"}",
        r"\toprule",
        "Schedule & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for tag, row in sorted(table.items()):
        vals = [f"{row.get(c,float('nan')):.4f}"
                if not np.isnan(row.get(c, float("nan"))) else "--"
                for c in cols]
        lines.append(tag.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [out] LaTeX → {tex_path}")

    # Full JSON
    json_path = out_dir / "results_full.json"
    with open(json_path, "w") as f:
        json.dump({t: {k: ([float(x) for x in v] if isinstance(v, list) else v)
                       for k,v in r.items()}
                   for t,r in table.items()}, f, indent=2)
    print(f"  [out] JSON  → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--schedules_dir", required=True,
                   help="Directory containing born_nfe*.npy files")
    p.add_argument("--model",  default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--out_dir", default="eval_results")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--n_prompts", type=int, default=len(T2I_PROMPTS))
    p.add_argument("--seeds",     type=int, nargs="+", default=[42])
    p.add_argument("--height",    type=int, default=512)
    p.add_argument("--width",     type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip generation; only score existing images")
    p.add_argument("--baseline_tag", default=None,
                   help="Tag for 1× speedup reference (default: uniform_nfe<max>)")
    args = p.parse_args()

    out_dir   = Path(args.out_dir);   out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "progress.json"
    prompts   = T2I_PROMPTS[:args.n_prompts]

    # ──  schedule ──────────────────────────────────────────────────────
    print(f"\n[1/3] Loading schedules from {args.schedules_dir} …")
    schedules = load_schedules(Path(args.schedules_dir))
    nfe_list  = sorted({int(t.split("nfe")[1]) for t in schedules
                        if t.startswith("born_")})
    print(f"  NFEs found: {nfe_list}")
    print(f"  Tags: {list(schedules.keys())}")

    # ── progress ───────────────────────────────────────────────────────
    progress = load_progress(ckpt_path)
    print(f"  Progress checkpoint: {len(progress)} items already done.")

    # ── generation ───────────────────────────────────────────────────────
    if not args.eval_only:
        print(f"\n[2/3] PHASE 1 — Generation (Flux only on GPU) …")
        phase_generate(
            model_name=args.model,
            schedules=schedules,
            prompts=prompts,
            seeds=args.seeds,
            out_dir=out_dir,
            progress=progress,
            ckpt_path=ckpt_path,
            device=args.device,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        )
        # phase_generate 
        print("  [2/3] Generation done. GPU now free for scoring.\n")
    else:
        print("[2/3] --eval_only: skipping generation.\n")

    # ── scoring ───────────────────────────────────────────────────────
    print("[3/3] PHASE 2 — Scoring (BLIP + CLIPIQA, no Flux) …")
    results = phase_score(
        schedules=schedules,
        prompts=prompts,
        seeds=args.seeds,
        progress=progress,
        device=args.device,
    )

    # ── output ───────────────────────────────────────────────────────────────
    baseline_tag = args.baseline_tag or f"uniform_nfe{max(nfe_list)}"
    table = aggregate(results, baseline_tag)
    print_table(table)
    save_results(table, out_dir)
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()