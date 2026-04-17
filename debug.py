"""
debug.py  — checkpoint-based FID sweep
================================================

Usage
----
python debug.py --nfe 12 --n_images 2000 --ckpt_every 500

python debug.py --nfe 8 10 12 --n_images 5000 \\
    --lr_list 3e-3 1e-3 3e-4 --max_iter_list 4000 8000

python debug.py --nfe 12 --lr_list 1e-3 --max_iter_list 8000 \\
    --seeds 0 1 2 --n_images 5000
"""

import os, gc, shutil, time, csv, json, argparse, itertools
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CkptResult:
    nfe:          int
    lr:           float
    max_iter:     int
    seed:         int
    ckpt_iter:    int      
    cost:         float
    fid_born:     float
    fid_baseline: float
    fid_delta:    float    
    elapsed_s:    float

@dataclass
class RunSummary:
    nfe:             int
    lr:              float
    max_iter:        int
    seed:            int
    grad_norm_step1: float
    grad_norm_step5: float
    final_grad_norm: float
    n_iter_actual:   int
    converged:       bool
    best_ckpt_iter:  int
    best_fid_born:   float
    best_fid_delta:  float
    fid_baseline:    float
    error_msg:       str


CKPT_COLS = ["nfe","lr","max_iter","seed","ckpt_iter",
             "cost","fid_born","fid_baseline","fid_delta","elapsed_s"]
RUN_COLS  = ["nfe","lr","max_iter","seed",
             "grad_norm_step1","grad_norm_step5","final_grad_norm",
             "n_iter_actual","converged",
             "best_ckpt_iter","best_fid_born","best_fid_delta",
             "fid_baseline","error_msg"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading & Sampling
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    from dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
    from diffusers import DDPMPipeline

    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    unet = pipe.unet.eval().to(device)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    ns = NoiseScheduleVP("discrete", alphas_cumprod=alphas_cumprod)
    del pipe.scheduler

    def raw_unet(x, t_input, **kwargs):
        return unet(x, t_input.to(x.device)).sample

    model_fn = model_wrapper(raw_unet, ns, model_type="noise",
                             guidance_type="uncond")
    dpm_solver = DPM_Solver(model_fn, ns, algorithm_type="dpmsolver")
    return unet, ns, dpm_solver


def generate_from_lambdas(dpm_solver, ns, lambdas, n_images, batch_size,
                           device, out_dir: Path, seed=42):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    lam_t   = torch.tensor(lambdas, dtype=torch.float32, device=device)
    t_outer = ns.inverse_lambda(lam_t)
    K       = len(t_outer) - 1

    saved = 0
    rng   = torch.Generator(device=device)
    while saved < n_images:
        rng.manual_seed(seed + saved)
        bs = min(batch_size, n_images - saved)
        x  = torch.randn(bs, 3, 32, 32, device=device, generator=rng)
        with torch.no_grad():
            for k in range(K):
                s = t_outer[k].reshape(1)
                t = t_outer[k+1].reshape(1)
                t_inner   = dpm_solver.get_time_steps("logSNR", s.item(),
                                                       t.item(), N=2, device=device)
                lam_inner = ns.marginal_lambda(t_inner)
                h         = lam_inner[-1] - lam_inner[0]
                r1        = float((lam_inner[1] - lam_inner[0]) / h)
                x         = dpm_solver.singlestep_dpm_solver_update(
                                x, s, t, order=2, solver_type="dpmsolver", r1=r1)
        x_np = ((x.clamp(-1,1).cpu().float() + 1) / 2 * 255)
        x_np = x_np.permute(0,2,3,1).numpy().astype("uint8")
        for i, img in enumerate(x_np):
            Image.fromarray(img).save(out_dir / f"{saved+i:06d}.png")
        saved += bs
        del x, x_np
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def generate_baseline(dpm_solver, ns, steps, n_images, batch_size,
                       device, out_dir: Path, seed=42):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    rng   = torch.Generator(device=device)
    while saved < n_images:
        rng.manual_seed(seed + saved)
        bs  = min(batch_size, n_images - saved)
        x_T = torch.randn(bs, 3, 32, 32, device=device, generator=rng)
        with torch.no_grad():
            x = dpm_solver.sample(x_T, steps=steps, t_start=ns.T,
                                  t_end=1.0/ns.total_N, order=2,
                                  skip_type="logSNR", method="singlestep")
        x_np = ((x.clamp(-1,1).cpu().float() + 1) / 2 * 255)
        x_np = x_np.permute(0,2,3,1).numpy().astype("uint8")
        for i, img in enumerate(x_np):
            Image.fromarray(img).save(out_dir / f"{saved+i:06d}.png")
        saved += bs
        del x, x_T, x_np
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def compute_fid(img_dir: Path) -> float:
    import torch_fidelity
    m = torch_fidelity.calculate_metrics(
        input1=str(img_dir), input2="cifar10-train",
        fid=True, isc=False, kid=False,
        cuda=torch.cuda.is_available(), batch_size=256,
        datasets_root=".fid_cache", datasets_download=True, verbose=False,
    )
    return float(m["frechet_inception_distance"])


def _offload(unet):
    unet.cpu(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _reload(unet, device):
    unet.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    nfe, lr, max_iter, seed,
    device, unet, ns, dpm_solver,
    n_images, batch_size,
    work_dir: Path,
    ckpt_every=500, tol=1e-7, keep_imgs=False,
) -> tuple[RunSummary, list[CkptResult]]:

    from scheduling_born import OptimalSchedule, _cost_functional

    tag      = f"nfe{nfe}_lr{lr:.0e}_iter{max_iter}_s{seed}"
    ckpt_dir = work_dir / tag / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary = RunSummary(
        nfe=nfe, lr=lr, max_iter=max_iter, seed=seed,
        grad_norm_step1=float("nan"), grad_norm_step5=float("nan"),
        final_grad_norm=float("nan"), n_iter_actual=0, converged=False,
        best_ckpt_iter=-1, best_fid_born=float("nan"),
        best_fid_delta=float("nan"), fid_baseline=float("nan"), error_msg="",
    )
    ckpt_results: list[CkptResult] = []

    try:
        torch.manual_seed(seed); np.random.seed(seed)
        born = OptimalSchedule(
            "google/ddpm-cifar10-32",
            r1=0.5, max_iter=max_iter, lr=lr, tol=tol,
            ckpt_dir=str(ckpt_dir), ckpt_every=ckpt_every,
            verbose=False,
        )
        K = nfe // 2
        try:
            born._optimise_with_hooks(K)   
        except AttributeError:
            born._optimise(K)              # fallback

        gh = getattr(born, "grad_history", [])
        summary.grad_norm_step1 = gh[0] if len(gh) >= 1 else float("nan")
        summary.grad_norm_step5 = gh[4] if len(gh) >= 5 else float("nan")
        summary.final_grad_norm = getattr(born, "final_grad_norm", float("nan"))
        summary.n_iter_actual   = getattr(born, "n_iter_actual", 0)
        summary.converged       = getattr(born, "converged", False)

    except Exception as e:
        summary.error_msg = f"optimise: {e}"
        return summary, ckpt_results

    # Baseline FID
    base_dir = work_dir / tag / "baseline"
    try:
        print(f"    [baseline] generating {n_images} imgs…")
        generate_baseline(dpm_solver, ns, nfe, n_images, batch_size,
                          device, base_dir, seed=seed)
        _offload(unet)
        summary.fid_baseline = compute_fid(base_dir)
        _reload(unet, device)
        print(f"    [baseline] FID = {summary.fid_baseline:.2f}")
    except Exception as e:
        summary.error_msg = f"baseline_fid: {e}"
        _reload(unet, device)
    finally:
        if not keep_imgs and base_dir.exists():
            shutil.rmtree(base_dir)

    # FID 
    def ckpt_sort_key(p):
        n = int(p.stem.replace("ckpt_iter", ""))
        return (1, 0) if n < 0 else (0, n)   
    ckpt_files = sorted(ckpt_dir.glob("ckpt_iter*.npz"), key=ckpt_sort_key)
    if not ckpt_files:
        summary.error_msg += " | no_ckpts"
        return summary, ckpt_results

    best_fid  = float("inf")
    best_iter = -1

    for ckpt_path in ckpt_files:
        iter_n     = int(ckpt_path.stem.replace("ckpt_iter", ""))
        iter_label = "final" if iter_n < 0 else str(iter_n)

        try:
            data     = np.load(ckpt_path)
            lambdas  = data["lambdas"]
            cost_val = float(data["cost"])
        except Exception as e:
            print(f"    [ckpt {iter_label}] load error: {e}")
            continue

        born_dir = work_dir / tag / f"born_{iter_label}"
        t0 = time.time()
        try:
            print(f"    [ckpt {iter_label:>6}] generating…", end="", flush=True)
            generate_from_lambdas(dpm_solver, ns, lambdas, n_images, batch_size,
                                  device, born_dir, seed=seed)
            _offload(unet)
            fid_born  = compute_fid(born_dir)
            _reload(unet, device)
            elapsed   = time.time() - t0
            fid_delta = summary.fid_baseline - fid_born

            marker = " ◀ BEST" if fid_born < best_fid else ""
            print(f"  FID={fid_born:.2f}  Δ={fid_delta:+.2f}  cost={cost_val:.3e}  "
                  f"({elapsed:.0f}s){marker}")

            cr = CkptResult(
                nfe=nfe, lr=lr, max_iter=max_iter, seed=seed,
                ckpt_iter=iter_n, cost=cost_val,
                fid_born=fid_born, fid_baseline=summary.fid_baseline,
                fid_delta=fid_delta, elapsed_s=elapsed,
            )
            ckpt_results.append(cr)

            if fid_born < best_fid:
                best_fid  = fid_born
                best_iter = iter_n

        except Exception as e:
            print(f"  ERROR: {e}")
            _reload(unet, device)
        finally:
            if not keep_imgs and born_dir.exists():
                shutil.rmtree(born_dir)

    summary.best_ckpt_iter = best_iter
    summary.best_fid_born  = best_fid
    summary.best_fid_delta = summary.fid_baseline - best_fid

    if not keep_imgs:
        shutil.rmtree(work_dir / tag, ignore_errors=True)

    return summary, ckpt_results


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_completed_runs(run_csv: Path) -> set:
    """comleted (nfe, lr, max_iter, seed) """
    if not run_csv.exists():
        return set()
    completed = set()
    with open(run_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["nfe"]), float(row["lr"]),
                       int(row["max_iter"]), int(row["seed"]))
                completed.add(key)
            except (KeyError, ValueError):
                pass
    return completed

def append_csv(path: Path, obj, cols):
    d = asdict(obj)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        w.writerow({k: d.get(k, "") for k in cols})


def print_run_summary(s: RunSummary):
    conv = "✓" if s.converged else "✗"
    err  = f"  !! {s.error_msg}" if s.error_msg else ""
    print(
        f"  {conv} NFE={s.nfe} lr={s.lr:.0e} iter={s.max_iter} seed={s.seed} | "
        f"g1={s.grad_norm_step1:.2e} g5={s.grad_norm_step5:.2e} "
        f"gfin={s.final_grad_norm:.2e} n={s.n_iter_actual} | "
        f"best_ckpt={s.best_ckpt_iter} "
        f"FID={s.best_fid_born:.2f} ΔFID={s.best_fid_delta:+.2f} "
        f"base={s.fid_baseline:.2f}" + err
    )


def print_global_summary(summaries: list[RunSummary]):
    from collections import defaultdict

    groups = defaultdict(list)
    for s in summaries:
        if not s.error_msg:
            groups[(s.nfe, s.lr, s.max_iter)].append(s)

    print(f"\n{'='*90}")
    print("  GLOBAL SUMMARY  (mean ± std over seeds)  — ranked by ΔFID↑")
    print(f"{'='*90}")
    print(f"{'NFE':>4} {'lr':>8} {'iter':>6}  "
          f"{'ΔFID↑':>8} {'±':>5}  {'FID_born':>9} {'±':>5}  "
          f"{'best_ckpt':>9}  {'g_step1':>9}  {'conv%':>6}")
    print("-" * 90)

    def stat(vals):
        v = [x for x in vals if not np.isnan(x)]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    rows = []
    for (nfe, lr, mi), rs in groups.items():
        dm, ds  = stat([r.best_fid_delta for r in rs])
        fm, fs  = stat([r.best_fid_born  for r in rs])
        bm, _   = stat([r.best_ckpt_iter for r in rs])
        g1m, _  = stat([r.grad_norm_step1 for r in rs])
        conv    = 100 * np.mean([r.converged for r in rs])
        rows.append((nfe, lr, mi, dm, ds, fm, fs, bm, g1m, conv))

    for row in sorted(rows, key=lambda r: -r[3]):
        nfe, lr, mi, dm, ds, fm, fs, bm, g1m, conv = row
        print(f"{nfe:>4} {lr:>8.0e} {mi:>6d}  "
              f"{dm:>8.3f} {ds:>5.3f}  {fm:>9.2f} {fs:>5.2f}  "
              f"{bm:>9.0f}  {g1m:>9.2e}  {conv:>5.0f}%")
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nfe",           type=int,   nargs="+", default=[12])
    p.add_argument("--lr_list",       type=float, nargs="+",
                   default=[1e-2, 3e-3, 1e-3, 3e-4])
    p.add_argument("--max_iter_list", type=int,   nargs="+", default=[4000, 8000])
    p.add_argument("--seeds",         type=int,   nargs="+", default=[0, 1, 2])
    p.add_argument("--ckpt_every",    type=int,   default=500)
    p.add_argument("--tol",           type=float, default=1e-7)
    p.add_argument("--n_images",      type=int,   default=5000)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--outdir",        type=str,   default="sweep_out")
    p.add_argument("--ckpt_csv",      type=str,   default="sweep_ckpt_fid.csv")
    p.add_argument("--run_csv",       type=str,   default="sweep_run_summary.csv")
    p.add_argument("--keep_imgs",     action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    configs = list(itertools.product(
        args.nfe, args.lr_list, args.max_iter_list, args.seeds))
    n_ckpts = len(range(args.ckpt_every,
                        max(args.max_iter_list)+1, args.ckpt_every)) + 1
    print(f"Total runs: {len(configs)},  ~{n_ckpts} checkpoints each\n")

    work_dir = Path(args.outdir)
    print("Loading model…")
    unet, ns, dpm_solver = load_model(device)
    print("Ready.\n")
    # ── 在 work_dir / load_model 之后加 ──
    completed = load_completed_runs(Path(args.run_csv))
    if completed:
        print(f"Resuming: {len(completed)} runs already done, skipping.\n")

    all_summaries = []
    for i, (nfe, lr, max_iter, seed) in enumerate(configs, 1):
        key = (nfe, lr, max_iter, seed)
        if key in completed:
            print(f"[{i}/{len(configs)}]  SKIP (done)  "
                  f"NFE={nfe} lr={lr:.0e} max_iter={max_iter} seed={seed}")
            continue                          # skip

        print(f"\n{'─'*60}")
        print(f"[{i}/{len(configs)}]  NFE={nfe}  lr={lr:.0e}  "
              f"max_iter={max_iter}  seed={seed}")

        summary, ckpt_results = run_one(
            nfe=nfe, lr=lr, max_iter=max_iter, seed=seed,
            device=device, unet=unet, ns=ns, dpm_solver=dpm_solver,
            n_images=args.n_images, batch_size=args.batch,
            work_dir=work_dir, ckpt_every=args.ckpt_every,
            tol=args.tol, keep_imgs=args.keep_imgs,
        )
        print_run_summary(summary)
        append_csv(Path(args.run_csv),  summary,      RUN_COLS)
        for cr in ckpt_results:
            append_csv(Path(args.ckpt_csv), cr, CKPT_COLS)
        all_summaries.append(summary)

    print_global_summary(all_summaries)
    print(f"\nCheckpoint FID details : {args.ckpt_csv}")
    print(f"Run summaries          : {args.run_csv}")


if __name__ == "__main__":
    main()