"""
BornSchedule evaluation suite.

Usage
-----
    python run_fid.py single --nfe 8 10 12 20 --n_images 10000 --batch 32

    python run_fid.py ckpt --nfe 10 --ckpt_dir runs/nfe10/ckpts \\
        --n_images 10000 --batch 32

    python run_evaluation.py grid --nfe 10 --n_images 5000 --batch 32 \\
        --w_rank1 0.5 1.0 2.0 --w_vres 0.5 1.0 2.0 --w_disc 0.5 1.0 2.0
"""

import os, gc, shutil, argparse, itertools, json, datetime
os.environ["HF_HUB_DISABLE_XET"]    = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from scheduling_born import OptimalSchedule, _cost_functional
from diffusers import DDPMPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Global constant: r1 fixed to match cost functional derivation
# ─────────────────────────────────────────────────────────────────────────────

R1 = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# VRAM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_free(*objs):
    for o in objs:
        try:    o.cpu()
        except: pass
        try:    del o
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unet_to(unet, device):
    unet.to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_uint8(x: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1]  ->  (B,H,W,3) uint8"""
    x = x.clamp(-1, 1).cpu().float()
    x = ((x + 1) / 2 * 255).byte()
    return x.permute(0, 2, 3, 1).numpy()


def _save_batch(imgs: np.ndarray, out_dir: Path, start: int):
    for i, img in enumerate(imgs):
        Image.fromarray(img).save(out_dir / f"{start + i:06d}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (UNet kept on CPU until generate() needs it)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    pipe      = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    unet      = pipe.unet.eval().cpu()
    alphas_cp = pipe.scheduler.alphas_cumprod
    ns        = NoiseScheduleVP("discrete",
                                alphas_cumprod=alphas_cp.to(device))
    del pipe
    gc.collect()
    return unet, ns


def make_solver(unet, ns):
    def _raw(x, t, **kw):
        return unet(x, t.to(x.device)).sample
    fn = model_wrapper(_raw, ns, model_type="noise", guidance_type="uncond")
    return DPM_Solver(fn, ns, algorithm_type="dpmsolver")


# ─────────────────────────────────────────────────────────────────────────────
# Outer time-grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _outer_from_lambdas(ns, lambdas: np.ndarray, device: str) -> torch.Tensor:
    """np.ndarray (K+1,) of lambda values -> decreasing t tensor."""
    lam_t = torch.tensor(lambdas, dtype=torch.float32, device=device)
    return ns.inverse_lambda(lam_t)


def _logsnr_uniform_outer(ns, K: int, device: str) -> torch.Tensor:
    """
    logSNR-uniform outer grid with K outer steps.
    Equivalent to solver.sample(skip_type='logSNR', steps=2*K, method='singlestep').
    """
    t_end   = 1.0 / ns.total_N
    t_start = float(ns.T)
    lam_max = float(ns.marginal_lambda(
        torch.tensor(t_start, dtype=torch.float32, device=device)))
    lam_min = float(ns.marginal_lambda(
        torch.tensor(t_end,   dtype=torch.float32, device=device)))
    lams = torch.linspace(lam_max, lam_min, K + 1, device=device)
    return ns.inverse_lambda(lams)


# ─────────────────────────────────────────────────────────────────────────────
# Single sampler — used by BOTH born and baseline.
# Only t_outer differs; everything else (including r1) is identical.
# ─────────────────────────────────────────────────────────────────────────────

def _sample_batch(solver, x_T: torch.Tensor,
                  t_outer: torch.Tensor, device: str) -> torch.Tensor:
    """
    Single-step DPM-Solver-2 over pre-computed outer grid t_outer (K+1,).
    r1 = R1 = 0.5, matching the cost functional derivation.
    Born and baseline both call this; only t_outer differs.
    """
    K = len(t_outer) - 1
    x = x_T
    with torch.no_grad():
        for k in range(K):
            s = t_outer[k    ].reshape(1)
            t = t_outer[k + 1].reshape(1)
            x = solver.singlestep_dpm_solver_update(
                x, s, t, order=2, solver_type="dpmsolver", r1=R1,
            )
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# UNet is moved to GPU at entry and back to CPU before return.
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    unet, ns, out_dir: Path,
    mode: str,              # "born" | "baseline"
    K: int,                 # number of outer steps (NFE = 2*K)
    n_images: int,
    batch_size: int,
    device: str,
    seed: int = 42,
    lambdas: np.ndarray = None,   # required when mode == "born"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"{n_images - 1:06d}.png").exists():
        print(f"  [skip] {out_dir} already complete ({n_images} images)")
        return

    _unet_to(unet, device)
    solver = make_solver(unet, ns)

    if mode == "born":
        assert lambdas is not None, "lambdas required for mode='born'"
        t_outer = _outer_from_lambdas(ns, lambdas, device)
    else:
        t_outer = _logsnr_uniform_outer(ns, K, device)

    saved = 0
    while saved < n_images:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed + saved)
        bs  = min(batch_size, n_images - saved)
        x_T = torch.randn(bs, 3, 32, 32, device=device, generator=rng)

        x = _sample_batch(solver, x_T, t_outer, device)

        _save_batch(_to_uint8(x), out_dir, saved)
        saved += bs
        del x, x_T
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{mode}] {saved}/{n_images}", end="\r", flush=True)

    print(f"  [{mode}] done -> {out_dir}          ")

    _unet_to(unet, "cpu")
    del solver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — single torch-fidelity pipeline (FID + optional P/R)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(img_dir: Path) -> dict:
    """
    All metrics go through torch-fidelity so Inception preprocessing
    is identical for every run — no separate feature extractor.
    Requires torch-fidelity >= 0.3.0 for P/R; older versions give FID only.
    """
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError("pip install torch-fidelity")

    base_kw = dict(
        input1=str(img_dir),
        input2="cifar10-train",
        fid=True,
        cuda=torch.cuda.is_available(),
        batch_size=256,
        datasets_root=".fid_cache",
        datasets_download=True,
        verbose=False,
    )

    try:
        m         = torch_fidelity.calculate_metrics(**base_kw, prc=True)
        precision = float(m.get("precision", float("nan")))
        recall    = float(m.get("recall",    float("nan")))
    except TypeError:
        # older torch-fidelity: FID only
        m         = torch_fidelity.calculate_metrics(**base_kw)
        precision = float("nan")
        recall    = float("nan")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fid":       float(m["frechet_inception_distance"]),
        "precision": precision,
        "recall":    recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def _optimise_born(
    nfe: int,
    w_rank1: float = 1.0,
    w_vres:  float = 1.0,
    w_disc:  float = 1.0,
    max_iter: int  = 5000,
    lr: float      = 1e-2,
    tol: float     = 1e-7,
    verbose: bool  = True,
    ckpt_dir: str  = None,
) -> tuple:
    """Return (lambdas_opt, born).  lambdas_opt: np.ndarray of shape (K+1,)."""
    born = OptimalSchedule(
        "google/ddpm-cifar10-32",
        r1=R1, max_iter=max_iter, lr=lr, tol=tol,
        verbose=verbose,
        w_rank1=w_rank1, w_vres=w_vres, w_disc=w_disc,
        ckpt_dir=ckpt_dir,
    )
    K       = nfe // 2
    lambdas = born._optimise(K)
    return lambdas, born


def _uniform_lambdas(born: OptimalSchedule, K: int) -> np.ndarray:
    return np.linspace(born.lambda_min, born.lambda_max, K + 1)


def _cost_at(born: OptimalSchedule, lambdas: np.ndarray) -> float:
    return float(_cost_functional(
        lambdas,
        g_fn=born.g_fn, sigma2_fn=born.sigma2_fn,
        sigma2_gpp_fn=born.sigma2_gpp_fn, ell_gpp=born.ell_gpp,
        phi_res_fn=born._phi_res_fn, rho_infty=born.rho_infty,
        r1=born.r1,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Result I/O and printing
# ─────────────────────────────────────────────────────────────────────────────

def _save_result(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _fmt(v):
    return f"{v:.3f}" if (isinstance(v, float) and v == v) else "  n/a"


def _print_header():
    print(f"\n  {'Tag':<32}  {'FID':>6}  {'Prec':>6}  {'Rec':>6}")
    print("  " + "-" * 55)


def _print_row(tag: str, m: dict):
    print(f"  {tag:<32}  {m['fid']:6.2f}  "
          f"{_fmt(m['precision']):>6}  {_fmt(m['recall']):>6}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 : single
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    unet, ns = load_model(device)
    all_results = {}

    for nfe in args.nfe:
        assert nfe % 2 == 0, f"NFE must be even, got {nfe}"
        K        = nfe // 2
        base_dir = Path(args.outdir) / f"nfe{nfe}"
        print(f"\n{'='*60}\n  NFE={nfe}  K={K}  r1={R1}\n{'='*60}")

        lambdas_opt, born = _optimise_born(
            nfe, max_iter=args.max_iter, lr=args.lr, tol=args.tol,
            verbose=args.verbose,
            ckpt_dir=str(base_dir / "ckpts"),
        )
        lam_unif  = _uniform_lambdas(born, K)
        cost_born = born.cost_at_schedule()
        cost_unif = _cost_at(born, lam_unif)

        print(f"  lambda nodes (born): {np.round(lambdas_opt, 4)}")
        print(f"  Born cost:           {cost_born:.4e}")
        print(f"  Uniform cost:        {cost_unif:.4e}")
        print(f"  Ratio unif/born:     {cost_unif / cost_born:.1f}x")

        born_dir  = base_dir / "born"
        base_imgs = base_dir / "baseline"

        print(f"\n[born] Generating {args.n_images} images -> {born_dir}")
        generate(unet, ns, born_dir, "born", K,
                 args.n_images, args.batch, device, args.seed, lambdas_opt)

        print(f"\n[baseline] Generating {args.n_images} images -> {base_imgs}")
        generate(unet, ns, base_imgs, "baseline", K,
                 args.n_images, args.batch, device, args.seed)

        print("\n[metrics] Computing…")
        m_born = compute_metrics(born_dir)
        m_base = compute_metrics(base_imgs)

        all_results[nfe] = {
            "born": m_born, "baseline": m_base,
            "lambdas_born": lambdas_opt.tolist(),
            "lambdas_unif": lam_unif.tolist(),
            "cost_born": cost_born, "cost_unif": cost_unif,
        }
        _save_result(base_dir / "results.json", all_results[nfe])

        _print_header()
        _print_row(f"born     NFE={nfe}", m_born)
        _print_row(f"baseline NFE={nfe}", m_base)

        if not args.keep_imgs:
            shutil.rmtree(born_dir,  ignore_errors=True)
            shutil.rmtree(base_imgs, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  Summary  ({args.n_images} images, CIFAR-10 train, r1={R1})")
    print(f"{'='*60}")
    _print_header()
    for nfe, r in all_results.items():
        _print_row(f"born     NFE={nfe}", r["born"])
        _print_row(f"baseline NFE={nfe}", r["baseline"])


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 : ckpt
# ─────────────────────────────────────────────────────────────────────────────

def run_ckpt(args):
    assert args.nfe % 2 == 0
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    unet, ns = load_model(device)
    nfe      = args.nfe
    K        = nfe // 2
    ckpt_dir = Path(args.ckpt_dir)

    npz_files = sorted(ckpt_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {ckpt_dir}")

    born_ref  = OptimalSchedule("google/ddpm-cifar10-32", verbose=False)
    lam_unif  = _uniform_lambdas(born_ref, K)
    cost_unif = _cost_at(born_ref, lam_unif)

    out_root    = Path(args.outdir) / f"nfe{nfe}_ckpt"
    all_results = {}

    # baseline once
    base_imgs = out_root / "baseline"
    print(f"\n[baseline] Generating {args.n_images} images -> {base_imgs}")
    generate(unet, ns, base_imgs, "baseline", K,
             args.n_images, args.batch, device, args.seed)
    print("[metrics] Baseline…")
    m_base = compute_metrics(base_imgs)
    all_results["baseline"] = m_base

    for npz in npz_files:
        data    = np.load(npz)
        lambdas = data["lambdas"]
        tag     = npz.stem
        cost    = float(data["cost"]) if "cost" in data else float("nan")

        print(f"\n-- {tag}  cost={cost:.4e}  unif={cost_unif:.4e}  "
              f"ratio={cost_unif / cost:.1f}x")
        print(f"   lambda: {np.round(lambdas, 4)}")

        img_dir = out_root / tag
        print(f"[ckpt/{tag}] Generating {args.n_images} images -> {img_dir}")
        generate(unet, ns, img_dir, "born", K,
                 args.n_images, args.batch, device, args.seed, lambdas)

        print(f"[metrics] {tag}…")
        m = compute_metrics(img_dir)
        all_results[tag] = {**m, "cost": cost, "lambdas": lambdas.tolist()}
        _save_result(out_root / "results.json", all_results)

        if not args.keep_imgs:
            shutil.rmtree(img_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  Checkpoint comparison  NFE={nfe}")
    print(f"{'='*60}")
    _print_header()
    _print_row("baseline (logSNR)", m_base)
    for tag, r in all_results.items():
        if tag == "baseline":
            continue
        _print_row(tag, r)

    if not args.keep_imgs:
        shutil.rmtree(base_imgs, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 : grid
# ─────────────────────────────────────────────────────────────────────────────

def run_grid(args):
    assert args.nfe % 2 == 0
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    unet, ns = load_model(device)
    nfe      = args.nfe
    K        = nfe // 2
    out_root = Path(args.outdir) / f"nfe{nfe}_grid"
    combos   = list(itertools.product(args.w_rank1, args.w_vres, args.w_disc))
    all_results = {}

    print(f"Grid: {len(combos)} combinations x NFE={nfe}  K={K}")

    # baseline once
    base_imgs = out_root / "baseline"
    print(f"\n[baseline] Generating {args.n_images} images -> {base_imgs}")
    generate(unet, ns, base_imgs, "baseline", K,
             args.n_images, args.batch, device, args.seed)
    print("[metrics] Baseline…")
    m_base = compute_metrics(base_imgs)
    all_results["baseline"] = m_base

    for wr, wv, wd in combos:
        tag = f"r{wr}_v{wv}_d{wd}"
        print(f"\n{'─'*55}")
        print(f"  w_rank1={wr}  w_vres={wv}  w_disc={wd}  -> {tag}")

        lambdas, born = _optimise_born(
            nfe,
            w_rank1=wr, w_vres=wv, w_disc=wd,
            max_iter=args.max_iter, lr=args.lr, tol=args.tol,
            verbose=args.verbose,
            ckpt_dir=str(out_root / "ckpts" / tag),
        )
        lam_unif  = _uniform_lambdas(born, K)
        cost_born = born.cost_at_schedule()
        cost_unif = _cost_at(born, lam_unif)

        print(f"  lambda: {np.round(lambdas, 4)}")
        print(f"  Born cost={cost_born:.4e}  Unif={cost_unif:.4e}  "
              f"ratio={cost_unif / cost_born:.1f}x")

        img_dir = out_root / tag
        print(f"[born/{tag}] Generating {args.n_images} images -> {img_dir}")
        generate(unet, ns, img_dir, "born", K,
                 args.n_images, args.batch, device, args.seed, lambdas)

        print(f"[metrics] {tag}…")
        m = compute_metrics(img_dir)
        all_results[tag] = {
            **m,
            "w_rank1": wr, "w_vres": wv, "w_disc": wd,
            "cost_born": cost_born, "cost_unif": cost_unif,
            "lambdas": lambdas.tolist(),
        }
        _save_result(out_root / "results.json", all_results)

        if not args.keep_imgs:
            shutil.rmtree(img_dir, ignore_errors=True)

    entries = [(k, v) for k, v in all_results.items() if k != "baseline"]
    entries.sort(key=lambda x: x[1]["fid"])

    print(f"\n{'='*60}")
    print(f"  Grid search summary  NFE={nfe}  ({args.n_images} images)")
    print(f"{'='*60}")
    _print_header()
    _print_row("baseline (logSNR)", m_base)
    for tag, r in entries:
        _print_row(tag, r)

    best_tag, best = entries[0]
    print(f"\n  Best: {best_tag}  FID={best['fid']:.2f}  "
          f"P={_fmt(best['precision'])}  R={_fmt(best['recall'])}")

    if not args.keep_imgs:
        shutil.rmtree(base_imgs, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _add_common(p):
    p.add_argument("--n_images",  type=int,   default=10000)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--outdir",    type=str,   default="fid_out")
    p.add_argument("--keep_imgs", action="store_true",
                   help="Keep generated PNGs after computing metrics")
    p.add_argument("--max_iter",  type=int,   default=20000)
    p.add_argument("--lr",        type=float, default=1e-2)
    p.add_argument("--tol",       type=float, default=1e-7)
    p.add_argument("--verbose",   action="store_true")


def main():
    root = argparse.ArgumentParser(
        description="BornSchedule FID / Precision / Recall evaluation suite")
    sub  = root.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single", help="Born vs logSNR baseline at one or more NFE")
    _add_common(p1)
    p1.add_argument("--nfe", type=int, nargs="+", default=[10])

    p2 = sub.add_parser("ckpt", help="Compare saved checkpoints vs logSNR baseline")
    _add_common(p2)
    p2.add_argument("--nfe",      type=int, required=True)
    p2.add_argument("--ckpt_dir", type=str, required=True,
                    help="Directory containing .npz checkpoint files")

    p3 = sub.add_parser("grid", help="Grid search over cost weights")
    _add_common(p3)
    p3.add_argument("--nfe",     type=int,   required=True)
    p3.add_argument("--w_rank1", type=float, nargs="+", default=[1.0])
    p3.add_argument("--w_vres",  type=float, nargs="+", default=[1.0])
    p3.add_argument("--w_disc",  type=float, nargs="+", default=[1.0])

    args = root.parse_args()
    print(f"[run_fid] {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"mode={args.mode}  r1={R1} (fixed, matches cost functional)")

    if   args.mode == "single": run_single(args)
    elif args.mode == "ckpt":   run_ckpt(args)
    elif args.mode == "grid":   run_grid(args)


if __name__ == "__main__":
    main()