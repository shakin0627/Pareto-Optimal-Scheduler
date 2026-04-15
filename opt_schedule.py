from __future__ import annotations

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

import numpy as np
import torch
from scipy.interpolate import interp1d, PchipInterpolator

# ─────────────────────────────────────────────────────────────────────────────
# VP-SDE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alpha(lam: float) -> float:
    return 1.0 / math.sqrt(1.0 + math.exp(-2.0 * lam))

def _sigma_vp(lam: float) -> float:
    return 1.0 / math.sqrt(1.0 + math.exp(2.0 * lam))


# ─────────────────────────────────────────────────────────────────────────────
# Φ builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_phi_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    quad_pts: int = 64,
    s_extend_factor: float = 3.0,
) -> Callable[[float, float], float]:
    rho_s    = np.asarray(rho_s,    dtype=np.float64)
    rho_vals = np.asarray(rho_vals, dtype=np.float64)
    rho_vals = np.clip(rho_vals, 1e-12, None)

    if rho_s[0] > 1e-10:
        rho_s    = np.concatenate([[0.0], rho_s])
        rho_vals = np.concatenate([[rho_vals[0]], rho_vals])

    log_rho    = np.log(rho_vals)
    rho_interp = PchipInterpolator(rho_s, log_rho, extrapolate=False)

    s_max  = rho_s[-1] * s_extend_factor
    n_fine = max(2000, int(s_max / (rho_s[-1] + 1e-12) * len(rho_s) * 4))
    s_fine = np.linspace(0.0, s_max, n_fine)

    log_rho_fine = rho_interp(s_fine)
    nan_mask = np.isnan(log_rho_fine)
    if nan_mask.any():
        last  = int(np.where(~nan_mask)[0][-1])
        slope = (log_rho_fine[last] - log_rho_fine[max(last - 1, 0)]) / (s_fine[1] - s_fine[0] + 1e-30)
        slope = min(slope, -1e-3)
        for idx in np.where(nan_mask)[0]:
            log_rho_fine[idx] = log_rho_fine[last] + slope * (s_fine[idx] - s_fine[last])
    rho_fine = np.exp(np.clip(log_rho_fine, -30.0, 0.0))

    ds     = s_fine[1] - s_fine[0]
    R_fine = np.zeros(n_fine)
    R_fine[1:] = np.cumsum(0.5 * (rho_fine[:-1] + rho_fine[1:]) * ds)
    R_interp = interp1d(s_fine, R_fine, kind="linear", bounds_error=False,
                        fill_value=(0.0, float(R_fine[-1])))

    _trapz = getattr(np, "trapezoid", None) or np.trapz

    def phi_fn(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 0.0
        a_int = min(a, b)
        b_int = max(a, b)
        u     = np.linspace(0.0, a_int, quad_pts)
        return float(_trapz(R_interp(u) + R_interp(b_int - u), u))

    return phi_fn


def _build_phi_res_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    rho_infty: float,
    quad_pts: int = 64,
) -> Callable[[float, float], float]:
    phi_full = _build_phi_fn(rho_s, rho_vals, quad_pts)

    def phi_res_fn(a: float, b: float) -> float:
        return phi_full(a, b) - rho_infty * a * b

    return phi_res_fn


# ─────────────────────────────────────────────────────────────────────────────
# ε_k and Γ
# ─────────────────────────────────────────────────────────────────────────────

def _compute_epsilon(
    lam_prev: float,
    lam_curr: float,
    g_fn: Callable,
    r1: float = 0.5,
) -> float:
    h      = lam_curr - lam_prev
    lam_s  = lam_prev + r1 * h
    I_full = math.exp(-lam_prev) - math.exp(-lam_curr)
    I_half = math.exp(-lam_prev) - math.exp(-lam_s)
    al_p   = _alpha(lam_prev)
    al_s   = _alpha(lam_s)
    a_k    = (1.0 - 1.0 / (2.0 * r1)) * al_p * I_full
    b_k    = -(al_s / (2.0 * r1)) * I_full
    phi_k  = 1.0 + al_p * I_half * g_fn(lam_prev)
    return a_k * g_fn(lam_prev) + b_k * g_fn(lam_s) * phi_k


def _compute_gamma(eps: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    N     = len(eps)
    Gamma = np.ones(N)
    for k in range(N - 2, -1, -1):
        Gamma[k] = Gamma[k + 1] * (1.0 + eps[k + 1])
    return Gamma


# ─────────────────────────────────────────────────────────────────────────────
# A_j  (rank-1 weight)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_A_j(
    lam_prev: float,
    lam_curr: float,
    sigma2_fn: Callable,
    g_fn: Callable,
    r1: float = 0.5,
    n_quad: int = 32,
    alpha_power: float = 0.5,
) -> float:
    """
    A_j^γ = α_{t_j}^γ · ∫_{full} e^{-μ} σ_η dμ  −  m_j · α_{s_j}^γ · ∫_{half} e^{-μ} σ_η dμ

    where m_j = (g/2r₁)·σ(λ_j)·(e^h−1) is the DPM-Solver-2 correction coefficient
    (unchanged from γ=1), and only the explicit α factors in the two-piece
    weight carry the power γ = alpha_power.

    γ=1  → original cost (α² weighting, ~22000× dynamic range at NFE=20)
    γ=0  → uniform weighting (no α preference, known to hurt FID)
    γ=0.5 → default, square-root scaling (~150× dynamic range)
    """
    h     = lam_curr - lam_prev
    lam_s = lam_prev + r1 * h
    al_j  = _alpha(lam_curr) ** alpha_power
    al_s  = _alpha(lam_s)    ** alpha_power

    # DPM-Solver-2 correction coefficient — not an α weight, stays γ-free
    m_j = (g_fn(lam_prev) / (2.0 * r1)) * _sigma_vp(lam_curr) * math.expm1(h)

    mu_full      = np.linspace(lam_prev, lam_curr, n_quad)
    sig_eta_full = np.sqrt(np.maximum(
        [sigma2_fn(float(m)) for m in mu_full], 0.0))
    int_full = float(np.trapezoid(np.exp(-mu_full) * sig_eta_full, mu_full))

    n_half       = max(2, n_quad // 2)
    mu_half      = np.linspace(lam_prev, lam_s, n_half)
    sig_eta_half = np.sqrt(np.maximum(
        [sigma2_fn(float(m)) for m in mu_half], 0.0))
    int_half = float(np.trapezoid(np.exp(-mu_half) * sig_eta_half, mu_half))

    return al_j * int_full - m_j * al_s * int_half


# ─────────────────────────────────────────────────────────────────────────────
# V_j^res
# ─────────────────────────────────────────────────────────────────────────────

def _compute_V_res(
    lam_prev: float,
    lam_curr: float,
    g_fn: Callable,
    sigma2_fn: Callable,
    phi_res_fn: Callable,
    r1: float = 0.5,
    alpha_power: float = 0.5,
) -> float:
    """
    V_j^res^γ = σ²_η(λ̄) · e^{-2λ_prev} · α_{t_j}^{2γ} · Q_j^res^γ

    Q_j^res^γ = φ^res(h,h) − 2c̃_j^γ φ^res(r₁h,h) + (c̃_j^γ)² φ^res(r₁h,r₁h)

    The normalised correction coefficient that factors out of the α_{t_j}^{2γ}
    prefactor is:
        c̃_j^γ = m_j · (α_{s_j}/α_{t_j})^γ
               = (g/2r₁) · α_{t_j}^{1−γ} · α_{s_j}^γ · I_full

    For γ=1: c̃_j^1 = (g/2r₁)·α_{s_j}·I_full  ← original code
    For γ=0.5: c̃_j^0.5 = (g/2r₁)·α_{t_j}^0.5·α_{s_j}^0.5·I_full
    """
    h       = lam_curr - lam_prev
    lam_s   = lam_prev + r1 * h
    lam_bar = 0.5 * (lam_prev + lam_curr)

    al_j = _alpha(lam_curr)
    al_s = _alpha(lam_s)

    prefactor = sigma2_fn(lam_bar) * math.exp(-2.0 * lam_prev) \
                * (al_j ** (2 * alpha_power))

    I_full = math.exp(-lam_prev) - math.exp(-lam_curr)
    # c̃_j^γ = (g/2r₁) · α_{t_j}^{1−γ} · α_{s_j}^γ · I_full
    c_k = (g_fn(lam_prev) / (2.0 * r1)) \
          * (al_j ** (1.0 - alpha_power)) \
          * (al_s **        alpha_power)  \
          * I_full

    phi_h     = phi_res_fn(h,       h)
    phi_r1h   = phi_res_fn(r1 * h,  r1 * h)
    phi_r1h_h = phi_res_fn(r1 * h,  h)

    Q_res = phi_h - 2.0 * c_k * phi_r1h_h + c_k**2 * phi_r1h
    return prefactor * max(Q_res, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# D_j  (discretization error)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_D_j(
    lam_prev: float,
    lam_curr: float,
    g_fn: Callable,
    sigma2_gpp_fn: Callable,
    ell_gpp: float,
    r1: float = 0.5,
    alpha_power: float = 0.5,
) -> float:
    """
    D_j^γ = σ²_{g''}(λ̄) · ℓ_{g''} · Q_j^{D,γ}

    Q_j^{D,γ} = (α_{t_j}^γ − m_j·α_{s_j}^γ)² · ∫_{half} e^{-2μ} dμ
              +  α_{t_j}^{2γ}                  · ∫_{tail} e^{-2μ} dμ

    m_j = (g/2r₁)·σ(λ_j)·(e^h−1)  is the DPM-Solver-2 correction, γ-free.
    Only the explicit α factors in the two-piece weight carry the power γ.
    """
    h       = lam_curr - lam_prev
    lam_s   = lam_prev + r1 * h
    lam_bar = 0.5 * (lam_prev + lam_curr)

    al_j_g = _alpha(lam_curr) ** alpha_power
    al_s_g = _alpha(lam_s)    ** alpha_power
    m_j    = (g_fn(lam_prev) / (2.0 * r1)) * _sigma_vp(lam_curr) * math.expm1(h)

    def int_exp2(a, b):
        return (math.exp(-2.0 * a) - math.exp(-2.0 * b)) / 2.0

    w_half = (al_j_g - m_j * al_s_g) ** 2
    w_full = al_j_g ** 2
    Q = w_half * int_exp2(lam_prev, lam_s) + w_full * int_exp2(lam_s, lam_curr)

    return sigma2_gpp_fn(lam_bar) * ell_gpp * max(Q, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Full cost functional
# ─────────────────────────────────────────────────────────────────────────────

def _cost_functional(
    lambdas: np.ndarray,
    g_fn: Callable,
    sigma2_fn: Callable,
    sigma2_gpp_fn: Callable,
    phi_res_fn: Callable,
    rho_infty: float,
    ell_gpp: float,
    r1: float = 0.5,
    n_quad: int = 32,
    barrier_weight: float = 0.0,
    w_rank1: float = 1.0,
    w_vres:  float = 1.0,
    w_disc:  float = 1.0,
    alpha_power: float = 0.5,
) -> float:
    N     = len(lambdas) - 1
    h_arr = np.diff(lambdas)

    eps   = np.array([_compute_epsilon(lambdas[k-1], lambdas[k], g_fn, r1)
                      for k in range(1, N+1)])
    Gamma = _compute_gamma(eps, lambdas)
    A     = np.array([_compute_A_j(lambdas[k-1], lambdas[k], sigma2_fn, g_fn,
                                    r1, n_quad, alpha_power)
                      for k in range(1, N+1)])
    Vres  = np.array([_compute_V_res(lambdas[k-1], lambdas[k], g_fn, sigma2_fn,
                                      phi_res_fn, r1, alpha_power)
                      for k in range(1, N+1)])
    D     = np.array([_compute_D_j(lambdas[k-1], lambdas[k], g_fn, sigma2_gpp_fn,
                                    ell_gpp, r1, alpha_power)
                      for k in range(1, N+1)])

    alpha_tN2  = _alpha(lambdas[-1]) ** 2
    rank1_term = w_rank1 * rho_infty * float(np.dot(A, Gamma)) ** 2
    vres_term  = w_vres  * float(np.dot(Vres, Gamma ** 2))
    disc_term  = w_disc  * float(np.dot(D,    Gamma ** 2))
    main_cost  = alpha_tN2 * (rank1_term + vres_term + disc_term)

    if barrier_weight > 0.0:
        h_uniform = (lambdas[-1] - lambdas[0]) / N
        barrier   = -barrier_weight * float(np.sum(np.log(h_arr / h_uniform)))
        return main_cost + barrier

    return main_cost


# ─────────────────────────────────────────────────────────────────────────────
# Gradient and projection
# ─────────────────────────────────────────────────────────────────────────────

def _project_ordering(
    interior: np.ndarray,
    lam0: float,
    lamN: float,
    min_gap: float,
) -> np.ndarray:
    x = np.sort(interior.copy())
    n = len(x)
    for i in range(n):
        x[i] = float(np.clip(x[i],
                              lam0 + (i + 1) * min_gap,
                              lamN - (n - i) * min_gap))
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rho_infty_and_ell(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
) -> tuple:
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.maximum(np.abs(rho_vals[:-1]), 0.05))
    MIN_FLAT = max(5, len(rho_vals) // 10)

    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i : i + MIN_FLAT] < 0.02):
            plateau_start = i
            break
    if plateau_start is None:
        warnings.warn("[BornSchedule] No plateau found — extend rho_s range.", UserWarning)
        plateau_start = int(0.5 * len(rho_vals))

    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i
            break

    mid = plateau_start + (plateau_end - plateau_start) // 2
    plateau_vals = rho_vals[plateau_start:mid]
    rho_infty    = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))

    cv = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(f"[BornSchedule] Plateau CV={cv:.3f} high, ρ_∞ unreliable.", UserWarning)

    denom    = max(1.0 - rho_infty, 1e-8)
    rho_res  = (rho_vals - rho_infty) / denom
    rho_res  = np.maximum(rho_res, 0.0)
    cross    = np.where(rho_res <= 1.0 / math.e)[0]
    ell_corr = float(rho_s[cross[0]]) if len(cross) > 0 else float(rho_s[-1])

    return rho_infty, ell_corr


_STATS_SEARCH_PATHS = [
    Path(os.environ.get("OPT_SCHEDULE_STATS_DIR", ".")),
    Path.home() / ".cache" / "opt_schedule",
]

MODEL_STATS_REGISTRY: Dict[str, dict] = {}


def register_model_stats(model_name: str, stats: dict) -> None:
    MODEL_STATS_REGISTRY[model_name] = stats


def load_model_stats_from_file(model_name: str, path: Union[str, Path]) -> dict:
    data     = np.load(path)
    required = ["lambda_grid", "g_values", "sigma2_values",
                "lambda_min", "lambda_max", "rho_s", "rho_values",
                "rho_s_gpp", "rho_values_gpp"]
    missing  = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Stats file '{path}' missing keys: {missing}")

    stats = {k: np.asarray(data[k], dtype=np.float64) for k in required}
    stats["lambda_min"] = float(data["lambda_min"])
    stats["lambda_max"] = float(data["lambda_max"])

    if "sigma2_gpp_values" in data:
        stats["sigma2_gpp_values"] = np.asarray(data["sigma2_gpp_values"], dtype=np.float64)
    elif "sigma2_gp_values" in data:
        warnings.warn(
            "[BornSchedule] sigma2_gpp_values not in .npz; falling back to "
            "sigma2_gp_values. Re-run estimate_model_stats.py with --estimate_gpp.",
            UserWarning,
        )
        stats["sigma2_gpp_values"] = np.asarray(data["sigma2_gp_values"], dtype=np.float64)
    else:
        warnings.warn("[BornSchedule] No g'' stats found; D_j = 0.", UserWarning)
        stats["sigma2_gpp_values"] = np.zeros_like(stats["sigma2_values"])

    register_model_stats(model_name, stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main scheduler class
# ─────────────────────────────────────────────────────────────────────────────

class OptimalSchedule:
    """BornSchedule v3 — with alpha²-normalisation for NFE-stable optimisation."""

    order = 2

    def __init__(
        self,
        model_name: str,
        r1: float = 0.5,
        max_iter: int = 3000,
        lr: float = 1e-2,
        lr_decay: float = 0.9995,
        tol: float = 1e-7,
        fd_eps: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        quad_pts: int = 64,
        barrier_weight_scale: float = 0.005,
        w_rank1: float = 1.0,
        w_vres:  float = 1.0,
        w_disc:  float = 1.0,
        alpha_power: float = 0.5,   # γ: α^{2γ} weighting. 1.0=original, 0.5=default, 0.0=uniform
        ckpt_dir: Optional[str] = None,
        ckpt_every: int = 500,
        verbose: bool = False,
        stats_override: Optional[dict] = None,
    ):
        self.model_name           = model_name
        self.r1                   = r1
        self.max_iter             = max_iter
        self.lr                   = lr
        self.lr_decay             = lr_decay
        self.tol                  = tol
        self.fd_eps               = fd_eps
        self.beta1                = beta1
        self.beta2                = beta2
        self.quad_pts             = quad_pts
        self.barrier_weight_scale = barrier_weight_scale
        self.verbose              = verbose
        self.w_rank1 = w_rank1
        self.w_vres  = w_vres
        self.w_disc      = w_disc
        self.alpha_power = alpha_power
        self.ckpt_dir   = ckpt_dir
        self.ckpt_every = ckpt_every

        stats = stats_override if stats_override is not None else self._load_stats(model_name)
        self._init_from_stats(stats)

        self.timesteps:           Optional[torch.Tensor] = None
        self.sigmas:              Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int]          = None
        self._lambdas_opt:        Optional[np.ndarray]   = None

    # ------------------------------------------------------------------
    # Stats loading
    # ------------------------------------------------------------------

    def _load_stats(self, model_name: str) -> dict:
        safe = model_name.replace("/", "--").replace(":", "--")
        for base in _STATS_SEARCH_PATHS:
            fpath = base / f"{safe}.npz"
            if fpath.exists():
                warnings.warn(f"[BornSchedule] Loading stats from {fpath}",
                              UserWarning, stacklevel=3)
                return load_model_stats_from_file(model_name, fpath)
        if model_name in MODEL_STATS_REGISTRY:
            return MODEL_STATS_REGISTRY[model_name]
        raise FileNotFoundError(
            f"[BornSchedule] No stats found for '{model_name}'. "
            "Run estimate_model_stats.py first."
        )

    def _init_from_stats(self, stats: dict) -> None:
        lam_grid        = np.asarray(stats["lambda_grid"],       dtype=np.float64)
        g_vals          = np.asarray(stats["g_values"],          dtype=np.float64)
        sigma2_vals     = np.asarray(stats["sigma2_values"],     dtype=np.float64)
        sigma2_gpp_vals = np.asarray(stats["sigma2_gpp_values"], dtype=np.float64)
        rho_s           = np.asarray(stats["rho_s"],             dtype=np.float64)
        rho_vals        = np.asarray(stats["rho_values"],        dtype=np.float64)
        rho_s_gpp       = np.asarray(stats["rho_s_gpp"],         dtype=np.float64)
        rho_vals_gpp    = np.asarray(stats["rho_values_gpp"],    dtype=np.float64)

        self.lambda_min = float(stats["lambda_min"])
        self.lambda_max = float(stats["lambda_max"])

        kw = dict(kind="linear", bounds_error=False)
        self._g_fn         = interp1d(lam_grid, g_vals,
                                      fill_value=(g_vals[0],          g_vals[-1]),          **kw)
        self._sigma2_fn    = interp1d(lam_grid, sigma2_vals,
                                      fill_value=(sigma2_vals[0],     sigma2_vals[-1]),     **kw)
        self._sigma2gpp_fn = interp1d(lam_grid, sigma2_gpp_vals,
                                      fill_value=(sigma2_gpp_vals[0], sigma2_gpp_vals[-1]), **kw)

        self.rho_infty, self.ell_corr = _extract_rho_infty_and_ell(rho_s, rho_vals)
        self._phi_res_fn = _build_phi_res_fn(
            rho_s, rho_vals, self.rho_infty, quad_pts=self.quad_pts,
        )

        from scipy.integrate import trapezoid
        rho_norm     = rho_vals_gpp / rho_vals_gpp[0]
        self.ell_gpp = float(trapezoid(rho_norm, rho_s_gpp))

        if self.verbose:
            print(f"  [BornSchedule] ρ_∞={self.rho_infty:.4f}  "
                  f"ℓ_corr={self.ell_corr:.4f}  "
                  f"λ∈[{self.lambda_min:.3f},{self.lambda_max:.3f}]")

    def g_fn(self,          lam: float) -> float: return float(self._g_fn(lam))
    def sigma2_fn(self,     lam: float) -> float: return max(float(self._sigma2_fn(lam)), 0.0)
    def sigma2_gpp_fn(self, lam: float) -> float: return max(float(self._sigma2gpp_fn(lam)), 0.0)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def lambda_to_sigma(lam):
        return 1.0 / np.sqrt(1.0 + np.exp(2.0 * np.asarray(lam, dtype=np.float64)))

    @staticmethod
    def sigma_to_lambda(sigma):
        s     = np.asarray(sigma, dtype=np.float64)
        alpha = np.sqrt(np.clip(1.0 - s**2, 1e-12, None))
        return np.log(alpha / np.clip(s, 1e-12, None))

    # ------------------------------------------------------------------
    # Cost / gradient helpers
    # ------------------------------------------------------------------

    def _kw(self, barrier_weight: float = 0.0) -> dict:
        """Common keyword args for _cost_functional, including alpha_power."""
        return dict(
            g_fn=self.g_fn, sigma2_fn=self.sigma2_fn,
            sigma2_gpp_fn=self.sigma2_gpp_fn, ell_gpp=self.ell_gpp,
            phi_res_fn=self._phi_res_fn, rho_infty=self.rho_infty,
            r1=self.r1, n_quad=self.quad_pts,
            barrier_weight=barrier_weight,
            w_rank1=self.w_rank1, w_vres=self.w_vres, w_disc=self.w_disc,
            alpha_power=self.alpha_power,
        )

    def _cost(self, lambdas: np.ndarray, barrier_weight: float = 0.0) -> float:
        return _cost_functional(lambdas, **self._kw(barrier_weight))

    def _grad(self, interior, lam0, lamN, barrier_weight):
        grad = np.zeros_like(interior)
        kw   = self._kw(barrier_weight)
        fd_h = self.fd_eps
        for i in range(len(interior)):
            xp = interior.copy(); xp[i] += fd_h
            xm = interior.copy(); xm[i] -= fd_h
            Cp = _cost_functional(np.concatenate([[lam0], xp, [lamN]]), **kw)
            Cm = _cost_functional(np.concatenate([[lam0], xm, [lamN]]), **kw)
            grad[i] = (Cp - Cm) / (2.0 * fd_h)
        return grad

    def _save_ckpt(self, it: int, interior: np.ndarray,
                   lam0: float, lamN: float, cost: float) -> None:
        import json
        if self.ckpt_dir is None:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        lams = np.concatenate([[lam0], interior, [lamN]])
        npz_path = os.path.join(self.ckpt_dir, f"ckpt_iter{it:06d}.npz")
        np.savez(npz_path,
                 lambdas=lams, sigmas=self.lambda_to_sigma(lams),
                 cost=np.array(cost), iter=np.array(it),
                 alpha_power=np.array(self.alpha_power))
        json_path = os.path.join(self.ckpt_dir, f"ckpt_iter{it:06d}.json")
        with open(json_path, "w") as f:
            import json as _json
            _json.dump({
                "iter": it, "cost": float(cost),
                "lambdas": lams.tolist(),
                "sigmas": self.lambda_to_sigma(lams).tolist(),
                "alpha_power": self.alpha_power,
            }, f, indent=2)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------


    def _optimise(self, N: int) -> np.ndarray:
        lam0    = self.lambda_min
        lamN    = self.lambda_max
        h_uni   = (lamN - lam0) / N
        min_gap = h_uni / 5.0

        # ── alpha² normalisation ──────────────────────────────────────────

        interior = np.linspace(lam0, lamN, N + 1)[1:-1].copy()

        if self.verbose:
            grad0 = self._grad(interior, lam0, lamN, barrier_weight=0.0)
            print(f"  gradient at uniform_logSNR: {grad0}")
            print(f"  gradient norm: {np.linalg.norm(grad0):.6f}")

        cost_uniform   = self._cost(np.linspace(lam0, lamN, N + 1), barrier_weight=0.0)
        barrier_weight = self.barrier_weight_scale * cost_uniform / N

        if self.verbose:
            print(f"  [BornSchedule] N={N}  C(uniform)={cost_uniform:.4e}  "
                  f"barrier_weight={barrier_weight:.3e}")

        m, v     = np.zeros_like(interior), np.zeros_like(interior)
        lr       = self.lr
        best_cost     = np.inf
        best_interior = interior.copy()

        for it in range(1, self.max_iter + 1):
            grad     = self._grad(interior, lam0, lamN, barrier_weight)
            m        = self.beta1 * m + (1.0 - self.beta1) * grad
            v        = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            mh       = m / (1.0 - self.beta1 ** it)
            vh       = v / (1.0 - self.beta2 ** it)
            interior = interior - lr * mh / (np.sqrt(vh) + 1e-8)
            interior = _project_ordering(interior, lam0, lamN, min_gap)

            cost = self._cost(np.concatenate([[lam0], interior, [lamN]]), barrier_weight)
            if cost < best_cost:
                best_cost, best_interior = cost, interior.copy()

            if self.ckpt_dir and (it % self.ckpt_every == 0):
                self._save_ckpt(it, interior, lam0, lamN, cost)

            gnorm = float(np.linalg.norm(grad))
            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
                print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  lr={lr:.4e}")
            if gnorm < self.tol:
                if self.verbose:
                    print(f"  converged  iter={it}  |g|={gnorm:.3e}")
                break
            lr *= self.lr_decay

        lams_best = np.concatenate([[lam0], best_interior, [lamN]])

        if self.verbose:
            self._print_cost_breakdown(lams_best)

        self._lambdas_opt = lams_best
        return lams_best

    def _optimise_with_hooks(self, N: int, capture_steps: int = 10) -> np.ndarray:
        lam0    = self.lambda_min
        lamN    = self.lambda_max
        h_uni   = (lamN - lam0) / N
        min_gap = h_uni / 5.0

        # ── alpha² normalisation ──────────────────────────────────────────

        interior = np.linspace(lam0, lamN, N + 1)[1:-1].copy()

        cost_uniform   = self._cost(np.linspace(lam0, lamN, N + 1), barrier_weight=0.0)
        barrier_weight = self.barrier_weight_scale * cost_uniform / N

        if self.verbose:
            print(f"  [BornSchedule] N={N}  C(uniform)={cost_uniform:.4e}  "
                  f"barrier_weight={barrier_weight:.3e}")

        m, v  = np.zeros_like(interior), np.zeros_like(interior)
        lr    = self.lr

        best_cost     = np.inf
        best_interior = interior.copy()
        grad_history: list = []

        for it in range(1, self.max_iter + 1):
            grad  = self._grad(interior, lam0, lamN, barrier_weight)
            gnorm = float(np.linalg.norm(grad))
            grad_history.append(gnorm)

            m  = self.beta1 * m + (1.0 - self.beta1) * grad
            v  = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            mh = m / (1.0 - self.beta1 ** it)
            vh = v / (1.0 - self.beta2 ** it)
            interior = interior - lr * mh / (np.sqrt(vh) + 1e-8)
            interior = _project_ordering(interior, lam0, lamN, min_gap)

            cost = self._cost(np.concatenate([[lam0], interior, [lamN]]), barrier_weight)
            if cost < best_cost:
                best_cost, best_interior = cost, interior.copy()

            if self.ckpt_dir and (it % self.ckpt_every == 0):
                self._save_ckpt(it, interior, lam0, lamN, cost)

            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
                print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  lr={lr:.4e}")

            if gnorm < self.tol:
                if self.verbose:
                    print(f"  converged  iter={it}  |g|={gnorm:.3e}")
                break
            lr *= self.lr_decay

        lams_best = np.concatenate([[lam0], best_interior, [lamN]])

        if self.ckpt_dir:
            self._save_ckpt(-1, best_interior, lam0, lamN, best_cost)

        if self.verbose:
            self._print_cost_breakdown(lams_best)

        self.grad_history    = grad_history
        self.final_grad_norm = grad_history[-1] if grad_history else float("nan")
        self.n_iter_actual   = len(grad_history)
        self.converged       = (grad_history[-1] < self.tol) if grad_history else False
        self._lambdas_opt    = lams_best

        return lams_best

    def _print_cost_breakdown(self, lams: np.ndarray) -> None:
        N     = len(lams) - 1
        eps   = np.array([_compute_epsilon(lams[k-1], lams[k], self.g_fn, self.r1)
                           for k in range(1, N+1)])
        Gamma = _compute_gamma(eps, lams)
        A     = np.array([_compute_A_j(lams[k-1], lams[k], self.sigma2_fn, self.g_fn,
                                        self.r1, self.quad_pts, self.alpha_power)
                          for k in range(1, N+1)])
        Vres  = np.array([_compute_V_res(lams[k-1], lams[k], self.g_fn, self.sigma2_fn,
                                          self._phi_res_fn, self.r1, self.alpha_power)
                          for k in range(1, N+1)])
        D     = np.array([_compute_D_j(lams[k-1], lams[k], self.g_fn, self.sigma2_gpp_fn,
                                        self.ell_gpp, self.r1, self.alpha_power)
                          for k in range(1, N+1)])
        a2 = _alpha(lams[-1]) ** 2
        print(f"  Cost breakdown (× α²_{{t_N}} = {a2:.4f}, alpha_power={self.alpha_power:.2f}):")
        print(f"    rank-1  ρ_∞(ΣA_jΓ_j)² : {a2 * self.rho_infty * float(np.dot(A,Gamma))**2:.4e}")
        print(f"    Σ Γ²_j V_j^res         : {a2 * float(np.dot(Vres, Gamma**2)):.4e}")
        print(f"    Σ Γ²_j D_j             : {a2 * float(np.dot(D,    Gamma**2)):.4e}")

    # ------------------------------------------------------------------
    # diffusers-compatible interface
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, "torch.device", None] = None,
    ) -> None:
        N = num_inference_steps
        if self.verbose:
            print(f"[BornSchedule] optimising {N}-step schedule for '{self.model_name}'")

        lambdas_opt       = self._optimise(N)
        self._lambdas_opt = lambdas_opt

        sigmas_np = self.lambda_to_sigma(lambdas_opt)
        sigma_min = float(sigmas_np[-1])
        sigma_max = float(sigmas_np[0])
        t_np      = (sigmas_np - sigma_min) / (sigma_max - sigma_min + 1e-12) * 999.0

        self.sigmas              = torch.tensor(sigmas_np, dtype=torch.float32, device=device)
        self.timesteps           = torch.tensor(t_np,      dtype=torch.float32, device=device)
        self.num_inference_steps = N

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_lambdas(self) -> Optional[np.ndarray]:
        return self._lambdas_opt

    def cost_at_schedule(self) -> Optional[float]:
        if self._lambdas_opt is None:
            return None
        return self._cost(self._lambdas_opt, barrier_weight=0.0)

    def equidistribution_residuals(self) -> Optional[np.ndarray]:
        if self._lambdas_opt is None:
            return None
        lams  = self._lambdas_opt
        N     = len(lams) - 1
        eps   = np.array([_compute_epsilon(lams[k-1], lams[k], self.g_fn, self.r1)
                           for k in range(1, N+1)])
        Gamma = _compute_gamma(eps, lams)
        Vres  = np.array([_compute_V_res(lams[k-1], lams[k], self.g_fn, self.sigma2_fn,
                                          self._phi_res_fn, self.r1, self.alpha_power)
                          for k in range(1, N+1)])
        D     = np.array([_compute_D_j(lams[k-1], lams[k], self.g_fn, self.sigma2_gpp_fn,
                                        self.ell_gpp, self.r1, self.alpha_power)
                          for k in range(1, N+1)])
        return (Vres + D) * Gamma ** 2

    def __repr__(self) -> str:
        return (f"OptimalSchedule(model='{self.model_name}', r1={self.r1}, "
                f"ρ_∞={self.rho_infty:.3f}, ℓ_corr={self.ell_corr:.4f}, "
                f"alpha_power={self.alpha_power:.2f}, "
                f"steps={self.num_inference_steps})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser(description="BornSchedule v3 - Verification & Diagnostics")
    parser.add_argument("--model",     type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--steps",     type=int, default=5)
    parser.add_argument("--r1",        type=float, default=0.5)
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--plot",      action="store_true")
    parser.add_argument("--alpha-power", type=float, default=0.5,
                        help="γ exponent: α^{2γ} step weighting (1.0=original, 0.5=default, 0.0=uniform)")
    parser.add_argument("--stats-dir", type=str, default=None)
    args = parser.parse_args()

    if args.stats_dir:
        os.environ["OPT_SCHEDULE_STATS_DIR"] = args.stats_dir

    print("=" * 80)
    print("BornSchedule v3  —  Verification (with alpha²-normalisation)")
    print("=" * 80)

    start_time = time.time()
    scheduler = OptimalSchedule(
        model_name=args.model, r1=args.r1, alpha_power=args.alpha_power,
        max_iter=3000, lr=1e-2, lr_decay=0.999, tol=1e-6, fd_eps=5e-6,
        verbose=args.verbose,
    )
    print(f"ρ_∞={scheduler.rho_infty:.5f}  ℓ_corr={scheduler.ell_corr:.5f}")

    scheduler.set_timesteps(num_inference_steps=args.steps)
    print(f"Optimisation: {time.time()-start_time:.2f}s  "
          f"alpha_power={scheduler.alpha_power:.2f}")

    lambdas   = scheduler.get_lambdas()
    residuals = scheduler.equidistribution_residuals()
    if residuals is not None:
        mean_r = float(np.mean(residuals))
        std_r  = float(np.std(residuals))
        print(f"Equidistribution CV = {std_r/mean_r*100:.1f}%  "
              f"({'excellent' if std_r/mean_r<0.25 else 'good' if std_r/mean_r<0.4 else 'moderate'})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].plot(lambdas, "o-");   axs[0].set_title("λ schedule");  axs[0].grid(True)
            axs[1].plot(scheduler.lambda_to_sigma(lambdas), "o-", color="orange")
            axs[1].set_title("σ schedule"); axs[1].grid(True)
            axs[2].plot(residuals, "o-", color="green")
            axs[2].axhline(mean_r, color="red", linestyle="--")
            axs[2].set_title("(V^res+D)·Γ² residuals"); axs[2].grid(True)
            plt.tight_layout(); plt.show()
        except ImportError:
            print("Matplotlib not available.")