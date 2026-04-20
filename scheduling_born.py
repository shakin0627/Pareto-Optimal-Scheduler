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
    """α(λ) = 1 / √(1 + exp(−2λ))"""
    return 1.0 / math.sqrt(1.0 + math.exp(-2.0 * lam))


def _sigma_vp(lam: float) -> float:
    """σ(λ) = 1 / √(1 + exp(2λ))"""
    return 1.0 / math.sqrt(1.0 + math.exp(2.0 * lam))


# ─────────────────────────────────────────────────────────────────────────────
# Φ builder  (full kernel — identical to v1)
# ─────────────────────────────────────────────────────────────────────────────

def _build_phi_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    quad_pts: int = 64,
    s_extend_factor: float = 3.0,
) -> Callable[[float, float], float]:
    """
    Build φ(a, b) = ∫₀ᵃ ∫₀ᵇ ρ(|u−u'|) du' du from empirical (rho_s, rho_vals).

    Algorithm:
      1. PCHIP on log ρ, extend tail with linear decay.
      2. Precompute R(s) = ∫₀ˢ ρ(v) dv on a dense grid.
      3. φ(a,b) = ∫₀ᵃ [R(u) + R(b−u)] du  via quad_pts-point trapz.

    Accepts rho_vals that do NOT start at 1.0 (e.g. residual kernel).
    The anchor rho_s[0]=0 is prepended if missing; rho_vals[0] is used as-is.
    """
    rho_s    = np.asarray(rho_s,    dtype=np.float64)
    rho_vals = np.asarray(rho_vals, dtype=np.float64)
    rho_vals = np.clip(rho_vals, 1e-12, None)

    # Anchor at s=0 using the provided rho(0) value (may be < 1 for residual kernel)
    if rho_s[0] > 1e-10:
        rho_s    = np.concatenate([[0.0], rho_s])
        rho_vals = np.concatenate([[rho_vals[0]], rho_vals])   # ← use actual rho(0), not 1

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
    """
    Build φ^res(a, b) = ∫∫ [ρ(|u−u'|) − ρ_∞] du du'
                      = φ(a, b) − ρ_∞ · a · b

    Uses the full φ_fn and subtracts the rank-1 plateau analytically.
    This is exact and preserves h² curvature for small h:
        φ^res(h,h) ≈ (1 − ρ_∞) h²   as h → 0
    avoiding the linear-objective collapse of the banded approximation.
    """
    phi_full = _build_phi_fn(rho_s, rho_vals, quad_pts)

    def phi_res_fn(a: float, b: float) -> float:
        return phi_full(a, b) - rho_infty * a * b

    return phi_res_fn


# ─────────────────────────────────────────────────────────────────────────────
# ε_k and Γ  (unchanged from v1)
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


def _compute_gamma(eps: np.ndarray, lambdas:np.ndarray) -> np.ndarray:
    """Γ[j] = Π_{k=j+1}^{N-1} (1 + ε[k]),  Γ[N-1] = 1."""
    N     = len(eps)
    Gamma = np.ones(N)
    for k in range(N - 2, -1, -1):
        Gamma[k] = Gamma[k + 1] * (1.0 + eps[k + 1])
    return Gamma

# def _compute_gamma(eps: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
#     N     = len(eps)
#     Gamma = np.ones(N)
#     EPS_CLIP = 1e-2   
#     for k in range(N - 2, -1, -1):
#         Gamma[k] = Gamma[k + 1] * max(1.0 + eps[k + 1], EPS_CLIP)
#     return Gamma

# ─────────────────────────────────────────────────────────────────────────────
# A_j  —  rank-1 weight  (1-D quadrature)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_A_j(
    lam_prev: float,
    lam_curr: float,
    sigma2_fn: Callable,
    g_fn: Callable,
    r1: float = 0.5,
    n_quad: int = 32,
) -> float:
    """
    A_j = α_{t_j} ∫_{λ_{j-1}}^{λ_j} e^{-μ} σ_η(μ) dμ
          - c_k α_{s_j} ∫_{λ_{j-1}}^{λ_{s_j}} e^{-μ} σ_η(μ) dμ

    The two-piece weight function from proposal (boxed, line 502-509):
      w_j(μ) = e^{-μ} (α_{t_j} - c_k α_{s_j})  for μ ∈ [λ_{j-1}, λ_{s_j}]
      w_j(μ) = e^{-μ} α_{t_j}                   for μ ∈ [λ_{s_j}, λ_j]

    c_k = g(λ_{j-1}) / (2r₁) · σ_VP(λ_j) · (e^h - 1)
    """
    h     = lam_curr - lam_prev
    lam_s = lam_prev + r1 * h
    al_s  = _alpha(lam_s)
    al_j  = _alpha(lam_curr)

    # half-step Jacobian correction coefficient (same c_k as in _compute_V_res)
    c_k = (g_fn(lam_prev) / (2.0 * r1)) * _sigma_vp(lam_curr) * math.expm1(h)

    # full-step integral: α_{t_j} ∫_{λ_{j-1}}^{λ_j} e^{-μ} σ_η(μ) dμ
    mu_full      = np.linspace(lam_prev, lam_curr, n_quad)
    sig_eta_full = np.sqrt(np.maximum(
        [sigma2_fn(float(m)) for m in mu_full], 0.0))
    int_full = float(np.trapezoid(np.exp(-mu_full) * sig_eta_full, mu_full))

    # half-step integral: c_k α_{s_j} ∫_{λ_{j-1}}^{λ_{s_j}} e^{-μ} σ_η(μ) dμ
    n_half       = max(2, n_quad // 2)
    mu_half      = np.linspace(lam_prev, lam_s, n_half)
    sig_eta_half = np.sqrt(np.maximum(
        [sigma2_fn(float(m)) for m in mu_half], 0.0))
    int_half = float(np.trapezoid(np.exp(-mu_half) * sig_eta_half, mu_half))

    return (al_j * int_full - c_k * al_s * int_half)

# ─────────────────────────────────────────────────────────────────────────────
# V_j^res  —  residual kernel contribution
# ─────────────────────────────────────────────────────────────────────────────

def _compute_V_res(
    lam_prev: float,
    lam_curr: float,
    g_fn: Callable,
    sigma2_fn: Callable,
    phi_res_fn: Callable,
    r1: float = 0.5,
) -> float:
    """
    V_j^res = σ²_η(λ̄_j) · e^{−2λ_{j-1}} · Q_j^res

    Q_j^res = φ^res(h,h) − 2c_k φ^res(r1·h, h) + c_k² φ^res(r1·h, r1·h)

    φ^res(a,b) = φ(a,b) − ρ_∞·a·b  ensures h² curvature for small h.
    c_k mirrors the DPM-Solver-2 midpoint correction (same as v1 _compute_V).
    """
    h       = lam_curr - lam_prev
    lam_s  = lam_prev + r1 * h
    lam_bar = 0.5 * (lam_prev + lam_curr)

    # lam_prev
    prefactor = sigma2_fn(lam_bar) * math.exp(-2.0 * lam_prev) * (_alpha(lam_curr)**2)
    # c_k       = (g_fn(lam_prev) / (2.0 * r1)) * _sigma_vp(lam_curr) * math.expm1(h)
    
    al_s = _alpha(lam_s)
    I_full = math.exp(-lam_prev) - math.exp(-lam_curr)
    c_k = (g_fn(lam_prev) / (2.0 * r1)) * al_s * I_full

    phi_h     = phi_res_fn(h,        h)
    phi_r1h   = phi_res_fn(r1 * h,   r1 * h)
    phi_r1h_h = phi_res_fn(r1 * h,   h)

    Q_res = phi_h - 2.0 * c_k * phi_r1h_h + c_k**2 * phi_r1h
    return prefactor * max(Q_res, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# D_j  —  discretization error (midpoint-rule leading term)
# ─────────────────────────────────────────────────────────────────────────────

# def _compute_D_j(
#     lam_prev: float,
#     lam_curr: float,
#     sigma2_gpp_fn: Callable,
# ) -> float:
#     """
#     D_j = (α²(λ_j) · e^{−2λ̄_j} / 576) · h_j^6 · σ²_{g''}(λ̄_j)

#     Derivation: DPM-Solver-2 with r1=0.5 is the midpoint rule for
#     g(λ) = e^{−λ} ε_θ.  Leading quadrature residual:
#         δ_disc ≈ −K(λ̄) g''(λ̄) / 24 · h³,   K(λ̄) = α_{t_j} e^{−λ̄}
#     Squaring and taking (1/d) E[·]:
#         D_j = K²(λ̄)/576 · h^6 · σ²_{g''}(λ̄)

#     sigma2_gpp_fn: (1/d) E[||g''(λ)||²] from three-point finite difference
#     in estimate_model_stats.py (saved as sigma2_gpp_values in .npz).
#     """
#     h        = lam_curr - lam_prev
#     lam_bar  = 0.5 * (lam_prev + lam_curr)
#     K2       = (_alpha(lam_curr) ** 2) * math.exp(-2.0 * lam_prev)
#     sig2_gpp = max(sigma2_gpp_fn(lam_bar), 0.0)
#     return (K2 / 576.0) * (h ** 6) * sig2_gpp

def _compute_D_j(
    lam_prev: float,
    lam_curr: float,
    g_fn: Callable,
    sigma2_gpp_fn: Callable,
    ell_gpp: float,
    r1: float = 0.5,
) -> float:
    h      = lam_curr - lam_prev
    lam_s  = lam_prev + r1 * h
    lam_bar = 0.5 * (lam_prev + lam_curr)

    al_j = _alpha(lam_curr)
    al_s = _alpha(lam_s)
    c_j  = (g_fn(lam_prev) / (2.0 * r1)) * _sigma_vp(lam_curr) * math.expm1(h)

    # ∫ e^{-2μ} dμ = (e^{-2a} - e^{-2b}) / 2
    def int_exp2(a, b):
        return (math.exp(-2.0*a) - math.exp(-2.0*b)) / 2.0

    w_half = (al_j - c_j * al_s) ** 2
    w_full = al_j ** 2
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
    w_rank1: float = 1.0,   # rank-1 
    w_vres:  float = 1.0,   # V_res 
    w_disc:  float = 1.0,   # D_j 
) -> float:
    N     = len(lambdas) - 1
    h_arr = np.diff(lambdas)

    eps   = np.array([_compute_epsilon(lambdas[k-1], lambdas[k], g_fn, r1)
                      for k in range(1, N+1)])
    Gamma = _compute_gamma(eps, lambdas)
    A     = np.array([_compute_A_j(lambdas[k-1], lambdas[k], sigma2_fn, g_fn, r1, n_quad)
                      for k in range(1, N+1)])
    Vres  = np.array([_compute_V_res(lambdas[k-1], lambdas[k], g_fn, sigma2_fn,
                                     phi_res_fn, r1)
                      for k in range(1, N+1)])
    D     = np.array([_compute_D_j(lambdas[k-1], lambdas[k], g_fn, sigma2_gpp_fn, ell_gpp, r1)
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

def _numerical_gradient(
    interior: np.ndarray,
    lam0: float,
    lamN: float,
    g_fn: Callable,
    sigma2_fn: Callable,
    sigma2_gpp_fn: Callable,
    phi_res_fn: Callable,
    rho_infty: float,
    r1: float,
    ell_gpp: float,
    barrier_weight: float,
    fd_h: float = 1e-5,
) -> np.ndarray:
    grad = np.zeros_like(interior)
    kw   = dict(g_fn=g_fn, sigma2_fn=sigma2_fn, sigma2_gpp_fn=sigma2_gpp_fn,
                phi_res_fn=phi_res_fn, ell_gpp=ell_gpp, rho_infty=rho_infty, r1=r1,
                barrier_weight=barrier_weight)
    for i in range(len(interior)):
        xp = interior.copy(); xp[i] += fd_h
        xm = interior.copy(); xm[i] -= fd_h
        Cp = _cost_functional(np.concatenate([[lam0], xp, [lamN]]), **kw)
        Cm = _cost_functional(np.concatenate([[lam0], xm, [lamN]]), **kw)
        grad[i] = (Cp - Cm) / (2.0 * fd_h)
    return grad


# def _project_ordering(
#     interior: np.ndarray,
#     lam0: float,
#     lamN: float,
#     min_gap: float,
# ) -> np.ndarray:
#     x = np.sort(interior.copy())
#     n = len(x)
#     for i in range(n):
#         x[i] = float(np.clip(x[i],
#                               lam0 + (i + 1) * min_gap,
#                               lamN - (n - i) * min_gap))
#     return x


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

# def _extract_rho_infty_and_ell(
#     rho_s: np.ndarray,
#     rho_vals: np.ndarray,
# ) -> tuple:
#     """
#     ρ_∞   : median of the last 20% of rho_vals (tail plateau).
#     ℓ_corr: s where normalised residual (ρ−ρ_∞)/(1−ρ_∞) first hits 1/e.
#     """
#     tail_start = max(1, int(0.80 * len(rho_vals)))
#     rho_infty  = float(np.median(rho_vals[tail_start:]))
#     rho_infty  = np.clip(rho_infty, 0.0, 1.0 - 1e-6)
#     tail_vals = rho_vals[tail_start:]
#     if np.std(tail_vals) / (np.mean(tail_vals) + 1e-8) > 0.1:
#         warnings.warn(
#             f"[BornSchedule] rho tail not converged (CV={np.std(tail_vals)/np.mean(tail_vals):.2f}). "
#             f"Extend rho_s range in estimate_model_stats.py.",
#             UserWarning,
#         )
#     denom   = max(1.0 - rho_infty, 1e-8)
#     rho_res = np.clip((rho_vals - rho_infty) / denom, 0.0, 1.0)
#     cross   = np.where(rho_res <= 1.0 / math.e)[0]
#     ell_corr = float(rho_s[cross[0]]) if len(cross) > 0 else float(rho_s[-1])

#     return rho_infty, ell_corr

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
        plateau_start = int(0.5 * len(rho_vals))  # fallback

    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)  # default: no slow decay
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

def _alpha_to_lambdas(alpha: np.ndarray,
                      lam0: float,
                      lamN: float) -> np.ndarray:
    """
    alpha ∈ R^N (unconstrained)
    h_i = softmax(alpha)_i · (lamN - lam0)   
    lambdas = [lam0, lam0+h_1, lam0+h_1+h_2, ..., lamN]  
    """
    span = lamN - lam0
    a    = alpha - alpha.max()          
    h    = np.exp(a) / np.exp(a).sum() * span
    interior = lam0 + np.cumsum(h)[:-1]
    return np.concatenate([[lam0], interior, [lamN]])

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
                "lambda_min", "lambda_max", "rho_s", "rho_values", "rho_s_gpp", "rho_values_gpp"]
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
            "sigma2_gp_values (first-derivative proxy). Re-run "
            "estimate_model_stats.py with --estimate_gpp for accuracy.",
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
    """
    BornSchedule v3 — rank-1 decomposed cost with residual kernel and D_j.

    Drop-in for diffusers schedulers: implements set_timesteps(N).
    """

    order = 2   # DPM-Solver-2

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
        ckpt_dir: Optional[str] = None,  
        ckpt_every: int = 500,            
        verbose: bool = True,
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

        stats = stats_override if stats_override is not None else self._load_stats(model_name)
        self._init_from_stats(stats)

        self.timesteps:           Optional[torch.Tensor] = None
        self.sigmas:              Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int]          = None
        self._lambdas_opt:        Optional[np.ndarray]   = None

        self.w_rank1    = w_rank1
        self.w_vres     = w_vres
        self.w_disc     = w_disc
        self.ckpt_dir   = ckpt_dir
        self.ckpt_every = ckpt_every

    # ------------------------------------------------------------------
    # Stats loading
    # ------------------------------------------------------------------

    def _load_stats(self, model_name: str) -> dict:
        safe = model_name.replace("/", "--").replace(":", "--")
        for base in _STATS_SEARCH_PATHS:
            fpath = base / f"{safe}.npz"
            if fpath.exists():
                warnings.warn(
                    f"[BornSchedule] Loading stats from {fpath}",
                    UserWarning, stacklevel=3,
                )
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
        rho_s_gpp   = np.asarray(stats["rho_s_gpp"],   dtype=np.float64)
        rho_vals_gpp = np.asarray(stats["rho_values_gpp"], dtype=np.float64)

        self.lambda_min = float(stats["lambda_min"])
        self.lambda_max = float(stats["lambda_max"])

        # Scalar function callables — consistent interface throughout
        kw = dict(kind="linear", bounds_error=False)
        self._g_fn         = interp1d(lam_grid, g_vals,
                                      fill_value=(g_vals[0],          g_vals[-1]),          **kw)
        self._sigma2_fn    = interp1d(lam_grid, sigma2_vals,
                                      fill_value=(sigma2_vals[0],     sigma2_vals[-1]),     **kw)
        self._sigma2gpp_fn = interp1d(lam_grid, sigma2_gpp_vals,
                                      fill_value=(sigma2_gpp_vals[0], sigma2_gpp_vals[-1]), **kw)
        
        # ρ_∞ and φ^res
        self.rho_infty, self.ell_corr = _extract_rho_infty_and_ell(rho_s, rho_vals)
        self._phi_res_fn = _build_phi_res_fn(
            rho_s, rho_vals, self.rho_infty, quad_pts=self.quad_pts,
        )

        from scipy.integrate import trapezoid
        rho_norm = rho_vals_gpp / rho_vals_gpp[0]
        self.ell_gpp  = float(trapezoid(rho_norm, rho_s_gpp))  # 0.1~0.3

        if self.verbose:
            print(f"  [BornSchedule] ρ_∞={self.rho_infty:.4f}  "
                  f"ℓ_corr={self.ell_corr:.4f}  "
                  f"λ∈[{self.lambda_min:.3f},{self.lambda_max:.3f}]")

    # Public wrappers (safe float → float)
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
    # Cost / gradient helpers (bind self's callables)
    # ------------------------------------------------------------------

    def _cost(self, lambdas: np.ndarray, barrier_weight: float = 0.0) -> float:
        return _cost_functional(
            lambdas,
            g_fn=self.g_fn, sigma2_fn=self.sigma2_fn,
            sigma2_gpp_fn=self.sigma2_gpp_fn, ell_gpp=self.ell_gpp,
            phi_res_fn=self._phi_res_fn, rho_infty=self.rho_infty,
            r1=self.r1, n_quad=self.quad_pts,
            barrier_weight=barrier_weight,
            w_rank1=self.w_rank1, w_vres=self.w_vres, w_disc=self.w_disc,
        )

    def _grad(self, interior, lam0, lamN, barrier_weight):
        grad = np.zeros_like(interior)
        kw = dict(
            g_fn=self.g_fn, sigma2_fn=self.sigma2_fn,
            sigma2_gpp_fn=self.sigma2_gpp_fn, phi_res_fn=self._phi_res_fn,
            ell_gpp=self.ell_gpp, rho_infty=self.rho_infty, r1=self.r1,
            barrier_weight=barrier_weight,
            w_rank1=self.w_rank1, w_vres=self.w_vres, w_disc=self.w_disc,
        )
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
                 lambdas=lams,
                 sigmas=self.lambda_to_sigma(lams),
                 cost=np.array(cost),
                 iter=np.array(it),
                 w_rank1=np.array(self.w_rank1),
                 w_vres=np.array(self.w_vres),
                 w_disc=np.array(self.w_disc))
        
        json_path = os.path.join(self.ckpt_dir, f"ckpt_iter{it:06d}.json")
        with open(json_path, "w") as f:
            json.dump({
                "iter": it,
                "cost": float(cost),
                "lambdas": lams.tolist(),
                "sigmas": self.lambda_to_sigma(lams).tolist(),
                "w_rank1": self.w_rank1,
                "w_vres":  self.w_vres,
                "w_disc":  self.w_disc,
            }, f, indent=2)
    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _optimise(self, N: int) -> np.ndarray:
        lam0      = self.lambda_min
        lamN      = self.lambda_max
        # ── 一次性诊断：打印均匀schedule下的中间量 ──────────────────────────
        lams_uniform = np.linspace(lam0, lamN, N + 1)
        print("\n=== DIAGNOSTIC: uniform schedule ===")
        for k in range(1, N + 1):
            lp = lams_uniform[k-1]
            lc = lams_uniform[k]
            h      = lc - lp
            lam_s  = lp + self.r1 * h
            I_full = math.exp(-lp) - math.exp(-lc)
            I_half = math.exp(-lp) - math.exp(-lam_s)
            al_p   = _alpha(lp)
            al_s   = _alpha(lam_s)
            a_k    = (1.0 - 1.0/(2.0*self.r1)) * al_p * I_full
            b_k    = -(al_s/(2.0*self.r1)) * I_full
            phi_k  = 1.0 + al_p * I_half * self.g_fn(lp)
            g_lp   = self.g_fn(lp)
            g_ls   = self.g_fn(lam_s)
            eps    = a_k * g_lp + b_k * g_ls * phi_k
            print(f"  step{k}: lp={lp:.3f} lc={lc:.3f} h={h:.3f} "
                f"I_full={I_full:.4f} a_k={a_k:.4f} b_k={b_k:.4f} "
                f"g(lp)={g_lp:.4f} g(ls)={g_ls:.4f} phi_k={phi_k:.4f} "
                f"eps={eps:.4f}")
        print("=====================================\n")
        # ────────────────────────────────────────────────────────────────────
        cost_uniform   = self._cost(np.linspace(lam0, lamN, N + 1), barrier_weight=0.0)
        # barrier_weight = self.barrier_weight_scale * cost_uniform / N
        barrier_weight = self.barrier_weight_scale * (lamN - lam0) / N
        if self.verbose:
            print(f"  [BornSchedule] N={N}  C(uniform)={cost_uniform:.4e}  "
                f"barrier_weight={barrier_weight:.3e}")
        # h_uniform = (lamN - lam0) / N
        # min_gap   = h_uniform / 5.0

        # interior = np.linspace(lam0, lamN, N + 1)[1:-1].copy()
        # grad = self._grad(interior, self.lambda_min, self.lambda_max, barrier_weight=0.0)
        # print(f"gradient at uniform_logSNR: {grad}")
        # print(f"gradient norm: {np.linalg.norm(grad):.6f}")

        # # Auto-scale barrier: C(uniform) / N  × scale
        # # At uniform schedule the log-barrier term = 0, so this only affects
        # # the curvature near degenerate boundaries, not the interior optimum.
        # cost_uniform = self._cost(np.linspace(lam0, lamN, N + 1), barrier_weight=0.0)
        # barrier_weight = self.barrier_weight_scale * cost_uniform / N
        # if self.verbose:
        #     print(f"  [BornSchedule] N={N}  C(uniform)={cost_uniform:.4e}  "
        #           f"barrier_weight={barrier_weight:.3e}")

        # m, v     = np.zeros_like(interior), np.zeros_like(interior)
        # lr       = self.lr
        # best_cost     = np.inf
        # best_interior = interior.copy()

        # for it in range(1, self.max_iter + 1):
        #     grad     = self._grad(interior, lam0, lamN, barrier_weight)
        #     m        = self.beta1 * m + (1.0 - self.beta1) * grad
        #     v        = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
        #     mh       = m / (1.0 - self.beta1 ** it)
        #     vh       = v / (1.0 - self.beta2 ** it)
        #     interior = interior - lr * mh / (np.sqrt(vh) + 1e-8)
        #     interior = _project_ordering(interior, lam0, lamN, min_gap)

        #     cost = self._cost(np.concatenate([[lam0], interior, [lamN]]), barrier_weight)
        #     if cost < best_cost:
        #         best_cost, best_interior = cost, interior.copy()

        #     if self.ckpt_dir and (it % self.ckpt_every == 0):
        #         self._save_ckpt(it, interior, lam0, lamN, cost)
                
        #     gnorm = float(np.linalg.norm(grad))
        #     if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
        #         print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  lr={lr:.4e}")
        #     if gnorm < self.tol:
        #         if self.verbose:
        #             print(f"  converged  iter={it}  |g|={gnorm:.3e}")
        #         break
        #     lr *= self.lr_decay

        # lams_best = np.concatenate([[lam0], best_interior, [lamN]])

        # if self.verbose:
        #     self._print_cost_breakdown(lams_best)

        # self._lambdas_opt = lams_best

        # return lams_best
        # softmax 参数：alpha=0 → 均匀步长
        alpha = np.zeros(N)
        m     = np.zeros_like(alpha)
        v     = np.zeros_like(alpha)
        lr    = self.lr
        best_cost  = np.inf
        best_alpha = alpha.copy()

        for it in range(1, self.max_iter + 1):
            # FD gradient 在 alpha 空间
            grad = np.zeros_like(alpha)
            fd_h = self.fd_eps          # alpha空间扰动，1e-3量级
            kw = dict(
                g_fn=self.g_fn, sigma2_fn=self.sigma2_fn,
                sigma2_gpp_fn=self.sigma2_gpp_fn, phi_res_fn=self._phi_res_fn,
                ell_gpp=self.ell_gpp, rho_infty=self.rho_infty, r1=self.r1,
                barrier_weight=barrier_weight,
                w_rank1=self.w_rank1, w_vres=self.w_vres, w_disc=self.w_disc,
            )
            for i in range(N):
                ap = alpha.copy(); ap[i] += fd_h
                am = alpha.copy(); am[i] -= fd_h
                Cp = _cost_functional(_alpha_to_lambdas(ap, lam0, lamN), **kw)
                Cm = _cost_functional(_alpha_to_lambdas(am, lam0, lamN), **kw)
                grad[i] = (Cp - Cm) / (2.0 * fd_h)

            m  = self.beta1 * m + (1.0 - self.beta1) * grad
            v  = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            mh = m / (1.0 - self.beta1 ** it)
            vh = v / (1.0 - self.beta2 ** it)
            alpha = alpha - lr * mh / (np.sqrt(vh) + 1e-8)
            lr   *= self.lr_decay

            lambdas = _alpha_to_lambdas(alpha, lam0, lamN)
            cost    = _cost_functional(lambdas, **kw)
            if cost < best_cost:
                best_cost  = cost
                best_alpha = alpha.copy()

            gnorm = float(np.linalg.norm(grad))
            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
                h = np.diff(lambdas)
                print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  "
                    f"h_cv={h.std()/h.mean():.3f}  lr={lr:.4e}")

            if gnorm < self.tol:
                if self.verbose:
                    print(f"  converged  iter={it}  |g|={gnorm:.3e}")
                break
            # # 诊断：打印每步的 eps, Gamma, V_j, D_j
            # lams = lambdas
            # N = len(lams) - 1
            # eps_arr = np.array([_compute_epsilon(lams[k-1], lams[k], self.g_fn, self.r1)
            #                     for k in range(1, N+1)])
            # Gamma = _compute_gamma(eps_arr, lams)
            # Vres  = np.array([_compute_V_res(lams[k-1], lams[k], self.g_fn, self.sigma2_fn,
            #                                 self._phi_res_fn, self.r1)
            #                 for k in range(1, N+1)])
            # D     = np.array([_compute_D_j(lams[k-1], lams[k], self.g_fn, self.sigma2_gpp_fn,
            #                                 self.ell_gpp, self.r1)
            #                 for k in range(1, N+1)])
            # print("eps:  ", np.round(eps_arr, 4))
            # print("Gamma:", np.round(Gamma,   4))
            # print("V_j:  ", np.round(Vres,    6))
            # print("D_j:  ", np.round(D,       8))
            # print("sigma2_eta at each lam_bar:",
            #     [round(self.sigma2_fn(0.5*(lams[k]+lams[k+1])), 6) for k in range(N)])
        lams_best = _alpha_to_lambdas(best_alpha, lam0, lamN)
        if self.verbose:
            self._print_cost_breakdown(lams_best)
        self._lambdas_opt = lams_best
        return lams_best
    
    def _optimise_with_hooks(
        self,
        N: int,
        capture_steps: int = 10,
    ) -> np.ndarray:

        lam0 = self.lambda_min
        lamN = self.lambda_max

        # ---------- alpha parameterization ----------
        alpha = np.zeros(N)
        m     = np.zeros_like(alpha)
        v     = np.zeros_like(alpha)
        lr    = self.lr

        # ---------- init ----------
        cost_uniform   = self._cost(np.linspace(lam0, lamN, N + 1), barrier_weight=0.0)
        # barrier_weight = self.barrier_weight_scale * cost_uniform / N
        barrier_weight = self.barrier_weight_scale * (lamN - lam0) / N

        if self.verbose:
            print(f"  [BornSchedule-α] N={N}  C(uniform)={cost_uniform:.4e}  "
                f"barrier_weight={barrier_weight:.3e}")

        best_cost  = np.inf
        best_alpha = alpha.copy()

        grad_history: list[float] = []

        # ---------- kwargs ----------
        kw = dict(
            g_fn=self.g_fn,
            sigma2_fn=self.sigma2_fn,
            sigma2_gpp_fn=self.sigma2_gpp_fn,
            phi_res_fn=self._phi_res_fn,
            ell_gpp=self.ell_gpp,
            rho_infty=self.rho_infty,
            r1=self.r1,
            barrier_weight=barrier_weight,
            w_rank1=self.w_rank1,
            w_vres=self.w_vres,
            w_disc=self.w_disc,
        )

        for it in range(1, self.max_iter + 1):

            # ---------- FD gradient (alpha space) ----------
            grad = np.zeros_like(alpha)
            fd_h = self.fd_eps

            for i in range(N):
                ap = alpha.copy(); ap[i] += fd_h
                am = alpha.copy(); am[i] -= fd_h

                Cp = _cost_functional(_alpha_to_lambdas(ap, lam0, lamN), **kw)
                Cm = _cost_functional(_alpha_to_lambdas(am, lam0, lamN), **kw)

                grad[i] = (Cp - Cm) / (2.0 * fd_h)

            gnorm = float(np.linalg.norm(grad))
            grad_history.append(gnorm)

            # ---------- Adam ----------
            m  = self.beta1 * m + (1.0 - self.beta1) * grad
            v  = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            mh = m / (1.0 - self.beta1 ** it)
            vh = v / (1.0 - self.beta2 ** it)

            alpha = alpha - lr * mh / (np.sqrt(vh) + 1e-8)

            # ---------- evaluate ----------
            lambdas = _alpha_to_lambdas(alpha, lam0, lamN)
            cost    = _cost_functional(lambdas, **kw)

            if cost < best_cost:
                best_cost  = cost
                best_alpha = alpha.copy()

            # ---------- checkpoint ----------
            if self.ckpt_dir and (it % self.ckpt_every == 0):
                self._save_ckpt(it, lambdas[1:-1], lam0, lamN, cost)

            # ---------- logging ----------
            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
                h = np.diff(lambdas)
                print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  "
                    f"h_cv={h.std()/h.mean():.3f}  lr={lr:.4e}")

            # ---------- stop ----------
            if gnorm < self.tol:
                if self.verbose:
                    print(f"  converged  iter={it}  |g|={gnorm:.3e}")
                break

            lr *= self.lr_decay

        # ---------- final ----------
        lams_best = _alpha_to_lambdas(best_alpha, lam0, lamN)

        if self.ckpt_dir:
            self._save_ckpt(-1, lams_best[1:-1], lam0, lamN, best_cost)

        if self.verbose:
            self._print_cost_breakdown(lams_best)

        # ---------- stats ----------
        self.grad_history    = grad_history
        self.final_grad_norm = grad_history[-1] if grad_history else float("nan")
        self.n_iter_actual   = len(grad_history)
        self.converged       = (grad_history[-1] < self.tol) if grad_history else False
        self._lambdas_opt    = lams_best

        return lams_best
    
    # def _optimise_with_hooks(
    #     self,
    #     N: int,
    #     capture_steps: int = 10,   
    # ) -> np.ndarray:
    #     """
    #     same as _optimise
    #     """
    #     lam0      = self.lambda_min
    #     lamN      = self.lambda_max
    #     h_uniform = (lamN - lam0) / N
    #     min_gap   = h_uniform / 5.0
 
    #     interior = np.linspace(lam0, lamN, N + 1)[1:-1].copy()
 
    #     cost_uniform   = self._cost(np.linspace(lam0, lamN, N + 1), barrier_weight=0.0)
    #     barrier_weight = self.barrier_weight_scale * cost_uniform / N
 
    #     if self.verbose:
    #         print(f"  [BornSchedule] N={N}  C(uniform)={cost_uniform:.4e}  "
    #               f"barrier_weight={barrier_weight:.3e}")
 
    #     m, v  = np.zeros_like(interior), np.zeros_like(interior)
    #     lr    = self.lr
 
    #     best_cost     = np.inf
    #     best_interior = interior.copy()
 
    #     grad_history: list[float] = []
 
    #     for it in range(1, self.max_iter + 1):
    #         grad = self._grad(interior, lam0, lamN, barrier_weight)
    #         gnorm = float(np.linalg.norm(grad))
    #         grad_history.append(gnorm)
 
    #         m  = self.beta1 * m + (1.0 - self.beta1) * grad
    #         v  = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
    #         mh = m / (1.0 - self.beta1 ** it)
    #         vh = v / (1.0 - self.beta2 ** it)
    #         interior = interior - lr * mh / (np.sqrt(vh) + 1e-8)
    #         interior = _project_ordering(interior, lam0, lamN, min_gap)
 
    #         cost = self._cost(np.concatenate([[lam0], interior, [lamN]]),
    #                           barrier_weight)
    #         if cost < best_cost:
    #             best_cost, best_interior = cost, interior.copy()
 
    #         # checkpoint
    #         if self.ckpt_dir and (it % self.ckpt_every == 0):
    #             self._save_ckpt(it, interior, lam0, lamN, cost)
 
    #         if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == 1):
    #             print(f"  iter {it:4d}  cost={cost:.4e}  |g|={gnorm:.3e}  lr={lr:.4e}")
 
    #         if gnorm < self.tol:
    #             if self.verbose:
    #                 print(f"  converged  iter={it}  |g|={gnorm:.3e}")
    #             break
 
    #         lr *= self.lr_decay
 
    #     lams_best = np.concatenate([[lam0], best_interior, [lamN]])
 
    #     if self.ckpt_dir:
    #         self._save_ckpt(-1, best_interior, lam0, lamN, best_cost)
 
    #     if self.verbose:
    #         self._print_cost_breakdown(lams_best)
 
    #     self.grad_history    = grad_history
    #     self.final_grad_norm = grad_history[-1] if grad_history else float("nan")
    #     self.n_iter_actual   = len(grad_history)
    #     self.converged       = (grad_history[-1] < self.tol) if grad_history else False
    #     self._lambdas_opt    = lams_best
 
    #     return lams_best
    
    def _print_cost_breakdown(self, lams: np.ndarray) -> None:
        N     = len(lams) - 1
        eps   = np.array([_compute_epsilon(lams[k-1], lams[k], self.g_fn, self.r1)
                           for k in range(1, N+1)])
        Gamma = _compute_gamma(eps, lams)
        # A     = np.array([_compute_A_j(lams[k-1], lams[k], self.sigma2_fn, self.quad_pts)
        #                   for k in range(1, N+1)])
        A     = np.array([_compute_A_j(lams[k-1], lams[k], self.sigma2_fn, self.g_fn, self.r1, self.quad_pts)
                          for k in range(1, N+1)])
        Vres  = np.array([_compute_V_res(lams[k-1], lams[k], self.g_fn, self.sigma2_fn,
                                         self._phi_res_fn, self.r1)
                          for k in range(1, N+1)])
        D     = np.array([_compute_D_j(lams[k-1], lams[k], self.g_fn, self.sigma2_gpp_fn, self.ell_gpp, self.r1)
                          for k in range(1, N+1)])
        a2    = _alpha(lams[-1]) ** 2
        print(f"  Cost breakdown (× α²_{{t_N}} = {a2:.4f}):")
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
        """(V_j^res + D_j) · Γ_j² — uniform at KKT optimum."""
        if self._lambdas_opt is None:
            return None
        lams  = self._lambdas_opt
        N     = len(lams) - 1
        eps   = np.array([_compute_epsilon(lams[k-1], lams[k], self.g_fn, self.r1)
                           for k in range(1, N+1)])
        Gamma = _compute_gamma(eps, lams)
        Vres  = np.array([_compute_V_res(lams[k-1], lams[k], self.g_fn, self.sigma2_fn,
                                         self._phi_res_fn, self.r1)
                          for k in range(1, N+1)])
        D     = np.array([_compute_D_j(lams[k-1], lams[k], self.g_fn, self.sigma2_gpp_fn, self.ell_gpp, self.r1)
                          for k in range(1, N+1)])
        return (Vres + D) * Gamma ** 2

    def __repr__(self) -> str:
        return (f"OptimalSchedule(model='{self.model_name}', r1={self.r1}, "
                f"ρ_∞={self.rho_infty:.3f}, ℓ_corr={self.ell_corr:.4f}, "
                f"steps={self.num_inference_steps})")

