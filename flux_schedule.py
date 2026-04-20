"""
flux_schedule.py  —  BornSchedule optimal timestep scheduler for FLUX FM models.

Cost functional:

  C = w_rank1 · ρ_∞ · (Σ_j γ_j · a_j)²      [rank-1, score-error bias]
    + w_vres  ·        Σ_j γ_j² · V_j^res    [residual score variance]
    + w_disc  ·        Σ_j γ_j² · D_j        [discretisation variance]

"""

import os, math, warnings
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d


# ══════════════════════════════════════════════════════════════════════════════
# φ / φ^res builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_phi_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    quad_pts: int = 64,
    s_extend_factor: float = 3.0,
) -> Callable[[float, float], float]:
    """φ(a,b) = ∫₀ᵃ ∫₀ᵇ ρ(|u−u'|) du' du  built from empirical (rho_s, rho_vals)."""
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
        slope = ((log_rho_fine[last] - log_rho_fine[max(last-1, 0)])
                 / (s_fine[1] - s_fine[0] + 1e-30))
        slope = min(slope, -1e-3)
        for idx in np.where(nan_mask)[0]:
            log_rho_fine[idx] = log_rho_fine[last] + slope*(s_fine[idx]-s_fine[last])
    rho_fine = np.exp(np.clip(log_rho_fine, -30.0, 0.0))

    ds     = s_fine[1] - s_fine[0]
    R_fine = np.zeros(n_fine)
    R_fine[1:] = np.cumsum(0.5*(rho_fine[:-1]+rho_fine[1:])*ds)
    R_interp = interp1d(s_fine, R_fine, kind="linear", bounds_error=False,
                        fill_value=(0.0, float(R_fine[-1])))

    _trapz = getattr(np, "trapezoid", None) or np.trapz

    def phi_fn(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 0.0
        a_int = min(a, b); b_int = max(a, b)
        u = np.linspace(0.0, a_int, quad_pts)
        return float(_trapz(R_interp(u) + R_interp(b_int - u), u))

    return phi_fn


def _build_phi_res_fn(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
    rho_infty: float,
    quad_pts: int = 64,
) -> Callable[[float, float], float]:
    """
    φ^res(a,b) = φ(a,b) − ρ_∞ · a · b

    Subtracts the rank-1 plateau analytically, preserving h² curvature:
        φ^res(h,h) ≈ (1−ρ_∞) h²  as h→0
    """
    phi_full = _build_phi_fn(rho_s, rho_vals, quad_pts)
    def phi_res_fn(a: float, b: float) -> float:
        return phi_full(a, b) - rho_infty * a * b
    return phi_res_fn


# ══════════════════════════════════════════════════════════════════════════════
# FM Euler propagator
# ══════════════════════════════════════════════════════════════════════════════

def _compute_epsilon_fm(sigma_prev: float, sigma_curr: float,
                        g_fn: Callable) -> float:
    """
    ε_k = (σ_k − σ_{k-1}) · g(σ_{k-1})   [< 0 since σ_k < σ_{k-1}]

    g(σ) = (1/d) Tr(∇_x v_θ) ≈ 0 for well-trained FM → γ_j ≈ 1.
    Kept for theoretical completeness.
    """
    return (sigma_curr - sigma_prev) * g_fn(sigma_prev)


def _compute_gamma(eps: np.ndarray) -> np.ndarray:
    """Γ[j] = ∏_{k=j+1}^{N} (1 + ε[k]),   Γ[N-1] = 1."""
    N     = len(eps)
    Gamma = np.ones(N)
    for k in range(N-2, -1, -1):
        Gamma[k] = Gamma[k+1] * (1.0 + eps[k+1])
    return Gamma


# ══════════════════════════════════════════════════════════════════════════════
# Per-step cost components
# ══════════════════════════════════════════════════════════════════════════════

def _compute_a_j(sigma_prev: float, sigma_curr: float,
                 sigma2_fn: Callable, n_quad: int = 32) -> float:
    """
    a_j = ∫_{σ_j}^{σ_{j-1}} σ_η(τ) dτ

    Rank-1 score-error weight.  No VP-SDE prefactors: FM error is in latent units.
    """
    tau     = np.linspace(sigma_curr, sigma_prev, n_quad)
    sig_eta = np.sqrt(np.maximum([sigma2_fn(float(t)) for t in tau], 0.0))
    return float((getattr(np, "trapezoid", None) or np.trapz)(sig_eta, tau))


def _compute_V_res_fm(sigma_prev: float, sigma_curr: float,
                      sigma2_fn: Callable,
                      phi_res_fn: Callable) -> float:
    """
    V_j^res = σ²_η(σ̄_j) · φ^res(h_j, h_j)

    Residual (banded-kernel) score-error variance.
    φ^res(h,h) ≈ (1−ρ_∞)h² preserves h² curvature → no linear-objective collapse.
    """
    h_abs   = sigma_prev - sigma_curr
    sig_bar = 0.5 * (sigma_prev + sigma_curr)
    s2      = max(sigma2_fn(sig_bar), 0.0)
    Q       = phi_res_fn(h_abs, h_abs)
    return s2 * max(Q, 0.0)


def _compute_D_j(sigma_prev: float, sigma_curr: float,
                 sigma2_vdot_fn: Callable) -> float:
    """
    D_j = (h_j⁴ / 4) · σ²_{v̇}(σ_{j-1})

    Discretisation variance.  Derivation: FM Euler local truncation error
        f_j^disc ≈ (h²/2) · v̇_θ(σ_{j-1})
    →   E‖f_j^disc‖²/d = (h⁴/4) · σ²_{v̇}(σ_{j-1})

    ρ_∞^d ≈ 0 empirically: v̇ errors decorrelate across steps,
    so no rank-1 disc term appears in the cost functional.
    """
    h_abs = sigma_prev - sigma_curr
    return (h_abs**4 / 4.0) * max(sigma2_vdot_fn(sigma_prev), 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Full cost functional
# ══════════════════════════════════════════════════════════════════════════════

def _cost_functional(
    sigmas: np.ndarray,          # decreasing: [σ_max, …, σ_min]
    g_fn: Callable,
    sigma2_fn: Callable,
    sigma2_vdot_fn: Callable,
    phi_res_fn: Callable,
    rho_infty: float,
    n_quad: int = 32,
    barrier_weight: float = 0.0,
    w_rank1: float = 1.0,
    w_vres:  float = 1.0,
    w_disc:  float = 1.0,
) -> float:
    """
    C = w_rank1 · ρ_∞ · (Σ_j γ_j a_j)²
      + w_vres  ·        Σ_j γ_j² V_j^res
      + w_disc  ·        Σ_j γ_j² D_j
      [+ log-barrier if barrier_weight > 0]
    """
    N = len(sigmas) - 1

    eps   = np.array([_compute_epsilon_fm(sigmas[k-1], sigmas[k], g_fn)
                      for k in range(1, N+1)])
    Gamma = _compute_gamma(eps)

    a_arr = np.array([_compute_a_j(sigmas[k-1], sigmas[k], sigma2_fn, n_quad)
                      for k in range(1, N+1)])
    V_arr = np.array([_compute_V_res_fm(sigmas[k-1], sigmas[k], sigma2_fn, phi_res_fn)
                      for k in range(1, N+1)])
    D_arr = np.array([_compute_D_j(sigmas[k-1], sigmas[k], sigma2_vdot_fn)
                      for k in range(1, N+1)])

    rank1_term = w_rank1 * rho_infty * float(np.dot(a_arr, Gamma))**2
    vres_term  = w_vres  * float(np.dot(V_arr, Gamma**2))
    disc_term  = w_disc  * float(np.dot(D_arr, Gamma**2))
    main_cost  = rank1_term + vres_term + disc_term

    if barrier_weight > 0.0:
        h_abs     = sigmas[:-1] - sigmas[1:]          # positive
        h_uniform = (sigmas[0] - sigmas[-1]) / N
        main_cost -= barrier_weight * float(np.sum(np.log(h_abs / h_uniform)))

    return main_cost


# ══════════════════════════════════════════════════════════════════════════════
# Gradient (numerical FD) and ordering projection
# ══════════════════════════════════════════════════════════════════════════════

def _numerical_gradient(
    interior: np.ndarray,
    sigma_max: float,
    sigma_min: float,
    g_fn, sigma2_fn, sigma2_vdot_fn, phi_res_fn,
    rho_infty: float,
    n_quad: int,
    barrier_weight: float,
    w_rank1: float, w_vres: float, w_disc: float,
    fd_h: float = 1e-5,
) -> np.ndarray:
    kw = dict(g_fn=g_fn, sigma2_fn=sigma2_fn, sigma2_vdot_fn=sigma2_vdot_fn,
              phi_res_fn=phi_res_fn, rho_infty=rho_infty,
              n_quad=n_quad, barrier_weight=barrier_weight,
              w_rank1=w_rank1, w_vres=w_vres, w_disc=w_disc)
    grad = np.zeros_like(interior)
    for i in range(len(interior)):
        xp = interior.copy(); xp[i] += fd_h
        xm = interior.copy(); xm[i] -= fd_h
        Cp = _cost_functional(np.concatenate([[sigma_max], xp, [sigma_min]]), **kw)
        Cm = _cost_functional(np.concatenate([[sigma_max], xm, [sigma_min]]), **kw)
        grad[i] = (Cp - Cm) / (2.0 * fd_h)
    return grad


# def _project_ordering(
#     interior: np.ndarray,
#     sigma_max: float,
#     sigma_min: float,
#     min_gap: float,
# ) -> np.ndarray:
#     """Enforce strict decreasing order with min_gap from boundaries."""
#     x = -np.sort(-interior.copy())   # descending
#     n = len(x)
#     for i in range(n):
#         lo = sigma_min + (n - i) * min_gap
#         hi = sigma_max - (i + 1) * min_gap
#         x[i] = float(np.clip(x[i], lo, hi))
#     return x

# ── Softmax Reparameterization ──────────────────
def _alpha_to_sigmas(alpha: np.ndarray,
                     sigma_max: float,
                     sigma_min: float) -> np.ndarray:
    """
    alpha ∈ R^N  (unconstrained)
    h_i = softmax(alpha)_i · (sigma_max - sigma_min)   
    sigmas = [sigma_max, sigma_max - h_1, ..., sigma_min]   
    """
    span = sigma_max - sigma_min
    a    = alpha - alpha.max()       
    h    = np.exp(a) / np.exp(a).sum() * span
    interior = sigma_max - np.cumsum(h)[:-1]
    return np.concatenate([[sigma_max], interior, [sigma_min]])

def _grad_alpha(
    alpha: np.ndarray,
    sigma_max: float,
    sigma_min: float,
    barrier_weight: float,
    fd_h: float,
    **cost_kw,
) -> np.ndarray:
    """FD gradient w.r.t. alpha."""
    grad = np.zeros_like(alpha)
    for i in range(len(alpha)):
        ap = alpha.copy(); ap[i] += fd_h
        am = alpha.copy(); am[i] -= fd_h
        cp = _cost_functional(_alpha_to_sigmas(ap, sigma_max, sigma_min),
                              barrier_weight=barrier_weight, **cost_kw)
        cm = _cost_functional(_alpha_to_sigmas(am, sigma_max, sigma_min),
                              barrier_weight=barrier_weight, **cost_kw)
        grad[i] = (cp - cm) / (2.0 * fd_h)
    return grad

# ══════════════════════════════════════════════════════════════════════════════
# Stats helper
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_infty(
    rho_s: np.ndarray,
    rho_vals: np.ndarray,
) -> float:
    drho     = np.abs(np.diff(rho_vals))
    rel_drho = drho / (np.abs(rho_vals[:-1]) + 1e-8)
    MIN_FLAT = max(5, len(rho_vals) // 10)

    plateau_start = None
    for i in range(len(rel_drho) - MIN_FLAT + 1):
        if np.all(rel_drho[i : i + MIN_FLAT] < 0.02):
            plateau_start = i
            break
    if plateau_start is None:
        warnings.warn("[flux_schedule] No plateau found — extend rho_s range.", UserWarning)
        plateau_start = len(rho_vals) // 2

    rho_at_start = rho_vals[plateau_start]
    plateau_end  = len(rho_vals)
    for i in range(plateau_start + MIN_FLAT, len(rho_vals)):
        if rho_vals[i] < rho_at_start * 0.95:
            plateau_end = i
            break

    plateau_vals = rho_vals[plateau_start:plateau_end]
    ri = float(np.clip(np.median(plateau_vals), 0.0, 1.0 - 1e-6))

    cv = np.std(plateau_vals) / (np.mean(plateau_vals) + 1e-8)
    if cv > 0.05:
        warnings.warn(
            f"[flux_schedule] Plateau CV={cv:.3f} — ρ_∞ estimate unreliable. "
            f"Detected plateau: indices [{plateau_start}, {plateau_end}), "
            f"s=[{rho_s[plateau_start]:.3f}, {rho_s[min(plateau_end, len(rho_s)-1)]:.3f}].",
            UserWarning,
        )
    return ri

def _load_stats(npz_path: str):
    """Load stats npz, return interpolators and rho_infty."""
    data   = np.load(npz_path)
    t_grid = data.get("t_grid",      data.get("lambda_grid")).astype(np.float64)
    s2_eta = data.get("sigma2_eta",  data.get("sigma2_values")).astype(np.float64)
    s2_vd  = data.get("sigma2_vdot_fd1", data.get("sigma2_gpp_values")).astype(np.float64)
    g_arr  = data.get("g_values").astype(np.float64)
    rho_s  = data.get("rho_s").astype(np.float64)
    rho_v  = data.get("rho_values").astype(np.float64)
    sigma_max = data.get("sigma_max").astype(np.float64)
    sigma_min = data.get("sigma_min").astype(np.float64)

    _s2    = PchipInterpolator(t_grid, s2_eta, extrapolate=True)
    _s2vd  = PchipInterpolator(t_grid, s2_vd,  extrapolate=True)
    _g     = PchipInterpolator(t_grid, g_arr,  extrapolate=True)

    def sigma2_fn(s):      return max(float(_s2(s)),   1e-10)
    def sigma2_vdot_fn(s): return max(float(_s2vd(s)), 1e-10)
    def g_fn(s):           return float(_g(s))

    rho_infty = _extract_rho_infty(rho_s, rho_v)
    phi_res   = _build_phi_res_fn(rho_s, rho_v, rho_infty, quad_pts=64)

    return sigma2_fn, sigma2_vdot_fn, g_fn, phi_res, rho_infty, sigma_max, sigma_min


# ══════════════════════════════════════════════════════════════════════════════
# Main optimizer
# ══════════════════════════════════════════════════════════════════════════════

def optimize_schedule(
    npz_path: str,
    nfe: int,
    sigma_max: float      = 0.97,
    sigma_min: float      = 0.02,
    n_steps: int          = 2000,
    lr: float             = 1e-3,
    lr_decay: float       = 0.995,
    min_gap: float        = 1e-4,
    barrier_weight: float = 0.0,
    fd_h: float           = 1e-3,
    n_quad: int           = 32,
    w_rank1: float        = 1.0,
    w_vres:  float        = 1.0,
    w_disc:  float        = 0.5,
    n_restarts: int       = 3,
    verbose: bool         = True,
) -> np.ndarray:
    """
    Returns optimal sigmas array (length nfe+1, strictly decreasing).
    Algorithm: Adam + logit reparametrisation + ordering projection, multi-restart.
    """
    sigma2_fn, sigma2_vdot_fn, g_fn, phi_res, rho_infty, sigma_max, sigma_min = _load_stats(npz_path)
    if verbose:
        print(f"  ρ_∞(η) = {rho_infty:.4f}  [disc rank-1 dropped: ρ_∞^d ≈ 0]")

    if barrier_weight == 0.0:
        barrier_weight = 1e-3 * nfe / 8.0   # NFE=8→1e-3, NFE=12→1.5e-3, NFE=20→2.5e-3

    cost_kw = dict(
        g_fn=g_fn, sigma2_fn=sigma2_fn, sigma2_vdot_fn=sigma2_vdot_fn,
        phi_res_fn=phi_res, rho_infty=rho_infty,
        n_quad=n_quad, barrier_weight=barrier_weight,
        w_rank1=w_rank1, w_vres=w_vres, w_disc=w_disc,
    )
    grad_kw = dict(sigma_max=sigma_max, sigma_min=sigma_min, fd_h=fd_h, **cost_kw)

    # span = sigma_max - sigma_min

    # def _to_sigma(u):
    #     return np.clip(sigma_min + span / (1.0 + np.exp(-u)),
    #                    sigma_min + 1e-6, sigma_max - 1e-6)

    # def _to_u(s):
    #     p = np.clip((s - sigma_min) / span, 1e-6, 1 - 1e-6)
    #     return np.log(p / (1.0 - p))

    # def _cost_u(u):
    #     interior = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
    #     return _cost_functional(np.concatenate([[sigma_max], interior, [sigma_min]]),
    #                             **cost_kw)

    # def _grad_u(u):
    #     s   = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
    #     raw = _numerical_gradient(s, **grad_kw)
    #     p   = (s - sigma_min) / span
    #     return raw * p * (1.0 - p) * span   # chain-rule through logistic

    # N_int      = nfe - 1
    best_cost  = float("inf")
    best_sigmas = None

    # for restart in range(n_restarts):
    #   init = np.linspace(sigma_max, sigma_min, nfe + 1)[1:-1]
    #   if restart > 0:
    #       init = _project_ordering(
    #           init + np.random.randn(N_int) * span * 0.05,
    #           sigma_max, sigma_min, min_gap)
    #     u  = _to_u(init)
    #     m  = np.zeros_like(u)
    #     v  = np.zeros_like(u)
    #     β1, β2, ε_a = 0.9, 0.999, 1e-8
    #     lr_t = lr; plateau = 0; cost_prev = float("inf")

    #     for step in range(1, n_steps + 1):
    #         g_vec = _grad_u(u)
    #         m     = β1*m + (1-β1)*g_vec
    #         v     = β2*v + (1-β2)*g_vec**2
    #         u    -= lr_t * (m/(1-β1**step)) / (np.sqrt(v/(1-β2**step)) + ε_a)
    #         lr_t *= lr_decay

    #         cost = _cost_u(u)
    #         if verbose and step % max(1, n_steps//10) == 0:
    #             interior = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
    #             eps_  = np.array([_compute_epsilon_fm(
    #                 np.concatenate([[sigma_max], interior, [sigma_min]])[k-1],
    #                 np.concatenate([[sigma_max], interior, [sigma_min]])[k], g_fn)
    #                 for k in range(1, nfe+1)])
    #             Gam_  = _compute_gamma(eps_)
    #             print(f"  [r{restart}] step {step:4d}  cost={cost:.4e}  "
    #                   f"γ∈[{Gam_.min():.3f},{Gam_.max():.3f}]")

    #         rel = abs(cost_prev - cost) / (abs(cost_prev) + 1e-12)
    #         plateau = (plateau + 1) if rel < 1e-7 else 0
    #         if plateau > 50:
    #             if verbose: print(f"  [r{restart}] early stop @ step {step}")
    #             break
    #         cost_prev = cost

    #     interior = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
    #     final_s  = np.concatenate([[sigma_max], interior, [sigma_min]])
    #     c        = _cost_functional(final_s, **cost_kw)
    #     if c < best_cost:
    #         best_cost   = c
    #         best_sigmas = final_s.copy()
    #         if verbose: print(f"  [r{restart}] ★ best={best_cost:.4e}")

    # return best_sigmas
    for restart in range(n_restarts):
        alpha = np.zeros(nfe)
        if restart > 0:
            alpha += np.random.randn(nfe) * 0.5   

        m  = np.zeros_like(alpha)
        v  = np.zeros_like(alpha)
        β1, β2, ε_a = 0.9, 0.999, 1e-8
        lr_t = lr; plateau = 0; cost_prev = float("inf")

        for step in range(1, n_steps + 1):
            g_vec = _grad_alpha(alpha, sigma_max, sigma_min,
                                barrier_weight, fd_h, **cost_kw)
            m   = β1*m + (1-β1)*g_vec
            v   = β2*v + (1-β2)*g_vec**2
            alpha -= lr_t * (m/(1-β1**step)) / (np.sqrt(v/(1-β2**step)) + ε_a)
            lr_t *= lr_decay

            sigmas = _alpha_to_sigmas(alpha, sigma_max, sigma_min)
            cost   = _cost_functional(sigmas, barrier_weight=barrier_weight, **cost_kw)

            if verbose and step % max(1, n_steps//10) == 0:
                h = sigmas[:-1] - sigmas[1:]
                print(f"  [r{restart}] step {step:4d}  cost={cost:.4e}  "
                      f"h∈[{h.min():.4f},{h.max():.4f}]  "
                      f"h_cv={h.std()/h.mean():.3f}")   # CV

            rel = abs(cost_prev - cost) / (abs(cost_prev) + 1e-12)
            plateau = (plateau + 1) if rel < 1e-7 else 0
            if plateau > 50:
                if verbose: print(f"  [r{restart}] early stop @ step {step}")
                break
            cost_prev = cost

        sigmas = _alpha_to_sigmas(alpha, sigma_max, sigma_min)
        c      = _cost_functional(sigmas, barrier_weight=barrier_weight, **cost_kw)
        if c < best_cost:
            best_cost   = c
            best_sigmas = sigmas.copy()
            if verbose: print(f"  [r{restart}] ★ best={best_cost:.4e}")

    return best_sigmas


