"""
flux_schedule.py  —  BornSchedule optimal timestep scheduler for FLUX FM models.

Cost functional (FM Euler, empirically validated):

  C = w_rank1 · ρ_∞ · (Σ_j γ_j · a_j)²      [rank-1, score-error bias]
    + w_vres  ·        Σ_j γ_j² · V_j^res    [residual score variance]
    + w_disc  ·        Σ_j γ_j² · D_j        [discretisation variance]

Note: the discretisation kernel ρ_∞^d ≈ 0 empirically (v̇ errors are
weakly correlated across steps), so the rank-1 disc term is dropped.
D_j enters only as a local diagonal term.

Definitions (FM / Euler):
  h_j   = σ_{j−1} − σ_j  > 0            (step size; σ decreasing)
  ε_k   = h_k · g(σ_{k−1}) · (−1)       (propagator correction; g≈0 for FM)
  γ_j   = ∏_{k=j+1}^{N} (1 + ε_k)       (scalar propagator ≈ 1)
  a_j   = ∫_{σ_j}^{σ_{j-1}} σ_η(τ) dτ   (rank-1 score-error weight)
  V_j^res = σ²_η(σ̄_j) · φ^res(h_j,h_j)  (residual score variance)
  D_j   = (h_j⁴/4) · σ²_{v̇}(σ_{j-1})   (discretisation variance)
  φ^res(a,b) = φ(a,b) − ρ_∞ · a · b      (residual kernel double-integral)
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
    D_j enters only as a local diagonal term Γ_j² · D_j.
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


def _project_ordering(
    interior: np.ndarray,
    sigma_max: float,
    sigma_min: float,
    min_gap: float,
) -> np.ndarray:
    """Enforce strict decreasing order with min_gap from boundaries."""
    x = -np.sort(-interior.copy())   # descending
    n = len(x)
    for i in range(n):
        lo = sigma_min + (n - i) * min_gap
        hi = sigma_max - (i + 1) * min_gap
        x[i] = float(np.clip(x[i], lo, hi))
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Importance scores  (segment-tree / Pareto sparsification)
# ══════════════════════════════════════════════════════════════════════════════

def compute_importance_scores(
    sigmas: np.ndarray,
    g_fn, sigma2_fn, sigma2_vdot_fn, phi_res_fn,
    w_vres: float = 1.0,
    w_disc: float = 1.0,
) -> np.ndarray:
    """
    I_k = γ_k² · (w_vres · V_k^res + w_disc · D_k)

    Greedy sparsification: remove step with smallest I_k to reduce NFE by 1.
    Because FM error is additive and γ≈1, I_k is approximately separable
    → O(N) greedy scan is near-optimal.
    """
    N   = len(sigmas) - 1
    eps = np.array([_compute_epsilon_fm(sigmas[k-1], sigmas[k], g_fn)
                    for k in range(1, N+1)])
    Gamma = _compute_gamma(eps)
    # Need phi_res for V_arr — caller must provide it
    V_arr = np.array([_compute_V_res_fm(sigmas[k-1], sigmas[k], sigma2_fn, phi_res_fn)
                      for k in range(1, N+1)])
    D_arr = np.array([_compute_D_j(sigmas[k-1], sigmas[k], sigma2_vdot_fn)
                      for k in range(1, N+1)])
    return Gamma**2 * (w_vres * V_arr + w_disc * D_arr)


# ══════════════════════════════════════════════════════════════════════════════
# Stats helper
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rho_infty(rho_s: np.ndarray, rho_vals: np.ndarray) -> float:
    tail_start = max(1, int(0.80 * len(rho_vals)))
    ri   = float(np.clip(np.median(rho_vals[tail_start:]), 0.0, 1.0 - 1e-6))
    tail = rho_vals[tail_start:]
    if np.std(tail) / (np.mean(tail) + 1e-8) > 0.1:
        warnings.warn("[flux_schedule] rho tail not converged — extend rho_s range.",
                      UserWarning)
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

    _s2    = PchipInterpolator(t_grid, s2_eta, extrapolate=True)
    _s2vd  = PchipInterpolator(t_grid, s2_vd,  extrapolate=True)
    _g     = PchipInterpolator(t_grid, g_arr,  extrapolate=True)

    def sigma2_fn(s):      return max(float(_s2(s)),   1e-10)
    def sigma2_vdot_fn(s): return max(float(_s2vd(s)), 1e-10)
    def g_fn(s):           return float(_g(s))

    rho_infty = _extract_rho_infty(rho_s, rho_v)
    phi_res   = _build_phi_res_fn(rho_s, rho_v, rho_infty, quad_pts=64)

    return sigma2_fn, sigma2_vdot_fn, g_fn, phi_res, rho_infty


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
    fd_h: float           = 1e-5,
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
    sigma2_fn, sigma2_vdot_fn, g_fn, phi_res, rho_infty = _load_stats(npz_path)
    if verbose:
        print(f"  ρ_∞(η) = {rho_infty:.4f}  [disc rank-1 dropped: ρ_∞^d ≈ 0]")

    cost_kw = dict(
        g_fn=g_fn, sigma2_fn=sigma2_fn, sigma2_vdot_fn=sigma2_vdot_fn,
        phi_res_fn=phi_res, rho_infty=rho_infty,
        n_quad=n_quad, barrier_weight=barrier_weight,
        w_rank1=w_rank1, w_vres=w_vres, w_disc=w_disc,
    )
    grad_kw = dict(sigma_max=sigma_max, sigma_min=sigma_min, fd_h=fd_h, **cost_kw)

    span = sigma_max - sigma_min

    def _to_sigma(u):
        return np.clip(sigma_min + span / (1.0 + np.exp(-u)),
                       sigma_min + 1e-6, sigma_max - 1e-6)

    def _to_u(s):
        p = np.clip((s - sigma_min) / span, 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    def _cost_u(u):
        interior = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
        return _cost_functional(np.concatenate([[sigma_max], interior, [sigma_min]]),
                                **cost_kw)

    def _grad_u(u):
        s   = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
        raw = _numerical_gradient(s, **grad_kw)
        p   = (s - sigma_min) / span
        return raw * p * (1.0 - p) * span   # chain-rule through logistic

    N_int      = nfe - 1
    best_cost  = float("inf")
    best_sigmas = None

    for restart in range(n_restarts):
        init = np.linspace(sigma_max, sigma_min, nfe + 1)[1:-1]
        if restart > 0:
            init = _project_ordering(
                init + np.random.randn(N_int) * span * 0.05,
                sigma_max, sigma_min, min_gap)
        u  = _to_u(init)
        m  = np.zeros_like(u)
        v  = np.zeros_like(u)
        β1, β2, ε_a = 0.9, 0.999, 1e-8
        lr_t = lr; plateau = 0; cost_prev = float("inf")

        for step in range(1, n_steps + 1):
            g_vec = _grad_u(u)
            m     = β1*m + (1-β1)*g_vec
            v     = β2*v + (1-β2)*g_vec**2
            u    -= lr_t * (m/(1-β1**step)) / (np.sqrt(v/(1-β2**step)) + ε_a)
            lr_t *= lr_decay

            cost = _cost_u(u)
            if verbose and step % max(1, n_steps//10) == 0:
                interior = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
                eps_  = np.array([_compute_epsilon_fm(
                    np.concatenate([[sigma_max], interior, [sigma_min]])[k-1],
                    np.concatenate([[sigma_max], interior, [sigma_min]])[k], g_fn)
                    for k in range(1, nfe+1)])
                Gam_  = _compute_gamma(eps_)
                print(f"  [r{restart}] step {step:4d}  cost={cost:.4e}  "
                      f"γ∈[{Gam_.min():.3f},{Gam_.max():.3f}]")

            rel = abs(cost_prev - cost) / (abs(cost_prev) + 1e-12)
            plateau = (plateau + 1) if rel < 1e-7 else 0
            if plateau > 50:
                if verbose: print(f"  [r{restart}] early stop @ step {step}")
                break
            cost_prev = cost

        interior = _project_ordering(_to_sigma(u), sigma_max, sigma_min, min_gap)
        final_s  = np.concatenate([[sigma_max], interior, [sigma_min]])
        c        = _cost_functional(final_s, **cost_kw)
        if c < best_cost:
            best_cost   = c
            best_sigmas = final_s.copy()
            if verbose: print(f"  [r{restart}] ★ best={best_cost:.4e}")

    return best_sigmas


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_schedule(
    sigmas: np.ndarray,
    npz_path: str,
    label: str = "BornSchedule",
    out_path: Optional[str] = None,
) -> dict:
    sigma2_fn, sigma2_vdot_fn, g_fn, phi_res, rho_infty = _load_stats(npz_path)

    N   = len(sigmas) - 1
    eps = np.array([_compute_epsilon_fm(sigmas[k-1], sigmas[k], g_fn)
                    for k in range(1, N+1)])
    Gam = _compute_gamma(eps)
    a_  = np.array([_compute_a_j(sigmas[k-1], sigmas[k], sigma2_fn)
                    for k in range(1, N+1)])
    V_  = np.array([_compute_V_res_fm(sigmas[k-1], sigmas[k], sigma2_fn, phi_res)
                    for k in range(1, N+1)])
    D_  = np.array([_compute_D_j(sigmas[k-1], sigmas[k], sigma2_vdot_fn)
                    for k in range(1, N+1)])
    I_  = Gam**2 * (V_ + D_)
    h_  = sigmas[:-1] - sigmas[1:]

    print(f"\n  ── {label}  NFE={N}  ρ_∞={rho_infty:.4f} ─────────────────────")
    print(f"  {'j':>3}  {'σ_prev':>7}  {'σ_curr':>7}  {'h_j':>7}  "
          f"{'γ_j':>7}  {'a_j':>9}  {'V_j^res':>9}  {'D_j':>9}  {'I_j':>9}")
    for k in range(N):
        print(f"  {k+1:>3}  {sigmas[k]:>7.4f}  {sigmas[k+1]:>7.4f}  {h_[k]:>7.4f}  "
              f"{Gam[k]:>7.4f}  {a_[k]:>9.3e}  {V_[k]:>9.3e}  "
              f"{D_[k]:>9.3e}  {I_[k]:>9.3e}")

    r1    = rho_infty * float(np.dot(a_, Gam))**2
    vt    = float(np.dot(V_, Gam**2))
    dt    = float(np.dot(D_, Gam**2))
    total = r1 + vt + dt
    print(f"\n  Cost breakdown:")
    print(f"    rank-1 (score) : {r1:.4e}  ({100*r1/total:.1f}%)")
    print(f"    V_j^res        : {vt:.4e}  ({100*vt/total:.1f}%)")
    print(f"    D_j            : {dt:.4e}  ({100*dt/total:.1f}%)")
    print(f"    TOTAL          : {total:.4e}")
    print(f"  γ ∈ [{Gam.min():.4f}, {Gam.max():.4f}]   "
          f"h ∈ [{h_.min():.4f}, {h_.max():.4f}]")

    if out_path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{label}  NFE={N}", fontsize=12)

        ax = axes[0, 0]
        h_unif = (sigmas[0]-sigmas[-1])/N
        ax.bar(range(1, N+1), h_, color="#2d6a9f", alpha=0.8)
        ax.axhline(h_unif, color="#e05c00", lw=1.5, ls="--",
                   label=f"uniform h={h_unif:.3f}")
        ax.set_xlabel("step j"); ax.set_ylabel("h_j")
        ax.set_title("Step Sizes h_j"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(range(1, N+1), Gam, "o-", color="#b5451b", lw=2)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("step j"); ax.set_ylabel("γ_j")
        ax.set_title("Propagator γ_j"); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        width = 0.35; x = np.arange(1, N+1)
        ax.bar(x - width/2, Gam**2 * V_, width, label="γ²V^res", color="#2d6a9f", alpha=0.8)
        ax.bar(x + width/2, Gam**2 * D_, width, label="γ²D",     color="#9b3fa0", alpha=0.8)
        ax.set_xlabel("step j"); ax.set_title("Per-step cost γ²V & γ²D")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        sigma_unif = np.linspace(sigmas[0], sigmas[-1], N+1)
        ax.plot(range(N+1), sigmas,     "o-",  color="#2d6a9f", lw=2, label=label)
        ax.plot(range(N+1), sigma_unif, "s--", color="#e05c00", lw=1.5,
                alpha=0.7, label="uniform")
        ax.set_xlabel("step"); ax.set_ylabel("σ")
        ax.set_title("σ Schedule"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [diag] → {out_path}")

    return dict(sigmas=sigmas, gamma=Gam, h=h_, a=a_, V=V_, D=D_, importance=I_)


# ══════════════════════════════════════════════════════════════════════════════
# Pareto curve
# ══════════════════════════════════════════════════════════════════════════════

def build_pareto_schedules(
    sigmas_full: np.ndarray,
    nfe_min: int,
    npz_path: str,
    w_vres: float = 1.0,
    w_disc: float = 0.5,
) -> dict:
    """
    Greedy NFE reduction: at each step remove the interior σ-point with
    the smallest importance score I_k = γ_k²(w_vres V_k + w_disc D_k).
    Returns {nfe: sigmas_array} for nfe = N_full, N_full-1, …, nfe_min.
    """
    sigma2_fn, sigma2_vdot_fn, g_fn, phi_res, rho_infty = _load_stats(npz_path)

    schedules = {len(sigmas_full)-1: sigmas_full.copy()}
    current   = sigmas_full.copy()

    while len(current) - 1 > nfe_min:
        N   = len(current) - 1
        eps = np.array([_compute_epsilon_fm(current[k-1], current[k], g_fn)
                        for k in range(1, N+1)])
        Gam = _compute_gamma(eps)
        V_  = np.array([_compute_V_res_fm(current[k-1], current[k], sigma2_fn, phi_res)
                        for k in range(1, N+1)])
        D_  = np.array([_compute_D_j(current[k-1], current[k], sigma2_vdot_fn)
                        for k in range(1, N+1)])
        I_  = Gam**2 * (w_vres * V_ + w_disc * D_)

        # Among interior steps (exclude j=0 and j=N-1 near boundaries), remove min I
        remove_idx = int(np.argmin(I_[1:-1])) + 1
        current    = np.delete(current, remove_idx)
        schedules[len(current)-1] = current.copy()
        print(f"  NFE {N} → {N-1}: removed step {remove_idx}  "
              f"I={I_[remove_idx]:.3e}")

    return schedules


# ══════════════════════════════════════════════════════════════════════════════
# Diffusers injection
# ══════════════════════════════════════════════════════════════════════════════

def sigmas_to_flux_scheduler(
    sigmas: np.ndarray,
    model_name: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cpu",
):
    """
    Inject custom σ schedule into FlowMatchEulerDiscreteScheduler.

        sched = sigmas_to_flux_scheduler(optimal_sigmas, model_name)
        pipeline.scheduler = sched
        images = pipeline(prompt=..., num_inference_steps=nfe).images
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    import torch
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler")
    ts = torch.tensor(sigmas, dtype=torch.float32, device=device)
    sched.timesteps = ts
    sched.sigmas    = ts
    return sched


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="BornSchedule optimal σ-schedule for FLUX FM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stats",      required=True,  help=".npz from stats_flux.py")
    p.add_argument("--nfe",        type=int,   default=28)
    p.add_argument("--sigma_max",  type=float, default=0.97)
    p.add_argument("--sigma_min",  type=float, default=0.02)
    p.add_argument("--n_steps",    type=int,   default=2000)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lr_decay",   type=float, default=0.995)
    p.add_argument("--n_restarts", type=int,   default=3)
    p.add_argument("--barrier",    type=float, default=0.0)
    p.add_argument("--w_rank1",    type=float, default=1.0)
    p.add_argument("--w_vres",     type=float, default=1.0)
    p.add_argument("--w_disc",     type=float, default=0.5)
    p.add_argument("--pareto",     action="store_true")
    p.add_argument("--pareto_min", type=int,   default=4)
    p.add_argument("--output_dir", default="schedules")
    p.add_argument("--n_quad",     type=int,   default=32)
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[flux_schedule] NFE={args.nfe}  stats={args.stats}")

    sigmas = optimize_schedule(
        npz_path=args.stats, nfe=args.nfe,
        sigma_max=args.sigma_max, sigma_min=args.sigma_min,
        n_steps=args.n_steps, lr=args.lr, lr_decay=args.lr_decay,
        barrier_weight=args.barrier, n_quad=args.n_quad,
        w_rank1=args.w_rank1, w_vres=args.w_vres, w_disc=args.w_disc,
        n_restarts=args.n_restarts,
    )

    stem = Path(args.stats).stem
    npy  = out / f"{stem}_nfe{args.nfe}.npy"
    np.save(str(npy), sigmas)
    print(f"\n[flux_schedule] → {npy}")
    print(f"  σ = {np.round(sigmas, 4).tolist()}")

    diagnose_schedule(sigmas, args.stats,
                      label=f"BornSchedule NFE={args.nfe}",
                      out_path=str(out / f"{stem}_nfe{args.nfe}_diag.png"))

    if args.pareto:
        print(f"\n[flux_schedule] Pareto NFE {args.nfe}→{args.pareto_min} …")
        schedules = build_pareto_schedules(
            sigmas, args.pareto_min, args.stats,
            w_vres=args.w_vres, w_disc=args.w_disc)
        for nfe_k, sg in schedules.items():
            np.save(str(out / f"{stem}_nfe{nfe_k}.npy"), sg)
        print(f"  saved NFE ∈ {sorted(schedules.keys())}")

    print("[flux_schedule] done.")


if __name__ == "__main__":
    main()