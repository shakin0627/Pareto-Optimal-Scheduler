from __future__ import annotations

import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
 
import numpy as np
import torch
from scipy.interpolate import interp1d, PchipInterpolator

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