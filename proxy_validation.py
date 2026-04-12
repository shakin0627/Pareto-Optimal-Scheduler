"""
proxy_validation.py  
    run_proxy_validation(born, dpm_solver, x_T, args.steps, ns, device, args.outdir)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from scheduling_born import (
    OptimalSchedule,
    _cost_functional,
    _alpha,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schedule builders
# ─────────────────────────────────────────────────────────────────────────────

def make_uniform_logsnr(born: OptimalSchedule, N: int) -> np.ndarray:
    """均匀 log-SNR 网格（当前 baseline）"""
    return np.linspace(born.lambda_min, born.lambda_max, N + 1)


def make_uniform_t(born: OptimalSchedule, N: int) -> np.ndarray:
    """
    均匀时间 t ∈ [0,1] → 对应 σ(t) 均匀插值 → 转回 λ。
    VP-SDE: σ(t) = sqrt(1 − α(t)²), 用线性 sigma 网格近似 uniform-t。
    """
    sigma_max = born.lambda_to_sigma(born.lambda_min)   # 高噪端
    sigma_min = born.lambda_to_sigma(born.lambda_max)   # 低噪端
    sigmas = np.linspace(sigma_max, sigma_min, N + 1)
    return born.sigma_to_lambda(sigmas)


def make_karras(born: OptimalSchedule, N: int, rho: float = 7.0) -> np.ndarray:
    """
    Karras et al. 2022 sigma schedule → 转回 λ。
    σ_i = (σ_max^{1/ρ} + i/(N-1) (σ_min^{1/ρ} - σ_max^{1/ρ}))^ρ
    """
    sigma_max = float(born.lambda_to_sigma(born.lambda_min))
    sigma_min = float(born.lambda_to_sigma(born.lambda_max))
    ramp = np.linspace(0, 1, N + 1)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return born.sigma_to_lambda(sigmas)


def make_quadratic_t(born: OptimalSchedule, N: int) -> np.ndarray:
    """
    二次时间调度：t_i = (i/N)^2，常用于 DDPM 变体。
    → sigma 插值 → λ
    """
    sigma_max = born.lambda_to_sigma(born.lambda_min)
    sigma_min = born.lambda_to_sigma(born.lambda_max)
    t = np.linspace(0, 1, N + 1) ** 2          # quadratic ramp
    sigmas = sigma_max + t * (sigma_min - sigma_max)
    return born.sigma_to_lambda(sigmas)


# ─────────────────────────────────────────────────────────────────────────────
# Cost computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost(born: OptimalSchedule, lambdas: np.ndarray) -> Dict[str, float]:
    """返回总 C 及三个分项，barrier=0（公平比较）。"""
    from scheduling_born import (
        _compute_epsilon, _compute_gamma,
        _compute_A_j, _compute_V_res, _compute_D_j,
    )
    N = len(lambdas) - 1
    eps   = np.array([_compute_epsilon(lambdas[k-1], lambdas[k], born.g_fn, born.r1)
                      for k in range(1, N+1)])
    Gamma = _compute_gamma(eps)
    A     = np.array([_compute_A_j(lambdas[k-1], lambdas[k], born.sigma2_fn, born.g_fn, born.r1, born.quad_pts)
                      for k in range(1, N+1)])
    Vres  = np.array([_compute_V_res(lambdas[k-1], lambdas[k], born.g_fn, born.sigma2_fn,
                                     born._phi_res_fn, born.r1)
                      for k in range(1, N+1)])
    D     = np.array([_compute_D_j(lambdas[k-1], lambdas[k], born.sigma2_gpp_fn)
                      for k in range(1, N+1)])

    a2       = _alpha(lambdas[-1]) ** 2
    rank1    = born.rho_infty * float(np.dot(A, Gamma)) ** 2
    vres_sum = float(np.dot(Vres, Gamma ** 2))
    disc_sum = float(np.dot(D,    Gamma ** 2))
    total    = a2 * (rank1 + vres_sum + disc_sum)

    return {
        "total":   total,
        "rank1":   a2 * rank1,
        "V_res":   a2 * vres_sum,
        "D":       a2 * disc_sum,
        "h_stats": (float(np.min(np.diff(lambdas))),
                    float(np.max(np.diff(lambdas))),
                    float(np.std(np.diff(lambdas)) / np.mean(np.diff(lambdas)))),  # CV
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helper (reuse your existing dpm_solver interface)
# ─────────────────────────────────────────────────────────────────────────────

def sample_with_lambdas(
    dpm_solver,
    x_T: torch.Tensor,
    lambdas: np.ndarray,
    device: str,
) -> torch.Tensor:
    """
    用指定 λ 网格直接驱动 dpm_solver.sample()。
    lambdas: 从 lambda_min (高噪) 到 lambda_max (低噪)，升序。
    转成 timesteps (降序 t) 传入。
    """
    from scheduling_born import OptimalSchedule
    sigmas = OptimalSchedule.lambda_to_sigma(lambdas)        # 降序 sigma
    # dpm_solver 期望降序 sigma（从高噪到低噪采样）
    # 对应 t ∈ [999, 0]，线性映射
    sigma_max = float(sigmas[0])
    sigma_min = float(sigmas[-1])
    t_np = (sigmas - sigma_min) / (sigma_max - sigma_min + 1e-12) * 999.0
    timesteps = torch.tensor(t_np, dtype=torch.float32, device=device)

    x = x_T.clone()
    # 调用你项目里的 sample 循环，这里给出通用接口
    # 请根据你的 dpm_solver API 调整
    x = dpm_solver.sample(
        x,
        steps=len(lambdas) - 1,
        order=2,
        skip_type="logSNR",          # 我们自己控制节点，skip_type 只是 fallback
        method="singlestep",
        timesteps_input=timesteps,   # 如果你的 API 支持直接传入
    )
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Main validation entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_proxy_validation(
    born:       OptimalSchedule,
    lambdas,
    dpm_solver,
    x_T:        torch.Tensor,
    N:          int,
    ns,                          # noise schedule, for baseline_sample compatibility
    device:     str,
    outdir:     str,
    run_sampling: bool = False,   # False 
):
    """
        run_proxy_validation(born, dpm_solver, x_T, args.steps, ns, device, args.outdir)
    """
    print("\n" + "═" * 70)
    print("PROXY VALIDATION  —  Cost C vs Schedule Quality")
    print("═" * 70)

    schedules: Dict[str, np.ndarray] = {
        "born_opt":      lambdas,
        "uniform_logSNR": make_uniform_logsnr(born, N),
        "uniform_t":      make_uniform_t(born, N),
        "karras_rho7":    make_karras(born, N, rho=7.0),
        "quadratic_t":    make_quadratic_t(born, N),
    }

    # ── 打印 λ 节点对比
    print(f"\n{'Schedule':<20}  {'λ nodes (first 4 … last 4)'}")
    print("-" * 70)
    for name, lam in schedules.items():
        head = "  ".join(f"{x:.3f}" for x in lam[:4])
        tail = "  ".join(f"{x:.3f}" for x in lam[-4:])
        print(f"{name:<20}  [{head} … {tail}]")

    # ── 计算 cost functional 分项
    print(f"\n{'Schedule':<20}  {'C_total':>12}  {'rank1':>10}  {'V_res':>10}  {'D':>10}  {'h_CV':>7}")
    print("-" * 70)
    costs: Dict[str, Dict] = {}
    for name, lam in schedules.items():
        c = compute_cost(born, lam)
        costs[name] = c
        h_min, h_max, h_cv = c["h_stats"]
        print(f"{name:<20}  {c['total']:>12.4e}  {c['rank1']:>10.4e}  "
              f"{c['V_res']:>10.4e}  {c['D']:>10.4e}  {h_cv:>7.3f}")

    # ── Cost 排序
    cost_ranking = sorted(costs.items(), key=lambda x: x[1]["total"])
    print(f"\nCost ranking (↑ = better predicted by theory):")
    for rank, (name, c) in enumerate(cost_ranking, 1):
        ratio = c["total"] / costs["born_opt"]["total"]
        print(f"  {rank}. {name:<20}  C={c['total']:.4e}  (×{ratio:.3f} vs born_opt)")
