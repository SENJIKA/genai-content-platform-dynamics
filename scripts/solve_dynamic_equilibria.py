"""Solve the paper's two-period and infinite-horizon platform equilibria.

The implementation is original to this project.  Public replication code is
used only to discipline the numerical workflow: a finite state/action grid,
bilinear interpolation, Bellman iteration, residual checks, and forward
simulation.  No source code from those packages is copied here.

The economic model follows the target-entry representation in ``main.tex``.
Creators' exposure, effort, quality, surplus, and the marginal entrant's cost
are substituted analytically.  The platform then chooses target entry and the
revenue share on a discrete grid.  Three policy environments are kept distinct:

* signed: an auxiliary benchmark in which the implementation transfer can be
  positive or negative;
* support_only: a uniform contract with a non-negative fixed transfer;
* governance: positive transfers when needed, and a quota at zero transfer
  when desired entry is below natural entry.  The quota is not treated as
  negative-transfer revenue.

The KuaiRec category file is used only for a normalized content-diversity
moment.  All remaining parameter values are transparent normalizations and are
reported with sensitivity checks; the exercise is not a structural estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "literature_recent_discrete_models" / "replication_code_sources" / "kuairec_caption_category.csv"
FIGURE_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output" / "computation"


@dataclass(frozen=True)
class Parameters:
    # Preferences, technology, and within-period contract primitives.
    beta: float = 0.92
    n_bar: float = 4.0
    i_bar: float = 2.0
    r: float = 2.0
    v: float = 0.50
    effort_cost: float = 5.0
    exposure_cost: float = 1.20
    outside_congestion: float = 1.20

    # State-dependent primitives A(K), kbar(K), eta(M), and theta(M,D).
    a_floor: float = 0.72
    a_gain: float = 0.62
    a_speed: float = 0.38
    cost_floor: float = 0.50
    cost_scale: float = 2.65
    cost_k_slope: float = 0.20
    eta_floor: float = 0.76
    eta_gain: float = 0.52
    eta_speed: float = 0.34
    theta_floor: float = 0.28
    theta_m_speed: float = 0.30
    theta_d_weight: float = 0.42

    # Portfolio discovery and convex review cost.
    discovery_weight: float = 1.15
    discovery_scale: float = 2.15
    discovery_speed: float = 0.90
    review_cost: float = 0.26

    # Laws of motion.  Each business batch is one model period.
    delta_k: float = 0.18
    delta_m: float = 0.12
    learn_k_entry: float = 0.22
    learn_k_effort: float = 0.46
    learn_m_entry: float = 0.16
    learn_m_diversity: float = 0.46
    learn_m_perf: float = 0.18

    k_max: float = 6.0
    m_max: float = 10.0


@dataclass(frozen=True)
class GridSpec:
    n_k: int = 31
    n_m: int = 41
    n_n: int = 49
    n_s: int = 17


@dataclass
class Environment:
    params: Parameters
    grid_spec: GridSpec
    diversity: float
    k_grid: np.ndarray
    m_grid: np.ndarray
    n_grid: np.ndarray
    s_grid: np.ndarray
    reward: Dict[str, np.ndarray]
    psi: np.ndarray
    effort: np.ndarray
    performance: np.ndarray
    next_k: np.ndarray
    next_m: np.ndarray
    interp_indices: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    interp_weights: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def category_diversity(path: Path) -> Dict[str, float]:
    """Return normalized Shannon entropy and related transparent moments."""
    counts: Dict[str, int] = {}
    total_rows = 0
    valid_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            name = (row.get("first_level_category_name") or "").strip()
            category_id = (row.get("first_level_category_id") or "").strip()
            if not name or name.upper() == "UNKNOWN" or category_id == "-124":
                continue
            counts[name] = counts.get(name, 0) + 1
            valid_rows += 1
    probabilities = np.array(list(counts.values()), dtype=float) / valid_rows
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized = entropy / math.log(len(counts))
    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "categories": len(counts),
        "shannon_entropy": entropy,
        "normalized_shannon_entropy": normalized,
        "effective_categories": float(math.exp(entropy)),
    }


def primitives(
    p: Parameters,
    diversity: float,
    k: np.ndarray,
    m: np.ndarray,
    n: np.ndarray,
    s: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Evaluate the analytic within-period equilibrium and transitions."""
    a = p.a_floor + p.a_gain * (1.0 - np.exp(-p.a_speed * k))
    k_upper = p.cost_floor + p.cost_scale / (1.0 + p.cost_k_slope * k)
    eta = p.eta_floor + p.eta_gain * (1.0 - np.exp(-p.eta_speed * m))
    theta = p.theta_floor + (1.0 - p.theta_floor) * np.exp(-p.theta_m_speed * m) * (
        1.0 - p.theta_d_weight * diversity
    )
    theta = np.clip(theta, p.theta_floor, 1.0)
    congestion = p.outside_congestion + theta * n
    platform_margin = p.v + (1.0 - s) * p.r
    zeta = 2.0 * s * p.r * platform_margin * eta**2 / (p.exposure_cost * congestion)
    if np.any(zeta >= p.effort_cost):
        raise ValueError("Creator effort is not globally concave: zeta >= effort_cost")
    effort = zeta * a / (p.effort_cost - zeta)
    quality = p.effort_cost * a / (p.effort_cost - zeta)
    exposure = platform_margin * eta * quality / (p.exposure_cost * congestion)
    performance = platform_margin * eta**2 * quality**2 / (p.exposure_cost * congestion)
    creator_surplus = p.effort_cost * zeta * a**2 / (2.0 * (p.effort_cost - zeta))
    platform_project_profit = platform_margin**2 * eta**2 * quality**2 / (
        2.0 * p.exposure_cost * congestion
    )

    marginal_cost = k_upper * n / p.n_bar
    psi = marginal_cost - creator_surplus
    discovery = p.discovery_weight * p.discovery_scale * (1.0 - np.exp(-p.discovery_speed * n))
    review = 0.5 * p.review_cost * n**2

    signed_reward = n * (platform_project_profit - psi) + discovery - review
    quota_reward = n * platform_project_profit + discovery - review
    support_feasible = (psi >= -1e-12) & (psi <= p.i_bar + 1e-12)
    support_reward = np.where(support_feasible, signed_reward, -np.inf)
    governance_reward = np.where(psi < 0.0, quota_reward, signed_reward)
    governance_reward = np.where(psi <= p.i_bar + 1e-12, governance_reward, -np.inf)

    k_investment = p.learn_k_entry * np.log1p(n) + p.learn_k_effort * np.log1p(n * effort)
    m_investment = (
        p.learn_m_entry * np.log1p(n)
        + p.learn_m_diversity * diversity * (1.0 - np.exp(-0.80 * n))
        + p.learn_m_perf * np.log1p(n * performance)
    )
    next_k = np.clip((1.0 - p.delta_k) * k + k_investment, 0.0, p.k_max)
    next_m = np.clip((1.0 - p.delta_m) * m + m_investment, 0.0, p.m_max)
    return {
        "a": a,
        "k_upper": k_upper,
        "eta": eta,
        "theta": theta,
        "zeta": zeta,
        "effort": effort,
        "quality": quality,
        "exposure": exposure,
        "performance": performance,
        "creator_surplus": creator_surplus,
        "project_profit": platform_project_profit,
        "psi": psi,
        "signed_reward": signed_reward,
        "support_only_reward": support_reward,
        "governance_reward": governance_reward,
        "next_k": next_k,
        "next_m": next_m,
    }


def interpolation_map(
    k_grid: np.ndarray, m_grid: np.ndarray, next_k: np.ndarray, next_m: np.ndarray
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Precompute indices and weights for bilinear continuation-value interpolation."""
    k_pos = np.clip((next_k - k_grid[0]) / (k_grid[1] - k_grid[0]), 0.0, len(k_grid) - 1.0)
    m_pos = np.clip((next_m - m_grid[0]) / (m_grid[1] - m_grid[0]), 0.0, len(m_grid) - 1.0)
    ik0 = np.minimum(np.floor(k_pos).astype(np.int32), len(k_grid) - 2)
    im0 = np.minimum(np.floor(m_pos).astype(np.int32), len(m_grid) - 2)
    wk = k_pos - ik0
    wm = m_pos - im0
    n_m = len(m_grid)
    idx00 = ik0 * n_m + im0
    idx10 = (ik0 + 1) * n_m + im0
    idx01 = ik0 * n_m + (im0 + 1)
    idx11 = (ik0 + 1) * n_m + (im0 + 1)
    w00 = (1.0 - wk) * (1.0 - wm)
    w10 = wk * (1.0 - wm)
    w01 = (1.0 - wk) * wm
    w11 = wk * wm
    return (idx00, idx10, idx01, idx11), (w00, w10, w01, w11)


def build_environment(p: Parameters, spec: GridSpec, diversity: float) -> Environment:
    k_grid = np.linspace(0.0, p.k_max, spec.n_k)
    m_grid = np.linspace(0.0, p.m_max, spec.n_m)
    n_grid = np.linspace(0.0, p.n_bar, spec.n_n)
    s_grid = np.linspace(0.0, 1.0, spec.n_s)
    kk, mm = np.meshgrid(k_grid, m_grid, indexing="ij")
    nn, ss = np.meshgrid(n_grid, s_grid, indexing="ij")
    state_k = kk.reshape(-1, 1)
    state_m = mm.reshape(-1, 1)
    action_n = nn.reshape(1, -1)
    action_s = ss.reshape(1, -1)
    values = primitives(p, diversity, state_k, state_m, action_n, action_s)
    indices, weights = interpolation_map(k_grid, m_grid, values["next_k"], values["next_m"])
    return Environment(
        params=p,
        grid_spec=spec,
        diversity=diversity,
        k_grid=k_grid,
        m_grid=m_grid,
        n_grid=n_grid,
        s_grid=s_grid,
        reward={
            "signed": values["signed_reward"],
            "support_only": values["support_only_reward"],
            "governance": values["governance_reward"],
        },
        psi=values["psi"],
        effort=values["effort"],
        performance=values["performance"],
        next_k=values["next_k"],
        next_m=values["next_m"],
        interp_indices=indices,
        interp_weights=weights,
    )


def expected_value(env: Environment, value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1)
    idx00, idx10, idx01, idx11 = env.interp_indices
    w00, w10, w01, w11 = env.interp_weights
    return w00 * flat[idx00] + w10 * flat[idx10] + w01 * flat[idx01] + w11 * flat[idx11]


def policy_arrays(env: Environment, action_index: np.ndarray) -> Dict[str, np.ndarray]:
    n_s = env.grid_spec.n_s
    state = np.arange(action_index.size)
    flat_action = action_index.reshape(-1)
    n = env.n_grid[flat_action // n_s]
    s = env.s_grid[flat_action % n_s]
    psi = env.psi[state, flat_action]
    effort = env.effort[state, flat_action]
    performance = env.performance[state, flat_action]
    next_k = env.next_k[state, flat_action]
    next_m = env.next_m[state, flat_action]
    shape = (env.grid_spec.n_k, env.grid_spec.n_m)
    return {
        "action_index": action_index.reshape(shape),
        "n": n.reshape(shape),
        "s": s.reshape(shape),
        "psi": psi.reshape(shape),
        "support": np.maximum(psi, 0.0).reshape(shape),
        "quota_wedge": np.maximum(-psi, 0.0).reshape(shape),
        "effort": effort.reshape(shape),
        "performance": performance.reshape(shape),
        "next_k": next_k.reshape(shape),
        "next_m": next_m.reshape(shape),
    }


def two_period(env: Environment, regime: str) -> Dict[str, Dict[str, np.ndarray]]:
    reward = env.reward[regime]
    terminal_action = np.argmax(reward, axis=1)
    terminal_value = reward[np.arange(reward.shape[0]), terminal_action].reshape(
        env.grid_spec.n_k, env.grid_spec.n_m
    )
    continuation = expected_value(env, terminal_value)
    first_q = reward + env.params.beta * continuation
    first_action = np.argmax(first_q, axis=1)
    first_value = first_q[np.arange(first_q.shape[0]), first_action].reshape(
        env.grid_spec.n_k, env.grid_spec.n_m
    )
    return {
        "period_1": {"value": first_value, **policy_arrays(env, first_action)},
        "period_2": {"value": terminal_value, **policy_arrays(env, terminal_action)},
    }


def infinite_horizon(
    env: Environment,
    regime: str,
    tolerance: float = 1e-7,
    max_iter: int = 1200,
) -> Dict[str, object]:
    reward = env.reward[regime]
    value = np.zeros((env.grid_spec.n_k, env.grid_spec.n_m))
    error_trace = []
    start = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        continuation = expected_value(env, value)
        q_value = reward + env.params.beta * continuation
        action = np.argmax(q_value, axis=1)
        new_value = q_value[np.arange(q_value.shape[0]), action].reshape(value.shape)
        error = float(np.max(np.abs(new_value - value)))
        error_trace.append(error)
        value = new_value
        if error < tolerance:
            break
    continuation = expected_value(env, value)
    bellman_q = reward + env.params.beta * continuation
    final_action = np.argmax(bellman_q, axis=1)
    bellman_value = bellman_q[np.arange(bellman_q.shape[0]), final_action].reshape(value.shape)
    residual = float(np.max(np.abs(bellman_value - value)))
    elapsed = time.perf_counter() - start
    if iteration == max_iter and error >= tolerance:
        raise RuntimeError(f"VFI did not converge for {regime}: error={error:g}")
    return {
        "value": value,
        "policy": policy_arrays(env, final_action),
        "iterations": iteration,
        "last_successive_error": error_trace[-1],
        "bellman_residual": residual,
        "elapsed_seconds": elapsed,
        "error_trace": error_trace,
    }


def bilinear_scalar(k_grid: np.ndarray, m_grid: np.ndarray, array: np.ndarray, k: float, m: float) -> float:
    k = float(np.clip(k, k_grid[0], k_grid[-1]))
    m = float(np.clip(m, m_grid[0], m_grid[-1]))
    ik = min(int((k - k_grid[0]) / (k_grid[1] - k_grid[0])), len(k_grid) - 2)
    im = min(int((m - m_grid[0]) / (m_grid[1] - m_grid[0])), len(m_grid) - 2)
    wk = (k - k_grid[ik]) / (k_grid[ik + 1] - k_grid[ik])
    wm = (m - m_grid[im]) / (m_grid[im + 1] - m_grid[im])
    return float(
        (1 - wk) * (1 - wm) * array[ik, im]
        + wk * (1 - wm) * array[ik + 1, im]
        + (1 - wk) * wm * array[ik, im + 1]
        + wk * wm * array[ik + 1, im + 1]
    )


def natural_entry(p: Parameters, diversity: float, k: float, m: float, s: float) -> float:
    """Solve Psi(n;s,K,M)=0 by bisection; Psi is monotone in the calibration."""
    def wedge(n: float) -> float:
        result = primitives(
            p,
            diversity,
            np.array(k),
            np.array(m),
            np.array(n),
            np.array(s),
        )
        return float(result["psi"])

    lo, hi = 0.0, p.n_bar
    if wedge(lo) >= 0.0:
        return lo
    if wedge(hi) <= 0.0:
        return hi
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if wedge(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(
    env: Environment,
    solution: Dict[str, object],
    periods: int = 60,
    initial_state: Tuple[float, float] = (0.20, 0.20),
) -> Dict[str, np.ndarray]:
    policy = solution["policy"]
    out = {name: np.zeros(periods + 1) for name in ["k", "m", "n", "s", "psi", "support", "quota", "natural"]}
    out["k"][0], out["m"][0] = initial_state
    for t in range(periods):
        k, m = out["k"][t], out["m"][t]
        n = bilinear_scalar(env.k_grid, env.m_grid, policy["n"], k, m)
        s = bilinear_scalar(env.k_grid, env.m_grid, policy["s"], k, m)
        period = primitives(
            env.params,
            env.diversity,
            np.array(k),
            np.array(m),
            np.array(n),
            np.array(s),
        )
        psi = float(period["psi"])
        out["n"][t] = n
        out["s"][t] = s
        out["psi"][t] = psi
        out["support"][t] = max(psi, 0.0)
        out["quota"][t] = max(natural_entry(env.params, env.diversity, k, m, s) - n, 0.0)
        out["natural"][t] = natural_entry(env.params, env.diversity, k, m, s)
        out["k"][t + 1] = float(period["next_k"])
        out["m"][t + 1] = float(period["next_m"])
    for key in ["n", "s", "psi", "support", "quota", "natural"]:
        out[key][-1] = out[key][-2]
    return out


def state_report(env: Environment, solution: Dict[str, object], states: Iterable[Tuple[float, float]]) -> list[dict]:
    policy = solution["policy"]
    rows = []
    for k, m in states:
        row = {"K": k, "M": m}
        for field in ["n", "s"]:
            row[field] = bilinear_scalar(env.k_grid, env.m_grid, policy[field], k, m)
        period = primitives(
            env.params,
            env.diversity,
            np.array(k),
            np.array(m),
            np.array(row["n"]),
            np.array(row["s"]),
        )
        row["psi"] = float(period["psi"])
        row["support"] = max(row["psi"], 0.0)
        row["quota_wedge"] = max(-row["psi"], 0.0)
        row["effort"] = float(period["effort"])
        row["performance"] = float(period["performance"])
        row["natural_entry"] = natural_entry(env.params, env.diversity, k, m, row["s"])
        row["quota_gap"] = max(row["natural_entry"] - row["n"], 0.0)
        rows.append(row)
    return rows


def find_switch(path: Dict[str, np.ndarray]) -> int | None:
    """First period after which positive support does not return on the path."""
    support_periods = np.flatnonzero(path["support"] > 1e-3)
    candidate = int(support_periods[-1] + 1) if support_periods.size else 0
    if candidate >= len(path["support"]):
        return None
    if np.any(path["quota"][candidate:] > 1e-3):
        return candidate
    return None


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    env: Environment,
    two: Dict[str, Dict[str, np.ndarray]],
    solutions: Dict[str, Dict[str, object]],
    path: Dict[str, np.ndarray],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    k_target = 1.5
    ik = int(np.argmin(np.abs(env.k_grid - k_target)))
    m = env.m_grid

    # Two-period backward induction.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)
    p1, p2 = two["period_1"], two["period_2"]
    natural_1 = np.array(
        [natural_entry(env.params, env.diversity, env.k_grid[ik], mj, p1["s"][ik, j]) for j, mj in enumerate(m)]
    )
    axes[0].plot(m, p1["n"][ik], lw=2.0, color="#145da0", label=r"period 1 $n_1^*$")
    axes[0].plot(m, p2["n"][ik], lw=1.8, color="#566573", ls="-.", label=r"terminal $n_2^*$")
    axes[0].plot(m, natural_1, lw=1.8, color="#b03a2e", ls="--", label=r"natural $n^{0,R}$")
    axes[0].set(xlabel=r"recommendation capital $M$", ylabel="entry scale", xlim=(0, env.params.m_max))
    axes[0].legend(frameon=False)
    axes[1].plot(m, p1["support"][ik], lw=2.0, color="#145da0", label="positive fixed payment")
    axes[1].plot(m, np.maximum(natural_1 - p1["n"][ik], 0.0), lw=2.0, color="#b03a2e", ls="--", label="quota gap")
    axes[1].axhline(0.0, color="#777777", lw=0.7)
    axes[1].set(xlabel=r"recommendation capital $M$", ylabel="instrument intensity", xlim=(0, env.params.m_max))
    axes[1].legend(frameon=False)
    fig.savefig(FIGURE_DIR / "two_period_equilibrium_solution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Infinite-horizon policy comparison.
    gov = solutions["governance"]["policy"]
    signed = solutions["signed"]["policy"]
    support = solutions["support_only"]["policy"]
    natural_g = np.array(
        [natural_entry(env.params, env.diversity, env.k_grid[ik], mj, gov["s"][ik, j]) for j, mj in enumerate(m)]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)
    axes[0].plot(m, gov["n"][ik], lw=2.1, color="#145da0", label="support + quota")
    axes[0].plot(m, signed["n"][ik], lw=1.7, color="#7d3c98", ls="-.", label="signed-transfer benchmark")
    axes[0].plot(m, support["n"][ik], lw=1.7, color="#566573", ls=":", label="support only")
    axes[0].plot(m, natural_g, lw=1.8, color="#b03a2e", ls="--", label=r"natural $n^{0,R}$")
    axes[0].set(xlabel=r"recommendation capital $M$", ylabel="stationary policy entry", xlim=(0, env.params.m_max))
    axes[0].legend(frameon=False)
    axes[1].plot(m, gov["support"][ik], lw=2.0, color="#145da0", label="fixed payment")
    axes[1].plot(m, np.maximum(natural_g - gov["n"][ik], 0.0), lw=2.0, color="#b03a2e", ls="--", label="quota gap")
    axes[1].set(xlabel=r"recommendation capital $M$", ylabel="stationary instrument", xlim=(0, env.params.m_max))
    axes[1].legend(frameon=False)
    fig.savefig(FIGURE_DIR / "infinite_horizon_policy_solution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Endogenous transition path.
    t = np.arange(len(path["k"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)
    axes[0].plot(t, path["k"], color="#7d3c98", lw=2.0, label=r"creative capital $K_t$")
    axes[0].plot(t, path["m"], color="#145da0", lw=2.0, label=r"recommendation capital $M_t$")
    axes[0].set(xlabel="business batch", ylabel="state", xlim=(0, t[-1]))
    axes[0].legend(frameon=False)
    axes[1].plot(t, path["n"], color="#273746", lw=2.0, label=r"entry $n_t$")
    axes[1].plot(t, path["support"], color="#145da0", lw=1.8, label="fixed payment")
    axes[1].plot(t, path["quota"], color="#b03a2e", lw=1.8, ls="--", label="quota gap")
    switch = find_switch(path)
    if switch is not None:
        axes[1].axvline(switch, color="#777777", lw=0.9, ls=":")
    axes[1].set(xlabel="business batch", ylabel="policy and entry", xlim=(0, t[-1]))
    axes[1].legend(frameon=False)
    fig.savefig(FIGURE_DIR / "infinite_horizon_transition_solution.pdf", bbox_inches="tight")
    plt.close(fig)


def sensitivity_cases(diversity: float) -> list[Tuple[str, Parameters, GridSpec, float]]:
    coarse = GridSpec(n_k=21, n_m=29, n_n=37, n_s=13)
    base = Parameters()
    return [
        ("coarse_grid", base, coarse, diversity),
        ("beta_0.88", replace(base, beta=0.88), coarse, diversity),
        ("beta_0.96", replace(base, beta=0.96), coarse, diversity),
        ("diversity_minus_0.10", base, coarse, max(0.05, diversity - 0.10)),
        ("diversity_plus_0.05", base, coarse, min(0.99, diversity + 0.05)),
        ("faster_obsolescence", replace(base, delta_k=0.24, delta_m=0.18), coarse, diversity),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    diversity_stats = category_diversity(DATA_PATH)
    diversity = diversity_stats["normalized_shannon_entropy"]
    params = Parameters()
    spec = GridSpec()
    env = build_environment(params, spec, diversity)

    two = two_period(env, "governance")
    solutions = {regime: infinite_horizon(env, regime) for regime in ["governance", "signed", "support_only"]}
    path = simulate(env, solutions["governance"])
    states = [(0.5, 0.5), (1.5, 3.0), (2.5, 6.0), (3.5, 8.5)]
    state_rows = state_report(env, solutions["governance"], states)
    write_csv(OUTPUT_DIR / "infinite_horizon_reference_states.csv", state_rows)

    two_rows = []
    ik = int(np.argmin(np.abs(env.k_grid - 1.5)))
    for j, m in enumerate(env.m_grid):
        p1 = two["period_1"]
        n0 = natural_entry(params, diversity, env.k_grid[ik], m, p1["s"][ik, j])
        two_rows.append(
            {
                "K": env.k_grid[ik],
                "M": m,
                "n_period_1": p1["n"][ik, j],
                "s_period_1": p1["s"][ik, j],
                "fixed_payment_period_1": p1["support"][ik, j],
                "quota_gap_period_1": max(n0 - p1["n"][ik, j], 0.0),
                "n_terminal": two["period_2"]["n"][ik, j],
                "natural_entry": n0,
            }
        )
    write_csv(OUTPUT_DIR / "two_period_policy_slice.csv", two_rows)

    path_rows = [{key: float(value[t]) for key, value in path.items()} | {"period": t} for t in range(len(path["k"]))]
    write_csv(OUTPUT_DIR / "infinite_horizon_transition_path.csv", path_rows)

    sensitivity_rows = []
    if not args.skip_sensitivity:
        for name, case_params, case_grid, case_diversity in sensitivity_cases(diversity):
            case_env = build_environment(case_params, case_grid, case_diversity)
            case_solution = infinite_horizon(case_env, "governance", tolerance=2e-7)
            case_path = simulate(case_env, case_solution)
            report = state_report(case_env, case_solution, [(0.5, 0.5), (3.5, 8.5)])
            sensitivity_rows.append(
                {
                    "case": name,
                    "beta": case_params.beta,
                    "delta_k": case_params.delta_k,
                    "delta_m": case_params.delta_m,
                    "diversity": case_diversity,
                    "iterations": case_solution["iterations"],
                    "bellman_residual": case_solution["bellman_residual"],
                    "initial_n": report[0]["n"],
                    "initial_support": report[0]["support"],
                    "mature_n": report[1]["n"],
                    "mature_quota_gap": report[1]["quota_gap"],
                    "switch_period": find_switch(case_path),
                    "terminal_K": case_path["k"][-1],
                    "terminal_M": case_path["m"][-1],
                }
            )
        write_csv(OUTPUT_DIR / "sensitivity.csv", sensitivity_rows)

    plot_results(env, two, solutions, path)
    summary = {
        "interpretation": "partial calibration and numerical mechanism check, not structural estimation",
        "data_moment": diversity_stats,
        "parameters": asdict(params),
        "grid": asdict(spec),
        "internal_checks": {
            "max_zeta": float(
                params.effort_cost
                * np.max(env.effort / np.maximum(env.effort + (
                    params.a_floor
                    + params.a_gain
                    * (1.0 - np.exp(-params.a_speed * np.repeat(env.k_grid, len(env.m_grid))[:, None]))
                ), 1e-12))
            ),
            "effort_cost": params.effort_cost,
            "minimum_discrete_psi_increment": float(
                np.min(np.diff(env.psi.reshape(-1, spec.n_n, spec.n_s), axis=1))
            ),
            "share_next_K_at_upper_boundary": float(np.mean(env.next_k >= params.k_max - 1e-10)),
            "share_next_M_at_upper_boundary": float(np.mean(env.next_m >= params.m_max - 1e-10)),
        },
        "source_boundaries": {
            "kuairec": "first-level category frequencies only; normalized Shannon diversity",
            "farboodi_veldkamp": "algorithmic organization only; no parameters or code copied",
            "structural_rl": "VFI and residual-check workflow only; no code copied",
        },
        "two_period": {
            "reference_K": float(env.k_grid[ik]),
            "period_1_initial_M": two_rows[0],
            "period_1_mature_M": two_rows[-1],
        },
        "infinite_horizon": {
            regime: {
                "iterations": int(solution["iterations"]),
                "last_successive_error": float(solution["last_successive_error"]),
                "bellman_residual": float(solution["bellman_residual"]),
                "elapsed_seconds": float(solution["elapsed_seconds"]),
            }
            for regime, solution in solutions.items()
        },
        "reference_states": state_rows,
        "transition": {
            "initial_state": [float(path["k"][0]), float(path["m"][0])],
            "switch_period": find_switch(path),
            "terminal_K": float(path["k"][-1]),
            "terminal_M": float(path["m"][-1]),
            "terminal_entry": float(path["n"][-1]),
        },
        "sensitivity": sensitivity_rows,
    }
    (OUTPUT_DIR / "equilibrium_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
