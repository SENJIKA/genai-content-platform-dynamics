"""Solve the paper's two online-learning extensions.

The first extension retains the Gaussian drifting match state from the paper
and adds an explicit test-exposure choice.  The platform trades current
matching revenue against a more precise signal and solves a belief-state
Bellman equation over posterior mean and precision.

The second extension replaces the Gaussian state with a two-category Markov
hotspot.  The platform chooses how much attention to place on category A;
balanced exposure is more informative about the relative hotspot, while
tilting toward the currently more likely category raises current revenue.

Both exercises condition on the paper's structural creative and recommendation
capital.  They are transparent mechanism checks rather than estimates of a
production recommendation system or claims about a particular platform.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solve_dynamic_equilibria import DATA_PATH, Parameters, category_diversity


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output" / "computation"


@dataclass(frozen=True)
class GaussianOnlineParameters:
    rho: float = 0.88
    process_sd: float = 0.16
    match_loading: float = 0.55
    exploration_cost: float = 0.06
    signal_scale: float = 1.35
    signal_exploration_gain: float = 2.50
    reference_k: float = 1.50
    reference_m: float = 3.00


@dataclass(frozen=True)
class GaussianGrid:
    n_mu: int = 25
    n_xi: int = 17
    n_n: int = 17
    n_s: int = 9
    n_a: int = 9
    mu_min: float = -0.60
    mu_max: float = 0.60
    xi_min: float = 1.50
    xi_max: float = 22.00


@dataclass(frozen=True)
class MarkovHotspotParameters:
    switch_probability: float = 0.08
    match_loading: float = 0.35
    signal_cap: float = 0.45
    signal_scale: float = 0.72
    comparative_signal_floor: float = 0.15
    early_k: float = 0.50
    early_m: float = 0.50
    mature_k: float = 3.50
    mature_m: float = 8.50


@dataclass(frozen=True)
class MarkovGrid:
    n_p: int = 81
    n_n: int = 17
    n_s: int = 9
    n_a: int = 21


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def structural_terms(p: Parameters, diversity: float, k: float, m: float) -> dict[str, float]:
    a = p.a_floor + p.a_gain * (1.0 - math.exp(-p.a_speed * k))
    k_upper = p.cost_floor + p.cost_scale / (1.0 + p.cost_k_slope * k)
    eta = p.eta_floor + p.eta_gain * (1.0 - math.exp(-p.eta_speed * m))
    overlap = p.theta_floor + (1.0 - p.theta_floor) * math.exp(-p.theta_m_speed * m) * (
        1.0 - p.theta_d_weight * diversity
    )
    return {"a": a, "k_upper": k_upper, "eta": eta, "overlap": float(np.clip(overlap, p.theta_floor, 1.0))}


def online_period(
    p: Parameters,
    diversity: float,
    k: float,
    m: float,
    n: np.ndarray,
    s: np.ndarray,
    match_factor: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the within-period equilibrium under an expected match multiplier.

    The multiplier scales effective matching productivity before effort and
    exposure are chosen.  Thus creators respond to the platform's public belief,
    not to the unobserved realization of the hotspot.
    """
    terms = structural_terms(p, diversity, k, m)
    congestion = p.outside_congestion + terms["overlap"] * n
    margin = p.v + (1.0 - s) * p.r
    eta_sq = terms["eta"] ** 2 * np.maximum(match_factor, 0.15)
    zeta = 2.0 * s * p.r * margin * eta_sq / (p.exposure_cost * congestion)
    feasible_concavity = zeta < p.effort_cost - 1e-10
    safe_denom = np.maximum(p.effort_cost - zeta, 1e-8)
    effort = zeta * terms["a"] / safe_denom
    quality = p.effort_cost * terms["a"] / safe_denom
    performance = margin * eta_sq * quality**2 / (p.exposure_cost * congestion)
    creator_surplus = p.effort_cost * zeta * terms["a"] ** 2 / (2.0 * safe_denom)
    project_profit = margin**2 * eta_sq * quality**2 / (2.0 * p.exposure_cost * congestion)
    marginal_cost = terms["k_upper"] * n / p.n_bar
    psi = marginal_cost - creator_surplus
    discovery = p.discovery_weight * p.discovery_scale * (1.0 - np.exp(-p.discovery_speed * n))
    review = 0.5 * p.review_cost * n**2
    signed_reward = n * (project_profit - psi) + discovery - review
    quota_reward = n * project_profit + discovery - review
    reward = np.where(psi < 0.0, quota_reward, signed_reward)
    feasible = feasible_concavity & (psi <= p.i_bar + 1e-12)
    reward = np.where(feasible, reward, -np.inf)
    return {
        "reward": reward,
        "psi": psi,
        "support": np.maximum(psi, 0.0),
        "quota_wedge": np.maximum(-psi, 0.0),
        "effort": effort,
        "performance": performance,
        "zeta": zeta,
    }


def online_natural_entry(
    p: Parameters,
    diversity: float,
    k: float,
    m: float,
    s: float,
    match_factor: float,
) -> float:
    """Zero-transfer entry under the online model's expected matching state."""
    def wedge(entry: float) -> float:
        outcome = online_period(
            p,
            diversity,
            k,
            m,
            np.array(entry),
            np.array(s),
            np.array(match_factor),
        )
        return float(outcome["psi"])

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


def interp1(grid: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    pos = np.clip((x - grid[0]) / (grid[1] - grid[0]), 0.0, len(grid) - 1.0)
    i0 = np.minimum(np.floor(pos).astype(np.int32), len(grid) - 2)
    w = pos - i0
    return (1.0 - w) * values[i0] + w * values[i0 + 1]


def interp2(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    xpos = np.clip((x - x_grid[0]) / (x_grid[1] - x_grid[0]), 0.0, len(x_grid) - 1.0)
    ypos = np.clip((y - y_grid[0]) / (y_grid[1] - y_grid[0]), 0.0, len(y_grid) - 1.0)
    ix = np.minimum(np.floor(xpos).astype(np.int32), len(x_grid) - 2)
    iy = np.minimum(np.floor(ypos).astype(np.int32), len(y_grid) - 2)
    wx = xpos - ix
    wy = ypos - iy
    return (
        (1.0 - wx) * (1.0 - wy) * values[ix, iy]
        + wx * (1.0 - wy) * values[ix + 1, iy]
        + (1.0 - wx) * wy * values[ix, iy + 1]
        + wx * wy * values[ix + 1, iy + 1]
    )


def solve_gaussian(
    base: Parameters,
    diversity: float,
    gp: GaussianOnlineParameters,
    grid: GaussianGrid,
    tolerance: float = 1e-7,
    max_iter: int = 1200,
) -> dict[str, object]:
    mu_grid = np.linspace(grid.mu_min, grid.mu_max, grid.n_mu)
    xi_grid = np.linspace(grid.xi_min, grid.xi_max, grid.n_xi)
    n_grid = np.linspace(0.0, base.n_bar, grid.n_n)
    s_grid = np.linspace(0.0, 1.0, grid.n_s)
    a_grid = np.linspace(0.0, 1.0, grid.n_a)

    mu = mu_grid[:, None, None, None, None]
    xi = xi_grid[None, :, None, None, None]
    n = n_grid[None, None, :, None, None]
    s = s_grid[None, None, None, :, None]
    explore = a_grid[None, None, None, None, :]
    current_match = (1.0 + gp.match_loading * np.tanh(mu)) * (
        1.0 - gp.exploration_cost * explore**2
    ) + 0.0 * xi
    period = online_period(base, diversity, gp.reference_k, gp.reference_m, n, s, current_match)
    reward_all = period["reward"]
    best_s_index = np.argmax(reward_all, axis=3)
    reward = np.max(reward_all, axis=3)
    selected = best_s_index[:, :, :, None, :]
    best_s = s_grid[best_s_index]
    best_psi = np.take_along_axis(period["psi"], selected, axis=3).squeeze(3)
    best_support = np.take_along_axis(period["support"], selected, axis=3).squeeze(3)
    best_quota = np.take_along_axis(period["quota_wedge"], selected, axis=3).squeeze(3)

    terms = structural_terms(base, diversity, gp.reference_k, gp.reference_m)
    signal_precision = (
        gp.signal_scale
        * np.log1p(n_grid[:, None])
        * (0.55 + 0.45 * diversity)
        * (1.0 + gp.signal_exploration_gain * a_grid[None, :])
        * terms["eta"]
    )
    xi_state = xi_grid[:, None, None]
    total_precision = xi_state + signal_precision[None, :, :]
    next_xi = 1.0 / (gp.rho**2 / total_precision + gp.process_sd**2)
    posterior_mean_var = gp.rho**2 * (1.0 / xi_state - 1.0 / total_precision)
    posterior_mean_sd = np.sqrt(np.maximum(posterior_mean_var, 0.0))

    nodes, weights = np.polynomial.hermite.hermgauss(5)
    weights = weights / math.sqrt(math.pi)
    value = np.zeros((grid.n_mu, grid.n_xi))
    error_trace: list[float] = []
    start = time.perf_counter()
    mu_mean = gp.rho * mu_grid[:, None, None, None]
    next_xi_full = next_xi[None, :, :, :]
    sd_full = posterior_mean_sd[None, :, :, :]
    for iteration in range(1, max_iter + 1):
        continuation = np.zeros((grid.n_mu, grid.n_xi, grid.n_n, grid.n_a))
        for node, weight in zip(nodes, weights):
            next_mu = mu_mean + math.sqrt(2.0) * node * sd_full
            continuation += weight * interp2(mu_grid, xi_grid, value, next_mu, next_xi_full)
        q_value = reward + base.beta * continuation
        flat = q_value.reshape(grid.n_mu, grid.n_xi, -1)
        action = np.argmax(flat, axis=2)
        new_value = np.max(flat, axis=2)
        error = float(np.max(np.abs(new_value - value)))
        error_trace.append(error)
        value = new_value
        if error < tolerance:
            break
    else:
        raise RuntimeError(f"Gaussian online VFI failed: error={error:g}")

    continuation = np.zeros_like(reward)
    for node, weight in zip(nodes, weights):
        next_mu = mu_mean + math.sqrt(2.0) * node * sd_full
        continuation += weight * interp2(mu_grid, xi_grid, value, next_mu, next_xi_full)
    q_value = reward + base.beta * continuation
    flat = q_value.reshape(grid.n_mu, grid.n_xi, -1)
    action = np.argmax(flat, axis=2)
    bellman_value = np.max(flat, axis=2)
    residual = float(np.max(np.abs(bellman_value - value)))
    in_idx = action // grid.n_a
    ia_idx = action % grid.n_a
    idx = np.indices((grid.n_mu, grid.n_xi))
    policy = {
        "n": n_grid[in_idx],
        "a": a_grid[ia_idx],
        "s": best_s[idx[0], idx[1], in_idx, ia_idx],
        "psi": best_psi[idx[0], idx[1], in_idx, ia_idx],
        "support": best_support[idx[0], idx[1], in_idx, ia_idx],
        "quota_wedge": best_quota[idx[0], idx[1], in_idx, ia_idx],
        "signal_precision": signal_precision[in_idx, ia_idx],
    }
    natural = np.empty_like(policy["n"])
    for im, mu_value in enumerate(mu_grid):
        for ix in range(grid.n_xi):
            factor = (1.0 + gp.match_loading * math.tanh(mu_value)) * (
                1.0 - gp.exploration_cost * policy["a"][im, ix] ** 2
            )
            natural[im, ix] = online_natural_entry(
                base,
                diversity,
                gp.reference_k,
                gp.reference_m,
                policy["s"][im, ix],
                factor,
            )
    policy["natural_entry"] = natural
    policy["quota_gap"] = np.maximum(natural - policy["n"], 0.0)
    return {
        "value": value,
        "policy": policy,
        "mu_grid": mu_grid,
        "xi_grid": xi_grid,
        "n_grid": n_grid,
        "s_grid": s_grid,
        "a_grid": a_grid,
        "iterations": iteration,
        "successive_error": error_trace[-1],
        "bellman_residual": residual,
        "elapsed_seconds": time.perf_counter() - start,
        "max_zeta": float(np.nanmax(np.where(np.isfinite(reward_all), period["zeta"], np.nan))),
    }


def gaussian_policy_at(solution: dict[str, object], mu: float, xi: float) -> dict[str, float]:
    mu_grid = solution["mu_grid"]
    xi_grid = solution["xi_grid"]
    im = int(np.argmin(np.abs(mu_grid - mu)))
    ix = int(np.argmin(np.abs(xi_grid - xi)))
    return {key: float(value[im, ix]) for key, value in solution["policy"].items()}


def simulate_gaussian(
    base: Parameters,
    diversity: float,
    gp: GaussianOnlineParameters,
    solution: dict[str, object],
    periods: int = 90,
    seed: int = 20260902,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    theta, mu, xi = 0.20, 0.0, 3.0
    rows = []
    for t in range(periods):
        policy = gaussian_policy_at(solution, mu, xi)
        rows.append({"period": t, "theta": theta, "mu": mu, "xi": xi, **policy})
        precision = policy["signal_precision"]
        if precision > 1e-12:
            signal = theta + rng.normal(0.0, 1.0 / math.sqrt(precision))
            filtered_xi = xi + precision
            filtered_mu = (xi * mu + precision * signal) / filtered_xi
        else:
            filtered_xi, filtered_mu = xi, mu
        theta = gp.rho * theta + rng.normal(0.0, gp.process_sd)
        mu = gp.rho * filtered_mu
        xi = 1.0 / (gp.rho**2 / filtered_xi + gp.process_sd**2)
    return rows


def solve_markov(
    base: Parameters,
    diversity: float,
    hp: MarkovHotspotParameters,
    grid: MarkovGrid,
    k: float,
    m: float,
    tolerance: float = 1e-8,
    max_iter: int = 1500,
) -> dict[str, object]:
    p_grid = np.linspace(0.0, 1.0, grid.n_p)
    n_grid = np.linspace(0.0, base.n_bar, grid.n_n)
    s_grid = np.linspace(0.0, 1.0, grid.n_s)
    a_grid = np.linspace(0.0, 1.0, grid.n_a)
    belief = p_grid[:, None, None, None]
    n = n_grid[None, :, None, None]
    s = s_grid[None, None, :, None]
    share_a = a_grid[None, None, None, :]
    match_probability = belief * share_a + (1.0 - belief) * (1.0 - share_a)
    match_factor = 1.0 + hp.match_loading * (match_probability - 0.5)
    period = online_period(base, diversity, k, m, n, s, match_factor)
    reward_all = period["reward"]
    best_s_index = np.argmax(reward_all, axis=2)
    reward = np.max(reward_all, axis=2)
    selected = best_s_index[:, :, None, :]
    best_s = s_grid[best_s_index]
    best_psi = np.take_along_axis(period["psi"], selected, axis=2).squeeze(2)
    best_support = np.take_along_axis(period["support"], selected, axis=2).squeeze(2)
    best_quota = np.take_along_axis(period["quota_wedge"], selected, axis=2).squeeze(2)

    terms = structural_terms(base, diversity, k, m)
    comparative_design = hp.comparative_signal_floor + (1.0 - hp.comparative_signal_floor) * (
        4.0 * a_grid[None, :] * (1.0 - a_grid[None, :])
    )
    accuracy = 0.5 + hp.signal_cap * (
        1.0
        - np.exp(
            -hp.signal_scale
            * n_grid[:, None]
            * terms["eta"]
            * comparative_design
        )
    )
    accuracy = np.clip(accuracy, 0.5, 0.5 + hp.signal_cap)
    p = p_grid[:, None, None]
    gamma = accuracy[None, :, :]
    prob_signal_a = p * gamma + (1.0 - p) * (1.0 - gamma)
    prob_signal_b = 1.0 - prob_signal_a
    posterior_a = np.divide(p * gamma, prob_signal_a, out=np.broadcast_to(p, prob_signal_a.shape).copy(), where=prob_signal_a > 1e-14)
    posterior_b = np.divide(
        p * (1.0 - gamma),
        prob_signal_b,
        out=np.broadcast_to(p, prob_signal_b.shape).copy(),
        where=prob_signal_b > 1e-14,
    )
    lam = hp.switch_probability
    next_p_a = lam + (1.0 - 2.0 * lam) * posterior_a
    next_p_b = lam + (1.0 - 2.0 * lam) * posterior_b

    value = np.zeros(grid.n_p)
    error_trace: list[float] = []
    start = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        continuation = prob_signal_a * interp1(p_grid, value, next_p_a) + prob_signal_b * interp1(
            p_grid, value, next_p_b
        )
        q_value = reward + base.beta * continuation
        flat = q_value.reshape(grid.n_p, -1)
        action = np.argmax(flat, axis=1)
        new_value = np.max(flat, axis=1)
        error = float(np.max(np.abs(new_value - value)))
        error_trace.append(error)
        value = new_value
        if error < tolerance:
            break
    else:
        raise RuntimeError(f"Markov hotspot VFI failed: error={error:g}")

    continuation = prob_signal_a * interp1(p_grid, value, next_p_a) + prob_signal_b * interp1(
        p_grid, value, next_p_b
    )
    q_value = reward + base.beta * continuation
    flat = q_value.reshape(grid.n_p, -1)
    action = np.argmax(flat, axis=1)
    bellman_value = np.max(flat, axis=1)
    residual = float(np.max(np.abs(bellman_value - value)))
    in_idx = action // grid.n_a
    ia_idx = action % grid.n_a
    ip = np.arange(grid.n_p)
    policy = {
        "n": n_grid[in_idx],
        "a": a_grid[ia_idx],
        "test_intensity": 4.0 * a_grid[ia_idx] * (1.0 - a_grid[ia_idx]),
        "s": best_s[ip, in_idx, ia_idx],
        "psi": best_psi[ip, in_idx, ia_idx],
        "support": best_support[ip, in_idx, ia_idx],
        "quota_wedge": best_quota[ip, in_idx, ia_idx],
        "signal_accuracy": accuracy[in_idx, ia_idx],
    }
    natural = np.empty_like(policy["n"])
    for ip, belief_value in enumerate(p_grid):
        match_probability_value = belief_value * policy["a"][ip] + (1.0 - belief_value) * (
            1.0 - policy["a"][ip]
        )
        factor = 1.0 + hp.match_loading * (match_probability_value - 0.5)
        natural[ip] = online_natural_entry(
            base,
            diversity,
            k,
            m,
            policy["s"][ip],
            factor,
        )
    policy["natural_entry"] = natural
    policy["quota_gap"] = np.maximum(natural - policy["n"], 0.0)
    symmetry_residual = max(
        float(np.max(np.abs(policy["a"] + policy["a"][::-1] - 1.0))),
        float(np.max(np.abs(policy["n"] - policy["n"][::-1]))),
        float(np.max(np.abs(policy["s"] - policy["s"][::-1]))),
    )
    if symmetry_residual > 1e-10:
        raise RuntimeError(f"Markov policy symmetry check failed: {symmetry_residual:g}")
    return {
        "value": value,
        "policy": policy,
        "p_grid": p_grid,
        "n_grid": n_grid,
        "s_grid": s_grid,
        "a_grid": a_grid,
        "iterations": iteration,
        "successive_error": error_trace[-1],
        "bellman_residual": residual,
        "elapsed_seconds": time.perf_counter() - start,
        "structural_state": {"K": k, "M": m},
        "max_zeta": float(np.nanmax(np.where(np.isfinite(reward_all), period["zeta"], np.nan))),
        "policy_symmetry_residual": symmetry_residual,
    }


def markov_policy_at(solution: dict[str, object], belief: float) -> dict[str, float]:
    ip = int(np.argmin(np.abs(solution["p_grid"] - belief)))
    return {key: float(value[ip]) for key, value in solution["policy"].items()}


def simulate_markov(
    hp: MarkovHotspotParameters,
    solution: dict[str, object],
    periods: int = 100,
    seed: int = 20260902,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    hidden_a, belief = True, 0.50
    rows = []
    for t in range(periods):
        policy = markov_policy_at(solution, belief)
        rows.append({"period": t, "hotspot_a": int(hidden_a), "belief_a": belief, **policy})
        correct = rng.random() < policy["signal_accuracy"]
        signal_a = hidden_a if correct else not hidden_a
        gamma = policy["signal_accuracy"]
        if signal_a:
            denom = belief * gamma + (1.0 - belief) * (1.0 - gamma)
            posterior = belief * gamma / max(denom, 1e-14)
        else:
            denom = belief * (1.0 - gamma) + (1.0 - belief) * gamma
            posterior = belief * (1.0 - gamma) / max(denom, 1e-14)
        if rng.random() < hp.switch_probability:
            hidden_a = not hidden_a
        belief = hp.switch_probability + (1.0 - 2.0 * hp.switch_probability) * posterior
    return rows


def policy_reference_gaussian(solution: dict[str, object]) -> list[dict]:
    return [
        {"mu": mu, "xi": xi, **gaussian_policy_at(solution, mu, xi)}
        for mu, xi in [(-0.30, 3.0), (0.0, 3.0), (0.0, 10.0), (0.30, 10.0), (0.0, 18.0)]
    ]


def policy_reference_markov(solution: dict[str, object], label: str) -> list[dict]:
    return [
        {"structural_state": label, "belief_a": belief, **markov_policy_at(solution, belief)}
        for belief in [0.10, 0.30, 0.50, 0.70, 0.90]
    ]


def gaussian_sensitivity(base: Parameters, diversity: float) -> list[dict]:
    coarse = GaussianGrid(n_mu=19, n_xi=13, n_n=17, n_s=9, n_a=9)
    baseline = GaussianOnlineParameters()
    cases = [
        ("baseline_coarse", baseline),
        ("low_persistence", replace(baseline, rho=0.70)),
        ("high_persistence", replace(baseline, rho=0.96)),
        ("high_process_noise", replace(baseline, process_sd=0.24)),
        ("high_test_cost", replace(baseline, exploration_cost=0.12)),
    ]
    rows = []
    for name, params in cases:
        solution = solve_gaussian(base, diversity, params, coarse, tolerance=3e-7)
        low = gaussian_policy_at(solution, -0.30, 3.0)
        high = gaussian_policy_at(solution, -0.30, 14.0)
        rows.append(
            {
                "case": name,
                "rho": params.rho,
                "process_sd": params.process_sd,
                "exploration_cost": params.exploration_cost,
                "iterations": solution["iterations"],
                "bellman_residual": solution["bellman_residual"],
                "test_share_low_precision": low["a"],
                "test_share_high_precision": high["a"],
                "entry_low_precision": low["n"],
                "entry_high_precision": high["n"],
            }
        )
    return rows


def markov_sensitivity(base: Parameters, diversity: float) -> list[dict]:
    coarse = MarkovGrid(n_p=61, n_n=13, n_s=7, n_a=17)
    baseline = MarkovHotspotParameters()
    cases = [
        ("slow_switching", replace(baseline, switch_probability=0.02)),
        ("baseline", baseline),
        ("fast_switching", replace(baseline, switch_probability=0.18)),
        ("weak_feedback", replace(baseline, signal_cap=0.30)),
    ]
    rows = []
    for name, params in cases:
        solution = solve_markov(base, diversity, params, coarse, params.early_k, params.early_m, tolerance=3e-8)
        center = markov_policy_at(solution, 0.50)
        tilted = markov_policy_at(solution, 0.65)
        rows.append(
            {
                "case": name,
                "switch_probability": params.switch_probability,
                "signal_cap": params.signal_cap,
                "iterations": solution["iterations"],
                "bellman_residual": solution["bellman_residual"],
                "test_intensity_p50": center["test_intensity"],
                "share_A_p65": tilted["a"],
                "test_intensity_p65": tilted["test_intensity"],
                "entry_p50": center["n"],
                "entry_p65": tilted["n"],
            }
        )
    return rows


def plot_gaussian(solution: dict[str, object], path: list[dict]) -> None:
    mu_grid, xi_grid = solution["mu_grid"], solution["xi_grid"]
    policy = solution["policy"]
    imu = int(np.argmin(np.abs(mu_grid)))
    ixi = int(np.argmin(np.abs(xi_grid - 8.0)))
    t = np.array([row["period"] for row in path])
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.7), constrained_layout=True)
    axes[0, 0].plot(xi_grid, policy["a"][imu], color="#b03a2e", lw=2.0, label="test exposure")
    axes[0, 0].plot(xi_grid, policy["n"][imu] / np.max(policy["n"][imu]), color="#145da0", lw=1.8, ls="--", label="entry (normalized)")
    axes[0, 0].set(xlabel=r"posterior precision $\xi$", ylabel="policy", title=r"Policy at $\mu=0$")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(mu_grid, policy["n"][:, ixi], color="#145da0", lw=2.0, label="entry")
    axes[0, 1].plot(mu_grid, policy["support"][:, ixi], color="#7d3c98", lw=1.8, ls="-.", label="fixed support")
    axes[0, 1].set(xlabel=r"posterior mean $\mu$", ylabel="policy", title=rf"Policy at $\xi\approx{xi_grid[ixi]:.1f}$")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(t, [row["theta"] for row in path], color="#273746", lw=1.6, label=r"latent match $\theta_t$")
    axes[1, 0].plot(t, [row["mu"] for row in path], color="#145da0", lw=1.7, label=r"belief mean $\mu_t$")
    axes[1, 0].set(xlabel="business batch", ylabel="state", title="Online filtering path")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(t, [row["a"] for row in path], color="#b03a2e", lw=1.7, label="test exposure")
    axes[1, 1].plot(t, [row["n"] for row in path], color="#145da0", lw=1.7, label="entry")
    axes[1, 1].set(xlabel="business batch", ylabel="policy", title="Endogenous refresh and entry")
    axes[1, 1].legend(frameon=False)
    fig.savefig(FIGURE_DIR / "online_gaussian_learning_solution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_markov(early: dict[str, object], mature: dict[str, object], path: list[dict]) -> None:
    p_grid = early["p_grid"]
    t = np.array([row["period"] for row in path])
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.7), constrained_layout=True)
    axes[0, 0].plot(p_grid, early["policy"]["a"], color="#145da0", lw=2.0, label="attention share on A")
    axes[0, 0].plot(p_grid, early["policy"]["test_intensity"], color="#b03a2e", lw=1.8, ls="--", label="comparative test intensity")
    axes[0, 0].set(xlabel=r"belief $p=\Pr(h=A)$", ylabel="allocation", title="Early structural state")
    axes[0, 0].legend(loc="lower left", frameon=True, framealpha=0.9)
    axes[0, 1].plot(p_grid, early["policy"]["n"], color="#145da0", lw=2.0, label="early entry")
    axes[0, 1].plot(p_grid, mature["policy"]["n"], color="#7d3c98", lw=1.8, ls="-.", label="mature entry")
    axes[0, 1].plot(p_grid, mature["policy"]["quota_gap"], color="#b03a2e", lw=1.6, ls="--", label="mature quota gap")
    axes[0, 1].set(xlabel=r"belief $p=\Pr(h=A)$", ylabel="policy", title="Structural capital comparison")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].step(t, [row["hotspot_a"] for row in path], where="post", color="#273746", lw=1.2, label="true hotspot A")
    axes[1, 0].plot(t, [row["belief_a"] for row in path], color="#145da0", lw=1.7, label="posterior belief")
    axes[1, 0].set(xlabel="business batch", ylabel="state / belief", ylim=(-0.05, 1.05), title="Hotspot switching and tracking")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(t, [row["a"] for row in path], color="#145da0", lw=1.7, label="attention share on A")
    axes[1, 1].plot(t, [row["test_intensity"] for row in path], color="#b03a2e", lw=1.5, ls="--", label="test intensity")
    axes[1, 1].plot(t, np.array([row["n"] for row in path]) / 4.0, color="#7d3c98", lw=1.5, ls="-.", label="entry / 4")
    axes[1, 1].set(xlabel="business batch", ylabel="policy", ylim=(-0.05, 1.05), title="Adaptive recommendation policy")
    axes[1, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    fig.savefig(FIGURE_DIR / "online_markov_hotspot_solution.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
        }
    )
    diversity_stats = category_diversity(DATA_PATH)
    diversity = diversity_stats["normalized_shannon_entropy"]
    base = Parameters()

    gp, gg = GaussianOnlineParameters(), GaussianGrid()
    gaussian = solve_gaussian(base, diversity, gp, gg)
    gaussian_path = simulate_gaussian(base, diversity, gp, gaussian)
    gaussian_refs = policy_reference_gaussian(gaussian)
    gaussian_sens = gaussian_sensitivity(base, diversity)
    write_csv(OUTPUT_DIR / "gaussian_online_policy_reference.csv", gaussian_refs)
    write_csv(OUTPUT_DIR / "gaussian_online_transition_path.csv", gaussian_path)
    write_csv(OUTPUT_DIR / "gaussian_online_sensitivity.csv", gaussian_sens)
    plot_gaussian(gaussian, gaussian_path)

    hp, hg = MarkovHotspotParameters(), MarkovGrid()
    markov_early = solve_markov(base, diversity, hp, hg, hp.early_k, hp.early_m)
    markov_mature = solve_markov(base, diversity, hp, hg, hp.mature_k, hp.mature_m)
    markov_path = simulate_markov(hp, markov_early)
    markov_refs = policy_reference_markov(markov_early, "early") + policy_reference_markov(markov_mature, "mature")
    markov_sens = markov_sensitivity(base, diversity)
    write_csv(OUTPUT_DIR / "markov_hotspot_policy_reference.csv", markov_refs)
    write_csv(OUTPUT_DIR / "markov_hotspot_transition_path.csv", markov_path)
    write_csv(OUTPUT_DIR / "markov_hotspot_sensitivity.csv", markov_sens)
    plot_markov(markov_early, markov_mature, markov_path)

    gaussian_boundary_mu = float(np.mean(np.abs(np.array([row["mu"] for row in gaussian_path])) >= gg.mu_max - 1e-8))
    gaussian_boundary_xi = float(
        np.mean(
            (np.array([row["xi"] for row in gaussian_path]) <= gg.xi_min + 1e-8)
            | (np.array([row["xi"] for row in gaussian_path]) >= gg.xi_max - 1e-8)
        )
    )
    summary = {
        "interpretation": "partial calibration of online-learning mechanisms, not a production recommender estimate",
        "data_moment": diversity_stats,
        "gaussian_drift": {
            "parameters": asdict(gp),
            "grid": asdict(gg),
            "iterations": gaussian["iterations"],
            "bellman_residual": gaussian["bellman_residual"],
            "elapsed_seconds": gaussian["elapsed_seconds"],
            "max_zeta": gaussian["max_zeta"],
            "reference_policies": gaussian_refs,
            "sensitivity": gaussian_sens,
            "simulation": {
                "periods": len(gaussian_path),
                "mean_test_exposure": float(np.mean([row["a"] for row in gaussian_path])),
                "mean_entry": float(np.mean([row["n"] for row in gaussian_path])),
                "rmse_belief": float(
                    np.sqrt(np.mean([(row["theta"] - row["mu"]) ** 2 for row in gaussian_path]))
                ),
                "share_mu_on_grid_boundary": gaussian_boundary_mu,
                "share_xi_on_grid_boundary": gaussian_boundary_xi,
            },
        },
        "markov_hotspot": {
            "parameters": asdict(hp),
            "grid": asdict(hg),
            "early": {
                "iterations": markov_early["iterations"],
                "bellman_residual": markov_early["bellman_residual"],
                "elapsed_seconds": markov_early["elapsed_seconds"],
                "max_zeta": markov_early["max_zeta"],
                "policy_symmetry_residual": markov_early["policy_symmetry_residual"],
            },
            "mature": {
                "iterations": markov_mature["iterations"],
                "bellman_residual": markov_mature["bellman_residual"],
                "elapsed_seconds": markov_mature["elapsed_seconds"],
                "max_zeta": markov_mature["max_zeta"],
                "policy_symmetry_residual": markov_mature["policy_symmetry_residual"],
            },
            "reference_policies": markov_refs,
            "sensitivity": markov_sens,
            "simulation": {
                "periods": len(markov_path),
                "mean_test_intensity": float(np.mean([row["test_intensity"] for row in markov_path])),
                "mean_entry": float(np.mean([row["n"] for row in markov_path])),
                "classification_accuracy": float(
                    np.mean(
                        [
                            (row["belief_a"] >= 0.5) == bool(row["hotspot_a"])
                            for row in markov_path
                        ]
                    )
                ),
            },
        },
        "source_boundaries": {
            "economics": "the model borrows dynamic-experimentation and belief-state organization only",
            "algorithms": "all NumPy code and parameterizations are original to this project",
            "platform": "no claim that a named platform uses these exact updates or parameters",
        },
    }
    (OUTPUT_DIR / "online_learning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
