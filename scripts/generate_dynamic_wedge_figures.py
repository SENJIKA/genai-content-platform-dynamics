from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Transparent illustrative parameters. They are not estimated or calibrated.
N_MIN = 0.9
DELTA_N = 1.2
RHO = 0.5
A = 0.8
B = 0.4
MU = 0.8
LEARNING = 1.8
CREATIVE = 0.2
CONGESTION = 0.9
IMPLEMENTATION_SLOPE = 0.8


def natural_entry(m):
    return N_MIN + DELTA_N * (1.0 - np.exp(-RHO * m))


def target_entry(m):
    marginal_intercept = (
        A
        + B * (1.0 - np.exp(-MU * m))
        + LEARNING / (1.0 + m)
        + CREATIVE
    )
    return marginal_intercept / CONGESTION


def signed_payment(m):
    return IMPLEMENTATION_SLOPE * (target_entry(m) - natural_entry(m))


threshold = brentq(lambda value: float(signed_payment(value)), 0.0, 8.0)
m_grid = np.linspace(0.0, 8.0, 500)
n_zero = natural_entry(m_grid)
n_target = target_entry(m_grid)
payment = signed_payment(m_grid)

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)

ax = axes[0]
ax.axvspan(0.0, threshold, color="#dceeff", alpha=0.65)
ax.axvspan(threshold, 8.0, color="#fde2df", alpha=0.65)
ax.plot(m_grid, n_target, color="#145da0", linewidth=2.0, label=r"target $n^d(M)$")
ax.plot(m_grid, n_zero, color="#b03a2e", linewidth=2.0, linestyle="--", label=r"natural $n^{0,R}(M)$")
ax.axvline(threshold, color="black", linewidth=1.0, linestyle=":")
ax.text(threshold + 0.12, 3.02, r"$M^\dagger$", fontsize=10)
ax.set_xlabel(r"recommendation capital $M$")
ax.set_ylabel("entry scale")
ax.set_xlim(0.0, 8.0)
ax.set_ylim(0.7, 3.25)
ax.legend(frameon=False, loc="center right")

ax = axes[1]
support = np.maximum(payment, 0.0)
selection = np.maximum(-payment, 0.0)
ax.axvspan(0.0, threshold, color="#dceeff", alpha=0.65)
ax.axvspan(threshold, 8.0, color="#fde2df", alpha=0.65)
ax.plot(m_grid, support, color="#145da0", linewidth=2.0, label=r"support $I^+(M)$")
ax.plot(m_grid, selection, color="#b03a2e", linewidth=2.0, linestyle="--", label="selection intensity")
ax.axvline(threshold, color="black", linewidth=1.0, linestyle=":")
ax.axhline(0.0, color="#777777", linewidth=0.7)
ax.set_xlabel(r"recommendation capital $M$")
ax.set_ylabel("policy intensity")
ax.set_xlim(0.0, 8.0)
ax.set_ylim(-0.03, 1.88)
ax.legend(frameon=False, loc="upper right")

fig.savefig(FIGURE_DIR / "dynamic_wedge_comparative_statics.pdf", bbox_inches="tight")
plt.close(fig)

# A deterministic transition used only to display how the policy regime changes.
depreciation = 0.2
learning_rate = 0.5
periods = np.arange(0, 21)
m_path = np.empty_like(periods, dtype=float)
m_path[0] = 0.2
for index in range(len(periods) - 1):
    m_path[index + 1] = (
        (1.0 - depreciation) * m_path[index]
        + learning_rate * target_entry(m_path[index])
    )

crossing_period = int(np.argmax(m_path >= threshold))
fig, ax = plt.subplots(figsize=(6.4, 3.25), constrained_layout=True)
ax.axhspan(0.0, threshold, color="#dceeff", alpha=0.65, label="universal-support region")
ax.axhspan(threshold, max(5.1, float(m_path.max()) + 0.2), color="#fde2df", alpha=0.65, label="selection region")
ax.plot(periods, m_path, color="#273746", linewidth=2.1, marker="o", markersize=3.5)
ax.axhline(threshold, color="black", linewidth=1.0, linestyle=":")
ax.axvline(crossing_period, color="#666666", linewidth=0.9, linestyle="--")
ax.text(12.2, threshold + 0.09, r"$M^\dagger$")
ax.set_xlabel("business batch $t$")
ax.set_ylabel(r"recommendation capital $M_t$")
ax.set_xlim(0, periods[-1])
ax.set_ylim(0.0, max(5.1, float(m_path.max()) + 0.2))
ax.set_xticks(np.arange(0, periods[-1] + 1, 2))
ax.legend(frameon=False, loc="lower right")
fig.savefig(FIGURE_DIR / "dynamic_wedge_transition_path.pdf", bbox_inches="tight")
plt.close(fig)

print(f"M_dagger={threshold:.4f}; crossing_period={crossing_period}; terminal_M={m_path[-1]:.4f}")
