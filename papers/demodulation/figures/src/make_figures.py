"""Figures 2 and 3 for the demodulation paper.

Every number here is copied from archived result.json files; the run names are
listed beside each value so a reader can trace them (evidence IDs in
evidence/claims.md: C024, C025, C032, C026). Palette and mark rules follow the
dataviz reference instance (one hue for a single series, gray for context,
hairline solid grid, thin marks, rounded data ends, text in ink tokens,
selective direct labels).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..")

BLUE = "#2a78d6"      # categorical slot 1 (emphasis)
GRAY = "#898781"      # muted / de-emphasis
INK = "#0b0b0b"       # primary text
INK2 = "#52514e"      # secondary text
GRID = "#e1e0d9"      # hairline grid
AXIS = "#c3c2b7"      # baseline
SURF = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "pdf.fonttype": 42,
})


def strip(ax, keep_left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, "src", name + ".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
# TUEV test kappa vs labelled training windows, seed 0 (full-data cells: three-seed means).
# runs: tuev-paclock_v2_n2160 / n6720 / n20000 ; full = tuev-paclock_v2 (3-seed mean 0.708)
#       tuev-sparcnet_n2160 / n6720 / n20000 ; full = sparcnet 3-seed mean 0.489
n = np.array([2160, 6720, 20000, 68445])
ours = np.array([0.652, 0.710, 0.677, 0.708])
sparc = np.array([0.492, 0.553, 0.364, 0.489])

fig, ax = plt.subplots(figsize=(3.35, 2.25))
ax.set_xscale("log")
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.plot(n, sparc, color=GRAY, linewidth=2, solid_capstyle="round", zorder=2,
        marker="o", markersize=6, markerfacecolor=GRAY, markeredgecolor=SURF, markeredgewidth=1.4,
        label="SPaRCNet (conv. over raw signal)")
ax.plot(n, ours, color=BLUE, linewidth=2, solid_capstyle="round", zorder=3,
        marker="o", markersize=6, markerfacecolor=BLUE, markeredgecolor=SURF, markeredgewidth=1.4,
        label="Demodulated grid + three-axis reader")
ax.annotate("0.652", (n[0], ours[0]), textcoords="offset points", xytext=(0, 7), ha="center", color=INK, fontsize=7.5)
ax.annotate("0.708", (n[-1], ours[-1]), textcoords="offset points", xytext=(0, 7), ha="center", color=INK, fontsize=7.5)
ax.annotate("0.489", (n[-1], sparc[-1]), textcoords="offset points", xytext=(0, -11), ha="center", color=INK2, fontsize=7.5)
ax.set_xticks(n)
ax.set_xticklabels(["2,160", "6,720", "20,000", "68,445"])
ax.minorticks_off()
ax.set_ylim(0.3, 0.8)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_xlabel("labelled training windows (log scale)")
ax.set_ylabel("TUEV test Cohen's $\\kappa$")
strip(ax)
ax.legend(frameon=False, loc="lower left", fontsize=7, handlelength=1.6)
fig.tight_layout(pad=0.4)
save(fig, "fig2_label_curve")


# ---------------------------------------------------------------- Figure 3
# (a) TUEV kappa and (b) CHB-MIT AUC-PR by token content, three-axis reader, 8 learned bands, three seeds.
#   waveform only  = paclock_rawtok   (TUEV 0.536, CHB 0.667)
#   envelope only  = paclock_rot2     (TUEV 0.733, CHB 0.506)
#   both           = paclock_duplex   (TUEV 0.690, CHB 0.699)
# (c) rotation direction, three-axis 16-band model on TUEV (seed in label):
#   identity 0.729 (s0) range_identity_sean; measured 0.714 (s0) / 0.716 (s2) quad_range_20260916 (mean 0.715);
#   unaligned weights 0.643 (s2) quad_range_weight; different-window direction 0.601 (s0) range_sample_shuffle_sean;
#   own phase 0.532 (s2) quad_range_own
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.9]})

content = ["wave", "env.", "both"]
tuev = [0.536, 0.733, 0.690]
chb = [0.667, 0.506, 0.699]


def bars(ax, vals, ylim, ylabel, title):
    x = np.arange(len(vals))
    for xi, v in zip(x, vals):
        ax.add_patch(FancyBboxPatch((xi - 0.22, ylim[0]), 0.44, v - ylim[0],
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    linewidth=0, facecolor=BLUE, mutation_aspect=1 / 8, zorder=3))
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", color=INK, fontsize=7.5)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(content)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.axhline(ylim[0], color=AXIS, linewidth=0.6, zorder=2)
    strip(ax)


bars(axes[0], tuev, (0.4, 0.8), "Cohen's $\\kappa$", "(a) TUEV, token content")
bars(axes[1], chb, (0.4, 0.8), "AUC-PR", "(b) CHB-MIT, token content")

ax = axes[2]
labels = ["no transform, s0", "measured alignment, s0/s2", "weights, no alignment, s2",
          "other window's direction, s0", "band's own phase, s2"]
vals = [0.729, 0.715, 0.643, 0.601, 0.532]
y = np.arange(len(vals))[::-1]
for yi, v in zip(y, vals):
    ax.add_patch(FancyBboxPatch((0.4, yi - 0.28), v - 0.4, 0.56,
                                boxstyle="round,pad=0,rounding_size=0.006",
                                linewidth=0, facecolor=BLUE, mutation_aspect=8, zorder=3))
    ax.annotate(f"{v:.3f}", (v, yi), textcoords="offset points", xytext=(4, 0), va="center", color=INK, fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0.4, 0.8)
ax.set_ylim(-0.6, len(vals) - 0.4)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_xlabel("TUEV test Cohen's $\\kappa$")
ax.set_title("(c) direction u of the test transform", fontsize=8, color=INK, loc="left")
ax.axvline(0.4, color=AXIS, linewidth=0.6, zorder=2)
strip(ax, keep_left=False)
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig3_content_and_rotation")
print("wrote fig2_label_curve and fig3_content_and_rotation (pdf + png)")
