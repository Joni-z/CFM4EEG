"""Compact, evidence-bounded analyses; run with pubfig 0.3 + matplotlib.

The window sweep and fixed-checkpoint diagnostic are different experiments.
Do not pool them, attach invented uncertainty, or relabel inference changes
as retrained component controls. Input provenance is stored alongside data.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import pubfig as pf

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
DATA = json.loads((ROOT / "cfft_analysis_input.json").read_text())
CORAL, SLATE = "#F45161", "#647182"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": .55,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(.55)
        ax.spines[k].set_color("#9BA2AC")
    ax.tick_params(length=2, width=.55, pad=2, labelsize=7)
    ax.grid(axis="y", color="#E7E9ED", linewidth=.45)
    ax.set_axisbelow(True)

fig, axs = plt.subplots(1, 2, figsize=(3.25, 1.7), sharey=True)
for ax, dataset, title in zip(axs, ("tuev", "chbmit"), ("TUEV", "CHB-MIT")):
    rows = sorted([r for r in DATA["window"] if r["dataset"] == dataset], key=lambda r:r["window_s"])
    assert len(rows) == 3
    x = np.array([r["window_s"] for r in rows])
    y = np.array([[r["primary"] for r in rows], [r["balanced_acc"] for r in rows]])
    pf.line(y.T, x=x, ax=ax, legend_show=False, x_label="", y_label="",
            color_palette=[CORAL, SLATE], line_width=1.05, marker_size=3.6,
            show_full_box=False, show_x_grid=False, show_y_grid=True)
    ax.set_xscale("log")
    ax.set_xlim(.20, 6.4)
    ax.set_ylim(.56, .765)
    ax.set_xticks([.25, 1, 5], ["0.25", "1", "5"])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(FixedLocator([.60, .65, .70, .75]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title(title, fontweight="normal", pad=4, fontsize=8)
    ax.set_xlabel("Moment window (s)", fontsize=7.5, labelpad=3)
    ax.axvline(.25, color="#CDD1D8", lw=.6, ls=":", zorder=0)
    tidy(ax)
axs[0].set_ylabel("Score", fontsize=7.5, labelpad=3)
handles = axs[0].get_lines()[:2]
fig.legend(handles, ["Primary metric", "Balanced accuracy"], loc="upper center",
           bbox_to_anchor=(.52,1.01), ncol=2, frameon=False, fontsize=7,
           handlelength=1.6, handletextpad=.4, columnspacing=1.1)
fig.subplots_adjust(left=.14, right=.985, bottom=.24, top=.73, wspace=.16)
# Preserve the shared-legend strip after pubfig's export relayout.
axs[0]._pubfig_post_layout = lambda: fig.subplots_adjust(
    left=.14, right=.985, bottom=.24, top=.73, wspace=.16)
for ext in ("pdf", "png"):
    pf.save_figure(fig, OUT / ("cfft_window_sensitivity."+ext), width=82.55,
                   height_mm=43.18, raster_dpi=250, trim=False)
plt.close(fig)

# One unchanged predictor: complete set of six interventions, not selected arms.
modes = ["measured", "identity", "own_phase", "weights_only", "shuffled_Z", "envelope_only"]
available = DATA["interventions"]["metrics"]
labels = ["Unchanged", "Identity", "Own phase", "Weights only", "Shuffled moments", "Envelope only"]
values = np.array([available[m]["cohen_kappa"] for m in modes])
fig, ax = plt.subplots(figsize=(3.25, 1.7))
ax.barh(np.arange(6), values, height=.52, color=[CORAL]+[SLATE]*5, zorder=3)
ax.set_yticks(np.arange(6), labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, .77)
ax.set_xticks([0, .2, .4, .6])
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.set_xlabel("TUEV Cohen's kappa", fontsize=7.5, labelpad=3)
for y, value in enumerate(values):
    ax.text(value+.014, y, f"{value:.3f}", va="center", fontsize=7, color="#303642")
tidy(ax)
ax.grid(False, axis="y")
ax.grid(True, axis="x", color="#E7E9ED", linewidth=.45)
ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=.34, right=.98, bottom=.23, top=.98)
for ext in ("pdf", "png"):
    pf.save_figure(fig, OUT / ("cfft_fixed_predictor."+ext), width=82.55,
                   height_mm=43.18, raster_dpi=250, trim=False)
plt.close(fig)
