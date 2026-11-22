"""Appendix figure: do the up-weighted band pairs overlap spectrally?
(a) trained magnitude responses, (b) power-response overlap of adjacent bands,
(c) alpha enrichment against overlap for all 120 eligible pairs.
Inputs: responses.npz (codex reconstruction of SincBandpass at nfft 32768) and
alpha-edges.csv (mean alpha per pair on the same TUEV seed-0 checkpoint).
Usage: python draw_filter_overlap.py <dir with inputs> <out.pdf>
"""
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

src, out = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(src / 'responses.npz'); f = d['frequency_hz']; H = d['trained_magnitude']
P = H ** 2; G = P @ P.T; n = np.sqrt(np.diag(G)); O = G / np.outer(n, n)
rows = list(csv.DictReader((src / 'alpha-edges.csv').open()))
pairs = [(int(r['target_band']), int(r['source_band']), float(r['log2_enrichment'])) for r in rows]
enr = np.array([p[2] for p in pairs]); ov = np.array([O[j - 1, i - 1] for j, i, _ in pairs])
adj = np.array([j - i == 1 for j, i, _ in pairs])
top = np.argsort(-enr)[:10]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7,
                     'axes.labelsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'axes.linewidth': .55, 'pdf.fonttype': 42})
blue = '#0072B2'; orange = '#D55E00'; gray = '#949DA7'; grid = '#E5E8EB'
fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.9))
fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.86, wspace=.42)
for ax, t in zip(axs, ['Trained filter responses', 'Adjacent-band overlap', 'Weight versus overlap']):
    ax.set_title(t, loc='left', fontweight='normal', pad=6)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2, pad=2); ax.set_axisbelow(True)

# (a) magnitude responses, coloured from low band (blue) to high band (gold)
ax = axs[0]; cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bg', ['#28617E', '#8A8F7A', '#A16B29'])
m = (f >= .4) & (f <= 90)
for b in range(16):
    ax.plot(f[m], H[b, m] / H[b].max(), color=cmap(b / 15), lw=.8)
ax.set_xscale('log'); ax.set_xlim(.4, 90); ax.set_ylim(0, 1.08)
ax.set_xticks([.5, 2, 8, 32, 75], ['0.5', '2', '8', '32', '75']); ax.minorticks_off()
ax.set_xlabel('Frequency (Hz; log scale)'); ax.set_ylabel('|H| / max')

# (b) cosine overlap of adjacent power responses
ax = axs[1]; b = np.arange(1, 16); o_adj = np.array([O[k - 1, k] for k in b])
ax.bar(b, o_adj, width=.7, color=[blue if v > .3 else gray for v in o_adj])
ax.axhline(.3, color=orange, lw=.7, ls='--')
ax.set_xticks([1, 5, 9, 13], ['1–2', '5–6', '9–10', '13–14'])
ax.set_xlabel('Adjacent band pair'); ax.set_ylabel('Power-response overlap'); ax.set_ylim(0, 1.05)
ax.grid(axis='y', color=grid, lw=.5)

# (c) enrichment against overlap, all 120 eligible pairs
ax = axs[2]
ax.scatter(ov[~adj], enr[~adj], s=7, color=gray, lw=0, label='non-adjacent')
ax.scatter(ov[adj], enr[adj], s=9, color=blue, marker='s', lw=0, label='adjacent (i = j−1)')
ax.scatter(ov[top], enr[top], s=34, facecolors='none', edgecolors=orange, lw=.8, label='ten largest weights')
ax.axhline(0, color=gray, lw=.6, ls='--')
rx = np.argsort(np.argsort(ov)); ry = np.argsort(np.argsort(enr)); rho = np.corrcoef(rx, ry)[0, 1]
ax.text(.98, .93, f'Spearman ρ = {rho:+.2f}', transform=ax.transAxes, ha='right', fontsize=6.3)
ax.set_xlim(-.03, 1.03); ax.set_xlabel('Power-response overlap')
ax.set_ylabel('log₂(α / uniform)'); ax.grid(axis='y', color=grid, lw=.5)
ax.legend(loc='lower right', frameon=False, fontsize=5.8, handlelength=1.1, borderpad=.2, labelspacing=.3)

fig.savefig(out); fig.savefig(out.with_suffix('.png'), dpi=300)
print('wrote', out, 'rho', round(rho, 3), 'adjacent overlaps', np.round(o_adj, 2).tolist())
