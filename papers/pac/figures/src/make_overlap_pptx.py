"""Appendix filter-overlap figure as an editable slide (396 x 136.8 pt sheet, three
panels), built from responses.npz and alpha-edges.csv in the given directory.
Usage: python make_overlap_pptx.py <data dir> <out.pptx>
"""
import sys, os, csv, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *

src, out = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(src, 'responses.npz')); f = d['frequency_hz']; Hm = d['trained_magnitude']
P = Hm ** 2; G = P @ P.T; nn_ = np.sqrt(np.diag(G)); O = G / np.outer(nn_, nn_)
rows = list(csv.DictReader(open(os.path.join(src, 'alpha-edges.csv'))))
pairs = [(int(r['target_band']), int(r['source_band']), float(r['log2_enrichment'])) for r in rows]
enr = np.array([p[2] for p in pairs]); ov = np.array([O[j - 1, i - 1] for j, i, _ in pairs])
adj = np.array([j - i == 1 for j, i, _ in pairs]); top = set(np.argsort(-enr)[:10].tolist())
rx_ = np.argsort(np.argsort(ov)); ry_ = np.argsort(np.argsort(enr)); rho = np.corrcoef(rx_, ry_)[0, 1]

blue = '0072B2'; orange = 'D55E00'; gray = '949DA7'; grid = 'E5E8EB'
W, H = 396, 136.8
sh = Sheet(W, H, font='Arial')
# subplots_adjust(left=.075,right=.985,bottom=.21,top=.86,wspace=.42)
aw = (.985 - .075) * W / (3 + 2 * .42); gx = .42 * aw
ay0, ay1 = .21 * H, .86 * H; ah = ay1 - ay0
X0 = [.075 * W + k * (aw + gx) for k in range(3)]


def axes(k, title, xlab, ylab):
    x0 = X0[k]
    sh.seg(x0, ay0, x0, ay1, INK, .55); sh.seg(x0, ay0, x0 + aw, ay0, INK, .55)
    sh.text(x0, ay1 + 7, title, 7, ha='l'); sh.text(x0 + aw / 2, ay0 - 15, xlab, 6.5, ha='c')
    t = sh.text(x0 - 19, ay0 + ah / 2, ylab, 6.5, ha='c', w=ah); t.rotation = -90
    return x0


def xtick(x0, x, s): sh.seg(x, ay0, x, ay0 - 2, INK, .5); sh.text(x, ay0 - 6.5, s, 6.5, ha='c')
def ytick(x0, y, s): sh.seg(x0 - 2, y, x0, y, INK, .5); sh.text(x0 - 3.5, y, s, 6.5, ha='r')


# (a) trained magnitude responses, blue -> gold by band
x0 = axes(0, 'Trained filter responses', 'Frequency (Hz; log scale)', '|H| / max')
fx = lambda v: x0 + (math.log(v) - math.log(.4)) / (math.log(90) - math.log(.4)) * aw
fy = lambda v: ay0 + v / 1.08 * ah
for v, s in [(.5, '0.5'), (2, '2'), (8, '8'), (32, '32'), (75, '75')]: xtick(x0, fx(v), s)
for v in (0, .2, .4, .6, .8, 1.0): ytick(x0, fy(v), f'{v:.1f}')
m = (f >= .4) & (f <= 90); fm = f[m]
for b in range(16):
    c = tint(GOLD, b / 15, base=BLUE) if False else None
    t = b / 15; c0, c1 = (0x28, 0x61, 0x7E), (0xA1, 0x6B, 0x29)
    col = ''.join('%02X' % round(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))
    # sample log-uniformly in frequency so the low bands are not drawn as polygons
    hb = Hm[b, m] / Hm[b].max()
    fs = np.geomspace(fm[0], fm[-1], 220); idx = np.unique(np.searchsorted(fm, fs).clip(0, len(fm) - 1))
    sh.curve([(fx(fm[i]), fy(hb[i])) for i in idx], col, .8)

# (b) adjacent-band overlap bars
x0 = axes(1, 'Adjacent-band overlap', 'Adjacent band pair', 'Power-response overlap')
bx = lambda b: x0 + (b - .3) / 15.4 * aw
by = lambda v: ay0 + v / 1.05 * ah
for v in (0, .2, .4, .6, .8, 1.0):
    sh.seg(x0, by(v), x0 + aw, by(v), grid, .5); ytick(x0, by(v), f'{v:.1f}')
for b in range(1, 16):
    v = O[b - 1, b]; w = .7 / 15.4 * aw
    sh.rect(bx(b) - w / 2, by(0), w, by(v) - by(0), blue if v > .3 else gray)
sh.seg(x0, by(.3), x0 + aw, by(.3), orange, .7, dash=DASH)
for b, s in [(1, '1–2'), (5, '5–6'), (9, '9–10'), (13, '13–14')]: xtick(x0, bx(b), s)

# (c) enrichment against overlap
x0 = axes(2, 'Weight versus overlap', 'Power-response overlap', 'log₂(α / uniform)')
sx = lambda v: x0 + (v + .03) / 1.06 * aw
ylo, yhi = -2.75, 1.15
sy = lambda v: ay0 + (v - ylo) / (yhi - ylo) * ah
for v in (-2, -1, 0, 1):
    sh.seg(x0, sy(v), x0 + aw, sy(v), grid, .5); ytick(x0, sy(v), str(v))
for v in (0, .5, 1.0): xtick(x0, sx(v), f'{v:.1f}')
sh.seg(x0, sy(0), x0 + aw, sy(0), gray, .6, dash=DASH)
for k, (j, i, e) in enumerate(pairs):
    if adj[k]: sh.marker(sx(ov[k]), sy(e), 2.6, blue, 's')
    else: sh.marker(sx(ov[k]), sy(e), 2.2, gray, 'o')
for k in top: sh.circle(sx(ov[k]), sy(enr[k]), 2.6, orange, .8)
sh.text(x0 + aw - 2, ay1 - 6, f'Spearman ρ = {rho:+.2f}', 6.3, ha='r')
lx, ly = x0 + aw - 4, ay0 + 22
sh.marker(lx - 58, ly, 2.2, gray, 'o'); sh.text(lx - 54, ly, 'non-adjacent', 5.8)
sh.marker(lx - 58, ly - 7.5, 2.6, blue, 's'); sh.text(lx - 54, ly - 7.5, 'adjacent (i = j−1)', 5.8)
sh.circle(lx - 58, ly - 15, 2.6, orange, .8); sh.text(lx - 54, ly - 15, 'ten largest weights', 5.8)

print('wrote', out, 'shapes:', sh.save(out))
