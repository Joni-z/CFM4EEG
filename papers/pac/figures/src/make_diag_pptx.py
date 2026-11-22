"""Figure 4 (inside the tokenizer) as an editable slide: four panels, every
interval, cell, step and marker a native shape. Data are the frozen TUEV seed-0
checkpoint diagnostics (smoke-bands.csv, alpha-edges.csv, derived.json). Sheet
396 x 295.2 pt, laid out like plot_diagnostics.py. Usage: python make_diag_pptx.py <out.pptx>
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *

BANDS = [  # band, initial_low, initial_high, learned_low, learned_high
    (1, .49999997, .68387336, .5008481, .70824033), (2, .68387336, .9353656, .6577398, .9171002),
    (3, .9353656, 1.2793432, .8970173, 1.2503605), (4, 1.2793432, 1.7498176, 1.2067361, 1.6667317),
    (5, 1.7498176, 2.3933074, 1.6594424, 2.2991316), (6, 2.3933074, 3.2734385, 2.2448564, 3.1042364),
    (7, 3.2734385, 4.477235, 2.99071, 4.084759), (8, 4.477235, 6.123725, 4.1434345, 5.715159),
    (9, 6.123725, 8.375705, 5.6345897, 7.745177), (10, 8.375705, 11.455844, 7.7323833, 10.626833),
    (11, 11.455844, 15.668694, 10.575079, 14.504265), (12, 15.668694, 21.430805, 14.3652, 19.633354),
    (13, 21.430805, 29.311913, 19.762697, 27.031229), (14, 29.311914, 40.091274, 26.983267, 36.916557),
    (15, 40.091274, 54.834713, 36.78828, 50.287724), (16, 54.834713, 75.0, 50.22896, 68.71762)]
ENRICH = {  # (target j, source i): log2(alpha / uniform), from alpha-edges.csv
 (2,1):0.0,(3,1):-.0549,(3,2):.0529,(4,1):-.1434,(4,2):-.0733,(4,3):.1942,(5,1):-.3054,(5,2):-.2119,
 (5,3):.0090,(5,4):.4018,(6,1):-.5359,(6,2):-.4262,(6,3):-.1927,(6,4):.1596,(6,5):.6545,(7,1):-.7477,
 (7,2):-.6303,(7,3):-.3883,(7,4):-.0411,(7,5):.4224,(7,6):.7503,(8,1):-.8818,(8,2):-.7606,(8,3):-.5179,
 (8,4):-.1778,(8,5):.2712,(8,6):.5982,(8,7):.6452,(9,1):-.8933,(9,2):-.7708,(9,3):-.5265,(9,4):-.1896,
 (9,5):.2523,(9,6):.5705,(9,7):.6123,(9,8):.1371,(10,1):-.8434,(10,2):-.7183,(10,3):-.4732,(10,4):-.1348,
 (10,5):.3086,(10,6):.6292,(10,7):.6813,(10,8):.2109,(10,9):-.6052,(11,1):-.8339,(11,2):-.7062,
 (11,3):-.4549,(11,4):-.1129,(11,5):.3422,(11,6):.6841,(11,7):.7731,(11,8):.4053,(11,9):-.4480,
 (11,10):-.9130,(12,1):-.8261,(12,2):-.6952,(12,3):-.4400,(12,4):-.0909,(12,5):.3748,(12,6):.7373,
 (12,7):.8634,(12,8):.5997,(12,9):-.1836,(12,10):-.8105,(12,11):-1.3469,(13,1):-.8688,(13,2):-.7338,
 (13,3):-.4733,(13,4):-.1185,(13,5):.3589,(13,6):.7462,(13,7):.9195,(13,8):.8007,(13,9):.2569,
 (13,10):-.5458,(13,11):-1.2018,(13,12):-1.7286,(14,1):-.9277,(14,2):-.7905,(14,3):-.5264,(14,4):-.1706,
 (14,5):.3095,(14,6):.7048,(14,7):.9060,(14,8):.8977,(14,9):.6409,(14,10):.0287,(14,11):-.8532,
 (14,12):-1.5682,(14,13):-2.0762,(15,1):-.9866,(15,2):-.8475,(15,3):-.5834,(15,4):-.2275,(15,5):.2538,
 (15,6):.6542,(15,7):.8597,(15,8):.8908,(15,9):.7937,(15,10):.5201,(15,11):-.1600,(15,12):-1.1680,
 (15,13):-1.8338,(15,14):-2.3587,(16,1):-1.0369,(16,2):-.8979,(16,3):-.6345,(16,4):-.2824,(16,5):.1892,
 (16,6):.5761,(16,7):.7707,(16,8):.7787,(16,9):.7013,(16,10):.6925,(16,11):.5861,(16,12):-.1871,
 (16,13):-1.2869,(16,14):-1.9919,(16,15):-2.5210}
RATIO_BINS = [1, 2, 4, 8, 16, 32, 64, 128]
MASS = [.3783, .2198, .1712, .1284, .0743, .0210, .0070]
UNIFORM = [.3758, .2202, .1513, .1264, .0785, .0342, .0137]
INTERV = [('Unchanged', .7139), ('Envelope only', .6912), ('Weights only', .3799),
          ('Shuffled Z', .2407), ('Own phase', .1875)]

blue = '0072B2'; orange = 'D55E00'; gray = '949DA7'; grid = 'E5E8EB'
W, H = 396, 295.2
sh = Sheet(W, H, font='Arial')

# gridspec(2,2,left=.105,right=.925,bottom=.12,top=.935,wspace=.62,hspace=.65)
aw = (.925 - .105) * W / (2 + .62); gx = .62 * aw
ah = (.935 - .12) * H / (2 + .65); gy = .65 * ah
X0 = [.105 * W, .105 * W + aw + gx]; Y0 = [.935 * H - ah, .12 * H]   # row 0 = top


def axes(col, row, title, xlab, ylab):
    x0, y0 = X0[col], Y0[row]
    sh.seg(x0, y0, x0, y0 + ah, INK, .55); sh.seg(x0, y0, x0 + aw, y0, INK, .55)
    sh.text(x0, y0 + ah + 9, title, 7, ha='l')
    sh.text(x0 + aw / 2, y0 - 16, xlab, 6.5, ha='c')
    if ylab:
        t = sh.text(x0 - 22, y0 + ah / 2, ylab, 6.5, ha='c', w=ah); t.rotation = -90
    return x0, y0


def xtick(x0, y0, x, s): sh.seg(x, y0, x, y0 - 2, INK, .5); sh.text(x, y0 - 6.5, s, 7, ha='c')
def ytick(x0, y0, y, s): sh.seg(x0 - 2, y, x0, y, INK, .5); sh.text(x0 - 3.5, y, s, 7, ha='r')


# ---- panel 1: cutoff intervals on a log axis ------------------------------------
x0, y0 = axes(0, 0, 'Frequency organization', 'Cutoff interval (Hz; log scale)', 'Band index')
fx = lambda f: x0 + (math.log(f) - math.log(.4)) / (math.log(90) - math.log(.4)) * aw
fy = lambda b: y0 + (b - .25) / (17 - .25) * ah
for f, s in [(.5, '0.5'), (2, '2'), (8, '8'), (32, '32'), (75, '75')]:
    sh.seg(fx(f), y0, fx(f), y0 + ah, grid, .5); xtick(x0, y0, fx(f), s)
for b in (1, 4, 8, 12, 16): ytick(x0, y0, fy(b), str(b))
for b, il, ih, ll, lh in BANDS:
    sh.seg(fx(il), fy(b - .16), fx(ih), fy(b - .16), gray, 1.5)
    sh.marker(fx(math.sqrt(il * ih)), fy(b - .16), 2.6, gray, 'o')
    sh.seg(fx(ll), fy(b + .16), fx(lh), fy(b + .16), blue, 1.5)
    sh.marker(fx(math.sqrt(ll * lh)), fy(b + .16), 2.6, blue, 's')
lx, ly = x0 + 6, y0 + ah - 6
sh.seg(lx, ly, lx + 9, ly, gray, 1); sh.marker(lx + 4.5, ly, 2.4, gray, 'o'); sh.text(lx + 12, ly, 'Initial', 6.5)
sh.seg(lx, ly - 8, lx + 9, ly - 8, blue, 1); sh.marker(lx + 4.5, ly - 8, 2.4, blue, 's'); sh.text(lx + 12, ly - 8, 'Trained', 6.5)

# ---- panel 2: enrichment heat map with colour bar --------------------------------
hw = aw * .88                                          # matplotlib steals ~12% for the colour bar
x0, y0 = X0[1], Y0[0]
sh.seg(x0, y0, x0, y0 + ah, INK, .55); sh.seg(x0, y0, x0 + hw, y0, INK, .55)
sh.text(x0, y0 + ah + 9, 'Cross-band weighting', 7, ha='l')
sh.text(x0 + hw / 2, y0 - 16, [('Source band ', {}), ('i', I)], 6.5, ha='c')
t = sh.text(x0 - 22, y0 + ah / 2, [('Target band ', {}), ('j', I)], 6.5, ha='c', w=ah); t.rotation = -90
cw, ch = hw / 16, ah / 16
BOUND = 2.75


def rdbu(v):
    """Approximate matplotlib RdBu_r on [-BOUND, BOUND]: blue negative, red positive."""
    stops = [(-1, (5, 48, 97)), (-.5, (67, 147, 195)), (0, (247, 247, 247)), (.5, (214, 96, 77)), (1, (103, 0, 31))]
    u = max(-1, min(1, v / BOUND))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= u <= b:
            w = (u - a) / (b - a)
            return ''.join('%02X' % round(ca[k] + (cb[k] - ca[k]) * w) for k in range(3))
    return 'F7F7F7'


for j in range(1, 17):
    for i in range(1, 17):
        v = ENRICH.get((j, i))
        sh.rect(x0 + (i - 1) * cw, y0 + (j - 1) * ch, cw + .15, ch + .15, 'F1F3F5' if v is None else rdbu(v))
for b in (1, 4, 8, 12, 16):
    xtick(x0, y0, x0 + (b - .5) * cw, str(b)); ytick(x0, y0, y0 + (b - .5) * ch, str(b))
cbx, cbw, cbh = x0 + hw + 6, 5, ah * .92; cby = y0 + (ah - cbh) / 2
for k in range(40):
    v = -BOUND + (k + .5) / 40 * 2 * BOUND
    sh.rect(cbx, cby + k / 40 * cbh, cbw, cbh / 40 + .15, rdbu(v))
sh.rect(cbx, cby, cbw, cbh, None, INK, .4)
for v in (-2, -1, 0, 1, 2):
    yy = cby + (v + BOUND) / (2 * BOUND) * cbh
    sh.seg(cbx + cbw, yy, cbx + cbw + 2, yy, INK, .4); sh.text(cbx + cbw + 3.5, yy, str(v), 6)
t = sh.text(cbx + cbw + 17, cby + cbh / 2, 'log₂(weight / uniform)', 6.5, ha='c', w=cbh); t.rotation = -90

# ---- panel 3: weight mass by frequency ratio --------------------------------------
x0, y0 = axes(0, 1, 'Frequency separation', 'Center-frequency ratio  fⱼ / fᵢ', 'Fraction of total weight')
rx = lambda r: x0 + math.log2(r) / 7 * aw
ymax = max(max(MASS), max(UNIFORM)) * 1.32
ry = lambda v: y0 + v / ymax * ah
for v in (0, .1, .2, .3, .4):
    sh.seg(x0, ry(v), x0 + aw, ry(v), grid, .5); ytick(x0, y0, ry(v), f'{v:.1f}')
for r in RATIO_BINS: xtick(x0, y0, rx(r), str(r))


def stairs(vals, color, lw, dash=None):
    pts = []
    for k, v in enumerate(vals):
        pts += [(rx(RATIO_BINS[k]), ry(v)), (rx(RATIO_BINS[k + 1]), ry(v))]
    sh.curve(pts, color, lw, dash=dash)


stairs(UNIFORM, gray, 1.3, DASH); stairs(MASS, blue, 1.5)
lx, ly = x0 + aw - 6, y0 + ah - 6
sh.seg(lx - 12, ly, lx, ly, gray, 1.3, dash=DASH); sh.text(lx - 15, ly, 'Uniform sources', 6.5, ha='r')
sh.seg(lx - 12, ly - 8, lx, ly - 8, blue, 1.5); sh.text(lx - 15, ly - 8, 'Measured α', 6.5, ha='r')

# ---- panel 4: test-time interventions ----------------------------------------------
x0, y0 = axes(1, 1, 'Test-time interventions', "Cohen’s κ", None)
kx = lambda v: x0 + v / .85 * aw
ky = lambda i: y0 + ah - (i + .6) / 5.25 * ah
for v in (0, .2, .4, .6, .8):
    sh.seg(kx(v), y0, kx(v), y0 + ah, grid, .5); xtick(x0, y0, kx(v), f'{v:.1f}')
sh.seg(kx(INTERV[0][1]), y0, kx(INTERV[0][1]), y0 + ah, gray, .8, dash=DASH)
for i, (name, v) in enumerate(INTERV):
    sh.seg(x0 - 2, ky(i), x0, ky(i), INK, .5); sh.text(x0 - 3.5, ky(i), name, 6.5, ha='r')
    sh.marker(kx(v), ky(i), 4, blue if i == 0 else orange, 'o' if i == 0 else 's')
    sh.text(kx(v) + 4.5, ky(i), f'{v:.3f}', 6.5)

print('wrote', sys.argv[1], 'shapes:', sh.save(sys.argv[1]))
