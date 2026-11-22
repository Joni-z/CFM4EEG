"""Figure 3 (encoder reuse bars) as an editable slide: bars, error bars, labels
and axes are all native shapes. Layout follows host-figure/plot.py, a 396 x 126
pt sheet with three panels. Usage: python make_host_pptx.py <out.pptx>
"""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *

# per-seed values from host-figure/data.csv (Native = three seeds, CroFreMo = one)
DATA = {
    ('TUEV', 'BIOT', 'Native'): [0.5038691, 0.5425016, 0.5507868],
    ('TUEV', 'BIOT', 'CroFreMo'): [0.5961068],
    ('TUEV', 'CBraMod', 'Native'): [0.5755035, 0.5366375, 0.5793720],
    ('TUEV', 'CBraMod', 'CroFreMo'): [0.6132694],
    ('IIIC', 'BIOT', 'Native'): [0.3537064, 0.3425021, 0.3709927],
    ('IIIC', 'BIOT', 'CroFreMo'): [0.3995526],
    ('IIIC', 'CBraMod', 'Native'): [0.3978563, 0.3957477, 0.3868954],
    ('IIIC', 'CBraMod', 'CroFreMo'): [],
    ('CHB-MIT', 'BIOT', 'Native'): [0.6491244, 0.5180293, 0.5312727],
    ('CHB-MIT', 'BIOT', 'CroFreMo'): [0.4737915],
    ('CHB-MIT', 'CBraMod', 'Native'): [0.4147999, 0.4545869, 0.0812583],
    ('CHB-MIT', 'CBraMod', 'CroFreMo'): [],
}
NATIVE = 'ACB5BF'; CRO = 'D99654'; DARK = '333333'; GRID = 'E2E2E2'
W, H = 396, 126
sh = Sheet(W, H, font='Arial')

# subplot boxes from subplots_adjust(left=.075,right=.985,bottom=.20,top=.73,wspace=.30)
aw = (0.985 - 0.075) * W / (3 + 2 * .30); gap = .30 * aw
ay0, ay1 = .20 * H, .73 * H
panels = [('TUEV', 0.075 * W), ('IIIC', 0.075 * W + aw + gap), ('CHB-MIT', 0.075 * W + 2 * (aw + gap))]

for ds, ax0 in panels:
    if ds == 'CHB-MIT': ylo, yhi, ticks, ylab = 0.0, 0.85, [0, .2, .4, .6, .8], 'AUC-PR'
    else: ylo, yhi, ticks, ylab = 0.3, 0.76, [.3, .4, .5, .6, .7], "Cohen’s κ"
    xlo, xhi = -.53, 1.53
    def px(x): return ax0 + (x - xlo) / (xhi - xlo) * aw
    def py(v): return ay0 + (v - ylo) / (yhi - ylo) * (ay1 - ay0)
    sh.text(ax0, ay1 + 9, ds, 7, ha='l')
    for t in ticks:                                   # horizontal grid, then spines
        sh.seg(ax0, py(t), ax0 + aw, py(t), GRID, .45)
        sh.seg(ax0 - 2, py(t), ax0, py(t), INK, .5)
        sh.text(ax0 - 4, py(t), f'{t:.1f}', 7, ha='r')
    sh.seg(ax0, ay0, ax0, ay1, INK, .55); sh.seg(ax0, ay0, ax0 + aw, ay0, INK, .55)
    if ds != 'CHB-MIT':                               # axis-break marks at the truncated bottom
        for dy in (-1.2, 1.2):
            sh.seg(ax0 - 2, ay0 + dy - 1.5, ax0 + 2, ay0 + dy + 1.5, DARK, .8)
    if ds == 'TUEV' or ds == 'CHB-MIT':
        lab = sh.text(ax0 - 20, (ay0 + ay1) / 2, ylab, 7, ha='c', w=44)
        lab.rotation = -90
    for hi, host in enumerate(['BIOT', 'CBraMod']):
        sh.seg(px(hi), ay0, px(hi), ay0 - 2, INK, .5)
        sh.text(px(hi), ay0 - 8, host, 7, ha='c')
        for vi, (variant, color) in enumerate([('Native', NATIVE), ('CroFreMo', CRO)]):
            x = hi + (-.16 if vi == 0 else .16); vals = DATA[(ds, host, variant)]
            if not vals:
                sh.text(px(x), py(.03 if ds == 'CHB-MIT' else .32), '—', 12, '777777', 'c'); continue
            m = statistics.mean(vals)
            sh.rect(px(x - .06), py(ylo), px(x + .06) - px(x - .06), py(m) - py(ylo), color)
            if len(vals) > 1:
                sd = statistics.stdev(vals); lo, hi_ = py(m - sd), py(m + sd)
                sh.seg(px(x), lo, px(x), hi_, DARK, .6)
                for yy in (lo, hi_): sh.seg(px(x) - 2, yy, px(x) + 2, yy, DARK, .6)
            top = py(max(vals)) + (.080 if vi == 0 else .025) * (ay1 - ay0) / (yhi - ylo)
            sh.text(px(x + (-.04 if vi == 0 else .04)), top + 4, f'{m:.3f}', 6, ha='c')

# legend, top centre
lx = .54 * W
sh.rect(lx - 50, H - 10, 16, 3, NATIVE); sh.text(lx - 31, H - 8.5, 'Native', 7)
sh.rect(lx + 2, H - 10, 16, 3, CRO); sh.text(lx + 21, H - 8.5, 'CroFreMo', 7)

print('wrote', sys.argv[1], 'shapes:', sh.save(sys.argv[1]))
