"""Figure 2 (architecture and reuse) as an editable slide. Layout follows
draw_sean_fig2.py: a 396 x 234 pt sheet, upper row 150 pt tall, lower row 80 pt.
Usage: python make_fig2_pptx.py <out.pptx>
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *

sh = Sheet(396, 234)
UP = 84  # the upper axes of the original sit on top of an 84 pt lower band


def U(y): return y + UP   # upper-row coordinate -> sheet coordinate


def tiles(x, y, w, h, values, color, rows=2):
    cols = len(values) // rows
    for k, v in enumerate(values):
        col, row = k % cols, k // cols; pw, ph = w / cols, h / rows
        sh.rect(x + col * pw + .4, y + row * ph + .4, pw - .8, ph - .8, tint(color, .10 + .58 * abs(v)))


# ---- upper row: tokenizer and encoder ------------------------------------------
sh.heading(3, U(143), 'a', 'CroFreMo tokenizer'); sh.heading(292, U(143), 'b', 'Encoder')
sh.rect(43, U(24), 242, 110, 'FAFBFC', LINE, .6, rounded=True)
for k in range(3):
    pts = [(5 + 26 * t, U(81 - 9 * k + 2.5 * math.sin((4 + k) * math.pi * t))) for t in [i / 60 for i in range(61)]]
    sh.curve(pts, BLUE, .75)
sh.text(18, U(98), 'EEG', 8, ha='c'); sh.text(18, U(43), [('C', I), (' × ', {}), ('T', I)], 7.5, ha='c')
sh.seg(32, U(72), 47, U(72), BLUE, .85, arrow=True)
sh.label(48, U(51), 32, 43, ['Sinc', 'bank', '×16'], BLUE, 7.4)
for y, c in [(112, GOLD), (77, BLUE), (40, PURPLE)]:
    sh.polyline([(80, U(72)), (88, U(72)), (88, U(y)), (97, U(y))], c, .75, arrow=True)
sh.label(97, U(101), 61, 22, ['Log-envelope', 'projection'], GOLD, 7.2)
sh.label(97, U(66), 61, 22, ['Waveform', 'projection'], BLUE, 7.2)
sh.label(97, U(29), 117, 22, ['Local moment → aligned phase'], PURPLE, 7.2)
sh.text(127, U(59), 'zero initialization', 6.5, MUTED, 'c')
av = [.3, .7, .9, .5, .8, .35, .6, .9]; wv = [.65, .4, .85, .25, .65, .85, .3, .5]
for y, c, values, s in [(105, GOLD, av, 'a'), (70, BLUE, wv, 'w')]:
    tiles(169, U(y), 42, 13, values, c)
    sh.seg(158, U(y + 6.5), 167, U(y + 6.5), c, .7, arrow=True)
    sh.text(190, U(y + 20), [(s, B)], 8, c, 'c')
sh.circle(218, U(96), 7, PURPLE, .7, fill='FFFFFF'); sh.text(218, U(96), '+', 10, PURPLE, 'c')
sh.circle(244, U(96), 10, PURPLE, .8, fill=tint(PURPLE, .06)); sh.text(244, U(96), '×', 13, PURPLE, 'c')
sh.polyline([(212, U(111)), (218, U(111)), (218, U(104))], GOLD, .8, arrow=True)
sh.polyline([(212, U(76)), (218, U(76)), (218, U(88))], BLUE, .8, arrow=True)
sh.text(220, U(66), [('i', I), ('w', I)], 7.7, BLUE, 'c')
sh.seg(225, U(96), 233, U(96), PURPLE, .8, arrow=True)
sh.text(230, U(108), [('q', I)], 8, PURPLE, 'c')
sh.polyline([(214, U(40)), (244, U(40)), (244, U(85))], PURPLE, .85, arrow=True)
sh.text(250, U(57), [('û', B)], 8, PURPLE)
for k in range(8):
    sh.rect(263 + k % 2 * 8, U(80 + k // 2 * 8), 7, 7, tint(PURPLE, .14 + .13 * (k % 3)), 'FFFFFF', .2)
sh.seg(254, U(96), 261, U(96), PURPLE, .85, arrow=True)
sh.text(271, U(122), [('h', B)], 8, PURPLE, 'c')
sh.text(159, U(13), 'One token per electrode, band and patch · width 128', 7.1, ha='c')

sh.text(343, U(132), [('C', I), (' × 16 × ', {}), ('P', I), (' × 128', {})], 7.4, ha='c')
sh.text(343, U(122), 'Pre-norm residual block · ×6', 6.5, MUTED, 'c')
sh.rect(297, U(38), 95, 78, PALE, LINE, .65, rounded=True)
for i, (name, c) in enumerate([('Temporal / RoPE', BLUE), ('Electrode', BLUE), ('Band', PURPLE), ('Feed-forward', INK)]):
    yy = 104 - 18 * i
    if i: sh.seg(301, U(yy + 9), 388, U(yy + 9), LINE, .45)
    if i == 2:
        for layer in [2, 1, 0]:
            sh.rect(303 + layer * 1.3, U(yy - 4 + layer * 2), 7, 5, tint(c, .45) if layer == 1 else 'FFFFFF', c, .35)
    else:
        for rr in range(3):
            for cc in range(3):
                selected = (rr == 1 if i == 0 else cc == 1 if i == 1 else True)
                sh.rect(303 + cc * 3, U(yy - 4 + rr * 3), 2.3, 2.3, tint(c, .68 if selected else .12))
    sh.text(318, U(yy), name, 7.05, c)
sh.polyline([(280, U(96)), (290, U(96)), (290, U(104)), (296, U(104))], PURPLE, .8, arrow=True)
sh.seg(345, U(38), 345, U(22), INK, .7, arrow=True)
sh.label(306, U(4), 78, 18, ['Mean pool → task head'], INK, 7.1)

# ---- lower row: reuse inside a host --------------------------------------------
sh.seg(3, 80, 393, 80, LINE, .65, dash=DASH)
sh.heading(3, 69, 'c', 'Tokenizer reuse')
sh.text(393, 69, 'BIOT / CBraMod · supervised from scratch', 6.8, MUTED, 'r')
sh.label(8, 25, 48, 26, ['Band', 'tokens'], PURPLE, 7.5)
sh.label(75, 25, 113, 26, ['Concatenate 16 bands', 'Average 4 time patches'], BLUE, 7)
sh.label(207, 25, 88, 26, ['Linear projection', '+ layer norm'], BLUE, 7)
sh.label(314, 25, 74, 26, ['Native host', '+ task head'], INK, 7)
for x, y in [(56, 74), (188, 206), (295, 313)]:
    sh.seg(x, 38, y, 38, BLUE, .75, arrow=True)
sh.text(198, 10, 'Adapt the token grid; retain the host encoder, positions and classifier', 7, MUTED, 'c')

print('wrote', sys.argv[1], 'shapes:', sh.save(sys.argv[1]))
