"""Figure 1 as an editable PowerPoint slide: every curve, circle, arrow and label
is a native shape, laid out on the same 396 x 151 pt sheet as the PDF, drawn at
2x so the slide is 11 in wide. Usage: python make_fig1_pptx.py <out.pptx>
"""
import sys, math
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from lxml import etree

OUT = sys.argv[1]
W, H, S = 396, 151, 2.0            # figure sheet in pt, drawn at 2x
INK = '202A35'; MUTED = '596575'; LINE = 'CDD5DD'; PALE = 'F4F6F8'
BLUE = '28617E'; GOLD = 'A16B29'; PURPLE = '75516F'; GREY = '9AA6AE'
FONT = 'Times New Roman'

prs = Presentation()
prs.slide_width = Pt(W * S); prs.slide_height = Pt(H * S)
slide = prs.slides.add_slide(prs.slide_layouts[6])
sh = slide.shapes
rgb = lambda h: RGBColor.from_string(h)
X = lambda x: Pt(x * S)
Y = lambda y: Pt((H - y) * S)           # figure y grows upward; slides grow downward


def stroke(shape, color, lw, dash=None):
    ln = shape.line; ln.color.rgb = rgb(color); ln.width = Pt(lw * S)
    if dash: ln.dash_style = dash


def nofill(shape):
    shape.fill.background()


def arrowhead(shape, size='med'):
    ln = shape.line._get_or_add_ln()
    tail = ln.find(qn('a:tailEnd'))
    if tail is None:
        tail = etree.SubElement(ln, qn('a:tailEnd'))
    tail.set('type', 'triangle'); tail.set('w', size); tail.set('len', size)


def seg(x1, y1, x2, y2, color, lw, arrow=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, X(x1), Y(y1), X(x2), Y(y2))
    stroke(c, color, lw)
    if arrow: arrowhead(c)
    return c


def polyline(pts, color, lw, arrow=False):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        seg(x1, y1, x2, y2, color, lw, arrow=arrow and (x2, y2) == pts[-1])


def curve(pts, color, lw):
    """Free-form polyline in figure coordinates."""
    b = sh.build_freeform(X(pts[0][0]), Y(pts[0][1]), scale=1.0)
    b.add_line_segments([(X(x), Y(y)) for x, y in pts[1:]], close=False)
    s = b.convert_to_shape(); nofill(s); stroke(s, color, lw); return s


def circle(cx, cy, r, color, lw, fill=None, dash=None):
    s = sh.add_shape(MSO_SHAPE.OVAL, X(cx - r), Y(cy + r), Pt(2 * r * S), Pt(2 * r * S))
    if fill is None: nofill(s)
    else: s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    stroke(s, color, lw, dash); s.shadow.inherit = False; return s


def rect(x, y, w, h, fill, line=None, lw=.3):
    s = sh.add_shape(MSO_SHAPE.RECTANGLE, X(x), Y(y + h), Pt(w * S), Pt(h * S))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line: stroke(s, line, lw)
    else: s.line.fill.background()
    s.shadow.inherit = False; return s


def arc(cx, cy, r, t1, t2, color, lw, arrow=False):
    """Counter-clockwise arc from math angle t1 to t2 (degrees), as a polyline."""
    n = max(8, int((t2 - t1) / 6))
    pts = [(cx + r * math.cos(math.radians(t1 + (t2 - t1) * k / n)),
            cy + r * math.sin(math.radians(t1 + (t2 - t1) * k / n))) for k in range(n + 1)]
    s = curve(pts, color, lw)
    if arrow: arrowhead(s, 'sm')
    return s


def text(x, y, runs, size, color=INK, ha='l', bold=False, w=None):
    """runs: str or list of (text, {'i':italic,'sub':subscript,'b':bold,'c':color})."""
    if isinstance(runs, str): runs = [(runs, {})]
    plain = ''.join(t for t, _ in runs)
    w = w or (len(plain) * size * 0.52 + 6)
    h = size * 1.5
    left = {'l': x, 'c': x - w / 2, 'r': x - w}[ha]
    tb = sh.add_textbox(X(left), Y(y + h / 2), Pt(w * S), Pt(h * S))
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[ha]
    for t, st in runs:
        r = p.add_run(); r.text = t; f = r.font
        f.name = FONT; f.size = Pt(size * S); f.bold = st.get('b', bold); f.italic = st.get('i', False)
        f.color.rgb = rgb(st.get('c', color))
        if st.get('sub'):
            r._r.get_or_add_rPr().set('baseline', '-25000')
    return tb


I = {'i': True}; SUB = {'i': True, 'sub': True}; B = {'b': True}


def heading(x, y, letter, title):
    text(x, y, letter, 9, bold=True); text(x + 12, y, title, 8.3, bold=True)


def phase_circle(cx, cy, r, pref, cur, color, name):
    circle(cx, cy, r, LINE, .6, fill='FFFFFF')
    for ang in (0, 90, 180, 270):
        t = math.radians(ang)
        seg(cx + (r - 1.6) * math.cos(t), cy + (r - 1.6) * math.sin(t),
            cx + r * math.cos(t), cy + r * math.sin(t), LINE, .45)
    tp, tc = math.radians(pref), math.radians(cur)
    seg(cx, cy, cx + r * math.cos(tp), cy + r * math.sin(tp), color, 1.0)
    circle(cx + r * math.cos(tc), cy + r * math.sin(tc), 1.7, color, .3, fill=color)
    arc(cx, cy, r + 4.5, pref, cur, color, .7)
    mid = math.radians((pref + cur) / 2)
    text(cx + (r + 9.5) * math.cos(mid), cy + (r + 9.5) * math.sin(mid), f'+{cur - pref}°', 6.8, color, 'c')
    text(cx, cy - r - 7, name, 6.9, color, 'c')


# ---- frame ---------------------------------------------------------------------
for x, c, title in [(3, 'a', 'Measure the coupling'), (141, 'b', 'Align the sources'),
                    (275, 'c', 'Rotate the token')]:
    heading(x, 143, c, title)
for x in (131, 265):
    seg(x, 12, x, 132, LINE, .5)

# ---- a: slow phase organizes the fast envelope --------------------------------
N = 260
def slow_curve():
    ax, ay, aw, ah = 7, 88, 113, 38
    return [(ax + aw * k / N, ay + ah * (math.cos(4 * math.pi * k / N) + 1.65) / 3.3) for k in range(N + 1)]
def fast_curves():
    ax, ay, aw, ah = 7, 46, 113, 39
    env = lambda t: 1 + .62 * math.cos(4 * math.pi * t - .65)
    y = lambda v: ay + ah * (v + 1.85) / 3.7
    M = 700
    sig = [(ax + aw * k / M, y(env(k / M) * math.cos(48 * math.pi * k / M))) for k in range(M + 1)]
    up = [(ax + aw * k / N, y(env(k / N))) for k in range(N + 1)]
    dn = [(ax + aw * k / N, y(-env(k / N))) for k in range(N + 1)]
    return sig, up, dn
rect(7 + 113 * .44, 46, 113 * .22, 80, 'F8F3EC')          # shaded patch behind both traces
curve(slow_curve(), BLUE, .85)
sig, up, dn = fast_curves()
curve(sig, GREY, .5); curve(up, GOLD, 1.05); curve(dn, GOLD, .65)
text(8, 128, [('Slow band ', {}), ('i', I), (': phase ', {}), ('φ', I), ('i', SUB)], 7.0, BLUE)
text(8, 88, [('Fast band ', {}), ('j', I), (': envelope ', {}), ('A', I), ('j', SUB)], 7.0, GOLD)
text(63, 33, [('per patch:  ', {}), ('Z', I), ('ij', SUB), (' = ⟨ ', {}), ('p', I), ('i', SUB),
              (' (', {}), ('A', I), ('j', SUB), (' − ', {}), ('Ā', I), ('j', SUB), (') ⟩', {})], 7.0, INK, 'c')
text(63, 21, [('|', {}), ('Z', I), ('ij', SUB), ('| strength  ·  ∠', {}), ('Z', I), ('ij', SUB),
              (' preferred phase', {})], 6.5, MUTED, 'c')

# ---- b: read each source at its preferred phase, then average -----------------
phase_circle(170, 100, 15, 40, 100, BLUE, 'source 1')
phase_circle(228, 100, 15, 70, 190, GOLD, 'source 2')
text(199, 68, [('tick: preferred phase ∠', {}), ('Z', I), ('ij', SUB), ('  ·  dot: current phase', {})],
     6.3, MUTED, 'c')
cx, cy, r = 199, 41, 14
circle(cx, cy, r, LINE, .6, fill='FFFFFF')
for ang, col in ((60, BLUE), (120, GOLD)):
    t = math.radians(ang)
    seg(cx, cy, cx + r * math.cos(t), cy + r * math.sin(t), col, .7)
seg(cx, cy, cx, cy + r, PURPLE, 1.3, arrow=True)
text(cx + r + 5, cy + 4, [('u', {'b': True}), ('j', SUB)], 8, PURPLE)
text(148, 41, [('|', {}), ('Z', I), ('|-weighted', {})], 6.4, MUTED)
text(148, 33, 'mean of offsets', 6.4, MUTED)
text(199, 19, [('u', B), ('j', SUB), (' is the direction the coupling picks', {})], 6.6, PURPLE, 'c')

# ---- c: rotate (envelope, waveform) by u_j; the size is untouched -------------
text(334, 129, [('Content of band ', {}), ('j', I), (' as one complex number', {})], 6.9, INK, 'c')
ox, oy, k = 289 + 45, 36 + 44, 88 / 11.8          # plane origin and pt per data unit
P = lambda u, v: (ox + k * u, oy + k * v)
L = 5.9
seg(*P(-L + .4, 0), *P(L - .4, 0), LINE, .55)
seg(*P(0, -L + .4), *P(0, L - .4), LINE, .55)
circle(ox, oy, 5 * k, LINE, .55, dash=MSO_LINE.DASH)
text(*P(L - .3, -.9), [('envelope ', {}), ('a', I)], 6.4, GOLD, 'r')
text(*P(-.5, L - .8), [('waveform ', {}), ('w', I)], 6.4, BLUE, 'r')
seg(*P(0, 0), *P(3, 4), INK, 1.0, arrow=True)
text(*P(3.35, 4.55), [('q', I), (' = ', {}), ('a', I), (' + ', {}), ('i', I), ('w', I)], 6.9, INK)
seg(*P(0, 0), *P(-4, 3), PURPLE, 1.15, arrow=True)
text(*P(-4.3, 3.6), [('h', I), (' = ', {}), ('q', I), (' ', {}), ('u', B), ('j', SUB)], 6.9, PURPLE, 'r')
arc(ox, oy, 2.2 * k, 53.13, 143.13, PURPLE, .7, arrow=True)
text(*P(0, 3.15), [('∠', {}), ('u', B), ('j', SUB)], 6.4, PURPLE, 'c')
text(334, 26, [('|', {}), ('h', I), ('| = |', {}), ('q', I), ('|: the direction changes, the size does not', {})],
     6.2, INK, 'c')
for i in range(6):
    rect(312 + i * 7.5, 11, 6.5, 6.5, 'E3DAE2' if i % 2 == 0 else 'CFC0CE', 'FFFFFF', .25)
text(308, 14.5, 'Re, Im interleaved →', 6.3, MUTED, 'r')
text(359, 14.5, 'token', 6.6, PURPLE)

# the coupling direction is what crosses from b into c
polyline([(214, 41), (262, 41), (262, 79), (287, 79)], PURPLE, .8, arrow=True)

prs.save(OUT)
print('wrote', OUT, 'shapes:', len(sh))
