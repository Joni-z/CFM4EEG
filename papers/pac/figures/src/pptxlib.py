"""Shared helpers for building the paper figures as editable PowerPoint slides.

Coordinates are the figure's own point coordinates (origin bottom-left, as in
matplotlib); everything is drawn at scale S so the slide is 2x the PDF.
"""
import math
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from lxml import etree

INK = '202A35'; MUTED = '596575'; LINE = 'CDD5DD'; PALE = 'F4F6F8'
BLUE = '28617E'; GOLD = 'A16B29'; PURPLE = '75516F'; GREY = '9AA6AE'
I = {'i': True}; B = {'b': True}; SUB = {'i': True, 'sub': True}; SUBR = {'sub': True}


class Sheet:
    def __init__(self, w, h, scale=2.0, font='Times New Roman'):
        self.W, self.H, self.S, self.font = w, h, scale, font
        self.prs = Presentation()
        self.prs.slide_width = Pt(w * scale); self.prs.slide_height = Pt(h * scale)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.sh = self.slide.shapes

    # -- coordinate helpers ----------------------------------------------------
    def X(self, x): return Pt(x * self.S)
    def Y(self, y): return Pt((self.H - y) * self.S)
    def L(self, v): return Pt(v * self.S)

    # -- primitives -------------------------------------------------------------
    def stroke(self, shape, color, lw, dash=None):
        ln = shape.line; ln.color.rgb = RGBColor.from_string(color); ln.width = Pt(lw * self.S)
        if dash: ln.dash_style = dash

    @staticmethod
    def arrowhead(shape, size='med', end='tail'):
        ln = shape.line._get_or_add_ln()
        tag = qn('a:tailEnd' if end == 'tail' else 'a:headEnd')
        el = ln.find(tag)
        if el is None: el = etree.SubElement(ln, tag)
        el.set('type', 'triangle'); el.set('w', size); el.set('len', size)

    @staticmethod
    def flat(shape):
        """Drop the theme's effect style (a drop shadow) from a connector or shape."""
        spPr = shape._element.spPr
        if spPr.find(qn('a:effectLst')) is None:
            etree.SubElement(spPr, qn('a:effectLst'))

    def seg(self, x1, y1, x2, y2, color, lw, arrow=False, dash=None):
        c = self.sh.add_connector(MSO_CONNECTOR.STRAIGHT, self.X(x1), self.Y(y1), self.X(x2), self.Y(y2))
        self.stroke(c, color, lw, dash); self.flat(c)
        if arrow: self.arrowhead(c)
        return c

    def polyline(self, pts, color, lw, arrow=False, dash=None):
        out = []
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            out.append(self.seg(x1, y1, x2, y2, color, lw, arrow=arrow and (x2, y2) == pts[-1], dash=dash))
        return out

    def curve(self, pts, color, lw, dash=None, fill=None):
        b = self.sh.build_freeform(self.X(pts[0][0]), self.Y(pts[0][1]), scale=1.0)
        b.add_line_segments([(self.X(x), self.Y(y)) for x, y in pts[1:]], close=fill is not None)
        s = b.convert_to_shape()
        if fill is None: s.fill.background()
        else: s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
        self.stroke(s, color, lw, dash); s.shadow.inherit = False; self.flat(s); return s

    def circle(self, cx, cy, r, color, lw, fill=None, dash=None):
        s = self.sh.add_shape(MSO_SHAPE.OVAL, self.X(cx - r), self.Y(cy + r), self.L(2 * r), self.L(2 * r))
        if fill is None: s.fill.background()
        else: s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
        self.stroke(s, color, lw, dash); s.shadow.inherit = False; return s

    def rect(self, x, y, w, h, fill, line=None, lw=.3, rounded=False):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
        s = self.sh.add_shape(kind, self.X(x), self.Y(y + h), self.L(w), self.L(h))
        if rounded: s.adjustments[0] = min(.12, 2.0 / min(w, h))
        if fill is None: s.fill.background()
        else: s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
        if line: self.stroke(s, line, lw)
        else: s.line.fill.background()
        s.shadow.inherit = False; return s

    def marker(self, x, y, size, color, kind='o'):
        if kind == 'o': return self.circle(x, y, size / 2, color, .3, fill=color)
        return self.rect(x - size / 2, y - size / 2, size, size, color, color, .3)

    def arc(self, cx, cy, r, t1, t2, color, lw, arrow=False):
        n = max(8, int(abs(t2 - t1) / 6))
        pts = [(cx + r * math.cos(math.radians(t1 + (t2 - t1) * k / n)),
                cy + r * math.sin(math.radians(t1 + (t2 - t1) * k / n))) for k in range(n + 1)]
        s = self.curve(pts, color, lw)
        if arrow: self.arrowhead(s, 'sm')
        return s

    def text(self, x, y, runs, size, color=INK, ha='l', bold=False, w=None, font=None, va='m'):
        """runs: str or list of (text, style) with style keys i, b, sub, c."""
        if isinstance(runs, str): runs = [(runs, {})]
        plain = ''.join(t for t, _ in runs)
        w = w or (len(plain) * size * 0.52 + 6)
        h = size * 1.5
        left = {'l': x, 'c': x - w / 2, 'r': x - w}[ha]
        top = {'m': y + h / 2, 't': y, 'b': y + h}[va]
        tb = self.sh.add_textbox(self.X(left), self.Y(top), self.L(w), self.L(h))
        tf = tb.text_frame; tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[ha]
        for t, st in runs:
            r = p.add_run(); r.text = t; f = r.font
            f.name = font or self.font; f.size = Pt(size * self.S)
            f.bold = st.get('b', bold); f.italic = st.get('i', False)
            f.color.rgb = RGBColor.from_string(st.get('c', color))
            if st.get('sub'): r._r.get_or_add_rPr().set('baseline', '-25000')
        return tb

    def label(self, x, y, w, h, lines, color, size=7.5, fill=None):
        """Rounded box with centered (possibly multi-line) text."""
        self.rect(x, y, w, h, fill or tint(color, .06), color, .6, rounded=True)
        n = len(lines)
        for k, ln in enumerate(lines):
            self.text(x + w / 2, y + h / 2 + (n - 1 - 2 * k) * size * .65, ln, size, color, 'c')

    def heading(self, x, y, letter, title):
        self.text(x, y, letter, 9, bold=True); self.text(x + 12, y, title, 8.3, bold=True)

    def save(self, path):
        self.prs.save(path); return len(self.sh)


def tint(hexcolor, alpha, base='FFFFFF'):
    """Blend a colour toward white; PowerPoint fills here are opaque."""
    c = [int(hexcolor[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(base[i:i + 2], 16) for i in (0, 2, 4)]
    return ''.join('%02X' % round(b[k] + (c[k] - b[k]) * alpha) for k in range(3))


DASH = MSO_LINE.DASH
