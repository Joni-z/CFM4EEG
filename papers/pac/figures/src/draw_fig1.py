"""Figure 1: from a measured coupling to a rotated token. Same sheet size, palette
and type as the previous version (396 x 151 pt, STIX), redrawn so that panel b
reads phase on circles instead of cramped linear axes, panel c shows the rotation
as a vector in the (envelope, waveform) plane, and the floating footer is gone.

Usage: python draw_fig1.py <output directory>
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Arc, Rectangle
from matplotlib.colors import to_rgba

OUT = Path(sys.argv[1]).resolve()
W, H = 396, 151
INK = '#202A35'; MUTED = '#596575'; LINE = '#CDD5DD'; PALE = '#F4F6F8'
BLUE = '#28617E'; GOLD = '#A16B29'; PURPLE = '#75516F'
STYLE = {'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
         'axes.linewidth': .55, 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'savefig.bbox': None, 'axes.spines.top': False, 'axes.spines.right': False}


def text(a, x, y, s, size=8, color=INK, ha='left', weight='normal', **kw):
    return a.text(x, y, s, fontsize=size, color=color, ha=ha, va='center',
                  weight=weight, zorder=8, **kw)


def line(a, p, color=INK, lw=.75, style='-', arrow=False, z=2):
    p = np.asarray(p, float)
    a.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=style, zorder=z, solid_capstyle='round')
    if arrow:
        a.annotate('', xy=p[-1], xytext=p[-2],
                   arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=5.5,
                                   shrinkA=0, shrinkB=0), zorder=z + 1)


def panel(a, x, y, w, h, color=LINE, fill='white', radius=2, lw=.6):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                               ec=color, fc=fill, lw=lw, zorder=1))


def label(a, x, y, w, h, s, color=INK, size=7.5):
    panel(a, x, y, w, h, color=color, fill=to_rgba(color, .06), lw=.6)
    text(a, x + w / 2, y + h / 2, s, size, color, ha='center')


def heading(a, x, y, letter, title):
    text(a, x, y, letter, 9, weight='bold'); text(a, x + 12, y, title, 8.3, weight='bold')


def mini_axes(f, x, y, w, h):
    ax = f.add_axes([x / W, y / H, w / W, h / H]); ax.axis('off'); return ax


def phase_circle(a, cx, cy, r, pref, cur, color, name):
    """One source band: tick at its preferred phase, dot at the current phase,
    and the offset between them, which is what alignment removes."""
    a.add_patch(Circle((cx, cy), r, fc='white', ec=LINE, lw=.6, zorder=3))
    for ang in (0, 90, 180, 270):
        t = np.deg2rad(ang)
        line(a, [(cx + (r - 1.6) * np.cos(t), cy + (r - 1.6) * np.sin(t)),
                 (cx + r * np.cos(t), cy + r * np.sin(t))], LINE, .45, z=4)
    tp, tc = np.deg2rad(pref), np.deg2rad(cur)
    line(a, [(cx, cy), (cx + r * np.cos(tp), cy + r * np.sin(tp))], color, 1.0, z=5)
    a.plot(cx + r * np.cos(tc), cy + r * np.sin(tc), 'o', ms=3.2, mfc=color, mec=color, zorder=6)
    a.add_patch(Arc((cx, cy), 2 * (r + 4.5), 2 * (r + 4.5), theta1=pref, theta2=cur,
                    ec=color, lw=.7, zorder=4))
    mid = np.deg2rad((pref + cur) / 2)
    text(a, cx + (r + 9.5) * np.cos(mid), cy + (r + 9.5) * np.sin(mid),
         f'+{cur - pref}°', 6.8, color, ha='center')
    text(a, cx, cy - r - 7, name, 6.9, color, ha='center')


with plt.rc_context(STYLE):
    f = plt.figure(figsize=(W / 72, H / 72))
    a = f.add_axes([0, 0, 1, 1]); a.set(xlim=(0, W), ylim=(0, H)); a.axis('off')

    for x, c, title in [(3, 'a', 'Measure the coupling'), (141, 'b', 'Align the sources'),
                        (275, 'c', 'Rotate the token')]:
        heading(a, x, 143, c, title)
    for x in (131, 265):
        line(a, [(x, 12), (x, 132)], LINE, .5)

    # ---- a: slow phase organizes the fast envelope inside one patch --------------
    t = np.linspace(0, 1, 700); phi = 4 * np.pi * t; env = 1 + .62 * np.cos(phi - .65)
    ax = mini_axes(f, 7, 88, 113, 38)
    ax.axvspan(.44, .66, color=to_rgba(GOLD, .075), lw=0)
    ax.plot(t, np.cos(phi), color=BLUE, lw=.85); ax.set(xlim=(0, 1), ylim=(-1.65, 1.65))
    text(a, 8, 128, r'Slow band $i$: phase $\phi_i$', 7.0, BLUE)
    ax = mini_axes(f, 7, 46, 113, 39)
    ax.axvspan(.44, .66, color=to_rgba(GOLD, .075), lw=0)
    ax.plot(t, env * np.cos(48 * np.pi * t), color='#9AA6AE', lw=.5)
    ax.plot(t, env, color=GOLD, lw=1.05); ax.plot(t, -env, color=GOLD, lw=.65)
    ax.set(xlim=(0, 1), ylim=(-1.85, 1.85))
    text(a, 8, 88, r'Fast band $j$: envelope $A_j$', 7.0, GOLD)
    text(a, 63, 33, r'per patch:  $Z_{ij}=\langle p_i\,(A_j-\bar A_j)\rangle$', 7.0, ha='center')
    text(a, 63, 21, r'$|Z_{ij}|$ strength  ·  $\angle Z_{ij}$ preferred phase', 6.5, MUTED, ha='center')

    # ---- b: each source read at its own preferred phase, then averaged ----------
    phase_circle(a, 170, 100, 15, 40, 100, BLUE, 'source 1')
    phase_circle(a, 228, 100, 15, 70, 190, GOLD, 'source 2')
    text(a, 199, 68, r'tick: preferred phase $\angle Z_{ij}$  ·  dot: current phase', 6.3, MUTED, ha='center')
    cx, cy, r = 199, 41, 14
    a.add_patch(Circle((cx, cy), r, fc='white', ec=LINE, lw=.6, zorder=3))
    for ang, col in ((60, BLUE), (120, GOLD)):
        tt = np.deg2rad(ang)
        line(a, [(cx, cy), (cx + r * np.cos(tt), cy + r * np.sin(tt))], col, .7, z=4)
    line(a, [(cx, cy), (cx, cy + r)], PURPLE, 1.3, arrow=True, z=6)
    text(a, cx + r + 5, cy + 4, r'$\mathbf{u}_j$', 8, PURPLE)
    text(a, 148, 41, r'$|Z|$-weighted', 6.4, MUTED, ha='left')
    text(a, 148, 33, 'mean of offsets', 6.4, MUTED, ha='left')
    text(a, 199, 19, r'$\mathbf{u}_j$ is the direction the coupling picks', 6.6, PURPLE, ha='center')

    # ---- c: rotate (envelope, waveform) by u_j; the size is untouched ------------
    text(a, 334, 129, r'Content of band $j$ as one complex number', 6.9, ha='center')
    px, py, pw, ph = 289, 36, 90, 88   # plane box in sheet points
    ax = f.add_axes([px / W, py / H, pw / W, ph / H]); ax.axis('off')
    L = 5.9; ax.set(xlim=(-L, L), ylim=(-L, L), aspect='equal')
    ax.plot([-L + .4, L - .4], [0, 0], color=LINE, lw=.55, zorder=1)
    ax.plot([0, 0], [-L + .4, L - .4], color=LINE, lw=.55, zorder=1)
    ax.add_patch(Circle((0, 0), 5, fc='none', ec=LINE, lw=.55, ls=(0, (2, 2)), zorder=1))
    ax.text(L - .3, -.75, r'envelope $a$', fontsize=6.4, color=GOLD, ha='right', va='top')
    ax.text(-.5, L - .4, r'waveform $w$', fontsize=6.4, color=BLUE, ha='right', va='top')
    ax.annotate('', xy=(3, 4), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.0, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(3.35, 4.25, r'$q=a+iw$', fontsize=6.9, color=INK, ha='left', va='bottom')
    ax.annotate('', xy=(-4, 3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.15, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(-4.3, 3.3, r'$h=q\,\mathbf{u}_j$', fontsize=6.9, color=PURPLE, ha='right', va='bottom')
    ax.add_patch(Arc((0, 0), 4.4, 4.4, theta1=53.13, theta2=143.13, ec=PURPLE, lw=.7, zorder=3))
    ax.annotate('', xy=(2.2 * np.cos(np.deg2rad(143.13)), 2.2 * np.sin(np.deg2rad(143.13))),
                xytext=(2.2 * np.cos(np.deg2rad(136)), 2.2 * np.sin(np.deg2rad(136))),
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=.7, mutation_scale=4.5, shrinkA=0, shrinkB=0))
    ax.text(0, 2.75, r'$\angle\mathbf{u}_j$', fontsize=6.4, color=PURPLE, ha='center', va='bottom')
    text(a, 334, 26, r'$|h|=|q|$: the direction changes, the size does not', 6.2, ha='center')
    for k in range(6):
        a.add_patch(Rectangle((312 + k * 7.5, 11), 6.5, 6.5,
                              fc=to_rgba(PURPLE, .16 + .16 * (k % 2)), ec='white', lw=.25, zorder=5))
    text(a, 308, 14.5, r'Re, Im interleaved $\rightarrow$', 6.3, MUTED, ha='right')
    text(a, 359, 14.5, 'token', 6.6, PURPLE)

    # the coupling direction is what crosses from b into c
    line(a, [(214, 41), (262, 41), (262, 79), (287, 79)], PURPLE, .8, arrow=True, z=6)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        f.savefig(OUT / f'fig1.{ext}', dpi=450, facecolor='white', bbox_inches=None)
    plt.close(f)
print('wrote', OUT / 'fig1.pdf')
