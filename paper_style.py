#!/usr/bin/env python3
"""
The shared look of the paper figures (paper_fig_*.py), after the figures of the Nature-family journals: Arial /
Helvetica type at 6-7 pt, bold 8 pt lowercase panel letters, 0.5 pt open axes with outward ticks, no grid, 1 pt data
lines, filled markers with a thin white rim, thin range bars drawn under the markers, the single runs shown as small
open dots where the axis can hold them, frameless legends. Sizes are in pt at print size: every figure is drawn at the
width it is printed at (\\columnwidth 3.5 in, \\textwidth 7.16 in), so it goes into the paper unscaled.

Only the drawing lives here; the figure scripts keep their own data loading.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullFormatter

INK, INK2, RULE = "#1a1a1a", "#595959", "#cfcfcf"       # text and axes / secondary text / reference rules
FS, FS_TICK, FS_PANEL = 7, 6, 8                         # axis labels / ticks and legends / panel letters
LW_AX, LW, MS = 0.5, 1.0, 4.2
DASH = (0, (3, 1.5))
FONTS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]   # Liberation Sans = Arial's metrics
# colour follows the modulation in every figure (and the Fig. 1 call-out)
STYLE = {"ook": dict(color="#1f5fbf", marker="o", label="OOK"),
         "fsk": dict(color="#00897b", marker="s", label="FSK")}
EB = dict(elinewidth=0.6, capsize=1.6, capthick=0.6)    # range bars
# the link's path in the frequency sweeps: direct contact is the neutral reference, one hue per TX electrode location
PATH = {"direct": dict(color="#555555", marker="D"), "back": dict(color="#D55E00", marker="^"),
        "knee": dict(color="#009E73", marker="s"), "ankle": dict(color="#6a4fb5", marker="o")}


def rc(scale=1.0):
    """rcParams of the look; scale > 1 for a figure drawn larger than it is printed"""
    s = scale
    return {"font.family": "sans-serif", "font.sans-serif": FONTS, "mathtext.default": "regular",
            "font.size": FS * s, "axes.labelsize": FS * s, "axes.titlesize": FS * s, "legend.fontsize": FS_TICK * s,
            "xtick.labelsize": FS_TICK * s, "ytick.labelsize": FS_TICK * s,
            "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
            "axes.linewidth": LW_AX * s, "axes.spines.top": False, "axes.spines.right": False,
            "axes.labelpad": 2 * s, "axes.axisbelow": True,
            "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.width": LW_AX * s, "ytick.major.width": LW_AX * s,
            "xtick.major.size": 2.5 * s, "ytick.major.size": 2.5 * s,
            "xtick.minor.width": 0.4 * s, "ytick.minor.width": 0.4 * s,
            "xtick.minor.size": 1.4 * s, "ytick.minor.size": 1.4 * s,
            "xtick.major.pad": 2 * s, "ytick.major.pad": 2 * s,
            "lines.linewidth": LW * s, "lines.markersize": MS * s, "lines.solid_capstyle": "round",
            "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3, "legend.columnspacing": 1.4,
            "hatch.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


RC = rc()


def figure(w=3.5, h=2.3, nrows=1, ncols=1, wspace=0.06, hspace=0.05, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=wspace, hspace=hspace)
    return fig, ax


def marks(color, marker="o", hollow=False, **kw):
    """line + marker keywords: a filled marker gets a thin white rim, so series that cross stay apart; hollow = white
    face with the series colour as rim (the second level of a two-level key)"""
    d = dict(color=color, marker=marker, lw=LW, ms=MS - 0.6 if hollow else MS, mfc="white" if hollow else color,
             mec=color if hollow else "white", mew=0.8 if hollow else 0.5)
    d.update(kw)
    return d


def series(ax, x, y, spreads=None, zorder=3, **kw):
    """a line with markers; under them, one range bar per point from spreads = [(lowest, highest) or None]
    (None, or a point that is not drawn: no bar)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = [i for i, s in enumerate(spreads or []) if s is not None and np.isfinite(y[i])]
    if ok:
        lo = [max(y[i] - spreads[i][0], 0.0) for i in ok]
        hi = [max(spreads[i][1] - y[i], 0.0) for i in ok]
        ax.errorbar(x[ok], y[ok], yerr=[lo, hi], fmt="none", ecolor=kw["color"], zorder=zorder - 0.4, **EB)
    return ax.plot(x, y, zorder=zorder, **kw)[0]


def run_dots(ax, x, values, color=INK, half_width=0.0, zorder=5, ms=1.9):
    """the single runs behind a pooled point: small open dots, spread evenly over x +- half_width"""
    v = np.asarray(values, float)
    dx = np.linspace(-half_width, half_width, len(v)) if len(v) > 1 else np.zeros(len(v))
    ax.plot(x + dx, v, ls="none", marker="o", ms=ms, mfc="white", mec=color, mew=0.45, zorder=zorder, clip_on=False)


def bars(ax, x, y, runs, color, width, label=None):
    """bars of pooled values with, per bar, the single runs as dots and their lowest-to-highest range bar"""
    ax.bar(x, y, width=width, color=color, lw=0, label=label, zorder=2)
    for xi, yi, r in zip(x, y, runs):
        if len(r) > 1:
            ax.errorbar([xi], [yi], yerr=[[max(yi - min(r), 0.0)], [max(max(r) - yi, 0.0)]], fmt="none", ecolor=INK,
                        zorder=4, **EB)
        run_dots(ax, xi, r, half_width=0.22 * width)


def finish(ax, categorical=False):
    """open axes, outward ticks; categorical: no tick marks under the category names"""
    ax.tick_params(which="both", colors=INK)
    if categorical:
        ax.tick_params(axis="x", which="both", length=0, pad=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def round_ticks(ax, nbins=5):
    """y ticks on round steps (0, 0.5, 1 ... rather than quarters)"""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))


def pct_axis(ax, top, headroom=1.10):
    """a linear axis from 0 that holds values up to top, on round ticks"""
    ax.set_ylim(0, top * headroom)
    round_ticks(ax)


def khz_axis(ax, car_hz):
    """the log carrier-frequency axis of the probe sweeps, in kHz"""
    ticks = [50, 100, 200, 500]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(car_hz[0] / 1e3 * 0.97, car_hz[-1] / 1e3 * 1.03)


def vmark(ax, x, text=None, y=0.99):
    """a reference frequency: thin dashed rule, its name beside the top of it"""
    ax.axvline(x, color=INK2, ls=DASH, lw=0.6, zorder=1)
    if text:
        ax.annotate(text, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points",
                    fontsize=FS_TICK, color=INK2, ha="left", va="top")


def band(ax, x0, x1, text=None):
    """a faint band behind a group of x positions, its name at the top of it"""
    ax.axvspan(x0, x1, color="#f1f1f1", lw=0, zorder=0)
    if text:
        ax.annotate(text, xy=((x0 + x1) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, -2),
                    textcoords="offset points", fontsize=FS_TICK, color=INK2, ha="center", va="top")


def trace(ax, x, y, color, marker, every, solid=True, **kw):
    """one dense sweep: a line with every few points marked; not solid = thin dashed, drawn underneath"""
    return ax.plot(x, y, color=color, lw=LW if solid else 0.7, ls="-" if solid else DASH, marker=marker, ms=3.0,
                   markevery=every, mec="white", mew=0.4, zorder=3 if solid else 2, **kw)[0]


def panel(ax, letter, dx=-27.0, dy=5.0, scale=1.0):
    """bold lowercase panel letter above the panel's top left corner, dx / dy pt from it"""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx * scale, dy * scale),
                textcoords="offset points", fontsize=FS_PANEL * scale, fontweight="bold", color=INK, ha="left",
                va="bottom", annotation_clip=False)


def note(ax, text, x=0.04, y=0.97, **kw):
    """a short label inside the panel (its title, in the Nature way: plain, top left)"""
    kw = {"fontsize": FS_TICK, "color": INK, "ha": "left", "va": "top", **kw}
    return ax.text(x, y, text, transform=ax.transAxes, **kw)


def top_legend(fig, handles, labels=None, ncol=None, **kw):
    """one frameless key centred above the panels"""
    labels = labels or [h.get_label() for h in handles]
    kw = {"prop": {"size": FS_TICK}, "borderaxespad": 0.1, **kw}
    return fig.legend(handles, labels, loc="outside upper center", ncol=ncol or len(handles), **kw)


def key(color=INK2, marker="o", hollow=False, ls="-", **kw):
    """a legend handle in the look of marks()"""
    return plt.Line2D([], [], **marks(color, marker, hollow, ls=ls, **kw))


def save(fig, out):
    """out without extension -> pdf (for the paper), svg, png (600 dpi)"""
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
