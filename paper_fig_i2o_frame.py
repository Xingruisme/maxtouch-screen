#!/usr/bin/env python3
"""
Paper figure: structure of the short i2o frame (i2o_frame.py), drawn like the i2i frame figure (Fig. 8): the frame
as a row of fields with the uncoded / coded parts bracketed above it, and the header opened up underneath.
Field names and sizes are read from i2o_frame.py, so the figure cannot drift from the frame the tests send.

    python paper_fig_i2o_frame.py [--out data_maxtouch/paper_fig_i2o_frame]
Output: fig_i2o_frame.pdf (for the paper), .svg, .png. Drawn at \\columnwidth (3.5 in), 8 pt type.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

import i2o_frame as I2O
import paper_style as PS

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = dict(src="Src", type="Type", frame_no="Frame no.", payload="Payload", crc="CRC")
LW = 0.9


def brace(ax, x0, x1, y, h, r=1.6):
    """a brace opening downwards over [x0, x1], its tip at y + h"""
    xm, ym = (x0 + x1) / 2, y + h / 2
    v = [(x0, y), (x0, ym), (x0 + r, ym), (xm - r, ym), (xm, ym), (xm, y + h), (xm, ym), (xm + r, ym), (x1 - r, ym),
         (x1, ym), (x1, y)]
    c = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3,
         Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(v, c), fc="none", ec=PS.INK, lw=LW, capstyle="round"))


def row(ax, cells, y, h):
    """cells [(x0, x1, name, bits)] -> boxes with 'name' over '(n bits)'"""
    for x0, x1, name, bits in cells:
        ax.add_patch(Rectangle((x0, y), x1 - x0, h, fc="none", ec=PS.INK, lw=LW, joinstyle="miter"))
        ax.text((x0 + x1) / 2, y + h / 2, f"{name}\n({bits} bits)", ha="center", va="center", linespacing=1.25)


def draw():
    hdr = [(n, b) for n, b in I2O.FIELDS if n in ("src", "type", "frame_no")]
    n_hdr = sum(b for _, b in hdr)
    sizes = dict(I2O.FIELDS)
    nsync = len(I2O.SYNC)
    fig = plt.figure(figsize=(3.5, 1.72))
    ax = fig.add_axes([0.005, 0.0, 0.99, 1.0])
    ax.set_xlim(-0.6, 100.6); ax.set_ylim(1.0, 50.0); ax.axis("off")
    y1, h1, y2, h2 = 27.5, 11.5, 2.5, 11.5
    x = [0.0, 33.0, 54.0, 80.0, 100.0]          # sync | header | payload | CRC (not to scale, as in the i2i figure)
    row(ax, [(x[0], x[1], "Sync", nsync), (x[1], x[2], "Header", n_hdr),
             (x[2], x[3], NAMES["payload"], sizes["payload"]), (x[3], x[4], NAMES["crc"], sizes["crc"])], y1, h1)
    brace(ax, x[0] + 0.4, x[1] - 0.7, y1 + h1 + 1.0, 2.6)
    brace(ax, x[1] + 0.7, x[4] - 0.4, y1 + h1 + 1.0, 2.6)
    ytxt = y1 + h1 + 4.6
    ax.text((x[0] + x[1]) / 2, ytxt, f"Uncoded: {nsync} symbols", ha="center", va="bottom")
    ax.text((x[1] + x[4]) / 2, ytxt, f"Rate-1/2 coded: {I2O.N_CODED} symbols", ha="center", va="bottom")
    w = 100.0 / len(hdr)                        # the header, opened up
    row(ax, [(i * w, (i + 1) * w, NAMES[n], b) for i, (n, b) in enumerate(hdr)], y2, h2)
    for xa, xb in ((x[1], 0.0), (x[2], 100.0)):
        ax.plot([xa, xb], [y1, y2 + h2], color=PS.INK, lw=0.7, ls=(0, (3, 2)), solid_capstyle="butt")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_i2o_frame"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with plt.rc_context(PS.RC):                 # the fonts are looked up when the figure is saved
        PS.save(draw(), os.path.join(a.out, "fig_i2o_frame"))
    print(I2O.describe())
    print(f"saved {os.path.relpath(a.out, HERE)}/fig_i2o_frame.pdf / .svg / .png")


if __name__ == "__main__":
    main()
