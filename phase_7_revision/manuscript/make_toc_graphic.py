"""
Table of Contents graphic: two negative classes, two screen behaviours.

ACS requires a TOC graphic with every manuscript and sets the terms this one is
built to. It must be original artwork rather than a figure lifted from the
paper, it must convey the essence "without providing specific results", and it
must fit 3.25 x 1.75 inches with sans serif text at 8 pt and never below 6 pt,
saved as TIF at 300 dpi.

So this carries no axes, no numbers and no measured values. It shows the one
idea the paper turns on: hold the positives and the model fixed, change where
the negatives come from, and the behaviour on real proteome fragments changes.

Palette is Okabe-Ito, as used by the manuscript figures, so it reads in
greyscale and for colour-blind readers.

OUTPUTS  ADP_TOC_graphic.tif (300 dpi) and a PNG for on-screen checking
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

W_IN, H_IN, DPI = 3.25, 1.75, 300

BLUE = "#0072B2"   # positives, identical on both paths
ORANGE = "#E69F00" # negatives drawn from a curated peptide database
GREY = "#999999"   # negatives cut from the proteome
VERM = "#D55E00"   # a fragment the screen wrongly accepts
INK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
})


def panel(ax, x0, y0, w, h):
    """A framed box, returned so its contents can be clipped to it."""
    r = Rectangle((x0, y0), w, h, facecolor="white", edgecolor="#CCCCCC",
                  linewidth=0.6, zorder=1)
    ax.add_patch(r)
    return r


def cloud(ax, cx, cy, n, rx, ry, colour, seed, box=None, size=6):
    """A small scatter standing for a class in feature space."""
    rng = np.random.default_rng(seed)
    pts = ax.scatter(rng.normal(cx, rx, n), rng.normal(cy, ry, n), s=size,
                     c=colour, linewidths=0, zorder=3)
    if box is not None:
        pts.set_clip_path(box) # a stray point outside its panel reads as an error


def arrow(ax, p, q, colour=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, color=colour, zorder=4,
                                 shrinkA=0, shrinkB=0))


def outcome(ax, x0, y0, w, h, n_wrong, seed):
    """A strip of proteome fragments, the wrongly accepted ones filled."""
    n, pad = 12, 0.004
    cw = (w - pad * (n - 1)) / n
    rng = np.random.default_rng(seed)
    wrong = set(rng.choice(n, n_wrong, replace=False))
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * (cw + pad), y0), cw, h,
                               facecolor=VERM if i in wrong else "#EDEDED",
                               edgecolor="none", zorder=3))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fs = 6.5 # above the 6 pt floor the guidelines set, and it buys room

    # left: one set of positives, shared by both paths
    box = panel(ax, 0.045, 0.38, 0.125, 0.30)
    cloud(ax, 0.1075, 0.53, 26, 0.019, 0.040, BLUE, 1, box)
    ax.text(0.1075, 0.345, "same\npositives", ha="center", va="top", color=INK,
            fontsize=fs, linespacing=1.25)

    arrow(ax, (0.175, 0.58), (0.255, 0.73))
    arrow(ax, (0.175, 0.48), (0.255, 0.33))

    # middle: the only thing that differs, where the negatives come from
    for y0, neg_colour, dx, label, seed in [
            (0.55, ORANGE, 0.055, "curated negatives", 2),
            (0.12, GREY, 0.008, "proteome negatives", 3)]:
        box = panel(ax, 0.265, y0, 0.215, 0.29)
        cx, cy = 0.345, y0 + 0.145
        cloud(ax, cx, cy, 22, 0.021, 0.034, BLUE, 4, box)
        cloud(ax, cx + dx, cy + (0.050 if dx > 0.02 else 0.0), 22, 0.021, 0.034,
              neg_colour, seed, box)
        ax.text(0.3725, y0 - 0.025, label, ha="center", va="top", color=INK,
                fontsize=fs)

    arrow(ax, (0.49, 0.695), (0.565, 0.695))
    arrow(ax, (0.49, 0.265), (0.565, 0.265))

    # right: what the same model then does on real proteome fragments
    ax.text(0.775, 0.955, "on proteome fragments", ha="center", va="top",
            color=INK, fontsize=fs)
    for y0, n_wrong, label, seed in [(0.655, 9, "mostly false positives", 5),
                                     (0.225, 2, "mostly correct", 6)]:
        outcome(ax, 0.585, y0, 0.38, 0.080, n_wrong, seed)
        ax.text(0.775, y0 - 0.025, label, ha="center", va="top", color=INK,
                fontsize=fs)

    os.makedirs(out_dir, exist_ok=True)
    tif = os.path.join(out_dir, "ADP_TOC_graphic.tif")
    png = os.path.join(out_dir, "ADP_TOC_graphic.png")
    fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)

    # ACS asks for TIF at 300 dpi; flatten to RGB so no alpha channel ships
    from PIL import Image
    Image.open(png).convert("RGB").save(tif, format="TIFF", dpi=(DPI, DPI),
                                        compression="tiff_lzw")
    print(f"wrote {tif} and {png} at {W_IN} x {H_IN} in, {DPI} dpi "
          f"({int(W_IN * DPI)} x {int(H_IN * DPI)} px)")


if __name__ == "__main__":
    main()
