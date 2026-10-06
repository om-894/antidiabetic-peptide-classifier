"""
Label the free N-terminal amine in panel (b) of Figure 8.

His point 8: the orange sphere marking the amine is not identified on the panel
itself, so the 6.5 A gap it illustrates has to be read out of the caption. This
adds an arrow and a label pointing at it. The sphere's position is found by
colour rather than typed in, so the script survives a re-render of the panel.

Nothing else in the figure changes and the image keeps its pixel dimensions, so
the drawing's extent in document.xml stays correct.

INPUTS   ms/word/media/image8.png
OUTPUTS  the same file, annotated
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = "ms/word/media/image8.png"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
LABEL = "free N-terminal amine"
ORANGE = (214, 110, 29)


def find_sphere(im):
    """Centre and radius of the orange amine sphere in the right-hand panel."""
    a = np.array(im.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 150) & (r < 245) & (g > 70) & (g < 160) & (b < 70) \
        & ((r - b) > 110) & ((r - g) > 50)
    ys, xs = np.nonzero(mask)
    keep = xs > a.shape[1] // 2 # panel b only
    ys, xs = ys[keep], xs[keep]

    # the thin dashed distance line is the same colour, so take the densest
    # window rather than the centroid of every orange pixel
    best, H, W = (0, 0, 0), a.shape[0], a.shape[1]
    for cy in range(0, H, 6):
        for cx in range(W // 2, W, 6):
            n = int(((ys >= cy) & (ys < cy + 22) & (xs >= cx) & (xs < cx + 22)).sum())
            if n > best[0]:
                best = (n, cx, cy)
    _, cx, cy = best
    sel = (ys >= cy - 5) & (ys < cy + 27) & (xs >= cx - 5) & (xs < cx + 27)
    return xs[sel].mean(), ys[sel].mean(), max(8.0, (sel.sum() / np.pi) ** 0.5)


def main():
    im = Image.open(SRC).convert("RGB")
    W, H = im.size
    sx, sy, rad = find_sphere(im)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 17)

    # label sits in the clear background to the right of the sphere
    tw = d.textlength(LABEL, font=font)
    tx, ty = min(sx + 62, W - tw - 8), sy - 30
    d.text((tx, ty), LABEL, font=font, fill=ORANGE)

    # arrow from just under the label's left edge to the sphere's edge
    x0, y0 = tx + 6, ty + 23
    dx, dy = sx - x0, sy - y0
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    x1, y1 = sx - ux * (rad + 3), sy - uy * (rad + 3) # stop at the sphere
    d.line([(x0, y0), (x1, y1)], fill=ORANGE, width=3)

    # arrowhead
    head, wing = 11, 0.45
    for s in (1, -1):
        d.line([(x1, y1),
                (x1 - head * (ux * np.cos(wing) - s * uy * np.sin(wing)),
                 y1 - head * (uy * np.cos(wing) + s * ux * np.sin(wing)))],
               fill=ORANGE, width=3)

    im.save(SRC)
    print(f"labelled sphere at ({sx:.0f}, {sy:.0f}) r={rad:.0f}; "
          f"image still {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
