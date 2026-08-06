#!/usr/bin/env python
"""
Movie of the simulated guider stamps, in the style of the summit_utils guider
mosaic (grayscale "Greys", origin="lower", nearest interpolation, per-frame
percentile scaling; gif via pillow, mp4 via ffmpeg).

The 8 guide sensors are laid out at their focal-plane corners; each frame is one
50 ms stamp, stepping through the visit.

  python guider_atmo_movie.py --outdir output/atmo_sim --visit 0 --save movie.gif

Reads guider_atmo_stamps_visit{v}.npy + guider_atmo_moments.parquet from --outdir.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _writer(path):
    ext = os.path.splitext(path)[1].lower()
    return {".gif": "pillow", ".mp4": "ffmpeg"}.get(ext, "pillow")


def _corner_rects(order):
    """Place each guider panel in its focal-plane quadrant (2 CCDs per corner)."""
    from collections import defaultdict
    quad = defaultdict(list)                       # (sx, sy) -> [axis-index, ...]
    for i, (_, tx, ty) in enumerate(order):
        quad[(np.sign(tx), np.sign(ty))].append(i)
    rects = {}
    m, pw, ph = 0.02, 0.225, 0.42                  # margin, panel w/h (fig fractions)
    for (sx, sy), members in quad.items():
        col0 = 0.02 if sx < 0 else 0.52            # left / right half
        row0 = 0.05 if sy < 0 else 0.53            # bottom / top half
        for slot, i in enumerate(sorted(members)):
            rects[i] = [col0 + slot * (pw + m), row0, pw, ph]
    return rects


def make_movie(outdir, visit=0, save=None, fps=5, plo=50.0, phi=99.0,
               cutout=None, hold=2):
    stamps = np.load(os.path.join(outdir, f"guider_atmo_stamps_visit{visit:02d}.npy"))
    df = pd.read_parquet(os.path.join(outdir, "guider_atmo_moments.parquet"))
    n_stamp, ng = stamps.shape[0], stamps.shape[1]

    # guider name + field position in stamps-axis order (rows at stamp 0, in order)
    first = df[(df.visit == visit) & (df.stamp == 0)].reset_index(drop=True)
    order = [(r.guider, r.theta_x_asec, r.theta_y_asec) for r in first.itertuples()]

    if cutout:                                     # optional centre crop
        c = stamps.shape[-1] // 2
        h = cutout // 2
        stamps = stamps[..., c - h:c + h, c - h:c + h]

    fig = plt.figure(figsize=(9, 9), facecolor="white")
    rects = _corner_rects(order)
    axes, ims = [], []
    for gi in range(ng):
        ax = fig.add_axes(rects[gi])
        vmin, vmax = np.nanpercentile(stamps[0, gi], [plo, phi])
        im = ax.imshow(stamps[0, gi], origin="lower", cmap="Greys",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(order[gi][0], fontsize=9, pad=2)
        axes.append(ax); ims.append(im)
    suptitle = fig.suptitle("", fontsize=12, y=0.99)

    def draw(frame):
        for gi in range(ng):
            arr = stamps[frame, gi]
            vmin, vmax = np.nanpercentile(arr, [plo, phi])
            ims[gi].set_data(arr)
            ims[gi].set_clim(vmin, vmax)
        suptitle.set_text(f"visit {visit}  stamp {frame}/{n_stamp-1}  "
                          f"t = {frame/fps:.2f} s")
        return ims + [suptitle]

    frames = [0] * hold + list(range(n_stamp)) + [n_stamp - 1] * hold
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)

    if save is None:
        save = os.path.join(outdir, f"guider_atmo_movie_visit{visit:02d}.gif")
    anim.save(save, fps=fps, dpi=90, writer=_writer(save))
    plt.close(fig)
    print(f"wrote {save}  ({n_stamp} stamps, {ng} sensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/atmo_sim")
    ap.add_argument("--visit", type=int, default=0)
    ap.add_argument("--save", default=None, help="output .gif or .mp4")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--cutout", type=int, default=None, help="centre-crop to N pix")
    ap.add_argument("--plo", type=float, default=50.0)
    ap.add_argument("--phi", type=float, default=99.0)
    args = ap.parse_args()
    make_movie(args.outdir, visit=args.visit, save=args.save, fps=args.fps,
               plo=args.plo, phi=args.phi, cutout=args.cutout)
