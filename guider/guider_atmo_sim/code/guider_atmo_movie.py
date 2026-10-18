#!/usr/bin/env python
"""
Movie of the simulated guider stamps in the RubinTV / summit_utils style: the 8
guide sensors laid out in the DVCS octagon (each corner raft's two CCDs adjacent),
each a pixelated cutout (grayscale, nearest interpolation, per-frame percentile
scaling) with 1"/2" guide circles, a crosshair, and the measured centroid (red).

The stamps are already binned at the ~0.2"/pixel plate scale by the simulator's
photon shooting; this just crops to a small cutout so the pixels are visible.

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

# DVCS octagon, matching summit_utils MosaicLayout.
GRID = [[".", "R40_SG1", "R44_SG0", "."],
        ["R40_SG0", "center", ".", "R44_SG1"],
        ["R00_SG1", ".", ".", "R04_SG0"],
        ["arrow", "R00_SG0", "R04_SG1", "."]]
GUIDERS = [c for row in GRID for c in row if c not in (".", "center", "arrow")]


def _writer(path):
    return {".gif": "pillow", ".mp4": "ffmpeg"}.get(
        os.path.splitext(path)[1].lower(), "pillow")


def make_movie(outdir, visit=0, save=None, fps=5, plo=50.0, phi=99.0,
               cutout=31, pixel_scale=0.2, hold=2):
    stamps = np.load(os.path.join(outdir, f"guider_atmo_stamps_visit{visit:02d}.npy"))
    df = pd.read_parquet(os.path.join(outdir, "guider_atmo_moments.parquet"))
    df = df[df.visit == visit]
    n_stamp, ng = stamps.shape[0], stamps.shape[1]

    # map guider name -> stamps-axis index (order rows were written at stamp 0)
    order = list(df[df.stamp == 0].sort_index().guider)
    gi_of = {name: i for i, name in enumerate(order)}
    # centroid (pixels from ROI centre) per (stamp, guider) for the red dot
    cen = {(r.stamp, r.guider): (r.cen_x, r.cen_y) for r in df.itertuples()}

    roi = stamps.shape[-1]
    c = roi // 2
    h = cutout // 2
    ext = h * pixel_scale                                  # arcsec half-width

    fig, axd = plt.subplot_mosaic(GRID, figsize=(9, 9),
                                  gridspec_kw=dict(hspace=0.05, wspace=0.05))
    ims, dots = {}, {}
    for name in GUIDERS:
        ax = axd[name]
        if name not in gi_of:
            ax.axis("off"); continue
        sub = stamps[0, gi_of[name], c - h:c + h, c - h:c + h]
        vmin, vmax = np.nanpercentile(sub, [plo, phi])
        ims[name] = ax.imshow(sub, origin="lower", cmap="Greys", vmin=vmin, vmax=vmax,
                              interpolation="nearest", extent=[-ext, ext, -ext, ext])
        for r in (1.0, 2.0):                               # guide circles [arcsec]
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--",
                                    color="tab:blue", lw=0.7, alpha=0.7))
        ax.axhline(0, ls="--", color="tab:blue", lw=0.5, alpha=0.5)
        ax.axvline(0, ls="--", color="tab:blue", lw=0.5, alpha=0.5)
        (dots[name],) = ax.plot([0], [0], "r.", ms=6)
        ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(name, fontsize=9, color="tab:blue", pad=2)

    axd["center"].axis("off")
    center_txt = axd["center"].text(0.5, 0.5, "", ha="center", va="center",
                                    fontsize=11, transform=axd["center"].transAxes)
    # DVCS orientation indicator
    aax = axd["arrow"]; aax.axis("off")
    aax.annotate("", xy=(0.7, 0.15), xytext=(0.15, 0.15),
                 arrowprops=dict(arrowstyle="->", color="tab:blue"))
    aax.annotate("", xy=(0.15, 0.7), xytext=(0.15, 0.15),
                 arrowprops=dict(arrowstyle="->", color="tab:blue"))
    aax.text(0.72, 0.15, "X DVCS", color="tab:blue", fontsize=7, va="center")
    aax.text(0.15, 0.74, "Y DVCS", color="tab:blue", fontsize=7, ha="center")

    def draw(frame):
        arts = []
        for name in GUIDERS:
            if name not in gi_of:
                continue
            sub = stamps[frame, gi_of[name], c - h:c + h, c - h:c + h]
            vmin, vmax = np.nanpercentile(sub, [plo, phi])
            ims[name].set_data(sub); ims[name].set_clim(vmin, vmax)
            cx, cy = cen.get((frame, name), (0.0, 0.0))
            dots[name].set_data([cx * pixel_scale], [cy * pixel_scale])
            arts += [ims[name], dots[name]]
        center_txt.set_text(f"guider atmo sim\nvisit {visit}\nstamp #: {frame}\n"
                            f"t = {frame/fps:.2f} s\norientation: dvcs")
        return arts + [center_txt]

    frames = [0] * hold + list(range(n_stamp)) + [n_stamp - 1] * hold
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    if save is None:
        save = os.path.join(outdir, f"guider_atmo_movie_visit{visit:02d}.gif")
    anim.save(save, fps=fps, dpi=90, writer=_writer(save))
    plt.close(fig)
    print(f"wrote {save}  ({n_stamp} stamps, {len(gi_of)} sensors, "
          f"cutout {cutout}px = +/-{ext:.1f}\")")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/atmo_sim")
    ap.add_argument("--visit", type=int, default=0)
    ap.add_argument("--save", default=None, help="output .gif or .mp4")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--cutout", type=int, default=31, help="cutout size [pix] (~0.2\"/pix)")
    ap.add_argument("--pixel-scale", type=float, default=0.2, dest="pixel_scale")
    ap.add_argument("--plo", type=float, default=50.0)
    ap.add_argument("--phi", type=float, default=99.0)
    args = ap.parse_args()
    make_movie(args.outdir, visit=args.visit, save=args.save, fps=args.fps,
               plo=args.plo, phi=args.phi, cutout=args.cutout,
               pixel_scale=args.pixel_scale)
