#!/usr/bin/env python
"""
Focal-plane maps from a full-FoV run (guider_atmo_sim.py --mode fov):
FWHM (scalar), ellipticity (whiskers), and 3rd-order spin-3 moment (whiskers),
one star per science CCD. Compare against typical on-sky PSF maps.

  python guider_atmo_fovmap.py --outdir output/fov --visit 0

Reads guider_atmo_fov_moments.parquet from --outdir.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _whiskers(ax, x, y, mag, ang, scale, color="k", key_label=""):
    """Draw headless whiskers of length ~scale*mag at position angle ang [rad]."""
    dx = scale * mag * np.cos(ang)
    dy = scale * mag * np.sin(ang)
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        ax.plot([xi - dxi / 2, xi + dxi / 2], [yi - dyi / 2, yi + dyi / 2],
                "-", color=color, lw=1.0)
    ax.set_aspect("equal")


def make_maps(outdir, visit=0, save=None):
    df = pd.read_parquet(os.path.join(outdir, "guider_atmo_fov_moments.parquet"))
    d = df[(df.visit == visit) & df.ok].copy()
    x, y = d.field_x_deg.values, d.field_y_deg.values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    # 1) FWHM scalar map
    sc = axes[0].scatter(x, y, c=d.fwhm_asec, s=90, cmap="viridis",
                         edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, ax=axes[0], label="FWHM [arcsec]", fraction=0.046)
    axes[0].set_title(f"FWHM  (median {np.median(d.fwhm_asec):.3f}\")")

    # 2) ellipticity whiskers: length |e|, position angle 0.5*atan2(e2,e1)
    e = np.hypot(d.e1, d.e2)
    ang_e = 0.5 * np.arctan2(d.e2, d.e1)
    span = max(x.max() - x.min(), y.max() - y.min())
    _whiskers(axes[1], x, y, e.values, ang_e.values,
              scale=0.15 * span / max(e.max(), 1e-3))
    axes[1].scatter(x, y, s=4, c="0.6")
    axes[1].set_title(f"ellipticity whiskers  (<|e|>={e.mean():.3f})")

    # 3) 3rd-order spin-3 whiskers: length |M3|, angle (1/3)*atan2(spin3_y,spin3_x)
    m3 = np.hypot(d.M_spin3_x, d.M_spin3_y)
    ang_3 = (1.0 / 3.0) * np.arctan2(d.M_spin3_y, d.M_spin3_x)
    _whiskers(axes[2], x, y, m3.values, ang_3.values,
              scale=0.15 * span / max(m3.max(), 1e-9), color="firebrick")
    axes[2].scatter(x, y, s=4, c="0.6")
    axes[2].set_title(f"3rd-moment spin-3  (<|M3|>={m3.mean():.2e} asec$^3$)")

    for ax in axes:
        ax.set_xlabel("field x [deg]"); ax.set_ylabel("field y [deg]")
        lim = 1.05 * max(np.abs(np.r_[x, y]))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle(f"FoV PSF maps — visit {visit}  ({len(d)} CCDs)", fontsize=13)
    fig.tight_layout()

    if save is None:
        save = os.path.join(outdir, f"guider_atmo_fovmap_visit{visit:02d}.png")
    fig.savefig(save, dpi=120)
    plt.close(fig)
    print(f"wrote {save}  ({len(d)} CCDs)")


def make_psf_mosaic(outdir, visit=0, cutout=None, save=None):
    """Place each CCD's PSF stamp at its focal-plane position -> look at how the
    PSF itself varies across the field. Needs guider_atmo_fov_stamps_visit{v}.npy."""
    npy = os.path.join(outdir, f"guider_atmo_fov_stamps_visit{visit:02d}.npy")
    if not os.path.isfile(npy):
        print(f"  (no {os.path.basename(npy)}; run the sim with fov output to view PSFs)")
        return
    stamps = np.load(npy)                                  # [n_ccd, roi, roi]
    df = pd.read_parquet(os.path.join(outdir, "guider_atmo_fov_moments.parquet"))
    d = df[df.visit == visit].reset_index(drop=True)
    x, y = d.field_x_deg.values, d.field_y_deg.values
    roi = stamps.shape[-1]
    c = roi // 2
    h = (cutout or min(roi, 41)) // 2
    span = max(x.max() - x.min(), y.max() - y.min())
    box = 0.11 * span                                      # each PSF panel size [deg]

    fig, ax = plt.subplots(figsize=(9, 9))
    for i in range(min(len(d), stamps.shape[0])):
        sub = stamps[i, c - h:c + h, c - h:c + h].astype(float)
        pk = np.nanpercentile(sub, 99.5)
        norm = np.sqrt(np.clip(sub, 0, None) / (pk if pk > 0 else 1))   # sqrt stretch
        ax.imshow(norm, origin="lower", cmap="Greys", vmin=0, vmax=1,
                  extent=[x[i] - box / 2, x[i] + box / 2, y[i] - box / 2, y[i] + box / 2],
                  interpolation="nearest", zorder=2)
    ax.scatter(x, y, s=2, c="tab:blue", zorder=1)
    lim = 1.08 * max(np.abs(np.r_[x, y])) + box
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("field x [deg]"); ax.set_ylabel("field y [deg]")
    ax.set_title(f"FoV PSF stamps — visit {visit}  ({len(d)} CCDs, "
                 f"{2*h}px cutout, sqrt stretch)")
    fig.tight_layout()
    if save is None:
        save = os.path.join(outdir, f"guider_atmo_fov_psfmosaic_visit{visit:02d}.png")
    fig.savefig(save, dpi=130)
    plt.close(fig)
    print(f"wrote {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/fov")
    ap.add_argument("--visit", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--cutout", type=int, default=None, help="PSF-mosaic cutout [pix]")
    args = ap.parse_args()
    make_maps(args.outdir, visit=args.visit, save=args.save)
    make_psf_mosaic(args.outdir, visit=args.visit, cutout=args.cutout)
