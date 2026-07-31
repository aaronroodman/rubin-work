#!/usr/bin/env python
"""Gallery of Rubin M1M3 and M2 bending-mode surfaces (batoid_rubin `bend`).

One page per mirror, 20 AOS bending modes in a 4x5 grid, for visual comparison
with Megias-Homar et al. 2024 (ApJ, arXiv:2406.04656).

Each panel is the mirror figure change for a unit (1 um) mode coefficient,
reconstructed as  Zernike(mode.zk) + grid_residual  (stored in meters ->
plotted in um of surface), on the true mirror footprint.  Modes are labeled
B1..B20 (1-indexed) with the underlying raw SVD index in parentheses; the AOS
set (from bend.yaml) is M1M3=[0..18,26], M2=[0..16,25,26,27].

Outputs (../output/):
    mode_gallery_M1M3.png
    mode_gallery_M2.png
"""
import argparse
from pathlib import Path

import numpy as np
import galsim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from batoid_rubin.builder import load_bend

DATA_DIR_DEFAULT = "/Users/roodman/LSST/batoid_rubin_data"


def mode_surface(bm, i, X, Y):
    """Full surface (um) of stored-mode i on grid (X,Y); off-annulus -> NaN."""
    zk = galsim.zernike.Zernike(np.asarray(bm.zk)[i],
                                R_outer=bm.R_outer, R_inner=bm.R_inner)
    surf = (zk(X, Y) + np.asarray(bm.z)[i]) * 1e6  # meters -> um
    R = np.hypot(X, Y)
    surf = np.where((R >= bm.R_inner) & (R <= bm.R_outer), surf, np.nan)
    return surf


def panel(ax, parts, title):
    """parts = list of (X, Y, surf); shared symmetric scale across parts."""
    vmax = max(np.nanmax(np.abs(s)) for _, _, s in parts)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for X, Y, s in parts:
        im = ax.pcolormesh(X, Y, s, cmap="RdBu_r", norm=norm, shading="auto")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return im, vmax


def build_page(mirror, sel_inds, data_dir, outpath):
    bd = str(Path(data_dir) / "bend")
    if mirror == "M1M3":
        modes = load_bend(bd, tuple(sel_inds))
        bms = [("M1", modes["M1"]), ("M3", modes["M3"])]
    else:
        modes = load_bend(bd, tuple(sel_inds))
        bms = [("M2", modes["M2"])]

    grids = {}
    for nm, bm in bms:
        x = np.asarray(bm.x); y = np.asarray(bm.y)
        grids[nm] = np.meshgrid(x, y)

    fig, axes = plt.subplots(4, 5, figsize=(13, 11))
    for k, ax in enumerate(axes.ravel()):
        parts = []
        for nm, bm in bms:
            X, Y = grids[nm]
            parts.append((X, Y, mode_surface(bm, k, X, Y)))
        _, vmax = panel(ax, parts, f"B{k+1}  (SVD {sel_inds[k]}) · {vmaxfmt(parts)}")
    fig.suptitle(f"Rubin {mirror} bending modes (unit 1µm coeff; µm surface) "
                 f"— AOS set, cf. Megias-Homar et al. 2024", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=130)
    print("wrote", outpath)


def vmaxfmt(parts):
    vmax = max(np.nanmax(np.abs(s)) for _, _, s in parts)
    return f"pk {vmax:.2f}µm"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    args = p.parse_args()
    import os
    os.environ.setdefault("BATOID_RUBIN_DATA_DIR", args.data_dir)

    import yaml
    cfg = yaml.safe_load(open(Path(args.data_dir) / "bend" / "bend.yaml"))
    m1m3_sel = cfg["use_m1m3_modes"]
    m2_sel = cfg["use_m2_modes"]

    outdir = Path(__file__).resolve().parent.parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    build_page("M1M3", m1m3_sel, args.data_dir, outdir / "mode_gallery_M1M3.png")
    build_page("M2", m2_sel, args.data_dir, outdir / "mode_gallery_M2.png")


if __name__ == "__main__":
    main()
