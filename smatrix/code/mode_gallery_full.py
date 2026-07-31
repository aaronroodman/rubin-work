#!/usr/bin/env python
"""Full bending-mode gallery -> multi-page PDF, 20 modes/page, both mirrors.

Uses the ZEMAX/OFC-basis full set (bend_zemax): 153 M1M3 + 69 M2 modes in
natural raw order (B_n = raw n-1 = DOF index).  Each panel is the mirror figure
for a unit (1 um) mode coefficient, reconstructed as Zernike(mode.zk) +
grid_residual on the true footprint (M1M3 = M1 outer annulus + M3 inner).

Output: ../output/mode_gallery_full.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import galsim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

from batoid_rubin.builder import load_bend

DATA = "/Users/roodman/LSST/batoid_rubin_data"


def mode_surface(bm, i, X, Y):
    zk = galsim.zernike.Zernike(np.asarray(bm.zk)[i], R_outer=bm.R_outer, R_inner=bm.R_inner)
    surf = (zk(X, Y) + np.asarray(bm.z)[i]) * 1e6            # meters -> um
    R = np.hypot(X, Y)
    return np.where((R >= bm.R_inner) & (R <= bm.R_outer), surf, np.nan)


def page(pdf, parts_list, titles, suptitle):
    """parts_list[k] = list of (X,Y,surf) drawn on panel k (M1M3 has 2 parts)."""
    fig, axes = plt.subplots(4, 5, figsize=(13, 11))
    for k, ax in enumerate(axes.ravel()):
        if k >= len(parts_list) or parts_list[k] is None:
            ax.axis("off"); continue
        parts = parts_list[k]
        vmax = max(np.nanmax(np.abs(s)) for _, _, s in parts) or 1e-9
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for X, Y, s in parts:
            ax.pcolormesh(X, Y, s, cmap="RdBu_r", norm=norm, shading="auto",
                          rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k], fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig); plt.close(fig)


def build_mirror(pdf, mirror, nmode, bend_dir):
    if mirror == "M1M3":
        modes = load_bend(bend_dir, tuple(range(nmode)))
        bms = [("M1", modes["M1"]), ("M3", modes["M3"])]
    else:
        modes = load_bend(bend_dir, tuple(range(nmode)))
        bms = [("M2", modes["M2"])]
    grids = {nm: np.meshgrid(np.asarray(bm.x), np.asarray(bm.y)) for nm, bm in bms}

    for p0 in range(0, nmode, 20):
        idxs = range(p0, min(p0 + 20, nmode))
        parts_list, titles = [], []
        for i in idxs:
            parts = []
            for nm, bm in bms:
                X, Y = grids[nm]
                parts.append((X, Y, mode_surface(bm, i, X, Y)))
            vmax = max(np.nanmax(np.abs(s)) for _, _, s in parts)
            parts_list.append(parts); titles.append(f"B{i+1}  (pk {vmax:.2f}µm)")
        page(pdf, parts_list, titles,
             f"Rubin {mirror} bending modes B{p0+1}-B{min(p0+20,nmode)} of {nmode}  "
             f"(unit 1µm coeff; µm surface; ZEMAX/OFC basis)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--bend-dir", default="bend_zemax")
    args = p.parse_args()
    import os
    os.environ.setdefault("BATOID_RUBIN_DATA_DIR", args.data_dir)
    bd = str(Path(args.data_dir) / args.bend_dir)
    import yaml
    cfg = yaml.safe_load(open(f"{bd}/bend.yaml"))
    n1 = len(cfg["use_m1m3_modes"]); n2 = len(cfg["use_m2_modes"])

    out = Path(__file__).resolve().parent.parent / "output" / "mode_gallery_full.pdf"
    with PdfPages(out) as pdf:
        build_mirror(pdf, "M1M3", n1, bd)
        build_mirror(pdf, "M2", n2, bd)
    print(f"wrote {out}  (M1M3 {n1} + M2 {n2} modes)")


if __name__ == "__main__":
    main()
