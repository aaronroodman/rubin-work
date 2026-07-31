#!/usr/bin/env python
"""M1M3 B52 -- the azimuthally symmetric mode not in the AOS-used 20.

B52 (raw index 51) is 96% m=0 (axisymmetric), dominated by Noll Z4/Z11/Z22
(defocus + primary & secondary spherical).  Its field-constant DZ is primarily
primary spherical Z11 -- i.e. B52 is the M1M3 mode that controls spherical
aberration, which the AOS 20-mode set omits.

Output: ../output/m1m3_B52.png
"""
from pathlib import Path
import numpy as np
import galsim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from batoid_rubin.builder import load_bend

OUT = Path(__file__).resolve().parent.parent / "output"
BZ = "/Users/roodman/LSST/batoid_rubin_data/bend_zemax"
I = 51   # B52 (0-based raw index)


def surf(bm, X, Y):
    zk = galsim.zernike.Zernike(np.asarray(bm.zk)[I], R_outer=bm.R_outer, R_inner=bm.R_inner)
    s = (zk(X, Y) + np.asarray(bm.z)[I]) * 1e6
    R = np.hypot(X, Y)
    return np.where((R >= bm.R_inner) & (R <= bm.R_outer), s, np.nan)


def main():
    import os
    os.environ.setdefault("BATOID_RUBIN_DATA_DIR", "/Users/roodman/LSST/batoid_rubin_data")
    modes = load_bend(BZ, tuple(range(60)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (1) surface
    ax = axes[0]
    vmax = 0
    parts = []
    for nm in ("M1", "M3"):
        bm = modes[nm]
        X, Y = np.meshgrid(np.asarray(bm.x), np.asarray(bm.y))
        s = surf(bm, X, Y); parts.append((X, Y, s))
        vmax = max(vmax, np.nanmax(np.abs(s)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for X, Y, s in parts:
        im = ax.pcolormesh(X, Y, s, cmap="RdBu_r", norm=norm, shading="auto")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("M1M3 B52 surface (µm, unit coeff)\naxisymmetric")
    fig.colorbar(im, ax=ax, shrink=0.8, label="µm")

    # (2) zk content (surface Zernike, m=0 highlighted)
    ax = axes[1]
    zk = np.asarray(modes["M1"].zk)[I]
    js = np.arange(1, 29)
    m0 = np.array([galsim.zernike.noll_to_zern(j)[1] == 0 for j in js])
    ax.bar(js[~m0], np.abs(zk[js][~m0]) * 1e6, color="lightgray", label="m≠0")
    ax.bar(js[m0], np.abs(zk[js][m0]) * 1e6, color="crimson", label="m=0 (axisym)")
    ax.set_xlabel("surface Noll Z"); ax.set_ylabel("|coef| (µm)")
    ax.set_title("B52 surface Zernike content\n96% in m=0 (Z1,Z4,Z11,Z22)")
    ax.legend()

    # (3) field-constant DZ (wavefront pupil-Noll sensitivity)
    ax = axes[2]
    sens = np.load(OUT / "smatrix_dz_31_29_232_r_bend_zemax.npy")
    fc = sens[1, :, 10 + I]                    # field-constant, pupil Noll
    js = np.arange(4, 23)
    ax.bar(js, fc[js], color=["crimson" if j in (4, 11, 22) else "steelblue" for j in js])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("pupil Noll Z"); ax.set_ylabel("µm WF / µm mode")
    ax.set_title("B52 field-constant wavefront sensitivity\nprimary spherical Z11 (+ Z4, Z22)")

    fig.suptitle("M1M3 B52 — axisymmetric spherical-aberration mode (not in AOS-used 20)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "m1m3_B52.png", dpi=140)
    print("wrote", OUT / "m1m3_B52.png")


if __name__ == "__main__":
    main()
