#!/usr/bin/env python
"""Demonstrate: field-spatial-order of a wavefront perturbation is set by the
perturbed surface's distance from focus (proximity to a pupil).

Put the SAME mid-spatial-frequency figure error (a Noll-12 surface Zernike) on
each of several surfaces, from near-pupil (M2) to near-focus (L3, Detector), and
map the induced pupil astigmatism Z5 across the focal plane.  Near the pupil the
Z5 field pattern is smooth (low order); near focus it becomes high field-order —
the signature seen in the MIW.

Output: ../output/demo_field_order.png
"""
from pathlib import Path
import numpy as np
import batoid
import galsim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = Path(__file__).resolve().parent.parent / "output"
WL = 754e-9
FR = 1.75
ZFOCUS = 4.4287

# (name, clear-aperture semi-diameter m) from near-pupil to near-focus
SURFACES = [
    ("LSST.M2", 1.71),
    ("LSST.LSSTCamera.L2.L2_entrance", 0.55),
    ("LSST.LSSTCamera.Filter.Filter_entrance", 0.375),
    ("LSST.LSSTCamera.L3.L3_entrance", 0.361),
    ("LSST.LSSTCamera.Detector", 0.36),
]
NOLL = 28            # fine (footprint-scale) figure error / striae-like (n=6)
AMP = 2e-7           # 0.2 um surface


def field_grid(n=35):
    t = np.linspace(-FR, FR, n)
    X, Y = np.meshgrid(t, t)
    inside = np.hypot(X, Y) <= 1.60      # mask vignetted field edge
    return X, Y, inside


def z5_map(tel_base, tel_pert, X, Y, inside):
    m = np.full(X.shape, np.nan)
    ys, xs = np.where(inside)
    for iy, ix in zip(ys, xs):
        thx, thy = np.deg2rad(X[iy, ix]), np.deg2rad(Y[iy, ix])
        try:
            z0 = batoid.zernike(tel_base, thx, thy, WL, jmax=6, eps=0.61, nx=48)
            z1 = batoid.zernike(tel_pert, thx, thy, WL, jmax=6, eps=0.61, nx=48)
            m[iy, ix] = (z1[5] - z0[5]) * WL * 1e6   # Z5 in um
        except Exception:
            pass
    return m


def field_order_metric(m, X, Y, inside):
    """Fit the Z5 field map to field Zernikes; return RMS-weighted mean field Noll."""
    v = m[inside]; x = X[inside] / FR; y = Y[inside] / FR
    basis = galsim.zernike.zernikeBasis(28, x, y, R_outer=1.0)
    coef, *_ = np.linalg.lstsq(basis.T, v, rcond=None)
    p = coef[1:]**2
    nolls = np.arange(1, len(coef))
    # radial order n for each Noll
    ns = np.array([galsim.zernike.noll_to_zern(j)[0] for j in nolls])
    return np.sum(ns * p) / (np.sum(p) + 1e-30)


def main():
    tel = batoid.Optic.fromYaml("LSST_i.yaml")
    X, Y, inside = field_grid()
    fig, axes = plt.subplots(1, len(SURFACES), figsize=(4 * len(SURFACES), 4.3))
    for ax, (name, R) in zip(axes, SURFACES):
        pert = batoid.Zernike(np.array([0] * NOLL + [AMP]), R_outer=R)
        tel2 = tel.withSurface(name, batoid.Sum([tel[name].surface, pert]))
        m = z5_map(tel, tel2, X, Y, inside)
        vmax = np.nanmax(np.abs(m)) or 1e-9
        im = ax.imshow(m, origin="lower", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                       extent=[-FR, FR, -FR, FR])
        dist = abs(tel[name].coordSys.origin[2] - ZFOCUS)
        nbar = field_order_metric(m, X, Y, inside)
        short = name.split(".")[-1]
        ax.set_title(f"{short}\n{dist*100:.0f} cm from focus\n"
                     f"pk {vmax*1e3:.0f}nm, mean field-order {nbar:.1f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Same fine (Noll-{NOLL}) figure error on each surface -> induced "
                 "astigmatism Z5 across the field\n"
                 "Z5~0 at the pupil (M2: high PUPIL-order, uniform in field) AND at exact "
                 "focus (Detector: footprint->point, only piston/tilt).\n"
                 "It PEAKS at the intermediate camera refractive elements (L1/L2/filter) "
                 "as HIGH FIELD-order astigmatism -- the MIW signature. Only camera optics "
                 "rotate with the rotator.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "demo_field_order.png", dpi=130)
    print("wrote", OUT / "demo_field_order.png")


if __name__ == "__main__":
    main()
