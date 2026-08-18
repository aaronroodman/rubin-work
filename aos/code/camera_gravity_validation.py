#!/usr/bin/env python
"""Validation plots for the camera-gravity model: what is actually being
computed, and what is included vs omitted.

Answers the two questions raised in the MIW investigation:
  (1) Is the gravity impact bulk (rigid) motion of the lenses, or surface
      DEFORMATION?  -> per-surface Zernike decomposition of the zer response.
  (2) Is there FEA for the sag of L1/L2 w.r.t. L3 and the focal plane, and is
      it applied?  -> the RB rigid-body files (present, NOT applied by
      batoid_rubin); their magnitudes vs elevation; and their incremental effect
      on the wavefront (surface-only vs surface+RB).

Output: ../output/camera_gravity/camera_gravity_validation.pdf

Laptop-runnable (needs only batoid_rubin).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

import camera_gravity as cg

OUT = Path(__file__).resolve().parents[1] / "output" / "camera_gravity"
OUT.mkdir(parents=True, exist_ok=True)

# Andy->Noll remap used by batoid_rubin for the zer/camera surface files
ZMAP = [1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
        17, 21, 16, 25, 24, 26, 23, 27, 22, 28]
SURFS = [("L1S1", "L1 ent"), ("L1S2", "L1 exit"), ("L2S1", "L2 ent"),
         ("L2S2", "L2 exit"), ("L3S1", "L3 ent"), ("L3S2", "L3 exit")]
GROUPS = [("piston+tilt\n(Z1-3, bulk)", 1, 3), ("power\n(Z4)", 4, 4),
          ("astig\n(Z5-6)", 5, 6), ("coma\n(Z7-8)", 7, 8), ("higher\n(Z9+)", 9, 28)]
RB_FILES = [("L1RB", "L1"), ("L2RB", "L2"), ("L3RB", "L3"),
            ("FRB", "Filter"), ("FPRB", "Focal plane")]
RB_COLS = ["dx", "dy", "dz", "Rx", "Ry", "Rz"]


def to_noll(row28):
    return np.asarray(row28)[[x - 1 for x in ZMAP]]


def page_surface_decomp(pdf, fea):
    """Per-surface zer response, grouped by Noll band (bulk vs deformation)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, case, row in zip(axes, ("axial (per cos z - 1)", "lateral (per sin z)"), (0, 1)):
        vals = np.zeros((len(SURFS), len(GROUPS)))
        for i, (stem, _) in enumerate(SURFS):
            n = to_noll(fits.getdata(f"{fea}/{stem}zer.fits.gz")[row, 3:]) * 1e3  # um
            for g, (_, lo, hi) in enumerate(GROUPS):
                vals[i, g] = np.sqrt(np.sum(n[lo - 1:hi] ** 2))
        x = np.arange(len(GROUPS)); w = 0.13
        for i, (_, lab) in enumerate(SURFS):
            ax.bar(x + (i - 2.5) * w, vals[i], w, label=lab)
        ax.set_yscale("log"); ax.set_ylim(1e-3, 200)
        ax.set_xticks(x); ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8)
        ax.set_title(f"Lens surface-figure gravity response — {case}", fontsize=10)
        ax.axhline(0.02, color="k", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("surface figure per unit gravity (µm RMS)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("What the gravity model APPLIES: per-surface figure on the 6 lens "
                 "surfaces (surface + Zernike).\nBulk piston/tilt (Z1-3) dominates axial; "
                 "the MIW-relevant ASTIG (Z5-6) comes from the LATERAL deformation of L1.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig); plt.close(fig)


def page_rb_magnitudes(pdf, fea, elevations):
    """RB rigid-body gravity displacement of each element vs elevation."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    els = np.linspace(20, 85, 40)
    for ci, col in enumerate(RB_COLS):
        ax = axes[ci]
        for stem, lab in RB_FILES:
            d = np.asarray(fits.getdata(f"{fea}/{stem}.fits.gz"))
            vals = []
            for el in els:
                z = np.deg2rad(90 - el)
                g = d[0, 3 + ci] * (np.cos(z) - 1) + d[1, 3 + ci] * np.sin(z)
                vals.append(g)
            vals = np.array(vals)
            scale = 1e3 if ci < 3 else 1e6   # mm->um  ;  rad->urad
            ax.plot(els, vals * scale, label=lab)
        ax.set_title(f"{col} ({'µm' if ci < 3 else 'µrad'})", fontsize=10)
        ax.set_xlabel("elevation (deg)"); ax.grid(alpha=0.3)
        ax.invert_xaxis()
        if ci == 0:
            ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("What the gravity model OMITS: rigid-body gravity sag from the RB "
                 "files (present in fea_legacy, never applied by batoid_rubin).\n"
                 "Focal-plane (FPRB) motion and lens decenters are entirely missing "
                 "from the shipped model.  (rot=0; ZCS raw columns.)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig); plt.close(fig)


def page_incremental(pdf, band, elevations):
    """Surface-only vs surface+RB: incremental Z4-Z8 effect on the wavefront."""
    b = cg.base_builder(band)
    wl = cg.WL[band] * 1e-9
    cg.C.RINGS, cg.C.SPOKES, cg.C.NX = 6, 15, 63
    dz_design = cg.C.double_zernike(b.build(), cg.FIELD_RAD, wl)
    t = np.linspace(-1.6, 1.6, 25); X, Y = np.meshgrid(t, t)
    inside = np.hypot(X, Y) <= 1.6; xg, yg = X[inside], Y[inside]
    nolls = (4, 5, 6, 7, 8)

    fig, axes = plt.subplots(1, len(nolls), figsize=(15, 4.2), sharey=True)
    for ax, j in zip(axes, nolls):
        so, rb = [], []
        for el in elevations:
            for store, use_rb in ((so, False), (rb, True)):
                dz, _ = cg.gravity_dz(b, el, band, 0.0, use_rb, dz_design)
                m = cg.dz_to_maps(dz, xg, yg, band, nolls=(j,))[j]
                store.append(cg.map_stats(m, xg, yg)[0])
        x = np.arange(len(elevations)); w = 0.35
        ax.bar(x - w / 2, so, w, label="surface only", color="tab:blue")
        ax.bar(x + w / 2, rb, w, label="surface + RB", color="tab:orange")
        ax.set_xticks(x); ax.set_xticklabels([f"{int(e)}" for e in elevations])
        ax.set_title(f"Z{j}", fontsize=11); ax.set_xlabel("elevation")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("µm RMS over field"); axes[0].legend(fontsize=9)
    fig.suptitle("Effect of adding the RB rigid-body sag: it changes mostly DEFOCUS "
                 "(Z4); astig/coma (Z5-Z8) are nearly unchanged.\n"
                 "=> the omitted focal-plane / lens-decenter motion does NOT hide a "
                 "large astig/coma gravity contribution.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    pdf.savefig(fig); plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="i")
    ap.add_argument("--elevations", type=float, nargs="+", default=[70, 45, 30])
    args = ap.parse_args()
    fea = str(cg.data_dir() / "fea_legacy")
    out = OUT / "camera_gravity_validation.pdf"
    with PdfPages(str(out)) as pdf:
        page_surface_decomp(pdf, fea)
        page_rb_magnitudes(pdf, fea, args.elevations)
        page_incremental(pdf, args.band, args.elevations)
    print("wrote", out)


if __name__ == "__main__":
    main()
