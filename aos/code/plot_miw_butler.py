#!/usr/bin/env python
"""Plot the MIW as ingested in the Butler (``intrinsicZernikes`` calib).

Reads ``lsst.ip.isr.IntrinsicZernikes`` from the Butler and writes a PDF of the
field maps (y_field vs x_field, i.e. thy vs thx in OCS/FIELD_ANGLE deg):

  * one panel per Noll index of the OCS (telescope-fixed) component, and
  * the Z4 CCS (camera-fixed: smooth camera field + per-CCD height) as a
    focal-plane map.

The OCS system is a single whole-field map shared by all detectors; the CCS
system is per-detector (Z4 piston = CCD height + camera piece).  We therefore
build the OCS maps from one detector's calib and the CCS Z4 map by reading many
detectors.

Two evaluation paths (the script tries the clean one first):
  A. interpolators  -- calib.interpolator_ocs / calib.interpolator evaluated on a
     grid: gives OCS and CCS SEPARATELY (preferred).
  B. fallback       -- calib.getIntrinsicZernikes(...) mosaicked over detectors
     at rotTelPos=0 (OCS+CCS combined; for Noll>=5 CCS~0 so this is ~OCS, and
     the Z4 map then shows OCS Z4 + the per-CCD CCS steps).

Usage (RSP):
  python code/plot_miw_butler.py \
      --repo /repo/main \
      --collection LSSTCam/calib/DM-55048/intrinsicZernikes.v2 \
      --physical-filter r_03 \
      --out output/miw_butler_maps.pdf
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lsst.daf.butler import Butler
import lsst.geom as geom
from lsst.afw.cameraGeom import FIELD_ANGLE


def detector_field_center(camera, det_id):
    """FIELD_ANGLE center of a detector, in degrees (thx, thy)."""
    det = camera[det_id]
    ctr = det.getCenter(FIELD_ANGLE)   # radians
    return np.rad2deg(ctr.getX()), np.rad2deg(ctr.getY())


def eval_maps_via_interpolators(calib, gx, gy):
    """Return (ocs_vals, ccs_vals) each (Npts, Nnoll) via the calib interpolators.

    Raises if the interpolator call signature differs (caller falls back).
    """
    pts = np.column_stack([gx, gy])
    ocs = np.asarray(calib.interpolator_ocs(pts), dtype=float)
    ccs = np.asarray(calib.interpolator(pts), dtype=float)
    if ocs.ndim != 2:
        raise ValueError("unexpected interpolator output shape")
    return ocs, ccs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--collection", required=True,
                    help="collection holding the intrinsicZernikes calib")
    ap.add_argument("--dataset-type", default="intrinsicZernikes")
    ap.add_argument("--physical-filter", required=True,
                    help="e.g. r_03 (OCS is filter-independent; pick any present)")
    ap.add_argument("--instrument", default="LSSTCam")
    ap.add_argument("--out", default="output/miw_butler_maps.pdf")
    ap.add_argument("--grid-n", type=int, default=200)
    ap.add_argument("--fp-radius-deg", type=float, default=1.75)
    args = ap.parse_args()

    butler = Butler(args.repo, collections=[args.collection])
    camera = butler.get("camera", instrument=args.instrument)

    # detectors that have the calib in this collection
    refs = list(butler.registry.queryDatasets(
        args.dataset_type, collections=args.collection,
        where=f"instrument='{args.instrument}' AND physical_filter='{args.physical_filter}'"))
    det_ids = sorted({r.dataId["detector"] for r in refs})
    if not det_ids:
        raise SystemExit("no intrinsicZernikes found for that filter/collection")
    print(f"{len(det_ids)} detectors with {args.dataset_type} "
          f"({args.physical_filter}) in {args.collection}")

    def get_calib(det):
        return butler.get(args.dataset_type, instrument=args.instrument,
                          detector=int(det), physical_filter=args.physical_filter)

    calib0 = get_calib(det_ids[len(det_ids) // 2])
    noll = list(np.asarray(calib0.noll_indices).astype(int))
    print(f"Noll indices: {noll}")

    # full-field grid inside the FoV
    g = np.linspace(-args.fp_radius_deg, args.fp_radius_deg, args.grid_n)
    GX, GY = np.meshgrid(g, g)
    inside = np.hypot(GX, GY) <= args.fp_radius_deg
    gx, gy = GX[inside], GY[inside]

    # ---- OCS whole-field maps ----
    used = "interpolators"
    try:
        ocs_vals, _ = eval_maps_via_interpolators(calib0, gx, gy)
        ocs_maps = {j: ocs_vals[:, k] for k, j in enumerate(noll)}
        ocs_xy = (gx, gy)
    except Exception as e:                                   # fallback path B
        used = f"getIntrinsicZernikes mosaic (interpolators failed: {e})"
        xs, ys, vals = [], [], []
        for det in det_ids:
            c = get_calib(det)
            fx = np.asarray(c.field_x, float); fy = np.asarray(c.field_y, float)
            v = np.asarray(c.getIntrinsicZernikes(field_x=fx, field_y=fy, rotTelPos=0.0))
            xs.append(fx); ys.append(fy); vals.append(v)
        gx = np.concatenate(xs); gy = np.concatenate(ys); V = np.concatenate(vals)
        ocs_maps = {j: V[:, k] for k, j in enumerate(noll)}
        ocs_xy = (gx, gy)
    print(f"OCS maps via: {used}")

    # ---- CCS Z4 per-detector (camera field + CCD height) ----
    cx, cy, cz4 = [], [], []
    j4 = noll.index(4) if 4 in noll else None
    for det in det_ids:
        c = get_calib(det)
        dx, dy = detector_field_center(camera, det)
        try:
            ccs = np.asarray(c.interpolator(np.array([[dx, dy]])), dtype=float)[0]
            z4 = ccs[j4] if j4 is not None else np.nan
        except Exception:
            # fallback: total - would need OCS subtraction; leave NaN
            z4 = np.nan
        cx.append(dx); cy.append(dy); cz4.append(z4)
    cx, cy, cz4 = map(np.asarray, (cx, cy, cz4))
    have_ccs = np.isfinite(cz4).any()

    # ---- plot ----
    ox, oy = ocs_xy
    with PdfPages(args.out) as pdf:
        # OCS maps, ~6 per page
        per = 6
        for start in range(0, len(noll), per):
            chunk = noll[start:start + per]
            fig, ax = plt.subplots(2, 3, figsize=(15, 9))
            for a, j in zip(ax.flat, chunk):
                v = ocs_maps[j]; vm = np.nanpercentile(np.abs(v), 98)
                sc = a.scatter(ox, oy, c=v, s=6, cmap="RdBu_r", vmin=-vm, vmax=vm, rasterized=True)
                a.set_title(f"Z{j} OCS  (rms={np.nanstd(v):.3f})", fontsize=10)
                a.set_aspect("equal"); a.set_xlabel("x_field [deg]"); a.set_ylabel("y_field [deg]")
                plt.colorbar(sc, ax=a, shrink=.8, label="µm")
            for a in ax.flat[len(chunk):]:
                a.axis("off")
            fig.suptitle(f"MIW OCS field maps (Butler {args.dataset_type}, {args.collection})", fontsize=12)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Z4 CCS focal-plane map
        fig, a = plt.subplots(figsize=(7.5, 6.5))
        if have_ccs:
            vm = np.nanpercentile(np.abs(cz4), 98)
            sc = a.scatter(cx, cy, c=cz4, s=90, cmap="RdBu_r", vmin=-vm, vmax=vm, edgecolor="k", lw=.2)
            plt.colorbar(sc, ax=a, shrink=.8, label="µm")
            a.set_title(f"Z4 CCS per detector (CCD height + camera field)\nrms={np.nanstd(cz4):.3f} µm")
        else:
            a.text(0.5, 0.5, "CCS interpolator not accessible;\nrerun path B or supply source tables",
                   ha="center", va="center")
            a.set_title("Z4 CCS (unavailable via this path)")
        a.set_aspect("equal"); a.set_xlabel("x_field [deg]"); a.set_ylabel("y_field [deg]")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
