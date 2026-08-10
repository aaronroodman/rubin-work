#!/usr/bin/env python
"""
RubinTV-style PSF-shape focal-plane plot for the simulation, using the REAL
summit_extras plotting code (lsst.summit.extras.plotting.psfPlotting).

The sim's FoV output (guider_atmo_sim.py --mode fov, or guiders --also-fov) already
carries the columns summit_extras' makeTableFromSourceCatalogs produces, all from
HSM adaptive moments: FWHM/e1/e2/e (2nd order) and coma1/coma2/trefoil1/trefoil2/
kurtosis (HSM-standardized higher-order). This builds the astropy Table + meta the
plotter expects, runs extendTable + makeAzElPlot, and writes the figure -- so the
sim and on-sky data go through the identical plotting path for direct comparison.

  # on the RSP (needs summit_extras + lsst.obs.lsst):
  python guider_atmo_psfshape.py --outdir output/atmo_sim_20260706_688 --save psfshape.png

Reads guider_atmo_fov_moments.parquet + guider_atmo_fov_meta.json from --outdir.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def build_table(outdir, visit, meta):
    """Assemble the summit_extras psf-shape astropy Table from the sim FoV output."""
    from astropy.table import Table
    import lsst.afw.cameraGeom as cg
    from lsst.obs.lsst import LsstCam
    from lsst.summit.extras.plotting.psfPlotting import extendTable, MM_TO_DEG

    df = pd.read_parquet(os.path.join(outdir, "guider_atmo_fov_moments.parquet"))
    df = df[(df.visit == visit) & df.ok].reset_index(drop=True)
    camera = LsstCam().getCamera()

    # focal-plane position [mm] + integer detector id, from the real geometry
    xfp, yfp, detid = [], [], []
    for name in df.det:
        det = camera[name]
        c = det.getCenter(cg.FOCAL_PLANE)
        xfp.append(c.getX()); yfp.append(c.getY()); detid.append(det.getId())

    t = Table()
    t["detector"] = detid
    t["x"], t["y"] = xfp, yfp                          # FP mm
    t["Ixx"], t["Iyy"], t["Ixy"] = df.Ixx_asec2, df.Iyy_asec2, df.Ixy_asec2   # arcsec^2
    t["T"] = t["Ixx"] + t["Iyy"]
    t["FWHM"] = np.sqrt(t["T"] / 2 * np.log(256))
    t["e1"], t["e2"], t["e"] = df.e1, df.e2, df.e
    for c in ("coma1", "coma2", "trefoil1", "trefoil2", "kurtosis"):
        t[c] = df[c]

    # rotations (same construction as makeTableFromSourceCatalogs)
    rtp = np.deg2rad(meta.get("rotTelPos_deg", 0.0))
    rsp = np.deg2rad(meta.get("rotSkyPos_deg", 0.0))
    srtp, crtp = np.sin(rtp), np.cos(rtp)
    aaRot = (np.array([[crtp, srtp], [-srtp, crtp]])
             @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]]))
    srsp, crsp = np.sin(rsp), np.cos(rsp)
    nwRot = np.array([[crsp, -srsp], [srsp, crsp]])
    t = extendTable(t, aaRot, "aa", MM_TO_DEG)
    t = extendTable(t, nwRot, "nw", MM_TO_DEG)
    t.meta["aaRot"], t.meta["nwRot"] = aaRot, nwRot
    t.meta["rotTelPos"], t.meta["rotSkyPos"] = rtp, rsp
    t.meta["az"], t.meta["el"] = meta.get("az", 0.0), meta.get("el", 90.0)
    t.meta["LSST BUTLER DATAID VISIT"] = (int(meta.get("dayObs", 0)) * 100000
                                          + int(meta.get("seqNum", 0)))
    return t, camera


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--visit", type=int, default=0)
    ap.add_argument("--save", default=None, help="output PNG (default in --outdir)")
    ap.add_argument("--band", default=None, help="override physical filter (else meta band)")
    ap.add_argument("--marker-size", type=float, default=40.0,
                    help="marker size for the FWHM/e1/e2 scatter panels (summit_extras uses 1)")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "guider_atmo_fov_meta.json")) as fh:
        meta = json.load(fh)
    t, camera = build_table(args.outdir, args.visit, meta)

    import matplotlib.collections as mcoll
    from lsst.summit.extras.plotting.psfPlotting import makeFigureAndAxes, makeAzElPlot
    fig, axs = makeFigureAndAxes(nrows=3)               # 3 rows -> incl. coma/trefoil/kurtosis
    band = args.band or meta.get("band", "i")
    makeAzElPlot(fig, axs, t, camera, physicalFilter=band, saveAs="")  # draw, don't save yet
    # enlarge the color-map scatter markers (FWHM, e1, e2) -- helpful for one-star-per-CCD
    for ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
        for coll in ax.collections:
            if isinstance(coll, mcoll.PathCollection):
                coll.set_sizes([args.marker_size])
    save = args.save or os.path.join(args.outdir, f"guider_atmo_psfshape_visit{args.visit:02d}.png")
    fig.savefig(save)
    print(f"wrote {save}  ({len(t)} CCDs)  via summit_extras makeAzElPlot")


if __name__ == "__main__":
    main()
