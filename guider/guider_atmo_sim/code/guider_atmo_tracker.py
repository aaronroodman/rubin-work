#!/usr/bin/env python
"""
Emit a GuiderStarTracker-format table from the simulation, for tomographic-seeing
analysis (Jackie). Option A: reproduce the tracker's exact output schema
(summit_utils DEFAULT_COLUMNS) by running the REAL geometry conversions
(convertToFocalPlane / convertToAltaz / computeOffsets / computeRotatorAngle /
convertEllipticity, + makeInitGuiderWcs / getCamRotAngle) on the sim's per-stamp
guider centroids. Also writes the Cn^2 profile Jackie compares against.

Per (guider, stamp): the star's CCD pixel is FIELD_ANGLE->PIXELS of (its field
angle + the DVCS centroid shift) -- both in the FOCAL_PLANE orientation the sim
stamps use, so no per-CCD rotation is needed -- then the real helpers give
xfp/yfp/alt/az and the offset/rotator/ellipticity columns.

  # RSP only (needs summit_utils + a butler for the visit's VisitInfo):
  python guider_atmo_tracker.py --outdir output/atmo_sim_20260706_688 --repo /repo/main \
      --save guider_tracker_table.parquet

Reads guider_atmo_moments.parquet + guider_atmo_meta.json (+ guider_atmo_atmosphere.json
for Cn^2) from --outdir.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def build_tracker_table(outdir, repo, sci_det=94, visit=0):
    import lsst.afw.cameraGeom as cg
    import lsst.geom as geom
    from lsst.daf.butler import Butler
    from lsst.obs.lsst import LsstCam
    from lsst.summit.utils.guiders.reading import getVisitInfo
    from lsst.summit.utils.guiders.transformation import (
        makeInitGuiderWcs, getCamRotAngle, convertToFocalPlane, convertToAltaz)
    from lsst.summit.utils.guiders.tracking import (
        computeOffsets, computeRotatorAngle, convertEllipticity)
    from lsst.summit.utils.guiders.detection import DEFAULT_COLUMNS
    from astropy.time import Time

    mom = pd.read_parquet(os.path.join(outdir, "guider_atmo_moments.parquet"))
    mom = mom[mom.visit == visit]
    with open(os.path.join(outdir, "guider_atmo_meta.json")) as fh:
        meta = json.load(fh)
    dayObs, seqNum = int(meta["dayObs"]), int(meta["seqNum"])
    cadence = float(meta.get("cadence", 0.2))
    band, ps, gain = meta.get("band", "i"), meta.get("pixel_scale", 0.2), meta.get("gain", 1.6)
    expid = dayObs * 100000 + seqNum

    camera = LsstCam().getCamera()
    butler = Butler(repo)
    visitInfo = getVisitInfo(butler, dayObs, seqNum, sci_det)
    wcsMap = makeInitGuiderWcs(camera, visitInfo)
    camRot = getCamRotAngle(visitInfo)
    obsTime = Time(visitInfo.date.toPython())
    R2A = 206265.0

    frames = []
    for name, g in mom.groupby("guider"):
        det = camera[name]
        detid = det.getId()
        fa2pix = det.getTransform(cg.FIELD_ANGLE, cg.PIXELS)
        g = g.sort_values("stamp")
        roi = None  # roi center from cen offset convention: xroi = roi/2 + cen; use meta
        roi = int(meta.get("roi", 200))
        # star field angle center (arcsec) + per-stamp DVCS shift (arcsec), -> CCD pix
        fx = (g["theta_x_asec"].to_numpy() + g["cen_x_asec"].to_numpy()) / R2A
        fy = (g["theta_y_asec"].to_numpy() + g["cen_y_asec"].to_numpy()) / R2A
        xccd, yccd = [], []
        for a, b in zip(fx, fy):
            p = fa2pix.applyForward(geom.Point2D(a, b))
            xccd.append(p.getX()); yccd.append(p.getY())
        xccd, yccd = np.array(xccd), np.array(yccd)
        n_e = np.clip(g["flux"].to_numpy() * gain, 1.0, None)
        sig_pix = g["fwhm_asec"].to_numpy() / (2.3548 * ps)
        cerr = sig_pix / np.sqrt(n_e)                        # centroid error [pix]
        df = pd.DataFrame(dict(
            detector=name, detid=detid, expid=expid, stamp=g["stamp"].to_numpy(),
            elapsed_time=g["stamp"].to_numpy() * cadence,
            timestamp=[(obsTime + k * cadence / 86400.0).isot for k in g["stamp"]],
            xroi=roi / 2.0 + g["cen_x"].to_numpy(), yroi=roi / 2.0 + g["cen_y"].to_numpy(),
            xccd=xccd, yccd=yccd, fwhm=g["fwhm_asec"].to_numpy(),
            ixx=g["Ixx"].to_numpy(), iyy=g["Iyy"].to_numpy(), ixy=g["Ixy"].to_numpy(),
            e1=g["e1"].to_numpy(), e2=g["e2"].to_numpy(),
            flux=g["flux"].to_numpy(), flux_err=np.sqrt(n_e) / gain,
            snr=n_e / np.sqrt(n_e), xerr=cerr, yerr=cerr,
            trackid=detid, ampname="C00", filter=band,
            exptime=float(meta.get("exptime", 0.05))))
        df["xfp"], df["yfp"] = convertToFocalPlane(xccd, yccd, detid)
        df["alt"], df["az"] = convertToAltaz(xccd, yccd, wcsMap[name], obsTime)
        frames.append(df)

    stars = pd.concat(frames, ignore_index=True)
    stars = computeOffsets(stars)                            # refs, dx/dy, dxfp/dyfp, dalt/daz
    stars = computeRotatorAngle(stars)                       # theta, dtheta, theta_err
    stars = convertEllipticity(stars, np.rad2deg(camRot))    # e1_altaz, e2_altaz
    for c in DEFAULT_COLUMNS:                                # ensure full schema
        if c not in stars.columns:
            stars[c] = np.nan
    return stars[list(DEFAULT_COLUMNS)], meta


def write_cn2(outdir, save):
    """Flat Cn^2 profile parquet for tomography (from the atmosphere validation JSON)."""
    p = os.path.join(outdir, "guider_atmo_atmosphere.json")
    if not os.path.isfile(p):
        return
    with open(p) as fh:
        a = json.load(fh)
    df = pd.DataFrame(a["layers"])
    df["r0_500_effective_m"] = a["r0_500_effective_m"]
    df["seeing_fwhm_arcsec"] = a["seeing_fwhm_target_arcsec"]
    df.to_parquet(save, index=False)
    print(f"wrote {save}  (Cn2 profile, {len(df)} layers)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repo", default="/repo/main", help="butler repo for VisitInfo")
    ap.add_argument("--sci-det", type=int, default=94, help="a science detNum for VisitInfo")
    ap.add_argument("--visit", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    stars, meta = build_tracker_table(args.outdir, args.repo, args.sci_det, args.visit)
    save = args.save or os.path.join(args.outdir, "guider_tracker_table.parquet")
    stars.to_parquet(save, index=False)
    print(f"wrote {save}  ({len(stars)} rows, GuiderStarTracker schema)")
    write_cn2(args.outdir, os.path.join(os.path.dirname(save) or ".",
                                        "guider_tracker_cn2.parquet"))


if __name__ == "__main__":
    main()
