#!/usr/bin/env python3
"""Adjacent-CCD PSF star moments for the guider comparison (RUN ON USDF).

For each science visit of a night, and each guider's adjacent science CCD
(from guider_adjacent_ccds.yaml), select clean PSF stars from the processed
collection (guider_science_collections.yaml) and record:
  - median 2nd moments from single_visit_star (ixx,iyy,ixy -> e1,e2), and
  - median 3rd moments (coma M21/M12, trefoil M30/M03) recomputed from
    preliminary_visit_image stamps via optatmo's HSM estimator.

Writes guider_psfmoments_<dayObs>.parquet, one row per (expId, guider,
adjacent_ccd), for joining to the guider moments on (expId, detector).

NOTE: moments here are HSM-*weighted* and in the science-CCD pixel (DVCS) frame;
``rot_deg`` and the CCD field angle are recorded so the comparison notebook can
rotate both sides to a common frame (the guider moments are unweighted -- treat
absolute-scale differences with that in mind).

    python extract_adjacent_psf_moments.py --day-obs 20260709 --out <dir> \
        [--repo /repo/main] [--max-per-ccd 45] [--snr-min 80] [--no-third]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "optatmo", "code"))

HALF = 16


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--out", required=True, help="Output dir (writes guider_psfmoments_<dayObs>.parquet).")
    p.add_argument("--repo", default="/repo/main")
    p.add_argument("--adjacent-yaml", default=os.path.join(_HERE, "guider_adjacent_ccds.yaml"))
    p.add_argument("--collections-yaml", default=os.path.join(_HERE, "guider_science_collections.yaml"))
    p.add_argument("--collection", default=None, help="Override the collection for this night.")
    p.add_argument("--max-per-ccd", type=int, default=45)
    p.add_argument("--snr-min", type=float, default=80.0)
    p.add_argument("--no-third", action="store_true", help="Skip 3rd-order (no image reads).")
    return p.parse_args(argv)


def cleanStars(st):
    snr = st.psfFlux / st.psfFluxErr
    return st[st.detect_isPrimary & (st.extendedness < 0.5) & (snr > 0)
             & (~st.pixelFlags_saturated) & (~st.pixelFlags_edge)
             & (~st.pixelFlags_interpolatedCenter) & (~st.pixelFlags_bad)
             & np.isfinite(st.psfFlux) & (st.psfFlux > 0)]


def visitRotDeg(butler, collection, visit, ccds):
    """Boresight rotation angle (deg) for a visit.

    The ``visit`` dimension record carries no rotation field, so read it from
    the image's ``visitInfo`` (a component get, no pixels loaded). Try each
    adjacent CCD in turn in case one is missing; nan if none resolve.
    """
    for ccd in ccds:
        try:
            vi = butler.get("preliminary_visit_image.visitInfo", collections=collection,
                            instrument="LSSTCam", visit=visit, detector=ccd)
            return float(vi.boresightRotAngle.asDegrees())
        except Exception:
            continue
    return float("nan")


def main(argv=None):
    args = parseArgs(argv)
    adj = yaml.safe_load(open(args.adjacent_yaml))
    guiderToCcd = {g: int(v["adjacent_ccd"]) for g, v in adj.items()}

    collection = args.collection
    if collection is None:
        coll_map = yaml.safe_load(open(args.collections_yaml))
        entry = coll_map.get(args.day_obs) or coll_map.get(str(args.day_obs))
        if not entry or not entry.get("n_visits"):
            print(f"{args.day_obs}: no processed science collection; writing empty.", file=sys.stderr)
            os.makedirs(args.out, exist_ok=True)
            pd.DataFrame(columns=["expId", "guider", "adjacent_ccd"]).to_parquet(
                os.path.join(args.out, f"guider_psfmoments_{args.day_obs}.parquet"), index=False)
            return
        # prefer a concrete run for reproducibility
        runs = entry.get("runs") or [entry["collection"]]
        collection = runs[0]

    import galsim
    from lsst.daf.butler import Butler
    from lsst.obs.lsst import LsstCam
    from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
    import lsst.geom as geom
    from moments_hsm import measure_hsm_moments   # optatmo/code

    butler = Butler(args.repo)
    cam = LsstCam.getCamera()
    ccds = sorted(set(guiderToCcd.values()))

    visits = sorted({r.dataId["visit"] for r in butler.registry.queryDatasets(
        "single_visit_star", collections=collection,
        where="instrument='LSSTCam' AND visit.day_obs = d", bind={"d": args.day_obs},
        findFirst=True)})
    print(f"{args.day_obs}: {len(visits)} visits in {collection}")

    rows = []
    for visit in visits:
        try:
            st = butler.get("single_visit_star", collections=collection,
                            instrument="LSSTCam", visit=visit).to_pandas()
        except Exception:
            continue
        ps = cleanStars(st)
        ps = ps[ps.detector.isin(ccds)]
        rot_deg = visitRotDeg(butler, collection, int(visit), ccds)

        for guider, ccd in guiderToCcd.items():
            sub = ps[ps.detector == ccd]
            sub = sub[(sub.psfFlux / sub.psfFluxErr) > args.snr_min]
            row = {"expId": int(visit), "dayObs": args.day_obs, "guider": guider,
                   "adjacent_ccd": ccd, "rot_deg": rot_deg, "n_stars": int(len(sub))}
            if len(sub):
                ixx, iyy, ixy = sub.ixx.median(), sub.iyy.median(), sub.ixy.median()
                tr = (ixx + iyy)
                row.update(ixx_ccd=float(ixx), iyy_ccd=float(iyy), ixy_ccd=float(ixy),
                           e1_ccd=float((ixx - iyy) / tr), e2_ccd=float(2 * ixy / tr))
            if len(sub) and not args.no_third:
                try:
                    exp = butler.get("preliminary_visit_image", collections=collection,
                                     instrument="LSSTCam", visit=visit, detector=ccd)
                    img = exp.image.array
                    ny, nx = img.shape
                    m21, m12, m30, m03 = [], [], [], []
                    for _, r in sub.sort_values("psfFlux", ascending=False).head(args.max_per_ccd).iterrows():
                        xi, yi = int(round(r.x)), int(round(r.y))
                        if xi < HALF or yi < HALF or xi >= nx - HALF or yi >= ny - HALF:
                            continue
                        stamp = np.ascontiguousarray(
                            img[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1].astype(float))
                        mom, _ = measure_hsm_moments(galsim.Image(stamp, scale=0.2),
                                                     fourth_order=False, radial=False)
                        if mom is None:
                            continue
                        m21.append(mom["M21"]); m12.append(mom["M12"])
                        m30.append(mom["M30"]); m03.append(mom["M03"])
                    if m21:
                        row.update(coma1_ccd=float(np.median(m21)), coma2_ccd=float(np.median(m12)),
                                   tref1_ccd=float(np.median(m30)), tref2_ccd=float(np.median(m03)),
                                   n_third=len(m21))
                except Exception as exc:  # noqa: BLE001
                    print(f"  visit {visit} ccd {ccd}: 3rd-order failed: {exc}", file=sys.stderr)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"guider_psfmoments_{args.day_obs}.parquet")
    df.to_parquet(out, index=False)
    print(f"{args.day_obs}: wrote {len(df)} rows ({df.expId.nunique()} visits) -> {out}")


if __name__ == "__main__":
    main()
