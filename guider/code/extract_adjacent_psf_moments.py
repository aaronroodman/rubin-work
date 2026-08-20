#!/usr/bin/env python3
"""Adjacent-CCD PSF star moments for the guider comparison (RUN ON USDF).

For each science visit of a night, and each guider's adjacent science CCD
(from guider_adjacent_ccds.yaml), select clean PSF stars from the processed
collection (guider_science_collections.yaml) and measure, from the
preliminary_visit_image star stamps via optatmo's galsim-HSM estimator, the
median AND RMS over that CCD's stars of:
  - 2nd moments: Q1/Q2/T (arcsec**2, directly comparable to the guider HSM
    hsm_Q1/hsm_Q2/hsm_T) and the normalized e1/e2; and
  - 3rd moments: coma (M21/M12) and trefoil (M30/M03), arcsec**3.

The SAME HSM estimator is applied to the guider coadd (guiderMoments hsm_*),
so the guider-vs-CCD comparison is matched-algorithm (no unweighted-vs-weighted
confound). n_stars = snr-passing single_visit_star count; n_hsm = stars actually
HSM-measured. Each stat has a companion ``*_rms`` column; ``pixscale_ccd``
records the 0.2 arcsec/pixel scale used. Writes guider_psfmoments_<dayObs>.parquet,
one row per (expId, guider, adjacent_ccd), for joining to the guider moments on
(expId, detector).

NOTE: moments are HSM-weighted, in the science-CCD pixel (DVCS) frame; ``rot_deg``
is recorded so the comparison notebook can rotate both sides to a common (OCS)
frame. T and magnitudes are frame-invariant; Q1/Q2/coma/trefoil components need
that rotation.

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
SCIENCE_PIXSCALE = 0.2   # arcsec/pixel; HSM image scale -> moments in arcsec**n


def medRms(a):
    """Median and RMS (std) of a 1-D array, as plain floats."""
    a = np.asarray(a, dtype=float)
    return float(np.nanmedian(a)), float(np.nanstd(a))


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
    p.add_argument("--image-types", nargs="+", default=["science"],
                   help="Keep only visits with these exposure observation_types "
                        "(default: science; excludes acq/eng where science CCDs aren't processed).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N science visits (0=all); for quick tests.")
    p.add_argument("--no-third", action="store_true",
                   help="Skip 3rd-order columns (2nd moments still measured from stamps).")
    return p.parse_args(argv)


def cleanStars(st):
    snr = st.psfFlux / st.psfFluxErr
    return st[st.detect_isPrimary & (st.extendedness < 0.5) & (snr > 0)
             & (~st.pixelFlags_saturated) & (~st.pixelFlags_edge)
             & (~st.pixelFlags_interpolatedCenter) & (~st.pixelFlags_bad)
             & np.isfinite(st.psfFlux) & (st.psfFlux > 0)]


def visitRotDeg(butler, collection, visit, ccds):
    """Physical camera rotator angle (deg), wrapped to (-180, 180].

    This is the mechanical rotator (rotTelPos, == ConsDB physical_rotator_angle),
    NOT visitInfo.boresightRotAngle (which is the sky position angle). It is
    derived from the two boresight angles (cf. aos/code/run_wfs_fam_compare.py):

        rotTelPos = parallacticAngle - boresightRotAngle - 90 deg

    verified against ConsDB physical_rotator_angle to ~0.3 deg. Read from the
    image's visitInfo (a component get, no pixels); try each adjacent CCD in
    turn in case one is missing; nan if none resolve.
    """
    for ccd in ccds:
        try:
            vi = butler.get("preliminary_visit_image.visitInfo", collections=collection,
                            instrument="LSSTCam", visit=visit, detector=ccd)
            rtp = (vi.boresightParAngle.asDegrees()
                   - vi.boresightRotAngle.asDegrees() - 90.0)
            return (rtp + 180.0) % 360.0 - 180.0
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

    # Restrict to the wanted observation_types (default science). Acq/eng visits
    # can have single_visit_star on only a stray CCD (no science-CCD photometry),
    # so they would contribute all-zero rows -- drop them (matches the guider side).
    wanted = {t.lower() for t in args.image_types}
    sci = {r.id for r in butler.registry.queryDimensionRecords(
        "exposure", where="instrument='LSSTCam' AND exposure.day_obs = d",
        bind={"d": args.day_obs})
        if (r.observation_type or "").lower() in wanted}
    nRaw = len(visits)
    visits = [v for v in visits if v in sci]
    print(f"{args.day_obs}: {len(visits)}/{nRaw} visits are {sorted(wanted)} "
          f"(dropped {nRaw - len(visits)} acq/eng).")
    if args.limit > 0:
        visits = visits[:args.limit]
        print(f"{args.day_obs}: limited to first {len(visits)} science visits (--limit).")
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
                   "adjacent_ccd": ccd, "rot_deg": rot_deg, "n_stars": int(len(sub)),
                   "pixscale_ccd": SCIENCE_PIXSCALE}
            if len(sub):
                # Measure 2nd AND 3rd moments with the SAME galsim-HSM estimator
                # used on the guider coadd -- matched algorithm, no weighting
                # confound. Q1/Q2/T in arcsec**2, coma/trefoil in arcsec**3
                # (scale=SCIENCE_PIXSCALE). Median + RMS over the CCD's stars.
                try:
                    exp = butler.get("preliminary_visit_image", collections=collection,
                                     instrument="LSSTCam", visit=visit, detector=ccd)
                    img = exp.image.array
                    ny, nx = img.shape
                    acc = {k: [] for k in ("Q1", "Q2", "T", "e1", "e2",
                                           "coma1", "coma2", "tref1", "tref2")}
                    for _, r in sub.sort_values("psfFlux", ascending=False).head(args.max_per_ccd).iterrows():
                        xi, yi = int(round(r.x)), int(round(r.y))
                        if xi < HALF or yi < HALF or xi >= nx - HALF or yi >= ny - HALF:
                            continue
                        stamp = np.ascontiguousarray(
                            img[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1].astype(float))
                        mom, _ = measure_hsm_moments(
                            galsim.Image(stamp, scale=SCIENCE_PIXSCALE),
                            third_order=not args.no_third, fourth_order=False, radial=False)
                        if mom is None:
                            continue
                        acc["Q1"].append(mom["M20"]); acc["Q2"].append(mom["M02"])
                        acc["T"].append(mom["M11"])
                        acc["e1"].append(mom["e1n"]); acc["e2"].append(mom["e2n"])
                        if not args.no_third:
                            acc["coma1"].append(mom["M21"]); acc["coma2"].append(mom["M12"])
                            acc["tref1"].append(mom["M30"]); acc["tref2"].append(mom["M03"])
                    row["n_hsm"] = len(acc["T"])
                    for base, arr in acc.items():
                        if arr:
                            med, rms = medRms(arr)
                            row[f"{base}_ccd"] = med
                            row[f"{base}_ccd_rms"] = rms
                except Exception as exc:  # noqa: BLE001
                    print(f"  visit {visit} ccd {ccd}: HSM moments failed: {exc}", file=sys.stderr)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"guider_psfmoments_{args.day_obs}.parquet")
    df.to_parquet(out, index=False)
    print(f"{args.day_obs}: wrote {len(df)} rows ({df.expId.nunique()} visits) -> {out}")


if __name__ == "__main__":
    main()
