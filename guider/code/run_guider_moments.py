#!/usr/bin/env python3
"""Per-exposure guider products (Snakemake runner).

Reads one guider exposure, tracks its guide stars, and writes three parquets
into --outdir, named by seqNum:

  <seqNum>_moments.parquet   NEW analysis: per-(detector) unweighted moment
                             decomposition (coadd/stamp/motion, 2nd + 3rd order
                             static/dynamic/cross, temporal covariances, PSDs).
  <seqNum>_stars.parquet     summit_utils GuiderStarTracker per-stamp table.
  <seqNum>_metrics.parquet   summit_utils GuiderMetricsBuilder per-exposure row.

Visits with missing/insufficient guider data are skipped (empty tables + a
SKIPPED.txt marker) so the batch and per-day combine still complete.

    python run_guider_moments.py --day-obs 20260709 --seq-num 808 --outdir <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guiderEdgeRecovery import makeTrackerConfig  # noqa: E402
from guiderMoments import (  # noqa: E402
    MomentConfig,
    SCHEMA_VERSION,
    decomposeExposure,
    summarizeExposure,
)


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, required=True)
    p.add_argument("--outdir", required=True, help="Output dir (per-seq files written here).")
    p.add_argument("--repo", default="main")
    p.add_argument("--collections", nargs="+", default=["LSSTCam/raw/guider", "LSSTCam/raw/all"])
    p.add_argument("--view", default="dvcs")
    p.add_argument("--guider-hz", type=float, default=5.0)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=3)
    p.add_argument("--min-finite-fraction", type=float, default=0.5)
    p.add_argument("--no-recover-edge-stars", action="store_true")
    p.add_argument("--ap-nsigma", type=float, default=4.0)
    p.add_argument("--cen-niter", type=int, default=3)
    p.add_argument("--strict", action="store_true",
                   help="Fail on missing/insufficient data instead of skipping.")
    return p.parse_args(argv)


def summitUtilsVersion() -> str:
    try:
        import lsst.summit.utils as su
        return str(getattr(su, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def paths(outdir, seqNum):
    return {k: os.path.join(outdir, f"{seqNum}_{k}.parquet")
            for k in ("moments", "stars", "metrics")}


def writeEmpty(outdir, seqNum, dayObs, reason):
    """Write empty per-seq tables + a SKIPPED marker so combine can flag it."""
    os.makedirs(outdir, exist_ok=True)
    for p in paths(outdir, seqNum).values():
        pd.DataFrame(columns=["expId", "dayObs", "seqNum", "detector"]).to_parquet(p, index=False)
    with open(os.path.join(outdir, f"{seqNum}_SKIPPED.txt"), "w") as fh:
        fh.write(f"{dayObs * 100000 + seqNum}\t{dayObs}\t{seqNum}\t{reason}\n")


def main(argv=None):
    args = parseArgs(argv)
    expId = args.day_obs * 100000 + args.seq_num
    out = paths(args.outdir, args.seq_num)

    from lsst.daf.butler import Butler
    from lsst.summit.utils.guiders.reading import GuiderReader
    from lsst.summit.utils.guiders.tracking import GuiderStarTracker
    from lsst.summit.utils.guiders.metrics import GuiderMetricsBuilder
    try:
        from lsst.daf.butler import DatasetNotFoundError
    except ImportError:
        from lsst.daf.butler._exceptions import DatasetNotFoundError

    butler = Butler(args.repo, collections=args.collections)
    reader = GuiderReader(butler, view=args.view)
    try:
        guiderData = reader.get(dayObs=args.day_obs, seqNum=args.seq_num, doSubtractMedian=True)
    except (DatasetNotFoundError, RuntimeError) as exc:
        if args.strict:
            raise
        print(f"{expId}: SKIPPED -- {exc}", file=sys.stderr)
        writeEmpty(args.outdir, args.seq_num, args.day_obs, str(exc))
        return

    recover = not args.no_recover_edge_stars
    config = makeTrackerConfig(
        minFiniteFraction=args.min_finite_fraction if recover else 1.0,
        minSnr=args.min_snr, maxEllipticity=args.max_ellipticity, edgeMargin=args.edge_margin,
    )
    stars = GuiderStarTracker(guiderData, config).trackGuiderStars(refCatalog=None)

    os.makedirs(args.outdir, exist_ok=True)
    stale = os.path.join(args.outdir, f"{args.seq_num}_SKIPPED.txt")
    if os.path.exists(stale):
        os.remove(stale)

    if stars.empty:
        for p in out.values():
            pd.DataFrame(columns=["expId", "dayObs", "seqNum", "detector"]).to_parquet(p, index=False)
        print(f"{expId}: no tracked stars; wrote empty tables")
        return

    # (1) summit_utils per-stamp tracker table (Jackie)
    starsOut = stars.copy()
    starsOut["dayObs"] = args.day_obs
    starsOut["seqNum"] = args.seq_num
    starsOut.to_parquet(out["stars"], index=False)

    # (2) summit_utils per-exposure metrics (Jackie)
    try:
        metrics = GuiderMetricsBuilder(stars, nMissingStamps=0).buildMetrics(expId)
        metrics = metrics.copy()
        metrics["expId"] = expId
        metrics["dayObs"] = args.day_obs
        metrics["seqNum"] = args.seq_num
    except Exception as exc:  # noqa: BLE001
        print(f"{expId}: GuiderMetricsBuilder failed: {exc}", file=sys.stderr)
        metrics = pd.DataFrame([{"expId": expId, "dayObs": args.day_obs, "seqNum": args.seq_num}])
    metrics.to_parquet(out["metrics"], index=False)

    # (3) NEW moment decomposition
    provenance = {
        "summit_utils_version": summitUtilsVersion(), "schema_version": SCHEMA_VERSION,
        "ap_nsigma": args.ap_nsigma, "cen_niter": args.cen_niter, "min_snr": args.min_snr,
        "edge_margin": args.edge_margin, "max_ellipticity": args.max_ellipticity,
        "min_finite_fraction": args.min_finite_fraction if recover else 1.0,
    }
    decomps = decomposeExposure(guiderData, stars, MomentConfig(apNSigma=args.ap_nsigma,
                                                                cenNIter=args.cen_niter))
    summary = summarizeExposure(guiderData, stars, decomps,
                                guiderHz=args.guider_hz, provenance=provenance)
    summary.to_parquet(out["moments"], index=False)
    print(f"{expId}: wrote moments({len(summary)}) stars({len(starsOut)}) metrics({len(metrics)})")


if __name__ == "__main__":
    main()
