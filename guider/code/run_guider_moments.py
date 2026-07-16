#!/usr/bin/env python3
"""Per-exposure guider second-moment decomposition (Snakemake runner).

Reads one guider exposure, tracks its guide stars, decomposes each sensor's
shape into coadd / static-per-stamp / image-motion second moments, and writes
a tidy per-exposure parquet table (see `guiderMoments.momentsToDataFrame`).

    python run_guider_moments.py --day-obs 20260709 --seq-num 808 \
        --output output/<dataset>/exposures/2026070900808/moments.parquet

The output has one row per (detector, kind) with kind in
{coadd, stamp_mean, motion, stamp+motion} and moments Mxx, Myy, Mxy, Q1, Q2, T
in arcsec**2, plus dayObs / seqNum / expId columns for downstream aggregation.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# guiderMoments / guiderEdgeRecovery live alongside this runner in code/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guiderEdgeRecovery import makeTrackerConfig  # noqa: E402
from guiderMoments import MomentConfig, decomposeExposure, momentsToDataFrame  # noqa: E402


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True, help="Observation date, YYYYMMDD.")
    p.add_argument("--seq-num", type=int, required=True, help="Sequence number in the night.")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.add_argument("--repo", default="main", help="Butler repository.")
    p.add_argument(
        "--collections",
        nargs="+",
        default=["LSSTCam/raw/guider", "LSSTCam/raw/all"],
        help="Butler collections.",
    )
    p.add_argument("--view", default="dvcs", help="Guider readout view.")
    # tracker
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=3)
    p.add_argument("--min-finite-fraction", type=float, default=0.5)
    p.add_argument(
        "--no-recover-edge-stars",
        action="store_true",
        help="Disable near-edge star recovery (use stock reject-any-NaN).",
    )
    # moments
    p.add_argument("--ap-nsigma", type=float, default=4.0)
    p.add_argument("--cen-niter", type=int, default=3)
    return p.parse_args(argv)


def emptyTable(dayObs, seqNum, expId):
    """Return a valid, empty per-exposure table (no stars tracked)."""
    cols = [
        "expId", "detector", "kind", "xfp", "yfp", "nStamps",
        "Mxx", "Myy", "Mxy", "Q1", "Q2", "T", "dayObs", "seqNum",
    ]
    return pd.DataFrame(columns=cols)


def main(argv=None):
    args = parseArgs(argv)
    expId = args.day_obs * 100000 + args.seq_num

    # Imports of the LSST stack are deferred so --help works without it.
    from lsst.daf.butler import Butler
    from lsst.summit.utils.guiders.reading import GuiderReader
    from lsst.summit.utils.guiders.tracking import GuiderStarTracker

    butler = Butler(args.repo, collections=args.collections)
    reader = GuiderReader(butler, view=args.view)
    guiderData = reader.get(dayObs=args.day_obs, seqNum=args.seq_num, doSubtractMedian=True)

    recover = not args.no_recover_edge_stars
    config = makeTrackerConfig(
        minFiniteFraction=args.min_finite_fraction if recover else 1.0,
        minSnr=args.min_snr,
        maxEllipticity=args.max_ellipticity,
        edgeMargin=args.edge_margin,
    )
    stars = GuiderStarTracker(guiderData, config).trackGuiderStars(refCatalog=None)

    if stars.empty:
        table = emptyTable(args.day_obs, args.seq_num, expId)
    else:
        momentConfig = MomentConfig(apNSigma=args.ap_nsigma, cenNIter=args.cen_niter)
        decomps = decomposeExposure(guiderData, stars, momentConfig)
        table = momentsToDataFrame(decomps)
        table["dayObs"] = args.day_obs
        table["seqNum"] = args.seq_num

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_parquet(args.output, index=False)
    print(f"{expId}: wrote {len(table)} rows for "
          f"{table['detector'].nunique() if not table.empty else 0} sensors -> {args.output}")


if __name__ == "__main__":
    main()
