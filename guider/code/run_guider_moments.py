#!/usr/bin/env python3
"""Per-exposure guider moment + summary row (Snakemake runner).

Reads one guider exposure, tracks its guide stars, decomposes each sensor's
shape into coadd / static-per-stamp / image-motion second moments, and writes
a tidy per-(expId, detector) summary parquet: the moment decomposition, the
temporal (Q1,Q2,T) covariance, binned PSDs of the centroid and shape series,
integral timescales, per-stamp HSM shape summaries, and the exposure-level
GuiderMetricsBuilder metrics for comparison. See guiderMoments.summarizeExposure.

    python run_guider_moments.py --day-obs 20260709 --seq-num 808 \
        --output output/<dataset>/exposures/2026070900808/moments.parquet
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# guiderMoments / guiderEdgeRecovery live alongside this runner in code/.
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
    p.add_argument("--guider-hz", type=float, default=5.0, help="Guider readout rate (Hz).")
    # tracker
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=3)
    p.add_argument("--min-finite-fraction", type=float, default=0.5)
    p.add_argument("--no-recover-edge-stars", action="store_true")
    # moments
    p.add_argument("--ap-nsigma", type=float, default=4.0)
    p.add_argument("--cen-niter", type=int, default=3)
    return p.parse_args(argv)


def summitUtilsVersion() -> str:
    """Best-effort summit_utils version string for provenance."""
    try:
        import lsst.summit.utils as su

        return str(getattr(su, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def emptyTable() -> pd.DataFrame:
    """A 0-row table (no stars tracked) that still carries the partition key."""
    return pd.DataFrame(columns=["expId", "dayObs", "seqNum", "detector"])


def main(argv=None):
    args = parseArgs(argv)
    expId = args.day_obs * 100000 + args.seq_num

    # Deferred so --help works without the stack.
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
        table = emptyTable()
    else:
        momentConfig = MomentConfig(apNSigma=args.ap_nsigma, cenNIter=args.cen_niter)
        decomps = decomposeExposure(guiderData, stars, momentConfig)
        provenance = {
            "summit_utils_version": summitUtilsVersion(),
            "schema_version": SCHEMA_VERSION,
            "ap_nsigma": args.ap_nsigma,
            "cen_niter": args.cen_niter,
            "min_snr": args.min_snr,
            "edge_margin": args.edge_margin,
            "max_ellipticity": args.max_ellipticity,
            "min_finite_fraction": args.min_finite_fraction if recover else 1.0,
        }
        table = summarizeExposure(
            guiderData, stars, decomps, guiderHz=args.guider_hz, provenance=provenance
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_parquet(args.output, index=False)
    nDet = table["detector"].nunique() if not table.empty else 0
    print(f"{expId}: wrote {len(table)} rows for {nDet} sensors -> {args.output}")


if __name__ == "__main__":
    main()
