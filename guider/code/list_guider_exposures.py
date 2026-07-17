#!/usr/bin/env python3
"""List the guide-sensor exposures for a night from the Butler.

Queries the registry for ``guider_raw`` datasets on the given dayObs and writes
the distinct seqNums (one per line) to --output (or stdout). Used by the
Snakemake `discover_night` checkpoint to fan out the per-exposure jobs.

    python list_guider_exposures.py --day-obs 20260709 \
        --repo main --collections LSSTCam/raw/guider LSSTCam/raw/all \
        --output output/night_20260709/exposures.txt
"""
from __future__ import annotations

import argparse
import os
import sys


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True, help="Observation date, YYYYMMDD.")
    p.add_argument("--repo", default="main", help="Butler repository.")
    p.add_argument(
        "--collections",
        nargs="+",
        default=["LSSTCam/raw/guider", "LSSTCam/raw/all"],
        help="Butler collections.",
    )
    p.add_argument("--instrument", default="LSSTCam")
    p.add_argument(
        "--image-types",
        nargs="+",
        default=["science"],
        help="Keep only these exposure observation_types (default: science).",
    )
    p.add_argument("--output", default=None, help="Output file (default: stdout).")
    p.add_argument("--limit", type=int, default=0, help="Keep only the first N seqNums (0 = all).")
    return p.parse_args(argv)


def discoverSeqNums(repo, collections, dayObs, instrument, imageTypes):
    """Return the sorted seqNums with guider_raw data on dayObs.

    Filtered to the given exposure ``observation_type`` values (case-insensitive).
    ``imageTypes`` may be None/empty to keep all types.
    """
    from lsst.daf.butler import Butler

    butler = Butler(repo, collections=collections)
    records = butler.registry.queryDimensionRecords(
        "exposure",
        datasets="guider_raw",
        collections=collections,
        where="instrument = inst AND exposure.day_obs = dayObs",
        bind={"inst": instrument, "dayObs": dayObs},
    )
    wanted = {t.lower() for t in imageTypes} if imageTypes else None
    seqNums = {
        int(r.seq_num)
        for r in records
        if wanted is None or (r.observation_type or "").lower() in wanted
    }
    return sorted(seqNums)


def main(argv=None):
    args = parseArgs(argv)
    seqNums = discoverSeqNums(
        args.repo, args.collections, args.day_obs, args.instrument, args.image_types
    )
    if args.limit > 0:
        seqNums = seqNums[: args.limit]

    text = "\n".join(str(s) for s in seqNums)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write(text + ("\n" if text else ""))
    else:
        print(text)
    print(f"{args.day_obs}: {len(seqNums)} guider exposures "
          f"(imagetypes={args.image_types})", file=sys.stderr)


if __name__ == "__main__":
    main()
