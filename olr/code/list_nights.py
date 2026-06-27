#!/usr/bin/env python3
"""List observing nights from ConsDB to choose which to run the OLR pipeline over.

SUMMIT-only (queries the summit ConsDB).  Prints, per night after a cutoff, the
`science_program` (BLOCK) values present and their exposure counts, so you can
identify the nights with the Feature-Based Scheduler (FBS / survey) running.

    # 1. see the per-night breakdown of science programs:
    python list_nights.py --after 20260417

    # 2. once you know the FBS/survey block substring, emit a config nights: block
    #    (a night qualifies if it has >= --min-exp science exposures in a matching
    #     program):
    python list_nights.py --after 20260417 --fbs-substr BLOCK-365 --min-exp 1

FBS survey visits are img_type='science' under the survey block; engineering /
calibration / AOS blocks (BLOCK-T..., calibrations) are separate.  Use the
breakdown from step 1 to pick the right --fbs-substr.
"""

import argparse
import os

import pandas as pd

from lsst.summit.utils import ConsDbClient

# ConsDB lives on an internal host; bypass any HTTP proxy for it.
if "no_proxy" in os.environ:
    os.environ["no_proxy"] += ",.consdb"
else:
    os.environ["no_proxy"] = ".consdb"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after", type=int, default=20260417,
                    help="day_obs cutoff, exclusive (default 20260417 -> nights from 0418)")
    ap.add_argument("--before", type=int, default=None,
                    help="optional upper day_obs cutoff, inclusive")
    ap.add_argument("--consdb-url", default="http://consdb-pq.consdb:8080/consdb")
    ap.add_argument("--fbs-substr", default=None,
                    help="science_program substring identifying FBS/survey; "
                         "if set, emit a config nights: YAML block")
    ap.add_argument("--min-exp", type=int, default=1,
                    help="min matching science exposures for a night to qualify (default 1)")
    ap.add_argument("--require-aos", action="store_true",
                    help="keep only nights that actually have aggregateAOSVisitTableAvg "
                         "datasets in the embargo Butler (the data OLR needs); drops nights "
                         "where exposures were taken but AOS/WEP processing is missing")
    args = ap.parse_args()

    where = f"e.day_obs > {args.after}"
    if args.before is not None:
        where += f" AND e.day_obs <= {args.before}"
    query = f"""
        SELECT e.day_obs AS day_obs,
               e.science_program AS science_program,
               e.img_type AS img_type
        FROM cdb_lsstcam.exposure AS e
        WHERE {where}
    """
    df = ConsDbClient(args.consdb_url).query(query).to_pandas()
    if len(df) == 0:
        print(f"No exposures found after day_obs {args.after}.")
        return

    df["science_program"] = df["science_program"].fillna("(none)")

    # --- per-night breakdown of science programs (all img_types) ---
    print(f"{'day_obs':>9} | science_program: n_exposures")
    print("-" * 70)
    for day, g in df.groupby("day_obs"):
        progs = g.groupby("science_program").size().sort_values(ascending=False)
        summary = "; ".join(f"{p}:{int(n)}" for p, n in progs.items())
        print(f"{int(day):>9} | {summary}")

    # --- distinct science programs (handy for picking --fbs-substr) ---
    print("\nDistinct science_program values in range:")
    for p in sorted(df["science_program"].unique()):
        print(f"  {p}")

    # --- optional: emit the nights: YAML for matching nights ---
    if args.fbs_substr or args.require_aos:
        if args.fbs_substr:
            sci = df[df["img_type"] == "science"].copy()
            match = sci[sci["science_program"].str.contains(args.fbs_substr, na=False, regex=False)]
            counts = match.groupby("day_obs").size()
            nights = sorted(int(d) for d, n in counts.items() if n >= args.min_exp)
            label = (f">= {args.min_exp} science exposures in a program containing "
                     f"{args.fbs_substr!r}")
        else:
            nights = sorted(int(d) for d in df["day_obs"].unique())
            label = "all nights in range"

        if args.require_aos:
            nights, dropped = filter_nights_with_aos(nights)
            for d in dropped:
                print(f"  note: {d} dropped — no aggregateAOSVisitTableAvg datasets")
            label += ", with AOS products"

        print(f"\n# {len(nights)} nights after {args.after} ({label}):")
        print("nights:")
        for n in nights:
            print(f"  - {n}")


def filter_nights_with_aos(nights):
    """Keep only nights with aggregateAOSVisitTableAvg in the embargo Butler.

    The exposure existing in ConsDB does not guarantee AOS/WEP products exist
    (those can be absent or unprocessed); OLR needs the wavefront datasets.
    Returns (kept, dropped).
    """
    import lsst.summit.utils.butlerUtils as butlerUtils

    butler = butlerUtils.makeDefaultButler("LSSTCam", embargo=True)
    kept, dropped = [], []
    for d in nights:
        try:
            refs = list(butler.query_datasets("aggregateAOSVisitTableAvg",
                                               where=f"visit.day_obs = {d}"))
            (kept if refs else dropped).append(d)
        except Exception:
            # EmptyQueryResultError (or any resolution failure) -> no usable AOS data
            dropped.append(d)
    return kept, dropped


if __name__ == "__main__":
    main()
