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
    if args.fbs_substr:
        sci = df[df["img_type"] == "science"].copy()
        match = sci[sci["science_program"].str.contains(args.fbs_substr, na=False, regex=False)]
        counts = match.groupby("day_obs").size()
        nights = sorted(int(d) for d, n in counts.items() if n >= args.min_exp)
        print(f"\n# {len(nights)} nights after {args.after} with >= {args.min_exp} "
              f"science exposures in a program containing {args.fbs_substr!r}:")
        print("nights:")
        for n in nights:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
