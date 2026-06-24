#!/usr/bin/env python3
"""CLI: build the nightly AOS table for one night and write it to parquet.

SUMMIT-only stage of the OPR pipeline (needs ConsDB + EFD + embargo Butler).

    python run_nightly_table.py --day-obs 20260329 --out output/20260329/nightly_aos_table.parquet

Thin wrapper around ``nightly_table.build_nightly_table`` (ported from the
canonical ``nightly_report_ts_version.ipynb``).
"""

import argparse
import asyncio
import os
import sys

from nightly_table import build_nightly_table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day-obs", type=int, required=True, help="Observation day, YYYYMMDD")
    ap.add_argument("--seq-min", type=int, default=0)
    ap.add_argument("--seq-max", type=int, default=9999)
    ap.add_argument("--out", required=True, help="Output parquet path")
    args = ap.parse_args()

    table = asyncio.run(
        build_nightly_table(args.day_obs, seq_min=args.seq_min, seq_max=args.seq_max)
    )

    if table is None or len(table) == 0:
        print(f"ERROR: empty table for day_obs={args.day_obs}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # build_nightly_table indexes by seq; keep it as a column for downstream.
    table = table.reset_index()
    table.to_parquet(args.out)
    print(f"Wrote {args.out}: {len(table)} rows, {len(table.columns)} cols")


if __name__ == "__main__":
    main()
