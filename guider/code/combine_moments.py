#!/usr/bin/env python3
"""Combine per-exposure guider summary tables into a partitioned dataset.

    python combine_moments.py exp1.parquet exp2.parquet ... --output moments/

Writes a Hive-partitioned parquet dataset under ``--output``, partitioned by
``dayObs`` (``moments/dayObs=YYYYMMDD/part-*.parquet``), so downstream reads can
prune by night:

    import pyarrow.dataset as pds
    df = pds.dataset("moments/", partitioning="hive").to_table(
        filter=(pds.field("dayObs") >= 20260701)).to_pandas()

Empty inputs (exposures with no tracked stars) are skipped.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="Per-exposure parquet files.")
    p.add_argument("--output", required=True, help="Output dataset directory.")
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)

    frames = []
    missing = []   # visits skipped for missing/insufficient data (have a marker)
    nostars = []   # visits with empty output but no marker (data present, no stars)
    for f in args.inputs:
        d = pd.read_parquet(f)
        if not d.empty:
            frames.append(d)
            continue
        eid = os.path.basename(os.path.dirname(f))
        marker = os.path.join(os.path.dirname(f), "SKIPPED.txt")
        if os.path.exists(marker):
            missing.append(open(marker).read().strip())
        else:
            nostars.append(eid)

    # Flag skipped visits in one place: to stdout (the combine log) and a report.
    lines = []
    if missing:
        lines.append(f"# {len(missing)} visit(s) SKIPPED -- missing/insufficient data:")
        lines += [f"  {m}" for m in missing]
    if nostars:
        lines.append(f"# {len(nostars)} visit(s) with no tracked stars: {', '.join(sorted(nostars))}")
    report = "\n".join(lines) if lines else "no skipped visits"
    print(report)
    parent = os.path.dirname(os.path.abspath(args.output.rstrip("/")))
    os.makedirs(parent, exist_ok=True)
    with open(os.path.join(parent, "skipped_visits.txt"), "w") as fh:
        fh.write(report + "\n")

    os.makedirs(args.output, exist_ok=True)
    if not frames:
        print(f"no non-empty inputs; empty dataset at {args.output}")
        return

    df = pd.concat(frames, ignore_index=True)
    df["dayObs"] = df["dayObs"].astype("int64")
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Partition by dayObs (Hive layout); the column is stripped from the files
    # and encoded in the folder names. delete_matching clears stale partitions.
    pds.write_dataset(
        table,
        base_dir=args.output,
        format="parquet",
        partitioning=pds.partitioning(pa.schema([("dayObs", pa.int64())]), flavor="hive"),
        existing_data_behavior="delete_matching",
    )
    print(f"wrote partitioned dataset -> {args.output} "
          f"({len(df)} rows, {df['dayObs'].nunique()} nights, {df['expId'].nunique()} exposures)")


if __name__ == "__main__":
    main()
