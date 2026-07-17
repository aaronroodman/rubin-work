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
    frames = [pd.read_parquet(f) for f in args.inputs]
    nonEmpty = [df for df in frames if not df.empty]

    os.makedirs(args.output, exist_ok=True)
    if not nonEmpty:
        print(f"no non-empty inputs; empty dataset at {args.output}")
        return

    df = pd.concat(nonEmpty, ignore_index=True)
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
