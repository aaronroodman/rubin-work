#!/usr/bin/env python3
"""Concatenate per-exposure guider moment tables into one dataset table.

    python combine_moments.py exp1.parquet exp2.parquet ... --output combined.parquet

The per-exposure tables are small (<= 8 sensors x 4 kinds), so a plain pandas
concat is sufficient. Empty inputs (exposures with no tracked stars) are
skipped; if every input is empty an empty table with the union schema is
written so downstream steps still find their input.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="Per-exposure parquet files.")
    p.add_argument("--output", required=True, help="Combined parquet path.")
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)
    frames = [pd.read_parquet(f) for f in args.inputs]
    nonEmpty = [df for df in frames if not df.empty]
    combined = pd.concat(nonEmpty, ignore_index=True) if nonEmpty else frames[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_parquet(args.output, index=False)
    nExp = combined["expId"].nunique() if not combined.empty else 0
    print(f"combined {len(args.inputs)} files -> {args.output} "
          f"({len(combined)} rows, {nExp} exposures)")


if __name__ == "__main__":
    main()
