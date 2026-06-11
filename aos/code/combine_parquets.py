#!/usr/bin/env python3
"""Concatenate parquet files row-wise into one combined parquet.

Used by the Snakemake pipeline to combine per-chunk donut / fits / visits
tables into a single param_set-level table for the downstream steps.

    python combine_parquets.py chunkA.parquet chunkB.parquet --output combined.parquet

Chunks are expected to be disjoint (different date ranges), so this is a
straight row concat — no row de-duplication (the tables carry list/array
columns that are not hashable).  A warning is printed if the natural
(day_obs, seq_num) key has duplicates across chunks.
"""
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', help='Input parquet files to concatenate')
    ap.add_argument('--output', required=True, help='Output combined parquet')
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        df = pd.read_parquet(p)
        frames.append(df)
        print(f'  {p}: {len(df)} rows')
    combined = pd.concat(frames, ignore_index=True)

    key = [c for c in ('day_obs', 'seq_num') if c in combined.columns]
    if len(key) == len(('day_obs', 'seq_num')):
        ndup = int(combined.duplicated(subset=key).sum())
        if ndup:
            print(f'  WARNING: {ndup} duplicate (day_obs, seq_num) rows across '
                  f'chunks — overlapping date ranges?')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_parquet(args.output)
    print(f'combined {len(args.inputs)} files -> {args.output}: '
          f'{len(combined)} rows x {len(combined.columns)} cols')


if __name__ == '__main__':
    main()
