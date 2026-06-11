#!/usr/bin/env python3
"""Concatenate parquet files into one, streaming row-group by row-group.

Used by the Snakemake pipeline to combine per-chunk donut / fits / visits
tables into a single param_set-level table.

    python combine_parquets.py chunkA.parquet chunkB.parquet --output combined.parquet

Streams one row group at a time (constant memory ~ a single row group), so it
combines the large per-donut tables without loading everything into RAM —
important on the 16 GiB RSP allocation.  Inputs are expected to share a schema
(same run_mktable) and be disjoint (different date chunks), so this is a
straight append; no row de-duplication.
"""
import argparse
import os
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', help='Input parquet files to concatenate')
    ap.add_argument('--output', required=True, help='Output combined parquet')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = None
    total_rows = 0
    try:
        for p in args.inputs:
            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                tbl = pf.read_row_group(i)
                if writer is None:
                    writer = pq.ParquetWriter(args.output, tbl.schema,
                                              compression='snappy')
                writer.write_table(tbl)
                total_rows += tbl.num_rows
            print(f'  {p}: {pf.num_row_groups} row groups, '
                  f'{pf.metadata.num_rows} rows')
    finally:
        if writer is not None:
            writer.close()

    print(f'combined {len(args.inputs)} files -> {args.output}: {total_rows} rows')


if __name__ == '__main__':
    main()
