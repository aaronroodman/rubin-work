#!/usr/bin/env python3
"""Concatenate parquet files into one, streaming row-group by row-group.

Used by the Snakemake pipeline to combine per-chunk donut / fits / visits
tables into a single param_set-level table.

    python combine_parquets.py chunkA.parquet chunkB.parquet --output combined.parquet

Streams one row group at a time (constant memory ~ a single row group), so it
combines the large per-donut tables without loading everything into RAM —
important on the 16 GiB RSP allocation.

Inputs come from the same run_mktable but can differ slightly in schema between
chunks: a chunk may be missing optional columns (e.g. ``*_donut_id`` strings
absent for some processing dates) or carry a narrower numeric type.  We build a
**unified schema** (union of all columns, type-promoted) up front and conform
every row group to it — casting types and null-filling absent columns — so the
append succeeds.  Columns not present in every input are reported.
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


def build_unified_schema(schemas):
    """Union of fields across schemas, with permissive type promotion."""
    try:
        return pa.unify_schemas(schemas, promote_options='permissive')
    except TypeError:
        # Older pyarrow without promote_options — union compatible types only.
        return pa.unify_schemas(schemas)


def conform_table(tbl, schema):
    """Return ``tbl`` rebuilt to match ``schema``: cast types, null-fill
    missing columns, and order columns as in ``schema``."""
    cols = set(tbl.column_names)
    arrays = []
    for field in schema:
        if field.name in cols:
            col = tbl.column(field.name)
            if not col.type.equals(field.type):
                col = col.cast(field.type)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(tbl.num_rows, type=field.type))
    return pa.table(arrays, schema=schema)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+', help='Input parquet files to concatenate')
    ap.add_argument('--output', required=True, help='Output combined parquet')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Pass 1: collect schemas and build the unified target schema.
    pfs = [pq.ParquetFile(p) for p in args.inputs]
    schemas = [pf.schema_arrow for pf in pfs]
    unified = build_unified_schema(schemas)

    # Report columns not present in every input (so silent null-fill is visible).
    all_names = [set(s.names) for s in schemas]
    common = set.intersection(*all_names) if all_names else set()
    for name in unified.names:
        if name not in common:
            missing = [args.inputs[i] for i, names in enumerate(all_names)
                       if name not in names]
            print(f'  note: column {name!r} absent from {len(missing)}/'
                  f'{len(args.inputs)} input(s); null-filled there')

    # Pass 2: stream row groups, conforming each to the unified schema.
    writer = pq.ParquetWriter(args.output, unified, compression='snappy')
    total_rows = 0
    try:
        for p, pf in zip(args.inputs, pfs):
            for i in range(pf.num_row_groups):
                tbl = conform_table(pf.read_row_group(i), unified)
                writer.write_table(tbl)
                total_rows += tbl.num_rows
            print(f'  {p}: {pf.num_row_groups} row groups, '
                  f'{pf.metadata.num_rows} rows')
    finally:
        writer.close()

    print(f'combined {len(args.inputs)} files -> {args.output}: {total_rows} rows')


if __name__ == '__main__':
    main()
