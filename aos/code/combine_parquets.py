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
absent for some processing dates) or carry a narrower numeric type (e.g.
``zk_CCS`` as ``list<float>`` in one chunk, ``list<double>`` in another).  We
build a **unified schema** (union of all columns, recursively type-promoted)
up front and conform every row group to it — casting types and null-filling
absent columns — so the append succeeds.  Columns not present in every input
are reported; genuinely incompatible types raise a clear, column-named error.
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat


def promote_type(a, b):
    """Common type two columns can both be cast to (recurses into lists)."""
    if a.equals(b):
        return a
    if pat.is_null(a):
        return b
    if pat.is_null(b):
        return a
    if pat.is_list(a) and pat.is_list(b):
        return pa.list_(promote_type(a.value_type, b.value_type))
    if pat.is_large_list(a) and pat.is_large_list(b):
        return pa.large_list(promote_type(a.value_type, b.value_type))
    if pat.is_floating(a) and pat.is_floating(b):
        return a if a.bit_width >= b.bit_width else b
    if pat.is_integer(a) and pat.is_integer(b):
        return pa.int64()
    if (pat.is_integer(a) or pat.is_floating(a)) and \
       (pat.is_integer(b) or pat.is_floating(b)):
        return pa.float64()
    raise TypeError(f'incompatible types {a} vs {b}')


def build_unified_schema(schemas):
    """Union of fields across schemas (first-seen order), type-promoted."""
    types, order = {}, []
    for sch in schemas:
        for f in sch:
            if f.name not in types:
                types[f.name] = f.type
                order.append(f.name)
            elif not types[f.name].equals(f.type):
                try:
                    types[f.name] = promote_type(types[f.name], f.type)
                except TypeError as e:
                    raise TypeError(f'column {f.name!r}: {e}')
    return pa.schema([pa.field(n, types[n]) for n in order])


def conform_table(tbl, schema):
    """Rebuild ``tbl`` to match ``schema``: cast types, null-fill missing
    columns, order columns as in ``schema``."""
    cols = set(tbl.column_names)
    data = {}
    for field in schema:
        if field.name in cols:
            col = tbl.column(field.name)
            if not col.type.equals(field.type):
                col = col.cast(field.type)
            data[field.name] = col
        else:
            data[field.name] = pa.nulls(tbl.num_rows, type=field.type)
    return pa.table(data, schema=schema)


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

    # Report columns not present in every input (so silent null-fill is visible)
    # and any column whose type had to be promoted.
    all_names = [set(s.names) for s in schemas]
    common = set.intersection(*all_names) if all_names else set()
    for field in unified:
        if field.name not in common:
            missing = [args.inputs[i] for i, names in enumerate(all_names)
                       if field.name not in names]
            print(f'  note: column {field.name!r} absent from {len(missing)}/'
                  f'{len(args.inputs)} input(s); null-filled there')
        else:
            srctypes = {s.field(field.name).type for s in schemas}
            if len(srctypes) > 1:
                print(f'  note: column {field.name!r} promoted to {field.type} '
                      f'from {sorted(str(t) for t in srctypes)}')

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
