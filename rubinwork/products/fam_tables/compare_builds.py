#!/usr/bin/env python3
"""Compare two ``fam_tables`` builds table by table: row counts, column sets and values.

The regression check for the phase 2 code move (``notes/status/organization_plan.md``):
build one chunk before the move and one after, then compare. Values are expected to match
exactly.

Rows must be aligned on a key before values can be compared at all. ``donuts.parquet`` has
one row per donut and its row *order* is not stable between runs, so comparing by position
reports almost every column as different. The key used here is
``day_obs, seq_num, detector, extra_donut_id``, verified unique on both sides; the
per-visit tables key on ``day_obs, seq_num``.

Usage::

    python -m rubinwork.products.fam_tables.compare_builds <dir-a> <dir-b>
    python -m rubinwork.products.fam_tables.compare_builds <dir-a> <dir-b> --tables visits fits
"""

import argparse
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

TABLE_KEYS = {
    "donuts": ["day_obs", "seq_num", "detector", "extra_donut_id"],
    "visits": ["day_obs", "seq_num"],
    "fits": ["day_obs", "seq_num"],
}
"""Row key per table: the columns that make a row unique, for alignment."""


def _column_values(series):
    """Values of one column as a comparable array.

    Returns
    -------
    values : `numpy.ndarray`
        The column stacked into an array. A column of per-row arrays (the Zernike
        vectors, in microns of wavefront) is stacked into a 2-D array.
    """
    if series.dtype == object and len(series) and isinstance(series.iloc[0], np.ndarray):
        return np.stack(series.values)
    return series.values


def compare_table(path_a, path_b, key, rtol=0.0, atol=0.0):
    """Compare one table between two builds.

    Parameters
    ----------
    path_a, path_b : `pathlib.Path`
        The two parquet files.
    key : `list` [`str`]
        Columns to sort on, so rows line up.
    rtol, atol : `float`, optional
        Tolerances passed to `numpy.allclose`. Both zero by default: an exact match
        is what the code move is expected to produce.

    Returns
    -------
    report : `dict`
        ``rows_a``, ``rows_b``, ``only_a``, ``only_b`` (column names), ``differing``
        (list of ``(column, max_abs_difference)``; the difference is `None` for a
        non-numeric column) and ``compared`` (count of columns compared, dimensionless).
    """
    import pandas as pd  # noqa: F401  (pandas is what to_pandas returns into)

    ta, tb = pq.read_table(path_a), pq.read_table(path_b)
    da, db = ta.to_pandas(), tb.to_pandas()
    rep = {"rows_a": len(da), "rows_b": len(db),
           "only_a": sorted(set(da.columns) - set(db.columns)),
           "only_b": sorted(set(db.columns) - set(da.columns)),
           "differing": [], "compared": 0, "key_aligned": None}

    usable = [k for k in key if k in da.columns and k in db.columns]
    if usable:
        da = da.sort_values(usable).reset_index(drop=True)
        db = db.sort_values(usable).reset_index(drop=True)
    if len(da) != len(db):
        return rep
    rep["key_aligned"] = bool(usable) and all(
        (da[k].values == db[k].values).all() for k in usable)

    for col in sorted(set(da.columns) & set(db.columns)):
        xa, xb = da[col], db[col]
        rep["compared"] += 1
        try:
            va, vb = _column_values(xa), _column_values(xb)
            if va.dtype.kind in "fc" and vb.dtype.kind in "fc":
                if not np.allclose(va, vb, rtol=rtol, atol=atol, equal_nan=True):
                    rep["differing"].append((col, float(np.nanmax(np.abs(va - vb)))))
            elif not xa.equals(xb):
                rep["differing"].append((col, None))
        except (ValueError, TypeError):
            # Ragged or non-stackable column: fall back to pandas equality.
            if not xa.equals(xb):
                rep["differing"].append((col, None))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", help="first build directory (the reference)")
    ap.add_argument("dir_b", help="second build directory")
    ap.add_argument("--tables", nargs="+", default=list(TABLE_KEYS),
                    help=f"tables to compare (default: {' '.join(TABLE_KEYS)})")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args(argv)

    a, b = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)
    failed = False
    for table in args.tables:
        pa, pb = a / f"{table}.parquet", b / f"{table}.parquet"
        print(f"--- {table}")
        if not pa.exists() or not pb.exists():
            print(f"    MISSING: {pa if not pa.exists() else pb}")
            failed = True
            continue
        rep = compare_table(pa, pb, TABLE_KEYS.get(table, []),
                            rtol=args.rtol, atol=args.atol)
        print(f"    rows: a={rep['rows_a']:,} b={rep['rows_b']:,}"
              f"{'  MISMATCH' if rep['rows_a'] != rep['rows_b'] else ''}")
        print(f"    columns compared: {rep['compared']} (count, dimensionless)"
              f" | only in a: {len(rep['only_a'])} | only in b: {len(rep['only_b'])}")
        if rep["only_a"]:
            print(f"      only in a: {rep['only_a'][:8]}"
                  f"{' ...' if len(rep['only_a']) > 8 else ''}")
        if rep["only_b"]:
            print(f"      only in b: {rep['only_b'][:8]}"
                  f"{' ...' if len(rep['only_b']) > 8 else ''}")
        if rep["key_aligned"] is False:
            print("    WARNING: key columns do not align, so value diffs are unreliable")
        if rep["rows_a"] != rep["rows_b"] or rep["differing"]:
            failed = True
        if rep["differing"]:
            print(f"    differing: {len(rep['differing'])} columns")
            for col, mx in rep["differing"]:
                print(f"      {col}: max abs difference "
                      f"{'(non-numeric)' if mx is None else f'{mx:.6g}'}")
        else:
            print("    values: identical")
    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
