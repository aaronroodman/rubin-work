#!/usr/bin/env python3
"""Compare two ``value_added`` databases table by table, at zero tolerance.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build a small reference with the unmoved code and the same one after, then compare. Values
are expected to match exactly everywhere **except** the build-time timestamp columns listed
in `EXPECTED_DIFF`, which two runs of identical code necessarily disagree on.

Unlike the parquet products this compares two **DuckDB** files, so the comparison is done
in SQL-loaded DataFrames rather than through
`rubinwork.products.fam_tables.compare_builds.compare_table`: a value-added table can hold
`LIST` columns (the 50 degrees of freedom, the 34 v-modes), which pandas holds as object
arrays that `!=` cannot compare elementwise.

Every table has a verified-unique key, asserted on both sides before any value is
compared, so a build that breaks uniqueness fails loudly instead of silently mis-aligning
rows.

Usage::

    python -m rubinwork.products.value_added.compare_builds <db-a> <db-b>
    python -m rubinwork.products.value_added.compare_builds <db-a> <db-b> --tables fam_dz
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

TABLE_KEYS = {
    "visit_telemetry": ("visit_id",),
    "optical_state": ("visit_id", "variant_id"),
    "m1m3_thermal_r2": ("visit_id",),
    "fam_dz": ("visit_id", "fam_variant_id"),
    "column_coverage": ("column_name",),
    "state_variant": ("variant_id",),
    "fam_variant": ("fam_variant_id",),
    "fetch_log": ("day_obs", "group_name"),
}
"""Row key per table: the columns that make a row unique, for alignment.

``optical_state`` and ``fam_dz`` are **long** tables keyed on the variant as well as the
visit, which is the whole point of their shape -- a variant is rows, not schema.
"""

EXPECTED_DIFF = {
    "fetch_log": ("attempted_at",),
    "column_coverage": ("updated_at",),
    "state_variant": ("created_at",),
    "fam_variant": ("created_at",),
    "optical_state": ("computed_at",),
    "fam_dz": ("computed_at",),
    "m1m3_thermal_r2": ("computed_at",),
}
"""Columns two runs of identical code may legitimately differ on.

Each is a wall-clock timestamp written at build time. The list was fixed in the phase 3
handoff **before** the reference build ran, so it cannot be widened after seeing a
difference. ``visit_telemetry`` carries no timestamp column at all and so appears here
with no entry: it must match exactly, column for column.
"""


def _load(db_path, table):
    """Read one whole table, or `None` when the database does not have it.

    Parameters
    ----------
    db_path : `pathlib.Path`
        A DuckDB database file.
    table : `str`
        Table name.

    Returns
    -------
    data : `pandas.DataFrame` or `None`
        The table, or `None` if it is absent.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        names = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main'").fetchall()}
        if table not in names:
            return None
        return con.execute(f"SELECT * FROM {table}").fetchdf()
    finally:
        con.close()


def _is_missing(x):
    """`True` for `None` and `pandas.NA`, the two forms a NULL cell arrives in."""
    return x is None or x is pd.NA


def _max_abs_diff(xs, ys):
    """Largest absolute numeric difference over paired cells, or `None` if not numeric.

    List cells are compared elementwise; a pair where either side is missing or NaN, or
    where the shapes disagree, is skipped (those already count in ``differing``).
    """
    best = None
    for x, y in zip(xs, ys):
        if _is_missing(x) or _is_missing(y):
            continue
        try:
            fx = np.asarray(x, dtype=float)
            fy = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            return None
        if fx.shape != fy.shape:
            continue
        d = np.abs(fx - fy)
        d = d[~np.isnan(d)]
        if d.size:
            m = float(d.max())
            best = m if best is None else max(best, m)
    return best


def _values_equal(a, b):
    """Whether two column values are equal, handling list-valued cells.

    Parameters
    ----------
    a, b : object
        Two cell values. Either may be a `numpy.ndarray` or `list` (the DOF and v-mode
        columns are ``LIST`` in DuckDB).

    Returns
    -------
    equal : `bool`
        `True` when the two are equal, treating NaN as equal to NaN so that an
        unrecovered visit's stored NaN does not read as a difference.
    """
    a_arr, b_arr = isinstance(a, (list, np.ndarray)), isinstance(b, (list, np.ndarray))
    if a_arr != b_arr:
        return False
    if a_arr:
        av, bv = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        if av.shape != bv.shape:
            return False
        try:
            af, bf = av.astype(float), bv.astype(float)
            return bool(np.array_equal(af, bf, equal_nan=True))
        except (TypeError, ValueError):
            return bool(np.array_equal(av, bv))
    a_na, b_na = _is_missing(a), _is_missing(b)
    if a_na or b_na:
        return a_na and b_na
    try:
        if isinstance(a, float) and isinstance(b, float):
            return bool(np.array_equal(a, b, equal_nan=True))
        fa, fb = float(a), float(b)
        return bool(np.array_equal(fa, fb, equal_nan=True))
    except (TypeError, ValueError):
        return a == b


def compare_table(da, db, key):
    """Compare two DataFrames of one table, aligned on `key`.

    Parameters
    ----------
    da, db : `pandas.DataFrame`
        The two tables.
    key : `list` [`str`] or `tuple` [`str`]
        Key columns, which must separate every row on both sides.

    Returns
    -------
    report : `dict`
        ``rows_a``, ``rows_b``, ``only_a`` and ``only_b`` (column names), ``differing``
        as a list of ``(column, n_differing_rows)``, ``max_abs`` as ``{column: largest
        absolute difference in the column's own units, or None if not numeric}``, ``compared`` (count of columns
        compared, dimensionless), ``keys_only_a`` and ``keys_only_b`` (counts of key
        tuples present on one side only), and ``unique_a`` / ``unique_b``.
    """
    key = [k for k in key if k in da.columns and k in db.columns]
    rep = {
        "rows_a": len(da), "rows_b": len(db),
        "only_a": sorted(set(da.columns) - set(db.columns)),
        "only_b": sorted(set(db.columns) - set(da.columns)),
        "differing": [], "compared": 0, "key": key,
        "unique_a": len(da.drop_duplicates(subset=key)) == len(da) if key else None,
        "unique_b": len(db.drop_duplicates(subset=key)) == len(db) if key else None,
    }
    if not key:
        return rep
    ia = da.set_index(key).sort_index()
    ib = db.set_index(key).sort_index()
    shared_keys = ia.index.intersection(ib.index)
    rep["keys_only_a"] = len(ia.index.difference(ib.index))
    rep["keys_only_b"] = len(ib.index.difference(ia.index))
    ia, ib = ia.loc[shared_keys], ib.loc[shared_keys]
    rep["n_shared"] = len(shared_keys)
    for col in sorted(set(ia.columns) & set(ib.columns)):
        rep["compared"] += 1
        xa, xb = ia[col].to_numpy(), ib[col].to_numpy()
        bad = [i for i, (x, y) in enumerate(zip(xa, xb)) if not _values_equal(x, y)]
        if bad:
            rep["differing"].append((col, len(bad)))
            rep.setdefault("max_abs", {})[col] = _max_abs_diff(xa[bad], xb[bad])
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db_a", help="first database (the reference)")
    ap.add_argument("db_b", help="second database")
    ap.add_argument("--tables", nargs="+", default=None,
                    help="restrict to tables whose name contains one of these strings "
                         "(default: all eight)")
    ap.add_argument("--strict-timestamps", action="store_true",
                    help="treat the EXPECTED_DIFF timestamp columns as differences too")
    args = ap.parse_args(argv)

    a, b = pathlib.Path(args.db_a), pathlib.Path(args.db_b)
    failed = False
    tables = list(TABLE_KEYS)
    if args.tables:
        tables = [t for t in tables if any(s in t for s in args.tables)]
    print(f"comparing {len(tables)} table(s) (count, dimensionless)")
    print(f"    a: {a}\n    b: {b}\n")

    for name in tables:
        da, dbf = _load(a, name), _load(b, name)
        print(f"--- {name}")
        if da is None and dbf is None:
            print("    absent from both databases -- nothing to compare")
            continue
        if da is None or dbf is None:
            print(f"    MISSING from {'a' if da is None else 'b'}")
            failed = True
            continue

        rep = compare_table(da, dbf, TABLE_KEYS[name])
        key_txt = "+".join(rep["key"]) or "(no usable key)"
        print(f"    rows: a={rep['rows_a']:,} b={rep['rows_b']:,}"
              f"{'  MISMATCH' if rep['rows_a'] != rep['rows_b'] else ''}")
        print(f"    key {key_txt}: unique on a={rep['unique_a']}, b={rep['unique_b']}")
        if rep["unique_a"] is not True or rep["unique_b"] is not True:
            print("    ERROR: the key does not separate every row, so a value "
                  "comparison would be meaningless")
            failed = True
        if rep.get("keys_only_a") or rep.get("keys_only_b"):
            print(f"    key tuples only in a: {rep.get('keys_only_a', 0):,} | "
                  f"only in b: {rep.get('keys_only_b', 0):,}")
            failed = True
        print(f"    columns compared: {rep['compared']} (count, dimensionless)"
              f" | only in a: {len(rep['only_a'])} | only in b: {len(rep['only_b'])}")
        if rep["only_a"]:
            print(f"      only in a: {rep['only_a']}")
            failed = True
        if rep["only_b"]:
            print(f"      only in b: {rep['only_b']}")
            failed = True
        if rep["rows_a"] != rep["rows_b"]:
            failed = True

        expected = () if args.strict_timestamps else EXPECTED_DIFF.get(name, ())
        unexpected = [(c, n) for c, n in rep["differing"] if c not in expected]
        as_expected = [(c, n) for c, n in rep["differing"] if c in expected]
        if as_expected:
            print("    differing as EXPECTED (build-time timestamps): "
                  + ", ".join(f"{c} ({n:,} rows)" for c, n in as_expected))
        if unexpected:
            print(f"    UNEXPECTED differences: {len(unexpected)} column(s)")
            for col, n in unexpected:
                m = rep.get("max_abs", {}).get(col)
                print(f"      {col}: {n:,} of {rep.get('n_shared', 0):,} rows differ"
                      + ("" if m is None else f", max abs difference {m:.6g}"
                         " (in the column's units, see docs/schema.md)"))
            failed = True
        elif not rep["differing"]:
            print("    values: identical")
        else:
            print("    values: identical apart from the expected timestamps")

    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
