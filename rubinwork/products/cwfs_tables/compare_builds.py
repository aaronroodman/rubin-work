#!/usr/bin/env python3
"""Compare two ``cwfs_tables`` builds table by table: row counts, column sets and values.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build one night before the move and one after, then compare. Values are expected to match
exactly, so the tolerances default to zero.

The per-table comparison itself is
`rubinwork.products.fam_tables.compare_builds.compare_table`, reused rather than copied;
what differs is the row key. Rows must be aligned on a key before values can be compared
at all, because ``donuts.parquet`` row *order* is not stable between runs.

**The key is not the ``fam_tables`` one.** These tables carry no donut-id column, so
``day_obs, seq_num, detector, extra_donut_id`` does not exist here, and
``day_obs, seq_num, detector`` alone is **not** unique on the paired variants -- several
donuts share one corner sensor in an exposure. The key used is
``day_obs, seq_num, detector, thx_OCS, thy_OCS`` (field angles in radians, Optical
Coordinate System), which is unique on every built variant including the ``unpaired``
ones, which carry no centroid columns. Uniqueness is asserted on both sides, so a future
variant that breaks it fails loudly instead of silently mis-aligning.

The build's ``wfs_mktable_validation.pdf`` is **not** compared: matplotlib embeds a
creation timestamp, so two runs of identical code produce different bytes. Check that it
exists and is of comparable size.

Usage::

    python -m rubinwork.products.cwfs_tables.compare_builds <dir-a> <dir-b>
    python -m rubinwork.products.cwfs_tables.compare_builds <dir-a> <dir-b> --tables visits
"""

import argparse
import pathlib
import sys

import pyarrow.parquet as pq

from ..fam_tables.compare_builds import compare_table
from .reader import DONUT_KEY, VISIT_KEY

TABLE_KEYS = {
    "donuts": list(DONUT_KEY),
    "visits": list(VISIT_KEY),
}
"""Row key per table: the columns that make a row unique, for alignment."""

PDF_NAME = "wfs_mktable_validation.pdf"
"""The validation plot, size-checked rather than byte-compared."""


def key_is_unique(path, key):
    """Whether `key` makes every row of a table unique.

    Parameters
    ----------
    path : `pathlib.Path`
        A parquet file.
    key : `list` [`str`]
        Candidate key columns.

    Returns
    -------
    unique : `bool`
        `True` if the key columns present in the table separate every row.
    n_unique : `int`
        Distinct key tuples (count, dimensionless).
    n_rows : `int`
        Rows in the table (count, dimensionless).
    """
    usable = [k for k in key if k in pq.ParquetFile(str(path)).schema.names]
    df = pq.read_table(path, columns=usable).to_pandas()
    n_unique = len(df.drop_duplicates())
    return n_unique == len(df), n_unique, len(df)


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

        key = TABLE_KEYS.get(table, [])
        for side, path in (("a", pa), ("b", pb)):
            unique, n_unique, n_rows = key_is_unique(path, key)
            print(f"    key {'+'.join(key)} on {side}: {n_unique} unique of {n_rows} rows"
                  f" (counts, dimensionless) -> {'unique' if unique else 'NOT UNIQUE'}")
            if not unique:
                print("    ERROR: the key does not separate every row, so a value "
                      "comparison would be meaningless")
                failed = True

        rep = compare_table(pa, pb, key, rtol=args.rtol, atol=args.atol)
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

    # The PDF carries a matplotlib creation timestamp, so compare sizes, not bytes.
    qa, qb = a / PDF_NAME, b / PDF_NAME
    print(f"--- {PDF_NAME}")
    if qa.exists() and qb.exists():
        sa, sb = qa.stat().st_size, qb.stat().st_size
        frac = abs(sa - sb) / sa if sa else 0.0
        print(f"    bytes: a={sa:,} b={sb:,} (difference {frac:.2%} of a; not "
              "byte-compared, matplotlib embeds a timestamp)")
    else:
        print(f"    MISSING: {qa if not qa.exists() else qb}")
        failed = True

    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
