#!/usr/bin/env python3
"""Compare two ``coadds`` builds: the two parquet tables and both npz grid files.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build a small reference before the move and the same one after, then compare. Values are
expected to match exactly, so the tolerances default to zero.

The parquet comparison is
`rubinwork.products.fam_tables.compare_builds.compare_table`, reused rather than copied.
Both tables key on **``block``** -- one row per detected block, verified unique on the
reference (3 of 3 rows on each, counts dimensionless). Uniqueness is asserted on both
sides, so a build that breaks it fails loudly instead of silently mis-aligning.

The npz files need their own comparison, because most of what this product carries is
arrays rather than table rows: ``block_grids.npz`` holds 42 arrays and
``block_grids_ext.npz`` holds 6. They are compared **array by array by name**, so an
array added or removed is reported rather than skipped. The per-block axis of a stacked
array is aligned on the ``block`` array, since array *order* within a build follows block
detection order and would otherwise be assumed rather than checked.

The PDFs are **not** compared: matplotlib embeds a creation timestamp, so two runs of
identical code produce different bytes. They are checked for existence and comparable
size.

Usage::

    python -m rubinwork.products.coadds.compare_builds <dir-a> <dir-b>
    python -m rubinwork.products.coadds.compare_builds <dir-a> <dir-b> --tables metrics
"""

import argparse
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

from ..fam_tables.compare_builds import compare_table
from . import reader

TABLE_KEYS = {
    "blocks_summary": ["block"],
    "coadd_metrics": ["block"],
}
"""Row key per table: ``block``, the detection index, one row per block."""

PDF_GLOBS = ("*.pdf",)
"""Plots to size-check rather than byte-compare."""


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


def compare_npz(path_a, path_b, rtol=0.0, atol=0.0):
    """Compare two npz archives array by array, aligned on ``block`` where it applies.

    Parameters
    ----------
    path_a, path_b : `pathlib.Path`
        The two archives.
    rtol, atol : `float`, optional
        Tolerances passed to `numpy.allclose` for floating-point arrays. Zero by default,
        which is exact equality.

    Returns
    -------
    report : `dict`
        ``n_arrays`` compared, ``only_a`` and ``only_b`` array names, ``differing`` as a
        list of ``(name, max abs difference or None)``, ``blocks_a`` and ``blocks_b`` the
        ``block`` arrays, and ``aligned`` saying whether the per-block axis was reordered
        to match.

    Notes
    -----
    A stacked array's first axis is the block axis when its length equals the number of
    blocks. Those arrays are reindexed on side b so that side b's ``block`` order matches
    side a's; everything else is compared as stored. ``block`` itself is compared as a
    set, since a different block *order* with the same blocks is not a difference in the
    data, only in detection order -- and that would be reported as a differing ``block``
    array anyway.
    """
    a = np.load(path_a, allow_pickle=True)
    b = np.load(path_b, allow_pickle=True)
    names_a, names_b = set(a.files), set(b.files)
    shared = sorted(names_a & names_b)

    blocks_a = np.asarray(a["block"]) if "block" in names_a else None
    blocks_b = np.asarray(b["block"]) if "block" in names_b else None
    order_b = None
    aligned = None
    if blocks_a is not None and blocks_b is not None:
        if sorted(blocks_a.tolist()) == sorted(blocks_b.tolist()):
            index = {int(v): i for i, v in enumerate(blocks_b)}
            order_b = np.array([index[int(v)] for v in blocks_a])
            aligned = bool(np.array_equal(order_b, np.arange(len(blocks_a))))
        else:
            aligned = False
    n_blocks = len(blocks_a) if blocks_a is not None else None

    differing = []
    for name in shared:
        arr_a, arr_b = a[name], b[name]
        if (order_b is not None and name != "block"
                and arr_b.ndim >= 1 and arr_b.shape[0] == n_blocks):
            arr_b = arr_b[order_b]
        if arr_a.shape != arr_b.shape:
            differing.append((name, None))
            continue
        if arr_a.dtype.kind in "fc" and arr_b.dtype.kind in "fc":
            same = np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
            if not same:
                with np.errstate(invalid="ignore"):
                    diff = np.abs(np.asarray(arr_a, float) - np.asarray(arr_b, float))
                differing.append((name, float(np.nanmax(diff)) if diff.size else None))
        elif not np.array_equal(arr_a, arr_b):
            differing.append((name, None))

    return {
        "n_arrays": len(shared),
        "only_a": sorted(names_a - names_b),
        "only_b": sorted(names_b - names_a),
        "differing": differing,
        "blocks_a": blocks_a,
        "blocks_b": blocks_b,
        "aligned": aligned,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", help="first build directory (the reference)")
    ap.add_argument("dir_b", help="second build directory")
    ap.add_argument("--tables", nargs="+", default=None,
                    help="restrict to tables and grid files whose name contains one of "
                         "these strings (default: all of them)")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args(argv)

    a, b = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)
    failed = False

    tables = list(TABLE_KEYS)
    grids = list(reader.GRIDS)
    if args.tables:
        tables = [t for t in tables if any(s in t for s in args.tables)]
        grids = [g for g in grids if any(s in g for s in args.tables)]
    print(f"comparing {len(tables)} table(s) and {len(grids)} grid file(s) "
          "(counts, dimensionless)\n")

    for name in tables:
        key = TABLE_KEYS[name]
        rel = f"{name}.parquet"
        pa, pb = a / rel, b / rel
        print(f"--- {name}  ({rel})")
        if not pa.exists() or not pb.exists():
            print(f"    MISSING: {pa if not pa.exists() else pb}")
            failed = True
            continue

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

    for name in grids:
        rel = f"{name}.npz"
        pa, pb = a / rel, b / rel
        print(f"\n--- {name}  ({rel})")
        if not pa.exists() or not pb.exists():
            print(f"    MISSING: {pa if not pa.exists() else pb}")
            failed = True
            continue
        rep = compare_npz(pa, pb, rtol=args.rtol, atol=args.atol)
        n_a = len(rep["blocks_a"]) if rep["blocks_a"] is not None else "?"
        n_b = len(rep["blocks_b"]) if rep["blocks_b"] is not None else "?"
        print(f"    blocks: a={n_a} b={n_b} (counts, dimensionless); "
              f"same blocks in the same order: {rep['aligned']}")
        if rep["aligned"] is False:
            print("    ERROR: the two builds do not hold the same blocks, so the "
                  "per-block arrays cannot be aligned")
            failed = True
        print(f"    arrays compared: {rep['n_arrays']} (count, dimensionless)"
              f" | only in a: {len(rep['only_a'])} | only in b: {len(rep['only_b'])}")
        if rep["only_a"]:
            print(f"      only in a: {rep['only_a']}")
            failed = True
        if rep["only_b"]:
            print(f"      only in b: {rep['only_b']}")
            failed = True
        if rep["differing"]:
            failed = True
            print(f"    differing: {len(rep['differing'])} arrays")
            for arr, mx in rep["differing"]:
                print(f"      {arr}: max abs difference "
                      f"{'(shape or non-numeric)' if mx is None else f'{mx:.6g}'}")
        else:
            print("    values: identical")

    # The PDFs carry a matplotlib creation timestamp, so compare sizes, not bytes.
    print("\n--- plots (size only; matplotlib embeds a timestamp)")
    for pattern in PDF_GLOBS:
        rels = sorted({str(p.relative_to(a)) for p in a.glob(pattern)}
                      | {str(p.relative_to(b)) for p in b.glob(pattern)})
        for rel in rels:
            pa, pb = a / rel, b / rel
            if not pa.exists() or not pb.exists():
                print(f"    MISSING: {pa if not pa.exists() else pb}")
                failed = True
                continue
            sa, sb = pa.stat().st_size, pb.stat().st_size
            frac = abs(sa - sb) / sa if sa else 0.0
            print(f"    {rel}: a={sa:,} b={sb:,} bytes (difference {frac:.2%} of a)")

    print("\nRESULT:", "DIFFERENCES FOUND" if failed else "builds match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
