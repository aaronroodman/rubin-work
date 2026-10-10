#!/usr/bin/env python3
"""Compare two ``miw`` builds table by table: row counts, column sets and values.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build a small reference before the move and the same one after, then compare. Values are
expected to match exactly, so the tolerances default to zero.

The per-table comparison itself is
`rubinwork.products.fam_tables.compare_builds.compare_table`, reused rather than copied;
what differs is the row key. Rows must be aligned on a key before values can be compared
at all, because parquet row *order* is not stable between runs.

**One key per table, each verified unique on the reference** (counts dimensionless):

``intrinsic_split_maps``
    ``thx_deg, thy_deg`` -- one row per field grid cell. 3,985 of 3,985 unique.
``intrinsic_split_decomp``
    ``j, part`` -- one row per (Noll index, real/imaginary part). 21 of 21.
``intrinsic_split_rms``
    ``j`` -- one row per Noll index. 21 of 21.
``fits``
    ``day_obs, seq_num`` -- one row per FAM visit. 169 of 169.
``zk_intrinsic`` and the per-CWFS ``wfs/<variant>/zk_intrinsic``
    ``day_obs, seq_num, detector, centroid_x_extra, centroid_y_extra``.
    ``day_obs, seq_num, detector`` alone is **not** unique -- 30,352 of 475,128 on the FAM
    sidecar and 4,485 of 34,631 on the corner one, because many donuts share a detector in
    one exposure -- so the extra-focal centroid is what separates the rows.
``build/rot_<lo>_<hi>/intrinsic_grid``
    ``thx_deg, thy_deg``. 3,831 of 3,831 on the ``rot_-3_3`` bin.
``build/rot_<lo>_<hi>/dz_fits``
    ``day_obs, seq_num``. 37 of 37 on that bin.
``build/rot_<lo>_<hi>/intrinsic_cov_edge``
    ``j``. 21 of 21.

Uniqueness is asserted on both sides, so a build that breaks it fails loudly instead of
silently mis-aligning.

The PDFs are **not** compared: matplotlib embeds a creation timestamp, so two runs of
identical code produce different bytes. They are checked for existence and comparable
size. The per-bin ``mi_config.yaml`` the builder writes IS byte-compared, since it is the
resolved configuration and must not drift.

Usage::

    python -m rubinwork.products.miw.compare_builds <dir-a> <dir-b>
    python -m rubinwork.products.miw.compare_builds <dir-a> <dir-b> --tables fits
"""

import argparse
import pathlib
import sys

import pyarrow.parquet as pq

from ..fam_tables.compare_builds import compare_table

DONUT_SIDECAR_KEY = ["day_obs", "seq_num", "detector",
                     "centroid_x_extra", "centroid_y_extra"]
"""Row key for either ``zk_intrinsic`` table: detector alone is not unique."""

TABLE_KEYS = {
    "intrinsic_split_maps": ["thx_deg", "thy_deg"],
    "intrinsic_split_decomp": ["j", "part"],
    "intrinsic_split_rms": ["j"],
    "zk_intrinsic": DONUT_SIDECAR_KEY,
    "fits": ["day_obs", "seq_num"],
}
"""Row key per build-level table: the columns that make a row unique, for alignment."""

BIN_TABLE_KEYS = {
    "intrinsic_grid": ["thx_deg", "thy_deg"],
    "dz_fits": ["day_obs", "seq_num"],
    "intrinsic_cov_edge": ["j"],
}
"""Row key per per-rotator-bin table, under ``build/rot_<lo>_<hi>/``."""

PDF_GLOBS = ("*.pdf", "build/*/*.pdf")
"""Plots to size-check rather than byte-compare."""

BYTE_COMPARED = ("build/*/mi_config.yaml",)
"""Files compared byte for byte: the resolved configuration the builder recorded."""


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


def discover_tables(a, b):
    """Every parquet both builds hold, as ``(label, relative path, key)``.

    Walks the build-level tables, the per-CWFS corner sidecars under ``wfs/`` and the
    per-rotator-bin tables under ``build/``, so a build with more bins or more paired CWFS
    variants is compared in full rather than up to a hardcoded list.

    Parameters
    ----------
    a, b : `pathlib.Path`
        The two build directories.

    Returns
    -------
    tables : `list` [`tuple`]
        ``(label, relative path, key columns)``, in a stable order. A table present in
        only one build is included, so the comparison reports it as missing.
    """
    found = {}
    for side in (a, b):
        for name, key in TABLE_KEYS.items():
            rel = f"{name}.parquet"
            if (side / rel).exists():
                found[rel] = (name, rel, key)
        for path in sorted(side.glob("wfs/*/zk_intrinsic.parquet")):
            rel = str(path.relative_to(side))
            found[rel] = (f"wfs sidecar {path.parent.name}", rel, DONUT_SIDECAR_KEY)
        for name, key in BIN_TABLE_KEYS.items():
            for path in sorted(side.glob(f"build/*/{name}.parquet")):
                rel = str(path.relative_to(side))
                found[rel] = (f"{path.parent.name}/{name}", rel, key)
    return [found[rel] for rel in sorted(found)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", help="first build directory (the reference)")
    ap.add_argument("dir_b", help="second build directory")
    ap.add_argument("--tables", nargs="+", default=None,
                    help="restrict to tables whose label contains one of these strings "
                         "(default: every parquet either build holds)")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args(argv)

    a, b = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)
    failed = False
    tables = discover_tables(a, b)
    if args.tables:
        tables = [t for t in tables if any(s in t[0] for s in args.tables)]
    print(f"comparing {len(tables)} table(s) (count, dimensionless)\n")

    for label, rel, key in tables:
        pa, pb = a / rel, b / rel
        print(f"--- {label}  ({rel})")
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

    print("\n--- byte-compared files")
    for pattern in BYTE_COMPARED:
        rels = sorted({str(p.relative_to(a)) for p in a.glob(pattern)}
                      | {str(p.relative_to(b)) for p in b.glob(pattern)})
        for rel in rels:
            pa, pb = a / rel, b / rel
            if not pa.exists() or not pb.exists():
                print(f"    MISSING: {pa if not pa.exists() else pb}")
                failed = True
                continue
            same = pa.read_bytes() == pb.read_bytes()
            print(f"    {rel}: {'identical' if same else 'DIFFERS'}")
            if not same:
                failed = True

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
