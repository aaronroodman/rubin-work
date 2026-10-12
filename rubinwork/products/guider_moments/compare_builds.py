#!/usr/bin/env python3
"""Compare two ``guider_moments`` nights table by table, at zero tolerance.

The regression check for the phase 3a code move (``notes/status/organization_plan.md``):
build a small reference with the unmoved code and the same one after, then compare.

The three combined tables are parquet, so the comparison reuses
`rubinwork.products.fam_tables.compare_builds` -- `key_is_unique` and `compare_table` --
rather than defining a second one. The ``moments`` table carries per-row arrays (the
centroid power spectral densities), which that module already stacks before comparing.

**Nothing is expected to differ.** Unlike ``value_added``, this product stamps no
wall-clock timestamp anywhere: the per-exposure writers record the tracker and moment
settings and the tables, and the combine step concatenates. The ``stars`` table's
``timestamp`` column is *exposure* time in Modified Julian Date (MJD), not build time, so
it must match exactly too. Any difference at all is therefore unexpected.

Usage::

    python -m rubinwork.products.guider_moments.compare_builds <night-a> <night-b> 20260706
"""

import argparse
import pathlib
import sys

from ..fam_tables import compare_builds as parquet_compare
from . import reader

TEXT_FILES = (reader.SKIPPED_FILE, reader.DETECTOR_COUNTS_FILE)
"""The text sidecars the combine step writes, compared byte for byte.

``detector_counts_<night>.txt`` is **not declared** by the ``aos``-era ``combine`` rule but
is written unconditionally, so the product rule declares it and the comparison checks it.
"""


def compare_night(root_a, root_b, night, tables=reader.TABLES):
    """Compare one night between two build roots.

    Parameters
    ----------
    root_a, root_b : `str` or `pathlib.Path`
        The two night directories, each holding ``guider_<table>_<night>.parquet``.
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.
    tables : `tuple` [`str`], optional
        Which of `reader.TABLES` to compare; all three by default.

    Returns
    -------
    reports : `dict`
        Keyed by table name, each the report
        `rubinwork.products.fam_tables.compare_builds.compare_table` returns, with
        ``unique_a``, ``n_unique_a`` and ``n_rows_a`` added from the key check.
    """
    root_a, root_b = pathlib.Path(root_a), pathlib.Path(root_b)
    night = int(night)
    reports = {}
    for table in tables:
        name = f'guider_{table}_{night}.parquet'
        key = list(reader.TABLE_KEYS[table])
        unique, n_unique, n_rows = parquet_compare.key_is_unique(root_a / name, key)
        rep = parquet_compare.compare_table(root_a / name, root_b / name, key)
        rep.update(unique_a=unique, n_unique_a=n_unique, n_rows_a=n_rows)
        reports[table] = rep
    return reports


def compare_text(root_a, root_b, night):
    """Compare the combine step's text sidecars byte for byte.

    Parameters
    ----------
    root_a, root_b : `str` or `pathlib.Path`
        The two night directories.
    night : `int` or `str`
        ``day_obs`` as ``YYYYMMDD``.

    Returns
    -------
    results : `dict`
        Keyed by file name, each ``True`` when the two files are byte-identical, ``False``
        when they differ, and ``None`` when the file is absent from both.
    """
    root_a, root_b = pathlib.Path(root_a), pathlib.Path(root_b)
    out = {}
    for template in TEXT_FILES:
        name = template.format(night=int(night))
        pa, pb = root_a / name, root_b / name
        if not pa.exists() and not pb.exists():
            out[name] = None
        elif not pa.exists() or not pb.exists():
            out[name] = False
        else:
            out[name] = pa.read_bytes() == pb.read_bytes()
    return out


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        0 when every table and sidecar matches, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog='python -m rubinwork.products.guider_moments.compare_builds',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root_a', help='the reference night directory')
    parser.add_argument('root_b', help='the second night directory')
    parser.add_argument('night', help='day_obs as YYYYMMDD')
    parser.add_argument('--tables', nargs='+', default=list(reader.TABLES),
                        help='tables to compare (default: all three)')
    args = parser.parse_args(argv)

    failed = False
    print(f'comparing {len(args.tables)} table(s) (count, dimensionless) '
          f'for day_obs {args.night}')
    print(f'    a: {args.root_a}\n    b: {args.root_b}\n')

    reports = compare_night(args.root_a, args.root_b, args.night, tuple(args.tables))
    for table, rep in reports.items():
        key = '+'.join(reader.TABLE_KEYS[table])
        print(f'--- {table}')
        print(f'    rows: a={rep["rows_a"]:,} b={rep["rows_b"]:,}'
              f'{"  MISMATCH" if rep["rows_a"] != rep["rows_b"] else ""}')
        print(f'    key {key}: unique on a={rep["unique_a"]} '
              f'({rep["n_unique_a"]:,} distinct of {rep["n_rows_a"]:,} rows)')
        if not rep['unique_a']:
            print('    ERROR: the key does not separate every row, so a value '
                  'comparison would be meaningless')
            failed = True
        print(f'    columns compared: {rep["compared"]} (count, dimensionless)'
              f' | only in a: {len(rep["only_a"])} | only in b: {len(rep["only_b"])}')
        for side in ('only_a', 'only_b'):
            if rep[side]:
                print(f'      {side}: {rep[side]}')
                failed = True
        if rep['rows_a'] != rep['rows_b']:
            failed = True
        if rep['key_aligned'] is False:
            print('    ERROR: key columns do not line up after sorting')
            failed = True
        if rep['differing']:
            print(f'    UNEXPECTED differences: {len(rep["differing"])} column(s)')
            for col, delta in rep['differing']:
                print(f'      {col}: max |a-b| = '
                      + ('non-numeric' if delta is None else f'{delta:.6g}'))
            failed = True
        else:
            print('    identical')

    print('\n--- text sidecars')
    for name, same in compare_text(args.root_a, args.root_b, args.night).items():
        if same is None:
            print(f'    {name}: absent from both')
        elif same:
            print(f'    {name}: byte-identical')
        else:
            print(f'    {name}: DIFFERS')
            failed = True

    print('\nRESULT: ' + ('builds DIFFER' if failed else 'builds match'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
