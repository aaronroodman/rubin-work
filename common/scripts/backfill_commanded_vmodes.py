"""Fill in the commanded hexapod LUT and Trim v-modes for `optical_state` rows that lack them.

A sharded `build_optical_state.py` run cannot compute the commanded terms: each shard writes
its own database, whose `visit_telemetry` table is created empty by the schema, so the
projector finds no telemetry row for any visit and stores NaN. The measured state is
unaffected -- only ``v_modes_lut`` and ``v_modes_trim`` are lost. This script runs over the
merged database, where telemetry is present, and fills them in.

The projection is `build_optical_state.make_commanded_projector`, i.e. the same code path a
non-sharded build uses, so a backfilled row is indistinguishable from a directly built one.
Each variant is projected in its own scheme, so all three v-mode terms of a visit share one
basis and ``v1_lut + v1_trim - v1_meas`` mixes none.

Usage
-----
    python common/scripts/backfill_commanded_vmodes.py
    python common/scripts/backfill_commanded_vmodes.py --variant v50_34__batoid__consdb_v1
    python common/scripts/backfill_commanded_vmodes.py --dry-run

Options
-------
--variant ID    only this variant (default: every variant with missing commanded terms)
--db PATH       database to update (default: the shared value-added database)
--chunk N       visits per UPDATE batch (default 5000)
--dry-run       report what would be filled, write nothing
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'aos' / 'code'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import efd_db  # noqa: E402
from build_optical_state import (build_state_estimator, make_commanded_projector,  # noqa: E402
                                 DEFAULT_OFC_VERSION)


def missing_counts(con, variant=None):
    """Rows per variant whose commanded v-modes are absent, and how many could be filled.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Connection to the merged database.
    variant : `str`, optional
        Restrict to this variant id.

    Returns
    -------
    rows : `list` [`tuple`]
        ``(variant_id, scheme, ofc_config_version, n_missing, n_fillable)`` -- `n_fillable`
        counts only those with a `visit_telemetry` row carrying a finite hexapod LUT, so
        `n_missing - n_fillable` is the irreducible remainder.
    """
    where = 'AND o.variant_id = ?' if variant else ''
    args = [variant] if variant else []
    return con.execute(
        'SELECT o.variant_id, s.scheme, s.ofc_config_version, COUNT(*) AS n_missing, '
        '       SUM(CASE WHEN t.visit_id IS NOT NULL AND t.lut_dof0 IS NOT NULL '
        '                THEN 1 ELSE 0 END) AS n_fillable '
        'FROM optical_state o '
        'JOIN state_variant s USING (variant_id) '
        'LEFT JOIN visit_telemetry t USING (visit_id) '
        'WHERE (o.v_modes_lut IS NULL OR isnan(o.v_modes_lut[1])) '
        f'{where} '
        'GROUP BY 1, 2, 3 ORDER BY 1', args).fetchall()


def backfill_variant(con, variant, scheme, ofc_version, chunk=5000, dry_run=False):
    """Project and store the commanded v-modes for one variant.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection, unless `dry_run`.
    variant : `str`
        Variant id to fill.
    scheme : `str`
        ``'22_12'`` or ``'50_34'`` -- sets the v-mode count.
    ofc_version : `str`
        OFC config version, e.g. ``'v13'``.
    chunk : `int`, optional
        Visits per UPDATE batch.
    dry_run : `bool`, optional
        Compute and report, write nothing.

    Returns
    -------
    n_filled : `int`
        Rows given finite commanded v-modes.
    """
    vids = [int(v) for (v,) in con.execute(
        'SELECT o.visit_id FROM optical_state o '
        'JOIN visit_telemetry t USING (visit_id) '
        'WHERE o.variant_id = ? AND (o.v_modes_lut IS NULL OR isnan(o.v_modes_lut[1])) '
        '  AND t.lut_dof0 IS NOT NULL '
        'ORDER BY o.visit_id', [variant]).fetchall()]
    if not vids:
        print(f'  {variant}: nothing fillable')
        return 0

    se, n_modes = build_state_estimator(scheme, ofc_version or DEFAULT_OFC_VERSION)
    project = make_commanded_projector(se, n_modes)
    n_filled = 0
    for i in range(0, len(vids), chunk):
        batch = vids[i:i + chunk]
        v_lut, v_trim = project(con, batch)
        good = np.isfinite(v_lut[:, 0]) & np.isfinite(v_trim[:, 0])
        n_filled += int(good.sum())
        if dry_run:
            continue
        payload = [[v_lut[j].tolist(), v_trim[j].tolist(), int(batch[j])]
                   for j in range(len(batch)) if good[j]]
        if payload:
            con.executemany(
                'UPDATE optical_state SET v_modes_lut = ?, v_modes_trim = ? '
                'WHERE visit_id = ? AND variant_id = ' + f"'{variant}'", payload)
    verb = 'would fill' if dry_run else 'filled'
    print(f'  {variant}: {verb} {n_filled} of {len(vids)} candidate row(s), '
          f'{n_modes} v-modes each (dimensionless)')
    return n_filled


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', default=None, help='only this variant id')
    p.add_argument('--db', default=None, help='database path')
    p.add_argument('--chunk', type=int, default=5000, help='visits per UPDATE batch')
    p.add_argument('--dry-run', action='store_true', help='report only, write nothing')
    a = p.parse_args(argv)

    con = efd_db.open_db(a.db, readonly=a.dry_run)
    todo = missing_counts(con, a.variant)
    if not todo:
        print('no rows are missing commanded v-modes')
        con.close()
        return 0

    print('rows missing commanded v-modes:')
    for vid, scheme, _ofc, n_missing, n_fillable in todo:
        print(f'  {vid}: {n_missing} missing, {n_fillable} with telemetry')
    print()

    total = 0
    for vid, scheme, ofc, _n_missing, n_fillable in todo:
        if not n_fillable:
            print(f'  {vid}: no telemetry for any missing row, skipped')
            continue
        total += backfill_variant(con, vid, scheme, ofc, chunk=a.chunk,
                                  dry_run=a.dry_run)
    if not a.dry_run:
        con.execute('CHECKPOINT')
    con.close()
    print(f'\n{"would fill" if a.dry_run else "filled"} {total} row(s) in total')
    return 0


if __name__ == '__main__':
    sys.exit(main())
