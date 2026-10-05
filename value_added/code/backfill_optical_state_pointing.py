"""Backfill `optical_state.elevation_deg` and `rotator_angle_deg` from ConsDB.

Both columns are properties of the visit rather than of the variant, so one fetch per
night fills every variant's row for that night and no optical-state recovery is rerun.
That is what makes this a backfill rather than a rebuild: the recovered DOF and v-modes
are not touched, only the two pointing columns added alongside them.

The elevation is ``cdb_lsstcam.exposure.altitude`` and the rotator angle is
``cdb_lsstcam.visit1_quicklook.physical_rotator_angle``, the same two values
`build_optical_state.visit_metadata` already returns -- the rotator angle is what the
intrinsic-wavefront evaluation uses, so it was being fetched and discarded.

Usage::

    python value_added/code/backfill_optical_state_pointing.py --day-obs 20250724-20260714
    python value_added/code/backfill_optical_state_pointing.py --day-obs 20260318 --dry-run
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from common.telemetry_clients import make_consdb_client

import build_optical_state as bos
import efd_db


def nights_needing_backfill(con, day_obs_range=None):
    """Nights with at least one `optical_state` row missing a pointing value.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
    day_obs_range : `tuple` [`int`], optional
        Inclusive ``(lo, hi)`` `day_obs` bounds.

    Returns
    -------
    nights : `list` [`int`]
        Ascending `day_obs`.

    Notes
    -----
    `day_obs` is derived as ``visit_id // 100000`` rather than by joining
    `visit_telemetry`, so a visit absent from that table is still backfilled.
    """
    where = ['(elevation_deg IS NULL OR rotator_angle_deg IS NULL)']
    params = []
    if day_obs_range is not None:
        lo, hi = day_obs_range
        if lo is not None:
            where.append('visit_id // 100000 >= ?')
            params.append(int(lo))
        if hi is not None:
            where.append('visit_id // 100000 <= ?')
            params.append(int(hi))
    rows = con.execute(
        f'SELECT DISTINCT visit_id // 100000 AS day_obs FROM optical_state '
        f'WHERE {" AND ".join(where)} ORDER BY day_obs', params).fetchall()
    return [int(r[0]) for r in rows]


def backfill_night(con, cdb, day_obs, dry_run=False):
    """Fill both pointing columns for every `optical_state` row of one night.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection, unless `dry_run`.
    cdb : `lsst.summit.utils.ConsDbClient`
    day_obs : `int`
    dry_run : `bool`, optional
        Report what would be written without writing.

    Returns
    -------
    stats : `dict`
        ``n_rows`` updated (all variants), ``n_visits`` matched, ``n_missing`` visits in
        the table with no ConsDB pointing, and ``elev_range`` / ``rot_range`` as
        ``(min, max)`` [deg] over the night, NaN where nothing matched.
    """
    meta = bos.visit_metadata(cdb, day_obs)
    have = con.execute(
        'SELECT DISTINCT visit_id FROM optical_state WHERE visit_id // 100000 = ?',
        [int(day_obs)]).fetchall()
    want = {int(r[0]) for r in have}
    nan2 = (float('nan'), float('nan'))
    if meta.empty or not want:
        return dict(n_rows=0, n_visits=0, n_missing=len(want), n_partial=0,
                    elev_range=nan2, rot_range=nan2)

    meta = meta[meta['visit_id'].isin(want)]
    elev = meta['altitude_deg'].to_numpy(float)
    rot = meta['rotator_angle_deg'].to_numpy(float)
    vids = meta['visit_id'].to_numpy('int64')
    # A non-finite pointing value must land as NULL, not as a NaN that reads as present.
    payload = [(None if not np.isfinite(e) else float(e),
                None if not np.isfinite(r) else float(r), int(v))
               for v, e, r in zip(vids, elev, rot)]
    # A visit can be in ConsDB with an elevation but no rotator angle, since
    # physical_rotator_angle comes from a left-join to visit1_quicklook. Count that
    # separately from a visit ConsDB does not have at all, or a partial record reads as
    # a clean fill.
    stats = dict(n_visits=len(payload), n_missing=len(want - set(vids.tolist())),
                 n_partial=int((np.isfinite(elev) != np.isfinite(rot)).sum()),
                 elev_range=(float(np.nanmin(elev)), float(np.nanmax(elev)))
                 if len(elev) and np.isfinite(elev).any() else nan2,
                 rot_range=(float(np.nanmin(rot)), float(np.nanmax(rot)))
                 if len(rot) and np.isfinite(rot).any() else nan2)
    if dry_run:
        n = con.execute(
            'SELECT COUNT(*) FROM optical_state WHERE visit_id // 100000 = ?',
            [int(day_obs)]).fetchone()[0]
        stats['n_rows'] = int(n)
        return stats

    # One statement per visit, across all variants: the update is keyed on visit_id
    # alone, so every variant's row for that visit gets the same pointing.
    con.executemany(
        'UPDATE optical_state SET elevation_deg = ?, rotator_angle_deg = ? '
        'WHERE visit_id = ?', payload)
    n = con.execute(
        'SELECT COUNT(*) FROM optical_state WHERE visit_id // 100000 = ? '
        'AND elevation_deg IS NOT NULL', [int(day_obs)]).fetchone()[0]
    stats['n_rows'] = int(n)
    return stats


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--day-obs', default=None,
                   help='single night 20260318 or inclusive range 20250724-20260714; '
                        'default is every night with a missing pointing value')
    p.add_argument('--db', default=None, help='database path, default efd_db.default_db_path()')
    p.add_argument('--consdb-url', default='auto')
    p.add_argument('--dry-run', action='store_true',
                   help='report per night without writing')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)

    rng = None
    if a.day_obs:
        lo, _, hi = a.day_obs.partition('-')
        rng = (int(lo), int(hi) if hi else int(lo))

    con = efd_db.open_db(a.db, readonly=a.dry_run)
    try:
        efd_db.create_schema(con) if not a.dry_run else None
        nights = nights_needing_backfill(con, rng)
        if not nights:
            print('no nights need a pointing backfill in that range')
            return 0
        print(f'{len(nights)} nights to backfill, {nights[0]} to {nights[-1]}'
              + (' (dry run)' if a.dry_run else ''))
        cdb = make_consdb_client(a.consdb_url)
        tot_rows = tot_vis = tot_miss = tot_part = 0
        for day in nights:
            try:
                s = backfill_night(con, cdb, day, dry_run=a.dry_run)
            except Exception as exc:
                # One night per ConsDB query, so a transient failure costs that night
                # only; rerun the script and it picks up whatever is still NULL.
                print(f'{day}: FAILED {type(exc).__name__}: {exc}')
                continue
            tot_rows += s['n_rows']
            tot_vis += s['n_visits']
            tot_miss += s['n_missing']
            tot_part += s['n_partial']
            if not a.quiet:
                print(f'{day}: {s["n_rows"]} rows, {s["n_visits"]} visits, '
                      f'elevation {s["elev_range"][0]:.2f} to {s["elev_range"][1]:.2f} deg, '
                      f'rotator {s["rot_range"][0]:.2f} to {s["rot_range"][1]:.2f} deg'
                      + (f', {s["n_missing"]} visits absent from ConsDB'
                         if s['n_missing'] else '')
                      + (f', {s["n_partial"]} visits with elevation but no rotator angle'
                         if s['n_partial'] else ''))
        print(f'\n{tot_rows} rows carry pointing over {len(nights)} nights, '
              f'{tot_vis} visits matched, {tot_miss} visits absent from ConsDB, '
              f'{tot_part} visits with only one of the two angles')
    finally:
        con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
