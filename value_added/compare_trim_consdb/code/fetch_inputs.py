"""Fetch the three inputs of the Trim comparison into one run directory.

* ``db_trim.parquet`` — the value-added database Trim, ``dof0..49`` per exposure.
* ``consdb_trim.parquet`` — the ConsDB transformed-EFD Trim, property
  ``mt_logevent_aggregated_dof`` of ``efd_lsstcam.exposure_efd_unpivoted``, pivoted to
  ``dof0..49``.
* ``consdb_exposure.parquet`` — ``cdb_lsstcam.exposure`` identity, times and ``img_type``.
* ``efd_dof_events.parquet`` — every ``lsst.sal.MTAOS.logevent_degreeOfFreedom`` event,
  ``aggregatedDoF0..49``, so both selection rules can be rebuilt from the raw events.

Run on an s3df interactive node (EFD and ConsDB both reachable)::

    python value_added/compare_trim_consdb/code/fetch_inputs.py
"""
import argparse

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from rubinwork.common import study_run
from rubinwork.common.dof_telemetry import DOF_TOPIC, _run_coro
from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.products import catalog
from rubinwork.products.value_added import reader

N_DOF = 50
TRIM_PROPERTY = 'mt_logevent_aggregated_dof'
FIRST_DAY_OBS, LAST_DAY_OBS = 20250415, 20260714   # the value-added database span
EFD_CHUNK_DAYS = 7


def month_ranges(first, last):
    """Inclusive ``(first, last)`` day_obs pairs, one per calendar month."""
    out = []
    start = pd.Timestamp(str(first))
    end = pd.Timestamp(str(last))
    while start <= end:
        stop = min(start + pd.offsets.MonthEnd(0), end)
        out.append((int(start.strftime('%Y%m%d')), int(stop.strftime('%Y%m%d'))))
        start = stop + pd.Timedelta(days=1)
    return out


def fetch_consdb(consdb):
    """ConsDB exposure table and pivoted unpivoted-Trim, month by month."""
    exp_parts, trim_parts = [], []
    for d0, d1 in month_ranges(FIRST_DAY_OBS, LAST_DAY_OBS):
        exp_parts.append(consdb.query(
            'SELECT exposure_id, day_obs, seq_num, img_type, obs_start, obs_end, '
            'exp_time, science_program, observation_reason '
            f'FROM cdb_lsstcam.exposure WHERE day_obs BETWEEN {d0} AND {d1}').to_pandas())
        long = consdb.query(
            'SELECT exposure_id, field, value FROM efd_lsstcam.exposure_efd_unpivoted '
            f"WHERE property = '{TRIM_PROPERTY}' AND day_obs BETWEEN {d0} AND {d1}"
        ).to_pandas()
        if len(long):
            long['dof'] = long['field'].str.replace('aggregatedDoF', 'dof', regex=False)
            wide = long.pivot_table(index='exposure_id', columns='dof', values='value',
                                    aggfunc='first')
            trim_parts.append(wide.reset_index())
        print(f'ConsDB {d0}-{d1}: {len(exp_parts[-1])} exposures, '
              f'{len(trim_parts[-1]) if len(long) else 0} with Trim')
    exposure = pd.concat(exp_parts, ignore_index=True)
    trim = pd.concat(trim_parts, ignore_index=True)
    cols = ['exposure_id'] + [f'dof{i}' for i in range(N_DOF)]
    return exposure, trim.reindex(columns=cols)


def fetch_efd_events(efd):
    """All degreeOfFreedom events over the span, in weekly chunks; UTC index."""
    fields = [f'aggregatedDoF{i}' for i in range(N_DOF)]
    # One hour before the first night covers the ConsDB start_offset; a day after the
    # last day_obs covers its morning.
    t0 = Time(f'{str(FIRST_DAY_OBS)[:4]}-{str(FIRST_DAY_OBS)[4:6]}-'
              f'{str(FIRST_DAY_OBS)[6:]}T00:00:00', scale='utc')
    t_end = Time(f'{str(LAST_DAY_OBS)[:4]}-{str(LAST_DAY_OBS)[4:6]}-'
                 f'{str(LAST_DAY_OBS)[6:]}T00:00:00', scale='utc') + TimeDelta(2, format='jd')
    parts = []
    step = TimeDelta(EFD_CHUNK_DAYS, format='jd')
    t = t0
    while t < t_end:
        t1 = min(t + step, t_end)
        # The influx index is already UTC (it equals private_sndStamp converted from
        # TAI); convert_influx_index=True would shift it 37 s early.
        df = _run_coro(efd.select_time_series(DOF_TOPIC, fields, t, t1,
                                              convert_influx_index=False))
        if df is not None and len(df):
            parts.append(df)
        print(f'EFD {t.isot[:10]}: {0 if df is None else len(df)} events')
        t = t1
    ev = pd.concat(parts).sort_index()
    ev = ev[~ev.index.duplicated(keep='first')]
    ev.index = pd.to_datetime(ev.index, utc=True)
    ev.index.name = 'time_utc'
    ev.columns = [c.replace('aggregatedDoF', 'dof') for c in ev.columns]
    return ev.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reuse', action='store_true',
                        help='skip any input already written to the run dir')
    args = parser.parse_args()

    run = study_run('value_added/compare_trim_consdb', 'inputs')

    def todo(name):
        return not (args.reuse and (run.path / name).exists())

    db_path = catalog.load('value_added', 'v1')
    if todo('db_trim.parquet'):
        cols = [f'dof{i}' for i in range(N_DOF)]
        db = reader.visits(day_obs_range=(FIRST_DAY_OBS, LAST_DAY_OBS), columns=cols,
                           db_path=db_path)
        db.to_parquet(run.path / 'db_trim.parquet')
        print(f'value-added DB ({db_path}): {len(db)} exposures')

    if todo('consdb_trim.parquet'):
        exposure, trim = fetch_consdb(make_consdb_client())
        exposure.to_parquet(run.path / 'consdb_exposure.parquet')
        trim.to_parquet(run.path / 'consdb_trim.parquet')

    if todo('efd_dof_events.parquet'):
        # Direct construction: makeEfdClient refuses s3df nodes (getSite() is 'local').
        from lsst_efd_client import EfdClient
        ev = fetch_efd_events(EfdClient('usdf_efd'))
        ev.to_parquet(run.path / 'efd_dof_events.parquet')
        print(f'EFD: {len(ev)} degreeOfFreedom events')
    print(f'wrote {run.path}')


if __name__ == '__main__':
    main()
