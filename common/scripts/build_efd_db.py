#!/usr/bin/env python3
"""Build the value-added telemetry database, one night and one group at a time.

Fetches the slow Engineering Facility Database (EFD) quantities and the value-added
quantities derived from them into ``output/value_added/aos_efd.duckdb``, so that later
analysis never re-pays the EFD cost. Consolidated Database (ConsDB) columns are **not**
copied — ConsDB is fast and is read live via `common.efd_db.join_consdb`.

Usage
-----
One night, every group (do this first, before any backfill)::

    python common/scripts/build_efd_db.py --day-obs 20251102 --groups all

A range, resuming an interrupted run::

    python common/scripts/build_efd_db.py --day-obs 20250415-20260913 --groups all --resume

Add a newly-defined group over nights already built, touching nothing else::

    python common/scripts/build_efd_db.py --day-obs 20250415-20260913 --groups turbulence

Groups
------
``trim``
    Accumulated Active Optics System (AOS) offset, ``MTAOS.logevent_degreeOfFreedom``,
    as-of each visit's ``obs_start``. 50 degrees of freedom (DOF), µm and deg.
``lut``
    Hexapod look-up-table (LUT) baseline, ``MTHexapod.logevent_compensationOffset``,
    salIndex 1 (camera) and 2 (M2). 10 values, µm and deg.
``tweak``
    Value-added: ``Tweak_i = Trim_i - Trim_(i-1)``, exactly 0.0 where the loop applied no
    new correction. Requires ``trim`` in the same pass.
``gradients``
    Value-added: M1M3 bulk thermal gradients [°C/m] from the raw thermocouples, one night
    at a time — a multi-night thermocouple query times out.
``camera``
    Camera-body temperatures [°C], ``lsst.MTCamera.utiltrunk_body``, which is in the
    camera's own InfluxDB database and in no ConsDB table.
``turbulence``
    3D sonic anemometers on ``ESS.airTurbulence`` at salIndex 123–126, the TMA top-ring
    quadrants, which are absent from the transformed EFD.
``wind_derived``
    Value-added: ``into_wind_deg = wrap180(wind_dir - azimuth)`` [deg], 0 deg being into
    the wind. Its two ConsDB inputs are stored alongside it, because the wrap convention
    is easy to get wrong.
``hexhist``
    Value-added: hexapod motion history within the night — cumulative and trailing-30-min
    ``|delta dz|`` [µm] and the commanded-move count. Requires ``lut`` and ``trim``.

Notes
-----
Nights are independent and each ``(day_obs, group)`` outcome is recorded in ``fetch_log``,
so an interrupted backfill resumes with ``--resume`` and a group that fails leaves NULLs
rather than aborting the night.

A quantity that post-dates the start of the range is NULL for the earlier nights, which is
expected rather than a failure; ``column_coverage`` records each column's true first and
last ``day_obs`` so a reader can tell that from a failed fetch.
"""
import argparse
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'aos' / 'code'))
sys.path.insert(0, str(_ROOT / 'aos' / 'code' / 'fam_processing'))
sys.path.insert(0, str(_ROOT / 'olr' / 'code'))

from common import efd_db                                            # noqa: E402
from common.telemetry_clients import make_consdb_client, make_efd_client  # noqa: E402

DEFAULT_FIRST_DAY_OBS = 20250415
INSTRUMENT = 'lsstcam'


# ---------------------------------------------------------------------------
# Night list and the visit spine
# ---------------------------------------------------------------------------
def parse_day_obs(spec, cdb=None):
    """Expand a ``--day-obs`` specification into a sorted list of nights.

    Parameters
    ----------
    spec : `str`
        A single night (``20251102``), an inclusive range (``20250415-20260913``), or a
        comma list of either.
    cdb : `lsst.summit.utils.ConsDbClient`, optional
        Used to list the nights that actually have exposures within a range, so an
        18-month range does not iterate over empty nights.

    Returns
    -------
    days : `list` [`int`]
    """
    out = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = (int(x) for x in part.split('-', 1))
            if cdb is not None:
                out.update(nights_with_exposures(cdb, lo, hi))
            else:
                raise ValueError('a day_obs range needs a ConsDB client to expand')
        else:
            out.add(int(part))
    return sorted(out)


def nights_with_exposures(cdb, first_day_obs, last_day_obs):
    """Nights in ``[first, last]`` that have at least one LSSTCam exposure.

    Returns
    -------
    days : `list` [`int`]
    """
    q = (f'SELECT DISTINCT day_obs FROM cdb_{INSTRUMENT}.exposure '
         f'WHERE day_obs >= {int(first_day_obs)} AND day_obs <= {int(last_day_obs)} '
         f'ORDER BY day_obs')
    return [int(d) for d in cdb.query(q).to_pandas()['day_obs']]


def visit_spine(cdb, day_obs):
    """The identity block plus the inputs the fetchers need, for one night.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
    day_obs : `int`

    Returns
    -------
    df : `pandas.DataFrame`
        One row per exposure: ``visit_id``, ``day_obs``, ``seq_num``, ``obs_start`` (TAI
        ISO-8601), ``mjd`` [days, TAI], ``azimuth_deg`` and ``wind_dir_deg`` /
        ``wind_speed_ms`` [deg, m/s] from the ConsDB weather station. Ordered by
        ``seq_num``, which the Tweak and hexapod-history derivations rely on.

    Notes
    -----
    This ConsDB read is fast, and the EFD fetchers need ``obs_start`` for their as-of
    lookups regardless, so the spine is re-read per night rather than stored.
    """
    q = (f'SELECT exposure_id AS visit_id, day_obs, seq_num, obs_start, obs_start_mjd, '
         f'azimuth, wind_dir, wind_speed '
         f'FROM cdb_{INSTRUMENT}.exposure WHERE day_obs = {int(day_obs)} '
         f'ORDER BY seq_num')
    df = cdb.query(q).to_pandas()
    if df.empty:
        return df
    df = df.rename(columns={'obs_start_mjd': 'mjd', 'azimuth': 'azimuth_deg',
                            'wind_dir': 'wind_dir_deg', 'wind_speed': 'wind_speed_ms'})
    for c in ('mjd', 'azimuth_deg', 'wind_dir_deg', 'wind_speed_ms'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['obs_start'] = df['obs_start'].astype(str)
    # aos_trim's fetchers take an astropy-Table-like object keyed on day_obs/seq_num.
    df['visit_id'] = df['visit_id'].astype('int64')
    return df


class _FitTable:
    """Minimal astropy-Table-like view of the spine, for the `aos_trim` fetchers.

    They test membership with ``.colnames`` and index columns with ``[name]``, which a
    DataFrame does not provide identically, so this wraps one rather than converting to a
    real `astropy.table.Table`.
    """

    def __init__(self, df):
        self._df = df

    @property
    def colnames(self):
        return list(self._df.columns)

    def __getitem__(self, key):
        return self._df[key].to_numpy()

    def __len__(self):
        return len(self._df)


# ---------------------------------------------------------------------------
# Group fetchers.  Each takes (spine, ctx) and returns a DataFrame carrying the
# identity block plus its own columns, or None when the group has nothing for the night.
# ---------------------------------------------------------------------------
def fetch_trim(spine, ctx):
    """Trim (accumulated AOS offset) per visit, ``dof0..49`` [µm, deg].

    Also stashes the source event ids on `ctx` so `derive_tweak_group` can distinguish
    "the loop applied no correction" (0.0) from "unknown" (NaN).
    """
    import aos_trim
    trim, info = aos_trim.fetch_aggregated_dof_for_visits(
        _FitTable(spine), efd_client=ctx['efd'], consdb_client=ctx['cdb'])
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    trim = np.asarray(trim, float)
    for i in range(efd_db.N_DOF):
        out[f'dof{i}'] = trim[:, i]
    ctx['trim'] = trim
    ctx['trim_event_ids'] = np.asarray(
        info.get('event_id', np.full(len(spine), np.nan)), float)
    return out


def fetch_lut(spine, ctx):
    """Hexapod LUT baseline per visit, ``lut_dof0..9`` [µm, deg].

    ``lut_dof0..4`` are the M2 hexapod (z, x, y, u, v) at salIndex 2 and ``lut_dof5..9``
    the camera hexapod at salIndex 1; z/x/y are µm and u/v are deg.
    """
    import aos_trim
    lut, _info = aos_trim.fetch_hexapod_lut_for_visits(
        _FitTable(spine), efd_client=ctx['efd'], consdb_client=ctx['cdb'])
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    lut = np.asarray(lut, float)
    for i in range(efd_db.N_HEX_LUT):
        out[f'lut_dof{i}'] = lut[:, i]
    ctx['lut'] = lut
    return out


def derive_tweak_group(spine, ctx):
    """Tweak per visit, ``tweak_dof0..49`` [µm, deg] — differenced from Trim.

    Requires ``trim`` earlier in the same pass; Trim is not re-read from the database,
    because the derivation also needs the per-visit source event ids, which are not stored.
    """
    from run_attach_telemetry import derive_tweak
    if 'trim' not in ctx:
        raise RuntimeError("group 'tweak' requires 'trim' in the same pass; "
                           "use --groups trim,tweak")
    tweak = derive_tweak(ctx['trim'], ctx['trim_event_ids'])
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    for i in range(efd_db.N_DOF):
        out[f'tweak_dof{i}'] = tweak[:, i]
    return out


def fetch_gradients(spine, ctx):
    """M1M3 bulk thermal gradients per visit [°C/m], one night only.

    Notes
    -----
    A multi-night thermocouple query times out, so this must be called per night. The
    column names are prefixed ``m1m3_`` and suffixed with their units here, while
    `olr.telemetry.GRAD_COLS` uses the bare ``x_gradient`` form.
    """
    import telemetry as olr_tel
    if olr_tel.ThermocoupleAnalysis is None:
        raise RuntimeError('lsst.ts.m1m3.utils.ThermocoupleAnalysis is unavailable; '
                           'the gradients group needs the AOS stack environment')
    data = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    data = olr_tel._run_coro(olr_tel.get_m1m3_gradients(ctx['efd'], data))
    ren = {f'{n}_gradient': f'm1m3_{n}_gradient_c_per_m'
           for n in ('x', 'y', 'z', 'radial')}
    return data.rename(columns=ren)


def fetch_camera_group(spine, ctx):
    """Camera-body temperatures per visit, ``cam_<field>`` [°C] plus ``cam_n_samp``.

    Notes
    -----
    Delegates to ``run_attach_telemetry.fetch_camera``, which builds its own EFD client
    **inside** the coroutine — aiohttp binds a session to the running event loop, so a
    client made outside cannot be used within it.
    """
    from run_attach_telemetry import fetch_camera
    keys = spine[['day_obs', 'seq_num', 'mjd']].copy()
    cam = fetch_camera(keys, verbose=False)
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].merge(
        cam, on=['day_obs', 'seq_num'], how='left')
    return out


def fetch_turbulence(spine, ctx):
    """3D sonic anemometer speeds and temperatures per visit, salIndex 123–126.

    Returns one column per (salIndex, field) averaged over a window around each visit, in
    m/s and °C, plus ``turb<idx>_n_samp``.

    Notes
    -----
    These TMA top-ring sensors are absent from the transformed EFD of the ConsDB, which is
    why they are fetched here; salIndex 110 *is* in the transformed EFD and is read live
    instead.

    Averaged over ``PAD_SEC['air_turbulence']`` = 15 s either side of each exposure, not
    the 0.2 s used for the ConsDB-transformed wind: the raw topic publishes at a measured
    5.00 s cadence, so a 0.2 s window lands a sample on only about 4% of exposures.
    """
    import asyncio
    from astropy.time import Time
    from common.telemetry_clients import PAD_SEC

    pad = PAD_SEC.get('air_turbulence', 15.0) / 86400.0
    fields = list(efd_db.TURB_FIELDS)
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    mjd = spine['mjd'].to_numpy(float)

    async def _go():
        client = make_efd_client()
        t0 = Time(np.nanmin(mjd) - pad, format='mjd', scale='utc')
        t1 = Time(np.nanmax(mjd) + pad, format='mjd', scale='utc')
        try:
            df = await client.select_time_series(
                'lsst.sal.ESS.airTurbulence', ['salIndex'] + fields, t0, t1,
                convert_influx_index=True)
        except Exception as e:
            print(f'    turbulence: EFD query failed ({type(e).__name__}: {e}); '
                  f'NaN for {len(spine)} visits')
            df = pd.DataFrame()
        if len(df):
            df = df.copy()
            df['mjd'] = Time(df.index).utc.mjd
        for holder in list(vars(client).values()):
            sess = getattr(holder, '_session', None)
            if sess is not None and not getattr(sess, 'closed', True):
                await sess.close()
        return df

    df = asyncio.run(_go())
    for idx in efd_db.TURB_SALINDEX:
        sub = (df[pd.to_numeric(df['salIndex'], errors='coerce') == idx]
               if len(df) and 'salIndex' in df else pd.DataFrame())
        nsamp, cols = [], {suffix: [] for suffix, _u in efd_db.TURB_FIELDS.values()}
        for m in mjd:
            sl = (sub[(sub.mjd >= m - pad) & (sub.mjd <= m + pad)]
                  if len(sub) and np.isfinite(m) else pd.DataFrame())
            nsamp.append(len(sl))
            for field, (suffix, _u) in efd_db.TURB_FIELDS.items():
                if not len(sl) or field not in sl:
                    cols[suffix].append(np.nan)
                    continue
                x = pd.to_numeric(sl[field], errors='coerce').to_numpy(float)
                good = np.isfinite(x)
                cols[suffix].append(float(np.mean(x[good])) if good.any() else np.nan)
        for suffix, vals in cols.items():
            out[f'turb{idx}_{suffix}'] = vals
        out[f'turb{idx}_n_samp'] = nsamp
    return out


def derive_wind(spine, ctx):
    """Into-the-wind angle per visit [deg], 0 deg being pointed into the wind.

    ``wind_dir`` from the ConsDB weather station is the direction the wind comes *from*,
    so ``into_wind_deg = wrap180(wind_dir_deg - azimuth_deg)``. Both inputs are stored
    alongside the result for provenance, since the wrap convention is easy to get wrong.
    """
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start',
                 'azimuth_deg', 'wind_dir_deg', 'wind_speed_ms']].copy()
    out['into_wind_deg'] = efd_db.wrap180(
        out['wind_dir_deg'].to_numpy(float) - out['azimuth_deg'].to_numpy(float))
    return out


def derive_hexhist(spine, ctx):
    """Hexapod motion history within the night — a heating/hysteresis proxy.

    Returns
    -------
    df : `pandas.DataFrame`
        ``cum_hex_dz_um`` — cumulative ``|delta dz|`` [µm] since the start of the night;
        ``recent_hex_dz_um`` — the same over a trailing 30 minute window [µm];
        ``n_moves_night`` — count of commanded moves so far in the night.

    Notes
    -----
    The physical camera-hexapod dz is LUT + Trim, neither alone, so this uses
    ``lut_dof5 + dof5`` [µm] and requires both groups in the same pass. A move is counted
    where that sum changes by more than ``1e-6`` µm between consecutive exposures.
    """
    if 'lut' not in ctx or 'trim' not in ctx:
        raise RuntimeError("group 'hexhist' requires 'lut' and 'trim' in the same pass; "
                           "use --groups trim,lut,hexhist")
    dz = ctx['lut'][:, 5] + ctx['trim'][:, 5]           # camera hexapod dz, µm
    mjd = spine['mjd'].to_numpy(float)
    if not len(dz):
        return spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    d = np.abs(np.diff(dz, prepend=dz[0]))      # first exposure of the night: 0.0 by
    d = np.where(np.isfinite(d), d, 0.0)        # construction, not unknown
    moved = d > 1e-6
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    out['cum_hex_dz_um'] = np.cumsum(d)
    out['n_moves_night'] = np.cumsum(moved).astype('int64')
    win = 30.0 / (24.0 * 60.0)                          # 30 minutes, in days
    recent = np.full(len(dz), np.nan)
    for i, m in enumerate(mjd):
        if not np.isfinite(m):
            continue
        sel = np.isfinite(mjd) & (mjd <= m) & (mjd >= m - win)
        recent[i] = float(np.sum(d[sel]))
    out['recent_hex_dz_um'] = recent
    return out


#: Group name -> (callable, needs_efd). Order matters: `derive_tweak_group` and
#: `derive_hexhist` consume what `fetch_trim` / `fetch_lut` leave on the context.
FETCHERS = {
    'trim': (fetch_trim, True),
    'lut': (fetch_lut, True),
    'camera': (fetch_camera_group, False),
    'turbulence': (fetch_turbulence, False),
    'gradients': (fetch_gradients, True),
    'tweak': (derive_tweak_group, False),
    'wind_derived': (derive_wind, False),
    'hexhist': (derive_hexhist, False),
}


def build_night(con, cdb, efd, day_obs, groups, refetch=False, done=(), verbose=True):
    """Fetch and store the requested groups for one night.

    Parameters
    ----------
    con : `duckdb.DuckDBPyConnection`
        Writable connection.
    cdb : `lsst.summit.utils.ConsDbClient`
    efd : `lsst_efd_client.EfdClient`
    day_obs : `int`
    groups : `iterable` [`str`]
        Group names, in `efd_db.GROUP_ORDER` order.
    refetch : `bool`, optional
        Fetch groups already recorded ``ok``, overwriting them.
    done : `set` [`tuple`], optional
        ``(day_obs, group)`` pairs to skip, from `efd_db.done_pairs`.
    verbose : `bool`, optional

    Returns
    -------
    n_visits : `int`
        Exposures on the night, 0 if none.

    Notes
    -----
    A group that raises is logged with status ``error`` and leaves its columns NULL; the
    remaining groups for the night still run. Derived groups whose input group was skipped
    raise, which is deliberate — a silently wrong Tweak is worse than a missing one.
    """
    spine = visit_spine(cdb, day_obs)
    if spine.empty:
        if verbose:
            print(f'{day_obs}: no exposures', flush=True)
        for g in groups:
            efd_db.log_fetch(con, day_obs, g, 'empty', 0)
        return 0
    if verbose:
        print(f'{day_obs}: {len(spine)} exposures', flush=True)
    # The identity block alone first, so a night exists in the table even if every group
    # fails -- and, being identity-only, without nulling any group's columns.
    efd_db.upsert_visits(con, spine)
    ctx = {'cdb': cdb, 'efd': efd}
    for g in groups:
        if not refetch and (int(day_obs), g) in done:
            if verbose:
                print(f'    {g:13s} skipped (already ok)', flush=True)
            continue
        fn, _needs_efd = FETCHERS[g]
        try:
            df = fn(spine, ctx)
            n = efd_db.upsert_visits(con, df, g) if df is not None else 0
            efd_db.log_fetch(con, day_obs, g, 'ok' if n else 'empty', n)
            if verbose:
                print(f'    {g:13s} {n} rows', flush=True)
        except Exception as e:
            efd_db.log_fetch(con, day_obs, g, 'error', 0,
                             f'{type(e).__name__}: {e}')
            if verbose:
                print(f'    {g:13s} FAILED {type(e).__name__}: {e}', flush=True)
                traceback.print_exc(limit=3)
    return len(spine)


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--day-obs', required=True,
                   help='single night, inclusive range (20250415-20260913), or a '
                        'comma list of either')
    p.add_argument('--groups', default='all',
                   help="'all' or a comma list: " + ','.join(efd_db.GROUP_ORDER))
    p.add_argument('--resume', action='store_true',
                   help="skip (day_obs, group) pairs already recorded 'ok' or 'empty'")
    p.add_argument('--refetch', action='store_true',
                   help='fetch even pairs already recorded ok, overwriting them')
    p.add_argument('--db', default=None,
                   help='database file; default output/value_added/aos_efd.duckdb')
    p.add_argument('--consdb-url', default='auto')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)

    if a.groups == 'all':
        groups = list(efd_db.GROUP_ORDER)
    else:
        groups = [g.strip() for g in a.groups.split(',') if g.strip()]
        bad = [g for g in groups if g not in FETCHERS]
        if bad:
            p.error(f'unknown group(s) {bad}; choose from {list(efd_db.GROUP_ORDER)}')
        groups = [g for g in efd_db.GROUP_ORDER if g in groups]   # canonical order

    cdb = make_consdb_client(a.consdb_url)
    days = parse_day_obs(a.day_obs, cdb=cdb)
    if not days:
        p.error(f'no nights matched --day-obs {a.day_obs}')
    efd = make_efd_client()

    con = efd_db.open_db(a.db, create=True)
    done = efd_db.done_pairs(con) if a.resume else set()
    print(f'building {len(days)} night(s) {days[0]}..{days[-1]}, '
          f'groups {",".join(groups)}, db {a.db or efd_db.default_db_path()}')
    total = 0
    for day in days:
        total += build_night(con, cdb, efd, day, groups, refetch=a.refetch,
                             done=done, verbose=not a.quiet)
    n_cov = efd_db.refresh_coverage(con)
    n_rows = con.execute('SELECT COUNT(*) FROM visit_telemetry').fetchone()[0]
    con.close()
    print(f'\n{total} exposures this run; {n_rows} rows in visit_telemetry; '
          f'{n_cov} columns described in column_coverage')
    return 0


if __name__ == '__main__':
    sys.exit(main())
