#!/usr/bin/env python3
"""Build the value-added telemetry database, one night and one group at a time.

Fetches the slow Engineering Facility Database (EFD) quantities and the value-added
quantities derived from them into ``value_added/output/aos_efd.duckdb``, so that later
analysis never re-pays the EFD cost. Consolidated Database (ConsDB) columns are **not**
copied — ConsDB is fast and is read live via `efd_db.join_consdb`.

Usage
-----
One night, every group (do this first, before any backfill)::

    python value_added/code/build_efd_db.py --day-obs 20251102 --groups all

A range, resuming an interrupted run::

    python value_added/code/build_efd_db.py --day-obs 20250415-20260913 --groups all --resume

Add a newly-defined group over nights already built, touching nothing else::

    python value_added/code/build_efd_db.py --day-obs 20250415-20260913 --groups turbulence

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
    quadrants, as a ±15 s mean about ``obs_start``. Superseded by ``sonic``.
``wind_derived``
    Value-added: ``into_wind_deg = wrap180(wind_dir - azimuth)`` [deg], 0 deg being into
    the wind. Its two ConsDB inputs are stored alongside it, because the wrap convention
    is easy to get wrong.
``hexhist``
    Value-added: hexapod motion history within the night — cumulative and trailing-30-min
    ``|delta dz|`` [µm] and the commanded-move count. Requires ``lut`` and ``trim``.
``twilight``
    Value-added: minutes after evening twilight at 0 deg and at -18 deg solar altitude, and
    the solar altitude at the exposure [deg]. Needs no query of its own — it is a function of
    ``day_obs`` and the exposure time — but is stored so consumers do not each re-solve it.
``sonic``
    3D sonic anemometers on ``ESS.airTurbulence`` at salIndex 110 and 123–126, over the
    exposure itself (``obs_start`` to ``obs_end``): per-axis mean and RMS wind velocity
    [m/s], mean speed magnitude [m/s], sonic temperature [°C]. 110 and 123–126 are also in
    the ConsDB transformed EFD, which agreed to 0.0000 m/s median on two test nights; this
    group keeps the full span and the RMS, which ConsDB does not form.
``psf_shape``
    Value-added from ConsDB per-CCD PSF moments, rotated to the RubinTV Az/El frame: the
    mean and 10/25/50/75/90% points of ``|e|``, ``e1``, ``e2`` over CCDs, and raw
    two-point functions of ``e1e1``, ``e1e2``, ``e2e2`` at 2 and 6 CCD separations
    (`rubinwork.psf_shape`). Reads ConsDB only.
``louvers``
    Dome louver opening over the exposure, ``MTDome.louvers`` ``positionActual0..33``
    [percent open]: louvers more than 50 percent open, the sum of positions / 100, and the
    names of the open louvers (A..N around the dome, rings 1 to 3).

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

_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))     # sibling builders

from rubinwork.products.value_added import reader as efd_db
from rubinwork.products.value_added import write_manifest
import rubinwork.common.utils as common_utils
from rubinwork.common.ess_telemetry import get_m1m3_gradients_sync
from rubinwork.common.telemetry_clients import make_consdb_client, make_efd_client

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
        ISO-8601), ``mjd`` and ``mjd_end`` [days, TAI, exposure start and end],
        ``azimuth_deg`` and ``wind_dir_deg`` /
        ``wind_speed_ms`` [deg, m/s] from the ConsDB weather station. Ordered by
        ``seq_num``, which the Tweak and hexapod-history derivations rely on.

    Notes
    -----
    This ConsDB read is fast, and the EFD fetchers need ``obs_start`` for their as-of
    lookups regardless, so the spine is re-read per night rather than stored.
    """
    q = (f'SELECT exposure_id AS visit_id, day_obs, seq_num, obs_start, obs_start_mjd, '
         f'obs_end_mjd, azimuth, wind_dir, wind_speed '
         f'FROM cdb_{INSTRUMENT}.exposure WHERE day_obs = {int(day_obs)} '
         f'ORDER BY seq_num')
    df = cdb.query(q).to_pandas()
    if df.empty:
        return df
    df = df.rename(columns={'obs_start_mjd': 'mjd', 'obs_end_mjd': 'mjd_end',
                            'azimuth': 'azimuth_deg',
                            'wind_dir': 'wind_dir_deg', 'wind_speed': 'wind_speed_ms'})
    for c in ('mjd', 'mjd_end', 'azimuth_deg', 'wind_dir_deg', 'wind_speed_ms'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['obs_start'] = df['obs_start'].astype(str)
    # The dof_telemetry fetchers take an astropy-Table-like object keyed on day_obs/seq_num.
    df['visit_id'] = df['visit_id'].astype('int64')
    return df


class _FitTable:
    """Minimal astropy-Table-like view of the spine, for the `dof_telemetry` fetchers.

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
    from rubinwork.common.dof_telemetry import fetch_aggregated_dof_for_visits
    trim, info = fetch_aggregated_dof_for_visits(
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
    from rubinwork.common.dof_telemetry import fetch_hexapod_lut_for_visits
    lut, _info = fetch_hexapod_lut_for_visits(
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
    from rubinwork.common.dof_telemetry import derive_tweak
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
    `common.ess_telemetry.GRAD_COLS` uses the bare ``x_gradient`` form.

    A night missing one of the four M1M3 cold-junction reference channels yields NaN in
    all four columns rather than failing the night; a missing `ThermocoupleAnalysis`
    means the wrong environment and does fail, since it would NaN out every night.
    """
    import rubinwork.common.ess_telemetry as ess
    if ess.ThermocoupleAnalysis is None:
        raise RuntimeError('lsst.ts.m1m3.utils.ThermocoupleAnalysis is unavailable; '
                           'the gradients group needs the AOS stack environment')
    data = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    data = get_m1m3_gradients_sync(ctx['efd'], data)
    ren = {f'{n}_gradient': f'm1m3_{n}_gradient_c_per_m'
           for n in ('x', 'y', 'z', 'radial')}
    return data.rename(columns=ren)


def fetch_camera_group(spine, ctx):
    """Camera-body temperatures per visit, ``cam_<field>`` [°C] plus ``cam_n_samp``.

    Notes
    -----
    Delegates to ``common.visit_telemetry.fetch_camera``, which builds its own EFD client
    **inside** the coroutine — aiohttp binds a session to the running event loop, so a
    client made outside cannot be used within it.
    """
    from rubinwork.common.visit_telemetry import fetch_camera
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
    The ConsDB transformed EFD now carries these sensors too, from 2026-02-16; when this
    group was written it did not. The ``sonic`` group supersedes this one.

    The window is centred on ``obs_start`` in UTC. The spine's ``mjd`` is ConsDB
    ``obs_start_mjd``, which is TAI, and the raw EFD index is UTC
    (``aos/docs/telemetry.md`` section 7). Reading the TAI MJD as UTC and also passing
    ``convert_influx_index=True`` put the window 74 s late in builds up to
    ``snapshots/20261016``.

    Averaged over ``PAD_SEC['air_turbulence']`` = 15 s either side of each exposure, not
    the 0.2 s used for the ConsDB-transformed wind: the raw topic publishes at a measured
    5.00 s cadence, so a 0.2 s window lands a sample on only about 4% of exposures.
    """
    import asyncio
    from astropy.time import Time
    from rubinwork.common.telemetry_clients import PAD_SEC

    pad = PAD_SEC.get('air_turbulence', 15.0) / 86400.0
    fields = list(efd_db.TURB_FIELDS)
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    mjd = spine['mjd'].to_numpy(float).copy()
    finite = np.isfinite(mjd)
    mjd[finite] = Time(mjd[finite], format='mjd', scale='tai').utc.mjd

    async def _go():
        client = make_efd_client()
        t0 = Time(np.nanmin(mjd) - pad, format='mjd', scale='utc')
        t1 = Time(np.nanmax(mjd) + pad, format='mjd', scale='utc')
        try:
            df = await client.select_time_series(
                'lsst.sal.ESS.airTurbulence', ['salIndex'] + fields, t0, t1,
                convert_influx_index=False)
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


def _close_efd_sessions(client):
    """Close the aiohttp sessions an EFD client holds; it has no close() of its own."""
    async def _close():
        for holder in list(vars(client).values()):
            sess = getattr(holder, '_session', None)
            if sess is not None and not getattr(sess, 'closed', True):
                await sess.close()
    return _close()


def sonic_window_stats(sub, t0, t1):
    """Wind statistics of one sensor's 5 s ``airTurbulence`` samples over a window.

    Parameters
    ----------
    sub : `pandas.DataFrame`
        One salIndex's samples, with ``mjd`` [days, UTC] and the raw fields.
    t0, t1 : `float`
        Window start and end [MJD, UTC].

    Returns
    -------
    out : `dict` [`str`, `float`]
        Keyed by the `efd_db.SONIC_STATS` suffixes.

    Notes
    -----
    Each sample is a 5 s mean (``speed<a>``) with its own standard deviation
    (``speedStdDev<a>``). The RMS about the window mean combines both:
    ``sqrt(mean(sigma^2) + var(means))``. That is the full variance of the velocity over
    the window when every sample covers the same length of time.

    A sample counts when its time stamp is within the window. A window shorter than the
    5 s cadence (a short exposure) can therefore hold no sample, and then the nearest one
    within 5 s of the window centre is used instead.
    """
    out = {k: np.nan for k in efd_db.SONIC_STATS}
    out['n_samp'] = 0
    if not len(sub) or not (np.isfinite(t0) and np.isfinite(t1)):
        return out
    m = sub['mjd'].to_numpy()
    sel = (m >= t0) & (m <= t1)
    if not sel.any():
        mid = 0.5 * (t0 + t1)
        j = int(np.argmin(np.abs(m - mid)))
        if abs(m[j] - mid) * 86400.0 > 5.0:
            return out
        sel = np.zeros(len(m), bool)
        sel[j] = True
    sl = sub[sel]
    out['n_samp'] = int(sel.sum())

    def col(name):
        return pd.to_numeric(sl[name], errors='coerce').to_numpy(float) \
            if name in sl else np.full(len(sl), np.nan)

    for a in range(3):
        mu, sd = col(f'speed{a}'), col(f'speedStdDev{a}')
        if np.isfinite(mu).any():
            out[f'mean_{a}_ms'] = float(np.nanmean(mu))
            var = np.nanvar(mu) + (np.nanmean(sd ** 2) if np.isfinite(sd).any() else 0.0)
            out[f'rms_{a}_ms'] = float(np.sqrt(var))
    for field, key in (('speedMagnitude', 'mag_mean_ms'),
                       ('sonicTemperature', 'sonic_temp_c')):
        v = col(field)
        if np.isfinite(v).any():
            out[key] = float(np.nanmean(v))
    return out


def fetch_sonic(spine, ctx):
    """Sonic-anemometer wind per visit over the exposure itself, salIndex 110 and 123-126.

    Returns ``sonic<idx>_<stat>`` for every `efd_db.SONIC_SALINDEX` and
    `efd_db.SONIC_STATS`: per-axis mean and RMS wind velocity [m/s], mean speed magnitude
    [m/s], sonic temperature [deg C] and the sample count.

    Notes
    -----
    The window is ``obs_start`` to ``obs_end`` converted from TAI to UTC, the time base
    of the raw EFD index. The ``turbulence`` group's ±15 s mean is kept alongside for
    continuity; this group is the one to use for wind during the exposure.
    """
    import asyncio
    from astropy.time import Time

    fields = (['salIndex'] + [f'speed{a}' for a in range(3)]
              + [f'speedStdDev{a}' for a in range(3)]
              + ['speedMagnitude', 'sonicTemperature'])
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    t0 = spine['mjd'].to_numpy(float).copy()
    t1 = spine['mjd_end'].to_numpy(float).copy()
    for t in (t0, t1):
        f = np.isfinite(t)
        t[f] = Time(t[f], format='mjd', scale='tai').utc.mjd
    pad = 10.0 / 86400.0

    async def _go():
        client = make_efd_client()
        try:
            df = await client.select_time_series(
                'lsst.sal.ESS.airTurbulence', fields,
                Time(np.nanmin(t0) - pad, format='mjd', scale='utc'),
                Time(np.nanmax(t1) + pad, format='mjd', scale='utc'),
                convert_influx_index=False)
        finally:
            await _close_efd_sessions(client)
        if len(df):
            df = df.copy()
            df['mjd'] = Time(df.index).utc.mjd
        return df

    df = asyncio.run(_go())
    sal = pd.to_numeric(df['salIndex'], errors='coerce') if len(df) else None
    for idx in efd_db.SONIC_SALINDEX:
        sub = df[sal == idx].sort_values('mjd') if len(df) else pd.DataFrame()
        rows = [sonic_window_stats(sub, a, b) for a, b in zip(t0, t1)]
        for k in efd_db.SONIC_STATS:
            out[f'sonic{idx}_{k}'] = [r[k] for r in rows]
    return out


def louver_window_stats(pos, m, t0, t1, max_gap_s=60.0):
    """Louver opening over one window from the ``MTDome.louvers`` samples.

    Parameters
    ----------
    pos : `numpy.ndarray`
        Shape ``(n_samp, 34)``, louver positions [percent open], telemetry order.
    m : `numpy.ndarray`
        Sample times [MJD, UTC], sorted.
    t0, t1 : `float`
        Window start and end [MJD, UTC].
    max_gap_s : `float`, optional
        With no sample inside the window, the last sample before it is used if it is
        within this many seconds of ``t0``.

    Returns
    -------
    out : `dict`
        ``louvers_n_open``, ``louvers_open_sum``, ``louvers_open_names``,
        ``louvers_n_samp``.

    Notes
    -----
    The telemetry arrives about every 1.5 s and the louvers take 30 s to travel fully, so a
    louver moving during the exposure gives a fractional ``louvers_n_open``.
    """
    out = {'louvers_n_open': np.nan, 'louvers_open_sum': np.nan,
           'louvers_open_names': None, 'louvers_n_samp': 0}
    if not len(m) or not (np.isfinite(t0) and np.isfinite(t1)):
        return out
    sel = (m >= t0) & (m <= t1)
    if not sel.any():
        j = int(np.searchsorted(m, t0, side='right')) - 1
        if j < 0 or (t0 - m[j]) * 86400.0 > max_gap_s:
            return out
        sel = np.zeros(len(m), bool)
        sel[j] = True
    p = pos[sel]
    ok = np.isfinite(p).all(axis=1)
    if not ok.any():
        return out
    p = p[ok]
    opened = p > efd_db.LOUVER_OPEN_PCT
    out['louvers_n_open'] = float(opened.sum(axis=1).mean())
    out['louvers_open_sum'] = float((p.sum(axis=1) / 100.0).mean())
    mid = opened[len(p) // 2]
    out['louvers_open_names'] = ','.join(n for n, o in zip(efd_db.LOUVER_NAMES, mid) if o)
    out['louvers_n_samp'] = int(len(p))
    return out


def fetch_louvers(spine, ctx):
    """Dome louver opening per visit over the exposure, ``MTDome.louvers``.

    Returns ``louvers_n_open`` (louvers more than `efd_db.LOUVER_OPEN_PCT` open),
    ``louvers_open_sum`` (sum of positions / 100, the equivalent number of fully open
    louvers), ``louvers_open_names`` and the sample count. Positions are
    ``positionActual0..33`` [percent open], named by `efd_db.LOUVER_NAMES`.

    Notes
    -----
    The window is ``obs_start`` to ``obs_end`` converted from TAI to UTC. NULL where the
    dome CSC published nothing within 60 s before the exposure.
    """
    import asyncio
    from astropy.time import Time

    n = len(efd_db.LOUVER_NAMES)
    fields = [f'positionActual{i}' for i in range(n)]
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    t0 = spine['mjd'].to_numpy(float).copy()
    t1 = spine['mjd_end'].to_numpy(float).copy()
    for t in (t0, t1):
        f = np.isfinite(t)
        t[f] = Time(t[f], format='mjd', scale='tai').utc.mjd
    pad = 120.0 / 86400.0

    async def _go():
        client = make_efd_client()
        try:
            df = await client.select_time_series(
                'lsst.sal.MTDome.louvers', fields,
                Time(np.nanmin(t0) - pad, format='mjd', scale='utc'),
                Time(np.nanmax(t1) + pad, format='mjd', scale='utc'),
                convert_influx_index=False)
        finally:
            await _close_efd_sessions(client)
        if len(df):
            df = df.copy()
            df['mjd'] = Time(df.index).utc.mjd
            df = df.sort_values('mjd')
        return df

    df = asyncio.run(_go())
    if len(df):
        pos = df[fields].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        m = df['mjd'].to_numpy(float)
    else:
        pos, m = np.empty((0, n)), np.empty(0)
    rows = [louver_window_stats(pos, m, a, b) for a, b in zip(t0, t1)]
    for k in ('louvers_n_open', 'louvers_open_sum', 'louvers_open_names',
              'louvers_n_samp'):
        out[k] = [r[k] for r in rows]
    return out


def fetch_psf_shape(spine, ctx):
    """Per-visit PSF ellipticity statistics from the ConsDB per-CCD moments, Az/El frame.

    See `rubinwork.psf_shape` for the frame and `rubinwork.psf_shape.visit_summary` for
    the statistics. Visits with no per-CCD moments are returned with NaN.
    """
    from rubinwork import psf_shape

    cdb = ctx['cdb']
    day = int(spine['day_obs'].iloc[0])
    q = (f'SELECT c.visit_id, c.detector, q.psf_ixx, q.psf_iyy, q.psf_ixy '
         f'FROM cdb_{INSTRUMENT}.ccdvisit1 c '
         f'JOIN cdb_{INSTRUMENT}.ccdvisit1_quicklook q ON q.ccdvisit_id = c.ccdvisit_id '
         f'JOIN cdb_{INSTRUMENT}.visit1 v ON v.visit_id = c.visit_id '
         f'WHERE v.day_obs = {day}')
    ccd = cdb.query(q).to_pandas()
    rot = cdb.query(
        f'SELECT q.visit_id, q.physical_rotator_angle '
        f'FROM cdb_{INSTRUMENT}.visit1_quicklook q '
        f'JOIN cdb_{INSTRUMENT}.visit1 v ON v.visit_id = q.visit_id '
        f'WHERE v.day_obs = {day}').to_pandas()
    rot = dict(zip(rot['visit_id'].astype('int64'),
                   pd.to_numeric(rot['physical_rotator_angle'], errors='coerce')))
    if 'pos' not in ctx:
        ctx['pos'] = psf_shape.detector_field_angles()
    pos = ctx['pos']
    for c in ('psf_ixx', 'psf_iyy', 'psf_ixy'):
        ccd[c] = pd.to_numeric(ccd[c], errors='coerce')
    ccd = ccd[ccd['detector'].astype(int).isin(pos.keys())]
    ccd['visit_id'] = ccd['visit_id'].astype('int64')
    rows = []
    for vid, g in ccd.groupby('visit_id'):
        r = rot.get(vid, np.nan)
        if not np.isfinite(r):
            continue
        e1, e2 = psf_shape.moments_to_azel(g['psf_ixx'].to_numpy(), g['psf_iyy'].to_numpy(),
                                           g['psf_ixy'].to_numpy(), r)
        xy = np.array([pos[int(d)] for d in g['detector']])
        s = psf_shape.visit_summary(e1, e2, xy[:, 0], xy[:, 1])
        row = {'visit_id': vid, 'psf_n_ccd': s['n_ccd'], 'psf_rot_deg': r}
        for qn in ('e', 'e1', 'e2'):
            for st in ('mean', 'p10', 'p25', 'p50', 'p75', 'p90'):
                row[f'psf_{qn}_azel_{st}'] = s[f'{qn}_{st}']
        for k in psf_shape.XI_SCALES_CCD:
            for c in ('e1e1', 'e1e2', 'e2e2'):
                row[f'psf_xi_{c}_r{k}'] = s[f'xi_{c}_r{k}']
            row[f'psf_xi_npair_r{k}'] = s[f'xi_npair_r{k}']
        rows.append(row)
    res = pd.DataFrame(rows)
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']]
    if res.empty:
        return None
    return out.merge(res, on='visit_id', how='left')


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


def derive_twilight(spine, ctx):
    """Time of each exposure relative to the night's evening twilight.

    Returns
    -------
    df : `pandas.DataFrame`
        ``min_after_twilight`` — minutes after geometric sunset, the 0 deg solar altitude an
        observer times the start of the night against [min]; ``min_after_twilight_18deg`` — the
        same against astronomical twilight at -18 deg [min]; ``sun_alt_deg`` — the Sun's
        apparent altitude at the exposure start [deg].

    Notes
    -----
    Derived, not fetched: this is a function of `day_obs` and the exposure time alone, so it
    needs no EFD or ConsDB query beyond the spine already in hand. It is stored anyway because
    every consumer otherwise recomputes it, and because the window used to bracket the evening
    crossing is easy to get wrong — see `common.utils.evening_twilight_mjd`, which solves it.

    The twilight crossing is solved once per night and subtracted from each exposure, so every
    exposure on a night shares one reference epoch.

    The spine's ``mjd`` is TAI while the solar ephemeris is evaluated against UTC, so these
    columns carry the current 37 s TAI-minus-UTC offset. That is 0.6 per cent of a minute and
    these are reported in minutes, so it is left uncorrected — but do not read sub-minute
    meaning into them.
    """
    out = spine[['visit_id', 'day_obs', 'seq_num', 'obs_start']].copy()
    mjd = spine['mjd'].to_numpy(float)
    night = spine['day_obs'].to_numpy(int)
    for col, alt in (('min_after_twilight', 0.0), ('min_after_twilight_18deg', -18.0)):
        out[col] = (mjd - common_utils.evening_twilight_mjd(night, alt_deg=alt)) * 1440.0
    out['sun_alt_deg'] = common_utils.sun_altitude_deg(mjd)
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
    'twilight': (derive_twilight, False),
    'sonic': (fetch_sonic, False),
    'psf_shape': (fetch_psf_shape, False),
    'louvers': (fetch_louvers, False),
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
                   help='database file; default value_added/output/aos_efd.duckdb')
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
    # Only connect to the EFD if a requested group actually reads it. The derived groups do
    # not, and the EFD does not resolve from a batch compute node, so a derive-only backfill
    # runs anywhere rather than failing on a client it would never use.
    efd = make_efd_client() if any(FETCHERS[g][1] for g in groups) else None

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
    # Last step of a build: the manifest is what puts it in the catalog.  A SCRATCH
    # database -- --db or AOS_EFD_DB pointing outside the catalog build directory -- gets
    # none, so a reference build can never be mistaken for a registered build.
    write_manifest.finish_build(
        a.db, build_options={'what': 'telemetry', 'day_obs': a.day_obs,
                             'groups': ','.join(groups)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
