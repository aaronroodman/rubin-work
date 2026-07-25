"""AOS Trim / Offset from the EFD.

The MTAOS closed loop accumulates per-DOF corrections relative to the
look-up-table (LUT) baseline; the running total is published as
``lsst.sal.MTAOS.logevent_degreeOfFreedom`` with fields
``aggregatedDoF0 .. aggregatedDoF49``.  This is the "Trim" (a.k.a. Offset):
the amount the AOS has moved each degree of freedom from the LUT value due
to closed-loop alignment.  Its 50-DOF ordering and units match the OFC DOF
state (see :data:`ofc_svd.LABELS_50DOF` / :data:`ofc_svd.DOF_UNITS_50`), so
it can be compared with / added to FAM-recovered DOF directly.

Per-visit lookup mirrors ``nightly_tablemaker`` / ``intrinsics_lib``: the
authoritative exposure start time ``obs_start`` (TAI) comes from ConsDB
(``cdb_lsstcam.exposure``, keyed by ``(day_obs, seq_num)``), and
``getMostRecentRowWithDataBefore`` returns the DOF state in effect just
before the exposure began.  The ``lsst.summit.utils`` imports are done
lazily so this module imports cleanly without the LSST stack.
"""
from __future__ import annotations

import numpy as np

DOF_TOPIC = 'lsst.sal.MTAOS.logevent_degreeOfFreedom'
N_DOF = 50
DEFAULT_CONSDB_URL = 'http://consdb-pq.consdb:8080/consdb'
DEFAULT_EXPOSURE_TABLE = 'cdb_lsstcam.exposure'

__all__ = [
    'DOF_TOPIC', 'N_DOF', 'DEFAULT_CONSDB_URL', 'DEFAULT_EXPOSURE_TABLE',
    'make_efd_client', 'make_consdb_client', 'fetch_obs_start',
    'fetch_aggregated_dof', 'fetch_aggregated_dof_for_visits',
]


def make_efd_client(efd_name='usdf_efd'):
    """Return an EFD client.

    ``makeEfdClient`` lives in ``lsst.summit.utils.efdUtils`` in current
    summit_utils (it used to be re-exported at the package top level);
    falls back to ``lsst_efd_client.EfdClient(efd_name)`` if neither is
    importable.
    """
    try:
        from lsst.summit.utils.efdUtils import makeEfdClient
        return makeEfdClient()
    except (ImportError, AttributeError):
        pass
    try:
        from lsst.summit.utils import makeEfdClient
        return makeEfdClient()
    except (ImportError, AttributeError):
        pass
    from lsst_efd_client import EfdClient
    return EfdClient(efd_name)


def make_consdb_client(url=DEFAULT_CONSDB_URL):
    """Return a ConsDB client (``lsst.summit.utils.ConsDbClient``).

    The internal ConsDB host (``*.consdb``) must bypass the RSP HTTP proxy,
    otherwise requests fail with ``502 Bad Gateway`` at the proxy.  This
    adds ``.consdb`` to ``$no_proxy`` if absent, matching nightly_tablemaker.
    """
    import os
    no_proxy = os.environ.get('no_proxy', '')
    if '.consdb' not in no_proxy:
        os.environ['no_proxy'] = (no_proxy + ',.consdb') if no_proxy else '.consdb'
    from lsst.summit.utils import ConsDbClient
    return ConsDbClient(url)


def fetch_obs_start(consdb_client, day_obs, seq_num,
                    exposure_table=DEFAULT_EXPOSURE_TABLE):
    """Exposure ``obs_start`` (TAI isot string) per visit, from ConsDB.

    Matched by ``(day_obs, seq_num)``; rows with no match are returned as
    None, aligned to the input order.
    """
    import pandas as pd

    day_obs = np.asarray(day_obs).astype(int)
    seq_num = np.asarray(seq_num).astype(int)
    day_list = ', '.join(str(d) for d in sorted(set(day_obs.tolist())))
    query = (f'SELECT e.day_obs, e.seq_num, e.obs_start '
             f'FROM {exposure_table} e '
             f'WHERE e.day_obs IN ({day_list}) '
             f'ORDER BY e.day_obs, e.seq_num')
    cdb = consdb_client.query(query).to_pandas()
    vi = pd.DataFrame({'day_obs': day_obs, 'seq_num': seq_num})
    vi = vi.merge(cdb[['day_obs', 'seq_num', 'obs_start']],
                  on=['day_obs', 'seq_num'], how='left')
    return [None if pd.isna(v) else str(v) for v in vi['obs_start'].values]


def _dof_at_times(times_utc, efd_client, topic=DOF_TOPIC, n_dof=N_DOF):
    """Core: aggregatedDoF + source event id at each anchor time.

    Returns ``(dof, event_ids)``: ``dof`` is (n, n_dof); ``event_ids`` is
    the ``visitId`` of the ``degreeOfFreedom`` event each anchor resolved
    to (NaN where none / unavailable).  A change in ``event_ids`` between
    consecutive visits marks an AOS re-alignment (Trim step).
    """
    from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore

    out = np.full((len(times_utc), n_dof), np.nan)
    event_ids = np.full(len(times_utc), np.nan)
    for i, t in enumerate(times_utc):
        if t is None:
            continue
        try:
            ev = getMostRecentRowWithDataBefore(efd_client, topic,
                                                timeToLookBefore=t)
            out[i] = [ev[f'aggregatedDoF{k}'] for k in range(n_dof)]
            try:
                event_ids[i] = float(ev.get('visitId', np.nan))
            except Exception:
                pass
        except Exception:
            continue
    return out, event_ids


def fetch_aggregated_dof(times_mjd, efd_client, scale='tai', topic=DOF_TOPIC,
                         n_dof=N_DOF):
    """Per-visit aggregated DOF (Trim) from the EFD, anchored on MJD times.

    Each visit uses the most-recent ``degreeOfFreedom`` event *before* its
    time.  ``scale`` is the MJD time scale ('tai' matches the ConsDB /
    obs_start convention).  Rows with no event found stay NaN.  Returns
    (n_visits, n_dof).
    """
    from astropy.time import Time

    times_mjd = np.asarray(times_mjd, dtype=float)
    times = [None if not np.isfinite(m)
             else Time(float(m), format='mjd', scale=scale).utc
             for m in times_mjd]
    dof, _ = _dof_at_times(times, efd_client, topic=topic, n_dof=n_dof)
    return dof


def fetch_aggregated_dof_for_visits(fit_table, efd_client=None,
                                    consdb_client=None,
                                    consdb_url=DEFAULT_CONSDB_URL,
                                    exposure_table=DEFAULT_EXPOSURE_TABLE,
                                    topic=DOF_TOPIC, n_dof=N_DOF,
                                    mjd_fallback_col='mjd', mjd_scale='tai'):
    """Per-visit aggregated DOF (Trim), anchored on the exposure ``obs_start``.

    The authoritative anchor is the ConsDB exposure ``obs_start`` (TAI),
    keyed by ``(day_obs, seq_num)`` — the same one nightly_tablemaker uses.
    Visits ConsDB can't match fall back to ``fit_table[mjd_fallback_col]``
    (scale ``mjd_scale``) if present.  Clients are created on demand.

    Returns ``(trim, info)`` where ``trim`` is (n_visits, n_dof) and
    ``info`` is a dict with ``n_obs_start`` / ``n_mjd_fallback`` / ``n_dof``
    (visits anchored by each source, and with a finite DOF result).
    """
    from astropy.time import Time

    if efd_client is None:
        efd_client = make_efd_client()
    day_obs = np.asarray(fit_table['day_obs']).astype(int)
    seq_num = np.asarray(fit_table['seq_num']).astype(int)
    n = len(day_obs)

    obs_start = [None] * n
    try:
        if consdb_client is None:
            consdb_client = make_consdb_client(consdb_url)
        obs_start = fetch_obs_start(consdb_client, day_obs, seq_num,
                                    exposure_table=exposure_table)
    except Exception as e:
        print(f'(ConsDB obs_start unavailable [{type(e).__name__}: {e}]; '
              f'falling back to {mjd_fallback_col!r})')

    mjd = (np.asarray(fit_table[mjd_fallback_col], dtype=float)
           if mjd_fallback_col in fit_table.colnames else np.full(n, np.nan))

    times, src = [], []
    for i in range(n):
        if obs_start[i] is not None:
            times.append(Time(obs_start[i], format='isot', scale='tai').utc)
            src.append('obs_start')
        elif np.isfinite(mjd[i]):
            times.append(Time(float(mjd[i]), format='mjd', scale=mjd_scale).utc)
            src.append('mjd')
        else:
            times.append(None)
            src.append('none')

    trim, event_ids = _dof_at_times(times, efd_client, topic=topic,
                                    n_dof=n_dof)
    info = {
        'n_obs_start': sum(s == 'obs_start' for s in src),
        'n_mjd_fallback': sum(s == 'mjd' for s in src),
        'n_dof': int(np.isfinite(trim).all(axis=1).sum()),
        'event_id': event_ids,
    }
    return trim, info


HEX_LUT_TOPIC = 'lsst.sal.MTHexapod.logevent_compensationOffset'
M1M3_ELEV_TOPIC = 'lsst.sal.MTM1M3.logevent_appliedElevationForces'
M2_AXIAL_TOPIC = 'lsst.sal.MTM2.axialForce'


def _run_coro(coro):
    """Run an async EFD coroutine from sync code (re-entrant under nest_asyncio)."""
    import asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _resolve_obs_times(fit_table, consdb_client, consdb_url, exposure_table,
                       mjd_fallback_col, mjd_scale):
    """Return (day_obs, times): times a list of astropy Time (UTC) or None."""
    from astropy.time import Time
    day_obs = np.asarray(fit_table['day_obs']).astype(int)
    seq_num = np.asarray(fit_table['seq_num']).astype(int)
    n = len(day_obs)
    obs_start = [None] * n
    try:
        if consdb_client is None:
            consdb_client = make_consdb_client(consdb_url)
        obs_start = fetch_obs_start(consdb_client, day_obs, seq_num,
                                    exposure_table=exposure_table)
    except Exception as e:
        print(f'(ConsDB obs_start unavailable [{type(e).__name__}: {e}])')
    mjd = (np.asarray(fit_table[mjd_fallback_col], dtype=float)
           if mjd_fallback_col in fit_table.colnames else np.full(n, np.nan))
    times = []
    for i in range(n):
        if obs_start[i] is not None:
            times.append(Time(obs_start[i], format='isot', scale='tai').utc)
        elif np.isfinite(mjd[i]):
            times.append(Time(float(mjd[i]), format='mjd', scale=mjd_scale).utc)
        else:
            times.append(None)
    return day_obs, times


def _asof_rows_by_night(efd_client, topic, columns, times, day_obs,
                        index=None, buffer_hours=6.0):
    """Most-recent topic row at/before each visit, querying ONCE per night.

    Bulk ``select_time_series`` over ``[min(obs) - buffer_hours, max(obs)]`` per
    night, then an as-of (backward) match per visit -- so the number of EFD
    queries is ~1 per night instead of one backward search per visit (which is
    ruinous for high-rate telemetry like MTM2.axialForce).  ``buffer_hours``
    should be a few hours for sparse logevents and small for high-rate topics.
    Returns a list (len n visits) of pandas Series or None.
    """
    import pandas as pd
    from astropy.time import TimeDelta
    out = [None] * len(times)
    nights = {}
    for i, (d, t) in enumerate(zip(day_obs, times)):
        if t is not None:
            nights.setdefault(int(d), []).append(i)
    buf = TimeDelta(buffer_hours * 3600.0, format='sec')
    tail = TimeDelta(60.0, format='sec')
    for d, idxs in nights.items():
        tvis = [times[i] for i in idxs]
        t0 = (min(tvis) - buf).utc
        t1 = (max(tvis) + tail).utc
        try:
            kw = {} if index is None else {'index': index}
            df = _run_coro(efd_client.select_time_series(
                topic, columns, t0, t1, convert_influx_index=True, **kw))
        except Exception as e:
            print(f'({topic} night {d} query failed [{type(e).__name__}: {e}])')
            continue
        if df is None or len(df) == 0:
            continue
        df = df.sort_index()
        di = pd.to_datetime(df.index, utc=True)
        for i in idxs:
            tt = pd.Timestamp(times[i].utc.datetime, tz='UTC')
            found = np.nonzero(np.asarray(di <= tt))[0]
            if len(found):
                out[i] = df.iloc[found[-1]]
    return out


def fetch_hexapod_lut_for_visits(fit_table, efd_client=None, consdb_client=None,
                                 consdb_url=DEFAULT_CONSDB_URL,
                                 exposure_table=DEFAULT_EXPOSURE_TABLE,
                                 mjd_fallback_col='mjd', mjd_scale='tai'):
    """Per-visit hexapod LUT (``MTHexapod.logevent_compensationOffset``).

    The compensation the hexapod applied from its LUT model (elevation / rotator
    / filter lookup) -- the 'total LUT' the ``aggregatedDoF`` Trim is measured
    *against*.  Queried once per night (bulk + as-of), anchored on obs_start.

    Returns ``(lut, info)`` with ``lut`` (n_visits, 10):
        dof0-4 = M2 hexapod (z, x, y, u, v)   [salIndex 2]
        dof5-9 = camera hex (z, x, y, u, v)   [salIndex 1]
    so ``lut[:, 5]`` is the camera-hexapod dz LUT (filter-dependent focus).
    z/x/y in micron; u/v in deg (angular axes may differ from the OFC arcsec
    convention, but dz is directly comparable to the Trim dof5).
    """
    if efd_client is None:
        efd_client = make_efd_client()
    day_obs, times = _resolve_obs_times(fit_table, consdb_client, consdb_url,
                                        exposure_table, mjd_fallback_col, mjd_scale)
    n = len(times)
    fields = ['z', 'x', 'y', 'u', 'v']
    cam = _asof_rows_by_night(efd_client, HEX_LUT_TOPIC, fields, times, day_obs,
                              index=1, buffer_hours=6.0)
    m2 = _asof_rows_by_night(efd_client, HEX_LUT_TOPIC, fields, times, day_obs,
                             index=2, buffer_hours=6.0)
    lut = np.full((n, 10), np.nan)
    for i in range(n):
        if m2[i] is not None:
            lut[i, 0:5] = [m2[i][k] for k in fields]
        if cam[i] is not None:
            lut[i, 5:10] = [cam[i][k] for k in fields]
    return lut, {'n_lut': int(np.isfinite(lut).all(axis=1).sum())}


def fetch_mirror_lut_for_visits(fit_table, config_dir=None, efd_client=None,
                                consdb_client=None, consdb_url=DEFAULT_CONSDB_URL,
                                exposure_table=DEFAULT_EXPOSURE_TABLE,
                                mjd_fallback_col='mjd', mjd_scale='tai',
                                m1m3_n=156, m2_n=72):
    """Per-visit M1M3 + M2 mirror LUT as bending-mode DOFs -> (n_visits, 40).

    The mirror LUT is stored as *forces*: M1M3 elevation LUT
    (``MTM1M3.logevent_appliedElevationForces.zForces``, 156 axial) and M2
    gravity LUT (``MTM2.axialForce.lutGravity``, 72 axial), converted to
    bending-mode amplitudes with ts_ofc ``BendModeToForce.bending_mode``.
    Columns map to OFC DOFs **dof10-29 (M1M3)** and **dof30-49 (M2)** -- same
    order/units as the aggregatedDoF Trim.  Queried once per night (bulk + as-of;
    small buffer for the high-rate M2 axialForce telemetry).

    NOTE (verify on the RSP): assumes the EFD force arrays are in the same
    actuator order as the ts_ofc influence matrix, and that ``bending_mode``
    returns >=20 modes per mirror.  M2 uses the gravity (elevation) LUT only;
    ``lutTemperature`` is available separately if the thermal LUT is also wanted.
    """
    from lsst.ts.ofc import OFCData, BendModeToForce

    if efd_client is None:
        efd_client = make_efd_client()
    ofc = OFCData('lsst', config_dir=config_dir)
    bmf_m1m3 = BendModeToForce('M1M3', ofc)
    bmf_m2 = BendModeToForce('M2', ofc)

    day_obs, times = _resolve_obs_times(fit_table, consdb_client, consdb_url,
                                        exposure_table, mjd_fallback_col, mjd_scale)
    n = len(times)
    zcols = [f'zForces{k}' for k in range(m1m3_n)]
    gcols = [f'lutGravity{k}' for k in range(m2_n)]
    # M1M3 elevation forces: sparse logevent -> few-hour buffer.
    m1 = _asof_rows_by_night(efd_client, M1M3_ELEV_TOPIC, zcols, times, day_obs,
                             buffer_hours=6.0)
    # M2 axialForce: high-rate telemetry -> small buffer (always has a sample).
    m2 = _asof_rows_by_night(efd_client, M2_AXIAL_TOPIC, gcols, times, day_obs,
                             buffer_hours=0.2)

    def _to20(vec):
        v = np.atleast_1d(np.asarray(vec, float))
        out = np.full(20, np.nan)
        out[:min(20, len(v))] = v[:20]
        return out

    lut = np.full((n, 40), np.nan)
    for i in range(n):
        if m1[i] is not None:
            zf = np.array([m1[i][c] for c in zcols], float)
            try:
                lut[i, 0:20] = _to20(bmf_m1m3.bending_mode(zf))
            except Exception as e:
                print(f'(M1M3 bending_mode failed visit {i} [{type(e).__name__}])')
        if m2[i] is not None:
            gf = np.array([m2[i][c] for c in gcols], float)
            try:
                lut[i, 20:40] = _to20(bmf_m2.bending_mode(gf))
            except Exception as e:
                print(f'(M2 bending_mode failed visit {i} [{type(e).__name__}])')
    return lut, {'n_lut': int(np.isfinite(lut).any(axis=1).sum())}
