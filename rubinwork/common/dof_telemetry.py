"""Per-visit degree-of-freedom (DOF) telemetry: look-up table, Trim and Tweak.

Three distinct 50-DOF quantities are served here, and they are **not**
interchangeable.  A physical hexapod position is LUT + Trim; neither alone is
the position.

=============== ==================== ==================================================
quantity        columns              source
=============== ==================== ==================================================
LUT (hexapod)   ``lut_dof0..9``      ``MTHexapod.logevent_compensationOffset`` -- a
                                     published position, read directly
LUT (mirror)    ``lut_dof10..49``    **derived**: no topic publishes mirror-LUT DOF, so
                                     the axial forces are converted to bending modes
Trim            ``dof0..49``         ``MTAOS.logevent_degreeOfFreedom.aggregatedDoF``
Tweak           ``tweak_dof0..49``   **derived**: ``Trim_i - Trim_(i-1)``
=============== ==================== ==================================================

The ordering and units of all three match the OFC DOF state (see
:data:`ofc_svd.LABELS_50DOF` / :data:`ofc_svd.DOF_UNITS_50`) -- µm for hexapod
translations, arcsec for rotations, dimensionless amplitude for bending modes --
so they can be compared with / added to FAM-recovered DOF directly.

**Trim** is the amount the MTAOS closed loop has moved each degree of freedom
away from the LUT baseline through closed-loop alignment.  Per-visit lookup
mirrors ``nightly_tablemaker`` / ``intrinsics_lib``: the authoritative exposure
start time ``obs_start`` (TAI) comes from ConsDB (``cdb_lsstcam.exposure``, keyed
by ``(day_obs, seq_num)``), and ``getMostRecentRowWithDataBefore`` returns the
DOF state in effect just before the exposure began.

**The mirror LUT has no published DOF value.**  Nothing in the SAL topic set
carries mirror-LUT bending amplitudes; the axial forces are the only record, so
``lut_dof10..29`` (M1M3) and ``lut_dof30..49`` (M2) exist only because
:func:`bending_modes_from_forces` derives them.  Two routes reach the same force
arrays and must therefore agree:

  - :func:`fetch_mirror_lut_for_visits` -- raw EFD, returning the 40 bending
    amplitudes directly.
  - :func:`fetch_lut_forces` -- the ConsDB transformed EFD, returning the raw
    axial forces in N for the caller to convert.

Those two are the same measurement at different stages, **not** interchangeable
outputs: one returns forces in N, the other dimensionless bending amplitudes.

The ``lsst.summit.utils`` and ``lsst.ts.ofc`` imports are done lazily so this
module imports cleanly without the LSST stack.
"""
from __future__ import annotations


import numpy as np

DOF_TOPIC = 'lsst.sal.MTAOS.logevent_degreeOfFreedom'
N_DOF = 50

# Client construction, endpoint selection and token handling live in
# common/telemetry_clients.py. The names are used throughout this module and re-exported
# unchanged, so that the aos/code/aos_trim.py shim can keep offering the surface untracked
# notebooks expect. New code should take them from common.telemetry_clients directly.
from .telemetry_clients import (
    IN_POD_CONSDB_URL, EXTERNAL_CONSDB_URL, DEFAULT_CONSDB_URL,
    DEFAULT_EXPOSURE_TABLE, in_rsp, make_efd_client, make_consdb_client, efd_window,
    PAD_SEC,
)

__all__ = [
    'DOF_TOPIC', 'N_DOF', 'DEFAULT_CONSDB_URL', 'IN_POD_CONSDB_URL',
    'EXTERNAL_CONSDB_URL', 'DEFAULT_EXPOSURE_TABLE', 'in_rsp',
    'make_efd_client', 'make_consdb_client', 'efd_window', 'PAD_SEC',
    'fetch_obs_start', 'fetch_aggregated_dof', 'fetch_aggregated_dof_for_visits',
    'fetch_hexapod_lut_for_visits', 'fetch_mirror_lut_for_visits',
    'LUT_PROPS', 'bending_modes_from_forces', 'fetch_lut_forces', 'derive_tweak',
]


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
# M1M3 elevation LUT: use the telemetry topic (high-rate); the logevent_ variant
# has no data in the EFD.
M1M3_ELEV_TOPIC = 'lsst.sal.MTM1M3.appliedElevationForces'
M2_AXIAL_TOPIC = 'lsst.sal.MTM2.axialForce'

# Mirror LUT array properties in the ConsDB efd_lsstcam.exposure_efd_unpivoted table:
# property -> (output prefix, axial field-name, array length). These are the same two
# force arrays that M1M3_ELEV_TOPIC / M2_AXIAL_TOPIC carry in the raw EFD.
LUT_PROPS = {
    'mt_m1m3_applied_elevation_forces_mean': ('m1m3elev', 'zForces', 156),
    'mt_m2_axial_force_lut_gravity_mean': ('m2grav', 'lutGravity', 72),
}


def bending_modes_from_forces(bmf, forces, n_mode=20):
    """Mirror bending-mode amplitudes from an axial-force array.

    This is the **definition** of the mirror look-up-table (LUT) degrees of freedom:
    no SAL topic publishes mirror-LUT DOF values, so ``lut_dof10..49`` exists only as
    the output of this conversion. Both the raw-EFD and the ConsDB route to the force
    arrays call this, so that the two agree.

    Parameters
    ----------
    bmf : `lsst.ts.ofc.BendModeToForce`
        Converter for the mirror in question, built as ``BendModeToForce('M1M3', ofc)``
        or ``BendModeToForce('M2', ofc)``.
    forces : `array_like`
        Axial forces in N: 156 for M1M3 (``zForces``), 72 for M2 (``lutGravity``).
        A non-finite entry is treated as 0 N -- see Notes.
    n_mode : `int`, optional
        Number of leading bending modes to return.

    Returns
    -------
    modes : `numpy.ndarray`
        Shape ``(n_mode,)`` bending-mode amplitudes, dimensionless. All-NaN if `forces`
        is None or holds no finite value at all, or if the conversion raises.

    Notes
    -----
    A single dropped actuator is routine on M1M3, so a partially-missing force array
    is treated as recoverable: the absent entries are filled with 0 N rather than
    poisoning all `n_mode` amplitudes with NaN. This is **not** what the hardware does
    -- the force-balance system redistributes the load of a failed actuator onto its
    neighbours, so a zero-filled conversion carries a small bias whose size has not
    been quantified. Modelling the redistribution is outstanding work, recorded in
    ``value_added/docs/status/build_progress.md``.

    Assumes (verify on the RSP) that the force arrays are in the same actuator order as
    the ts_ofc influence matrix, and that ``bending_mode`` returns at least `n_mode`
    modes per mirror. That assumption is what makes the derived DOF correct.
    """
    if forces is None:
        return np.full(n_mode, np.nan)
    forces = np.asarray(forces, dtype=float)
    if not np.isfinite(forces).any():
        return np.full(n_mode, np.nan)
    try:
        modes = bmf.bending_mode(np.nan_to_num(forces, nan=0.0))
    except Exception as e:
        print(f'(bending_mode failed [{type(e).__name__}])', flush=True)
        return np.full(n_mode, np.nan)
    modes = np.atleast_1d(np.asarray(modes, dtype=float))
    out = np.full(n_mode, np.nan)
    out[:min(n_mode, len(modes))] = modes[:n_mode]
    return out


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


def _tqdm(iterable, total=None, desc=None):
    """tqdm progress bar that degrades to a plain iterable if tqdm is absent."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _top1(efd_client, topic, columns, t, index=None):
    """Most recent single row of ``topic`` at/before ``t`` (fast influx LIMIT 1).

    Uses ``select_top_n(.., 1, time_cut=t)`` -- one row regardless of the topic's
    sample rate, so it is cheap even for high-rate force telemetry.
    """
    try:
        kw = {} if index is None else {'index': index}
        df = _run_coro(efd_client.select_top_n(topic, columns, 1, time_cut=t.utc, **kw))
        return df.iloc[0] if (df is not None and len(df)) else None
    except Exception as e:
        print(f'({topic} top1 failed [{type(e).__name__}: {e}])')
        return None


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
    fields = ['z', 'x', 'y', 'u', 'v']
    # One bulk select_time_series per night per hexapod (salIndex) + as-of match,
    # i.e. ~2 EFD queries per night instead of 2 per visit.  compensationOffset
    # is a sparse logevent, so keep a generous lookback buffer.
    m2_rows = _asof_rows_by_night(efd_client, HEX_LUT_TOPIC, fields, times,
                                  day_obs, index=2, buffer_hours=6.0)
    cam_rows = _asof_rows_by_night(efd_client, HEX_LUT_TOPIC, fields, times,
                                   day_obs, index=1, buffer_hours=6.0)
    lut = np.full((len(times), 10), np.nan)
    for i in range(len(times)):
        if m2_rows[i] is not None:
            lut[i, 0:5] = [m2_rows[i][k] for k in fields]
        if cam_rows[i] is not None:
            lut[i, 5:10] = [cam_rows[i][k] for k in fields]
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
    bending-mode amplitudes by :func:`bending_modes_from_forces` -- no topic
    publishes these DOF directly.  Columns map to OFC DOFs **dof10-29 (M1M3)**
    and **dof30-49 (M2)** -- same order/units as the aggregatedDoF Trim.
    Queried once per night (bulk + as-of; small buffer for the high-rate M2
    axialForce telemetry).

    M2 uses the gravity (elevation) LUT only; ``lutTemperature`` is available
    separately if the thermal LUT is also wanted.  See
    :func:`bending_modes_from_forces` for the actuator-order assumption and the
    treatment of dropped actuators.
    """
    from lsst.ts.ofc import OFCData, BendModeToForce

    if efd_client is None:
        efd_client = make_efd_client()
    ofc = OFCData('lsst', config_dir=config_dir)
    bmf_m1m3 = BendModeToForce('M1M3', ofc)
    bmf_m2 = BendModeToForce('M2', ofc)

    day_obs, times = _resolve_obs_times(fit_table, consdb_client, consdb_url,
                                        exposure_table, mjd_fallback_col, mjd_scale)
    zcols = [f'zForces{k}' for k in range(m1m3_n)]
    gcols = [f'lutGravity{k}' for k in range(m2_n)]

    # One bulk select_time_series per night per topic + as-of match, instead of
    # a per-visit backward search.  The per-visit path on the 156-col M1M3 and
    # high-rate 72-col M2 force topics is what blew the batch wall clock (a full
    # night of science visits x 2 queries each).  Both topics are continuous
    # telemetry, so a short lookback buffer suffices to find a prior sample.
    m1_rows = _asof_rows_by_night(efd_client, M1M3_ELEV_TOPIC, zcols, times,
                                  day_obs, buffer_hours=2.0)
    m2_rows = _asof_rows_by_night(efd_client, M2_AXIAL_TOPIC, gcols, times,
                                  day_obs, buffer_hours=2.0)
    lut = np.full((len(times), 40), np.nan)
    for i in range(len(times)):
        m1 = m1_rows[i]
        if m1 is not None:
            lut[i, 0:20] = bending_modes_from_forces(
                bmf_m1m3, m1[zcols].to_numpy(float))
        m2 = m2_rows[i]
        if m2 is not None:
            lut[i, 20:40] = bending_modes_from_forces(
                bmf_m2, m2[gcols].to_numpy(float))
    return lut, {'n_lut': int(np.isfinite(lut).any(axis=1).sum())}


def fetch_lut_forces(cdb, visit_ids):
    """Mirror LUT axial forces per visit from the ConsDB transformed EFD.

    The ConsDB route to the same two force arrays that
    :func:`fetch_mirror_lut_for_visits` reads from the raw EFD. Returns the forces
    themselves rather than bending modes, so the caller converts with
    :func:`bending_modes_from_forces`; the two functions are the same measurement at
    different stages and are **not** interchangeable outputs.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
        ConsDB client.
    visit_ids : `list` [`int`]
        Visit (exposure) ids.

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit`` plus ``<prefix>_<n>`` columns of axial force in N: 156 for M1M3
        elevation (``m1m3elev_*``), 72 for M2 gravity (``m2grav_*``). Missing visits
        are absent rather than NaN-filled.
    """
    import pandas as pd

    frames = []
    for prop, (prefix, field, n) in LUT_PROPS.items():
        rows = []
        for i in range(0, len(visit_ids), 400):
            part = visit_ids[i:i + 400]
            inl = ','.join(str(int(v)) for v in part)
            q = (f"SELECT exposure_id, field, value "
                 f"FROM efd_lsstcam.exposure_efd_unpivoted "
                 f"WHERE property='{prop}' AND exposure_id IN ({inl})")
            try:
                rows.append(cdb.query(q).to_pandas())
            except Exception as e:
                print(f'    {prefix} chunk {i}: {type(e).__name__}: {str(e)[:90]}')
        if not rows:
            continue
        d = pd.concat(rows, ignore_index=True)
        p = d.pivot_table(index='exposure_id', columns='field', values='value')
        # field names are e.g. zForces0..zForces155; order them numerically
        order = [f'{field}{k}' for k in range(n) if f'{field}{k}' in p.columns]
        p = p[order]
        p.columns = [f'{prefix}_{k}' for k in range(len(order))]
        frames.append(p.reset_index().rename(columns={'exposure_id': 'visit'}))
    if not frames:
        return pd.DataFrame(columns=['visit'])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on='visit', how='outer')
    return out


def derive_tweak(trim, event_ids):
    """Tweak per visit, differenced from Trim.

    Parameters
    ----------
    trim : `numpy.ndarray`
        Shape ``(n_visits, n_dof)`` Trim values, in the OFC DOF units (µm for hexapod
        translations, arcsec for rotations, dimensionless for bending amplitudes).
    event_ids : `numpy.ndarray`
        Shape ``(n_visits,)`` ``visitId`` of the source ``degreeOfFreedom`` event, NaN
        where none resolved. A change between consecutive visits marks a re-alignment.

    Returns
    -------
    tweak : `numpy.ndarray`
        Shape ``(n_visits, n_dof)``, same units as `trim`. **0.0** where the AOS applied
        no new correction between this visit and the previous one, since that is a real
        measurement of "no correction" rather than missing information. **NaN** only where
        the value is genuinely unknown: the first row (no predecessor), or where either
        visit's Trim or source event id could not be resolved.

    Notes
    -----
    Tweak has no EFD topic and no ConsDB property; ``Tweak = PID(optical_state)`` and
    ``Trim_(i+1) = Trim_i + Tweak``, so differencing Trim is the only route. The
    `event_ids` array that :func:`fetch_aggregated_dof_for_visits` returns under the
    ``'event_id'`` key is what distinguishes a real zero from an unknown, so the two
    functions must be used together.

    Where consecutive visits share one ``degreeOfFreedom`` event the difference is exactly
    zero by construction, and it is written as 0.0 rather than recomputed -- guarding
    against a float subtraction of two equal Trim values landing on a denormal instead of
    a clean zero.
    """
    trim = np.asarray(trim, dtype=float)
    ev = np.asarray(event_ids, dtype=float)
    tweak = np.full_like(trim, np.nan)
    for i in range(1, len(trim)):
        # Unknown Trim on either side -> genuinely unknown Tweak.
        if not (np.isfinite(trim[i]).any() and np.isfinite(trim[i - 1]).any()):
            continue
        if np.isfinite(ev[i]) and np.isfinite(ev[i - 1]) and ev[i] == ev[i - 1]:
            tweak[i] = 0.0            # loop ran, emitted no new correction
        else:
            tweak[i] = trim[i] - trim[i - 1]
    return tweak
