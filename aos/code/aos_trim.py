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
    """Return a ConsDB client (``lsst.summit.utils.ConsDbClient``)."""
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
    """Core: aggregatedDoF at each (UTC astropy Time or None) anchor."""
    from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore

    out = np.full((len(times_utc), n_dof), np.nan)
    for i, t in enumerate(times_utc):
        if t is None:
            continue
        try:
            ev = getMostRecentRowWithDataBefore(efd_client, topic,
                                                timeToLookBefore=t)
            out[i] = [ev[f'aggregatedDoF{k}'] for k in range(n_dof)]
        except Exception:
            continue
    return out


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
    return _dof_at_times(times, efd_client, topic=topic, n_dof=n_dof)


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

    trim = _dof_at_times(times, efd_client, topic=topic, n_dof=n_dof)
    info = {
        'n_obs_start': sum(s == 'obs_start' for s in src),
        'n_mjd_fallback': sum(s == 'mjd' for s in src),
        'n_dof': int(np.isfinite(trim).all(axis=1).sum()),
    }
    return trim, info
