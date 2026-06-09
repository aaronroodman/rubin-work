"""AOS Trim / Offset from the EFD.

The MTAOS closed loop accumulates per-DOF corrections relative to the
look-up-table (LUT) baseline; the running total is published as
``lsst.sal.MTAOS.logevent_degreeOfFreedom`` with fields
``aggregatedDoF0 .. aggregatedDoF49``.  This is the "Trim" (a.k.a. Offset):
the amount the AOS has moved each degree of freedom from the LUT value due
to closed-loop alignment.  Its 50-DOF ordering and units match the OFC DOF
state (see :data:`ofc_svd.LABELS_50DOF` / :data:`ofc_svd.DOF_UNITS_50`), so
it can be compared with / added to FAM-recovered DOF directly.

Mirrors the per-visit lookup used in ``nightly_tablemaker``:
``getMostRecentRowWithDataBefore`` returns the DOF state in effect when the
exposure was taken.  The ``lsst.summit.utils`` imports are done lazily so
this module imports cleanly without the LSST stack.
"""
from __future__ import annotations

import numpy as np

DOF_TOPIC = 'lsst.sal.MTAOS.logevent_degreeOfFreedom'
N_DOF = 50

__all__ = ['DOF_TOPIC', 'make_efd_client', 'fetch_aggregated_dof']


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


def fetch_aggregated_dof(times_mjd, efd_client, scale='tai', topic=DOF_TOPIC,
                         n_dof=N_DOF):
    """Per-visit aggregated DOF (Trim) from the EFD.

    Parameters
    ----------
    times_mjd : array-like (n_visits,)
        Visit timestamps as MJD.  For each, the most-recent
        ``degreeOfFreedom`` event *before* that time is used.
    efd_client : EFD client
        From :func:`make_efd_client`.
    scale : str
        Time scale of ``times_mjd`` ('tai' matches the ConsDB / obs_start
        convention used by nightly_tablemaker; flip to 'utc' if the
        visit-to-event matching looks off by a few seconds near a
        correction boundary).
    topic : str
    n_dof : int

    Returns
    -------
    trim : ndarray (n_visits, n_dof)
        ``aggregatedDoF`` per visit; rows with no event found stay NaN.
    """
    from astropy.time import Time
    from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore

    times_mjd = np.asarray(times_mjd, dtype=float)
    out = np.full((len(times_mjd), n_dof), np.nan)
    for i, mjd in enumerate(times_mjd):
        if not np.isfinite(mjd):
            continue
        t = Time(float(mjd), format='mjd', scale=scale).utc
        try:
            ev = getMostRecentRowWithDataBefore(efd_client, topic,
                                                timeToLookBefore=t)
            out[i] = [ev[f'aggregatedDoF{k}'] for k in range(n_dof)]
        except Exception:
            continue
    return out
