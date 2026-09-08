"""Shared EFD and ConsDB client construction, and EFD query-window helpers.

Every topic in this repository that reads telescope telemetry needs the same three
things: an Engineering Facility Database (EFD) client, a Consolidated Database (ConsDB)
client pointed at an endpoint that actually resolves from wherever the code is running,
and a time window in which to query. Before this module those were reimplemented per
topic — 16 files constructed their own client, 12 hardcoded a ConsDB URL literal in one of
two forms, and 5 reimplemented token-file handling.

Import from the repo root::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))
    from common.telemetry_clients import make_consdb_client, make_efd_client

Notes
-----
The two ConsDB endpoints are not interchangeable. ``consdb-pq.consdb`` resolves **only**
inside the Rubin Science Platform (RSP) Nublado pod; from an S3DF login or batch node it
fails with a DNS error. Use ``url='auto'`` unless there is a reason not to.

Two ConsDB server bugs to route around, both returning HTTP 500:

* ``WHERE <col> LIKE 'prefix%'`` — use exact ``=`` and filter client-side.
* a cross-database ``LEFT JOIN`` between ``cdb_lsstcam`` and ``efd_lsstcam`` — issue two
  queries instead. Same-database joins are fine.
"""
import os
from pathlib import Path

# In-pod host (resolves only inside the RSP Nublado pod) versus the public RSP endpoint
# (token-injected, works from S3DF/sdfiana too).
IN_POD_CONSDB_URL = 'http://consdb-pq.consdb:8080/consdb'
EXTERNAL_CONSDB_URL = 'https://usdf-rsp.slac.stanford.edu/consdb'
DEFAULT_CONSDB_URL = IN_POD_CONSDB_URL      # kept for back-compat with existing callers
DEFAULT_EXPOSURE_TABLE = 'cdb_lsstcam.exposure'
DEFAULT_EFD_NAME = 'usdf_efd'

# EFD query padding, in seconds, by the character of the quantity. These were four
# unreconciled literals spread across four files (0.2 s in olr/telemetry.py and
# olr/nightly_table.py, 120 s in run_backfill_camera_telemetry.py, a 60 s tail in
# aos_trim.py) -- a 600x spread applied to the same job. Collected here so a padding
# choice is a named, reviewable decision rather than a buried constant.
#
# The distinction that matters is the publication rate of the topic relative to the
# exposure length: a slow quantity needs the window widened to catch the nearest sample,
# a fast one does not.
PAD_SEC = {
    'ess_temperature': 0.2,      # ESS temps: ConsDB transform already averages per exposure
    'wind': 0.2,                 # as above
    'm1m3_thermocouple': 60.0,   # ~0.05 Hz array; widen to land a sample
    'camera_body': 120.0,        # utility-trunk housekeeping, slowest of the set
    'dof_event': 60.0,           # MTAOS logevent: as-of lookup, tail only
    'mirror_force': 5.0,         # high-rate M2 axialForce; a small buffer suffices
}

__all__ = [
    'IN_POD_CONSDB_URL', 'EXTERNAL_CONSDB_URL', 'DEFAULT_CONSDB_URL',
    'DEFAULT_EXPOSURE_TABLE', 'DEFAULT_EFD_NAME', 'PAD_SEC',
    'in_rsp', 'make_efd_client', 'make_consdb_client', 'efd_window',
]


def in_rsp():
    """True when running inside the RSP (Nublado) JupyterLab pod.

    Returns
    -------
    inside : `bool`
        Detected via the ``/etc/nublado`` marker directory the Nublado spawner mounts
        into every RSP pod; absent on S3DF login/batch nodes (sdfiana, slacrd) and on a
        laptop.
    """
    return os.path.isdir('/etc/nublado')


def make_efd_client(efd_name=DEFAULT_EFD_NAME):
    """Return an EFD client.

    Parameters
    ----------
    efd_name : `str`, optional
        EFD instance name, used only by the bare-``lsst_efd_client`` fallback.

    Returns
    -------
    client : `lsst_efd_client.EfdClient`
        A connected client.

    Notes
    -----
    ``makeEfdClient`` lives in `lsst.summit.utils.efdUtils` in current summit_utils; it
    used to be re-exported at the package top level. Both spellings are tried before
    falling back to constructing `lsst_efd_client.EfdClient` directly.
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


def make_consdb_client(url='auto', token_file=None):
    """Return a ConsDB client, with the endpoint and token resolved for this environment.

    Parameters
    ----------
    url : `str`, optional
        ``'auto'`` (the default) picks the in-pod host inside the RSP and the external
        token-injected endpoint elsewhere, via `in_rsp`. Pass an explicit URL to force
        one.
    token_file : `str` or `pathlib.Path`, optional
        Override the token path, default ``~/.lsst/consdb_token``.

    Returns
    -------
    client : `lsst.summit.utils.ConsDbClient`
        A client for the resolved endpoint.

    Notes
    -----
    The two access modes:

    * **In-pod** — ``consdb-pq.consdb`` resolves only inside the RSP JupyterLab (Nublado)
      pod, and must bypass the RSP HTTP proxy or it returns ``502 Bad Gateway``;
      ``.consdb`` is appended to ``$no_proxy`` here. No token needed.
    * **External / S3DF** — the internal host does not resolve from an S3DF login or
      batch node, so the public endpoint is used with an RSP access token injected as
      ``https://user:<token>@host/consdb``.

    The token **file** is preferred over the ``ACCESS_TOKEN`` environment variable: it is
    read at call time, so a long-queued batch job still picks up a current token, whereas
    a value frozen into the job environment by ``--export=ALL`` may have expired by the
    time the job runs (401 Unauthorized).
    """
    if url == 'auto':
        url = IN_POD_CONSDB_URL if in_rsp() else EXTERNAL_CONSDB_URL
    no_proxy = os.environ.get('no_proxy', '')
    if '.consdb' not in no_proxy:
        os.environ['no_proxy'] = (no_proxy + ',.consdb') if no_proxy else '.consdb'
    if '@' not in url and 'consdb-pq.consdb' not in url:
        tf = Path(token_file) if token_file else Path.home() / '.lsst' / 'consdb_token'
        token = tf.read_text().strip() if tf.exists() else os.environ.get('ACCESS_TOKEN')
        if token:
            url = url.replace('://', f'://user:{token}@', 1)
    from lsst.summit.utils import ConsDbClient
    return ConsDbClient(url)


def efd_window(obs_start, obs_end=None, kind='ess_temperature',
               pre_sec=None, post_sec=None, scale='tai'):
    """EFD query window around an exposure, as a (begin, end) pair of `astropy.time.Time`.

    Parameters
    ----------
    obs_start : `str` or `astropy.time.Time`
        Exposure start. A string is parsed as ISOT in `scale` — this is the ConsDB
        ``obs_start`` convention.
    obs_end : `str` or `astropy.time.Time`, optional
        Exposure end. Defaults to `obs_start`, giving a window centred on the start.
    kind : `str`, optional
        Key into `PAD_SEC` selecting the padding appropriate to the topic's publication
        rate. Ignored where `pre_sec` or `post_sec` is given.
    pre_sec, post_sec : `float`, optional
        Explicit padding in seconds before `obs_start` and after `obs_end`, overriding
        `kind`.
    scale : `str`, optional
        Time scale for string inputs; ConsDB ``obs_start``/``obs_end`` are TAI.

    Returns
    -------
    begin : `astropy.time.Time`
        Window start, in UTC — what the EFD client expects.
    end : `astropy.time.Time`
        Window end, in UTC.

    Notes
    -----
    This exists because `lsst.summit.utils.efdUtils.getEfdData` builds its window from a
    Butler `~lsst.daf.butler.DimensionRecord`, and the code here anchors on ConsDB
    ``obs_start``/``obs_end`` instead, frequently with no Butler client available. Prefer
    ``getEfdData`` for a genuine single-window query where a record is in hand; use this
    when working from ConsDB times, and query **once per night in bulk** rather than once
    per visit when there are thousands of visits.
    """
    from astropy.time import Time, TimeDelta
    t0 = obs_start if isinstance(obs_start, Time) else Time(str(obs_start),
                                                            format='isot', scale=scale)
    if obs_end is None:
        t1 = t0
    else:
        t1 = obs_end if isinstance(obs_end, Time) else Time(str(obs_end),
                                                            format='isot', scale=scale)
    pad = PAD_SEC.get(kind, 0.2)
    pre = pad if pre_sec is None else pre_sec
    post = pad if post_sec is None else post_sec
    return (t0.utc - TimeDelta(pre, format='sec'),
            t1.utc + TimeDelta(post, format='sec'))
