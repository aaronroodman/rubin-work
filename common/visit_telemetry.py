"""Per-visit environmental telemetry: dome wind and camera-body temperatures.

Two quantities that attach to exposures but belong to neither the Environmental Sensor
System (ESS) air-temperature set in ``common/ess_telemetry.py`` nor the degree-of-freedom
set in ``common/dof_telemetry.py``:

  - :func:`fetch_wind` -- inside-dome sonic-anemometer and outside-dome airflow readings
    from the Consolidated Database (ConsDB) transformed Engineering Facility Database
    (EFD). Present on 88.6% of Full Array Mode (FAM) exposures.
  - :func:`fetch_camera` -- camera-body and housing temperatures from the camera
    housekeeping topic, which the ConsDB transform does not carry at all, so this is
    raw-EFD only.

Import from the repo root::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))
    from common.visit_telemetry import fetch_wind, fetch_camera
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # repo root
from common.telemetry_clients import PAD_SEC    # noqa: E402

# Wind and airflow, from the ConsDB transformed EFD.
# ConsDB transformed-EFD column -> our name.
WIND_COLS = {
    'mt_salindex110_wind_speed_magnitude_mean': 'wind_speed_inside',
    'mt_salindex301_airflow_speed_mean': 'wind_speed_outside',
    'mt_salindex301_airflow_direction_mean': 'wind_dir_outside',
    'mt_salindex110_wind_speed_0_mean': 'wind_inside_x',
    'mt_salindex110_wind_speed_1_mean': 'wind_inside_y',
    'mt_salindex110_wind_speed_2_mean': 'wind_inside_z',
    'mt_salindex110_wind_speed_maxmagnitude_mean': 'wind_inside_maxmag',
    'mt_salindex110_sonic_temperature_mean': 'sonic_temperature',
}

# Camera-body temperatures come from the camera housekeeping measurement in its OWN
# InfluxDB database (lsst.MTCamera), not the main `efd` one, so they need a second EFD
# client. Values are deg C; a reading outside CAM_T_RANGE is a dropout, not a temperature.
CAM_TOPIC = 'lsst.MTCamera.utiltrunk_body'
CAM_DB = 'lsst.MTCamera'
CAM_T_RANGE = (-50.0, 60.0)
CAM_OK_STATE = 1.0                  # <field>_state == 1 marks a valid reading
CAM_FIELDS = [
    'AverageTemp',
    'CamBodyXPlusTemp', 'CamBodyYPlusTemp', 'CamBodyYMinusTemp',
    'CamHousXPlusTemp', 'CamHousXMinusTemp', 'CamHousYPlusTemp', 'CamHousYMinusTemp',
    'BackFlngXMinusTemp', 'BackFlngYMinusTemp',
    'ShrdRngXPlusTemp', 'ShrdRngXMinusTemp', 'ShrdRngYPlusTemp',
    'L1XMinusTemp', 'L1YMinusTemp', 'L2XPlusTemp', 'L2XMinusTemp', 'L2YPlusTemp',
    'DomeYMinusTemp', 'VPPlenumInTemp',
    'ShtrEboxRtnAirTemp', 'ShtrMtrRtnAirTemp', 'ChgrYMinusRtnAirTemp', 'AmbAirtemp',
]
# DomeXMinusTemp is deliberately absent: it read 0 of 3385 visits finite while its
# DomeYMinusTemp sibling read 98.6%, so the sensor is taken to be non-functional.

__all__ = [
    'WIND_COLS', 'CAM_TOPIC', 'CAM_DB', 'CAM_T_RANGE', 'CAM_OK_STATE', 'CAM_FIELDS',
    'fetch_wind', 'fetch_camera',
]


def fetch_wind(cdb, visit_ids):
    """Wind and airflow per visit from the ConsDB transformed EFD.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
        ConsDB client.
    visit_ids : `list` [`int`]
        Visit (exposure) ids.

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit`` plus the columns named in `WIND_COLS`; speeds in m/s, directions in deg,
        `sonic_temperature` in deg C. Missing visits are absent rather than NaN-filled.
    """
    out = []
    sel = ', '.join(WIND_COLS)
    for i in range(0, len(visit_ids), 700):
        part = visit_ids[i:i + 700]
        inl = ','.join(str(int(v)) for v in part)
        q = (f'SELECT exposure_id, {sel} FROM efd_lsstcam.exposure_efd '
             f'WHERE exposure_id IN ({inl})')
        try:
            out.append(cdb.query(q).to_pandas())
        except Exception as e:
            print(f'    wind chunk {i}: {type(e).__name__}: {str(e)[:90]}')
    if not out:
        return pd.DataFrame(columns=['visit'] + list(WIND_COLS.values()))
    df = pd.concat(out, ignore_index=True).rename(columns=WIND_COLS)
    return df.rename(columns={'exposure_id': 'visit'})


def fetch_camera(keys, pad_sec=None, require_state=False, verbose=True):
    """Camera-body temperatures per visit, averaged over a window around each visit.

    Parameters
    ----------
    keys : `pandas.DataFrame`
        Must carry ``day_obs``, ``seq_num`` and ``mjd``.
    pad_sec : `float`, optional
        Half-width of the averaging window in seconds; defaults to
        ``PAD_SEC['camera_body']``.
    require_state : `bool`, optional
        Keep only samples whose companion ``<field>_state`` equals `CAM_OK_STATE`. Off by
        default: the ``_state`` columns are empty in the EFD for the range checked
        (2026-07), so requiring them discards every sample.
    verbose : `bool`, optional
        Print a per-night row count.

    Returns
    -------
    df : `pandas.DataFrame`
        ``day_obs``, ``seq_num``, ``cam_n_samp`` (samples averaged) and ``cam_<field>``
        for each of `CAM_FIELDS`, in deg C. NaN where no valid sample fell in the window.

    Notes
    -----
    Queried once per night in bulk and sliced per visit, rather than one query per visit.
    Readings are kept when finite and inside `CAM_T_RANGE`; the ``<field>_state`` flag is
    consulted only if `require_state` is set.

    The EFD client is constructed **inside** the event loop: aiohttp binds its session to
    the running loop, so a client built before ``asyncio.run`` cannot be used within it.
    """
    import asyncio
    from astropy.time import Time
    from lsst_efd_client import EfdClient
    pad = (PAD_SEC['camera_body'] if pad_sec is None else pad_sec) / 86400.0
    tmin, tmax = CAM_T_RANGE
    v = keys.dropna(subset=['mjd']).copy()
    v['day_obs'] = v['day_obs'].astype(int)
    v['seq_num'] = v['seq_num'].astype(int)
    cols = CAM_FIELDS + ([f + '_state' for f in CAM_FIELDS] if require_state else [])
    out = {'day_obs': [], 'seq_num': [], 'cam_n_samp': []}
    for f in CAM_FIELDS:
        out['cam_' + f] = []

    async def _go():
        efd_cam = EfdClient('usdf_efd', db_name=CAM_DB)
        for day, g in v.groupby('day_obs'):
            t0 = Time(g.mjd.min() - pad, format='mjd', scale='utc')
            t1 = Time(g.mjd.max() + pad, format='mjd', scale='utc')
            try:
                df = await efd_cam.select_time_series(CAM_TOPIC, cols, t0, t1)
            except Exception as e:
                print(f'    camera {day}: EFD query failed ({type(e).__name__}); '
                      f'NaN for {len(g)} visits')
                df = pd.DataFrame()
            if len(df):
                df = df.copy()
                df['mjd'] = Time(df.index).utc.mjd
            for _, r in g.iterrows():
                out['day_obs'].append(int(r.day_obs))
                out['seq_num'].append(int(r.seq_num))
                if not len(df):
                    out['cam_n_samp'].append(0)
                    for f in CAM_FIELDS:
                        out['cam_' + f].append(np.nan)
                    continue
                sl = df[(df.mjd >= r.mjd - pad) & (df.mjd <= r.mjd + pad)]
                out['cam_n_samp'].append(int(len(sl)))
                for f in CAM_FIELDS:
                    if f not in sl or not len(sl):
                        out['cam_' + f].append(np.nan)
                        continue
                    x = pd.to_numeric(sl[f], errors='coerce').to_numpy(float)
                    good = np.isfinite(x) & (x >= tmin) & (x <= tmax)
                    st_col = f + '_state'
                    if require_state and st_col in sl:
                        st = pd.to_numeric(sl[st_col], errors='coerce').to_numpy(float)
                        good &= (st == CAM_OK_STATE)
                    out['cam_' + f].append(float(np.mean(x[good])) if good.any()
                                           else np.nan)
            if verbose:
                print(f'    camera {day}: {len(df)} EFD rows, {len(g)} visits')
        for holder in list(vars(efd_cam).values()):
            sess = getattr(holder, '_session', None)
            if sess is not None and not getattr(sess, 'closed', True):
                await sess.close()

    asyncio.run(_go())
    return pd.DataFrame(out)
