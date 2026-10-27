"""Per-visit thermal history from EFD time series: rates of change, state at twilight.

For each night, the Environmental Sensor System (ESS) air temperatures (camera 111, M2
112, M1M3 113, outside 301), the mean of the two TMA truss channels (ESS 122 items 6 and
7) and the camera body average (``MTCamera.utiltrunk_body``) are read from 1.5 h before
evening twilight (0 deg solar altitude) to the last visit. Each series is median-binned
to 1 min and smoothed over 5 min, then for every visit gives:

- ``<s>_rate30``, ``<s>_rate60``: change over the previous 30 or 60 min, in deg C/h;
- ``<s>_twi``: value at evening twilight, in deg C;
- ``<s>_since_twi``: visit value minus twilight value, in deg C.

Plus ``hours_after_twilight`` (h). Visit times are the ``mjd`` (TAI) of `cwfs_science`,
converted to UTC to match the EFD.

Run from this study's directory::

    python code/fetch_history.py
"""

import pathlib
import time

import numpy as np
import pandas as pd
from astropy.time import Time

from rubinwork.common.ess_telemetry import _run_coro
from rubinwork.common.telemetry_clients import make_efd_client
from rubinwork.common.utils import evening_twilight_mjd
from rubinwork.common.visit_telemetry import CAM_DB, CAM_T_RANGE, CAM_TOPIC
from rubinwork.products import cwfs_science

VARIANT = "d13-blitz_rso1071"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "output" / "cache" / "history_per_visit.parquet"
ESS_AIR = {"cam_air": 111, "m2_air": 112, "m1m3_air": 113, "outside": 301}
SENSORS = (*ESS_AIR, "truss", "cam_body")
PRE_TWILIGHT_H = 1.5
TAI_MINUS_UTC_D = 37.0 / 86400


def smooth(series):
    """1 min median bins, then a centred 5 min mean; index in MJD (UTC)."""
    s = series.dropna()
    if len(s) == 0:
        return None
    s = s.resample("1min").median().rolling(5, center=True, min_periods=2).mean().dropna()
    mjd = Time(s.index.to_pydatetime(), scale="utc").mjd
    return mjd, s.to_numpy()


async def night_series(efd, efd_cam, t0, t1):
    a, b = Time(t0, format="mjd", scale="utc"), Time(t1, format="mjd", scale="utc")
    out = {}
    for name, idx in ESS_AIR.items():
        d = await efd.select_time_series("lsst.sal.ESS.temperature", ["temperatureItem0"],
                                         a, b, index=idx)
        out[name] = smooth(d["temperatureItem0"]) if len(d) else None
    d = await efd.select_time_series("lsst.sal.ESS.temperature",
                                     ["temperatureItem6", "temperatureItem7"], a, b, index=122)
    out["truss"] = smooth(d[["temperatureItem6", "temperatureItem7"]].mean(axis=1,
                          skipna=False)) if len(d) else None
    # Range check only, as in visit_telemetry.fetch_camera; the _state flag is often empty.
    d = await efd_cam.select_time_series(CAM_TOPIC, ["AverageTemp"], a, b)
    if len(d):
        out["cam_body"] = smooth(d["AverageTemp"].where(d["AverageTemp"].between(*CAM_T_RANGE)))
    else:
        out["cam_body"] = None
    return out


def interp(series, t):
    if series is None:
        return np.full(np.shape(t), np.nan)
    mjd, val = series
    # No extrapolation past the series, and no bridging of gaps over 10 min.
    v = np.interp(t, mjd, val, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(mjd, t), 1, len(mjd) - 1)
    v[(mjd[j] - mjd[j - 1]) > 10 / 1440] = np.nan
    return v


def features(series, t, t_twi):
    row = {}
    for s in SENSORS:
        now = interp(series[s], t)
        for minutes in (30, 60):
            row[f"{s}_rate{minutes}"] = (now - interp(series[s], t - minutes / 1440)) * 60 / minutes
        twi = interp(series[s], np.array([t_twi]))[0]
        row[f"{s}_twi"] = np.full(len(t), twi)
        row[f"{s}_since_twi"] = now - twi
    return row


async def fetch():
    v = (cwfs_science.load_table("ccd_visits", VARIANT)[["visit", "day_obs", "mjd"]]
         .drop_duplicates("visit"))
    v["mjd_utc"] = v["mjd"] - TAI_MINUS_UTC_D
    nights = np.sort(v["day_obs"].unique())
    twi = dict(zip(nights, evening_twilight_mjd(nights)))
    from lsst_efd_client import EfdClient
    efd = make_efd_client()
    efd_cam = EfdClient("usdf_efd", db_name=CAM_DB)
    frames = []
    t_start = time.time()
    for k, night in enumerate(nights):
        sub = v[v["day_obs"] == night]
        t = sub["mjd_utc"].to_numpy()
        t0 = min(twi[night], t.min()) - PRE_TWILIGHT_H / 24
        try:
            series = await night_series(efd, efd_cam, t0, t.max() + 5 / 1440)
        except Exception as e:
            print(f"{night}: EFD failed [{type(e).__name__}: {e}]", flush=True)
            series = {s: None for s in SENSORS}
        f = pd.DataFrame(features(series, t, twi[night]), index=sub.index)
        f["visit"] = sub["visit"].to_numpy()
        f["hours_after_twilight"] = (t - twi[night]) * 24
        frames.append(f)
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{len(nights)} nights, {time.time() - t_start:.0f} s", flush=True)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = _run_coro(fetch())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"{len(df)} visits written to {CACHE}")
    print(df.notna().mean().round(3).to_string())
