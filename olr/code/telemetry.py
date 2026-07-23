"""Shared EFD environmental-telemetry helpers for the AOS analyses.

Single source of truth for the thermal + wind telemetry attached per visit, so
the logic is not duplicated across ``nightly_table.py``, the block-T539
notebook, and the FAM summary build.  All functions are ``async`` (they await
the ``lsst_efd_client`` EfdClient).

Provided:
  - ``fetch_thermal_telemetry`` -- ESS air temperatures (camera / M2 / M1M3 /
    outside), the derived delta-Ts, the M1M3 bulk thermal gradients
    (x/y/z/radial), and the TMA truss temperatures.
  - ``fetch_dome_wind`` -- outside-dome wind speed/direction from
    ``lsst.sal.ESS.airFlow`` (weather tower) and inside-dome wind speed from
    ``lsst.sal.ESS.airTurbulence`` (TMA sonic anemometers).
  - ``get_m1m3_gradients`` -- M1M3 bulk thermal gradients interpolated onto each
    visit's ``obs_start`` (kept public; re-exported by ``nightly_table``).

Each visit is identified by ``(day_obs, seq)`` and its TAI ``obs_start`` /
``obs_end`` ISO strings, matching ``nightly_table.AOSDatabase``.
"""
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

# ThermocoupleAnalysis (M1M3 bulk gradients) lives in lsst.ts.m1m3.utils on the
# RSP stack; guard the import so this module can at least be imported off-stack.
try:  # pragma: no cover - stack-only
    from lsst.ts.m1m3.utils import ThermocoupleAnalysis
except Exception:  # pragma: no cover
    ThermocoupleAnalysis = None

# --- ESS salIndex -> physical air-temperature sensor (from the nightly report) --
ESS_TEMP_INDEX = {
    "outside_temp": 301,   # outside weather station
    "cam_air_temp": 111,   # camera
    "m2_air_temp": 112,    # M2
    "m1m3_air_temp": 113,  # air above M1M3
}

# --- TMA truss temperatures: ESS index 2, temperatureItem6/7 = +x+y / -x-y ------
TRUSS_ESS_INDEX = 2
TRUSS_ITEMS = {6: "tma_truss_temp_pxpy", 7: "tma_truss_temp_mxmy"}

# --- Wind: outside from ESS.airFlow, inside from ESS.airTurbulence ---------------
# Outside dome = the weather-tower anemometer (ESS.airFlow; has speed + direction).
OUTSIDE_AIRFLOW_INDEX = 301
# Inside dome = the TMA sonic anemometers (ESS.airTurbulence, ``speedMagnitude``
# only -- no direction field).  110 = TMA deployable platform; 123-126 = TMA
# top-ring quadrants (-x-y / +x-y / +x+y / -x+y).  Locations documented in
# ts_config_ocs ESS/v8/_init.yaml.  fetch_dome_wind averages speedMagnitude over
# these per visit.
INSIDE_AIRTURB_INDICES = (110, 123, 124, 125, 126)

DEFAULT_TEMP_WINDOW = TimeDelta(0.2, format="sec")


# ----------------------------------------------------------------------------
# M1M3 bulk thermal gradients
# ----------------------------------------------------------------------------
async def get_m1m3_gradients(client, data):
    """Interpolate the M1M3 bulk thermal gradients onto each row's obs_start.

    Adds ``x_gradient``, ``y_gradient``, ``z_gradient`` and ``radial_gradient``
    to ``data`` (which must have an ``obs_start`` column of TAI ISO strings) and
    returns it.  Moved verbatim from ``nightly_table.get_m1m3_gradients`` so both
    callers share one implementation.
    """
    date_strings = Time(
        [str(x) for x in data["obs_start"].values], format="isot", scale="tai"
    ).utc.isot
    data_times = pd.to_datetime(date_strings, format="ISO8601", utc=True)
    sorted_data_times = data_times.sort_values()
    start = Time(sorted_data_times[0])
    end = Time(sorted_data_times[-1])
    data_times = data_times.astype("int64")
    thermocouples = ThermocoupleAnalysis(client)
    await thermocouples.load(start, end, time_bin=30)
    gradients = thermocouples.xyz_r_gradients
    grad_times = pd.to_datetime(
        gradients.index, format="ISO8601", utc=True
    ).astype("int64")
    t0 = grad_times[0]
    grad_times -= t0
    grad_times /= 1e9
    data_times -= t0
    data_times /= 1e9
    names = ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]
    for name in names:
        values = gradients[name].values
        val_series = pd.Series(values)
        val_interpolated = val_series.interpolate()
        data[name] = np.interp(data_times, grad_times, val_interpolated)
    return data


# ----------------------------------------------------------------------------
# TMA truss temperatures
# ----------------------------------------------------------------------------
async def _query_tma_truss(client, start_date, end_date):
    """TMA truss temperatures (ESS index 2, items 6/7) as a 1-minute series.

    Adapted from A. Roodman's snippet; returns a time-indexed DataFrame with
    columns ``tma_truss_temp_pxpy`` and ``tma_truss_temp_mxmy``.
    """
    topic_name = "lsst.sal.ESS.temperature"
    frames = []
    for item, name in TRUSS_ITEMS.items():
        fields = [f"mean(temperatureItem{item}) AS ch{item}"]
        query = client.build_time_range_query(
            topic_name, fields, start_date, end_date, index=TRUSS_ESS_INDEX
        )
        query += " GROUP BY time(1m)"
        table = await client._do_query(query)
        table.rename(columns={f"ch{item}": name}, inplace=True)
        frames.append(table)
    return pd.concat(frames, axis=1)


def _interp_series_to_rows(ts_df, data, cols):
    """Interpolate a time-indexed series onto each row's obs_start (TAI)."""
    date_strings = Time(
        [str(x) for x in data["obs_start"].values], format="isot", scale="tai"
    ).utc.isot
    data_times = pd.to_datetime(date_strings, format="ISO8601", utc=True).astype("int64")
    ts_times = pd.to_datetime(ts_df.index, utc=True).astype("int64")
    if len(ts_times) == 0:
        for c in cols:
            data[c] = np.nan
        return data
    t0 = ts_times[0]
    ts_x = (ts_times - t0) / 1e9
    data_x = (data_times - t0) / 1e9
    for c in cols:
        vals = pd.Series(ts_df[c].values).interpolate().values
        data[c] = np.interp(data_x, ts_x, vals)
    return data


# ----------------------------------------------------------------------------
# Thermal telemetry (temps + delta-Ts + gradients + truss)
# ----------------------------------------------------------------------------
async def fetch_thermal_telemetry(
    efd_client,
    day_seq,
    temp_window=DEFAULT_TEMP_WINDOW,
    include_gradients=True,
    include_truss=True,
):
    """Per-visit thermal telemetry from the EFD.

    Parameters
    ----------
    efd_client : lsst_efd_client.EfdClient
    day_seq : pandas.DataFrame
        One row per unique visit, with columns ``day_obs``, ``seq``,
        ``obs_start`` and ``obs_end`` (TAI ISO strings).
    temp_window : astropy.time.TimeDelta
        Padding added to ``obs_end`` for the ESS temperature averaging window.
    include_gradients, include_truss : bool
        Toggle the M1M3 gradient / TMA-truss columns.

    Returns
    -------
    pandas.DataFrame keyed on ``(day_obs, seq)`` with columns:
        cam_air_temp, m2_air_temp, m1m3_air_temp, outside_temp,
        dome_delta_t, m2_delta_t, cam_m1m3_delta_t,
        x_gradient, y_gradient, z_gradient, radial_gradient   [gradients],
        tma_truss_temp_pxpy, tma_truss_temp_mxmy              [truss].
    """

    async def _temp(index, rec_start, rec_end):
        d = await efd_client.select_time_series(
            "lsst.sal.ESS.temperature",
            ["temperatureItem0"],
            Time(rec_start, scale="tai").utc,
            Time(rec_end, scale="tai").utc + temp_window,
            index=index,
            convert_influx_index=True,
        )
        return d["temperatureItem0"].mean() if "temperatureItem0" in d else np.nan

    rows = {k: [] for k in ("day_obs", "seq", "cam_air_temp", "m2_air_temp",
                            "m1m3_air_temp", "outside_temp")}
    for _, r in day_seq.iterrows():
        rows["day_obs"].append(int(r["day_obs"]))
        rows["seq"].append(int(r["seq"]))
        rows["outside_temp"].append(await _temp(ESS_TEMP_INDEX["outside_temp"], r["obs_start"], r["obs_end"]))
        rows["m2_air_temp"].append(await _temp(ESS_TEMP_INDEX["m2_air_temp"], r["obs_start"], r["obs_end"]))
        rows["cam_air_temp"].append(await _temp(ESS_TEMP_INDEX["cam_air_temp"], r["obs_start"], r["obs_end"]))
        rows["m1m3_air_temp"].append(await _temp(ESS_TEMP_INDEX["m1m3_air_temp"], r["obs_start"], r["obs_end"]))
    out = pd.DataFrame(rows)

    out["m2_delta_t"] = out["m2_air_temp"] - out["m1m3_air_temp"]
    out["dome_delta_t"] = out["outside_temp"] - out["m1m3_air_temp"]
    out["cam_m1m3_delta_t"] = out["cam_air_temp"] - out["m1m3_air_temp"]

    if include_gradients:
        if ThermocoupleAnalysis is None:
            warnings.warn("ThermocoupleAnalysis unavailable; M1M3 gradients skipped")
        else:
            grad = await get_m1m3_gradients(
                efd_client, day_seq[["day_obs", "seq", "obs_start"]].copy()
            )
            gcols = ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]
            out = out.merge(grad[["day_obs", "seq"] + gcols], on=["day_obs", "seq"], how="left")

    if include_truss:
        span0 = Time(min(day_seq["obs_start"]), scale="tai")
        span1 = Time(max(day_seq["obs_end"]), scale="tai")
        truss = await _query_tma_truss(efd_client, span0, span1)
        tcols = list(TRUSS_ITEMS.values())
        ds = _interp_series_to_rows(truss, day_seq[["day_obs", "seq", "obs_start"]].copy(), tcols)
        out = out.merge(ds[["day_obs", "seq"] + tcols], on=["day_obs", "seq"], how="left")

    return out


# ----------------------------------------------------------------------------
# Wind telemetry (inside / outside dome)
# ----------------------------------------------------------------------------
async def fetch_dome_wind(
    efd_client,
    day_seq,
    inside_indices=INSIDE_AIRTURB_INDICES,
    outside_index=OUTSIDE_AIRFLOW_INDEX,
    wind_window=DEFAULT_TEMP_WINDOW,
):
    """Per-visit inside- and outside-dome wind.

    Outside dome comes from ``lsst.sal.ESS.airFlow`` at ``outside_index`` (the
    weather tower, index 301): mean ``speed`` and ``direction``.  Inside dome
    comes from ``lsst.sal.ESS.airTurbulence`` at ``inside_indices`` (the TMA
    sonic anemometers): the mean ``speedMagnitude`` over those sensors.  The
    airTurbulence topic has no direction field, so there is no inside direction.

    Returns
    -------
    pandas.DataFrame keyed on ``(day_obs, seq)`` with columns:
        wind_speed_inside   -- mean airTurbulence speedMagnitude [m/s],
        wind_speed_outside  -- airFlow speed [m/s],
        wind_dir_outside    -- airFlow direction [deg].
    """
    inside_set = set(inside_indices) if inside_indices else set()

    async def _outside(rec_start, rec_end):
        if outside_index is None:
            return (np.nan, np.nan)
        d = await efd_client.select_time_series(
            "lsst.sal.ESS.airFlow",
            ["speed", "direction"],
            Time(rec_start, scale="tai").utc,
            Time(rec_end, scale="tai").utc + wind_window,
            index=outside_index,
            convert_influx_index=True,
        )
        speed = d["speed"].mean() if "speed" in d else np.nan
        direction = d["direction"].mean() if "direction" in d else np.nan
        return (speed, direction)

    async def _inside(rec_start, rec_end):
        if not inside_set:
            return np.nan
        # one query for all instances, then average over the inside sensors
        d = await efd_client.select_time_series(
            "lsst.sal.ESS.airTurbulence",
            ["salIndex", "speedMagnitude"],
            Time(rec_start, scale="tai").utc,
            Time(rec_end, scale="tai").utc + wind_window,
            convert_influx_index=True,
        )
        if "speedMagnitude" not in d or len(d) == 0:
            return np.nan
        sub = d[d["salIndex"].isin(inside_set)] if "salIndex" in d else d
        vals = sub["speedMagnitude"].to_numpy(dtype=float)
        return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan

    rows = {k: [] for k in ("day_obs", "seq", "wind_speed_inside",
                            "wind_speed_outside", "wind_dir_outside")}
    for _, r in day_seq.iterrows():
        rows["day_obs"].append(int(r["day_obs"]))
        rows["seq"].append(int(r["seq"]))
        so, do = await _outside(r["obs_start"], r["obs_end"])
        rows["wind_speed_inside"].append(await _inside(r["obs_start"], r["obs_end"]))
        rows["wind_speed_outside"].append(so)
        rows["wind_dir_outside"].append(do)
    return pd.DataFrame(rows)
