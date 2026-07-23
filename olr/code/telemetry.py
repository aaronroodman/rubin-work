"""Shared EFD environmental-telemetry helpers for the AOS analyses.

Single source of truth for the thermal + wind telemetry attached per visit, so
the logic is not duplicated across ``nightly_table.py``, the block-T539 build,
and the FAM summary.  The core functions are ``async`` (they await the
``lsst_efd_client`` EfdClient); ``*_sync`` wrappers are provided for notebooks /
sync callers.

Provided:
  - ``fetch_thermal_telemetry`` -- ESS air temperatures (camera / M2 / M1M3 /
    outside), the derived delta-Ts, the M1M3 bulk thermal gradients
    (x/y/z/radial), and the TMA truss temperatures.
  - ``fetch_dome_wind`` -- outside-dome wind speed/direction from
    ``lsst.sal.ESS.airFlow`` (weather tower) and inside-dome wind speed from
    ``lsst.sal.ESS.airTurbulence`` (TMA sonic anemometers).
  - ``get_m1m3_gradients`` -- M1M3 bulk thermal gradients interpolated onto each
    visit's ``obs_start`` (re-exported by ``nightly_table``).

Robustness: the M1M3-gradient and TMA-truss loads are done **per night** (never
over a multi-night span, which times out), and every EFD call is wrapped so a
timeout / missing sensor yields NaN for that piece rather than aborting the run.
Pass ``progress=True`` (default) for a tqdm bar over the slow loops.

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

# --- TMA truss temperatures --------------------------------------------------
# These are channels of the M2-hexapod ESS instance (salIndex 122, per
# ts_config_ocs ESS/v8/_init.yaml): temperatureItem0-5 = M2 hexapod struts 1-6,
# temperatureItem6 = "+X/+Y Truss Structure", temperatureItem7 = "-X/-Y Truss".
# (The older config used salIndex 2; it was renumbered.)
TRUSS_ESS_INDEX = 122
TRUSS_ITEMS = {6: "tma_truss_temp_pxpy", 7: "tma_truss_temp_mxmy"}

# --- Wind: outside from ESS.airFlow, inside from ESS.airTurbulence --------------
# Outside dome = the weather-tower anemometer (ESS.airFlow; has speed + direction).
OUTSIDE_AIRFLOW_INDEX = 301
# Inside dome = the TMA sonic anemometers (ESS.airTurbulence, ``speedMagnitude``
# only -- no direction field).  110 = TMA deployable platform; 123-126 = TMA
# top-ring quadrants (-x-y / +x-y / +x+y / -x+y).  Locations documented in
# ts_config_ocs ESS/v8/_init.yaml.  fetch_dome_wind averages speedMagnitude over
# these per visit.
INSIDE_AIRTURB_INDICES = (110, 123, 124, 125, 126)

DEFAULT_TEMP_WINDOW = TimeDelta(0.2, format="sec")

GRAD_COLS = ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]


def _tqdm(iterable, total=None, desc=None, progress=True):
    """tqdm wrapper that degrades to a plain iterable if tqdm is unavailable."""
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


# ----------------------------------------------------------------------------
# M1M3 bulk thermal gradients
# ----------------------------------------------------------------------------
async def get_m1m3_gradients(client, data):
    """Interpolate the M1M3 bulk thermal gradients onto each row's obs_start.

    Adds ``x_gradient``, ``y_gradient``, ``z_gradient`` and ``radial_gradient``
    to ``data`` (which must have an ``obs_start`` column of TAI ISO strings) and
    returns it.  Call it over a **single night's** span -- loading over a
    multi-night span makes the thermocouple query time out.
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
    for name in GRAD_COLS:
        values = gradients[name].values
        val_series = pd.Series(values)
        val_interpolated = val_series.interpolate()
        data[name] = np.interp(data_times, grad_times, val_interpolated)
    return data


# ----------------------------------------------------------------------------
# TMA truss temperatures
# ----------------------------------------------------------------------------
async def _query_tma_truss(client, start_date, end_date):
    """TMA truss temperatures as a time-indexed DataFrame.

    The truss RTDs are channels 6/7 of the M2-hexapod ESS instance
    (``TRUSS_ESS_INDEX`` = 122): ``temperatureItem6`` -> ``tma_truss_temp_pxpy``,
    ``temperatureItem7`` -> ``tma_truss_temp_mxmy``.  Query over one night's span.
    """
    cols = {f"temperatureItem{item}": name for item, name in TRUSS_ITEMS.items()}
    d = await client.select_time_series(
        "lsst.sal.ESS.temperature",
        list(cols),
        Time(start_date).utc,
        Time(end_date).utc,
        index=TRUSS_ESS_INDEX,
        convert_influx_index=True,
    )
    out = pd.DataFrame(index=getattr(d, "index", None))
    for src, name in cols.items():
        out[name] = d[src] if (len(d) and src in d) else np.nan
    return out


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


def _nan_cols(sub, cols):
    """A (day_obs, seq) frame with NaN placeholder columns (failure fallback)."""
    out = sub[["day_obs", "seq"]].copy()
    for c in cols:
        out[c] = np.nan
    return out


# ----------------------------------------------------------------------------
# Thermal telemetry (temps + delta-Ts + gradients + truss)
# ----------------------------------------------------------------------------
async def fetch_thermal_telemetry(
    efd_client,
    day_seq,
    temp_window=DEFAULT_TEMP_WINDOW,
    include_gradients=True,
    include_truss=True,
    progress=True,
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
        Toggle the M1M3-gradient / TMA-truss columns.
    progress : bool
        Show a tqdm bar over the per-visit and per-night loops.

    Returns
    -------
    pandas.DataFrame keyed on ``(day_obs, seq)`` with columns:
        cam_air_temp, m2_air_temp, m1m3_air_temp, outside_temp,
        dome_delta_t, m2_delta_t, cam_m1m3_delta_t,
        x_gradient, y_gradient, z_gradient, radial_gradient   [gradients],
        tma_truss_temp_pxpy, tma_truss_temp_mxmy              [truss].

    Every EFD call is guarded: a timeout / missing sensor yields NaN for that
    entry rather than aborting.  Gradient/truss loads are done per night.
    """

    async def _temp(index, rec_start, rec_end):
        try:
            d = await efd_client.select_time_series(
                "lsst.sal.ESS.temperature",
                ["temperatureItem0"],
                Time(rec_start, scale="tai").utc,
                Time(rec_end, scale="tai").utc + temp_window,
                index=index,
                convert_influx_index=True,
            )
            return d["temperatureItem0"].mean() if ("temperatureItem0" in d and len(d)) else np.nan
        except Exception:
            return np.nan

    rows = {k: [] for k in ("day_obs", "seq", "cam_air_temp", "m2_air_temp",
                            "m1m3_air_temp", "outside_temp")}
    for _, r in _tqdm(list(day_seq.iterrows()), total=len(day_seq),
                      desc="ESS temps/visit", progress=progress):
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

    # --- M1M3 gradients, loaded per night (a multi-night load times out) ---
    if include_gradients and ThermocoupleAnalysis is None:
        warnings.warn("ThermocoupleAnalysis unavailable; M1M3 gradients skipped")
    elif include_gradients:
        parts = []
        for day, sub in _tqdm(list(day_seq.groupby("day_obs")),
                              desc="M1M3 gradients/night", progress=progress):
            try:
                g = await get_m1m3_gradients(
                    efd_client, sub[["day_obs", "seq", "obs_start"]].copy())
                parts.append(g[["day_obs", "seq"] + GRAD_COLS])
            except Exception as e:
                warnings.warn(f"M1M3 gradients failed for day_obs={day}: "
                              f"{type(e).__name__}: {e}")
                parts.append(_nan_cols(sub, GRAD_COLS))
        out = out.merge(pd.concat(parts, ignore_index=True),
                        on=["day_obs", "seq"], how="left")

    # --- TMA truss temps, loaded per night ---
    if include_truss:
        tcols = list(TRUSS_ITEMS.values())
        parts = []
        for day, sub in _tqdm(list(day_seq.groupby("day_obs")),
                              desc="TMA truss/night", progress=progress):
            try:
                span0 = Time(min(sub["obs_start"]), scale="tai")
                span1 = Time(max(sub["obs_end"]), scale="tai")
                ts = await _query_tma_truss(efd_client, span0, span1)
                part = _interp_series_to_rows(
                    ts, sub[["day_obs", "seq", "obs_start"]].copy(), tcols)
                parts.append(part[["day_obs", "seq"] + tcols])
            except Exception as e:
                warnings.warn(f"TMA truss failed for day_obs={day}: "
                              f"{type(e).__name__}: {e}")
                parts.append(_nan_cols(sub, tcols))
        out = out.merge(pd.concat(parts, ignore_index=True),
                        on=["day_obs", "seq"], how="left")

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
    progress=True,
):
    """Per-visit inside- and outside-dome wind.

    Outside dome comes from ``lsst.sal.ESS.airFlow`` at ``outside_index`` (the
    weather tower, index 301): mean ``speed`` and ``direction``.  Inside dome
    comes from ``lsst.sal.ESS.airTurbulence`` at ``inside_indices`` (the TMA
    sonic anemometers): the mean ``speedMagnitude`` over those sensors.  The
    airTurbulence topic has no direction field, so there is no inside direction.

    Every EFD call is guarded (timeout / missing sensor -> NaN).

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
        try:
            d = await efd_client.select_time_series(
                "lsst.sal.ESS.airFlow",
                ["speed", "direction"],
                Time(rec_start, scale="tai").utc,
                Time(rec_end, scale="tai").utc + wind_window,
                index=outside_index,
                convert_influx_index=True,
            )
            speed = d["speed"].mean() if ("speed" in d and len(d)) else np.nan
            direction = d["direction"].mean() if ("direction" in d and len(d)) else np.nan
            return (speed, direction)
        except Exception:
            return (np.nan, np.nan)

    async def _inside(rec_start, rec_end):
        if not inside_set:
            return np.nan
        try:
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
        except Exception:
            return np.nan

    rows = {k: [] for k in ("day_obs", "seq", "wind_speed_inside",
                            "wind_speed_outside", "wind_dir_outside")}
    for _, r in _tqdm(list(day_seq.iterrows()), total=len(day_seq),
                      desc="dome wind/visit", progress=progress):
        rows["day_obs"].append(int(r["day_obs"]))
        rows["seq"].append(int(r["seq"]))
        so, do = await _outside(r["obs_start"], r["obs_end"])
        rows["wind_speed_inside"].append(await _inside(r["obs_start"], r["obs_end"]))
        rows["wind_speed_outside"].append(so)
        rows["wind_dir_outside"].append(do)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Synchronous wrappers (for notebooks / sync callers)
# ----------------------------------------------------------------------------
# The async functions above are the native interface (nightly_table awaits
# them).  Notebooks that use the summit_utils sync EFD helpers must NOT mix a
# top-level ``await`` with those helpers -- summit_utils applies ``nest_asyncio``
# to the kernel loop, and a subsequent native ``await`` breaks it ("pop from an
# empty deque").  These wrappers run the coroutine via run_until_complete, which
# is re-entrant under nest_asyncio and is the pattern the rest of the notebook
# already relies on.
def _run_coro(coro):
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


def fetch_thermal_telemetry_sync(efd_client, day_seq, **kwargs):
    """Synchronous wrapper around :func:`fetch_thermal_telemetry`."""
    return _run_coro(fetch_thermal_telemetry(efd_client, day_seq, **kwargs))


def fetch_dome_wind_sync(efd_client, day_seq, **kwargs):
    """Synchronous wrapper around :func:`fetch_dome_wind`."""
    return _run_coro(fetch_dome_wind(efd_client, day_seq, **kwargs))
