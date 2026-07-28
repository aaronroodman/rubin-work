"""ConsDB transformed-EFD telemetry -- the fast, no-per-visit-raw-EFD path.

Pulls per-exposure MEAN AOS telemetry from the ConsDB Consolidated (transformed)
EFD in a handful of queries, instead of per-visit raw-EFD downloads (which ran
the night table out of batch wall time).  Mirrors Guillem's querying_efd_consdb
notebook.

Array quantities from ``efd_lsstcam.exposure_efd_unpivoted`` (property/field/value):
  - ``mt_logevent_aggregated_dof``               -> DOF Trim (50)          -> dof0-49
  - ``mt_m1m3_applied_elevation_forces_mean``    -> M1M3 elevation LUT     -> lut_dof10-29
  - ``mt_m1m3_applied_azimuth_forces_mean``      -> M1M3 azimuth LUT       -> m1m3azim_dof0-19
  - ``mt_m1m3_applied_thermal_forces_mean``      -> M1M3 thermal LUT       -> m1m3therm_dof0-19
  - ``mt_m2_axial_force_lut_gravity_mean``       -> M2 gravity LUT         -> lut_dof30-49
  - ``mt_m2_axial_force_lut_temperature_mean``   -> M2 thermal LUT         -> m2temp_dof0-19
axial forces (M1M3 zForces[156] / M2 lut*[72]) -> bending amplitudes via ts_ofc
``BendModeToForce``.  (M1M3 azimuth is partly lateral; only its axial zForces are
used, and it is skipped cleanly if the transform stores no zForces for it.)

Scalars from ``efd_lsstcam.exposure_efd`` (pivoted), queried in independent topic
groups so one missing column only drops its group:
  - ESS temps 111/112/113/301 + TMA truss 122 + sonic 110 + camera-hex ESS 1
  - wind: 110 sonic components/magnitude + 301 airflow speed/direction
  - mirror stress scalars m1m3_stress / m2_stress
  - hexapod compensation_offset (LUT -> lut_dof0-9) + aos_corrections (Trim -> trim_hex_dof0-9)
Plus weather-tower ``wind_speed`` / ``wind_dir`` from ``cdb_lsstcam.exposure``.

Column names match ``olr/code/telemetry.py`` + ``aos_trim`` so the ConsDB and
raw-EFD paths yield the SAME schema and are interchangeable.  NOT in the
transform: M1M3 spatial gradients (x/y/z/radial, from the thermocouple array)
and the 123-126 inside anemometers -- fetch those from raw EFD if wanted.

Keyed on ``exposure_id`` (== ``visit_id`` for these single-snap AOS visits).
"""
import numpy as np
import pandas as pd

# --- unpivoted array properties -> (out_prefix, axial field-name, length) -----
UNPIVOT_PROPS = {
    "mt_logevent_aggregated_dof":             ("dof",       "aggregatedDoF",  50),
    "mt_m1m3_applied_elevation_forces_mean":  ("m1m3elev",  "zForces",       156),
    "mt_m1m3_applied_azimuth_forces_mean":    ("m1m3azim",  "zForces",       156),
    "mt_m1m3_applied_thermal_forces_mean":    ("m1m3therm", "zForces",       156),
    "mt_m2_axial_force_lut_gravity_mean":     ("m2grav",    "lutGravity",     72),
    "mt_m2_axial_force_lut_temperature_mean": ("m2temp",    "lutTemperature", 72),
}

# --- pivoted scalar columns: transformed-EFD name -> our output name ----------
TEMP_COLS = {
    "mt_salindex111_temperature_0_mean": "cam_air_temp",
    "mt_salindex112_temperature_0_mean": "m2_air_temp",
    "mt_salindex113_temperature_0_mean": "m1m3_air_temp",
    "mt_salindex301_temperature_0_mean": "outside_temp",
    "mt_salindex122_temperature_6":      "tma_truss_temp_pxpy",   # +X+Y truss
    "mt_salindex122_temperature_7":      "tma_truss_temp_mxmy",   # -X-Y truss
    "mt_salindex110_sonic_temperature_mean": "sonic_temperature",
    **{f"mt_salindex1_temperature_{i}_mean": f"cam_hex_temp_{i}" for i in range(8)},
}
WIND_COLS = {
    "mt_salindex110_wind_speed_magnitude_mean": "wind_speed_inside",
    "mt_salindex301_airflow_speed_mean":        "wind_speed_outside",
    "mt_salindex301_airflow_direction_mean":    "wind_dir_outside",
    "mt_salindex110_wind_speed_0_mean":         "wind_inside_x",
    "mt_salindex110_wind_speed_1_mean":         "wind_inside_y",
    "mt_salindex110_wind_speed_2_mean":         "wind_inside_z",
    "mt_salindex110_wind_speed_maxmagnitude_mean": "wind_inside_maxmag",
}
STRESS_COLS = {"m1m3_stress": "m1m3_stress", "m2_stress": "m2_stress"}

HEX_AXES = ["z", "x", "y", "u", "v"]           # drop w to match the OFC DOF layout
HEX_LUT_COLS = ([f"m2_hexapod_compensation_offset_{a}" for a in HEX_AXES]
                + [f"camera_hexapod_compensation_offset_{a}" for a in HEX_AXES])
HEX_TRIM_COLS = ([f"m2_hexapod_aos_corrections_{a}" for a in HEX_AXES]
                 + [f"camera_hexapod_aos_corrections_{a}" for a in HEX_AXES])

_DELTAS = {  # derived, matching telemetry.py
    "m2_delta_t":      ("m2_air_temp", "m1m3_air_temp"),
    "dome_delta_t":    ("outside_temp", "m1m3_air_temp"),
    "cam_m1m3_delta_t": ("cam_air_temp", "m1m3_air_temp"),
}


def _chunks(seq, n=800):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _to20(vec):
    v = np.atleast_1d(np.asarray(vec, float))
    out = np.full(20, np.nan)
    out[:min(20, len(v))] = v[:20]
    return out


def fetch_arrays_unpivoted(cdb, visit_ids, props=None):
    """DOF Trim + mirror LUT force arrays per exposure from exposure_efd_unpivoted.

    ``props`` (default ``UNPIVOT_PROPS``) is the {property: (out_prefix, axial
    field-name, length)} map to fetch.  Returns ``exposure_id -> {out_prefix:
    np.ndarray}``; each array holds the fields whose name starts with that
    property's axial field-name (e.g. ``zForces*``), all-NaN if absent.
    """
    props = props if props is not None else UNPIVOT_PROPS
    propnames = "', '".join(props)
    frames = []
    for ids in _chunks(list(map(int, visit_ids))):
        idlist = ", ".join(str(v) for v in ids)
        q = (f"SELECT exposure_id, property, field, value "
             f"FROM efd_lsstcam.exposure_efd_unpivoted "
             f"WHERE exposure_id IN ({idlist}) AND property IN ('{propnames}')")
        frames.append(cdb.query(q).to_pandas())
    unp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = {}
    if len(unp) == 0:
        return out
    unp["value"] = pd.to_numeric(unp["value"], errors="coerce")
    fstr = unp["field"].astype(str)
    unp["k"] = fstr.str.extract(r"(\d+)$")[0].astype(float)
    for eid, g in unp.groupby("exposure_id"):
        rec = {}
        for prop, (prefix, fname, n) in props.items():
            sub = g[(g["property"] == prop)
                    & g["field"].astype(str).str.startswith(fname)]
            arr = np.full(n, np.nan)
            kk = sub["k"].to_numpy()
            vv = sub["value"].to_numpy()
            ok = np.isfinite(kk) & (kk >= 0) & (kk < n)
            arr[kk[ok].astype(int)] = vv[ok]
            rec[prefix] = arr
        out[int(eid)] = rec
    return out


def _query_cols(cdb, table, cols, visit_ids, gname):
    """SELECT exposure_id + cols from `table` for visit_ids, chunked + guarded."""
    sel = ", ".join(["exposure_id"] + cols)
    frames = []
    for ids in _chunks(list(map(int, visit_ids))):
        idlist = ", ".join(str(v) for v in ids)
        q = f"SELECT {sel} FROM {table} WHERE exposure_id IN ({idlist})"
        try:
            frames.append(cdb.query(q).to_pandas())
        except Exception as e:
            print(f"(ConsDB '{gname}' query failed [{type(e).__name__}: {e}])",
                  flush=True)
            return None
    return pd.concat(frames, ignore_index=True) if frames else None


def fetch_scalars_pivoted(cdb, visit_ids, hexapod=True):
    """ESS temps/truss/sonic/cam-hex, wind, stress, hexapod LUT/Trim (per group).

    Also joins weather-tower ``wind_speed`` / ``wind_dir`` from
    ``cdb_lsstcam.exposure``.  Independent groups so one bad column drops only
    its group.  ``hexapod=False`` skips the (sparse, logevent-sourced) hexapod
    groups -- take those from the raw EFD instead.  Returns a DataFrame indexed
    by ``exposure_id``.
    """
    groups = [
        ("efd_lsstcam.exposure_efd", list(TEMP_COLS),  "temps"),
        ("efd_lsstcam.exposure_efd", list(WIND_COLS),  "wind"),
        ("efd_lsstcam.exposure_efd", list(STRESS_COLS), "stress"),
        ("cdb_lsstcam.exposure",     ["wind_speed", "wind_dir"], "weather"),
    ]
    if hexapod:
        groups += [("efd_lsstcam.exposure_efd", HEX_LUT_COLS, "hexlut"),
                   ("efd_lsstcam.exposure_efd", HEX_TRIM_COLS, "hextrim")]
    merged = None
    for table, cols, gname in groups:
        g = _query_cols(cdb, table, cols, visit_ids, gname)
        if g is None or len(g) == 0:
            continue
        g = g.drop_duplicates("exposure_id")
        merged = g if merged is None else merged.merge(g, on="exposure_id", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["exposure_id"]).set_index("exposure_id")
    return merged.set_index("exposure_id")


def collect_consdb_telemetry(cdb, visits, config_dir, visit_col="visit_id",
                             dof=True, hexapod=True, m1m3_azim_therm=False):
    """Attach DOF/mirror-LUT/hexapod/temps/wind/stress to ``visits`` (a copy).

    ``dof=False`` / ``hexapod=False`` skip the DOF Trim (dof0-49) / hexapod LUT
    (lut_dof0-9, trim_hex_dof0-9) -- these are SAL logevents that the transform
    captures in only a few % of exposures, so the hybrid path takes them from the
    raw EFD (as-of) instead.  The mirror LUT (lut_dof10-49, always continuous
    telemetry) and temps/wind/stress are always taken from ConsDB.

    ``m1m3_azim_therm=False`` (default) skips the M1M3 azimuth + thermal LUT
    bending modes (m1m3azim_dof0-19, m1m3therm_dof0-19) -- currently all zero
    on-sky.  Set True to compute/store them if those LUT terms ever activate.

    Does NOT add v-modes (caller derives from dof0-49) or M1M3 gradients (raw EFD).
    """
    from lsst.ts.ofc import OFCData, BendModeToForce
    ofc = OFCData("lsst", config_dir=config_dir)
    bmf_m1m3 = BendModeToForce("M1M3", ofc)
    bmf_m2 = BendModeToForce("M2", ofc)

    def _bend(bmf, arr):
        if arr is None or not np.isfinite(arr).any():
            return None
        try:
            return _to20(bmf.bending_mode(np.nan_to_num(arr, nan=0.0)))
        except Exception as e:
            print(f"(bending_mode failed [{type(e).__name__}])", flush=True)
            return None

    visits = visits.copy()
    vids = visits[visit_col].astype(int).to_numpy()
    n = len(vids)
    dof_arr = np.full((n, 50), np.nan)
    lut = np.full((n, 50), np.nan)              # 0-9 hex, 10-29 M1M3 elev, 30-49 M2 grav
    m2temp = np.full((n, 20), np.nan)
    m1m3azim = np.full((n, 20), np.nan)
    m1m3therm = np.full((n, 20), np.nan)
    props = dict(UNPIVOT_PROPS)
    if not m1m3_azim_therm:                     # off by default (all zero on-sky)
        props = {k: v for k, v in props.items() if v[0] not in ("m1m3azim", "m1m3therm")}
    try:
        arrays = fetch_arrays_unpivoted(cdb, vids, props=props)
        for i, vid in enumerate(vids):
            rec = arrays.get(int(vid))
            if not rec:
                continue
            if np.isfinite(rec["dof"]).any():
                dof_arr[i] = rec["dof"]
            for src, dst in ((_bend(bmf_m1m3, rec["m1m3elev"]),  ("lut", 10, 30)),
                             (_bend(bmf_m2,   rec["m2grav"]),    ("lut", 30, 50))):
                if src is not None:
                    lut[i, dst[1]:dst[2]] = src
            v = _bend(bmf_m2, rec["m2temp"]);      m2temp[i] = v if v is not None else m2temp[i]
            if m1m3_azim_therm:
                v = _bend(bmf_m1m3, rec.get("m1m3azim"));  m1m3azim[i] = v if v is not None else m1m3azim[i]
                v = _bend(bmf_m1m3, rec.get("m1m3therm")); m1m3therm[i] = v if v is not None else m1m3therm[i]
    except Exception as e:
        print(f"(ConsDB unpivoted arrays failed [{type(e).__name__}: {e}])", flush=True)

    try:
        scal = fetch_scalars_pivoted(cdb, vids, hexapod=hexapod).reindex(vids)
    except Exception as e:
        print(f"(ConsDB pivoted scalars failed [{type(e).__name__}: {e}])", flush=True)
        scal = pd.DataFrame(index=vids)

    def _col(name):
        return (pd.to_numeric(scal[name], errors="coerce").to_numpy()
                if name in scal else np.full(n, np.nan))

    if hexapod:
        for j, c in enumerate(HEX_LUT_COLS):
            lut[:, j] = _col(c)

    blocks = {}
    if dof:
        blocks.update({f"dof{i}": dof_arr[:, i] for i in range(50)})
    lut_lo = 0 if hexapod else 10              # skip lut_dof0-9 (hexapod) in hybrid
    blocks.update({f"lut_dof{i}": lut[:, i] for i in range(lut_lo, 50)})
    blocks.update({f"m2temp_dof{i}": m2temp[:, i] for i in range(20)})
    if m1m3_azim_therm:
        blocks.update({f"m1m3azim_dof{i}": m1m3azim[:, i] for i in range(20)})
        blocks.update({f"m1m3therm_dof{i}": m1m3therm[:, i] for i in range(20)})
    if hexapod:
        for j, c in enumerate(HEX_TRIM_COLS):
            blocks[f"trim_hex_dof{j}"] = _col(c)
    for src, name in {**TEMP_COLS, **WIND_COLS, **STRESS_COLS}.items():
        blocks[name] = _col(src)
    blocks["wind_speed_weather"] = _col("wind_speed")
    blocks["wind_dir_weather"] = _col("wind_dir")

    add = pd.DataFrame(blocks, index=visits.index)
    for name, (a, b) in _DELTAS.items():
        if a in add and b in add:
            add[name] = add[a] - add[b]
    cam_hex = [f"cam_hex_temp_{i}" for i in range(8)]
    if all(c in add for c in cam_hex):
        add["cam_hex_temp_avg"] = add[cam_hex].mean(axis=1)
    visits = pd.concat([visits, add], axis=1)

    n_dof = int(np.isfinite(dof_arr).all(axis=1).sum())
    n_mir = int(np.isfinite(lut[:, 10:50]).any(axis=1).sum())
    extra = ""
    if m1m3_azim_therm:
        extra = (f", azimuth {int(np.isfinite(m1m3azim).any(axis=1).sum())}, "
                 f"thermal {int(np.isfinite(m1m3therm).any(axis=1).sum())}")
    print(f"ConsDB EFD: DOF {n_dof}/{n} (added={dof}), mirror LUT {n_mir}/{n}{extra}",
          flush=True)
    return visits
