"""ConsDB transformed-EFD telemetry -- the fast, no-per-visit-raw-EFD path.

Pulls per-exposure MEAN AOS telemetry from the ConsDB Consolidated (transformed)
EFD in ~2 queries, instead of per-visit raw-EFD downloads (which ran the night
table out of batch wall time).  Mirrors Guillem's querying_efd_consdb notebook:

  * ``efd_lsstcam.exposure_efd_unpivoted`` (property/field/value rows) supplies
    the ARRAY quantities:
      - ``mt_logevent_aggregated_dof``               -> DOF Trim (50)
      - ``mt_m1m3_applied_elevation_forces_mean``    -> M1M3 elevation LUT (156 zForces)
      - ``mt_m2_axial_force_lut_gravity_mean``       -> M2 gravity LUT (72)
      - ``mt_m2_axial_force_lut_temperature_mean``   -> M2 thermal LUT (72)
    forces -> bending amplitudes via ts_ofc ``BendModeToForce``.
  * ``efd_lsstcam.exposure_efd`` (pivoted) supplies the SCALARS: ESS temps, TMA
    truss, wind/airflow, and hexapod ``compensation_offset`` (LUT) / ``aos_corrections``.

Column names match ``olr/code/telemetry.py`` + ``aos_trim`` (``dof*``, ``lut_dof*``,
``cam_air_temp`` ...), so the ConsDB and raw-EFD paths yield the SAME schema and
are interchangeable.  Mirror terms are kept SEPARATE (per the analysis choice):
    lut_dof10-29 = M1M3 elevation,   lut_dof30-49 = M2 gravity,
    m2temp_dof0-19 = M2 temperature   (M2 total LUT = lut_dof30-49 + m2temp_dof0-19).

NOT in the transform (still need raw EFD if wanted): the M1M3 spatial gradients
(x/y/z/radial_gradient, from the thermocouple array) and the 123-126 inside
anemometers (only salIndex 110 is transformed).  Everything else is here.

Keyed on ``exposure_id`` (== ``visit_id`` for these single-snap AOS visits).
"""
import re

import numpy as np
import pandas as pd

# --- unpivoted array properties -> short prefix used to rebuild each array ----
UNPIVOT_PROPS = {
    "mt_logevent_aggregated_dof":             ("dof",      "aggregatedDoF", 50),
    "mt_m1m3_applied_elevation_forces_mean":  ("m1m3elev", "zForces",       156),
    "mt_m2_axial_force_lut_gravity_mean":     ("m2grav",   "lutGravity",    72),
    "mt_m2_axial_force_lut_temperature_mean": ("m2temp",   "lutTemperature", 72),
}

# --- pivoted scalar columns: transformed-EFD name -> our output name ----------
TEMP_COLS = {
    "mt_salindex111_temperature_0_mean": "cam_air_temp",
    "mt_salindex112_temperature_0_mean": "m2_air_temp",
    "mt_salindex113_temperature_0_mean": "m1m3_air_temp",
    "mt_salindex301_temperature_0_mean": "outside_temp",
    "mt_salindex122_temperature_6":      "tma_truss_temp_pxpy",  # +X+Y truss
    "mt_salindex122_temperature_7":      "tma_truss_temp_mxmy",  # -X-Y truss
}
WIND_COLS = {
    "mt_salindex110_wind_speed_magnitude_mean": "wind_speed_inside",   # TMA sonic (110 only)
    "mt_salindex301_airflow_speed_mean":        "wind_speed_outside",  # weather tower
    "mt_salindex301_airflow_direction_mean":    "wind_dir_outside",
    "mt_salindex110_wind_speed_0_mean":         "wind_inside_x",       # sonic components
    "mt_salindex110_wind_speed_1_mean":         "wind_inside_y",
    "mt_salindex110_wind_speed_2_mean":         "wind_inside_z",
    "mt_salindex110_wind_speed_maxmagnitude_mean": "wind_inside_maxmag",
}
HEX_AXES = ["z", "x", "y", "u", "v"]           # drop w to match the OFC DOF layout
# lut_dof0-4 = M2 hex, lut_dof5-9 = cam hex  (LUT = compensation_offset)
HEX_LUT_COLS = ([f"m2_hexapod_compensation_offset_{a}" for a in HEX_AXES]
                + [f"camera_hexapod_compensation_offset_{a}" for a in HEX_AXES])
# hexapod AOS-correction Trim (dof-space), stored as trim_hex_dof0-9
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


def fetch_arrays_unpivoted(cdb, visit_ids):
    """DOF Trim + mirror LUT force arrays per exposure from exposure_efd_unpivoted.

    Returns a dict ``exposure_id -> {prefix: np.ndarray}`` for prefixes
    ``dof`` (50), ``m1m3elev`` (156), ``m2grav`` (72), ``m2temp`` (72).
    """
    props = "', '".join(UNPIVOT_PROPS)
    frames = []
    for ids in _chunks(list(map(int, visit_ids))):
        idlist = ", ".join(str(v) for v in ids)
        q = (f"SELECT exposure_id, property, field, value "
             f"FROM efd_lsstcam.exposure_efd_unpivoted "
             f"WHERE exposure_id IN ({idlist}) AND property IN ('{props}')")
        frames.append(cdb.query(q).to_pandas())
    unp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = {}
    if len(unp) == 0:
        return out
    unp["value"] = pd.to_numeric(unp["value"], errors="coerce")
    unp["k"] = unp["field"].astype(str).str.extract(r"(\d+)$").astype(float)
    for eid, g in unp.groupby("exposure_id"):
        rec = {}
        for prop, (prefix, _field, n) in UNPIVOT_PROPS.items():
            sub = g[g["property"] == prop]
            arr = np.full(n, np.nan)
            kk = sub["k"].to_numpy()
            vv = sub["value"].to_numpy()
            ok = np.isfinite(kk) & (kk >= 0) & (kk < n)
            arr[kk[ok].astype(int)] = vv[ok]
            rec[prefix] = arr
        out[int(eid)] = rec
    return out


def fetch_scalars_pivoted(cdb, visit_ids):
    """ESS temps, TMA truss, wind, and hexapod LUT/Trim per exposure (pivoted).

    Queried in independent topic groups so a single missing/renamed column in
    the deployed schema only drops its own group (NaN) rather than the whole
    scalar block.
    """
    groups = {"temps": list(TEMP_COLS), "wind": list(WIND_COLS),
              "hexlut": HEX_LUT_COLS, "hextrim": HEX_TRIM_COLS}
    ids_all = list(map(int, visit_ids))
    merged = None
    for gname, cols in groups.items():
        sel = ", ".join(["exposure_id"] + cols)
        frames, ok = [], True
        for ids in _chunks(ids_all):
            idlist = ", ".join(str(v) for v in ids)
            q = (f"SELECT {sel} FROM efd_lsstcam.exposure_efd "
                 f"WHERE exposure_id IN ({idlist})")
            try:
                frames.append(cdb.query(q).to_pandas())
            except Exception as e:
                print(f"(ConsDB pivoted group '{gname}' failed "
                      f"[{type(e).__name__}: {e}])", flush=True)
                ok = False
                break
        if not ok or not frames:
            continue
        g = pd.concat(frames, ignore_index=True)
        merged = g if merged is None else merged.merge(g, on="exposure_id", how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["exposure_id"])


def collect_consdb_telemetry(cdb, visits, config_dir, visit_col="visit_id"):
    """Attach DOF/mirror-LUT/hexapod/temps/wind to ``visits`` from the ConsDB
    transformed EFD.  Returns the enriched DataFrame (a copy).

    Adds: dof0-49, lut_dof0-49 (hexapod + M1M3 elevation + M2 gravity),
    m2temp_dof0-19, trim_hex_dof0-9, ESS temps + deltas + truss, wind columns.
    Does NOT add v-modes (caller derives from dof) or M1M3 gradients (raw EFD).
    """
    from lsst.ts.ofc import OFCData, BendModeToForce
    ofc = OFCData("lsst", config_dir=config_dir)
    bmf_m1m3 = BendModeToForce("M1M3", ofc)
    bmf_m2 = BendModeToForce("M2", ofc)

    visits = visits.copy()
    vids = visits[visit_col].astype(int).to_numpy()

    # --- arrays: DOF Trim + mirror bending modes (guarded) ---
    dof = np.full((len(vids), 50), np.nan)
    lut = np.full((len(vids), 50), np.nan)      # 0-9 hex (filled below), 10-49 mirror
    m2temp = np.full((len(vids), 20), np.nan)
    try:
        arrays = fetch_arrays_unpivoted(cdb, vids)
        for i, vid in enumerate(vids):
            rec = arrays.get(int(vid))
            if not rec:
                continue
            if np.isfinite(rec["dof"]).any():
                dof[i] = rec["dof"]
            if np.isfinite(rec["m1m3elev"]).any():
                lut[i, 10:30] = _to20(bmf_m1m3.bending_mode(rec["m1m3elev"]))
            if np.isfinite(rec["m2grav"]).any():
                lut[i, 30:50] = _to20(bmf_m2.bending_mode(rec["m2grav"]))
            if np.isfinite(rec["m2temp"]).any():
                m2temp[i] = _to20(bmf_m2.bending_mode(rec["m2temp"]))
    except Exception as e:
        print(f"(ConsDB unpivoted arrays failed [{type(e).__name__}: {e}])", flush=True)

    # --- scalars: hexapod LUT/Trim + temps + wind (guarded) ---
    scal = None
    try:
        scal = fetch_scalars_pivoted(cdb, vids)
    except Exception as e:
        print(f"(ConsDB pivoted scalars failed [{type(e).__name__}: {e}])", flush=True)
    if scal is not None and len(scal):
        scal = scal.set_index("exposure_id")
        scal = scal.reindex(vids)               # align to visit order
        for j, c in enumerate(HEX_LUT_COLS):    # -> lut_dof0-9
            if c in scal:
                lut[:, j] = pd.to_numeric(scal[c], errors="coerce").to_numpy()

    # assemble DataFrame columns via one concat (avoid fragmentation)
    blocks = {}
    for i in range(50):
        blocks[f"dof{i}"] = dof[:, i]
    for i in range(50):
        blocks[f"lut_dof{i}"] = lut[:, i]
    for i in range(20):
        blocks[f"m2temp_dof{i}"] = m2temp[:, i]
    if scal is not None and len(scal):
        for j, c in enumerate(HEX_TRIM_COLS):   # hexapod Trim (aos_corrections)
            blocks[f"trim_hex_dof{j}"] = (pd.to_numeric(scal[c], errors="coerce").to_numpy()
                                          if c in scal else np.full(len(vids), np.nan))
        for src, name in {**TEMP_COLS, **WIND_COLS}.items():
            blocks[name] = (pd.to_numeric(scal[src], errors="coerce").to_numpy()
                            if src in scal else np.full(len(vids), np.nan))
    add = pd.DataFrame(blocks, index=visits.index)
    # derived delta-Ts (match telemetry.py)
    for name, (a, b) in _DELTAS.items():
        if a in add and b in add:
            add[name] = add[a] - add[b]
    visits = pd.concat([visits, add], axis=1)
    n_dof = int(np.isfinite(dof).all(axis=1).sum())
    n_mir = int(np.isfinite(lut[:, 10:50]).any(axis=1).sum())
    print(f"ConsDB EFD: DOF {n_dof}/{len(vids)}, mirror LUT {n_mir}/{len(vids)}",
          flush=True)
    return visits
