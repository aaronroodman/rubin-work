---
name: transformed-efd-consdb
description: "What the ConsDB transformed EFD (efd_lsstcam) has for AOS telemetry — mirror forces+DOF in the UNPIVOTED table, hexapod/wind/thermal pivoted; only M1M3 gradients absent"
metadata: 
  node_type: memory
  type: reference
  originSessionId: d6ed390e-ee48-44ee-8c0a-a6efca6a5fdf
  modified: 2026-07-28T17:08:02.798Z
---

ConsDB **transformed EFD** schema: https://sdm-schemas.lsst.io/efd_lsstcam.html
USDF-only, but reachable both on the RSP **and on slaciana/sdfiana** (unlike the
in-pod-only `consdb-pq.consdb`). Tables: `exposure_efd` (per-exposure),
`visit1_efd` (per-visit, join key `visit_id`) — identical columns. Values are
per-exposure/visit aggregates (mean/median/stddev/min/max), not point-in-time.

**Two table shapes matter.** PIVOTED `exposure_efd`/`visit1_efd` = one column per
scalar. UNPIVOTED `exposure_efd_unpivoted`/`visit1_efd_unpivoted` = (property,
field, value) rows — this is where the ARRAY quantities live (mirror force
vectors, aggregated DOF). Searching only the pivoted schema WRONGLY concludes the
mirror forces are absent — they are in the unpivoted table (Guillem's
querying_efd_consdb notebook, 2026-07-27).

**DONE 2026-07-27:** rubin-work now has a full ConsDB telemetry path —
`aos/code/aos_consdb_efd.py` (`collect_consdb_telemetry`), used by
`blocks/code/telemetry_pipeline.py` via `telemetry_source` (config /
`--telemetry-source`): `consdb` (default) or `efd` (raw-EFD fallback, kept).
Replaces the slow per-visit raw-EFD mirror download that timed out the night
table. Relevant to [[optatmo-moments-project]] / blocks T539 + per-night pipeline.

PRESENT — UNPIVOTED (`exposure_efd_unpivoted`, property/field/value):
- `mt_logevent_aggregated_dof` (field aggregatedDoF0-49) = **DOF Trim (50)** → dof0-49.
- `mt_m1m3_applied_elevation_forces_mean` (zForces0-155) → M1M3 elev LUT → lut_dof10-29.
  Also `_azimuth_forces_mean` / `_thermal_forces_mean` (azimuth partly lateral).
- `mt_m2_axial_force_lut_gravity_mean` / `_lut_temperature_mean` (0-71) → M2 LUT.
  forces → bending via ts_ofc BendModeToForce. Kept SEPARATE: lut_dof30-49=M2 grav,
  m2temp_dof0-19=M2 temp; total M2 LUT = grav + temp (Guillem sums them).

PRESENT — PIVOTED (`exposure_efd`):
- Hexapod: `*_compensation_offset_{z,x,y,u,v,w}` = **LUT** (=lut_dof0-9);
  `*_aos_corrections_*` = **Trim** (=trim_hex_dof0-9). aos_corrections=Trim,
  compensation_offset=LUT. See [[aos-dof-terminology]].
- ESS temps `mt_salindex{111,112,113,301}_temperature_0_mean` (cam/M2/M1M3/outside),
  truss `mt_salindex122_temperature_6/7` (note: NO `_mean` suffix), sonic
  `mt_salindex110_sonic_temperature_mean`, cam-hex ESS `mt_salindex1_temperature_0-7_mean`.
- Wind: `mt_salindex110_wind_speed_0/1/2_mean` (x/y/z) + `_magnitude_mean` (only 110;
  123-126 NOT transformed → inside wind is 110-only, not the 5-sensor avg). Outside
  `mt_salindex301_airflow_speed_mean` + `_direction_mean`. Weather tower
  `wind_speed`/`wind_dir` are in `cdb_lsstcam.exposure` (not the efd tables).
- Mirror `m1m3_stress` / `m2_stress` (RSS-of-bending-mode scalars).

ABSENT — still raw-EFD only:
- **M1M3 spatial gradients** (x/y/z/radial_gradient): only 4 discrete M1M3 temps
  in the transform, not the FCU thermocouple array → consdb mode still pulls
  gradients from raw EFD (ThermocoupleAnalysis, per night).
- The 123-126 inside anemometers (only salIndex 110 transformed).
