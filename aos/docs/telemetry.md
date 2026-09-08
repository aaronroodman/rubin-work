# Telemetry inventory — what exists, where it comes from, and what reaches the tables

> **Status:** current · **Last updated:** 2026-09-08 · **Kind:** reference (telemetry inventory)

Every per-visit telemetry quantity used in the AOS work: its name in the Consolidated
Database (ConsDB) transformed Engineering Facility Database (EFD) or in the raw EFD,
whether it is documented, its measured coverage on Full Array Mode (FAM) exposures, its
units, and which source a fetch should prefer.

All coverage figures were measured on 2026-09-08 against `param_set`
`fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` — 3385 FAM visits, `day_obs` 20250415 to
20260713 — using the USDF ConsDB at `https://usdf-rsp.slac.stanford.edu/consdb`. Samples
of 400–500 visits are the most recent ones unless stated.

**Read this first:** some telemetry simply did not exist early in commissioning and was
added gradually through 2025. A low coverage figure below is often that history, not a
bug. Filling those visits with NaN is the intended behaviour.

## The short version

| what | source to use | why |
|---|---|---|
| ESS air temperatures | **ConsDB** | 88.6% on FAM exposures, one bulk query |
| Wind and airflow | **ConsDB** | 88.6%; was unused, now fetched by `run_attach_telemetry.py` |
| TMA truss temperatures | **ConsDB** | 88.6% |
| M1M3 elevation LUT, M2 gravity LUT | **ConsDB** | 400/400 sampled FAM visits |
| **Trim (all 50 DOF)** | **EFD, as-of-time** | **0% on FAM exposures in ConsDB** — no ConsDB path exists |
| Tweak | **derived** | no topic or property exists; difference consecutive Trim (0.0 where no correction was applied) |
| M1M3 spatial gradients | **EFD** | not in the transform |
| Camera-body temperatures | **EFD** | not in ConsDB at all |
| Mirror stress | neither | 0% in ConsDB on FAM exposures |

## 1. What actually reaches `visits.parquet` today

The combined `output/<ps>/visits.parquet` has **58 columns**. Measured finite fractions:

| group | columns | finite | notes |
|---|---|---|---|
| ESS air temps | `cam_air_temp`, `m2_air_temp`, `m1m3_air_temp`, `outside_temp` | 94.3% (`cam_air_temp`) | deg C |
| derived deltas | `m2_delta_t`, `dome_delta_t`, `cam_m1m3_delta_t` | as above | deg C, differences of the four |
| M1M3 gradients | `x_gradient`, `y_gradient`, `z_gradient`, `radial_gradient` | 100% except the first chunk | deg C per m; **41%** in `20250415_20250531` |
| TMA truss | `tma_truss_temp_pxpy`, `tma_truss_temp_mxmy` | **0% in the two oldest 2025 chunks** | deg C |
| camera body | 26 `cam_*Temp` columns | 98.6% | deg C; **combined table only** |
| pointing | `alt`, `az`, `rotator_angle`, `ra`, `dec`, `skyAngle` | — | deg |
| identity | `visit`, `day_obs`, `seq_num`, `band`, `science_program`, `mjd` | — | |
| donut counts | `n_donuts`, `n_detectors`, `median_blur_arcsec`, `visit_quality_pass` | — | arcsec for the blur |
| **commanded DOF** | — | **absent entirely** | see §4 |
| **wind / airflow** | — | **absent entirely** | see §3, available in ConsDB |

### Camera-body temperatures are combined-table only, by design

`run_backfill_camera_telemetry.py` writes a per-chunk `camera_telemetry.parquet` sidecar
and `--merge` joins it into the **combined** `visits.parquet` only, deliberately leaving
the expensive per-chunk `mktable` outputs untouched. Measured: **28 `cam_*` columns in the
combined table, 2 per chunk.**

The consequence is a real trap: **re-running `combine_visits` drops all 28 columns**, and
they have to be restored with `--merge --skip-produce`. The unified attach script
(§6) removes this fragility.

One column is dead: **`cam_DomeXMinusTemp` is 0.0% finite** across all 3385 visits, while
its sibling `cam_DomeYMinusTemp` is 98.6%. Either the sensor does not exist or the field
name is wrong.

## 2. ConsDB structure — pivoted versus unpivoted

Two shapes, and the difference matters for both querying and documentation.

**Pivoted** — `efd_lsstcam.exposure_efd`, one row per exposure, real named columns.
Documented in the [`efd_lsstcam` schema browser](https://sdm-schemas.lsst.io/efd_lsstcam.html)
with types and units.

**Unpivoted** — `efd_lsstcam.exposure_efd_unpivoted`, one row per
(exposure, property, field). The schema browser documents only its **7 generic columns**
(`day_obs`, `seq_num`, `exposure_id`, `property`, `field`, `value`, `created_at`) and gives
**no controlled vocabulary for `property`**. So the property names below are undocumented
and were found with `SELECT DISTINCT property`, which returns 13 values.

The six AOS array properties:

| property | quantity | fields per exposure |
|---|---|---|
| `mt_logevent_aggregated_dof` | Trim, `aggregatedDoF0..49` | 50 |
| `mt_m1m3_applied_elevation_forces_mean` | M1M3 elevation LUT, `zForces` | 156 |
| `mt_m1m3_applied_azimuth_forces_mean` | M1M3 azimuth LUT | 156 |
| `mt_m1m3_applied_thermal_forces_mean` | M1M3 thermal LUT | 156 |
| `mt_m2_axial_force_lut_gravity_mean` | M2 gravity LUT, `lutGravity` | 72 |
| `mt_m2_axial_force_lut_temperature_mean` | M2 thermal LUT | 72 |

### Two ConsDB query bugs to route around

- `WHERE science_program LIKE 'BLOCK-T539%'` returns **HTTP 500**. Use exact `=` and
  filter client-side after `SELECT DISTINCT science_program`.
- A cross-database `LEFT JOIN` between `cdb_lsstcam` and `efd_lsstcam` returns **HTTP
  500**. Same-database joins are fine. Split into two queries.

## 3. Environmental telemetry — ConsDB, and wind is free

Measured on the 500 most recent FAM visits, all in `efd_lsstcam.exposure_efd`:

| ConsDB column | our name | finite | range | unit |
|---|---|---|---|---|
| `mt_salindex111_temperature_0_mean` | `cam_air_temp` | 88.6% | 8.19–17.93 | deg C |
| `mt_salindex112_temperature_0_mean` | `m2_air_temp` | 88.6% | 7.34–17.05 | deg C |
| `mt_salindex113_temperature_0_mean` | `m1m3_air_temp` | 88.6% | 8.78–18.99 | deg C |
| `mt_salindex301_temperature_0_mean` | `outside_temp` | 88.6% | 7.12–18.25 | deg C |
| `mt_salindex122_temperature_6` | `tma_truss_temp_pxpy` | 88.6% | 8.07–17.26 | deg C |
| `mt_salindex110_sonic_temperature_mean` | `sonic_temperature` | 88.6% | 10.53–19.57 | deg C |
| `mt_salindex110_wind_speed_magnitude_mean` | `wind_speed_inside` | 88.6% | 0.30–16.95 | m/s |
| `mt_salindex301_airflow_speed_mean` | `wind_speed_outside` | 88.6% | 0.82–11.55 | m/s |
| `mt_salindex301_airflow_direction_mean` | `wind_dir_outside` | 88.6% | 3.38–354.15 | deg |
| `mt_salindex110_wind_speed_0_mean` | `wind_inside_x` | 88.6% | −2.55–4.67 | m/s |
| `m1m3_stress` | `m1m3_stress` | **0.0%** | — | |
| `m2_stress` | `m2_stress` | **0.0%** | — | |

**Wind and airflow are fully available and currently thrown away.**
`aos/code/aos_consdb_efd.py` already defines all seven wind columns plus
`sonic_temperature`, but nothing in the FAM pipeline calls that module, so none of them
reach `visits.parquet`. Adding them is a one-line change to the attach script's column
list — no new query.

The uniform 88.6% is the commissioning-history effect: these are present for recent
visits and absent for the earliest ones.

## 4. Commanded degrees of freedom — the important finding

**No DOF column of any kind is in the pipeline output.** `visits.parquet` (58 columns),
`fits.parquet` (456) and the MI-refit `fits.parquet` (653) contain no Trim, Tweak or LUT.
Name searches for `dof`, `trim`, `tweak`, `lut`, `hex`, `bend` and `offset` return only
false positives: `m1m3_air_temp`, `m2_delta_t`, `cam_m1m3_delta_t`, and about 200
`m1m3_tc_*` / `m1m3_dt_*` **thermocouple** channels.

### Terminology — the three are different quantities

Per `../../notes/claude-memory/aos-dof-terminology.md`:

- **optical_state** — DOF recovered from the measured wavefront deviation through the
  sensitivity-matrix SVD. Computed here, not telemetry.
- **Tweak** = `PID(optical_state)` — the per-iteration correction MTAOS emits.
- **Trim** — the accumulated offset, `Trim_(i+1) = Trim_i + Tweak`, reported by the EFD
  topic `lsst.sal.MTAOS.logevent_degreeOfFreedom` as `aggregatedDoF0..49`.

**Tweak has no EFD topic and no ConsDB property.** It is derived by differencing
consecutive Trim values, `Tweak_i = Trim_i - Trim_(i-1)`.

The `event_id` array returned by `aos_trim.fetch_aggregated_dof_for_visits` (the `visitId`
of the source `degreeOfFreedom` event) distinguishes the two cases that matter:

- consecutive visits sharing one event mean the AOS applied **no new correction**, so
  Tweak is **0.0** — a real measurement, not missing data;
- **NaN** is reserved for genuinely unknown values: the first visit of a chunk, or one
  whose Trim or event id could not be resolved.

Measured on `20260514_20260731` (54 visits): 53 known, 8 with a non-zero correction, 45
with none applied, 1 NaN (the first visit). The round-trip invariant
`Trim_i = Trim_0 + cumsum(Tweak)` holds to 1e-6 in DOF units, confirming Trim and Tweak
are mutually consistent.

### ConsDB carries Trim, but never on FAM exposures

Coverage of `mt_logevent_aggregated_dof` by `img_type`, joining
`exposure_efd_unpivoted` against `cdb_lsstcam.exposure`, all exposures since 2025-04:

| `img_type` | exposures with Trim |
|---|---|
| `science` | 46 562 |
| `acq` | 183 |
| `dark` | 10 |
| **`cwfs`** | **6** |
| `engtest` | 5 |
| `flat` | 1 |

FAM visits are `img_type='cwfs'`. An `exposure_id` join therefore recovers Trim for
**1 of 3385** visits. `visit1_efd_unpivoted` behaves identically (0 of 600 sampled).

The same gate applies to the *documented* pivoted hexapod columns. Measured on 3000
science, 156 cwfs and 777 acq exposures since 2026-06:

| ConsDB column | quantity, unit | science | cwfs | acq |
|---|---|---|---|---|
| `camera_hexapod_aos_corrections_{x,y,z}` | Trim, camera hexapod, µm | 32.7% | **0.0%** | **0.0%** |
| `camera_hexapod_aos_corrections_{u,v,w}` | Trim, camera hexapod, deg | 32.7% | **0.0%** | **0.0%** |
| `m2_hexapod_aos_corrections_*` | Trim, M2 hexapod, µm / deg | 32.7% | **0.0%** | **0.0%** |
| `camera_hexapod_compensation_offset_z` | hexapod **LUT**, µm | 5.0% | 7.7% | 3.6% |
| `m2_hexapod_compensation_offset_z` | hexapod LUT, µm | 1.4% | 0.0% | 0.6% |
| `mt_hexapod_uncompensated_position_z_mean` | raw hexapod position | **0.0%** | 0.0% | 0.0% |

Note the three distinct behaviours. `aos_corrections` and `aggregated_dof` are both
**Trim** and are science-gated, landing on *exactly the same* exposures.
`compensation_offset` is the **LUT**, is low everywhere, and is *not* img_type-gated.
`uncompensated_position` appears entirely unpopulated even on science exposures.

The OFC DOF layout uses only 5 hexapod axes per hexapod (z, x, y, u, v); ConsDB also
carries `w`, which `aos_consdb_efd.HEX_AXES` correctly drops.

### It is not about closed-loop operation, and not about time coverage

Both plausible explanations were tested and fail:

- **Closed loop.** `BLOCK-T539` (`infocus_initial_alignment`, `acq`/`cwfs`, MTAOS
  demonstrably running) has Trim on **3 of 900** exposures; `BLOCK-T539_hexapods` on **0
  of 632**. The 3 that do are metadata-identical to hundreds that do not — same
  `img_type`, `exp_time=30.0 s`, `target_name=Rubin_SV_225_-40`,
  `observation_reason=infocus_initial_alignment`.
- **Time coverage.** On `day_obs=20251214` a FAM block ran mid-night at `seq_num`
  745–1067. That night had 198 exposures carrying Trim, **187 of them interleaved inside
  that span**, and **zero** on any of the 8 FAM exposures. Two other mid-night FAM nights
  (`20260409`, `20260419`) show the same zero intersection. FAM does not always run first
  in the night, so time of night is not the discriminator — `img_type` is.

**The EFD data is present.** Querying `lsst.sal.MTAOS.logevent_degreeOfFreedom` directly
for 2026-07-11:

| window (UTC) | MTAOS events | dof5 range (µm) |
|---|---|---|
| FAM block, 23:00–24:00 | **52** | −1481.83 … 1518.17 |
| science, 00:13–01:13 | 31 | −342.19 … −119.72 |

More events during FAM than during science, over a far wider excursion — as expected,
since FAM deliberately offsets the optical state.

**Likely mechanism.** On a science night Trim lands on a strict subset of science
exposures (247 of 822 on 2026-07-11), spaced about every third `seq_num`, consistent with
"an MTAOS event fell inside this exposure's window". Combined with 0% on `cwfs`/`acq`
despite 52 events in that hour, the transform appears to (a) require the event to fall
within the exposure window rather than doing an as-of-most-recent lookup, and (b) apply a
science-visit selection that excludes AOS image types.

**Consequence for code:** the as-of-time EFD lookup in
`aos_trim.fetch_aggregated_dof_for_visits` — anchor on ConsDB `obs_start` (TAI), then
`getMostRecentRowWithDataBefore` on the topic — is **required**. It must not be
"simplified" into a ConsDB join.

### The mirror LUTs, by contrast, join perfectly

On the same FAM exposures where Trim fails:

| property | coverage | fields |
|---|---|---|
| `mt_m1m3_applied_elevation_forces_mean` | **400/400** sampled visits | 156 axial forces |
| `mt_m2_axial_force_lut_gravity_mean` | **400/400** sampled visits | 72 axial forces |

So the exposure rows themselves exist in the transform; the gap is specific to the AOS
Trim quantities. `aos_trim.fetch_mirror_lut_for_visits` converts these axial forces to
bending-mode amplitudes with ts_ofc `BendModeToForce.bending_mode`, mapping to DOF 10–29
(M1M3) and 30–49 (M2).

**Unverified assumption**, carried in that function's own docstring: that the EFD force
arrays share the actuator order of the ts_ofc influence matrix, and that `bending_mode`
returns at least 20 modes per mirror. Worth checking before the bending amplitudes are
trusted quantitatively.

## 5. EFD time windows — currently inconsistent

`summit_utils` provides `getEfdData`, which builds the query window for you (from a
`dayObs`, an explicit begin/end, a timespan, a TMA event, or a Butler exposure record)
with `prePadding` / `postPadding`. In this repo it is used in **one** file,
`guider/code/guider_utils.py`. Six files call the raw `select_time_series` and build their
own window; `expRecordToTimespan` is used nowhere.

The result is four unreconciled padding conventions for the same job:

| file | padding |
|---|---|
| `olr/code/telemetry.py` | `DEFAULT_TEMP_WINDOW = 0.2 s` |
| `olr/code/nightly_table.py` | `time_window` and `temp_time_window` both 0.2 s |
| `aos/code/run_backfill_camera_telemetry.py` | `--pad-sec`, default 120 s |
| `aos/code/aos_trim.py` | `buffer_hours` plus a 60 s tail |

A 600× spread. Some variation is legitimate — a high-rate M2 `axialForce` stream needs a
different window than a slow ESS temperature — but at present it is accidental and
undocumented rather than reasoned.

Two reasons `getEfdData` is not a drop-in replacement:

- **It queries one window per call.** For 3385 visits that is 3385 round-trips.
  `aos_trim` and `olr/telemetry` deliberately query **once per night in bulk** and then do
  as-of lookups in memory; that is the fix for what previously ran the OLR night table out
  of batch wall time.
- **`expRecordToTimespan` needs a Butler `DimensionRecord`**, whereas everything here
  anchors on ConsDB `obs_start`/`obs_end`, often with no Butler client in hand. It is also
  **deprecated** in `efdUtils` (removal after `w_2026_01`) and now lives in
  `lsst.summit.utils.dateTime`.

`makeEfdClient` and `getMostRecentRowWithDataBefore` *are* used consistently and should
stay.

## 6. Where the code lives, and where it is going

Today, all of it is under `aos/code/` even though six topics query the EFD or ConsDB:
**29 tracked Python files across `aos`, `olr`, `blocks`, `optatmo`, `guider` and `common`
mention EFD or ConsDB, and 16 of them construct their own client.** Duplicated concretely:
12 hardcoded ConsDB URL literals in two forms
(`http://consdb-pq.consdb:8080/consdb` ×7, `https://usdf-rsp.slac.stanford.edu/consdb` ×5)
and token-file handling reimplemented in 5 files.

The in-pod URL does not resolve outside the RSP, which is a live failure mode, not a
hypothetical.

| module | role | destination |
|---|---|---|
| `aos/code/aos_trim.py` | Trim, hexapod and mirror LUT fetchers, `make_consdb_client` | client layer to `common/`, AOS physics stays |
| `aos/code/aos_consdb_efd.py` | ConsDB transformed-EFD bulk path | `common/` |
| `aos/code/aos_state.py` | per-visit state helpers, `DOF22` | stays in `aos/` |
| `aos/code/run_backfill_thermal.py` | thermal repair, rewrites per-chunk visits in place | folded into the unified attach |
| `aos/code/run_backfill_camera_telemetry.py` | camera-body sidecars, merges to combined only | folded into the unified attach |

The two backfills exist because thermal retrieval fails when `mktable` runs on a Slurm
node with no EFD access, while the expensive donut streaming succeeds. Both need a node
where the EFD and ConsDB resolve: the RSP terminal or a slaciana/slacrd interactive node,
**not** a batch node. Their split — one rewriting per-chunk files, the other writing
sidecars and merging only to the combined table — is the source of the `cam_*` fragility
in §1, and is what the unified script replaces.

## See also

- [Processing](../README.md#processing) — the chunk build these quantities attach to
- [`status/dof_telemetry_availability.md`](status/dof_telemetry_availability.md) — the DOF survey in detail
- [`studies/fam_processing.md`](studies/fam_processing.md) — the chunk-status review tooling
- [`efd_lsstcam` schema](https://sdm-schemas.lsst.io/efd_lsstcam.html) · [`cdb_lsstcam` schema](https://sdm-schemas.lsst.io/cdb_lsstcam.html)
- `../../notes/claude-memory/aos-dof-terminology.md` — optical_state vs Tweak vs Trim
