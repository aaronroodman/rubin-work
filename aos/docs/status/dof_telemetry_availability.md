# DOF telemetry: Trim, Tweak and LUT — what is available where

> **Status:** current · **Last updated:** 2026-09-08 · **Kind:** working state (availability survey)

Measured on 2026-09-08 against `param_set` `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`
(3385 visits, `day_obs` 20250415–20260713) and the USDF ConsDB at
`https://usdf-rsp.slac.stanford.edu/consdb`.

## The headline

**No degree-of-freedom (DOF) column of any kind is in the pipeline output.** Checked
directly against the parquet schemas:

| table | columns | Trim / Tweak / LUT DOF present |
|---|---|---|
| `output/<ps>/visits.parquet` | 58 | none |
| `output/<ps>/fits.parquet` | 456 | none |
| `output/<ps>/<mi>/fits.parquet` | 653 | none |

Name searches for `dof`, `trim`, `tweak`, `lut`, `hex`, `bend` and `offset` return only
**false positives** — `m1m3_air_temp`, `m2_delta_t`, `cam_m1m3_delta_t`, and roughly 200
`m1m3_tc_*` / `m1m3_dt_*` **thermocouple** channels in the MI-refit table. Those are
temperatures, not degrees of freedom.

The retrieval code exists in `aos/code/aos_trim.py` and `aos/code/aos_consdb_efd.py`, but
`mktable` never calls it. Its only callers are `code/bounce/run_bounce.py` and the sibling
topic `blocks/code/telemetry_pipeline.py`, so every study needing commanded state re-fetches
it itself.

## Terminology first — the three are not interchangeable

Per `../../../notes/claude-memory/aos-dof-terminology.md`:

- **optical_state** — DOF recovered from the measured wavefront deviation through the
  sensitivity-matrix SVD. Computed here, not telemetry.
- **Tweak** = `PID(optical_state)` — the per-iteration correction the controller emits.
- **Trim** — the accumulated offset, `Trim_(i+1) = Trim_i + Tweak`. This is what the EFD
  topic `lsst.sal.MTAOS.logevent_degreeOfFreedom` reports as `aggregatedDoF0..49`.

**Tweak is not a retrievable quantity.** There is no EFD topic or ConsDB property for it,
and correspondingly no fetcher in `aos_trim.py`. It has to be *derived* by differencing
consecutive Trim values, and only across an actual re-alignment — `_dof_at_times` returns
an `event_ids` array (the `visitId` of the source `degreeOfFreedom` event) precisely so
that consecutive visits sharing one event are not differenced to a spurious zero.

## ConsDB now carries Trim and both LUTs — but not on the FAM exposures

Schema browser: [`efd_lsstcam`](https://sdm-schemas.lsst.io/efd_lsstcam.html) and
[`cdb_lsstcam`](https://sdm-schemas.lsst.io/cdb_lsstcam.html), indexed at
[sdm-schemas.lsst.io](https://sdm-schemas.lsst.io/). Note the transformed-EFD page
documents the *table structure* of `exposure_efd_unpivoted` (`property`, `field`, `value`)
but does **not** enumerate which properties populate it, so the property names below were
established by querying, not from the docs.

All six AOS array properties exist in `efd_lsstcam.exposure_efd_unpivoted`:

| property | quantity | fields/visit |
|---|---|---|
| `mt_logevent_aggregated_dof` | Trim, `aggregatedDoF0..49` | 50 |
| `mt_m1m3_applied_elevation_forces_mean` | M1M3 elevation LUT, `zForces` | 156 |
| `mt_m1m3_applied_azimuth_forces_mean` | M1M3 azimuth LUT | 156 |
| `mt_m1m3_applied_thermal_forces_mean` | M1M3 thermal LUT | 156 |
| `mt_m2_axial_force_lut_gravity_mean` | M2 gravity LUT, `lutGravity` | 72 |
| `mt_m2_axial_force_lut_temperature_mean` | M2 thermal LUT | 72 |

**Coverage against this param_set's 3385 FAM visits, joining on `exposure_id`:**

| quantity | visits matched | note |
|---|---|---|
| M1M3 elevation LUT | **400 / 400** sampled | full coverage, 156 axial forces each |
| M2 gravity LUT | **400 / 400** sampled | full coverage, 72 axial forces each |
| Trim | **1 / 3385** | effectively absent on FAM exposure IDs |

Trim is not missing from ConsDB — the property holds **2 338 450 rows spanning
`exposure_id` 2025041400656 to 2026071300861**, which brackets the whole FAM date range.
And it is present *on the same nights*: sampling eight FAM nights gives 81–247 exposures
carrying Trim per night, against 12–55 FAM visits on those nights.

The rows simply sit on **different exposures**. The transform records `aggregated_dof`
against the exposures where the AOS event landed, not against the FAM donut exposures, so
a direct `exposure_id` equality join finds almost nothing. `visit1_efd_unpivoted` behaves
identically (0 / 600 sampled), so moving to the visit-level table does not help.

**Consequence for design.** The LUTs can come from ConsDB with a fast bulk query. Trim
cannot, and needs the as-of-time lookup that `aos_trim.fetch_aggregated_dof_for_visits`
already implements — anchor on the ConsDB exposure `obs_start` (TAI), then
`getMostRecentRowWithDataBefore` on the EFD topic. That is exactly why that function is
written the way it is, and it should not be "simplified" into a ConsDB join.

## Units, for whatever consumes these

- Trim `aggregatedDoF0..49`: dof0–4 M2 hexapod, dof5–9 camera hexapod, dof10–29 M1M3
  bending, dof30–49 M2 bending. Hexapod z/x/y in µm, u/v in arcsec (OFC convention);
  bending modes dimensionless amplitude.
- Hexapod LUT (`fetch_hexapod_lut_for_visits`) returns (n_visits, 10): dof0–4 M2, dof5–9
  camera, **z/x/y in µm but u/v in deg** — the angular axes may differ from the OFC arcsec
  convention, so do not compare u/v against Trim without converting. `lut[:, 5]` (camera
  hexapod dz) is directly comparable to Trim dof5.
- Mirror LUT (`fetch_mirror_lut_for_visits`) converts axial forces to bending amplitudes
  via ts_ofc `BendModeToForce.bending_mode`, mapping to dof10–29 and dof30–49, same order
  and units as Trim. Its docstring carries an unverified assumption that the EFD force
  arrays share the actuator order of the ts_ofc influence matrix.

## Open

- `run_backfill_dof.py` is the agreed fix: add Trim, the LUTs, and a derived Tweak to
  `visits.parquet` the way `run_backfill_thermal.py` adds thermal columns. Not yet written.
- The mirror-LUT actuator-order assumption above wants verifying against ts_ofc before its
  bending amplitudes are trusted quantitatively.
