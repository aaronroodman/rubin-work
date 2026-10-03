# Build progress

> **Status:** current · **Last updated:** 2026-10-03 · **Kind:** working state (build log)

What has been built into the value-added database, what is known sparse, and what failed.
Row counts and spans read from the live database's `fetch_log` and `column_coverage` on
2026-09-24; the `optical_state` variant table and the item 2 section refreshed 2026-10-03.

## Contents

- [`visit_telemetry` coverage](#visit_telemetry-coverage)
- [What the 2025 nights do and do not carry](#what-the-2025-nights-do-and-do-not-carry)
- [Sparse columns to check before using](#sparse-columns-to-check-before-using)
- [`optical_state` and `fam_dz`](#optical_state-and-fam_dz)
- [Mirror LUT zero-fill on a dropped actuator](#mirror-lut-zero-fill-on-a-dropped-actuator)
- [`m1m3_thermal_r2` coverage](#m1m3_thermal_r2-coverage)
- [Rebuilding](#rebuilding)

## `visit_telemetry` coverage

**366 nights**, `day_obs` 20250415 to 20260714, 213,704 exposures — every night that has
`lsstcam` exposures in the Consolidated Database (ConsDB) over that span. All eight groups
are recorded `ok` for all 366 nights, with no night in `fetch_log` in the `error` state.

`ok` means the fetch ran to completion, **not** that the telemetry existed: a group whose
source topic was not yet reporting stores NaN and is still `ok`. The next section gives the
one era boundary that matters.

## What the 2025 nights do and do not carry

Over the 154 nights from `day_obs` 20250415 to 20251101, 84,462 exposures, several
instrument subsystems were not yet publishing, so those columns hold NaN. One representative
column per group, counted as **non-null exposures** (dimensionless count, out of the era's
exposure total):

| representative column | 20250415–20251101 (of 84,462 exposures) | 20251102–20260714 (of 129,242 exposures) |
|---|---|---|
| `dof0` (Trim) | 84,440 | 129,240 |
| `cam_ChgrYMinusRtnAirTemp` (camera body) | 83,282 | 127,542 |
| `lut_dof0` (hexapod LUT) | 76,982 | 125,600 |
| `m1m3_z_gradient_c_per_m` | 64,271 | 128,883 |
| `turb123_speed_mag_ms` | **0** | 90,817 |

**The 2025 nights carry no turbulence data at all.** The four Telescope Mount Assembly
top-ring sonic anemometers (`ESS.airTurbulence`, salIndex 123 to 126) were not reporting, so
every `turb12*` column is NaN across all 84,462 exposures. A turbulence-conditioned analysis
is therefore restricted to `day_obs` 20251102 and later, and a join that does not check will
silently drop the whole 2025 sample.

M1M3 bulk thermal gradients are absent over 36 nights, 19,849 exposures, `day_obs` 20250415
to 20260221. Two causes, both genuine telemetry gaps that store NaN:

- No thermocouple reported at all — the in-glass grid was not publishing before roughly
  `day_obs` 20250513, with isolated later nights 20250610 and 20250713.
- A missing cold-junction reference channel on `day_obs` 20260217 (`coldJunction114`) and
  20260221 (`coldJunction117`).

193,154 of 213,704 exposures carry gradient values.

## Sparse columns to check before using

`column_coverage` tracks all 193 telemetry columns of `visit_telemetry`; the 194th column is
the `visit_id` primary key, which is never NULL and is not tracked.

Coverage is **not** uniform across those 193 columns. The `turbulence` group is the one that
bites: its columns range from 90,814 to 213,704 non-null values out of 213,704 exposures, so
a turbulence-conditioned analysis can silently lose up to **58%** of the sample without any
error. Most of that loss is the 2025 era above, where the anemometers read nothing.

`lut` is mildly sparse at 202,582–205,771 non-null, about 95% of exposures.

Check before trusting a column:

```sql
SELECT column_name, units, source, n_non_null, first_day_obs, last_day_obs
FROM column_coverage
WHERE group_name = 'turbulence'
ORDER BY n_non_null;
```

## `optical_state` and `fam_dz`

`optical_state` holds 90,695 rows, `day_obs` 20251102 to 20260713 — a genuinely narrower
span than `visit_telemetry`, which reaches back to 20250415. The 2025 nights have telemetry
but no recovered optical state, so a join of the two drops them. Four variants are
**registered** in `state_variant`:

| variant_id | rows | state |
|---|---|---|
| `v50_34__batoid__consdb_v1` | 90,695 | built, but **pre-item-2 vintage**: no `_olr` or `fwhm_cwfs_arcsec` columns. Awaiting the full-span rerun |
| `v50_34_rbr__batoid__consdb_v1` | 916 | one night (20260318), built as the costing check |
| `v22_12__batoid__consdb_v1` | 0 | registered, never built |
| `v50_34__miw__consdb_v1` | 0 | registered, never built |

The **Measured Intrinsic Wavefront (MIW) route is not built**, only the batoid-design route.
This is the trap in this table: `efd_db.optical_state('v50_34__miw__consdb_v1')` returns an
empty DataFrame rather than raising, so an analysis that names the MIW variant gets zero rows
and no error. Building it is outstanding work, deferred to todo item 6.

### Item 2 full-span build, pending submission

Three batoid variants are to be built over the full 366-night span, replacing the existing
`v50_34__batoid__consdb_v1` rows rather than extending them, so all three come from identical
code against identical inputs. **Measured** on `day_obs` 20260318 (916 exposures, 915
recovered — near the busiest night; the mean is 584 exposures/night):

| scheme | median CWFS FWHM contribution [arcsec] | wall clock [s/night, 916 exposures] | median `resid_rms_um` [µm of wavefront] |
|---|---|---|---|
| `22_12` | 0.4372 | 39.8 | 0.1165 |
| `50_34` | 0.2428 | 39.1 | 0.0587 |
| `50_34_rbr` | 0.2650 | 59.0 | 0.0812 |

Note the `resid_rms_um` column means different things across those rows — subspace residual
for the two truncated schemes, achieved residual for the range-bounded one. See
[`../schema.md`](../schema.md).

At 24 nights per shard that is 16 shards per variant, 48 jobs in all; a mean shard runs about
10 min (15 min for RBR) and the worst-case night-heavy shard about twice that, inside the
1 h default `SB_TIME`.

Two verifications already done, both on 20260318:

- The rebuilt `50_34` deviation-recovered state reproduces the existing variant to
  **6.0e-14** (dimensionless v-mode amplitude) and **3.4e-11** µm/arcsec in DOF across all
  915 matched visits, so the rerun changes nothing but the added columns.
- `v_modes_olr = v_modes − v_modes_trim` to **1.3e-15** (dimensionless), and the optical
  state `Trim − Deviation` is its exact negative, matching the `thermal_focus`
  `MEASURED_SIGN = -1.0` convention.

The pre-item-2 rows are archived at
`output/scratch/archive/v50_34__batoid__consdb_v1_pre_item2.parquet` (90,695 rows verified,
95 MB, gitignored) pending their removal before the rerun.

`fam_dz` holds 2,528 visits over `day_obs` 20250415 to 20260713, under one registered
`fam_variant_id`. All 2,528 join to `visit_telemetry` on `visit_id`, and all 2,528 carry a
Trim value (`dof0` non-null), so a DOF-conditioned join loses nothing.

A **turbulence**-conditioned join is the one that costs: only 1,026 of the 2,528 FAM visits
have `turb123_speed_mag_ms`, because 1,000 of them fall in the pre-20251102 era where the
anemometers were not reporting. See
[What the 2025 nights do and do not carry](#what-the-2025-nights-do-and-do-not-carry).

**The apparent v-mode-1 sign error is resolved.** Both hexapod dz axes of v-mode 1 are
genuinely negative — camera degree of freedom (DOF) 5 is −8.9144254e-04 and M2 DOF 0 is
−9.1026032e-04, both dimensionless v-mode-1 amplitude per µm of hexapod dz — and the helper
that returns the conversion constant returns a magnitude only, 0.5 × (|c5| + |c0|), by
design: the sign is carried separately by a `MEASURED_SIGN = -1.0` (dimensionless) constant.
There was no sign error in that helper.

**One sign disagreement remains genuinely unverified.** `fam_dz.v_modes` is built by a
different engine than `optical_state.v_modes`, and the two disagree on the **sign of
v-mode 1**: `ofc_svd.vmodes()` divides by the positive singular values, while the arbitrary
per-mode sign convention on the other route comes out opposite. Do not compare or combine
v-mode 1 across the two tables without resolving the sign first. The diagnosis is in
[`../../aos/docs/status/rerun_needed.md`](../../aos/docs/status/rerun_needed.md) under "The
two v-mode engines disagree on the sign of v1";
[`../../thermal_focus/docs/thermal_focus.md`](../../thermal_focus/docs/thermal_focus.md)
carries the resolved half of this history.

A separate, earlier change to the v-mode basis also flips v1 for **every stored** `v1`,
`v1_lut` and `v1_trim` — see `rerun_needed.md`. Stored v-mode values therefore need
regenerating, not just reinterpreting.

## Mirror LUT zero-fill on a dropped actuator

The forty mirror look-up-table (LUT) degrees of freedom `lut_dof10` to `lut_dof49` have no
published value: they are derived from the M1M3 and M2 axial forces, in newtons, through
`common/dof_telemetry.bending_modes_from_forces`. A single dropped actuator is routine on
M1M3, and the converter currently substitutes **0 N** for any actuator reporting NaN, which
is what every stored value was built with.

Zero force is not what the hardware does. M1M3's force-balance system redistributes a failed
actuator's load onto its neighbours, so the physical substitute is a redistributed force
pattern, not zero. The stored bending amplitudes (dimensionless) are therefore slightly
biased on exposures with a dropped actuator, by an amount that has not been quantified.

Modelling the redistribution, and rebuilding the affected `lut_dof10` to `lut_dof49` values,
is outstanding work. The alternative considered and rejected was propagating NaN, which
would discard every visit with one dead actuator.

## `m1m3_thermal_r2` coverage

**365 nights**, `day_obs` 20250415 to 20260714, 211,922 exposures, of which **192,079 carry a
fit**. Built by `code/build_m1m3_thermal_r2.py` over 331 nights with usable M1M3 thermocouple
telemetry; the remaining 34 nights hold NaN.

The NaN nights are 32 contiguous nights `day_obs` 20250415 to 20250518 (18,995 exposures),
where the in-glass thermocouple grid was not publishing at all, plus two isolated nights,
20250610 (9 exposures) and 20250713 (839 exposures). The reason is logged per night as the
build runs.

**The quadratic terms cover more exposures than the bulk gradients do** — 192,079 against
193,154 over the whole database, but on the thermal-focus science sample 100.0% of visits
against 99.9%. The bulk gradients come from
`lsst.ts.m1m3.utils.ThermocoupleAnalysis.calculate_gradients_xyz_r`, which returns NaN for a
whole time bin if any one thermocouple dropped out; the quadratic fit instead groups the time
bins by their unique finite-sensor pattern and fits each group against the sensors it actually
has, so a single dropped thermocouple costs one column of the design matrix rather than the
row. The two nights 20260217 and 20260221, which the bulk gradients lose to a missing
cold-junction reference channel, are for the same reason still absent here: that channel is
needed to convert any thermocouple at all.

Two nights failed transiently during the first pass, `day_obs` 20260424 and 20260628, both
with `ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)` out of the
Engineering Facility Database (EFD) query. Each night is wrapped in its own `try`/`except`
so one bad night does not stop the build, and both succeeded on a `--refetch` retry — 822 and
958 exposures fitted respectively. No nights remain failed.

Rebuilding is per night, and one night per EFD query is a hard constraint rather than a tuning
choice, since a multi-night thermocouple span times out:

```bash
cd ~/notebooks/rubin-work
python -u value_added/code/build_m1m3_thermal_r2.py --day-obs 20250415-20260714 --resume
python -u value_added/code/build_m1m3_thermal_r2.py --day-obs 20260424 --refetch
```

Redirect the output and Python buffers it, so use `python -u` or the per-night progress lines
do not appear until the run ends.

## Rebuilding

Shards are written per night-range and merged, because DuckDB's file lock is process-wide
and excludes readers as well as writers — several processes cannot write one database file.

```bash
cd ~/notebooks/rubin-work/value_added
./code/run_build.sh --what telemetry --day-obs 20251103-20260418 --chunk 24
python code/merge_db_shards.py --shards 'output/shards/telemetry_*.duckdb'
```

`--what telemetry` is EFD-bound and runs `--mode local` only: the EFD resolves from the RSP
and from s3df **interactive** nodes (`slacrd`, `sdfiana*`) but not from batch compute nodes,
so a batch job would fetch nothing. `--what state` is ConsDB-bound and does work in batch.

Wall time per night is dominated by the `trim` group, and it is wildly uneven — most nights
finish in tens of seconds while the worst take close to an hour, scaling with the number of
`degreeOfFreedom` events rather than with the exposure count. Budget for the tail: a night
of about 1,800 exposures has taken roughly 3,200 s (53 min) for `trim` alone. Use `--resume`,
which skips any `(day_obs, group)` pair already recorded `ok` or `empty`, so an interrupted
build restarts without refetching. Do not wrap the build in an external `timeout` shorter
than the tail — killing it mid-transaction rolls the night back and leaves a write-ahead log
behind with no committed rows.

**Batch submission is a hard must-ask** — see the root `CLAUDE.md`.
