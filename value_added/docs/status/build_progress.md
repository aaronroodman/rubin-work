# Build progress

> **Status:** current · **Last updated:** 2026-10-04 · **Kind:** working state (build log)

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
- [Corrupt `vt_day_obs` index, open](#corrupt-vt_day_obs-index-open)
- [Rebuilding](#rebuilding)

## `visit_telemetry` coverage

**366 nights**, `day_obs` 20250415 to 20260714, 213,704 exposures — every night that has
`lsstcam` exposures in the Consolidated Database (ConsDB) over that span. The eight original
groups are recorded `ok` for all 366 nights, with no night in `fetch_log` in the `error`
state. The ninth group, `twilight`, covers 365 of them: `day_obs` 20260620 is blocked by a
[corrupt `vt_day_obs` index](#corrupt-vt_day_obs-index-open).

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

`optical_state` holds 307,389 rows over three built variants, `day_obs` 20250724 to
20260714, 231 nights — a genuinely narrower span than `visit_telemetry`, which reaches back
to 20250415. Nights before 20250724 have telemetry but no corner-WFS quicklook in ConsDB, so
there is no wavefront to invert and they carry no optical state; a join of the two tables
drops them. Four variants are **registered** in `state_variant`:

| variant_id | rows | state |
|---|---|---|
| `v22_12__batoid__consdb_v1` | 102,463 | built full span, item 2 vintage |
| `v50_34__batoid__consdb_v1` | 102,463 | built full span, item 2 vintage; replaced the 90,695 pre-item-2 rows |
| `v50_34_rbr__batoid__consdb_v1` | 102,463 | built full span, item 2 vintage |
| `v50_34__miw__consdb_v1` | 0 | registered, never built |

96,278 rows per variant (94.0%) carry a recovered optical state and so a
`fwhm_cwfs_arcsec`; the other 6,185 are per-visit recovery failures. The **same** visits
succeed and fail in all three variants, so cross-scheme comparisons are paired without
further filtering — but `fwhm_cwfs_arcsec IS NOT NULL` is still the screen to apply, since
an unrecovered row stores NaN rather than NULL in the array columns.

The **Measured Intrinsic Wavefront (MIW) route is not built**, only the batoid-design route.
This is the trap in this table: `efd_db.optical_state('v50_34__miw__consdb_v1')` returns an
empty DataFrame rather than raising, so an analysis that names the MIW variant gets zero rows
and no error. Building it is outstanding work, deferred to todo item 6.

### Item 2 full-span build, done 2026-10-03

Three batoid variants built over the full span, 16 shards each at 24 nights per shard, 48
jobs. All three come from identical code against identical inputs. The results are in
[`../../../olr/docs/scheme_comparison.md`](../../../olr/docs/scheme_comparison.md); measured
wall clock on `day_obs` 20260318 (916 exposures, near the busiest night; the mean is 584)
was 39.8 / 39.1 / 59.0 s per night for `22_12` / `50_34` / `50_34_rbr`.

`resid_rms_um` means different things across the variants — subspace residual for the two
truncated schemes, achieved residual for the range-bounded one. See
[`../schema.md`](../schema.md).

**Twelve of the 48 jobs failed, all in the pre-20250724 era, and the fix was to skip it.**
Shards 01–04 of each variant covered `day_obs` 20250415–20250723, where ConsDB has exposures
but no corner-WFS quicklook. Two defects surfaced there, neither visible on any later night:

- `get_intrinsic_zernikes` raised `RuntimeError: Invalid filter name OTHER:PINHOLE` on
  engineering-mask exposures, crashing shards 01 and 04.
- A night with zero complete corner OPD still wrote rows whose recovered and open-loop
  columns were all NaN — shards 02 and 03 exited 0 having written 11,562 such rows.

Fixed in `80b5578`: the band is screened against `OFCData.intrinsic_zk` rather than an
enumeration of known-bad values, and a night with no complete corner OPD records `empty`.
Those 12 shard files were discarded rather than rerun, since with the fix every night in
their range yields zero rows. Covered by `code/test_build_optical_state.py`.

Three verifications:

- The rebuilt `50_34` deviation-recovered state reproduces the pre-item-2 variant to
  **6.0e-14** (dimensionless v-mode amplitude) and **3.4e-11** µm/arcsec in DOF across all
  915 matched visits of 20260318, so the rerun changed nothing but the added columns.
- `v_modes_olr = v_modes − v_modes_trim` to **1.3e-15** (dimensionless), and the optical
  state `Trim − Deviation` is its exact negative, matching the `thermal_focus`
  `MEASURED_SIGN = -1.0` convention.
- 20260318 reproduces its single-night costing values exactly in the full build, so that run
  was arithmetically sound — but it was a favourable night, and overstated the 50/34 image
  quality gain by about 50% against the full sample.

The pre-item-2 `v50_34__batoid__consdb_v1` rows are archived at
`output/scratch/archive/v50_34__batoid__consdb_v1_pre_item2.parquet` (90,695 rows verified,
95 MB, gitignored). The merge overwrote all of them in place — the new build is a strict
superset, with **0** rows left stale — so the archive is a rollback copy only and can be
deleted once the merged table has been checked.

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

## Corrupt `vt_day_obs` index, open

**Status as of 2026-10-08: open, one night affected, no data lost, low priority.** The equality
index on `visit_telemetry(day_obs)` is corrupt for `day_obs` 20260620. The symptom is that the
two query paths disagree:

```sql
SELECT COUNT(*) FROM visit_telemetry WHERE day_obs = 20260620;                      -- 0
SELECT COUNT(*) FROM visit_telemetry WHERE day_obs BETWEEN 20260620 AND 20260620;   -- 1782
```

The equality predicate uses the index and returns nothing; the range scan bypasses it and
finds all 1,782 rows. The rows are intact — the archive copy agrees, and `fetch_log` records
all eight original groups `ok` at 1,782 rows. **Only the index is wrong.**

A write that touches the night fails with

```
FATAL Error: Invalid Input Error: Failed to delete all rows from index.
Only deleted 0 out of 561 rows.
```

because `upsert_visits` deletes before inserting and the delete matches nothing. That is why
the `twilight` backfill covers 365 of 366 nights: 20260620 is the one it could not write.

**Why this is low priority.** Two things limit the damage. The reader API builds range
predicates — `_day_obs_clause` emits `day_obs >= ? AND day_obs <= ?` — so `efd_db.visits()` and
everything downstream of it already return this night correctly; that is why
`thermal_focus_truss_all.parquet` carries all 1,782 rows. And the night is **all calibration**
(1,766 cbp, 10 dark, 3 bias, 3 flat, no science, acq or cwfs), so the columns it is missing are
meaningless for it anyway.

What remains is a latent trap rather than an active bug: hand-written SQL of the form
`WHERE day_obs = 20260620` returns 0 rows with no error, so a loop over nights would skip it
silently. Prefer a range predicate over an equality one when scanning nights, which the reader
API already does.

The fix is to rebuild the index, which needs a write handle and is a DDL change on the shared
database, so it has not been run:

```bash
cd ~/notebooks/rubin-work/value_added && \
    python -c "
import sys; sys.path.insert(0, 'code')
import efd_db
con = efd_db.open_db(None, readonly=False)
con.execute('DROP INDEX vt_day_obs')
con.execute('CREATE INDEX vt_day_obs ON visit_telemetry(day_obs)')
con.execute('FORCE CHECKPOINT')
print('20260620 rows:', con.execute('SELECT COUNT(*) FROM visit_telemetry WHERE day_obs = 20260620').fetchone()[0])
con.close()"
```

Then fill the night's twilight columns:

```bash
cd ~/notebooks/rubin-work/value_added && \
    python code/build_efd_db.py --day-obs 20260620 --groups twilight
```

Most likely cause, not proven: 20260620 is a 1,782-exposure night, among the largest in the
span, and the `trim` group on a night that size has taken close to an hour. An interrupted
write on such a night is the documented hazard in [Rebuilding](#rebuilding) — the rollback
leaves no committed rows, and here it appears to have left the index inconsistent with the
rows instead.

**It is the only affected night.** All 366 were checked by comparing the equality count
against the range-scan count per night; 20260620 is the single disagreement.

**The rows belong in the table even though the night is all calibration.** `visit_spine` applies
no `img_type` filter by design, so the table is one row per exposure and selecting on-sky types
is the consumer's job — which is how a 68,079-visit science sample comes out of 213,704 rows.
Calibration is not a special case here: 54 of the 366 nights carry no science, acq or cwfs
exposure at all, and calibration is about 32 per cent of the table (flat 36,364, dark 16,929,
bias 13,081, cbp 2,248 over the span). Those rows also carry real telemetry — the truss has a
temperature whether or not the shutter is open on sky, and 20260620 sits at 6.68 to 8.65 °C —
which is what the database-wide truss page of the `thermal_focus` report plots.

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
