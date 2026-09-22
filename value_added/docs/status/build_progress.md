# Build progress

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** working state (build log)

What has been built into the value-added database, what is known sparse, and what failed.
Read from the live database's `fetch_log` and `column_coverage` on 2026-09-18.

## Contents

- [`visit_telemetry` coverage](#visit_telemetry-coverage)
- [Two failed nights](#two-failed-nights)
- [Sparse columns to check before using](#sparse-columns-to-check-before-using)
- [`optical_state` and `fam_dz`](#optical_state-and-fam_dz)
- [Mirror LUT zero-fill on a dropped actuator](#mirror-lut-zero-fill-on-a-dropped-actuator)
- [Rebuilding](#rebuilding)

## `visit_telemetry` coverage

**212 nights**, `day_obs` 20251102 to 20260714, 129,242 exposures. Every group is built for
all 212 nights except `gradients`, which has 210:

| group | nights built | status |
|---|---|---|
| `trim`, `tweak`, `lut`, `camera`, `turbulence`, `hexhist`, `wind_derived` | 212 | complete |
| `gradients` | 210 | 2 nights failed, see below |

## Two failed nights

The `gradients` group failed on two nights with a missing M1M3 thermocouple channel:

| day_obs | error |
|---|---|
| 20260217 | `KeyError: "['coldJunction114'] not in index"` |
| 20260221 | `KeyError: "['coldJunction117'] not in index"` |

Two different cold-junction channels, four days apart. The M1M3 bulk thermal gradients are
derived from raw thermocouple telemetry, so a channel absent from the EFD for that night
makes the derivation fail rather than return a partial result. These are almost certainly
genuine telemetry gaps rather than a code defect, but that has not been confirmed — the
builder could equally be assuming a fixed channel list where it should tolerate a missing
one.

Consequence: any analysis using M1M3 gradients has no data for those two nights and will
see NaN. 128,883 of 129,242 exposures have gradient values.

## Sparse columns to check before using

Coverage is **not** uniform across the 194 columns. The `turbulence` group is the one that
bites: its columns range from 90,814 to 129,242 non-null values, so a
turbulence-conditioned analysis can silently lose about **30%** of the sample without any
error.

`lut` is mildly sparse at 125,171–125,600 non-null, about 97% of exposures.

Check before trusting a column:

```sql
SELECT column_name, units, source, n_non_null, first_day_obs, last_day_obs
FROM column_coverage
WHERE group_name = 'turbulence'
ORDER BY n_non_null;
```

## `optical_state` and `fam_dz`

`optical_state` holds 90,695 rows, `day_obs` 20251102 to 20260713. Three variants are
**registered** in `state_variant` but only one is populated:

| variant_id | rows |
|---|---|
| `v50_34__batoid__consdb_v1` | 90,695 |
| `v22_12__batoid__consdb_v1` | 0 — registered, never built |
| `v50_34__miw__consdb_v1` | 0 — registered, never built |

The **Measured Intrinsic Wavefront (MIW) route is not built**, only the batoid-design route.
This is the trap in this table: `efd_db.optical_state('v50_34__miw__consdb_v1')` returns an
empty DataFrame rather than raising, so an analysis that names the MIW variant gets zero rows
and no error. Building it is outstanding work.

`fam_dz` holds 2,528 visits over `day_obs` 20250415 to 20260713, under one registered
`fam_variant`. Note the earlier start date than `visit_telemetry`: the FAM DZ fits reach
back to April 2025, while the telemetry build starts in November 2025. A join of the two on
`visit_id` therefore drops the 2025 FAM visits.

**A known defect affects the v-modes here, found 2026-09-18 and not yet fixed.**
`fam_dz.v_modes` is built by a different engine than `optical_state.v_modes`, and the two
disagree on the **sign of v-mode 1**: `ofc_svd.vmodes()` divides by the positive singular
values, while the arbitrary per-mode sign convention on the other route comes out opposite.
Do not compare or combine v-mode 1 across the two tables without resolving the sign first.
The diagnosis and the fix are in
[`../../aos/docs/status/rerun_needed.md`](../../aos/docs/status/rerun_needed.md) under "Two
v-mode sign errors"; `aos/docs/studies/fam_focus.md` carries the same warning.

A separate, earlier change to the v-mode basis also flips v1 for **every stored** `v1`,
`v1_lut` and `v1_trim` — see the same document. Stored v-mode values therefore need
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

**Batch submission is a hard must-ask** — see the root `CLAUDE.md`.
