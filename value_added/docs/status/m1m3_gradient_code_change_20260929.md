# M1M3 thermal gradients: what the 2026-09-29 pull changed

> **Status:** current · **Last updated:** 2026-09-29 · **Kind:** working state (impact assessment)

`aos_efd.duckdb` was built with `ts_m1m3_utils` at **v0.6.2** (`956f732`, 2026-05-05).
A `git pull` on 2026-09-29 moved it to **v0.6.2-7-g57b153b** (`57b153b`, 2026-08-21),
which refactored the gradient plane fit out of `thermocouples.py` into a new
`thermal_gradients.py`. The old commit is recoverable from the reflog and recorded in
`common/output/packages.lock.prepull-20260929.txt`.

**The table is a single code vintage.** `fetch_log` shows all 366 nights of the
`gradients` group fetched between **2026-09-14 and 2026-09-22** (56 + 18 + 136 nights,
plus a 156-night retry on 09-22 that returned rows for all 156). The latest `attempted_at`
in the whole database is `2026-09-22 15:47:41`, and no group was fetched on or after
09-23. So every stored gradient predates the pull and came from `956f732` — there is no
mixed-vintage subset to hunt for. Re-check this with `fetch_log` before concluding
otherwise if any partial rebuild happens later.

## Bottom line

**The stored gradient values do not need recomputing because of the code change.**
The fitted coefficients are numerically identical. There is a separate, small
*configuration* change that does affect them — see
[The AirNozzles change](#the-airnozzles-change-this-one-does-matter).

## Contents

- [What the code change did](#what-the-code-change-did)
- [The error bug you dodged](#the-error-bug-you-dodged)
- [The AirNozzles change](#the-airnozzles-change-this-one-does-matter)
- [What is not affected](#what-is-not-affected)

## What the code change did

The builder path is unchanged in shape: `common/ess_telemetry.get_m1m3_gradients` calls
`ThermocoupleAnalysis.load(start, end, time_bin=30)` and reads `.xyz_r_gradients`, which
runs `calculate_gradients_xyz_r()` with its defaults `use_3d_dataset=True`,
`remove_nonstandard_cells=True`, `radius_limit=None`. All of that still holds.

Three real differences in the fit itself:

1. **Coefficients: identical.** Old code solved the normal equations
   `lstsq(AᵀA, Aᵀt)`; new code solves `lstsq(A, t)` directly. Same least-squares
   solution. Verified numerically on a synthetic 3-plane dataset: max
   |Δβ| = 7e-15, i.e. floating-point noise. **The four stored columns are unaffected.**
2. **Degrees of freedom:** old `dof = max(n - ncols, 1)`; new `dof = n - rank`. Only
   differs for a rank-deficient design, which does not occur here.
3. **NaN handling (new):** the new `fit_plane_gradients` drops non-finite temperatures
   and positions, and returns all-NaN if fewer than `nparam+1` sensors survive. The old
   code fed NaNs straight into `lstsq`, so a single bad thermocouple in a time bin
   poisoned the whole fit to NaN. **This is a genuine improvement**, and it means the new
   code will return a *finite* gradient for some time bins where your DB currently
   holds NaN. Of 213,704 rows, 193,154 have a gradient — so ~20,550 are NaN. Some
   fraction of those are recoverable now; the rest are real telemetry gaps (the
   pre-2025-11 era boundary documented in `build_progress.md`).

## The error bug you dodged

The old `calculate_gradients_xyz_r` had a copy-paste bug in the Cartesian branch:

```python
sigma2 = float(np.sum((y - y_hat) ** 2) / dof)   # `y` is the y-COORDINATE array
```

`y` there is the thermocouple y positions in metres, not the temperatures — the residual
was computed against the wrong vector. (The radial branch below it correctly used
`temperature - y_hat`.) On a synthetic case this inflated every Cartesian error bar by a
factor of ~586, and the factor is data-dependent, so it was not even a constant scale.

**You dodged this entirely:** `visit_telemetry` stores only
`m1m3_{x,y,z,radial}_gradient_c_per_m` and no `*_err` or `intercept` columns, so the
corrupted uncertainties never entered the database. If you ever add error columns, take
them from the new code only.

## The AirNozzles change (this one *does* matter)

The same pull moved `ts_config_mttcs` (`996537b` → `ae5cca1`), and commit `fde6022`
"Update air nozzle orifice diameters" rewrote `MTM1M3TS/v1/tables/AirNozzles.csv`.

That table feeds `nonstandard_thermocouples()`, which decides which thermocouples are
*excluded* from the fit when `remove_nonstandard_cells=True` — your default. So the
sensor set changed even though the code did not.

Of 1650 nozzles, **4 changed type** (314 changed orifice diameter only, which the
gradient fit ignores):

| Nozzle | Old | New | Excluded before → after |
|---|---|---|---|
| A149 | `INSTALLED` | `COVERED` | no → **yes** |
| B1 | `INSTALLED` | `COVERED` | no → **yes** |
| C47 | `INSTALLED` | `SUPER_SHORT` | no → **yes** |
| C46 | `SUPER_SHORT` | `INSTALLED` | **yes** → no |

Net: the excluded set goes from 196 to 198 cells, +3 / −1.

**Expected impact: small but nonzero.** Three of ~150 fitted sensors leave the fit and
one rejoins, out of a well-conditioned plane fit — so a per-visit shift at the level of a
few times 1e-3 °C/m on gradients whose operational band is ±0.4 (x, y) and ±0.1
(radial, z). That is below the band edges but not obviously below your analysis
sensitivity.

Note this cuts the other way from the code change: it means the *old* values were fit
with a nozzle table that has since been corrected. Whether to rebuild is a judgement
call about how much a few-times-1e-3 °C/m systematic matters downstream.

**Suggested check before committing to a full rebuild.** Rebuild one or two nights with
the new code and compare, rather than reprocessing 366 nights on principle:

```bash
cd ~/notebooks/rubin-work
python value_added/code/build_efd_db.py --groups gradients \
    --day-obs 20260714 --db /tmp/grad_new.duckdb
```

then join on `(day_obs, seq)` against `visit_telemetry` and look at the distribution of
differences in the four columns, plus how many NaNs became finite. If the spread is
few-times-1e-3 °C/m as expected, the `AirNozzles` fix is the whole story and you can
decide on cost; if it is larger, something else is in play and this note is wrong.

## What is not affected

- **`ThermocoupleCache` / `max_missing` 12 → 3** (`be42a7c`): a new class for live
  EFD-streaming use. `ThermocoupleAnalysis.load` does not go through it. No effect
  on the builder.
- **`m1m3_thermal_r2`** (211,922 rows): built by `build_m1m3_thermal_r2.py`, which does
  its own fitting and does not call `ThermocoupleAnalysis`. Unaffected by this pull.
- **`bulk_glass_temperature_metrics`** (the `mean_glass_temp` path): the plane-fit
  refactor did not touch it.
