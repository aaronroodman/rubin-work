# Thermal response of all 34 v-modes

> **Status:** current · **Last updated:** 2026-10-06 · **Kind:** reference (study)

> **Code:** `code/thermal_vmodes.py`, `code/run_thermal_vmodes.py`, `code/test_thermal_vmodes.py`
> **Output:** `output/thermal_vmodes/`
> **Notebooks:** `notebooks/`

Extends the [thermal-focus](thermal_focus.md) deliverable from v-mode 1 to every v-mode the
50-degree-of-freedom (DOF) recovery retains, asking **which other v-modes carry a component
predictable from telescope thermal telemetry**.

The thermal-focus model predicts one quantity: v-mode 1, which is almost pure uniform
defocus, from five thermal channels. Nothing in that work says whether astigmatism, coma or
any higher mode behaves the same way. This study runs the same fit on all 34 and reports
per-mode skill against a null.

## The response

For mode `k` the response is the **optical state**, `Trim − Deviation`, which is the negative
of the stored open-loop column:

```
y_k [dimensionless v-mode amplitude] = −v{k}_olr = v{k}_trim − v{k}
```

Verified against `thermal_focus_lib.MEASURED_SIGN` to 6.7e-16 (dimensionless v-mode
amplitude) over modes 1 through 34, so this is the v-mode-1 convention generalized rather
than a new one.

**The response is left dimensionless**, not divided into µm of equivalent hexapod dz as the
v-mode-1 deliverable is. That conversion (`DZ_UM_PER_UM_WF`, −63.9902 µm of equivalent
hexapod dz per µm of wavefront) is specific to defocus; no single physical axis stands in for
the higher modes, so a per-mode physical conversion would be invented rather than derived.

## Why this is not just the existing fit run 34 times

**The null matters more than the fit.** Most modes are expected to carry no thermal signal,
so the question is per-mode skill against an intercept-only model, not goodness of fit. Skill
is the fractional reduction in out-of-fold residual normalized median absolute deviation
(nMAD), both terms night-grouped. The null's intercept is the **median** of the training
folds, not the mean — the robust counterpart of the Huber fit it is compared against. A
mean-based null would be pulled by the same one-sided tail the Huber fit is chosen to resist,
flattering the fit.

**34 simultaneous tests need a correction.** `mode_table` applies a Benjamini-Hochberg
false-discovery-rate cut at `q = 0.05` (dimensionless), so a mode is called thermal only if it
survives. Skill has no analytic null distribution here — the response is spatially correlated
between modes and the folds are not independent — so the empirical null is taken from the 34
modes themselves. **This is a screening rule for which modes deserve a closer look, not a
calibrated significance claim**; a mode near the threshold should be confirmed with
`thermal_focus_fit.nested_comparison` on that mode alone.

**The high modes are measurement noise.** Four corner wavefront sensors (CWFS) constrain 84
Zernike values and the recovery's scatter grows with mode index. `noise_floor_table` reports
per-mode within-night against between-night scatter, so a reader does not take a high-mode
slope at face value. Within-night scatter is estimated from successive-visit differences
divided by sqrt(2), which does not assume the state is constant across a night: a slow
thermal drift contributes to the between-night term instead. A mode whose between/within
ratio is near 1 (dimensionless) carries no more night-to-night structure than its own
measurement noise and is not interpretable however it scores.

## Nights are held out whole

Only 2.7% of the Telescope Mount Assembly truss temperature's variance is within-night
(dimensionless, within-night over total), so consecutive visits are near-duplicates in
feature space and a visit-level split leaks. Every fit here uses
`thermal_focus_fit.evaluate`, which is `GroupKFold` on `day_obs`. The measured optimism of a
visit-level split is a factor of 3.1 (dimensionless) for boosted trees and about 1.04 for the
Huber linear fit; see [`thermal_focus.md`](thermal_focus.md).

## Variants

| role | variant | why |
|---|---|---|
| primary | `v50_34_rbr__batoid__consdb_v1` | range-bounded recovery, physically realizable on 99.5% of visits |
| intrinsic check | `v50_34__batoid__consdb_v1` and `v50_34__miw__consdb_v1` | the two intrinsic routes at fixed solver |

The primary result uses the range-bounded recovery because the unconstrained 50/34 solution
asks a median 33x the available actuator stroke on every visit, so its high-mode amplitudes
are partly fitting an unreachable state (`olr/docs/scheme_comparison.md`).

The intrinsic check holds the solver fixed and varies only the intrinsic — there is no
range-bounded arm on the measured intrinsic wavefront (MIW) route. **The expectation is that
the two routes agree on which modes are thermal**: the MIW differs from the batoid prediction
by a static offset per rotator angle, and a static offset moves a fitted intercept, not a
thermal slope. A mode where they disagree is either dominated by the rotator-angle-dependent
part of the MIW−batoid difference — which correlates with elevation through the observing
pattern, and so with temperature — or is marginal on both routes.

## Features and selection

The five thermal channels of the thermal-focus deliverable, `DELIVERABLE_GROUPS`: the TMA
truss temperature and the four M1M3 bulk thermal gradients. The whole selection funnel is
reused from `run_thermal_focus.load_science`, including the LUT-epoch night exclusion and the
20 °C truss-temperature cut, so the sample is the one the published v-mode-1 result is fitted
on. `load_science` gained a `keep_extra` argument to carry the 34 `v*_olr` columns through;
its default behaviour is unchanged.

## Code

| function | what it does |
|---|---|
| `run_thermal_vmodes.py` | the run: the per-mode table, the noise floor, both intrinsic routes |
| `attach_mode_response` | sets `y` to one mode's optical state from the stored open-loop column |
| `null_nmad` | out-of-fold residual nMAD of a median-intercept null, nights held out |
| `fit_mode` | night-grouped Huber fit and null for one mode |
| `bh_threshold` | Benjamini-Hochberg cut over the 34 per-mode statistics |
| `mode_table` | the per-mode table, sorted by skill, with the cut and the noise flag applied |
| `noise_floor_table` | per-mode within-night against between-night scatter |
| `intrinsic_comparison` | per-mode skill on the two intrinsic routes, side by side |

`code/test_thermal_vmodes.py` has 8 tests. Two matter: the response sign is pinned against
`thermal_focus_lib.MEASURED_SIGN`, and the screening rule is checked on a synthetic frame
where one planted thermal mode must be the only one called — and on a pure-noise frame where
nothing must be.

## Results

**V-mode 1 is the only thermal mode.** Over 72,835 science visits across 175 nights on
`v50_34_rbr__batoid__consdb_v1`, one mode of 34 survives the false-discovery-rate cut at
`q = 0.05` (dimensionless), and it is the defocus mode the thermal-focus deliverable already
models. The gap below it is wide, not marginal: v1 scores a skill of +0.808 and the next mode,
v10, scores +0.243 (both dimensionless, fractional reduction in out-of-fold residual nMAD).

| mode | skill | nMAD null | nMAD fit | between/within | well constrained | thermal |
|---|---|---|---|---|---|---|
| v1 | +0.808 | 0.3092 | 0.0593 | 37.07 | yes | **yes** |
| v10 | +0.243 | 0.6818 | 0.5162 | 4.01 | yes | no |
| v18 | +0.222 | 0.1870 | 0.1455 | 1.78 | no | no |
| v15 | +0.219 | 0.6605 | 0.5161 | 9.72 | no | no |
| v19 | +0.208 | 0.6168 | 0.4888 | 3.75 | no | no |
| v13 | +0.169 | 0.5766 | 0.4794 | 8.68 | no | no |
| v22 | +0.160 | 0.1648 | 0.1384 | 2.70 | no | no |
| v3 | +0.152 | 0.0741 | 0.0628 | 5.74 | yes | no |

Skill and the between/within ratio are dimensionless; the nMAD columns are dimensionless v-mode
amplitude. The v1 truss coefficient is +0.1132 (dimensionless v-mode-1 amplitude per °C of mean
TMA truss temperature), the same channel and sign the deliverable reports.

The modes scoring between +0.15 and +0.25 are **not** a weak thermal signal to be chased. Three
things place them: none survives the multiple-comparison cut; the empirical null's own scale is
set by that cluster, so they define the noise rather than stand out from it; and the residual
nMAD they leave is an order of magnitude larger than v1's in absolute terms (0.49 to 0.52
against 0.059 dimensionless v-mode amplitude), so even taken at face value they would predict
little of what is there.

**The noise floor is not monotonic in mode index**, which is why the ratio is measured rather
than assumed. 19 of 34 modes carry between-night structure at more than twice their own
within-night scatter; the 15 that do not are v6, v11, v12, v17, v18 and v23 through v32. So v34
(ratio 5.15) is better determined night to night than v11 (ratio 0.96) or v12 (ratio 1.02).
`WELL_CONSTRAINED_MAX = 12` remains a useful flag for the recovery's conditioning, but it is not
the same statement as this ratio and the table reports both.

**The two intrinsic routes agree, as predicted.** On the unconstrained 50/34 pair, 32 of 34
modes get the same thermal flag and the median absolute skill difference is 0.0016
(dimensionless). Both disagreements are threshold artefacts rather than physics: v3 scores
+0.172 on both routes to three decimals and falls on opposite sides of the cut, and v18 differs
by 0.030 (dimensionless) with neither route calling it thermal in the primary result. Neither
route finds a thermal mode the other misses.

The practical consequence: the published v-mode-1 correction is the whole thermal feed-forward
available from these five channels. There is no second mode to add to it.

Reproduce with:

```bash
python code/run_thermal_vmodes.py
```

Products in `output/thermal_vmodes/`: `mode_table_<variant>.parquet`, `noise_floor.parquet`,
`intrinsic_comparison.parquet`.

## Reference

- [`thermal_focus.md`](thermal_focus.md) — the v-mode-1 deliverable: the five features, the
  Huber pipeline, the night-grouped evaluation and the physical conversion.
- `olr/docs/scheme_comparison.md` — the three solvers on 96,278 paired visits.
- `value_added/docs/schema.md` — `optical_state` columns and the two sign conventions.
- `aos/docs/studies/cwfs_lut.md` — the pointing-dependence study specified alongside this one.
- `notes/status/vmode_thermal_and_lut_handoff.md` — working state for both studies.
