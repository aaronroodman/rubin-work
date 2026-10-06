# Thermal response of all 34 v-modes

> **Status:** in progress · **Last updated:** 2026-10-06 · **Kind:** reference (study)

> **Code:** `code/thermal_vmodes.py`, `code/test_thermal_vmodes.py`
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

Not yet produced over the full sample. The screening rule is verified on synthetic data
(recovers a single planted mode out of 34, calls nothing on pure noise) but has not been run
against the 214-night sample.

## Reference

- [`thermal_focus.md`](thermal_focus.md) — the v-mode-1 deliverable: the five features, the
  Huber pipeline, the night-grouped evaluation and the physical conversion.
- `olr/docs/scheme_comparison.md` — the three solvers on 96,278 paired visits.
- `value_added/docs/schema.md` — `optical_state` columns and the two sign conventions.
- `aos/docs/studies/cwfs_lut.md` — the pointing-dependence study specified alongside this one.
- `notes/status/vmode_thermal_and_lut_handoff.md` — working state for both studies.
