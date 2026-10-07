# Study: `cwfs_lut` — pointing dependence of the open-loop state, from science visits

> **Status:** current · **Last updated:** 2026-10-07 · **Kind:** reference (study)

> **Code:** `code/cwfs_lut/` · **Notebooks:** `notebooks/cwfs_lut/`
> **Output:** `output/cwfs_lut/`

Dependence of the open-loop optical state on Telescope Mount Assembly (TMA) elevation and
camera rotator angle, measured from the corner wavefront sensors (CWFS) over the whole
science survey, for look-up-table (LUT) development. Compared against the
[`bounce`](bounce.md) study's elevation and rotator bounce tests.

Distinct from [`lut`](lut.md), which averages the Full Array Mode (FAM) Double Zernike fits
over all pointings and so carries no pointing dependence. This study is built on single-visit
corner-sensor recoveries and the pointing dependence is the result.

## The quantity

The stored **open-loop** degree-of-freedom (DOF) vector, `Deviation − Trim` — the state that
would have been present with the loop open, which is what a LUT must supply. Read from
`dof_olr` in the value-added database's `optical_state` table, expanded by
`efd_db.optical_state(..., wide=True)` into `dof0_olr` to `dof49_olr`.

The sign is the opposite of the optical state: `optical state = Trim − Deviation = −dof_olr`.
See `value_added/docs/schema.md` for both conventions.

## Sample

| | |
|---|---|
| nights | 214 with a recovered state, `day_obs` 20250728–20260713 |
| visits | 96,278 paired across intrinsic routes |
| elevation range | 17.11 to 83.19 deg |
| rotator angle range | −79.87 to +79.51 deg |

Pointing comes from `optical_state.elevation_deg` (ConsDB `exposure.altitude`) and
`optical_state.rotator_angle_deg` (ConsDB `visit1_quicklook.physical_rotator_angle`). Every
visit with a recovered state carries both.

## Two choices, and what each costs

**Absolute trends, not within-night paired differences.** Science visits sweep both angles
widely inside a single night — `day_obs` 20260713 covers elevation 23.68 to 82.43 deg and
rotator −79.43 to +78.52 deg — so pairing would discard most of the available leverage, far
more than the bounce test's paired ±3 deg legs provide. The cost: an absolute elevation trend
**confounds gravity-driven flexure with thermal drift** that tracks elevation through the
observing pattern. That is a caveat on every elevation number here, not something the fit
removes. The [`thermal_focus`](../../../thermal_focus/docs/thermal_focus.md) topic models the
thermal part of v-mode 1 directly and is the route to separating them.

**Both intrinsic routes, neither as the reference.** The Deviation is `OPD − intrinsic`, so
the assumed intrinsic shifts every recovered DOF. The measured intrinsic wavefront (MIW)
differs from the batoid ray-trace prediction by 0.0547 µm of wavefront at the corner field
points even at rotator angle 0.0 deg, rising to 0.0796 µm at +60 deg. On `day_obs` 20260318
the resulting open-loop M2 hexapod dx median moves from +251.04 µm (batoid) to +487.58 µm
(MIW) — more than the term's own value. The spread between routes is part of the result.

| variant | intrinsic | solver |
|---|---|---|
| `v50_34__batoid__consdb_v1` | batoid | truncated SVD |
| `v50_34__miw__consdb_v1` | MIW `danish_1_2_A_50_34_i_5rot` | truncated SVD |
| `v50_34_rbr__batoid__consdb_v1` | batoid | range-bounded recovery (RBR) |

The intrinsic comparison uses the first two, which hold the solver fixed; there is no RBR arm
on the MIW route. The RBR variant is reported alongside because a LUT must command a
physically reachable state, and the unconstrained 50/34 recovery asks a median 33x the
available actuator stroke on every visit (`olr/docs/scheme_comparison.md`).

## DOF units: the four hexapod tilts are deg on both sides

The 50-element DOF vector is stored with **DOF 3, 4, 8, 9 — the M2 and camera hexapod
rx/ry — in deg** and the other 46 entries in µm.
`lsst.ts.intrinsic.wavefront.ofc_svd.DOF_UNITS_50` labels those same four **arcsec**, and the
bounce-test tables copy that label into their own `unit` column — but **neither side ever
scales a value to match it**. The label is wrong; the numbers are not. Both are deg, and
nothing converts in either direction.

Four independent checks say so:

| check | value | reading |
|---|---|---|
| ts_ofc `default_rb_stroke()` | tilts 0.12, 0.24 | 0.12 deg = 432 arcsec is a real hexapod stroke; 0.12 arcsec is not |
| ts_ofc sensitivity matrix, tilt over decentre column | 5.6 m lever arm read as deg | read as arcsec it implies 20 km, absurd by three orders of magnitude |
| controller `max_integral` | 0.01 on the tilts against 500/100 µm | 0.01 deg = 36 arcsec is a sane clamp |
| `grep 3600` in the bounce, `ofc_svd` and `regularized_inversion` DOF path | no matches | no conversion exists to have been applied |

This study previously applied 3600 arcsec/deg to those four entries on the strength of the
label. That was wrong and is removed; `HEX_TILT_UNIT` now records the convention in one
greppable place, and two tests fail if the factor reappears.

The ts_ofc layout is DOF 0–4 M2 hexapod, DOF 5–9 camera hexapod, each as (dz, dx, dy, rx,
ry) — so DOF 1–2 are decentres in µm and DOF 3–4 are tilts in deg, not the reverse.

## Scope of the bounce-test comparison

Restricted to the **hexapod pistons and decentres**, `cwfs_lut_lib.BOUNCE_COMPARABLE_DOF`.
The bounce test retrieves a full-focal-plane Double Zernike field from FAM data; this study
has four corner field points, which cannot constrain mirror figure the same way.

That limit is now **measured rather than argued**: over the rotator leg the two retrievals
agree at cosine similarity +0.829 (dimensionless) on the six hexapod translations and +0.093
on the 40 bending modes, even though 25 of those 40 are individually significant above 3 sigma
on the bounce side. The bending modes are not noise — they are retrieved, and the two
retrievals disagree about them.

## The arm matters more than the solver

The bounce Δ is a paired difference of the recovered **deviation** and never subtracts the
Trim. The stored `dof*_olr` is the **open-loop** state, `Deviation − Trim`. Comparing one
against the other mixes two quantities that differ by the Trim, and it costs real agreement:

| M2 + camera dx, µm per deg of rotator | value | agreement |
|---|---|---|
| bounce Δ | +20.33 | — |
| survey, deviation arm (`dofN`) | +21.15 | **4.0%** |
| survey, open-loop arm (`dofN_olr`) | +17.17 | 15.6% |

The matched deviation arm agrees four times more closely. `bounce_compare` therefore carries
the arm as an explicit axis and defaults to the deviation arm.

## Code

| module | what it does |
|---|---|
| `cwfs_lut_lib.py` | DOF labels and units, Huber trend fits, per-DOF trend tables, intrinsic-route comparison |
| `bounce_compare.py` | the bounce-test comparison: the bounce slope per leg, the survey slope on either arm, and the three comparison spaces |
| `run_cwfs_lut.py` | the run: both angles on three variants, the intrinsic comparison, the bounce comparison, the figures |
| `cwfs_lut_figures.py` | the nine figure pages, one function each |
| `test_cwfs_lut_lib.py` | 17 tests; two of them fail if the spurious 3600 arcsec/deg returns |
| `test_bounce_compare.py` | 21 tests; the ones that matter pin the wrong `unit` label being ignored, the 0-based against 1-based v-mode indexing, and the sign of a negative throw |

Fits are Huber robust linear (`statsmodels.RLM` with `HuberT`), the repository default for
AOS correlations, reporting both Pearson r and Spearman rho. `huber_trend` refuses a fit over
under 5 deg of angle span or under 200 visits — a slope from a narrow span extrapolates badly
and is not reportable. Its `slope_err` is the formal RLM standard error and **understates the
true uncertainty**: successive visits are correlated, so the effective sample is smaller than
`n`.

## Results

96,278 visits across 214 nights on the range-bounded recovery, elevation spanning 17.11 to
83.19 deg and rotator angle −79.87 to +79.51 deg.

**One term dominates: M2 hexapod dx against rotator angle, +20.39 µm/deg of rotator**, with
Pearson r = +0.718 and Spearman rho = +0.759 (both dimensionless, n = 96,278 visits). That is
the only rigid-body trend in either angle with a correlation above 0.6, and it is a clean
candidate for a LUT term. Second is M2 hexapod ry against rotator angle at
−1.701e-4 deg/deg (−0.612 arcsec/deg), r = −0.525, rho = −0.656.

| DOF | vs elevation [unit/deg] | Pearson r | vs rotator [unit/deg] | Pearson r | unit |
|---|---|---|---|---|---|
| M2 hexapod dz | +1.570 | +0.011 | −0.684 | −0.030 | µm |
| M2 hexapod dx | +5.937 | +0.083 | **+20.39** | **+0.718** | µm |
| M2 hexapod dy | −7.586 | −0.068 | −4.860 | −0.111 | µm |
| M2 hexapod rx | −9.981e-5 | −0.104 | −4.428e-5 | −0.105 | deg |
| M2 hexapod ry | −4.850e-5 | −0.054 | **−1.701e-4** | **−0.525** | deg |
| camera hexapod dz | −0.879 | +0.008 | +1.039 | +0.066 | µm |
| camera hexapod dx | +3.957 | +0.042 | +2.981 | +0.051 | µm |
| camera hexapod dy | +0.979 | −0.003 | −0.246 | −0.022 | µm |
| camera hexapod rx | +1.371e-5 | +0.033 | +2.681e-5 | +0.125 | deg |
| camera hexapod ry | −2.677e-6 | −0.003 | +4.799e-6 | −0.007 | deg |

**Elevation carries no strong trend.** No elevation slope reaches r = 0.09 (dimensionless), and
the largest in magnitude — camera hexapod dy at −14.59 µm/deg on the two-night smoke test —
drops to +0.979 µm/deg over the full sample. A trend that changes sign when the sample grows
from 2 nights to 214 is a per-night offset being read as a slope, not elevation dependence.
Gravity-driven elevation terms are presumably already removed by the hexapod LUT in force during
the survey, which is what the open-loop residual is measured against.

**The camera hexapod decentres are far noisier than M2's, which is why none of them trends.**
Against rotator angle the robust residual scatter is 1,290 µm for camera hexapod dy and 681 µm
for camera hexapod dx, against 504 µm for the M2 hexapod dx term that does trend and 429 to
483 µm for the other M2 axes. On the figures the four camera-hexapod panels are vertical smears
at every rotator angle. So the absence of a camera-hexapod LUT term here is a statement about
what four corner sensors can constrain, not a claim that the camera hexapod has no pointing
dependence; a term as large as M2 dx's +20.39 µm/deg would still be visible, but a term a few
times smaller would not be.

**The two intrinsic routes agree far better than the single-night check suggested.** On the
unconstrained 50/34 pair the MIW−batoid slope differences are sub-percent on every large term:
M2 hexapod dx against rotator angle differs by −0.024 µm/deg out of +8.79 (0.3%, dimensionless
MIW over batoid), and the largest absolute difference anywhere is camera hexapod dy at
+0.290 µm/deg. The earlier 23% figure came from `day_obs` 20260713 alone and does not survive
the full sample — a single night does not constrain a slope well enough to compare routes.

So the intrinsic route matters for the **absolute** open-loop state, where the lateral decentres
shift by more than their own value, but not for the **trends** a LUT is built from. A static
intrinsic offset moves an intercept and not a slope, and that is what the data shows.

**The solver matters much more than the intrinsic.** The headline M2 dx term against rotator
angle is +20.39 µm/deg under the range-bounded recovery and +8.79 µm/deg unconstrained — a
factor of 2.3 (dimensionless, RBR over unconstrained) — and its Pearson r goes from +0.218 to
+0.718 (dimensionless, n = 96,278 visits both). That is the expected direction, since the
unconstrained solution spends amplitude on states the actuators cannot reach, but the size of it
means a LUT term must state which recovery it was fitted on. The primary result here is RBR.

The correlation gap comes from the **signal**, not from extra noise: the robust residual scatter
is 503.6 µm under RBR against 509.0 µm unconstrained, near-identical. So the range-bounded
recovery recovers a rotator dependence 2.3 times larger over the same per-visit scatter, rather
than recovering the same dependence more cleanly. On the figure the unconstrained panel looks
visibly broader, but that is its longer outlier tail, which the nMAD ignores and the eye does
not — the slope ratio and the correlation ratio are the same statement.

### The bounce-test comparison

Measured on the T724 rotator leg, which is the sharp test: a single +60.017 deg throw with
elevation pinned to 0.003 deg, 31 pairs. Headline pairing is the bounce unconstrained arm
against the survey deviation arm on `v50_34__batoid__consdb_v1`.

**Agreement by subspace**, cosine similarity of the two slope vectors (dimensionless), over the
entries clearing |Δ|/error = 3 on the bounce side:

| subspace | terms significant | batoid | MIW | RBR |
|---|---|---|---|---|
| 6 hexapod translations | 3 of 6 | **+0.829** | +0.840 | +0.524 |
| 4 hexapod tilts | 3 of 4 | +0.815 | +0.827 | +0.638 |
| 40 bending modes | 25 of 40 | **+0.093** | +0.021 | −0.200 |
| 34 v-modes | 19 of 34 | −0.225 | −0.185 | +0.261 |

**The hexapod rigid-body terms agree and the mirror bending modes do not.** The bending
disagreement is not an absence of signal — 25 of those 40 modes are individually significant
above 3 sigma on the bounce side. Four corner field points do not constrain mirror figure the
way a full-focal-plane Double Zernike retrieval does, and this is the measurement of that.

**V-mode space is the dirtiest comparison, not the cleanest.** It looks like the
retrieval-independent space and is not: the basis is bending-dominated, so it inherits the
bending disagreement, buries the translation agreement, and even changes sign with the solver.

**RBR degrades the agreement**, from +0.829 to +0.524 on the translations. Consistent with the
bounce note's own warning that RBR biases rigid-body amplitudes toward zero by construction and
that an RBR amplitude is not a measurement of hexapod motion. The unconstrained recovery is the
estimator for a LUT fit on both sides.

**Per axis the two disagree; summed over the two hexapods they agree.** Slopes in µm per deg of
rotator angle:

| term | bounce | survey | ratio |
|---|---|---|---|
| M2 hexapod dx | +4.48 ± 0.40 | +8.08 | 1.80 |
| camera hexapod dx | +15.85 ± 0.41 | +12.91 | 0.81 |
| **M2 + camera dx** | **+20.33** | **+21.15** | **1.04** |
| M2 + camera dy | +10.21 | +0.38 | 0.04 |
| M2 + camera dz | −0.09 | −0.01 | 0.07 |

The dx sum agrees to **4%** while its two constituent axes differ by factors of 1.80 and 0.81 in
opposite directions — the signature of the rigid-body degeneracy rather than of either method
being wrong. The bounce note says the same from its own side: the split is not uniquely pinned,
while the rigid-body wavefront is preserved. The dy sum does **not** agree, and that is
unexplained; dz carries no signal on either side.

**The elevation legs.** A weighted fit through the origin over the five legs, against the survey
Huber slope over the same elevation range:

| term | bounce linear fit [µm/deg] | chi2/dof (dof 4) | cos-fit chi2/dof |
|---|---|---|---|
| M2 hexapod dy | −26.77 ± 1.04 | 7.39 | 4.29 |
| camera hexapod dy | +19.19 ± 0.91 | 2.19 | 2.78 |
| camera hexapod dx | +16.71 ± 0.76 | 0.61 | 1.13 |
| M2 hexapod dx | −3.86 ± 0.37 | 0.80 | 0.90 |

A chi2/dof above 1 here is **not** grounds to reject the linear form: the bounce test carries
atmospheric turbulence and other stochastic terms the per-leg error bars do not capture, so a
perfect chi2/dof is not expected. The linear term is a first-order approximation to a dependence
physically closer to cos(elevation), and it is much better than no correction.

The survey elevation trends are near-flat (all |Pearson r| ≤ 0.09) while these bounce slopes are
large. Both sides were taken with the hexapod look-up table **active**, so both measure LUT
*residual* and the comparison is like-for-like — which makes this a real disagreement rather
than two different quantities. It is the open question this comparison leaves.

Reproduce with:

```bash
python code/cwfs_lut/run_cwfs_lut.py
```

Products in `aos/output/cwfs_lut/`: `trend_<variant>_<angle>.parquet` for three variants and two
angles, `intrinsic_spread_<angle>.parquet`, `bounce_comparable_<angle>.parquet` — the ten
rigid-body axes with a `comparable` flag marking the six decentres — the five
`bounce_compare_*.parquet` tables, and `cwfs_lut.pdf`.

`--no-bounce` skips the comparison. It is also skipped automatically, with a message, when the
bounce run is not on disk: those products belong to the [`bounce`](bounce.md) study and this
study's own results do not depend on them.

## Figures

`cwfs_lut.pdf`, nine pages when the bounce run is present and five without, written by the same
run. `--no-figures` writes the tables alone.
Per-visit points are drawn as hexbin rather than scatter: near 100,000 visits across ten axes
and two angles would be unreadable as points and would make a very large vector PDF.

| page | what it shows |
|---|---|
| the headline term | the strongest rigid-body trend, picked from the data rather than hardcoded, with its residual against the same angle beside it. Curvature in that residual would mean a straight line is the wrong LUT form |
| solver against solver | the same term under the range-bounded and unconstrained recoveries, shared y axis, both on the batoid intrinsic. The slopes differ by a factor near 2.3 and the correlations by much more, so the range-bounded solution is visibly the tighter function rather than a scaled copy |
| every axis against elevation | all ten rigid-body axes in one 2×5 grid, Huber line and Pearson r on each. This is how the elevation null is *shown* — every panel is flat and a reader can see none was omitted |
| every axis against rotator | the same grid for rotator angle; panels clearing \|r\| = 0.6 are outlined, so the one real term does not hide among the ten |
| slope summary | slope per axis for all three variants with formal error bars, split into a µm panel and a deg panel, with the measured bounce slopes as a fourth marker per axis. The two results in one view: the intrinsic routes overlap while the solvers separate |
| agreement by subspace | cosine similarity and amplitude ratio per subspace, every arm pairing, the headline pairing picked out. The page the quantitative scope rests on: translations agree, bending does not |
| bounce against survey per axis | the six comparable axes side by side, and the same information summed over the two hexapods. The pair is the argument — individual axes disagree by factors of a few while their sum agrees, which is a degeneracy rather than one method being wrong |
| bounce against survey in v-mode space | all 34 modes against exact agreement, significant ones filled, and the per-mode slopes beside it. Included because v-mode space *looks* like the retrieval-independent comparison and is not |
| bounce elevation legs | the five legs with error bars, the weighted linear fit, the cos(elevation) fit and the survey slope. A chi2/dof above 1 here reflects turbulence the per-leg errors do not capture, not a rejected linear model |

Error bars on the survey arms are the **formal** RLM standard errors and understate the true
uncertainty, since successive visits are correlated; the pages say so. They are there to compare
arms against each other, not as confidence intervals. The bounce marker's bars are a per-leg
median standard error divided by the throw and are not the same kind of quantity — the per-axis
comparison page is where the two are put on a common footing.

## Reference

- [`bounce`](bounce.md) — the comparison target. Its physical-DOF table labels the tilts
  **arcsec** and they are **deg**, as are this study's; see the units section above.
- [`lut`](lut.md) — the earlier FAM-averaged DOF table, with no pointing dependence.
- `value_added/docs/schema.md` — `optical_state` columns, the two sign conventions, the DOF
  unit section and the pointing section.
- `olr/docs/scheme_comparison.md` — the three solvers on 96,278 paired visits, and why the
  unconstrained 50/34 state is not realizable.
- `notes/status/vmode_thermal_and_lut_handoff.md` — working state for this study and the
  all-v-mode thermal study it was specified with.
