# Study: `thermal_focus` — the focus error as a function of temperature

> **Status:** current · **Last updated:** 2026-09-23 · **Kind:** reference (study)

> **Code:** `code/` · **Notebooks:** `notebooks/`
> **Output:** `output/` (`thermal_focus.pdf`, `thermal_focus.parquet`,
> `<fam_dir>/thermal_focus_fam.parquet`)

Prediction of the Rubin telescope's uniform-defocus error from thermal telemetry alone, so that
focus can be set open-loop from a table instead of being driven by the wavefront sensors. The
measurement is made on ordinary **science** exposures, whose four Corner Wavefront Sensors (CWFS)
the Consolidated Database (ConsDB) records for every visit, so the sample is the whole survey
rather than the few hundred dedicated Full Array Mode (FAM) visits.

**Result.** Five thermal channels — the Telescope Mount Assembly (TMA) truss temperature and the
four M1M3 bulk thermal gradients — fitted jointly with one band-independent Huber robust linear
model predict the focus error to **59.9 µm of equivalent hexapod dz** from an uncorrected
**336.8 µm**, which is 18% of the original scatter (dimensionless, residual normalized median
absolute deviation (nMAD) over uncorrected nMAD). The truss temperature carries most of it at
**+125.09 µm of equivalent hexapod dz per °C**. After that correction **no elevation dependence
remains**, so temperature alone sets the table. The correction works night to night and **not
within an observing block**, where the commanded Trim the model is really predicting is frozen.

## The response

For every visit the response is the focus error the closed loop had accumulated but not yet
corrected:

```
response [um of equivalent hexapod dz] = (v1_trim + MEASURED_SIGN * v1) / v1_per_um_dz
```

with `MEASURED_SIGN = -1` (dimensionless), so the response is `Trim − measured`. Here `v1` is the
amplitude of the first singular vector of the Active Optics System (AOS) sensitivity matrix —
v-mode 1, essentially uniform defocus.

The elevation- and temperature-dependent hexapod look-up-table (LUT) baseline is **deliberately
excluded**. A physical hexapod position is LUT + Trim, but the LUT term is a known commanded
function of elevation: it carries essentially the whole elevation dependence and about 37 times the
measured term's scatter, so including it would put a large elevation dependence into the response
that has nothing to do with the thermal question. Both commanded vectors come from the Engineering
Facility Database (EFD) through the value-added database, because the ConsDB copy of the Trim
reaches only about a third of science exposures.

### Units: equivalent hexapod dz

The response is expressed throughout as **equivalent hexapod dz [µm]**: the total defocus travel,
shared as 0.5 µm on the camera hexapod and 0.5 µm on the M2 hexapod. The conversion is
`v1_per_um_dz = 9.00851e-04` dimensionless v-mode-1 amplitude per µm of total dz travel, the mean
magnitude of camera degree of freedom (DOF) 5 and M2 DOF 0.

Both axes are genuinely **negative** — v-mode 1 per µm is −8.9144254e-04 for the camera hexapod
(DOF 5) and −9.1026032e-04 for M2 (DOF 0) — and `v1_per_um_dz_value` returns the magnitude
`0.5*(|c5| + |c0|)` **by design**, with the sign carried separately by `MEASURED_SIGN`. That is
not a lost sign. The two axes share a sign and agree to 2.1% (dimensionless), so their sum over
their mean is 2.00000 (dimensionless) — that factor of two is what makes the unit *total* travel
rather than one hexapod's.

The **shared** and **camera-alone** conventions are two ways of expressing one optical state, not
two estimates of one number. Per unit v-mode-1 amplitude they are 1110.1 µm and 1121.8 µm of
hexapod dz at the 50-DOF/34-mode projection, and 1108.9 µm and 1120.6 µm at 10/1. The study reports
the shared convention everywhere.

The sign of the measured term is a convention fixed by observation, not by a fit: on the
`BLOCK-T539` `infocus_initial_alignment` sequence the AOS answered a +3.87 µm of wavefront focus
error with −119.7 µm of camera hexapod dz, so the commanded motion opposes the measured Z4 and a
surviving measured residual enters the sum with the sign that cancels it.

Two unit traps apply when commanded vectors are added: `lut_dof3/4/8/9` are hexapod tilts in
**deg** as `MTHexapod` reports them, while the Trim `dof3/4/8/9` are in **arcsec** following the
Optical Feedback Control (OFC) convention; and the LUT covers only the 10 hexapod DOF, so the
mirror bending entries are zero rather than absent.

### The measured term: recovered optical state, not four-corner mean Z4

Four field points cannot separate a field-constant defocus from real field tilt, so a four-corner
mean Z4 is an estimator of convenience. The fitted model instead uses the per-corner Zernike
**deviations** — the ConsDB total optical path difference (OPD) minus an intrinsic wavefront — run
through the OFC sensitivity-matrix singular value decomposition (SVD), recovering DOF and v-mode
amplitudes properly.

That recovery is a value-added quantity held in `value_added/output/aos_efd.duckdb` as one
`optical_state` variant per combination of scheme (`22_12`, what the AOS runs online, or `50_34`),
intrinsic route (`batoid`, the ts_ofc design intrinsic, or `miw`, the Measured Intrinsic Wavefront
evaluated at the corner field points) and OPD version. The deliverable model is fitted on
`v50_34__batoid__consdb_v1`, which is the only variant with rows; the other two are registered and
unpopulated.

The notebook `notebooks/corner_z4_vs_temperature_science.ipynb` uses the
four-corner mean Z4 instead, and is kept as the independent route to the same physical question.

## Sample

68,079 visits over 147 nights, `day_obs` 20251103 to 20260713, bands u g r i z y. The selection
funnel, printed by the build stage rather than summarised, because every stage of it has cost a
real misunderstanding at some point:

| step | count |
|---|---|
| visits with a recovered `v50_34__batoid__consdb_v1` optical state | 85,386 |
| joined to `visit_telemetry` gradients | 85,280 (99.9%) |
| `img_type = 'science'` | 70,769 |
| in bands u g r i z y | 70,769 |
| excluding 7 LUT-epoch nights | −1,204 visits |
| excluding truss temperature above +20 °C | −217 visits on 2 nights |
| with a finite response | 69,348 |
| **with all five thermal features** | **68,079** |

The LUT-epoch nights (`LUT_EPOCH_OFFSET_NIGHTS`) ran a different hexapod LUT configuration; their
per-night offsets sit far from the rest because the commanded baseline itself changed.

The truss cut (`TRUSS_TEMP_MAX_C = 20.0` °C) removes an isolated warm population: 217 visits on
`day_obs` 20251118 and 20251119, spanning +22.88 to +25.07 °C. They are detached from the rest of
the sample by an empty interval of **5.1792 °C** — the largest gap anywhere above +14 °C runs from
+17.7016 to +22.8808 °C — so the 20 °C threshold sits mid-gap and the cut has no boundary
sensitivity. Removing them shifts the truss coefficient by 0.48% (dimensionless) and the
night-grouped residual nMAD from 60.1 to 59.9 µm of equivalent hexapod dz, so no conclusion turns
on it; the cross-validated R² improves from 0.474 to 0.518 (dimensionless) because the warm
outliers were inflating the variance being explained rather than being predicted.

The uncorrected response has median +172.2 µm and nMAD 341.2 µm of equivalent hexapod dz over the
69,348 visits with a finite response, and nMAD 336.8 µm over the 68,079 with all five features.
The truss temperature is filled by within-night interpolation for 6,254 of 69,348 visits (9.0%);
`truss_temp_mean_c_interpolated` is carried so the analysis can cut on it.

Feature ranges over the fitted sample, which bound where the model may be used:

| feature | mean | minimum | maximum | unit |
|---|---|---|---|---|
| `truss_temp_mean_c` | +11.278 | +3.877 | +17.702 | °C |
| `m1m3_z_gradient_c_per_m` | −0.0647 | −0.7692 | +0.6800 | °C per m |
| `m1m3_y_gradient_c_per_m` | −0.0196 | −0.1458 | +0.0332 | °C per m |
| `m1m3_radial_gradient_c_per_m` | −0.0167 | −0.2344 | +0.1374 | °C per m |
| `m1m3_x_gradient_c_per_m` | +0.0017 | −0.0179 | +0.0464 | °C per m |

The x gradient spans only 0.064 °C per m in total, so its large coefficient acts over a narrow
lever arm.

## The fitted model

One Huber robust linear model, band independent, on five thermal features. The pipeline is a
median imputer, a standardizing scaler, then `HuberRegressor`; the coefficients below are the
physical ones, recovered from the standardized fit and verified to reproduce the pipeline's own
prediction to 1.0e-12 µm of equivalent hexapod dz over all 68,079 visits.

```
response [um of equivalent hexapod dz, 0.5 um on each hexapod]

  = -1392.01
    +  125.09 * truss_temp_mean_c              [per deg C]
    -  811.32 * m1m3_z_gradient_c_per_m        [per deg C per m]
    - 1254.02 * m1m3_y_gradient_c_per_m        [per deg C per m]
    -  949.08 * m1m3_radial_gradient_c_per_m   [per deg C per m]
    - 3374.73 * m1m3_x_gradient_c_per_m        [per deg C per m]
```

The intercept is the response at zero in every feature. That is a long extrapolation from the
sample means above, so it is not a physically meaningful offset on its own, only the constant that
makes the five slopes land on the data.

Night-grouped residual nMAD is **59.9 µm of equivalent hexapod dz**, and the per-fold coefficients
are all sign-stable. The truss term is the stable one; the three weaker gradients scatter more
across folds, the radial term's scatter being comparable to its own magnitude.

### Night-grouped evaluation is required

Within a night the thermal telemetry drifts slowly, so consecutive visits are near-duplicates in
feature space: only **2.7%** of the truss temperature's variance is within-night, while **90.7%**
of the response variance is between nights. A visit-level train/test split therefore lets a model
identify the night from its temperature and recall that night's offset, and every score here comes
from `GroupKFold` grouped on `day_obs`. The size of the trap depends on model capacity — a factor
of 1.25 (dimensionless, visit-level nMAD over night-grouped nMAD) for boosted trees, and 1.03 for
the five-coefficient linear fit actually used. The visit-level number is reported for comparison;
it is not a performance estimate.

The same reasoning applies within the sample itself: the per-night offset nMAD of 50.1 µm against
a within-night residual nMAD of 32.7 µm is a ratio of 1.53 (dimensionless, offset nMAD over
residual nMAD), so night-to-night offset variation is the larger of the two and is what a held-out
night must be predicted through.

### Model choice

Night-grouped 5-fold, truss plus the four M1M3 gradients, against the uncorrected baseline of
336.8 µm of equivalent hexapod dz:

| model | residual nMAD [µm equiv hexapod dz] | cross-validated R² [dimensionless] |
|---|---|---|
| **Huber linear** | **59.9** | 0.518 |
| Ridge linear | 66.4 | 0.484 |
| RandomForest | 92.4 | 0.152 |
| HistGB | 99.4 | 0.254 |
| uncorrected | 336.8 | 0.000 |

The response is close to linear and the trees are worse, fitting night-specific structure that
does not transfer to held-out nights. The linear fit is kept because it is interpretable as a
coefficient per °C and is what a look-up table needs.

### What adds nothing

No feature set beats truss + the four gradients by more than 0.2% (dimensionless, nMAD gain over
the deliverable nMAD). Elevation, wind, camera-body temperature and hexapod motion history each
add nothing measurable. This is the result that justifies a five-channel linear model.

### The FAM truss cross-check compares commanded slopes

`FAM_TRUSS_SLOPE = 0.09634` (dimensionless v-mode-1 amplitude per °C) from the FAM Double Zernike
(DZ) fits is a slope of the **commanded Trim**, not of the response. The like-for-like science-image
test is therefore `v1_trim` against truss temperature, and it gives a pooled
**+0.09850 ± 0.00021 per °C**, consistent with the FAM value to within the study's 0.03 per °C
tolerance. Per-band slopes spread more widely
around it; a single band is not an independent measurement of this slope.

Comparing the fitted **response** coefficient against `FAM_TRUSS_SLOPE` instead is not a valid
cross-check — it compares two different quantities and produces an apparent 16.5% disagreement
that means nothing.

### Camera-body temperature

The camera-body `AverageTemp` from `lsst.MTCamera.utiltrunk_body` resolves 98.28% of visits against
the truss temperature's 100.00%, and is indistinguishable from the truss as a regressor. The two
thermometers correlate at **Pearson r +0.9943, Spearman rho +0.9928** over 66,908 visits, with a
Huber slope of **+0.8850 °C camera-body per °C truss**. Against the response the truss gives
**+109.16 ± 0.25 µm of equivalent hexapod dz per °C** (Pearson r +0.6328, Spearman rho +0.8190,
n = 68,079) and the camera body **+126.78 ± 0.27 µm per °C** (Pearson r +0.6271, Spearman rho
+0.8283, n = 66,908). The truss stays primary on physical grounds — it is the
load path setting the M1M3-to-camera spacing — and the camera-body channel is excluded because its
collinearity with the truss destabilises the truss coefficient.

### Residual shape

The residual has a heavy, one-sided positive tail, which is why every fit here is robust rather
than least squares. Two residuals answer two different questions, and the study reports both:
about a **truss-only per-band** fit the positive excess is large, saying the truss relation alone
leaves a population of visits above it; about the **full five-feature** fit that asymmetry is
largely absorbed, saying the gradients account for much of it.

An earlier one-sided-tail result quoted in the `correlations` study (beyond +3 nMAD 2.38/2.01/3.04/
0.88% against 0.25/0.38/0.34/0.04% below −3 nMAD) is **a different statistic** — measured on
`v1_total`, which includes the LUT, in dimensionless v-mode units over four bands, about a
truss-only per-band fit. It is not restated here as reproduced.

### Band independence and filter changes

The model is fitted once for all bands. Fitting per band would let the coefficients change at every
filter change, injecting a step into the corrected residual where nothing physical has happened.
The three-way comparison of the median absolute step between consecutive visits within a night
tests exactly that claim:

| correction | band change [µm equiv hexapod dz] | same band | ratio [dimensionless] |
|---|---|---|---|
| uncorrected | 18.3 (n = 1,132) | 8.7 (n = 66,800) | 2.12 |
| **per-band models** | **25.7** | 8.8 | 2.94 |
| shared thermal model | 18.8 | 8.7 | 2.15 |

Per-band fitting makes the band-change step **worse**, 25.7 against 18.3 µm uncorrected, while the
shared model leaves it essentially unchanged at 18.8 µm. The shared model is the right choice. A
band-independent correction cannot remove a real per-band focus offset, and a small one remains;
it would have to be added separately.

## Elevation: nothing remains

Once the thermal correction is applied, the residual carries no useful elevation dependence. Over
**125 nights** with enough visits to fit, the median per-night residual-against-elevation slope is
**−0.005 µm of equivalent hexapod dz per deg** with nMAD **0.782 µm per deg**, scattering about
zero against a median formal error of 0.162 µm per deg. The per-night offset at 60 deg elevation
has median −1.8 µm and nMAD 50.1 µm.

Splitting each night into rising and falling legs over the 109 nights with both, the median
rising-minus-falling slope difference is **+0.254 µm per deg** with nMAD **0.878 µm per deg**, and
the rising leg is steeper on 64 of 109 nights (sign-test p = 0.084, dimensionless) — **no
consistent direction dependence**, so no hysteresis term is warranted. The slew direction is taken
from a **centred 21-visit rolling median of elevation with a deadband**, not from the sign of the
per-visit elevation difference, which is dominated by pointing jitter.

The reason nothing remains is that the hexapod LUT already handles elevation: fitted on its own
against elevation the LUT term has a slope two orders of magnitude larger than what survives in
the residual. No elevation stage is therefore subtracted, and the diagnostic showing that nothing
is left is the result.

## FAM blocks: the correction does not work within a block

A FAM block is a run of triplets taken at one fixed pointing over tens of minutes. The in-focus
`acq` visit of each triplet carries a CWFS optical state, so the same response can be read per
triplet and followed across the block. This is a timescale the model was never fitted on.

**A FAM block is derived, not stored.** `assign_blocks` is a greedy fixed-pointing walk over
`(science_program, day_obs)` in `acq_seq_num` order: a new block starts when the `seq_num` span
reaches 36, or when altitude, azimuth or camera rotator angle drifts beyond 2.0 deg (azimuth
compared circularly). `select_sets` then requires exactly 12 visits and a constant `seq_num` step
of 3. Grouping by night instead would substitute a whole night for a 12-triplet block and silently
change what every within-set number means, so `assign_blocks` **raises** rather than falling back
to a coarser grouping when a pointing column is absent.

Of 326 blocks, 45 hold exactly 12 triplets with a constant step of 3 — 540 visits over 18 nights.

| quantity | median within-set peak-to-peak | unit |
|---|---|---|
| uncorrected response | **34.9** | µm equiv hexapod dz |
| thermally corrected | **47.0** | µm equiv hexapod dz |
| the prediction's own swing | 20.8 | µm equiv hexapod dz |
| truss temperature | 0.0658 | °C |

**The thermal correction makes within-block scatter worse**, a ratio of **1.35 (dimensionless,
corrected over uncorrected)**, improving only **5 of 45** sets.

### Why: inside a block the commanded term is frozen

The response is `(v1_trim + MEASURED_SIGN * v1) / v1_per_um_dz` with `MEASURED_SIGN = -1.0`
(dimensionless), so it carries a commanded term and a measured term of opposite sign. Between
nights the **commanded** term dominates: `v1_trim` carries a between-night variance fraction of
**91.2%** (dimensionless, between-night over total) against **25.0%** for the measured `v1`. The
fitted model is therefore, to a good approximation, a model of what the AOS commanded.

**Inside a FAM block the Trim is exactly constant.** Its within-set peak-to-peak is identically
zero in **44 of the 45** clean sets; the one exception steps by 7.6e-03 (dimensionless v-mode-1
amplitude). The AOS does not re-command Trim while a ladder runs. So within a block the response
reduces to `-v1 / v1_per_um_dz` — the measured term alone, entering with the opposite sign from the
one the model was fitted on.

That is what reverses the slope against truss temperature:

| slope of response against truss temperature | value [µm equiv hexapod dz per °C] |
|---|---|
| science, between nights | +124.38 |
| science, within a night (n = 68,079) | +78.69 ± 0.62 |
| FAM, within a 12-triplet set (n = 540) | **−81.10 ± 17.58** |

The reversal is neither a sign error nor telemetry noise. The truss temperature is genuinely
resolved inside a block — 12 distinct values per 12-visit set, monotonic in 26 of the 45 sets, with
the within-night interpolation flag raised on only 50 of 540 rows (9.3%) — and a within-set
permutation test puts the observed slope about 4 null-sigma out: the shuffled null is
+0.01 ± 20.61 µm equiv hexapod dz per °C, with 0 of 200 draws reaching the observed magnitude. The
within-set slope of response against prediction is **−0.7207 ± 0.0458** (dimensionless, Huber),
where a correct correction would give ≈ +1.

**Do not flip the sign to repair this.** Subtracting the prediction gives 46.7 µm of within-set
peak-to-peak and improves 6 of 45 sets; adding it gives 32.7 µm and improves 27 of 45, against
34.9 µm uncorrected. The improvement is real but meaningless: it fits the measured term with a
model of the commanded term, and the agreement would not survive a block in which the AOS did
re-command Trim.

The model describes night-to-night thermal drift, which is what it was built for and what the
open-loop feed-forward term needs. It cannot describe a block over which the quantity it actually
models does not move.

The result is not an artifact of the 12-triplet floor. Relaxing it:

| floor | sets | nights | median within-set peak-to-peak [µm equiv hexapod dz] |
|---|---|---|---|
| at least 12 triplets | 45 | 18 | 34.9 |
| at least 8 | 64 | 22 | 34.4 |
| at least 6 | 70 | 22 | 33.5 |

### DZ(k=1, j=4) cross-check

The defocused pair of each triplet carries its own measurement of the same physical quantity: the
DZ fit's term at focal (field) Zernike order k=1 and pupil Noll index j=4, read from the `fam_dz`
table as `dz_k1_j4` [µm of wavefront]. Its within-set peak-to-peak is median **0.3286 µm of
wavefront** over 45 sets, which is **21.0 µm of equivalent hexapod dz** through

```
DZ_UM_PER_UM_WF = -63.9902 um of equivalent hexapod dz per um of wavefront
```

against the response's **34.9 µm**. The in-focus corner-sensor state swings about 1.7 times as much
as the FAM pair's own defocus over the same set.

Two readings of that difference are open and this measurement does not separate them: the FAM pair
is a defocused exposure pair whose fit spans the whole focal plane while the `acq` v-mode 1 comes
from the four corner sensors at best focus, so they differ in both what they average over and how
they are retrieved; and DZ(k=1, j=4) is one term of the FAM wavefront rather than the whole of it.

**Still unverified:** `fam_dz.v_modes` is built by a different engine than
`optical_state.v_modes`, and the two carry **opposite v-mode-1 sign conventions**. The DZ
coefficient columns used here are unaffected, but any future comparison of the two `v_modes`
columns must establish the relative sign first.

## The v-mode-1 conversion across projection schemes

The conversion is re-derived from the sensitivity matrix at each projection rather than assumed to
carry over, because the online system runs a different projection from the one the fit uses:

| scheme | v1 per µm total dz [dimensionless per µm] | shared [µm dz per unit v1] | camera-alone |
|---|---|---|---|
| 50 DOF / 34 modes | 9.00851e-04 | 1110.1 | 1121.8 |
| 22 DOF / 12 modes | 9.00942e-04 | 1109.9 | 1121.7 |
| 10 DOF / 1 mode | 9.01828e-04 | 1108.9 | 1120.6 |

The three agree to **0.108%** (dimensionless, spread over the 50/34 value), far below the fit's own
uncertainty, so the choice of projection does not affect any conclusion. The shared-against-
camera-alone difference of 1.1% is a definition choice, not a scheme uncertainty; conflating the
two comparisons is easy and wrong.

## Code

| file | role |
|---|---|
| `code/thermal_focus_lib.py` | the response definition, the conversions and the feature groups — one definition, so nothing can drift |
| `code/run_thermal_focus.py` | build: the value-added database plus live ConsDB, writing the cached tables |
| `code/thermal_focus_fit.py` | the fitting core: the models, night-grouped evaluation, the block assignment and the diagnostics |
| `code/run_thermal_focus_analysis.py` | the analysis: eleven sections and one document, no network |
| `code/trim_calculator.py` | the standalone online calculator: numpy only, no repository imports |

### The network seam

Only one quantity forces a network call. `truss_temp_mean_c` is **not a stored column**: it is
computed inside `value_added/code/efd_db.py` (`join_consdb`) as the mean of the
`tma_truss_temp_pxpy` and `tma_truss_temp_mxmy` thermometers and then interpolated within the
night, so there is no offline route to the study's headline regressor. That is why the build and
analysis stages are separate: the build pays the network cost once and caches to parquet, and the
analysis needs no network at all, which is what makes iterating on a fit cheap.

The DuckDB file lock is process-wide and excludes readers as well as writers, so every connection
is read-only; a stray read-write connection blocks every other process, including a running build.

This topic imports `aos_state` from `aos/code` for the v-modes and the DOF sets, through
`sys.path.insert`, as `blocks/`, `olr/`, `optatmo/`, `smatrix/` and `value_added/` do.

### The standalone calculator

`trim_calculator.py` imports numpy and argparse and nothing else, so it can be copied to a summit
machine and run there. Every coefficient is inlined with its units and provenance, and it carries
worked test cases. Inlining can drift from the fit silently, so section 11 of the analysis checks
the calculator against the pipeline it fitted: **max |difference| 0.0034 µm of equivalent hexapod
dz** over 68,079 visits, which is the two-decimal rounding of the inlined coefficients.

The calculator is a **night-to-night feed-forward term**. It does not read the wavefront and does
not know what the AOS has already commanded, so applying it blind on top of an already-converged
loop would double-count the correction; and it must not be used to chase focus within a block, for
the reason the FAM section gives.

Keep two conversions distinct: the dz-equivalent conversion is a one-hexapod-motion equivalent,
fine for reading v1 physically, and is **not** the trim-adjustment code, which must resemble the
online scheme.

### Output

| product | content |
|---|---|
| `thermal_focus.parquet` | one row per science visit: identity, band, pointing, the v-mode-1 components, the response [µm equiv hexapod dz] and the thermal telemetry |
| `<fam_dir>/thermal_focus_fam.parquet` | one row per FAM triplet whose `acq` visit has a recovered optical state, with the triplet's own DZ coefficients |
| `thermal_focus.pdf` | the analysis document: eleven sections, from the sample funnel to the calculator check |

`output/` has no data-axis level: the products depend on the database and the optical
prescription, not on a Butler collection or processing variant. The FAM table is the exception,
depending on which reduction produced the DZ coefficients, so it sits under the FAM variant's short
directory name.

## Notebook

`notebooks/corner_z4_vs_temperature_science.ipynb` asks the same physical question
from the four-corner mean Z4 rather than the recovered optical state, assembling

```
v1_total = v1(hexapod LUT) + v1(Trim) + MEASURED_SIGN * v1_equivalent(four-corner mean Z4)
```

This includes the LUT term, unlike the fitted model, and is a different and noisier estimator of
the measured state; it is kept as an independent route to the same conclusion. It needs ConsDB and
the EFD, so it runs on the Rubin Science Platform or USDF.

## Relation to the other focus studies

| study | measurement | timescale | sample |
|---|---|---|---|
| [`lut`](../../../aos/docs/studies/lut.md) | FAM DZ fits over 189 detectors, collapsed to one static DOF vector | static | dedicated FAM visits |
| [`correlations`](../../../aos/docs/studies/correlations.md) | DZ and v-mode correlations against telemetry on the MI-refit residual | per visit | FAM visits |
| `thermal_focus` | the temperature-dependent focus surface from the CWFS optical state | night to night, and within a block | all science visits, and FAM `acq` visits |

The `lut` study produces one static vector; this study produces a dependence on temperature. They
are separate outputs and neither consumes the other.

## Outstanding work

One follow-up is known and not attempted here:

- Collect all M1M3 cell temperature values to form separate M1 and M3 focus variables, looking for
  an r²-like radial thermal mode the four bulk gradients cannot express.

The comparison of the 50 DOF / 34 mode, 22/12 and 10/1 projections is **done** — the three
`v1_per_um_dz` values agree to 0.108% (dimensionless, spread over the 50/34 value), far below the
fit's own uncertainty, so no refit under another projection is needed. The conversion section above
carries the table.
