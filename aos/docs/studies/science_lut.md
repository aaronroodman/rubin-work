# Study: `science_lut` — a focus look-up table from science exposures

> **Status:** current · **Last updated:** 2026-09-17 · **Kind:** reference (study)

> **Code:** `code/science_lut/` · **Notebooks:** `notebooks/science_lut/`
> **Output:** `output/science_lut/` (`science_lut_analysis.pdf`, `science_lut_fits.parquet`)

Construction of a focus Look-Up Table (LUT) for the Active Optics System (AOS) from ordinary
**science** exposures rather than from Full Array Mode (FAM) data. The measured quantity is the
optical state at the four Corner Wavefront Sensors (CWFS), which the Consolidated Database
(ConsDB) records for every science visit, so the sample is the whole survey instead of the few
hundred dedicated FAM visits.

The goal is to predict the telescope's uniform-defocus error from temperature telemetry alone, so
focus can be set open-loop from a table instead of being driven by the wavefront sensors.

**Result.** Five thermal channels — the Telescope Mount Assembly (TMA) truss temperature and the
four M1M3 thermal gradients — fitted jointly with one band-independent Huber robust linear model
predict the focus error to **61.3 µm of equivalent hexapod dz** from an uncorrected **330.7 µm**,
which is 19% of the original scatter (dimensionless, residual normalized median absolute deviation
(nMAD) over uncorrected nMAD). The truss temperature carries most of it at **+124.64 µm of
equivalent hexapod dz per °C**, agreeing to 16.5% (dimensionless) with the independent FAM Double
Zernike (DZ) measurement. **After that correction no elevation dependence remains**, so temperature
alone sets the table.

## The response

For every science visit the response is the focus error the closed loop had accumulated but not
yet corrected:

```
response = v1(commanded Trim) + MEASURED_SIGN * v1(measured state)
```

with `MEASURED_SIGN = -1`, so the response is `Trim − measured`. Here `v1` is the amplitude of the
first singular vector of the AOS sensitivity matrix — v-mode 1, essentially uniform defocus.

The elevation- and temperature-dependent hexapod LUT baseline is **deliberately excluded** from the
response. A physical hexapod position is LUT + Trim, but the LUT term is a known commanded
function of elevation, so including it would put a large elevation dependence into the response
that has nothing to do with the thermal question. Both commanded vectors come from the Engineering
Facility Database (EFD) through the value-added database, because the ConsDB copy of the Trim
reaches only about a third of science exposures.

The response is expressed throughout as **equivalent hexapod dz [µm]**: the total defocus travel,
shared as 0.5 µm on the camera hexapod and 0.5 µm on the M2 hexapod. The conversion is
`v1_per_um_dz = 9.00851e-04` dimensionless v-mode-1 amplitude per µm of total dz travel, the mean
magnitude of camera degree of freedom (DOF) 5 and M2 DOF 0. The two axes share a sign and agree to
2.1% (dimensionless), so their sum over their mean is 2.00000 (dimensionless) — that factor of two
is what makes the unit *total* travel rather than one hexapod's.

The sign is a convention fixed by observation, not by a fit: on the `BLOCK-T539`
`infocus_initial_alignment` sequence the AOS answered a +3.87 µm of wavefront focus error with
−119.7 µm of camera hexapod dz, so the commanded motion opposes the measured Z4 and a surviving
measured residual enters the sum with the sign that cancels it.

Two unit traps apply when commanded vectors are added: `lut_dof3/4/8/9` are hexapod tilts in
**deg** as `MTHexapod` reports them, while the Trim `dof3/4/8/9` are in **arcsec** following the
Optical Feedback Control (OFC) convention; and the LUT covers only the 10 hexapod DOF, so the
mirror bending entries are zero rather than absent.

## The measured term: optical state, not four-corner mean Z4

Four field points cannot separate a field-constant defocus from real field tilt, so the
four-corner mean Z4 used in the `correlations` notebook is an estimator of convenience. This
study instead runs the per-corner Zernike **deviations** — the ConsDB total optical path
difference (OPD) minus an intrinsic wavefront — through the OFC sensitivity-matrix singular
value decomposition (SVD), recovering DOF and v-mode amplitudes properly.

That recovery is a value-added quantity in the repository's EFD/ConsDB database rather than a
calculation in this study: it is held in `output/value_added/aos_efd.duckdb` as one
`optical_state` variant per combination of

* **scheme** — `22_12` (22 DOF, 12 v-modes, what the AOS runs online) or `50_34`;
* **intrinsic route** — `batoid` (the ts_ofc design intrinsic) or `miw` (the Measured Intrinsic
  Wavefront evaluated at the corner field points);
* **OPD version** — a provenance tag, so a reprocessing of the measured Zernikes arrives as new
  rows rather than overwriting the existing numbers.

The deliverable model is fitted on `v50_34__batoid__consdb_v1`.

## Sample

68,317 visits over 149 nights, `day_obs` 20251103 to 20260713, bands u g r i z y — from 73,982
visits over 161 nights before cuts. Two cuts apply:

* **8 nights running a different hexapod LUT configuration** are dropped (1,300 visits;
  `LUT_EPOCH_OFFSET_NIGHTS`, kept with `--keep-lut-epoch-offset-nights`). Their per-night offsets
  sit far from the rest because the commanded baseline itself changed.
* **341 visits on 20260116** have no truss temperature after in-night interpolation.

Per band: u 2,374, g 5,540, r 9,586, i 26,342, z 14,432, y 10,043 visits. The uncorrected response
has median +163.4 µm and nMAD 330.7 µm of equivalent hexapod dz.

## The fitted model

One Huber robust linear model, band independent, on five thermal features. The pipeline is a
median imputer, a standardizing scaler, then `HuberRegressor`; the coefficients below are the
physical ones, recovered from the standardized fit and verified to reproduce the pipeline's own
prediction to 1.4e-12 µm over all 68,317 visits.

```
response [um of equivalent hexapod dz, 0.5 um on each hexapod]

  = -1387.61
    +  124.64 * truss_temp_mean_c              [per deg C]
    -  808.72 * m1m3_z_gradient_c_per_m        [per deg C per m]
    - 1218.34 * m1m3_y_gradient_c_per_m        [per deg C per m]
    -  975.01 * m1m3_radial_gradient_c_per_m   [per deg C per m]
    - 3334.94 * m1m3_x_gradient_c_per_m        [per deg C per m]
```

The intercept is the response at zero in every feature. That is a long extrapolation from the
sample, whose feature means are truss +11.346 °C and the four gradients −0.0654, −0.0193, −0.0171
and +0.00159 °C per m; the intercept is not a physically meaningful offset on its own, only the
constant that makes the five slopes land on the data.

Coefficient stability, as the mean and scatter over the 5 night-grouped folds:

| feature | full sample | fold mean ± scatter | unit |
|---|---|---|---|
| `truss_temp_mean_c` | +124.64 | +124.87 ± 1.53 | µm equiv hexapod dz per °C |
| `m1m3_z_gradient_c_per_m` | −808.72 | −812.07 ± 38.93 | µm equiv hexapod dz per (°C per m) |
| `m1m3_y_gradient_c_per_m` | −1218.34 | −1237.72 ± 147.99 | µm equiv hexapod dz per (°C per m) |
| `m1m3_radial_gradient_c_per_m` | −975.01 | −906.76 ± 292.83 | µm equiv hexapod dz per (°C per m) |
| `m1m3_x_gradient_c_per_m` | −3334.94 | −3331.63 ± 283.41 | µm equiv hexapod dz per (°C per m) |

The truss term is stable to 1.2% across folds; the three weaker gradients are not, with the radial
term's scatter comparable to its own magnitude.

### FAM cross-check

The truss coefficient converts to **+0.11228 dimensionless v-mode-1 amplitude per °C** on the full
sample, and **+0.11249 per °C** as the fold mean. The independent FAM DZ value from
`code/correlations/run_dz14_truss.py` is **+0.09634 per °C**, so the two agree to **16.5%**
(full sample) and **16.8%** (fold mean), both dimensionless. A grossly different slope would be a
units or index error rather than a discovery.

### Calibration against the response

The out-of-fold prediction against the measured response, with a Huber slope of measured on
predicted (dimensionless — a calibrated model gives 1):

| band | n | Pearson r | Spearman rho | robust slope | residual nMAD [µm equiv hexapod dz] |
|---|---|---|---|---|---|
| all | 68,317 | +0.8716 | +0.9581 | +0.992 | 61.3 |
| u | 2,374 | +0.9158 | +0.9370 | +0.908 | 64.7 |
| g | 5,540 | +0.8239 | +0.9691 | +0.934 | 53.6 |
| r | 9,586 | +0.9579 | +0.9611 | +0.974 | 57.4 |
| i | 26,342 | +0.8321 | +0.9515 | +0.997 | 62.3 |
| z | 14,432 | +0.9765 | +0.9760 | +1.017 | 63.2 |
| y | 10,043 | +0.7489 | +0.8992 | +0.982 | 62.6 |

Pooled the model is calibrated at +0.992, and no band departs from unity by more than 9%.

## Elevation: nothing remains

Once the thermal correction is applied, the residual carries no useful elevation dependence.
Pooled over all bands the residual-against-elevation Huber slope is **+0.347 ± 0.023 µm of
equivalent hexapod dz per deg** with **Pearson r +0.061** (dimensionless), and removing it moves
the residual nMAD only from **61.3 to 60.9 µm** — a 0.7% gain. Adding elevation to the feature set
gives the same answer from the other direction: 61.3 µm without it, 61.0 µm with it.

Per night the picture is the same: over 127 nights with enough visits to fit, the median
elevation slope is **+0.025 µm of equivalent hexapod dz per deg** with nMAD **0.801 µm per deg**,
scattering about zero. Splitting each night into rising and falling legs, the median
rising-minus-falling slope difference is **+0.233 µm per deg** with nMAD **0.822 µm per deg**, and
63 of 109 nights have the rising leg steeper (sign-test p = 0.125) — no consistent direction
dependence, so no hysteresis term is warranted.

No elevation stage is therefore subtracted. The diagnostic plot showing that nothing is left is
the result, and it stays in the document.

The reason is that the hexapod LUT already handles elevation. Fitted on its own against
elevation the LUT term has a slope of about **−17.4 µm of equivalent hexapod dz per deg** with
Pearson r near −0.95 in every band, two orders of magnitude larger than what survives in the
residual.

## Method notes

### Night-grouped evaluation is required

Within a night the thermal telemetry drifts slowly, so consecutive visits are near-duplicates in
feature space: only 1.9% of the truss temperature's variance is within-night. A visit-level
train/test split therefore lets a model identify the night from its temperature and recall that
night's offset, and every score here comes from `GroupKFold` grouped on `day_obs`. The size of the
trap depends on model capacity — a factor of 3.1 (dimensionless, visit-level nMAD over
night-grouped nMAD) for boosted trees, and small for the five-coefficient linear fit actually
used. `--leaky-split` reproduces the visit-level number for comparison; it is not a performance
estimate.

The same reasoning applies within the sample itself: the per-night offset nMAD of 55.6 µm against
a median within-night residual nMAD of 32.5 µm is a ratio of 1.71 (dimensionless), so
night-to-night offset variation is the larger of the two and is what a held-out night must be
predicted through.

### Model choice

Night-grouped 5-fold, truss plus the four M1M3 gradients, against the uncorrected baseline of
330.7 µm of equivalent hexapod dz:

| model | residual nMAD [µm equiv hexapod dz] | fraction of baseline [dimensionless] | R² [dimensionless] |
|---|---|---|---|
| Huber + splines (4 knots) | 58.0 | 0.175 | 0.813 |
| **Huber linear** | **61.3** | **0.185** | **0.751** |
| Ridge linear | 70.2 | 0.212 | 0.749 |
| HistGB | 79.9 | 0.242 | 0.780 |
| RandomForest | 88.1 | 0.266 | 0.696 |

The response is close to linear: splines buy 5% and the trees are worse, fitting night-specific
structure that does not transfer to held-out nights. The linear fit is kept because it is
interpretable as a coefficient per °C and is what a look-up table needs.

### Feature set

Truss temperature stays primary on physical grounds — the truss is the load path setting the
M1M3-to-camera spacing. Huber, night-grouped 5-fold:

| feature groups | n features | residual nMAD [µm equiv hexapod dz] | R² [dimensionless] |
|---|---|---|---|
| truss | 1 | 151.1 | 0.579 |
| truss + M1M3 z gradient | 2 | 67.2 | 0.744 |
| **truss + four M1M3 gradients** | 5 | **61.3** | **0.751** |
| truss + four gradients + elevation | 6 | 61.0 | 0.752 |
| truss + four gradients + wind | 8 | 60.7 | 0.753 |

The z gradient does most of the work beyond the truss; the other three gradients add 6 µm; and
neither elevation nor wind adds anything. `cam_AverageTemp` correlates marginally better with the
response than the truss temperature but is collinear with it at Pearson r 0.9663, which
destabilises the truss coefficient and hides the FAM cross-check, so it is excluded.

### Fitting

Huber is the primary fit at every stage — `HuberRegressor` inside the model pipeline, and
`RLM(y, X, M=HuberT())` for the diagnostic slopes on the pages. Every fit reports both Pearson r
and Spearman rho, n, and `nmad(residuals)`. Scatter is quoted as nMAD rather than standard
deviation because the tails are heavy: the plain root mean square (RMS) of the measured state runs
1.2 to 1.5 times the robust RMS depending on band.

### Band independence and filter changes

The model is fitted once for all bands, not per band. Fitting per band would let the coefficients
change at every filter change, injecting a step into the corrected residual where nothing physical
has happened.

A band-independent correction cannot remove a real per-band focus offset, and the residual medians
show a small one — u +18.0 µm against z −13.2 µm of equivalent hexapod dz. Consistent with that,
the median absolute step between consecutive visits is 18.3 µm at a band change (n = 1,161) against
8.7 µm within a band (n = 67,007) before the correction, a ratio of 2.12 (dimensionless), and
18.8 / 8.8 µm for a ratio of 2.15 after it. The correction is band-blind by construction, so the
step survives it unchanged; a per-band offset would have to be added separately.

## Code

| file | role |
|---|---|
| `code/science_lut/run_science_lut.py` | assembly: live ConsDB queries plus the value-added database, writing the per-visit table |
| `code/science_lut/run_science_lut_analysis.py` | the whole analysis: the thermal model, its coefficients and calibration, the elevation null result, and the per-night diagnostics, as one document |

The analysis script reads `science_lut.parquet` and nothing else — no ConsDB or EFD access — so it
runs on a laptop from synced output. Its options: `--variant`, `--features`, `--model` and
`--n-splits` select what is fitted; `--day-obs` adds per-night detail pages; `--all-nights` adds
the full per-night grid; `--modulators` adds the second-order candidate pages;
`--no-model-scan` drops the model-comparison page; `--keep-lut-epoch-offset-nights` restores the
8 dropped nights; `--leaky-split` reproduces the visit-level score.

Output goes to `output/science_lut/`. That is the top level, outside any `param_set` or `mi_name`:
the study consumes science exposures and the database rather than the FAM donut tables, so no
Butler collection or processing variant enters. Where a variant's optical state does depend on a
Measured Intrinsic Wavefront build, that dependence is recorded in the `state_variant` registry
as `intrinsic_ref` and carried in the `variant` column of every output row, so it stays visible
without appearing in the path.

| product | content |
|---|---|
| `science_lut.parquet` | one row per visit per variant: identity, band, program, pointing, the v-mode-1 components and their total, and the telemetry used |
| `science_lut_fits.parquet` | one row per (variant, band, stage, method): slope with units, intercept, Pearson r, Spearman rho, n, `nmad(residuals)`, `chi2/dof` |
| `science_lut_analysis.pdf` | the single analysis document: 22 pages by default, opening with a description of the study and closing with the per-night diagnostics |
| `science_lut_model.parquet` | one row per model term plus the intercept: full-sample coefficient, fold mean and scatter, feature mean, units, and the sample and score columns |
| `science_lut_predictions.parquet` | one row per visit: `visit_id`, `day_obs`, `seq_num`, `band`, elevation, response, out-of-fold prediction and residual, in µm of equivalent hexapod dz |
| `science_lut_nights.parquet` | one row per night: elevation slope and offset at 60 deg for all points and for the rising and falling legs, within-night residual nMAD, and n |

The database itself is built by `common/scripts/build_efd_db.py` and
`common/scripts/build_optical_state.py`, outside this topic, because it serves the whole
repository rather than this study.

## Relation to the other focus studies

| study | measurement | sample |
|---|---|---|
| [`lut`](lut.md) | FAM DZ fits over 189 detectors, collapsed to one static DOF vector | dedicated FAM visits |
| [`correlations`](correlations.md) | the same truss-temperature question, on both FAM DZ(1,4) and the four-corner mean Z4 | FAM visits and science visits |
| `science_lut` | the temperature-dependent focus surface from the CWFS optical state | all science visits |
| [`fam_focus`](fam_focus.md) | drift of the same v-mode-1 response within one fixed-pointing FAM block | FAM `acq` visits |

The `lut` study produces one static vector; this study produces a dependence on temperature. They
are separate outputs and neither consumes the other.

`fam_focus` does consume this study: it applies the model fitted here, unchanged, to the in-focus
visit of each FAM triplet, and finds that the correction **increases** within-block scatter by a
factor 1.31 (dimensionless, corrected over uncorrected median peak-to-peak). The model is fitted
between nights, and within one block the thermal inputs move too little for it to describe the
drift.

## Notebook

`notebooks/science_lut/science_lut_explore.ipynb` reads `science_lut.parquet` and the database
only, with no EFD access, so it runs on a laptop from synced output.
