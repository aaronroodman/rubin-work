# Study: `thermal_focus` — the focus error as a function of temperature

> **Status:** current · **Last updated:** 2026-09-24 · **Kind:** reference (study)

> **Code:** `code/` · **Notebooks:** `notebooks/`
> **Output:** `output/` (`thermal_focus.pdf`, `thermal_focus.parquet`,
> `thermal_focus_t539.parquet`, `<fam_dir>/thermal_focus_fam.parquet`)

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
| visits with a recovered `v50_34__batoid__consdb_v1` optical state | 85,819 |
| joined to `visit_telemetry` gradients | 85,713 (99.9%) |
| joined to the quadratic radial terms | 85,819 (100.0%) |
| `img_type = 'science'` | 71,069 |
| in bands u g r i z y | 71,069 |
| excluding 8 LUT-epoch nights | −1,504 visits |
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

### Quadratic radial M1M3 terms

The four bulk gradients are all **linear**: the temperature field is fitted as
`T = a0 + a1·x + a2·y + a3·z` and as `T = a0 + a1·r + a3·z` over the 146 thermocouples in the
mirror glass. A field going as radius squared bends the mirror into a shape much closer to pure
defocus than a linear radial ramp does, and so is the term most likely to move focus — but no
linear gradient can express it. Three such terms are therefore computed and tested here, over
three thermocouple populations: the whole mirror, the M1 annulus alone (80 thermocouples, 2.997
to 4.197 m) and the M3 inner disc alone (66 thermocouples, 0.555 to 2.533 m). M1 and M3 are one
monolithic blank but two optical surfaces at different radii and curvatures, so a thermal
expansion confined to one of them is a different optical perturbation than the same expansion
over both.

They live in the value-added database's `m1m3_thermal_r2` table, built by
`value_added/code/build_m1m3_thermal_r2.py`; the reduction and the mirror split are documented in
[`value_added/docs/schema.md`](../../value_added/docs/schema.md). Two points about the unit matter
for reading a coefficient here:

- The coefficient is **°C per unit normalized radius-squared amplitude**, not °C/m². The
  quadratic shape is Gram-Schmidt orthogonalized against the constant, linear-radius and depth
  terms over each population's own sensor positions and scaled to unit root-mean-square. Fitting
  raw `r²` beside the linear terms is not viable: over M1's narrow annulus `r²` is 99.875%
  explained by them, a variance inflation factor of 801.8 (dimensionless).
- Because of that orthogonalization the coefficient carries **only** the radial curvature the
  linear terms cannot express, which is what makes it the right variable for an "above and beyond
  the gradients we already have" test rather than a second copy of the radial gradient.

Orthogonalizing against sensor *positions* fixes the design matrix, not the time series. The
quadratic terms remain correlated with the existing radial gradient over time — over the science
sample, Pearson r +0.8991 for the whole-mirror term, +0.4301 for M1 and +0.5571 for M3
(dimensionless, n = 68,079 visits; Spearman rho +0.7607, +0.3692, +0.4051) — because the upstream
radial gradient is itself fitted with no quadratic term and so absorbs whatever curvature exists.
That redundancy is a question for the regression, not for the reduction, which is why it is
answered here by partial correlation and a nested model comparison rather than by trying to remove
it upstream.

Section 5b of the analysis reports four things, in increasing strength of claim:

1. the **raw** Huber relation of each term to the focus error, with Pearson r and Spearman rho;
2. how much each **duplicates** the existing M1M3 radial gradient;
3. the **partial** correlation, with the truss temperature and the four bulk gradients regressed
   out of both the candidate and the response — the "above and beyond" test;
4. a **night-grouped nested** comparison, `GroupKFold` on `day_obs`, of the five-feature
   deliverable against the same features plus the candidates, scored on identical rows so a
   sparser candidate is not credited with an easier sample. Only this one is a performance claim.

A fifth row covers **substitution** rather than addition: the truss temperature plus the three
quadratic terms fitted in place of the truss plus the four bulk gradients, on the same rows. That
is the comparison a decision to switch from the gradients would rest on, and it is reported
separately because a term can add information without being a better replacement.

#### What the quadratic terms are worth

Measured over 68,079 science visits on 147 nights, `day_obs` 20251103 to 20260713. Raw and
partial Huber relations to the focus error, the partial being with the truss temperature and the
four bulk gradients regressed out of both sides:

| term | raw Pearson r | partial Pearson r | partial Spearman rho | partial slope [µm equiv hexapod dz per unit normalized r² amplitude] |
|---|---|---|---|---|
| whole-mirror `m1m3_r2_coeff_c` | −0.1306 | −0.0715 | −0.0689 | −115.4 ± 16.3 |
| M1 `m1_r2_coeff_c` | −0.2235 | −0.0641 | −0.2043 | −1992.2 ± 30.4 |
| M3 `m3_r2_coeff_c` | −0.1490 | −0.1342 | −0.0870 | −377.1 ± 14.4 |

All three correlations are dimensionless; the raw Huber slopes are −1515.7 ± 33.1, −5586.5 ± 75.1
and −2278.9 ± 53.3 µm of equivalent hexapod dz per unit normalized r² amplitude. Every partial
slope is many times its formal error, so the information is real, but every partial correlation is
small: the quadratic curvature is a **weak** predictor once the bulk gradients are in the model.

The night-grouped nested comparison is the performance claim, baseline residual nMAD 59.9 µm of
equivalent hexapod dz on the same 68,079 visits:

| added to the five deliverable features | residual nMAD [µm equiv hexapod dz] | gain [dimensionless, baseline nMAD over extended nMAD] | ΔR² [dimensionless] |
|---|---|---|---|
| whole-mirror term alone | 60.0 | 0.9988 | +0.0004 |
| M1 and M3 terms | 57.1 | **1.0484** | +0.0011 |
| all three terms | 57.8 | 1.0356 | −0.0001 |

The whole-mirror term adds nothing — it is the one most nearly duplicated by the existing radial
gradient, at Pearson r +0.8991. **The M1 and M3 split is what carries the new information**, a
4.8% reduction in robust residual scatter, and adding the whole-mirror term back on top of the
split makes it slightly worse, which is what redundancy looks like in a cross-validated score.
Splitting the mirror was therefore the part of the design that mattered, not the quadratic radial
shape by itself.

Substitution goes the other way. On the same 68,079 visits over 147 nights, the truss temperature
plus the four bulk gradients gives residual nMAD 59.9 µm of equivalent hexapod dz at R² +0.5181,
while the truss temperature plus the three quadratic terms gives 72.4 µm at R² +0.5008 — a ratio
of **0.8269** (dimensionless, gradient nMAD over quadratic nMAD), where above 1 would favour
switching. **The quadratic terms do not replace the bulk gradients**; the deliverable feature set
is unchanged, and the useful form of this result is the M1-plus-M3 pair added to the existing five
features.

One property of the reduction is worth knowing when the sample sizes differ: its coverage is
**higher** than the bulk gradients'. The upstream reduction NaNs an entire time sample if any one
thermocouple dropped out, while this fit groups samples by their finite pattern and reuses one
pseudo-inverse per pattern. On the science sample the quadratic terms resolve **100.0%** of visits
against 99.9% for the bulk gradients.

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

#### Where `FAM_TRUSS_SLOPE` comes from, and the truss expansion it implies

The constant is the Huber slope of v-mode 1 of the Trim against truss temperature over the
FAM DZ fits on the `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` param_set. Two properties of that
measurement are load-bearing and are the reason the constant is a slope of the commanded Trim.

**The Trim falls into two populations that must be fitted separately.** v-mode 1 from the Trim
alone against truss temperature forms two bands of similar slope separated by an offset of
**+2.545** (dimensionless v-mode-1 amplitude) at fixed temperature, across an empty gap of 1.871
in the residual about a common line. Pooling them reverses the apparent correlation — a
Simpson's-paradox artefact:

| population | n [visits] | Huber slope [dimensionless per °C] | intercept [dimensionless] | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| pooled (misleading) | 1,591 | +0.0799 ± 0.0017 | −2.105 | −0.051 | +0.425 |
| upper | 224 | +0.0740 ± 0.0036 | +0.204 | +0.734 | +0.726 |
| lower | 1,367 | +0.0963 ± 0.0013 | −2.342 | +0.902 | +0.869 |

`FAM_TRUSS_SLOPE` is the **lower** population's slope. The split is mostly but not purely
temporal — the upper covers 14 nights (day_obs 20251023 to 20251219), the lower 44 nights
(20251103 to 20260713), and three nights carry visits from both — so no single date boundary
defines it.

**The slope is what a thermally expanding steel truss predicts, to 14%.** Three conversions
chain together:

| quantity | value | source |
|---|---|---|
| d(v1) / d(truss T) | +0.09634 per °C (dimensionless v-mode-1 amplitude per °C) | Huber slope of the lower Trim population, n = 1,367 visits |
| v1 per hexapod dz | 9.0095 × 10⁻⁴ per µm | mean of the camera- and M2-hexapod dz coefficients of v-mode 1 |
| DZ(1,4) to hexapod dz | −1110.0 µm of hexapod dz per µm of wavefront | `U_eff[(1,4),0]` = −0.999919 (dimensionless) |

Dividing the first by the second gives **106.9 µm of hexapod dz per °C**, equivalently 0.0963 µm
of wavefront of DZ(k=1, j=4) per °C. A steel truss of the LTS-213 length — 7,835 mm from the
elevation axis to the top of the lower top-end right light baffle — expands **94 µm per °C** at a
coefficient of thermal expansion of 12 ppm per °C. The ratio is **1.14** (dimensionless, measured
over predicted). Attributing the excess to geometry alone would need an effective length of
8,911 mm; to material alone, 13.6 ppm per °C at the LTS-213 length. The LTS-213 assembly drawing
is at <https://docushare.lsst.org/docushare/dsweb/Get/LTS-213> and is not kept in this repository.

This is the *commanded* sensitivity. It does not conflict with the measured DZ(1,4) being nearly
uncorrelated with truss temperature (Huber slope −0.0330 ± 0.0052 µm of wavefront per °C, Pearson
r −0.124, Spearman rho −0.108, n = 1,591 visits): the hexapod is driven as though the truss were
expanding thermally, and what survives that correction is the response this study models.

**The hexapod LUT is two-dimensional**, in elevation and camera rotator angle, so neither
one-dimensional projection is a single curve and structure against rotator angle is not a defect.
v1 from the LUT against elevation traces parallel branches, one per LUT version, with the last LUT
change on or about day_obs 20251209. Restricting to day_obs ≥ 20251209 so one version is in force
gives a single-valued map over 5 deg × 10 deg bins in (elevation, camera rotator angle): 1,267
visits, 56 of 180 bins occupied, elevation 22.9 to 75.0 deg, rotator angle −70.1 to +60.2 deg.
The uniform defocus is no quieter after that change — median per-night nMAD 0.1670 µm of wavefront
over the 65 nights before against 0.2258 µm of wavefront over the 32 nights on and after — so the
intra-night focus excursions are not an artefact of an unsettled LUT version.

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

## The correction as degrees of freedom

The response is one number per visit, a v-mode-1 amplitude. An observer acts on degrees of freedom
(DOF), so the measured amplitude is projected back into the DOF it is built from, using the
library's own inverse, `StateEstimator.get_dofs_from_vmodes`:

```
dof = normalization_matrix @ (v_modes @ Vh)
```

**`Vh[0]` alone is not the answer.** The normalization matrix is not optional: taking the raw right
singular vector gives a DOF vector whose forward projection is v-mode-1 amplitude +0.0141 with
another mode at 0.227, instead of the +1.0000000000 with a largest other mode of 2.6e-16
(dimensionless) that the normalized inverse round-trips to.

Setting every other v-mode to zero is **exact, not an approximation**. `Vh` is orthonormal, so the
result is the unique minimum-norm DOF vector consistent with the amplitude being projected — the
smallest motion that delivers the required defocus, which is what an observer wants.

At v-mode-1 amplitude 1.0 (dimensionless), `dof_set` `all_50` with 34 modes retained:

| DOF | index | value [µm per unit v-mode-1 amplitude] |
|---|---|---|
| camera hexapod dz | `dof5` | −645.6579 |
| M2 hexapod dz | `dof0` | −463.8983 |
| M1M3 bending mode B3 | `dof12` | +0.0094 |
| M2 bending mode B5 | `dof34` | +0.0076 |

The two hexapod dz values carry the defocus and move together in a fixed ratio, because v-mode 1 is
one direction in DOF space. The two mirror bending modes are real but tiny: at the 99th-percentile
absolute amplitude the correction asks for, 0.71968 (dimensionless), they reach only **6.7576 nm and
5.4340 nm**, so an observer applying this correction can leave them alone. M2 bending mode B4 does not
appear at all — it enters at +0.0002 µm per unit v-mode-1 amplitude, below even those two.

### The Trim the correction would command

The quantity worth tabulating is the motion an observer would command, not what is left over after
commanding it. The predicted focus error is converted back to an amplitude,
`v1_applied = predicted focus error × v1_per_um_dz`, and back-projected. The commanded Trim term
enters the response with a positive sign, so the amplitude needed in the Trim to cancel a predicted
error is that error in v-mode-1 units, with **no sign flip**. Over the 68,079-visit sample:

| DOF | median | nMAD | 1st pct | 99th pct | unit |
|---|---|---|---|---|---|
| camera hexapod dz | −86.6304 | 189.1334 | −432.5783 | +425.6159 | µm |
| M2 hexapod dz | −62.2430 | 135.8903 | −310.8029 | +305.8005 | µm |
| M1M3 bending mode B3 | +1.2599 | 2.7505 | −6.1897 | +6.2909 | nm |
| M2 bending mode B5 | +1.0131 | 2.2118 | −4.9773 | +5.0587 | nm |

The correction asks for **hundreds of µm** of hexapod dz and **single-digit nm** of either bending
mode, which is the practical statement: this is a two-axis hexapod correction and the mirror figure
can be left alone. How well the correction works, rather than how large it is, is what the training
section measures.

### Start of night

The first visit of a night is the one an open-loop focus setting has to be right for, before any
wavefront measurement has been folded in, so the Trim the correction asks for there is the size of
the motion that matters most. Over 147 nights, MJD 60983.202 to 61235.045:

| DOF | median | nMAD | slope against date, per d | unit |
|---|---|---|---|---|
| camera hexapod dz | −107.0654 | 190.0861 | +0.52648 ± 0.21942 | µm |
| M2 hexapod dz | −76.9254 | 136.5748 | +0.37827 ± 0.15765 | µm |
| M1M3 bending mode B3 | +1.5570 | 2.7644 | −0.00766 ± 0.00319 | nm |
| M2 bending mode B5 | +1.2521 | 2.2229 | −0.00616 ± 0.00257 | nm |

Every row gives the same significance, **2.4 standard errors** (Huber, n = 147 nights), the same
Pearson r **+0.1840** and the same Spearman rho **+0.2224** — with the sign reversed on the two
bending modes — because all four DOF are a fixed multiple of the one v-mode-1 amplitude. That is a
property of the projection, not four independent measurements.

At 2.4 standard errors over 147 nights this is a **weak positive trend, not a detection**: the Trim
the correction asks for at the start of a night is mostly scatter about a fixed offset. It is worth
re-testing as the season lengthens rather than quoting as a measured drift.

The start-of-night spread is **larger than the night-to-night spread of the rest of the night**.
Comparing like with like — both statistics over the same 147 nights — the camera hexapod dz has nMAD
190.0861 µm at the first visit against 164.9279 µm across the per-night medians, a factor of
**1.15** (dimensionless, start-of-night nMAD over per-night-median nMAD). The ratio is the same
1.15 for all four DOF, for the same reason the correlation coefficients are.

A first visit needing a larger correction than the night's own centre is consistent with the
telescope being furthest from thermal equilibrium at the start of a night, but that is an
interpretation: this study measures the spread and does not separate a loop-convergence transient
from a genuinely larger thermal excursion.

### Against the Trim the initial alignment block settled on

Every residual above is against the optical state recovered from the corner wavefront sensors, which
is the quantity the model was fitted to. `BLOCK-T539` is the initial alignment block run at the start
of each night, and it converges the commanded Trim without reference to the thermal telemetry, so the
Trim it arrives at is an **independent** measurement of the focus the telescope actually needed.

The two epochs compared are the **first** visit of the block's start-of-night run, whose thermal
telemetry feeds the prediction, and the **last** visit of that run, whose Trim is the settled value.
Per night, the exposures of `img_type` `science` or `acq` are sorted by `seq_num`, the first 10 are
taken, those carrying a `BLOCK-T539` program are selected, and the run is extended through the
contiguous `seq_num` from the lowest of them. Biases and darks are skipped by the `img_type` filter,
and `cwfs` is excluded because a wavefront pair is not the alignment exposure itself.

Two properties of the block that the selection has to accommodate:

- **Two program labels exist** — `BLOCK-T539` (166 nights) and `BLOCK-T539_hexapods` (12 nights), the
  latter a November 2025-era label — so the match is on the `BLOCK-T539` prefix.
- **The run is not a fixed 10 exposures.** Over the 178 selected nights the median is 17, the
  minimum 2 and the maximum 45 exposures (16 as the median over the 163 that survive the cuts);
  long runs are one label running 20 or more contiguous `acq`, not two chained. So the
  separation between the two epochs varies by night, and the run length is carried per night rather
  than assumed. 118 of 178 nights also have further block exposures later in the night, which is what
  the start-of-night restriction excludes.

The block is genuinely converging the Trim across the run: the camera hexapod dz Trim changes by a
median of **−253.9 µm** over it, and 170 of 178 nights move it by more than 1 µm.

Of 178 nights with a start-of-night run over `day_obs` 20251102 to 20260714, **163** survive the same
cuts the science sample takes — 8 LUT-epoch nights and 2 nights above 20 °C truss temperature. On
those 163 nights, with the prediction back-projected through v-mode 1 and every fit Huber:

| quantity | Pearson r | Spearman rho | Huber slope | actual − predicted, median | nMAD | unit |
|---|---|---|---|---|---|---|
| camera hexapod dz | +0.1123 | +0.2701 | +0.445 ± 0.105 | −99.763 | 239.279 | µm |
| M2 hexapod dz | +0.5819 | +0.7120 | +1.406 ± 0.100 | +77.599 | 140.011 | µm |
| M1M3 bending mode B3 | +0.0840 | +0.1022 | +3.530 ± 5.981 | −67.368 | 148.561 | nm |
| M2 bending mode B5 | +0.0914 | +0.1115 | +3.642 ± 5.209 | −48.458 | 107.824 | nm |
| v-mode-1 amplitude of the pair | +0.3870 | +0.7375 | +0.874 ± 0.041 | +0.052 | 0.139 | dimensionless |

The slope is dimensionless in every row, actual per predicted in that row's own unit.

**Pearson and Spearman disagree, and the Spearman value is the one to read.** The relation is far
more monotonic than it is linear, because a few nights with large commanded Trim dominate a
least-squares view of it. That gap is why every fit here is Huber rather than ordinary least squares,
and why both statistics are reported.

**The per-hexapod rows are the weaker ones because of how the alignment splits focus**, not because
the prediction is worse for one hexapod. The alignment is free to put focus on either, and does: it
leaves the camera hexapod dz Trim at exactly zero on 8 of the 163 nights and the M2 hexapod on
another 8. That split carries no optical meaning. Projecting the pair onto the v-mode-1 direction,

```
combined v-mode-1 amplitude = (dof5 × u5 + dof0 × u0) / (u5² + u0²)
```

with `u5` and `u0` the unit content from the table above, is insensitive to the split, and it is that
combined row — Spearman rho **+0.7375** over 163 nights — that answers the physical question. The two
hexapod rows are kept so the split stays visible rather than hidden inside the combination.

The agreement is not expected to be exact: the two epochs are separated by the whole run, so the
telescope's thermal state has moved between them, and the block's own convergence is not error-free.
The **correlation**, not the offset, is what this comparison establishes.

**The two mirror figure DOF cannot show a correlation at this amplitude.** v-mode 1 contains
**+9.390 nm** of M1M3 bending mode B3 and **+7.551 nm** of M2 bending mode B5 per unit amplitude,
against −645.7 µm and −463.9 µm for the two hexapod dz, so over these nights the predicted bending
Trim spans only **8.485 nm** and **6.823 nm** while the actual Trim has a standard deviation of
**242.6 nm** and **170.7 nm** — larger by factors of **28.6** and **25.0** (dimensionless, actual
standard deviation over predicted span). Whatever sets the mirror figure Trim, it is not this focus
correction, and the Huber slopes of +3.530 and +3.642 are consistent with zero at well under one
standard error of unity.

## Code

| file | role |
|---|---|
| `code/thermal_focus_lib.py` | the response definition, the conversions and the feature groups — one definition, so nothing can drift |
| `code/run_thermal_focus.py` | build: the value-added database plus live ConsDB, writing the cached tables |
| `code/thermal_focus_fit.py` | the fitting core: the models, night-grouped evaluation, the block assignment and the diagnostics |
| `code/run_thermal_focus_analysis.py` | the analysis: sixteen sections and one document, no network |
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

`dof_trim` is the form to command online. It returns the correction as the degrees of freedom the
Optical Feedback Control system sets — the camera and M2 hexapod dz plus the two mirror figure
bending modes v-mode 1 contains — rather than as a single focus number:

```python
from trim_calculator import dof_trim
out = dof_trim(truss_temp_c=8.4, z_gradient_c_per_m=0.10, y_gradient_c_per_m=-0.05,
               radial_gradient_c_per_m=0.02, x_gradient_c_per_m=0.01)
out['dof5'], out['dof0'], out['dof12'], out['dof34']   # um, ts_ofc DOF ordering
```

**Two split conventions coexist and must not be confused.** `trim_adjustment` splits the predicted
travel **evenly**, half on each hexapod, which is what the dz-equivalent unit means. `dof_trim`
back-projects through v-mode 1, which splits it **unevenly** — 58.2% of the travel on the camera
hexapod against 41.8% on M2, a ratio of 1.1638 (dimensionless, back-projected camera dz over
even-split camera dz) — because that is the shape of the optical mode. The two hexapod dz entries sum
to −1109.556 µm per unit v-mode-1 amplitude, the same total travel the dz-equivalent conversion
inverts, so the conventions agree on the total and differ only on the split. Both are printed side by
side by the module's command line.

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
| `thermal_focus_t539.parquet` | one row per night of the initial alignment block: the run's first and last visit, the thermal telemetry at the first suffixed `_first`, and the Trim DOF at the last suffixed `_last` |
| `<fam_dir>/thermal_focus_fam.parquet` | one row per FAM triplet whose `acq` visit has a recovered optical state, with the triplet's own DZ coefficients |
| `thermal_focus.pdf` | the analysis document, in three parts: before the correction, the training, and all the data |

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

Both known follow-ups are now closed.

The r²-like radial thermal mode is **done**: three quadratic radial terms over the whole mirror,
the M1 annulus and the M3 inner disc are built into the value-added database and tested in the
section above. The result is that the M1 and M3 pair adds a real but modest 4.8% reduction in
robust residual scatter on top of the five deliverable features, while the whole-mirror term adds
nothing and no quadratic set replaces the bulk gradients. Whether to adopt the M1 and M3 pair into
the deliverable feature set is a decision left open; the deliverable is unchanged pending it.

The comparison of the 50 DOF / 34 mode, 22/12 and 10/1 projections is **done** — the three
`v1_per_um_dz` values agree to 0.108% (dimensionless, spread over the 50/34 value), far below the
fit's own uncertainty, so no refit under another projection is needed. The conversion section above
carries the table.
