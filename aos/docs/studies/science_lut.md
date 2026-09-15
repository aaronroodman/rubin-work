# Study: `science_lut` — a focus look-up table from science exposures

> **Status:** current · **Last updated:** 2026-09-14 · **Kind:** reference (study)

Construction of a focus Look-Up Table (LUT) for the Active Optics System (AOS) from ordinary
**science** exposures rather than from Full Array Mode (FAM) data. The measured quantity is the
optical state at the four Corner Wavefront Sensors (CWFS), which the Consolidated Database
(ConsDB) records for every science visit, so the sample is the whole survey instead of the few
hundred dedicated FAM visits.

The dependence being characterized has two terms, extracted in that order:

1. **Telescope Mount Assembly (TMA) truss temperature.** The uniform-defocus content of the
   commanded optical state tracks truss temperature at close to +0.1 dimensionless v-mode-1
   amplitude per °C, consistent with the thermal expansion of a steel truss.
2. **Elevation, in the residual.** Once the truss-temperature relation is removed, a
   substantial elevation dependence remains — larger than the gravitational flexure the
   existing hexapod LUT already compensates, and apparently modulated by factors that are not
   yet identified.

Four candidate modulators of that residual are tested: a hexapod motion-history proxy for
heating or hysteresis, the M1M3 thermal gradients, wind direction relative to the pointing
azimuth, and the Danish wavefront-estimation version epoch.

## Relation to the other focus studies

| study | measurement | sample |
|---|---|---|
| [`lut`](lut.md) | FAM Double Zernike (DZ) fits over 189 detectors, collapsed to one static DOF vector | dedicated FAM visits |
| [`correlations`](correlations.md) | the same truss-temperature question, on both FAM DZ(1,4) and the four-corner mean Z4 | FAM visits and science visits |
| `science_lut` | the pointing- and temperature-**dependent** focus surface from the CWFS optical state | all science visits |

The `lut` study produces one static vector; this study produces a dependence on temperature and
elevation. They are separate outputs and neither consumes the other.

## The measured term: optical state, not four-corner mean Z4

Four field points cannot separate a field-constant defocus from real field tilt, so the
four-corner mean Z4 used in the `correlations` notebook is an estimator of convenience. This
study instead runs the per-corner Zernike **deviations** — the ConsDB total optical path
difference (OPD) minus an intrinsic wavefront — through the OFC sensitivity-matrix singular
value decomposition (SVD), recovering degrees of freedom (DOF) and v-mode amplitudes properly.

That recovery is a value-added quantity in the repository's EFD/ConsDB database rather than a
calculation in this study: it is held in `output/value_added/aos_efd.duckdb` as one
`optical_state` variant per combination of

* **scheme** — `22_12` (22 DOF, 12 v-modes, what the AOS runs online) or `50_34`;
* **intrinsic route** — `batoid` (the ts_ofc design intrinsic) or `miw` (the Measured Intrinsic
  Wavefront evaluated at the corner field points);
* **OPD version** — a provenance tag, so a reprocessing of the measured Zernikes arrives as new
  rows rather than overwriting the existing numbers.

`run_science_lut.py` reads those variants and can be given several at once, which turns the
batoid-versus-MIW and 22/12-versus-50/34 comparisons into one run.

## The commanded term and the sign convention

The total v-mode-1 amplitude per visit is

```
v1_total = v1(hexapod LUT) + v1(Trim) + MEASURED_SIGN * v1(measured)
```

with `MEASURED_SIGN = -1`. The LUT is the elevation- and temperature-dependent hexapod
baseline and Trim is the accumulated closed-loop offset; a physical hexapod position is
LUT + Trim, neither alone. Both come from the Engineering Facility Database (EFD) through the
value-added database, because the ConsDB copy of the Trim reaches only about a third of science
exposures.

The sign is a convention fixed by observation, not by a fit: on the `BLOCK-T539`
`infocus_initial_alignment` sequence the AOS answered a +3.87 µm of wavefront focus error with
−119.7 µm of camera hexapod dz, so the commanded motion opposes the measured Z4 and a surviving
measured residual enters the sum with the sign that cancels it.

Two unit traps apply when the two commanded vectors are added: `lut_dof3/4/8/9` are hexapod
tilts in **deg** as `MTHexapod` reports them, while the Trim `dof3/4/8/9` are in **arcsec**
following the OFC convention; and the LUT covers only the 10 hexapod DOF, so the mirror bending
entries are zero rather than absent.

## Fitting

Huber `RLM(y, X, M=HuberT())` is the primary fit at every stage, with Theil-Sen on the same
sample as a cross-check. Theil-Sen is non-parametric in the slope, so a large disagreement
between the two flags leverage from the elevation-extreme points rather than a real trend. Every
fit reports both Pearson r and Spearman rho, n, and `nmad(residuals)`.

Fits are done **per band**: filter thickness changes the camera-hexapod dz look-up table, so a
pooled fit would leave a band-to-band focus offset in the residual as four offset clusters.

The truss slope is cross-checked against the FAM commanded value of +0.09634 dimensionless
v-mode-1 amplitude per °C from `code/correlations/run_dz14_truss.py`. A grossly different slope
is a units or index error rather than a discovery.

## Code

| file | role |
|---|---|
| `code/science_lut/run_science_lut.py` | the whole study: assemble, read the optical state, fit truss temperature, fit the elevation residual and its modulators |

Reads the value-added database (`common/efd_db.py`) plus live ConsDB queries, and writes to
`output/science_lut/`. That is the top level, outside any `param_set` or `mi_name`: the study
consumes science exposures and the database rather than the FAM donut tables, so no Butler
collection or processing variant enters. Where a variant's optical state does depend on a
Measured Intrinsic Wavefront build, that dependence is recorded in the `state_variant` registry
as `intrinsic_ref` and carried in the `variant` column of every output row, so it stays visible
without appearing in the path.

| product | content |
|---|---|
| `science_lut.parquet` | one row per visit per variant: identity, band, program, pointing, the v-mode-1 components and their total, the telemetry used, and the residual after each fit stage |
| `science_lut_fits.parquet` | one row per (variant, band, stage, method): slope with units, intercept, Pearson r, Spearman rho, n, `nmad(residuals)`, `chi2/dof` |
| `science_lut.pdf` | program inventory, the truss fits per band, the elevation residual, one page per candidate modulator, and the column-coverage table |

The database itself is built by `common/scripts/build_efd_db.py` and
`common/scripts/build_optical_state.py`, outside this topic, because it serves the whole
repository rather than this study.

## Notebook

`notebooks/science_lut/science_lut_explore.ipynb` reads `science_lut.parquet` and the database
only, with no EFD access, so it runs on a laptop from synced output.
