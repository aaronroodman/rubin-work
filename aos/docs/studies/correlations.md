# Study: `correlations` — what does the residual DZ correlate with?

> **Status:** current · **Last updated:** 2026-09-09 · **Kind:** reference (study)

Correlation analysis of the per-visit Double Zernike (DZ) coefficients remaining after
the measured intrinsic is subtracted: against each other, against Optical Feedback
Control (OFC) v-modes, and against telescope telemetry. Also the per-donut
primary→secondary aberration-pair correlations, on the single-Zernike values rather than
the DZ fits.

The first four scripts run on the **MI-refit** residual (`output/<ps>/<mi>/fits.parquet`),
not the raw DZ. All five are pipeline rules. Knobs live in `analysis_config.yaml`, kept
separate from `mi_config.yaml` so editing an analysis knob never re-triggers a slow
intrinsic build.

## Code

| file | role |
|---|---|
| `run_dz_correlations.py` | DZ_kj ↔ DZ_k'j' Pearson heatmap, top-\|r\| scatters, astigmatism-symmetry pairs, conjugate-orbit grids, Fisher-z significance. Also a `_optcorr` variant on the post-OFC-correction residual |
| `run_vmode_correlations.py` | project the MI-subtracted DZ onto the OFC SVD and correlate v-modes, for both 50/34 and 22/12 schemes |
| `run_thermal_correlations.py` | DZ_kj × EFD temperature-variable Pearson heatmap plus per-term scatter pages |
| `run_dz_explained.py` | per-visit fraction of the measured DZ explained by the OFC sensitivity subspace, 22/12 and 50/34 |
| `run_aberration_pairs.py` | per-donut primary→secondary aberration pairs (defocus→spherical, astigmatism→2nd astigmatism, and so on), split into quartiles of the primary |
| `run_dz14_truss.py` | the focal-plane-uniform defocus DZ(k=1, j=4) against Telescope Mount Assembly (TMA) truss temperature, and against v-mode 1 reconstructed from the commanded degrees of freedom in three cumulative forms (LUT, LUT+Trim, LUT+Trim+Deviation); includes a hexapod-LUT validation page and per-night traces of DZ(1,4) overlaid with truss temperature |

`run_aberration_pairs.py` works on the **Phase-1** per-donut `zk_<coord>` values in
`donuts.parquet`, so it needs no `mi_name` and writes to `output/<ps>/correlations/`.
It streams the donut table by row group.

`run_dz14_truss.py` reads the raw DZ fit (`output/<ps>/fits.parquet`), not the MI-refit
residual, because the quantity of interest is the absolute mean focus rather than a
residual. It takes the commanded degrees of freedom from `visits.parquet`: the hexapod
look-up table (LUT) as `lut_dof0..9` and the Trim as `dof0..49`. A hexapod position is
LUT + Trim, and on the camera-hexapod dz axis the two are anti-correlated with a Huber
slope of −1.089 ± 0.014 (dimensionless, v1 from LUT per unit v1 from Trim), so a v-mode
reconstructed from the Trim alone has the wrong variance, not merely a missing offset.
See [`../telemetry.md`](../telemetry.md) for the LUT / Trim / Tweak column mapping.

The **Deviation** term is v-mode 1 of the *measured* DZ, obtained by projecting the raw DZ
onto the same OFC singular-value decomposition that defines the v-modes, so
LUT + Trim + Deviation approximates the true optical state. Its sign is not fixed a
priori, so the script evaluates both and uses whichever gives the tighter relation with
truss temperature, recording both residual scatters in the summary parquet. On the
param_set below it chose +1 (residual nMAD 0.4625 versus 0.4832 µm of wavefront), and the
term is small either way: v1 Deviation has nMAD 0.0281 against 0.7321 for LUT + Trim
(both dimensionless).

## Outputs

`<mi>/plots/dz_correlations{,_optcorr}.{pdf,_pairs.parquet}`,
`vmode_correlations_{50_34,22_12}.pdf` + summary parquets,
`thermal_correlations.pdf` + `_summary.parquet`, `dz_explained.{pdf,parquet}`,
`<ps>/correlations/aberration_pairs.{pdf,_summary.parquet}`, and
`<ps>/correlations/dz14_truss_<dz_prefix>.{pdf,_summary.parquet}` (11 pages: four
one-per-page truss scatters, LUT validation, v1 LUT vs v1 Trim coloured by time, the
pooled and split Trim populations, then three pages of per-night traces).

The first four currently share `<mi>/plots/` with the bounce and coadd output; splitting
them per study is outstanding work.

## DZ(k=1, j=4) and truss temperature

On the `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` param_set, 2465 quality-passing FAM
visits of which 1591 carry a truss temperature: truss temperature accounts for almost
none of the uniform defocus. Huber slope −0.0330 ± 0.0052 µm of wavefront per °C, Pearson
r = −0.124, Spearman rho = −0.108, n = 1591, residual nMAD 0.444 µm of wavefront. Across
the full 13 °C range spanned by the data that trend gives only 0.43 µm of wavefront,
against a total DZ(1,4) scatter of 0.490 µm of wavefront (nMAD, n = 2465).

The per-night traces make the same point without any averaging. On day_obs 20260423,
DZ(1,4) spans 1.438 µm of wavefront while the truss temperature moves 0.163 °C, within a
single LUT plateau; at the pooled slope that temperature motion predicts 0.005 µm of
wavefront, some 270× smaller than observed. day_obs 20260404 is comparable at 1.633 µm of
wavefront against 0.401 °C. Whatever drives the intra-night focus excursions, truss
temperature is not it.

The commanded v-mode 1 does track truss temperature, as expected from a hexapod LUT that
is a function of elevation and temperature, and adding successive terms barely changes
that (n = 1591 throughout):

| v-mode 1 form | Huber slope [dimensionless per °C] | Pearson r | Spearman rho |
|---|---|---|---|
| LUT alone | +0.1212 ± 0.0046 | +0.413 | +0.556 |
| LUT + Trim | +0.1811 ± 0.0043 | +0.588 | +0.683 |
| LUT + Trim + Deviation | +0.1829 ± 0.0043 | +0.589 | +0.680 |

## Two Trim populations

v-mode 1 from the Trim alone against truss temperature falls into two bands of similar
slope separated by an offset of +2.545 (dimensionless) at fixed temperature, across a
completely empty gap of 1.871 in the residual about a common line. Pooling them reverses
the apparent correlation — a Simpson's-paradox artefact — so they must be fitted
separately:

| population | n | Huber slope [dimensionless per °C] | intercept [dimensionless] | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| pooled (misleading) | 1591 | +0.0799 ± 0.0017 | −2.105 | −0.051 | +0.425 |
| upper | 224 | +0.0740 ± 0.0036 | +0.204 | +0.734 | +0.726 |
| lower | 1367 | +0.0963 ± 0.0013 | −2.342 | +0.902 | +0.869 |

The split is *mostly* but not purely temporal: the upper population covers 14 nights
(20251023 to 20251219) and the lower 44 nights (20251103 to 20260713), and three nights —
20251211, 20251212 and 20251218 — contain visits from both. No single date boundary
therefore defines the populations, which is why the script splits on the residual gap and
then reports the date ranges rather than assuming a cut date.

## Hexapod LUT validation

v-mode 1 from the LUT against elevation traces several distinct parallel curves rather
than one, each a different LUT version. Fitting a cubic in elevation and taking the
residual separates the LUT changes from the elevation dependence itself, and confirms that
the LUT settled at the end of 2025: residual nMAD 0.9310 (n = 904) before day_obs
20251101 versus 0.0859 (n = 1515) on and after, a factor 10.8 reduction, both
dimensionless. Against camera rotator angle v1 LUT is flat, as it must be — the hexapod
LUT has no rotator dependence, so that panel is a null test.

The v1 LUT versus v1 Trim scatter separates into two clusters that the time colouring
identifies as epochs, not distinct physical states: the lower-right cluster is the first
~100 days of the sample and the upper-left cluster days 200 to 350.

## Statistical cautions

These are correlation studies, so the reporting rules matter more than usual:

- **Robust methods, and ask which** before implementing. Report **both** Pearson r and
  Spearman rho (`robust-fits-aos`). `run_aberration_pairs.py` uses a quartile-of-primary
  ordinary least-squares slope, which predates that preference.
- Every number needs its quantity name and units, or an explicit "dimensionless" with
  numerator and denominator named. `chi2` always as `chi2/dof` with dof stated.
  Correlations need the statistic, both variables with units, and `n`.
- The review backlog flags real issues here that may still be live: significance computed
  with a global complete-case `n` rather than per-pair `n`; an `r` clamp near ±1 that
  manufactures huge significances on the `_optcorr` run; and no detrending in the thermal
  correlations, where both temperature and AOS state drift through the night. See
  [`../status/code_review_findings.md`](../status/code_review_findings.md) — verify
  against current source, the line anchors are stale.

## Running

```bash
cd ~/notebooks/rubin-work/aos
./run_snake.sh --until dz_correlations
./run_snake.sh -n            # check what is stale first
```

`run_vmode_correlations.py` and `run_dz_explained.py` build the OFC SVD, so they need
`lsst.ts.ofc` — RSP only.

`run_dz14_truss.py` is not a pipeline rule; it is run directly and needs only the two
parquet files, plus `lsst.ts.ofc` for the v-mode projection:

```bash
cd ~/notebooks/rubin-work/aos
python code/correlations/run_dz14_truss.py \
  --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
```

The nights on the trace pages are drawn at random from those with at least 10 visits and
10 truss-temperature readings — a requirement on coverage, not on any value of DZ(1,4), so
the figure is not selected on the result. `--seed` fixes the draw and `--n-days` sets how
many nights to show, 6 panels per page.

## Notebooks

None. The first five analyses are Snakemake rules; `run_dz14_truss.py` is run directly.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes come from
- [`telemetry.md`](../telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
