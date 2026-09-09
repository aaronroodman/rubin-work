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
| `run_dz14_truss.py` | the focal-plane-uniform defocus DZ(k=1, j=4) against Telescope Mount Assembly (TMA) truss temperature and against v-mode 1 reconstructed from the commanded degrees of freedom, split into within-FAM-sequence and between-sequence variation |

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

## Outputs

`<mi>/plots/dz_correlations{,_optcorr}.{pdf,_pairs.parquet}`,
`vmode_correlations_{50_34,22_12}.pdf` + summary parquets,
`thermal_correlations.pdf` + `_summary.parquet`, `dz_explained.{pdf,parquet}`,
`<ps>/correlations/aberration_pairs.{pdf,_summary.parquet}`, and
`<ps>/correlations/dz14_truss_<dz_prefix>.{pdf,_summary.parquet}`.

The first four currently share `<mi>/plots/` with the bounce and coadd output; splitting
them per study is outstanding work.

## DZ(k=1, j=4) and truss temperature

On the `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` param_set, 2465 quality-passing FAM
visits of which 1591 carry a truss temperature: truss temperature accounts for almost
none of the uniform defocus. Huber slope −0.0330 ± 0.0052 µm of wavefront per °C, Pearson
r = −0.124, Spearman rho = −0.108, n = 1591, residual nMAD 0.444 µm of wavefront. Across
the full 13 °C range spanned by the data that trend gives only 0.43 µm of wavefront,
against a total DZ(1,4) scatter of 0.490 µm of wavefront (nMAD, n = 2465).

The within-sequence and between-sequence split separates the two effects. Over the 156
FAM sequences with at least 5 visits, the median within-sequence scatter is 0.112 µm of
wavefront in DZ(1,4) against 0.026 °C in truss temperature; the between-sequence scatter
(nMAD of per-sequence medians) is 0.454 µm of wavefront against 2.43 °C. At the pooled
slope, the 0.026 °C of within-sequence temperature motion predicts 0.0009 µm of
wavefront — some two orders of magnitude below the 0.112 µm actually seen. The commanded
v-mode 1 is essentially frozen within a sequence as well (3 × 10⁻⁵ dimensionless), so the
swing is neither thermal nor commanded.

Between sequences, the commanded v-mode 1 does track truss temperature (Huber slope
+0.181 ± 0.004 dimensionless per °C, Pearson r = +0.588, Spearman rho = +0.683,
n = 1591), as expected from a hexapod LUT that is a function of elevation and
temperature. The measured DZ(1,4) does not follow it (Pearson r = −0.107, Spearman
rho = −0.097, n = 2419 against v1 from LUT + Trim).

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
parquet files:

```bash
cd ~/notebooks/rubin-work/aos
python code/correlations/run_dz14_truss.py \
  --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
```

## Notebooks

None. The first five analyses are Snakemake rules; `run_dz14_truss.py` is run directly.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes come from
- [`telemetry.md`](../telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
