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
| `run_dz14_truss.py` | the focal-plane-uniform defocus DZ(k=1, j=4) against Telescope Mount Assembly (TMA) truss temperature, and against v-mode 1 reconstructed from the commanded degrees of freedom in three cumulative forms (LUT, LUT+Trim, LUT+Trim+Deviation); includes a DZ(1,4) time history, a hexapod-LUT validation page and per-night traces of DZ(1,4) overlaid with truss temperature |

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
`<ps>/correlations/dz14_truss_<dz_prefix>.{pdf,_summary.parquet}` (20 pages: four
one-per-page truss scatters, LUT validation, v1 LUT vs v1 Trim coloured by time, the
pooled and split Trim populations, the DZ(1,4) time history, nine pages of per-night
traces, then the conversion constants and page 1 of the LTS-213 drawing).

Every DZ(1,4) panel carries a second y-axis giving the equivalent hexapod dz in µm. The
conversion is taken from the singular-value decomposition rather than fitted: v-mode 1 is
99.98% DZ(k=1, j=4) by the first entry of `U_eff` (−0.999919, dimensionless), and the
camera- and M2-hexapod dz coefficients of v-mode 1 agree to 2.1%, so their mean
9.0095 × 10⁻⁴ per µm gives **−1110.03 µm of hexapod dz per µm of wavefront of DZ(1,4)**.

The first four currently share `<mi>/plots/` with the bounce and coadd output; splitting
them per study is outstanding work.

## DZ(k=1, j=4) and truss temperature

On the `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` param_set, 2465 quality-passing FAM
visits of which 1591 carry a truss temperature: truss temperature accounts for almost
none of the uniform defocus. Huber slope −0.0330 ± 0.0052 µm of wavefront per °C, Pearson
r = −0.124, Spearman rho = −0.108, n = 1591, residual nMAD 0.444 µm of wavefront. Across
the full 13 °C range spanned by the data that trend gives only 0.43 µm of wavefront,
against a total DZ(1,4) scatter of 0.490 µm of wavefront (nMAD, n = 2465).

The per-night traces make the same point without any averaging. On day_obs 20251026,
DZ(1,4) spans 3.290 µm of wavefront while the truss temperature moves 0.446 °C, across only
two LUT plateaux; at the pooled slope that temperature motion predicts 0.015 µm of
wavefront, some 220× smaller than observed. day_obs 20251215 spans the same 3.290 µm of
wavefront against 0.811 °C, and day_obs 20260423 spans 1.438 µm of wavefront against
0.163 °C inside a single LUT plateau. Whatever drives the intra-night focus excursions,
truss temperature is not it.

The trace pages show every night carrying at least one contiguous block of 12 Full Array
Mode (FAM) triplets, using the same block definition as the coadd study — a new block
starts whenever `day_obs` changes or `seq_num` advances by other than the triplet spacing of
3. That is a criterion on the observing pattern, not on any value of DZ(1,4), so the figure
is not selected on the result. 49 of 97 nights qualify, of which 34 also carry a truss
temperature; the remaining 15 are shown with an empty truss trace rather than dropped.

The commanded v-mode 1 does track truss temperature, as expected from a hexapod LUT that
is a function of elevation and temperature, and adding successive terms barely changes
that (n = 1591 throughout):

| v-mode 1 form | Huber slope [dimensionless per °C] | Pearson r | Spearman rho |
|---|---|---|---|
| LUT alone | +0.1212 ± 0.0046 | +0.413 | +0.556 |
| LUT + Trim | +0.1811 ± 0.0043 | +0.588 | +0.683 |
| LUT + Trim + Deviation | +0.1829 ± 0.0043 | +0.589 | +0.680 |

## Thermal interpretation of the commanded focus

The commanded focus tracking is quantitatively consistent with a thermally expanding steel
truss, which the final page of the PDF sets out as a chain of three conversions:

| quantity | value | source |
|---|---|---|
| d(v1) / d(truss T) | +0.09634 per °C (dimensionless v-mode amplitude per °C) | Huber slope of the lower Trim population, n = 1367 |
| v1 per hexapod dz | 9.0095 × 10⁻⁴ per µm | mean of the camera- and M2-hexapod dz coefficients of v-mode 1 |
| DZ(1,4) to hexapod dz | −1110.0 µm per µm of wavefront | `U_eff[(1,4),0]` = −0.999919 |

Dividing the first by the second gives **106.9 µm of hexapod dz per °C**, equivalently
0.0963 µm of wavefront of DZ(k=1, j=4) per °C. A steel truss of the LTS-213 length,
7835 mm from the elevation axis to the top of the lower top-end right light baffle, expands
94 µm per °C at a coefficient of thermal expansion of 12 ppm per °C. The ratio is 1.14
(dimensionless, measured over predicted) — agreement to 14%. Attributing the excess to
geometry alone would need an effective length of 8911 mm; attributing it to material alone
would need 13.6 ppm per °C at the LTS-213 length.

Note that this is the *commanded* sensitivity, from the LUT and Trim. It does not contradict
the measured DZ(1,4) being nearly uncorrelated with truss temperature: the hexapod is being
driven as though the truss were expanding thermally, and the residual defocus that survives
that correction is what the earlier sections show has no truss-temperature dependence.

Page 1 of the LTS-213 assembly drawing is appended to show where the length is measured, and
is available at <https://docushare.lsst.org/docushare/dsweb/Get/LTS-213>. It is not kept in
this repository; `--lts213-pdf` points at a local copy, defaulting to
`~/Documents/LSST/LTS-213.pdf`, and the page is skipped if the file is absent. Its dimension callouts render as mojibake under
both MuPDF and Ghostscript because the drawing embeds Identity-H Arial subsets whose font
programs neither engine can parse; the geometry and the notes block are unaffected.

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

The hexapod LUT is a two-dimensional function of elevation and camera rotator angle, so
neither one-dimensional projection is expected to be a single curve, and structure against
rotator angle is not a defect. v-mode 1 from the LUT against elevation traces several
distinct parallel branches, each a different LUT version, and the time history places the
last LUT change on or about day_obs 20251209. The map itself is therefore drawn as the mean
v1 LUT over 5 deg × 10 deg bins in (elevation, camera rotator angle), restricted to
day_obs ≥ 20251209 so that a single LUT version is in force: 1267 visits, 56 of 180 bins
occupied, spanning elevation 22.9 to 75.0 deg and rotator angle −70.1 to +60.2 deg.

The v1 LUT versus v1 Trim scatter separates into two clusters that the time colouring
identifies as epochs, not distinct physical states: the lower-right cluster is the first
~100 days of the sample and the upper-left cluster days 200 to 350.

The DZ(1,4) time-history page is information only, with no fit. It marks the 2025-12-09 LUT
change, after which the uniform defocus is no quieter: median per-night nMAD 0.1670 µm of
wavefront over the 65 nights before against 0.2258 µm of wavefront over the 32 nights on
and after, with pooled nMAD 0.5272 (n = 1198) and 0.4732 (n = 1267). The intra-night
excursions above are therefore not an artefact of an unsettled LUT version.

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

`--min-contig-triplets` sets how long a contiguous FAM block a night must contain to appear
on the trace pages, which are laid out 6 panels per page. `--lts213-pdf` points at the
LTS-213 drawing for the final page.

## Notebooks

None. The first five analyses are Snakemake rules; `run_dz14_truss.py` is run directly.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes come from
- [`telemetry.md`](../telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
