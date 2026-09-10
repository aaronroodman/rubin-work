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

The six analyses above are scripts — five Snakemake rules plus `run_dz14_truss.py`, which is
run directly. Two notebooks sit alongside them in `notebooks/correlations/`.

### `consdb_vs_efd_aos_dof_20260513.ipynb`

Where the Consolidated Database (ConsDB) copy of the AOS degree-of-freedom values is filled,
whether it agrees with the Engineering Facility Database (EFD) where filled, and where it is
empty. Written for colleagues outside this project, so it imports **only released LSST code**
— `lsst.summit.utils`, `lsst_efd_client` — and nothing from `rubin-work`; the client
construction is inlined rather than taken from `common/telemetry_clients.py`. Keeping that
constraint is the point of the notebook, so any edit that adds a repo import defeats it.

One night, `day_obs = 20260513`, chosen for its mix of `acq`, `cwfs` and `science` exposures.
It plots 22 Trim quantities and the 10 hexapod LUT axes against `seq_num`, ConsDB points over
the EFD as-of-`obs_start` step trace, with the exact ConsDB column and EFD topic named in every
panel title, plus an `img_type` reference panel and a coverage table broken down by `img_type`.

Results on that night: the ConsDB Trim (`mt_logevent_aggregated_dof`) reaches 24.1% of 510
science exposures and **0.0% of all 114 `acq` and all 110 `cwfs` exposures**, confirming the
`img_type='science'` gate. The hexapod LUT (`compensation_offset`) is 1.8% science, 4.4% `acq`,
5.5% `cwfs` for the camera hexapod and 0.2% for M2 — sparse everywhere and *not* img_type-gated,
a different failure mode. The EFD as-of lookup resolves 816 of 822 exposures for every type,
from 437 MTAOS events.

It also settles the ConsDB transform rule, which `telemetry.md` previously only inferred. Of
the 123 exposures carrying both values, 113 agree to within 1 × 10⁻⁶ µm and 10 disagree; on
**all 10**, exactly one MTAOS event fell inside the exposure window and the ConsDB value equals
that event, while the EFD as-of value is the state at exposure start. The ConsDB takes a
**within-exposure** event, not the most-recent-before one, which explains both the sparse
coverage and the residual disagreement.

### `corner_z4_vs_temperature_science.ipynb`

The same thermal question as `run_dz14_truss.py`, asked on ordinary **science** exposures
instead of FAM ones: `day_obs` 20260419 to 20260713, bands g/r/i/z. Science exposures carry
corner-wavefront-sensor Zernikes for free, so the sample is much larger and differently
selected than the 2465 FAM visits.

The measured term is the **four-corner mean Z4 (OPD)** from `cdb_lsstcam.ccdvisit1_quicklook`,
via `aos_state.fetch_corner_zernikes_consdb`. It is deliberately not called DZ(1,4): four field
points cannot separate a field-constant defocus from real field tilt, and it is a different
estimator from the 189-detector DZ fit. The batoid intrinsic is not subtracted; the notebook
evaluates it with `lsst.ts.ofc.utils.get_intrinsic_zernikes` and finds it identical across the
four corners to five decimal places (0.0050 µm of wavefront in g, 0.0191 µm in i), so dropping
it moves the zero point and cannot affect any slope, though the 0.014 µm of wavefront g-to-i
difference is a per-band offset absorbed by each per-band intercept.

The total is `v1_total = v1(LUT) + v1(Trim) + v1_equivalent(four-corner mean Z4)`, with LUT and
Trim taken from the **EFD** through `aos_trim.fetch_hexapod_lut_for_visits` and
`fetch_aggregated_dof_for_visits` rather than the ConsDB, whose coverage of these is a third of
science exposures at best. Both v-mode routes are used and cross-checked before any result
depends on them: `aos_state.make_state_estimator` + `vmodes_from_dofs` for DOF → v-mode, and
`ofc_svd.build_ofc_svd(..., n_dof=DOF22)` + `U_eff` for wavefront → v-mode.

That cross-check produced a result worth recording: the canonical DOF route agrees with the
five-coefficient `V1_HEX` / `V1_BEND` shortcut in `run_dz14_truss.py` **exactly on the hexapod
dz axes** (−8.9153 × 10⁻⁴ per µm for the camera hexapod, −9.1035 × 10⁻⁴ per µm for M2), but the
shortcut omits nine smaller coefficients (DOF 3, 4, 8, 9, 10, 11, 13, 15, 16). Those multiply
hexapod tilts in arcsec, which are numerically large, so on a general 22-DOF vector the two
differ by roughly 10% (dimensionless, canonical over shortcut). The notebook uses the canonical
route throughout.

Temperatures come from ConsDB `efd_lsstcam.exposure_efd` using the `aos_consdb_efd.TEMP_COLS`
names. The camera-body environmental sensing system (ESS) at salIndex 1 — `cam_hex_temp_0..7` —
carries **no data on any of the 27671 science visits in this range**, so the camera variable is
the camera *air* temperature from the salIndex-111 ESS, whose channel 0 is the only populated
one; truss and ambient are 89.7% populated. The notebook measures this coverage rather than
assuming it, and labels the axis for whichever source it actually used.

The per-visit pull is cached to
`output/notebooks/correlations/corner_z4_vs_temperature_science_<first>_<last>.parquet`, since
the EFD LUT and Trim queries cost roughly 20 s per night over about 50 nights.

#### Results on science exposures

27671 science visits in g/r/i/z over 52 nights, of which 25430 carry a finite `v1_total`; the
LUT and Trim resolve for all 27671, the four corner Z4 values for 91.0%, and the temperatures
for 89.7%. Pooled over bands:

| v-mode 1 form | Huber slope [dimensionless per °C] | Pearson r | Spearman rho | n | residual nMAD |
|---|---|---|---|---|---|
| LUT + Trim | +0.10524 ± 0.00163 | +0.341 | +0.396 | 24831 | 0.5171 |
| LUT + Trim + measured | +0.09242 ± 0.00186 | +0.289 | +0.321 | 22824 | 0.5660 |
| measured Z4 alone | −0.00150 ± 0.00050 | −0.007 | −0.008 | 22824 | 0.1540 |

Two conclusions, both consistent with the FAM analysis reached by an independent route:

- **The commanded state tracks truss temperature at the same rate.** LUT + Trim gives
  +0.10524 ± 0.00163 per °C here against +0.1811 ± 0.0043 per °C pooled on FAM visits, and the
  per-band g value of +0.0966 ± 0.0018 per °C sits on top of the +0.09634 per °C from the lower
  FAM Trim population. Through the same chain that page 20 of the `run_dz14_truss.py` PDF sets
  out, +0.10524 per °C implies **116.8 µm of hexapod dz per °C**, against 102.6 µm per °C for
  `v1_total` and 94 µm per °C for a 7835 mm steel truss at 12 ppm per °C.
- **The measured wavefront carries no truss-temperature signal.** The four-corner mean Z4 term
  alone has Huber slope −0.00150 ± 0.00050 per °C with Pearson r = −0.007 and Spearman
  rho = −0.008 over n = 22824 — a slope 70× smaller than the commanded one and a correlation
  indistinguishable from zero. This is the science-exposure counterpart of the FAM result that
  truss temperature accounts for almost none of the uniform defocus, and it is reached from
  corner-sensor OPD rather than a 189-detector DZ fit.

The per-band slopes for `v1_total` against truss temperature run +0.1153 ± 0.0022 (g, n = 2394),
+0.1156 ± 0.0017 (r, n = 5224), +0.0913 ± 0.0012 (i, n = 9551) and +0.1017 ± 0.0016 per °C
(z, n = 5655) — consistent to about 20% across bands, with the i band lowest. Ambient and camera
air temperature give the same picture at slightly shallower slopes (`v1_total`: +0.07083 ± 0.00174
and +0.07925 ± 0.00178 per °C respectively), as expected for quantities correlated with the truss
temperature rather than independent of it.

The overall sign with which the measured term enters the sum is a convention, set by the
notebook's `MEASURED_SIGN` parameter rather than fitted. Two sign conventions compose into it,
neither pinned down by anything measured here: the sign of the ConsDB corner OPD Z4 relative to
the batoid/OFC wavefront convention, and the sign of the −1110.03 µm-per-µm conversion. The
truss fit cannot resolve the choice — the two options differ by 0.56% in residual nMAD, because
the measured term (nMAD 0.1535, dimensionless) is only 0.26× the size of LUT + Trim (nMAD
0.5957, ratio dimensionless).

A weaker but physically meaningful diagnostic does favour one. A measured residual should report
the part of the commanded focus that was *not* achieved, so adding it with the correct sign
should reduce the scatter of the total rather than increase it. Sign −1 gives nMAD 0.5939 against
0.5957 for LUT + Trim alone (−0.30% relative, dimensionless); sign +1 gives 0.6065 (+1.82%).
Sign −1 is therefore the notebook's default, which makes the composed map from OPD Z4 to v-mode 1
net positive. The margin is small enough that this is stated as an indication, not a
determination, and flipping `MEASURED_SIGN` and re-running from Section 7 reproduces the
alternative. No conclusion above depends on the choice.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes come from
- [`telemetry.md`](../telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
