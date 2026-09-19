# Study: `correlations` — what does the residual DZ correlate with?

> **Status:** current · **Last updated:** 2026-09-10 · **Kind:** reference (study)

> **Code:** `code/correlations/` · **Notebooks:** `notebooks/correlations/`
> **Output:** `output/<param_set>/<mi_name>/correlations/`, `output/<param_set>/correlations/aberration_pairs.*`, `output/<param_set>/correlations/dz14_truss_*`


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

The Engineering Facility Database (EFD) as the source of record for the AOS degree-of-freedom
Trim and hexapod look-up table (LUT), and where the Consolidated Database (ConsDB) copy of
those quantities is filled. Written for colleagues outside this project, so it imports **only
released LSST code** — `lsst.summit.utils`, `lsst_efd_client` — and nothing from `rubin-work`;
the client construction is inlined rather than taken from `common/telemetry_clients.py`.
Keeping that constraint is the point of the notebook, so any edit that adds a repo import
defeats it.

One night, `day_obs = 20260513`, chosen for its mix of `acq`, `cwfs` and `science` exposures.

**The EFD side.** Three checks establish that the terms are correctly identified before any
ConsDB comparison is made.

`MTHexapod.logevent_compensationMode` is checked first, because `compensationOffset` publishes
a computed LUT value whether or not the hexapod acts on it. Of the 822 exposures, 737 ran with
compensation enabled and 85 with it disabled — and the 85 are entirely `bias` and `dark`
frames, so every `science`, `acq`, `cwfs` and `flat` exposure of the night had the LUT applied.
It is a state-change event (nine events for the camera hexapod over a seven-day window), so it
needs an as-of lookup with a lookback of days, not a per-exposure join.

The identity `compensatedPosition = uncompensatedPosition + compensationOffset` holds
**exactly**: maximum residual 0 µm on x, y, z and 0 deg on u, v, w, on both hexapods, over 261
(M2) and 427 (camera) matched event triples, unchanged when restricted to compensation-enabled
events. So `uncompensatedPosition` is the accumulated Offset (Trim) alone, `compensationOffset`
is the LUT alone, and the LUT has two independent EFD routes. The matching tolerance is the one
trap: a single-stage match with a loose window pairs values across a hexapod move during a slew
and shows spurious residuals of hundreds of µm, which is a property of the pairing rather than
of the identity. The notebook matches in two nearest-in-time stages and reports both a 1 s and a
5 s tolerance.

**The ConsDB gaps.** The Trim (`mt_logevent_aggregated_dof`) reaches 24.1% of 510 science
exposures and **0.0% of all 114 `acq` and all 110 `cwfs` exposures**, confirming the
`img_type='science'` gate. The hexapod LUT (`compensation_offset`) is a few percent on all
three on-sky types for the camera hexapod and under a percent for M2 — sparse everywhere and
*not* img_type-gated, a different failure mode. There is **no ConsDB column for
`compensatedPosition`**, so a physical position can be rebuilt from the ConsDB only where both
term families happen to be populated.

**What the ConsDB columns actually hold**, measured by comparing each pivoted family against
all three `MTHexapod` topics rather than inferred from its name:

- `{camera,m2}_hexapod_aos_corrections_*` is the **accumulated Offset**, closest to
  `uncompensatedPosition` on 6 of 6 translation axes with the runner-up topic wrong by a factor
  of 11 to 264. The name suggests a per-iteration correction; it is not one, and the magnitudes
  agree — thousands of µm on x and y is a standing alignment.
- `{camera,m2}_hexapod_compensation_offset_*` is the **LUT**, closest to `compensationOffset`
  on 6 of 6 axes.

**Where populated, the ConsDB values are correct.** The LUT difference from the EFD as-of value
at `obs_start` (about 10 µm median on the camera hexapod x axis) collapses to 1.2 µm or less
against the best-matching `compensationOffset` event inside the exposure, and to exactly 0 µm
for M2 — a fraction of a percent of the 1959 µm (x), 1730 µm (y) and 4000 µm (z) range the LUT
covers in the night. The ConsDB value is a sample of the same stream at a different instant;
the LUT genuinely moves during an exposure.

The transform rule itself is settled by the Trim, which `telemetry.md` previously only
inferred. Of the 123 exposures carrying both values, 113 agree to within 1 × 10⁻⁶ µm and 10
disagree; on **all 10**, exactly one MTAOS event fell inside the exposure window and the ConsDB
value equals that event, while the EFD as-of value is the state at exposure start. The ConsDB
takes a **within-exposure** event, not the most-recent-before one, which explains both the
sparse coverage and the residual disagreement. For a step-function quantity like the Trim that
is the wrong instant.

Scatter plots of ConsDB against EFD, one panel per axis and coloured by `img_type`, are given
for all three families, so the coverage question and the value-agreement question are answered
separately.

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

Air temperatures come from ConsDB `efd_lsstcam.exposure_efd` using the
`aos_consdb_efd.TEMP_COLS` names: truss and ambient at 89.7% populated, and the camera *air*
temperature from the salIndex-111 environmental sensing system (ESS), whose channel 0 is the only
populated one. The camera **body** temperature is a different quantity and does not come from the
ESS at all — the salIndex-1 ESS (`cam_hex_temp_0..7`) carries no data on any of the 27671 science
visits in this range. It is camera housekeeping published to the camera's own InfluxDB database
`lsst.MTCamera`, topic `lsst.MTCamera.utiltrunk_body`, field `AverageTemp`, and the notebook
fetches it with a second EFD client constructed against that database, following the pattern in
`code/fam_processing/run_attach_telemetry.py`. A reading outside (−50.0, 60.0) °C is a sensor
dropout rather than a temperature. Both the body and the air temperature near the camera are kept
as independent thermal channels.

The per-visit pull is cached to
`output/notebooks/correlations/corner_z4_vs_temperature_science_<first>_<last>_<version>.parquet`,
since the EFD LUT, Trim and camera-body queries cost roughly 20 s per night over about 50 nights.
The version suffix is bumped when a new column is added, which forces a refetch rather than
silently serving a cache that predates it.

One section steps outside that sample to fix the sign of the measured term, reading
`MTAOS.logevent_wavefrontError` (the Optical Feedback Control input, four rows per visit, one per
corner sensor) and `MTAOS.logevent_degreeOfFreedom` (which co-emits `opticalState`, `visitDoF`,
`aggregatedDoF` and the PID gains on one event) over a single initial-alignment sequence. See
[The sign of the measured term](#the-sign-of-the-measured-term).

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

#### The sign of the measured term

The overall sign with which the measured term enters the sum is the notebook's `MEASURED_SIGN`
parameter. Two sign conventions compose into it: the sign of the ConsDB corner OPD Z4 relative to
the batoid/OFC wavefront convention, and the sign of the −1110.03 µm-per-µm conversion.

**The temperature fits cannot settle it.** The two choices differ by 0.56% in residual nMAD about
the truss relation, because the measured term (nMAD 0.1535, dimensionless) is only 0.26× the size
of LUT + Trim (nMAD 0.5957, ratio dimensionless). A weaker diagnostic — that a measured residual
should reduce the scatter of the total rather than increase it — favours −1 (nMAD 0.5939 against
0.5957 for LUT + Trim alone, −0.30% relative; sign +1 gives 0.6065, +1.82%), but not decisively.

**The initial-alignment sequence settles it.** `MEASURED_SIGN = -1`, determined from the AOS
closing the loop in real time rather than from any fit. Each night opens with a BLOCK-T539
`infocus_initial_alignment` sequence that walks the telescope in from a large focus error; on
20260513 that is seq_num 10–19, ten `acq` exposures. Because those errors are several µm of
wavefront, the direction of the response is unambiguous:

| seq_num | four-corner mean Z4 [µm of wavefront] | Tweak, camera hexapod dz [µm] | Tweak, M2 hexapod dz [µm] |
|---|---|---|---|
| 10 | +3.870 | −119.75 | −86.04 |
| 12 | +2.346 | −74.59 | −53.59 |

The commanded hexapod dz opposes the measured Z4. A residual Z4 surviving after the command is
therefore the part of the error not yet removed, and enters the commanded state with the sign
that cancels it. Since the −1110.03 µm-per-µm conversion is itself negative, `MEASURED_SIGN = -1`
makes the composed map from OPD Z4 to v-mode 1 net positive. Over the sequence the four-corner
mean Z4 converges +3.870 → +2.346 → +0.327 → −0.351 → +0.007 µm of wavefront.

Three properties of the block are verified in the same section rather than assumed, and each is
a check that the repo's model of the control chain matches what the AOS actually does:

- **Correction latency.** The AOS emitted a `degreeOfFreedom` event only on seq_num 10, 12, 14,
  16, 18 and none on 11, 13, 15, 17, 19 — image *n*'s correction is applied at *n+2* and
  *n+1*'s is discarded.
- **Trim accumulates the Tweak.** `Trim_n = Trim_{n−1} + Tweak_n` holds exactly (largest residual
  0.0 µm of hexapod dz over both hexapod dz axes and all five events).
- **Proportional gain.** `kpGain` is published per DOF on the same event and reads 0.80 on
  seq_num 10 and 12, then 0.75 on 14, 16 and 18 — this block runs a higher gain than the usual
  0.3 (all dimensionless). `kiGain` and `kdGain` are zero, so the controller is pure proportional
  here.

The sign conclusion is independent of the DOF subset in force, which changes partway through the
sequence between the 10-DOF/5-v-mode truncation and the usual 22-DOF/12-v-mode scheme: either way
the hexapod dz axes carry the focus correction. For that reason no measured-to-commanded gain
line is drawn on the scatter panel — only the quadrant is meaningful. The three small-|Z4| visits
(seq_num 14, 16, 18, all under 0.4 µm of wavefront) do not all fall in the opposing quadrant and
are not expected to: once the focus error approaches the corner-to-corner spread, the correction
is driven by the other Zernikes and DOF in the fit rather than by defocus.

#### Structure in the residual about the truss relation

Subtracting each band's own Huber fit of `v1_total` against mean truss temperature leaves a
residual with nMAD 0.2152 (g), 0.2432 (r), 0.2414 (i) and 0.2670 (z), all dimensionless — 239
to 296 µm of equivalent hexapod dz, pooling to 0.2450 dimensionless or 271.9 µm over n = 22824.
Truss temperature is therefore far from a complete description of the commanded focus state.

The residual core is close to Gaussian but carries a **one-sided positive tail** in every band:
2.38% (g), 2.01% (r), 3.04% (i) and 0.88% (z) of visits lie beyond +3 nMAD against 0.25%,
0.38%, 0.34% and 0.04% beyond −3 nMAD, where a Gaussian would put 0.135% on each side. The
positive/negative ratio runs 5 to 25 (dimensionless) and in the same direction in all four
bands, so it is not symmetric measurement noise on the wavefront term but a population of
visits whose commanded focus sat above the truss relation — the reason the fits are robust
rather than ordinary least squares.

The same fit is also run against the **commanded state alone**, `v1_lut_trim = v1_lut + v1_trim`,
dropping the measured wavefront term. That residual is 15% tighter — pooled nMAD 0.2091 against
0.2450, both dimensionless, a ratio of 0.854 — while the within-night focus excursions remain, so
`v1_meas` contributes more scatter than it removes at this level. It is a four-field-point
estimate of a field-constant quantity, which is the expected reason. Both residuals are stored in
the summary table, so either can be used as the starting point for the time-history work.

#### No thermal channel adds information beyond the truss

The ten-channel `CORE_THERMAL` set is scored against a truss-only baseline with `GroupKFold` on
`day_obs` — grouping is load-bearing, since thermal state is correlated within a night and an
ungrouped split leaks near-duplicate rows into the test fold and inflates every score. Ridge on
all ten channels and histogram gradient-boosted trees are compared to the baseline on identical
rows, scored once on accumulated out-of-fold predictions.

Out-of-sample R² (dimensionless): truss-only +0.3898 (g), +0.3545 (r), +0.1603 (i), +0.3255
(z); ridge +0.4759, +0.3932, +0.1381, +0.3952; gradient boosting +0.2501, +0.2964, −0.0181,
+0.2236.

**The answer is no, on two grounds.** The mean ridge gain of +0.0431 in R² is smaller than the
0.0880 across-band standard deviation of the baseline R² itself, and it is not uniform — the i
band, the largest sample at n = 9397, gets worse by −0.0222. Gradient boosting is worse than
the baseline in every band, inflating the residual nMAD by 4.4% to 10.9% (dimensionless, GB
over truss-only), so there is no exploitable nonlinearity. Permutation importance on the
held-out fold puts `cam_AverageTemp` (mean drop in R² 0.3596) and `tma_truss_temp_pxpy` (0.3498)
far above the other eight channels, all under 0.08 and several negative. The channels are
largely redundant, as expected.

#### The residual does not drift within a night

The residual is plotted against hours since two per-night zero points: (a) zero-degree evening
twilight, the descending crossing of geometric sun altitude 0.0 deg computed from the site
coordinates by a coarse scan plus bisection, and (b) the first `science` or `acq` exposure of
the night, queried separately from ConsDB because the notebook's own visit list is
`img_type='science'` in g/r/i/z only. The first aligns nights by solar time, the second by
operational time; they differ by the observing-start delay, median +0.99 h with a range of
+0.75 to +5.26 h over the 52 nights.

Read through the median across nights in one-hour bins — the per-night lines are too noisy
visit-to-visit to carry a shape — the residual sits flat on zero to within about ±0.05
dimensionless (±55 µm of equivalent hexapod dz) from the start of the night to the end, in every
band and under both alignments, with error bars of the same size as the excursions. The pooled
Huber slopes are small and **inconsistent in sign between bands** (g −0.00680 ± 0.00171, r
+0.00786 ± 0.00106 dimensionless per h under alignment (a), both at |slope/err| above 3), the
nMAD of the per-night slopes (0.042 to 0.084 per h) is 3 to 180× the median per-night slope, and
even the largest pooled slope integrated over a 9-hour night gives 0.07 dimensionless, under
30% of the residual nMAD it would need to explain. Neither zero point does better than the
other.

Twilight validates as a smooth 22.25 h to 21.75 h UTC seasonal walk over April to July (a
0.50 h swing) and precedes the first exposure on all 52 nights. On 8 of those nights the first
exposure fell after 00:00 UTC, so the validation panel plots its hour unwrapped past midnight.

#### Per-visit summary table

The notebook writes
`output/notebooks/correlations/corner_z4_v1_summary_<first>_<last>_<version>.parquet`, one row
per science visit (27671 rows, 36 columns) for follow-up work outside the notebook:

| column | quantity |
|---|---|
| `day_obs`, `seq_num`, `visit_id`, `band`, `exp_midpt_mjd` | visit identification and exposure midpoint MJD (UTC) |
| `elevation_deg`, `azimuth_deg` | telescope pointing, deg, 100% populated |
| `rotator_deg` | camera physical rotator angle, deg, from `cdb_lsstcam.visit1_quicklook.physical_rotator_angle`, 91.93% populated |
| `sky_rotation_deg` | sky position angle, deg — carried alongside the rotator, and a different quantity |
| `v1_lut`, `v1_trim`, `v1_meas` | the three v-mode-1 components, dimensionless; `v1_meas` is stored with its own sign as measured, before `MEASURED_SIGN` is applied |
| `v1_total` | `v1_lut + v1_trim + MEASURED_SIGN * v1_meas` with `MEASURED_SIGN = -1`, dimensionless |
| `v1_lut_trim` | `v1_lut + v1_trim`, the commanded state with no measured term, dimensionless |
| `v1_truss_fit`, `v1_resid_truss` | the per-band Huber prediction of `v1_total` from truss temperature and the residual about it, dimensionless |
| `v1_truss_fit_lutptrim`, `v1_resid_truss_lutptrim` | the same pair fitted to `v1_lut_trim` instead, dimensionless |
| `truss_temp`, `cam_body_temp`, `outside_temp` | mean TMA truss, camera body `AverageTemp`, and outside air temperature, deg C |
| `tma_truss_temp_pxpy`, `tma_truss_temp_mxmy` | the two TMA truss thermocouples individually, deg C |
| `cam_air_temp`, `m2_air_temp`, `m1m3_air_temp` | ESS air temperatures near the camera, M2 and M1M3 (salindex 111/112/113), deg C |
| `m2_delta_t`, `cam_m1m3_delta_t`, `dome_delta_t` | air-temperature differences against `m1m3_air_temp`, deg C |
| `pressure_pa` | ambient air pressure, **Pa** — median about 74300 Pa at 2663 m, not hPa |
| `donut_blur_fwhm`, `aos_fwhm` | ConsDB donut-blur and AOS FWHM contributions, arcsec, 91.85% populated |
| `psf_sigma_median`, `psf_fwhm` | ConsDB median PSF Gaussian sigma in pixels, and the FWHM derived from it in arcsec |
| `delta_mjd_twilight`, `delta_mjd_firstimg` | exposure midpoint minus each per-night zero point, days |

Both fit/residual pairs are rebuilt from the same per-band fits Section 10 subtracts, so
`v1_truss_fit + v1_resid_truss` reproduces `v1_total`, and the `_lutptrim` pair reproduces
`v1_lut_trim`, to floating-point round-off. The parquet is read back and compared against the
in-memory frame in the same cell.

The commanded state alone gives a **15% smaller** residual about the truss relation than the
total does: pooled nMAD 0.2091 against 0.2450 (both dimensionless), a ratio of 0.854, or 232.1
against 271.9 µm of equivalent hexapod dz. Per band the ratio runs 0.779 (g) to 0.870 (i). The
within-night focus excursions survive in both, so the difference is measurement noise carried by
the four-field-point `v1_meas` term rather than signal removed. Its commanded-only truss slope in
g band, +0.09664 ± 0.00182 per °C, also sits close to the +0.09634 per °C that
`run_dz14_truss.py` finds on FAM exposures — an independent check on units and indexing.

`psf_fwhm` is derived, not stored by the ConsDB: `psf_sigma_median` is a Gaussian sigma in
pixels, scaled by 2.3548 (dimensionless) and 0.2 arcsec/pixel. The three image-quality columns
arrive as `object` dtype from the ConsDB and are cast to numeric before use.

The camera physical rotator angle required extending the visit query with a `LEFT JOIN` onto
`visit1_quicklook`, since `visit1` itself carries only `sky_rotation`; `pressure` comes from
`visit1` and the image-quality columns from `visit1_quicklook`. Because all of these come from
that ConsDB query rather than from the slow EFD pull, the cache loader back-fills them into a
cache written before they were added instead of forcing a refetch.

## See also

- [`smatrix_vmode.md`](smatrix_vmode.md) — where the v-modes come from
- [`telemetry.md`](../telemetry.md) — where the temperature columns come from
- [`../miw_pipeline.md`](../miw_pipeline.md#phase-3--analyses-on-the-mi-refit-fits-per-param_set--mi_name)
