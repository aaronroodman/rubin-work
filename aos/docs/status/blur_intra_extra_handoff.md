# Status: intra vs extra donut blur in the Danish 1.3 blitz FAM ensemble

> **Status:** current · **Last updated:** 2026-10-03 · **Kind:** status (handoff)

## Done and committed

Commit `3f600c9` "Keep per-side blur and Zernikes in the blitz recast":

- `aos/code/fam_processing/blitz_reader.py` — new `SIDE_COLUMNS` constant (18 columns):
  `blur_intra` / `blur_extra` in arcsec, per-side fit cost and optimality, and the
  per-side Zernike vectors in OCS and CCS (total, intrinsic, deviation) in micrometres of
  wavefront. `visit_metrics` gained `median_blur_intra_arcsec` and
  `median_blur_extra_arcsec`. Paired columns keep their names and meaning (the two-side
  mean), so a Danish 1.2 reader is unaffected.
- `aos/code/fam_processing/run_blitz_mktable.py` — the two per-side visit medians appended
  to `VISITS_COLUMNS`.
- `aos/notebooks/fam_processing/blur_intra_extra_offset.ipynb` — the plot and the
  offset-model test, executed with outputs.
- `aos/docs/studies/fam_processing.md` — notebook row plus a paragraph on the per-side
  columns.

Tables rebuilt at `aos/output/fam_processing/danish_1_3_test/` (966 visits, 3,155,734
donut rows, 66 columns, 966 row groups, 13.3 GB).

Output at `aos/output/fam_processing/blur_intra_extra/`: `blur_vs_fam_ordinal.pdf`,
`blur_offset_models.pdf`, `quad_diff_thermal_cumulative.pdf`,
`quad_diff_thermal_final.pdf`, and the cached `thermal_join_per_side.parquet` (966 rows,
the per-exposure thermal telemetry for both sides).

Three PDFs in that directory are **orphaned** by the 2026-10-04 rewrite and no cell
produces them any more: `blur_diff_vs_thermal.pdf`,
`blur_diff_thermal_raw_vs_within_night.pdf` and `blur_diff_vs_joint_thermal_fit.pdf`. They
belong to the de-meaned analysis described under "Thermal dependence" below. The old
`thermal_join.parquet` cache is likewise superseded by the `_per_side` one. Safe to delete,
but not deleted — deletion needs Aaron's say-so.

Second pass the same day added the `day_obs` annotations and per-night medians to the
ordinal plot, and the thermal section below.

## The result

873 FAM triplets passing `visit_quality_pass`, 15 nights, 20260315–20260619:

| quantity | value |
|---|---|
| median intra-focal blur | 1.0526 arcsec |
| median extra-focal blur | 0.6636 arcsec |
| median (intra − extra) | +0.3440 arcsec |
| median (intra / extra) | 1.4999 (dimensionless) |
| fraction with intra > extra | 0.9954 (dimensionless, of 873) |

Three one-parameter models, each fitted by a median, robust residual scatter as nMAD of
the intra-blur residual in arcsec:

| model | parameter | resid nMAD (arcsec) |
|---|---|---|
| constant `b_i = b_e + c` | c = +0.34395 arcsec | 0.20379 |
| fractional `b_i = f·b_e` | f = +1.49989 (dimensionless) | 0.23984 |
| quadrature `b_i² = b_e² + q` | q = +0.56140 arcsec² | 0.17863 |

**Quadrature wins**, and the discriminator is unambiguous: a Huber RLM of (intra − extra)
in arcsec on extra blur in arcsec gives slope **−0.57640 ± 0.03444** (dimensionless),
intercept +0.76206 ± 0.02435 arcsec, n = 873. A constant offset predicts slope 0; a
fractional offset predicts slope +0.49989. The measured slope is negative and ~17σ from
zero, which is the quadrature signature — the difference *shrinks* as blur grows. Binned
medians in the notebook's middle panel show the same fall model-free.

Supporting correlations: (intra − extra) vs extra blur, Pearson r = −0.5061, Spearman
rho = −0.4830, n = 873. Intra vs extra blur, Pearson r = +0.4003, Spearman rho = +0.3934,
n = 873.

Both the full 966-triplet sample and the 873-triplet quality sample give the same answer;
the full sample's slope is −0.46926 ± 0.02759 (dimensionless).

Read physically: the intra-focal side carries an extra blur contribution of
sqrt(0.5614) ≈ 0.749 arcsec added **in quadrature**, not a constant arcsec adder and not
a fixed 1.5× scaling. The ~0.75 arcsec quadrature term is close to the +0.762 arcsec
Huber intercept, as it should be when the quadrature model is the right one.

## Thermal dependence

Added 2026-10-03 in the same notebook, after Aaron noticed on the annotated ordinal plot
that the per-night median difference runs from about +0.14 to +0.58 arcsec — much more
night-to-night structure than within-night scatter.

Thermal telemetry is joined **per FAM triplet**, not per side: `visits.parquet` carries
only the in-focus `visit`, so one thermal value serves both exposures. These quantities
vary on tens of minutes, far longer than the intra-to-extra gap, so the granularity is
right — but **nothing in this section can resolve a difference between the two
exposures.** The join is cached at
`aos/output/fam_processing/blur_intra_extra/thermal_join.parquet` (966 rows). The air
temperatures and truss come from ConsDB transformed EFD (`efd_db.join_consdb`, group
`thermal`); the M1M3 gradients and the quadratic-in-radius term from the local
value-added DuckDB.

The target is the **quadrature residual** — observed intra blur minus what the fitted
quadrature model predicts from the extra side — so the blur dependence established above
is already removed.

**The full-sample correlations are mostly night-level covariance and must not be quoted
alone.** Both the thermal state and the seeing are near-constant within a night and differ
strongly between nights, so a quantity merely cold on the large-difference nights
correlates without being a cause. Every number is therefore given both ways; the
within-night column de-means both variables by `day_obs`, over the 14 nights with at least
5 quality triplets:

| quantity | rho full sample | rho within night |
|---|---|---|
| M1M3 radial gradient (°C/m) | +0.0209 | **−0.3583** |
| truss − M1M3 air (°C) | −0.7126 | **−0.3157** |
| M1M3 y gradient (°C/m) | +0.4746 | **+0.2835** |
| M2 air − M1M3 air (°C) | −0.4681 | **−0.2825** |
| M1M3 quadratic-in-radius (°C) | −0.1409 | −0.2701 |
| camera air − M1M3 air (°C) | −0.3788 | −0.1227 |
| M1M3 thermocouple scatter (°C) | +0.4002 | +0.1191 |
| truss − M2 air (°C) | −0.4678 | +0.0732 |
| mean truss temperature (°C) | +0.4669 | −0.0266 |
| M2 air − outside (°C) | −0.4776 | −0.0153 |
| outside air temperature (°C) | +0.5612 | +0.0090 |
| M1M3 z gradient (°C/m) | +0.5005 | −0.0063 |

All coefficients Spearman rho, dimensionless, n = 794 triplets for the air-temperature
differences and 873 for the M1M3 quantities; within-night n = 793, 14 nights.

The bottom five rows are the lesson: outside air temperature falls from rho = +0.5612 to
+0.0090 and the M1M3 z gradient from +0.5005 to −0.0063. Those are **not** thermal
dependences, they are the same night twice.

The top four survive. A joint Huber fit of the night-de-meaned residual on the top four,
n = 793 triplets over 14 nights:

| quantity | slope |
|---|---|
| M1M3 radial gradient | −4.64592 ± 0.56052 arcsec per °C/m |
| truss − M1M3 air | −0.12287 ± 0.01949 arcsec per °C |
| M1M3 y gradient | +2.78300 ± 0.57705 arcsec per °C/m |
| M2 air − M1M3 air | +0.00352 ± 0.01486 arcsec per °C |

Within-night robust scatter falls from 0.07194 to 0.06239 arcsec, a reduction of 0.1327
(dimensionless, fraction of robust scatter in arcsec — **not** of variance). Joint
prediction versus the residual: Pearson r = +0.4592, Spearman rho = +0.4604, n = 793.

`blur_diff_vs_joint_thermal_fit.pdf` plots the observed quadrature residual against this
joint prediction. Left panel, both axes de-meaned by night: the Huber slope of observed
on predicted is +1.0000 ± 0.0661 (dimensionless) by construction — the prediction is a
Huber fit of the same points, so that slope tests nothing and only the scatter about the
line carries information. Right panel adds each night's mean residual back to the
prediction, which is the honest picture of the total: its Pearson r = +0.8944 is
**circular** and must not be quoted as a figure of merit, since the night mean then sits
on both axes. The useful numbers are the three robust scatters in arcsec: 0.18369 raw,
0.07194 after removing the night mean, 0.06239 after also removing the thermal model.
That ordering is the finding — the night level dominates and the thermal term is a small
correction on top of it.

So Aaron's thermal suspicion is **supported but is not the whole story**. The signal is
real — four quantities survive the night de-meaning, two at |rho| > 0.31, and the M1M3
radial gradient has essentially *no* full-sample correlation (+0.0209) yet the strongest
within-night one (−0.3583), which is the opposite of the covariance artefact and hard to
explain any other way. But the within-night part is the smaller part: the night-de-meaned
scatter is 0.07194 arcsec against a per-night median difference that itself ranges over
roughly 0.44 arcsec. **What sets the level night to night is not identified**, and the
`M2 air − M1M3 air` slope is consistent with zero in the joint fit despite rho = −0.2825
alone, so the four are partly redundant.

## Thermal dependence, simplified (2026-10-04)

Aaron asked for the analysis visit by visit, on the quadrature difference **directly**
with nothing subtracted, with the variables ranked one at a time and then added
cumulatively. That replaced the de-meaned section above; the numbers in it are kept for
the record but the section below is the current result.

Target, in arcsec, over 873 quality triplets:

```
q = b_intra**2 - b_extra**2                      [arcsec^2]
quad_diff_arcsec = sign(q) * sqrt(|q|)           [arcsec]
```

Median +0.74927 arcsec, range −0.50257 to +1.39783 arcsec, robust scatter (nMAD) 0.24547
arcsec. 4 of 873 are negative — the intra side sharper — and are kept, which is why the
square root is signed.

**Three data-layer changes made this possible** (see "Infrastructure added" below): the
M1M3 mean glass temperature now exists, `visits.parquet` carries `intra_visit` and
`extra_visit`, and the ESS air temperatures are interpolated within the night, which lifts
coverage from 794 to 871 of 873 triplets.

Ranked one at a time, Huber slope in arcsec per unit, |Spearman rho| descending
(full-sample rho, then rho with each night's mean removed, both dimensionless):

| quantity | slope (arcsec/unit) | rho | rho within night |
|---|---|---|---|
| truss − M1M3 air (°C) | −0.35687 ± 0.01132 | −0.7054 | −0.3134 |
| M1M3 air temperature (°C) | +0.05674 ± 0.00279 | +0.5804 | +0.1675 |
| outside air temperature (°C) | +0.05525 ± 0.00261 | +0.5746 | +0.0709 |
| M1M3 z gradient (°C/m) | +0.74607 ± 0.04048 | +0.5228 | +0.0358 |
| M1M3 y gradient (°C/m) | +11.49413 ± 0.68928 | +0.4905 | +0.2882 |
| M1M3 glass temperature (°C) | +0.04646 ± 0.00292 | +0.4903 | +0.2390 |
| M1M3 glass − M1M3 air (°C) | −0.14659 ± 0.01228 | −0.4849 | +0.0223 |
| M1 glass − M3 glass (°C) | −4.48104 ± 0.31471 | −0.4435 | −0.2998 |
| M2 air − M1M3 air (°C) | −0.32173 ± 0.02128 | −0.4414 | −0.2685 |
| M1M3 radial gradient (°C/m) | −1.44604 ± 0.86339 | +0.0181 | −0.3413 |

n = 871 for the air differences, 873 for the M1M3 quantities; 14 nights contribute to the
within-night column. Full 20-row table in the notebook.

**The final model is four quantities**, chosen by adding in rank order with two guards: a
candidate is skipped if its variance inflation factor against those already in exceeds 10
(dimensionless), or if it fails to cut the residual nMAD by 1% of its current value.

| quantity | slope |
|---|---|
| truss − M1M3 air | −0.26277 ± 0.01350 arcsec per °C |
| M1M3 air temperature | +0.02243 ± 0.00208 arcsec per °C |
| M1M3 z gradient | +0.34369 ± 0.03727 arcsec per °C/m |
| M1M3 quadratic-in-radius | −5.28962 ± 0.29589 arcsec per °C |

Intercept −0.01608 arcsec, n = 871 triplets over 15 nights. Robust scatter falls from
0.24506 to 0.11000 arcsec, a reduction of 0.5511 (dimensionless, fraction of robust
scatter in arcsec, **not** of variance). Prediction versus observed: Pearson r = +0.8110,
Spearman rho = +0.8270.

Three things to carry forward rather than quote the headline alone:

- **About half of that rho is night-level.** With each night's mean removed from both
  sides, Spearman rho = +0.4125 and the residual nMAD moves only 0.10385 → 0.10031
  arcsec. The thermal state explains the night-to-night level well and the within-night
  scatter much less.
- **The M1M3 z gradient is in the model but has essentially no within-night correlation of
  its own** (rho = +0.0358 alone). It earns its place jointly, not individually. Do not
  quote it as an independent driver.
- **The 4 negative triplets are not predicted.** All four sit near +0.5 arcsec in the
  prediction against −0.3 to −0.5 arcsec observed. The model has nothing to say about the
  only visits where the intra side is sharper.

Stability, leave-one-night-out over 15 nights: the highest-leverage night is 20260619, and
dropping it gives Spearman rho = +0.8603, residual nMAD = 0.10086 arcsec. **Every slope
sign is preserved on every holdout**, but the largest fractional slope change is 0.407
(dimensionless, of the all-night slope) — the quadratic-in-radius coefficient goes
−5.28962 → −7.43156 arcsec per °C. Treat the signs as established and the magnitudes as
good to tens of percent, no better.

Plots: `quad_diff_thermal_cumulative.pdf` (one panel per step) and
`quad_diff_thermal_final.pdf`.

### The per-side test, now possible and negative

With `intra_visit` / `extra_visit` in place, each side's telemetry is joined on its own
exposure identifier. The two exposures are adjacent (`extra − intra` = 1 sequence number
on all 966 visits), and **the telemetry is effectively identical between them**: the nMAD
of the per-side difference over the nMAD of the quantity itself is 0.0013 for the mean
truss temperature, 0.0016 for the M1M3 glass temperature and 0.0020 for the M1M3 air
temperature (all dimensionless). The strongest correlation of any per-side difference with
the quadrature difference is Spearman rho = −0.1572 for the mean truss temperature,
n = 871.

**This is a real answer, not a missing measurement.** These quantities drift over tens of
minutes and the two exposures are about a minute apart, so no per-side thermal difference
exists to find. The thermal dependence is of the *size* of the asymmetry on the telescope's
thermal state, not of a difference between the two exposures. Next step 3 below is
therefore resolved and closed.

## Infrastructure added 2026-10-04

- **M1M3 mean glass temperature.** `fit_r2_terms` in `value_added/code/m1m3_thermal_r2.py`
  now also returns `<p>_mean_temp_c`, `<p>_std_temp_c` and `<p>_range_temp_c` [°C] per
  population (`m1m3`, `m1`, `m3`), computed row-wise over the reporting thermocouples. No
  new EFD query — the raw temperatures were already loaded for the shape fit. The nine
  columns are in `efd_db.R2_TABLE_COLS`, the `m1m3_thermal_r2` DDL (now 28 columns) and an
  additive `ALTER TABLE` migration in `create_schema`. Documented in
  `value_added/docs/schema.md`.
- **Backfilled 20260315–20260619 only**, 58,066 rows, all 58,066 with a finite glass
  temperature: median 11.225 °C, range 4.293 to 18.054 °C. The other 153,856 rows of the
  table still have NULL glass columns — **the full backfill has not been run** (see next
  actions).
- **`intra_visit` / `extra_visit` in `visits.parquet`**, 23 columns, 966 rows.
  `blitz_reader.visit_meta_record` already emitted both; they were simply absent from
  `VISITS_COLUMNS` in `run_blitz_mktable.py`, now added. The existing table was patched by
  `aos/code/fam_processing/backfill_visit_sides.py` reading only each visit's blitz
  `.meta`, so `donuts.parquet` was untouched.
- **ESS air temperatures interpolated within the night** by `efd_db.join_consdb`, listed in
  the new `efd_db.INTERPOLATED_AIR_COLS`, each with its own `<col>_interpolated` provenance
  flag. Coverage of the air differences rose from 794 to 871 of 873 triplets.

## Not done / next concrete action

1. **Explain the quadrature term.** 0.749 arcsec added in quadrature on the intra side
   only is large and its cause is not established here. Candidates not yet tested: a
   genuine optical difference between the two defocal positions, a difference in the
   defocal offset magnitude between sides, or something in how blitz computes
   `group_fwhm` per exposure. The thermal section narrows but does not settle this.
2. **Find what sets the per-night level.** Only 14 nights, so a night-level regression has
   little power — but the per-night mean thermal quantities against the per-night mean
   residual is one line of code and was left at a single check (M1M3 radial gradient,
   Spearman rho = +0.1824, p = 0.53, n = 14 nights: nothing). A night-level cause that is
   *not* thermal is as likely at this point.
3. ~~**Per-side telemetry would be a genuine test.**~~ **Done and answered negatively** on
   2026-10-04 — see "The per-side test" above. The two exposures see the same thermal
   state to a few parts in a thousand, so there is no per-side difference to find.
4. **Run the full glass-temperature backfill.** Only 20260315–20260619 is done. The rest
   of `m1m3_thermal_r2` has NULL glass columns, so any other study reading
   `m1m3_mean_temp_c` outside that range gets nothing. This is a batch job and **Aaron
   submits it**; the commands are in the chat reply that accompanied this update. ~450
   nights at roughly 25 s each.
5. **Check the defocal offsets per side.** `blitz_reader.focus_side` already reads the
   varying element of `defocal_offsets` from the table metadata and validates its sign
   against `visit_id`. Whether the two sides have equal *magnitude* is not checked, and an
   asymmetry there would produce exactly this signature. This is the cheapest next test
   and needs no new Butler pass.
6. **Per-side Zernikes are now available and unexamined.** The 12 per-side Zernike columns
   were written but no analysis reads them yet.

## Tried and rejected, and why

- **Do not try to answer this from the old `donuts.parquet`.** The pre-2026-10-03 recast
  stored only the two-side mean of `group_fwhm`, so the per-side blur was unrecoverable.
  That is why the schema changed; the question is not answerable without the rebuild.
- **`group_fwhm` is per-donut, not per-exposure.** Worth stating because the code comment
  at `blitz_reader.py` warns it is a fit-*group* quantity and "not a substitute for `blur`
  in any per-donut analysis". It does vary donut to donut — 3413 distinct values among
  3463 intra donuts on visit 1 — so a median over donuts per side is well defined. The
  warning is about `group_fwhm` not tracking the Danish 1.2 per-donut `blur` star by star
  (Pearson r = 0.208), which is a different claim.
- **One FAM triplet cannot answer this.** The 8-visit probe gave a slope of +0.280 ± 0.451
  (dimensionless) — consistent with all three models. Seeing drift between the intra and
  extra exposures of a triplet is the noise; only the 873-triplet ensemble separates it
  from a systematic offset.
- **This does not resolve the Z11/Z14 intra/extra split** listed as an open question in
  `aos/CLAUDE.md`. That is about Zernike coefficients in micrometres of wavefront in the
  unpaired CWFS; this is about blur FWHM in arcsec on the FAM science CCDs. Do not
  conflate them or claim one explains the other.
- **A full-sample thermal correlation on this ensemble is misleading.** Outside air
  temperature gives Spearman rho = +0.5612 against the quadrature residual and collapses
  to +0.0090 once each night's mean is removed; the M1M3 z gradient goes +0.5005 to
  −0.0063. Both the thermal state and the seeing are near-constant within a night, so any
  slowly varying quantity inherits that night's difference. Always report the within-night
  figure. The notebook's raw-versus-within-night scatter plot
  (`blur_diff_thermal_raw_vs_within_night.pdf`) exists to make this unmissable.
- **The thermal telemetry *was* per triplet; it no longer is.** `visits.parquet` carried
  only the extra-focal `visit` until 2026-10-04, when `intra_visit` and `extra_visit` were
  added. The per-side comparison has since been done and found no difference between the
  two exposures, so the original conclusion — do not read a thermal result as a statement
  about a difference *between* the exposures — still stands, but now because it was
  measured rather than because it could not be.
- **Status line and `truss_temp_mean_c`.** Not a stored column; `efd_db.join_consdb`
  derives it as the mean of `tma_truss_temp_pxpy` and `tma_truss_temp_mxmy` and
  interpolates within the night. 965 of 966 FAM visits have it; the air temperatures are
  870 of 966 because the ESS sensors drop out. Camera-hexapod temperatures
  (`cam_hex_temp_0..7`) are present in `TEMP_COLS` but **entirely NaN** on these nights —
  do not plan an analysis on them without checking first.
- **M1M3 glass temperature was missing and has been added.** It needed no EFD pass after
  all: `fit_r2_terms` already had the full per-thermocouple table in memory for the shape
  fit, so the mean was free. Worth knowing *why* it was absent — the shape coefficients are
  blind to it by construction, since the design matrix carries a constant column and a
  uniform offset of the whole mirror leaves every gradient and the quadratic term exactly
  unchanged. The glass-minus-air difference is now formed and ranks mid-table (Spearman
  rho = −0.4849 full sample, +0.0223 within night), i.e. it is almost entirely night-level
  covariance and did not enter the final model.
- **Do not rank these quantities on the full sample alone.** It is what Aaron asked for and
  the notebook delivers it, but four of the top ten full-sample correlations essentially
  vanish within night: outside air temperature +0.5746 → +0.0709, the M1M3 z gradient
  +0.5228 → +0.0358, M2 air − outside −0.4883 → −0.0890, M1M3 glass − air −0.4849 →
  +0.0223 (Spearman rho, dimensionless). Both the thermal state and the seeing are
  near-constant within a night, so a quantity merely cold on the large-difference nights
  ranks high without being a cause. Every table in the notebook carries the within-night
  column beside the full-sample one for this reason; read both.
- **An in-sample residual cannot justify adding a variable.** The cumulative build-up needs
  its two guards or it will happily accept everything: `mean truss temperature` has a
  variance inflation factor of **inf** against the model (it is exactly `truss − M1M3 air`
  plus `M1M3 air temperature`, both already in), and 11 further candidates reduce the
  residual nMAD by less than 1% of its value. Without the guards the "model" would be 20
  collinear variables and the reported scatter reduction would be meaningless.
- **The quadrature difference must keep its sign.** 4 of 873 triplets have the intra side
  sharper. Using `sqrt(b_i**2 - b_e**2)` would make them NaN and silently drop the only
  visits that contradict the trend; the signed square root keeps them, and the final model
  visibly fails on all four. That failure is information.

- **Mean-of-medians vs median-of-means.** `median_blur_arcsec` is the median over donuts
  of the two-side *mean*, not the mean of the two per-side medians. The two differ
  slightly; the quality cut uses the former, so it was left alone.
- **Process hygiene, cost a wasted run:** launching the build twice concurrently (two
  `nohup` attempts) left a truncated `donuts.parquet`. The script's `--overwrite` guard
  catches an existing file but nothing guards against two live writers. Check
  `pgrep -f run_blitz_mktable` before launching. Note the wrapper exits 0 even when the
  Python process fails, so the log must be read rather than the exit code trusted.

## Leftovers to clean up

- `aos/output/fam_processing/danish_1_3_test/archive_pre_side_columns/` — the pre-rebuild
  `donuts.parquet` (5.06 GB) and `visits.parquet`, kept deliberately for comparison.
  Delete when no longer wanted.
- `aos/output/fam_processing/_timing_probe/` — 8-visit probe output. Safe to delete.
