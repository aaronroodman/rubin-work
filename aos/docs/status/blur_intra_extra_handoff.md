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
donut rows, 66 columns, 966 row groups, 13.3 GB). Plots at
`aos/output/fam_processing/blur_intra_extra/`.

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

## Not done / next concrete action

1. **Explain the quadrature term.** 0.749 arcsec added in quadrature on the intra side
   only is large and its cause is not established here. Candidates not yet tested: a
   genuine optical difference between the two defocal positions, a difference in the
   defocal offset magnitude between sides, or something in how blitz computes
   `group_fwhm` per exposure.
2. **Check the defocal offsets per side.** `blitz_reader.focus_side` already reads the
   varying element of `defocal_offsets` from the table metadata and validates its sign
   against `visit_id`. Whether the two sides have equal *magnitude* is not checked, and an
   asymmetry there would produce exactly this signature. This is the cheapest next test
   and needs no new Butler pass.
3. **Per-side Zernikes are now available and unexamined.** The 12 per-side Zernike columns
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
