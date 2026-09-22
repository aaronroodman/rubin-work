# Provenance — `aos-bounce-test-summary`

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** reference (provenance)

Everything needed to reproduce the numbers in [`note.md`](note.md).

## Software

| item | value |
|---|---|
| `rubin-work` git SHA | `04d2bf5` |
| `ts_intrinsic_wavefront` | `v2.1-6-gf4f23b6`, at `~/u/LSST/packages/ts_intrinsic_wavefront` |
| environment | Aaron's AOS/CWFS environment on the USDF (needs `lsst.ts.ofc`, which is not in `lsst_distrib`) |

## Configuration

| item | value |
|---|---|
| `param_set` | `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` (`dir_name` `danish_1_2`) |
| `mi_name` (MIW leg) | `pathA_50_34_i_5rot` (`dir_name` `A_50_34_i_5rot`), frozen at `day_obs_max: 20260513` |
| bounce definitions | `aos/analysis_config.yaml`, `overrides: fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x: bounce:` — set at **param_set** level so both intrinsic legs resolve the same legs |
| DZ grid | fit prefix `z1toz6`; focal `k = 1…6`; pupil Noll `j = 4…19, 22…26` (21 indices), stated explicitly as `pupil_j_range` because the Phase-1 table carries no `nollIndices` column |
| OFC subspace | `n_dof: 50`, `n_keep: 34`; sensitivity matrix evaluated at camera rotator angle 0.0 deg |
| quality cut | `--min-detectors 160` (CCDs with enough donuts per visit), matching the Snakefile `bounce` rule |
| per-night floor | `night_min_visits: 3` |
| thresholds | `pass_nsigma_threshold: 3.5` (dimensionless) with `pass_delta_threshold_um: 0.1` µm of wavefront, or `pass_sigma_only_threshold: 5.0` (dimensionless) alone |

## Inputs

| table | visits | day_obs range | mtime |
|---|---|---|---|
| `aos/output/fam_processing/danish_1_2/fits.parquet` (batoid intrinsic) | 2528 over 98 nights | 20250415–20260713 | 2026-08-24 |
| `aos/output/miw/danish_1_2_A_50_34_i_5rot/fits.parquet` (MIW refit) | 1126 | 20260315–20260513 | 2026-07-07 |

Butler collection for the FAM processing: Danish 1.2,
`wep_17_6_1` with refit WCS and 2×2 binning, in `/repo/main`.

## Outputs

| directory | intrinsic | coverage |
|---|---|---|
| `aos/output/bounce/danish_1_2_A_50_34_i_5rot/` | measured intrinsic wavefront (MIW) | April/May nights only — lead result for the 40 deg elevation leg and the rotator bounce |
| `aos/output/bounce/danish_1_2_batoid/` | batoid design intrinsic | all six BLOCK-T720 nights including July — the only source for the 60/50/30/75 deg legs |

Each holds `bounce_kj_stats.parquet`, `bounce_fwhm_metric.parquet` and `plots/bounce_*.pdf`.
DOF and v-mode Δ are rendered into the PDFs only; they are not persisted to a parquet, so the
DOF table in the note was recomputed from the fits table with the same `ofc_svd` projection the
script uses.

## How the two runs were produced

Both are **hand-run**, not Snakemake — every input already existed, and no pipeline rule was
re-triggered (the MIW `fits.parquet` kept its 2026-07-07 mtime through both runs). Each was
invoked as `python code/bounce/run_bounce.py` with the fits table above, `--min-detectors 160`,
and an explicit `--out-dir`, since the script otherwise derives its output path from
`--param-set` / `--mi-name` and the two intrinsic legs would collide.

## Caveats carried into the note

1. **July uses the batoid intrinsic.** The MIW build is frozen at `day_obs` 20260513. The
   paired Δ cancels any frame-fixed intrinsic, so this was expected to matter little, and the
   note quotes the measured bound: on the identically-selected 20260418 `Elev=40` leg, the two
   intrinsics give Δ differing by median −0.0001 µm of wavefront, `nmad` 0.0003 µm of
   wavefront, against a 0.0370 µm RMS signal, Pearson r = 1.000, Spearman rho = 0.983, n = 126.
2. **Elevation 75 deg is a +5 deg upward throw**, a near-null control rather than a flexure
   measurement.
3. **20260419 and 20260513 drop out of the batoid per-night breakdown** on the 40 deg leg: the
   Phase-1 table keeps only 2 visits there (at 170–172 CCDs with enough donuts) where the MIW
   refit keeps 12 and 8 respectively (down to 29 CCDs), because the refit runs with the quality
   cut relaxed (`--no-quality-cut`) while the Phase-1 table was cut upstream. Two visits is
   below the 3-visit `night_min_visits` floor. No July visit is affected — all 47 sit at
   176–180 CCDs and pass the quality cut.
4. **Small pair counts on the new legs** — 4 to 6 pairs per July leg. The quoted errors reflect
   this; the monotonic throw trend across legs is stronger evidence than any single leg.

## Code change behind this analysis

`bounce_lib.bounce_nights` previously required **every** configured comparison leg to clear
`night_min_visits` for a night to enter the per-night breakdown. With five legs no night can
satisfy that, so the per-night product would have come back empty. The gate now qualifies a
night on its populated legs (at least one, and every populated one clearing the floor).
Behaviour-preserving for single-leg bounces, and verified so: 882 pre-existing rows are
bit-identical, `max|difference| = 0.000e+00` on `delta`, `delta_err`, `significance`, `n_ref`
and `n_comp`, and the BLOCK-T724 FWHM metric reproduces exactly.
