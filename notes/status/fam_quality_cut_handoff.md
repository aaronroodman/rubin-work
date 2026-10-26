# FAM visit quality cut: regularized, refit, current

> **Status:** d12 chain rebuilt and current; Danish 1.3 MIW builds not rebuilt · **Last updated:** 2026-10-25 · **Kind:** handoff

## Done

- `rubinwork/products/fam_tables/quality.py` defines `visit_quality_pass`: `n_donuts` > 500,
  `n_detectors_with_min_donuts` > 100, `median_blur_arcsec` < 1.8 arcsec. `run_blitz_mktable`
  and the Snakefile use it (`2573e4c`).
- `builders/refit_build.py` made build `20261025` of `danish_1_2`, `danish_1_3`,
  `danish_1_3_test` and `danish_1_3_v1000` from their current builds, reusing the donut
  tables (Slurm job 41904532). Set current 2026-10-25.
- `qa/evaluate_quality_cut.py` reports the cut and DZ outliers (`15aad3d`).

## Decisions (Aaron, 2026-10-25)

- Drop the |DZ| < 2.0 µm cut and the fitter's `bad_fit` flag from the consumers (MIW build,
  LUT, coadds, correlations, `fam_dz` `good_only`).
- Replace it with a fixed-level flag: a fit is bad if more than 10 coefficients (count) have
  |DZ| > 2.0 µm. Not an nMAD test: that depends on the ensemble.

- `bad_fit` is now "more than 5 coefficients with |DZ| > 2.0 µm, or n_donuts < 200"
  (`ts_intrinsic_wavefront` `0f0a2ad` on `tickets/RSO-809`: `flag_bad_fits`,
  `_flag_bad_visits`, `run_dz_fit --max-large-coeffs`, MIW build). Builds `20261025b` keep
  the `20261025` fits and recompute only `bad_fit` (`refit_build --reflag`); they flag
  20250707 seq 507 and 510 in `danish_1_2`, nothing in the 1.3 variants. Set current.
- `max_coeff_um` set to null in `aos/analysis_config.yaml` and the correlations defaults
  (`3def770`).
- `fam_dz` rewritten from `danish_1_2@20261025b` (3268 rows, 2 `bad_fit`); snapshot
  `value_added/v1/snapshots/20261025` is current.
- `aos/run_d12_downstream.sbatch` (job 41942117, 50 min) built `20261025` of miw, dof_lut
  and bounce_tables for `d12-A_50_34_i` and `_5rot`, and coadds `d12-A_50_34_i_5rot`. All
  set current. MIW split maps differ from `20261014` by at most 0.001 µm (from the rotator
  bin mean angles); the LUT uses 970 visits instead of 969, largest change M2 dz 5 µm,
  0.013 of its MAD scatter. Coadds and bounce tables have the same rows.

## Next

1. Danish 1.3 MIW builds are still old-tree builds from `fam_tables` `20261013`; rebuild
   if wanted (batch, ask first).
2. Rerun downstream with the prior cuts otherwise unchanged: MIW, LUT, coadds, value_added
   `fam_dz`, bounce. Batch submits need Aaron's OK. The MIW split reads the FAM
   `fits.parquet` for each rotator bin's mean angle, so the larger visit set shifts those
   means unless pinned.
3. Scratch dirs to remove, with Aaron's OK: `/sdf/data/rubin/user/roodman/LSST/scratch_refit_test`,
   `/tmp/qcut`.

## Tried and rejected

- |DZ| < 2.0 µm as a general cut: it mostly removes deliberate perturbation sequences
  (T503, T539, T559 ± pairs), not broken fits; only 1 T614 visit per variant.
- Per-visit 5-nMAD outlier count: rejected by Aaron (ensemble-dependent). It also flags
  good fits: 20 of the 26 `danish_1_2` visits with more than 10 such coefficients have no
  |DZ| > 2.0 µm.
- The >10 coefficients at > 2.0 µm flag, k=1..6 fit, catches only 20250707 seq 507 in
  `danish_1_2` (16 coefficients) and nothing in the 1.3 variants. 20250707 seq 510, the
  other broken fit, has 6.
