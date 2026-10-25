# FAM visit quality cut: regularized, refit, current

> **Status:** builds `20261025` current; consumer cuts not yet changed · **Last updated:** 2026-10-25 · **Kind:** handoff

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

## Next

1. Confirm the flag definition (below), then implement it and change the consumers.
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
