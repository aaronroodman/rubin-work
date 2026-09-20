# Provenance — two-epochs note

- **Date:** 2026-06-20
- **Author:** Aaron Roodman
- **Data:** AOS FAM, `param_set = fam_danish_1_0_wep17_3_0_bin2x`
  (Butler `LSSTCam/runs/aos/fam/danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2/paired`,
  wep 17.3.0, donut_viz 4.2.0, bin ×2), `mi_name = pathA_50_34_i` (n_dof 50, n_keep 34).
- **Epochs:** Family A = day_obs {20260315, 16, 17, 24} (rot 0/±15/±60), 192 visits;
  Family B = day_obs 20260409 (rot ±30/±45), 82 visits.
- **Figure sources:**
  - `rotmaps_Z{5,7}_*.png` — per-rotator-bin `intrinsic_grid.parquet`
    (`aos/output/<ps>/<mi>/build/rot_*/`).
  - `two_epochs_intrinsic.png` — WFS-radius azimuth cut from the raw DZ fit
    (`aos/output/<ps>/fits.parquet`) + `z_gradient` from the MI-refit fit
    (`aos/output/<ps>/<mi>/fits.parquet`).
  - `two_epochs_fieldmaps.png` — reconstructed low-order DZ field (per-epoch median).
- **Figure-source paths** are the layout in use on 2026-06-20. `aos/output/` was
  reorganized on 2026-09-20 to `output/<study>/<data>/`, so the equivalents are now
  `aos/output/miw/<P>_<M>/build/rot_*/`, `aos/output/fam_processing/<P>/fits.parquet` and
  `aos/output/miw/<P>_<M>/fits.parquet`. This `param_set` was retired before that move and
  its products are under `aos/output/archive/`, which keeps the original layout.
- **Regenerate:** `make_figures.py` (reads the above; writes `figures/`).
- **Stack / env:** USDF `lsst-scipipe-13.0.0` (numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.8).
- **Code commit:** record the `rubin-work` git SHA when the build was produced
  (the A4 half-open-rotator-bin fix is in `ed13d35`; re-state after the in-flight
  rebuild job 29357754 completes, in case the grids are regenerated).
- **Caveat:** maps are the low-order (k=1..6) double-Zernike field model the pipeline
  fits; elevation is matched (~70°) between epochs, so the driver is thermal
  (`z_gradient`), not gravity flexure.
