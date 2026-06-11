# AOS — Active Optics System

Analysis of the Rubin Active Optics System: sensitivity matrix decomposition,
nightly data extraction, PID control simulation, and reference wavefront studies.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `nightly_tablemaker.ipynb` | Extract per-exposure AOS data (EFD + ConsDB + Butler) into a single parquet table with vmodes, per-corner Zernikes, and summary arrays. Based on `nightly_report_ts_version`. | 2026-03-07 | 2026-03-13 |
| `aos_nightly_plots.ipynb` | Plots of AOS FWHM and Zernike deviations from multiple nights. Loads nightly_tablemaker output, computes mean 4-corner Zernike deviations, and produces time-series and histogram plots of vmodes and DOF states. | 2026-03-14 | 2026-03-14 |
| `aos_openloop.ipynb` | Reconstruct closed-loop PID behavior of the AOS system. Builds sensitivity matrix SVD, projects 4-corner Zernikes onto vmodes, runs per-vmode PID simulation, and validates against actual DOF corrections. | 2026-03-11 | 2026-03-13 |
| `smatrix_vmode_info.ipynb` | Comprehensive analysis of the AOS sensitivity matrix: SVD (`StateEstimator` + custom-SVD validation), v-mode composition, wavefront signatures, control equations, noise/gain, plus a normalization-scheme deep-dive (unit-invariance) and double-Zernike field patterns & physical impact. Consolidates the former `normalization_study` and `smatrix_doublez`. Shared primitives in `code/ofc_svd.py`. | 2026-03-08 | 2026-06-11 |
| `intrinsics_mktable.ipynb` | Create a table of Zernike wavefront measurements from Full Array Mode (FAM) cwfs images with model intrinsic values. Queries ConsDB for FAM visits, extracts Zernikes via Butler, interpolates Batoid intrinsic model, saves to parquet. | 2026-02-23 | 2026-03-13 |
| `intrinsics_plots.ipynb` | Analyze the FAM Zernike table from `intrinsics_mktable`. Plots data vs model comparisons (trio plots) for each Zernike term across the focal plane. | 2026-02-23 | 2026-02-23 |
| `study_compare_donuts.ipynb` | Compare per-donut wavefront Zernikes between two processing runs (param_set A vs B — code version, binning, or algorithm). Per-CCD positional donut matching, then coverage maps, per-visit large-\|Δ\|, density (hist or hexbin) over the full focal plane and an edge annulus, difference histograms, focal-plane Δ maps (OCS+CCS), and an optional per-visit double-Zernike-fit comparison. Consolidates the former `study_danish_v0p6_vs_v1`, `study_binning`, and `donutalgo_comparison`. Shared code in `code/compare_donuts.py`. | 2026-06-11 | 2026-06-11 |
| `build_measured_intrinsic.ipynb` | Build the empirical focal-plane intrinsic Zernike map from FAM donuts via two DZ-removal methods (Path A U-mode constrained, Path B reachability-thresholded), iterated. Includes OFC SVD/reachability reference, per-path validation, FWHM-equivalent diagnostics, and DOF recovery. Params cell is papermill-tagged. | 2026-04-28 | 2026-05-14 |
| `build_measured_intrinsic_batch.ipynb` | Papermill driver that runs `build_measured_intrinsic.ipynb` once per parameter set (scan over elevation/rotator ranges, filter bands, n_keep). Each run writes to its own self-describing output_dir; executed notebooks + a run manifest land in `output/build_measured_intrinsic/_runs/`. | 2026-05-14 | 2026-05-14 |
| `study_wfs_mimic.ipynb` | Study whether the 4 corner WFS can reconstruct the optical state from FAM observations. Mimics WFS measurements by averaging FAM donuts in annular wedges at the WFS field radius, subtracts measured intrinsic, builds WFS-specific SVD of the sensitivity matrix, recovers DOFs, and compares against full FAM DOF analysis. | 2026-06-04 | 2026-06-04 |
| `intrinsic_camera_telescope_split.ipynb` | Decompose the measured intrinsic Zernike maps into a telescope-fixed (OCS) component and a camera-fixed (CCS) component that rotates with the rotator, for all Noll Zernikes in use (4–19, 22–26). Spin-aware: astig/coma/trefoil/… doublets are combined as Z_cos+i·Z_sin and decomposed with the spin model (camera coefficients allowed to rotate as spin |m|); hole-aware least-squares keeps O hole-free. Per-Zernike page-1 maps with data/model/O/C/residual RMS, plus a telescope-vs-camera RMS summary. Core in `code/intrinsic_split.py`. | 2026-06-10 | 2026-06-10 |
| `study_bounce.ipynb` | Analyze FAM-triplet *bounce* tests (BLOCK-T720 elevation 40↔70°, BLOCK-T724 rotator 0↔60°). Computes the paired-difference Δ (median of time-ordered within-night comp−ref pairs; err = scaled-MAD/√n_pairs) for Double-Zernike coefficients, OFC v-modes, and physical DOF (50 DoF, n_keep=34). Produces Δ/significance/pass heatmaps over (k, j), DZ/v-mode/DOF vs-ordinal-image plots (per bounce; DOF pages also show FAM DOF + the AOS closed-loop Trim = EFD MTAOS aggregatedDoF), night-vs-night cross-scatter, and a 5-panel DOF night-A-vs-night-B scatter. | 2026-05-06 | 2026-06-09 |

## Typical workflow

1. Run `nightly_tablemaker.ipynb` to extract a night's AOS data into a parquet file
2. Run `aos_nightly_plots.ipynb` to visualize multi-night AOS performance trends
3. Run `aos_openloop.ipynb` to simulate PID control and validate against actual corrections
4. Use `smatrix_vmode_info.ipynb` to understand the sensitivity matrix (SVD, normalization, double-Zernike)
5. Use `intrinsics_mktable.ipynb` + `intrinsics_plots.ipynb` for reference wavefront studies
6. Use `study_compare_donuts.ipynb` to compare two processing runs (version/binning/algorithm)

## Data dependencies

- **nightly_tablemaker**: Requires EFD, ConsDB, and Butler access (run on RSP)
- **aos_nightly_plots**: Requires parquet output from nightly_tablemaker
- **aos_openloop**: Requires parquet output from nightly_tablemaker + `ts_config_mttcs` OFC config
- **smatrix_vmode_info**: Requires `lsst.ts.ofc` + `lsst.ts.wep` and `$TS_CONFIG_MTTCS_DIR` OFC config (run on RSP)
- **intrinsics_mktable**: Requires Butler FAM collections + ConsDB (run on RSP)
- **intrinsics_plots**: Requires parquet output from intrinsics_mktable
- **study_compare_donuts**: Requires two runs' `output/<param_set>/{donuts,visits}.parquet` (and `fits.parquet` for the optional DZ-fit comparison); numpy/scipy/pyarrow only (no LSST stack)
- **study_wfs_mimic**: Requires output from `build_measured_intrinsic` (dz_fits + grid parquets), donut parquets, `lsst.ts.ofc`, and CCD height map (run on RSP)
- **intrinsic_camera_telescope_split**: Requires `build_measured_intrinsic` grid parquets at several rotator angles (e.g. `output/build_measured_intrinsic/<set>/nkeep_34_OCS_*_rot_*`) and a FAM `*_fits.parquet` for the actual rotator angles; numpy/scipy only (no LSST stack)
- **study_bounce**: Requires a `*_fits.parquet` (per-visit DZ fits); the v-mode / DOF sections additionally need `lsst.ts.ofc` + `$TS_CONFIG_MTTCS_DIR` (run on RSP; DZ-only analysis runs without them)
