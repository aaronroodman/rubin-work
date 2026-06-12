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
| `build_measured_intrinsic.ipynb` | Build the empirical focal-plane intrinsic Zernike map from FAM donuts via two DZ-removal methods (Path A U-mode constrained, Path B reachability-thresholded), iterated. Includes OFC SVD/reachability reference, per-path validation, FWHM-equivalent diagnostics, and DOF recovery. **Superseded by the Phase-2 scripts** (`code/run_build_intrinsic.py` + Snakemake); kept as the reference until that path is validated on the RSP. | 2026-04-28 | 2026-05-14 |
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

## Pipeline (Snakemake)

`Snakefile` orchestrates the per-`param_set` processing; configs in
`snake_config.yaml` (date chunks), `mi_config.yaml` (measured-intrinsic build
configs — an input of `build_intrinsic`, so editing it re-triggers the builds),
and `analysis_config.yaml` (LUT / aberration-pair knobs, kept separate so
editing them re-runs only the analyses, never the builds). Launch detached with
`./run_snake.sh` (see its header).

- **Phase 1** — per chunk: `mktable` (Butler → donuts/visits) → `fit`
  (Double-Zernike, tabulated intrinsic) → `combine` → validation `plots`.
  Outputs `output/<param_set>/{donuts,fits,visits}.parquet` + `plots/`.
- **`aberration_pairs`** — per `param_set`, per-donut primary→secondary
  aberration-pair analysis on `donuts.parquet` (e.g. defocus→spherical,
  astig→2nd-astig): quartile-of-primary OLS slope/r + density pages.
  `code/run_aberration_pairs.py` → `plots/aberration_pairs.pdf` +
  `aberration_pairs_summary.parquet`. (Port of `study_aberrationpairs.ipynb`.)
- **Phase 2** — per `(param_set, mi_name)` from `mi_config.yaml`, the *measured
  intrinsic*: `build_intrinsic` (Path-A U-mode-constrained build, once per
  rotator bin; `code/run_build_intrinsic.py`) → `intrinsic_split` (OCS/CCS
  spin decomposition; `run_intrinsic_split.py`) → `intrinsic_sidecar`
  (per-donut `zk_intrinsic` row-aligned to `donuts.parquet`, full spin
  reconstruction + CCD-height Z4; `run_make_intrinsic_sidecar.py`) →
  `refit_mi` (DZ refit using the measured intrinsic; `run_dz_fit.py
  --intrinsic-sidecar`). Outputs under `output/<param_set>/<mi_name>/`.
- **`build_lut`** — per `(param_set, mi_name)`, an averaged-DOF look-up
  table: projects the per-visit DZ fits onto the OFC SVD (settable
  `n_dof`/`n_keep` via the `lut` block in `mi_config.yaml`), recovers DOF per
  visit, and collapses over **all** elevation and rotator angle (median by
  default). `code/run_build_lut.py` → `output/<param_set>/<mi_name>/lut/`
  `lut.parquet` (per-DOF) + `lut_dz.parquet` (per-(k,j) raw/fit/residual DZ).
  (Supersedes the former `study_50dofLUT.ipynb`, now removed.)
- **MI-refit analyses** — per `(param_set, mi_name)`, run on the *residual*
  (MI-subtracted) DZ in `output/<ps>/<mi>/fits.parquet`; knobs in
  `analysis_config.yaml`:
  - **`dz_correlations`** (`run_dz_correlations.py`, port of
    `study_doublezernike` §7–§10) — DZ_kj↔DZ_k'j' Pearson heatmap, top-|r|
    pair scatters, astigmatism-symmetry pairs, optional exhaustive
    (k1,j1)×(k2,j2) scan. → `plots/dz_correlations.pdf` + `_pairs.parquet`.
  - **`thermal_correlations`** (`run_thermal_correlations.py`, port of
    `intrinsics_thermal_correlations`) — DZ_kj × temperature-variable Pearson
    heatmap + per-term scatter pages. → `plots/thermal_correlations.pdf` +
    `_summary.parquet`.
  - **`bounce`** (`run_bounce.py` + `bounce_lib.py`, port of `study_bounce`) —
    FAM bounce-test time-ordered paired Δ (DZ / v-mode / DOF) with
    significance/pass heatmaps, vs-ordinal pages, night cross-scatter; optional
    EFD MTAOS Trim overlay (`add_dof_trim`). → `plots/bounce_*.pdf` +
    `bounce_kj_stats.parquet`.

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
