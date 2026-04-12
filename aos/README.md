# AOS — Active Optics System

Analysis of the Rubin Active Optics System: sensitivity matrix decomposition,
nightly data extraction, PID control simulation, and reference wavefront studies.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `nightly_tablemaker.ipynb` | Extract per-exposure AOS data (EFD + ConsDB + Butler) into a single parquet table with vmodes, per-corner Zernikes, and summary arrays. Based on `nightly_report_ts_version`. | 2026-03-07 | 2026-03-13 |
| `aos_nightly_plots.ipynb` | Plots of AOS FWHM and Zernike deviations from multiple nights. Loads nightly_tablemaker output, computes mean 4-corner Zernike deviations, and produces time-series and histogram plots of vmodes and DOF states. | 2026-03-14 | 2026-03-14 |
| `aos_openloop.ipynb` | Reconstruct closed-loop PID behavior of the AOS system. Builds sensitivity matrix SVD, projects 4-corner Zernikes onto vmodes, runs per-vmode PID simulation, and validates against actual DOF corrections. | 2026-03-11 | 2026-03-13 |
| `smatrix_vmode_info.ipynb` | Comprehensive analysis of the AOS sensitivity matrix SVD: U/S/V decomposition, v-mode composition in DOF space, wavefront signatures, and control equations. | 2026-03-08 | 2026-03-13 |
| `smatrix_doublez.ipynb` | Double-Zernike decomposition of the sensitivity matrix. Identifies which telescope DOFs produce specific double-Zernike field patterns (e.g. astigmatism varying as astigmatism across the field). | 2026-03-18 | 2026-03-18 |
| `intrinsics_mktable.ipynb` | Create a table of Zernike wavefront measurements from Full Array Mode (FAM) cwfs images with model intrinsic values. Queries ConsDB for FAM visits, extracts Zernikes via Butler, interpolates Batoid intrinsic model, saves to parquet. | 2026-02-23 | 2026-03-13 |
| `intrinsics_plots.ipynb` | Analyze the FAM Zernike table from `intrinsics_mktable`. Plots data vs model comparisons (trio plots) for each Zernike term across the focal plane. | 2026-02-23 | 2026-02-23 |
| `donutalgo_comparison.ipynb` | Compare Zernike wavefront coefficients between two different donut wavefront retrieval algorithms. Matches donuts by position, produces hexbin density plots and difference histograms. | 2026-03-24 | 2026-03-24 |

## Typical workflow

1. Run `nightly_tablemaker.ipynb` to extract a night's AOS data into a parquet file
2. Run `aos_nightly_plots.ipynb` to visualize multi-night AOS performance trends
3. Run `aos_openloop.ipynb` to simulate PID control and validate against actual corrections
4. Use `smatrix_vmode_info.ipynb` and `smatrix_doublez.ipynb` to understand the sensitivity matrix
5. Use `intrinsics_mktable.ipynb` + `intrinsics_plots.ipynb` for reference wavefront studies
6. Use `donutalgo_comparison.ipynb` to compare different wavefront estimation algorithms

## Data dependencies

- **nightly_tablemaker**: Requires EFD, ConsDB, and Butler access (run on RSP)
- **aos_nightly_plots**: Requires parquet output from nightly_tablemaker
- **aos_openloop**: Requires parquet output from nightly_tablemaker + `ts_config_mttcs` OFC config
- **smatrix_vmode_info**: Requires `lsst.ts.ofc` (available on RSP)
- **smatrix_doublez**: Requires `lsst.ts.ofc` (available on RSP)
- **intrinsics_mktable**: Requires Butler FAM collections + ConsDB (run on RSP)
- **intrinsics_plots**: Requires parquet output from intrinsics_mktable
- **donutalgo_comparison**: Requires parquet output from intrinsics_mktable (two algorithm runs)
