# Nightly Image Quality (nightlyiq)

Study image quality **image-by-image for a given night** (`day_obs`), combining
science-image PSF metrics with guider, AOS, and pointing diagnostics. Data is
pulled per image from the ConsDB (`visit1_quicklook`) at USDF.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `guider_iq.ipynb` | Per-image PSF quality vs. guider metrics over a `day_obs` range, split **by filter band** into two multipage PDFs. **Group 1** (guider quality cuts): per-band robust (Theil-Sen) fits of `FIT_VARIABLE` (default `donut_blur`, else `fwhm`) vs. `guider_total_seeing` and vs. the detrended pointing RMS (`sqrt(alt_rms² + az_rms²)`, arcsec), correcting each guider variable to that equivalent; signed quadrature residuals `FWHM² − equiv²` (always vs. science FWHM); 2-D histograms of the relations/equivalents and of each residual vs. `aos_fwhm²`; residual histograms in `N_AOS_QUANTILES` quantiles of `aos_fwhm`. **Group 2** (no guider-star cut, `donut_blur < DONUT_BLUR_MAX`): `FWHM² − donut_blur²` vs. `aos_fwhm²` as 2-D histogram and in aos_fwhm quantiles. Selectable `img_type`; coverage-based axis limits (`AXIS_COVERAGE`, default 0.99). | 2026-07-13 | 2026-07-14 |
| `nightly_tablemaker.ipynb` | Extract per-exposure AOS data (EFD + ConsDB + Butler) into a single parquet table with vmodes, per-corner Zernikes, and summary arrays. Based on `nightly_report_ts_version`. | 2026-03-07 | 2026-03-13 |
| `aos_nightly_plots.ipynb` | Plots of AOS FWHM and Zernike deviations from multiple nights. Loads nightly_tablemaker output, computes mean 4-corner Zernike deviations, and produces time-series and histogram plots of vmodes and DOF states. | 2026-03-14 | 2026-03-14 |
| `aos_openloop.ipynb` | Reconstruct closed-loop PID behavior of the AOS system. Builds sensitivity matrix SVD, projects 4-corner Zernikes onto vmodes, runs per-vmode PID simulation, and validates against actual DOF corrections. | 2026-03-11 | 2026-03-13 |

## Conventions

- ConsDB access uses the **USDF** endpoint
  `https://user:{token}@usdf-rsp.slac.stanford.edu/consdb` with the RSP
  `ACCESS_TOKEN` (auto-set in the USDF notebook environment).
- Science FWHM is derived from `psf_sigma_median` as
  `psf_sigma_median * 2*sqrt(2*ln2) * 0.2` (arcsec); ConsDB has no direct
  `psf_fwhm_median` column.
- Notebooks follow the repo template (`common/notebook_template.ipynb`).
- Outputs go in `output/` (gitignored).

## Data dependencies

- **guider_iq**: ConsDB `cdb_lsstcam` schema (`visit1`, `visit1_quicklook`) — see
  the guider metric definitions in `lsst.summit.utils.guiders`.
- **nightly_tablemaker**: EFD, ConsDB, and Butler access (run on RSP).
- **aos_nightly_plots**: parquet output from nightly_tablemaker.
- **aos_openloop**: parquet output from nightly_tablemaker + `ts_config_mttcs` OFC config.
