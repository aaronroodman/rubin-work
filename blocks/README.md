# Blocks — Observing Blocks

Tools for identifying and tabulating Rubin Observatory observing blocks and test programs, and for trending image-quality metrics across them.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `locate_test_blocks.ipynb` | Query ConsDB Visit1 for all observations in a date range, map BLOCK test case IDs to human-readable names via the Zephyr Scale API, and tabulate band counts by day_obs for selected test blocks. | 2026-01-27 | 2026-03-14 |
| `image_quality_trending.ipynb` | Trend delivered image quality (PSF FWHM, ellipticity, moment score, AOS contributions) for LSSTCam survey visits across a date range. Fetches per-CCD PSF moments and per-visit AOS / atmosphere metrics from ConsDB via a chunked, cached helper (`code/consdb_fetch.py`), aggregates to per-visit and per-night summaries, and produces histograms, time-series, and a 7-quantity per-visit corner plot. | 2026-05-04 | 2026-05-04 |
| `m1m3_zgradient_survey.ipynb` | Survey the M1M3 vertical thermal gradient (`z_gradient`) across all `science`/`acq` exposures over a date range. `z_gradient` is EFD-derived (`lsst.ts.m1m3.utils.ThermocoupleAnalysis`), **not** a ConsDB column; computed per exposure night-by-night from the Butler exposure list (via `ts_intrinsic_wavefront` `get_m1m3_data`, tqdm per-day progress), cached to parquet, and plotted as z_gradient vs ordinal day_obs (scatter + nightly median) and a per-night violin. | 2026-07-07 | 2026-07-07 |

## Data dependencies

- **locate_test_blocks**: Requires ConsDB access and Zephyr Scale API credentials (run on RSP)
- **image_quality_trending**: Requires ConsDB access (run on RSP). Uses `blocks/code/consdb_fetch.py` for chunked queries with per-chunk parquet caching.
- **m1m3_zgradient_survey**: Requires the Butler (exposure records) and the EFD (M1M3 thermocouples via `lsst.ts.m1m3.utils` + `ts_intrinsic_wavefront.intrinsics_lib.get_m1m3_data`) — run on RSP. Caches to `z_gradient_science_acq.parquet`.
