# Blocks — Observing Blocks

Tools for identifying and tabulating Rubin Observatory observing blocks and test programs, and for trending image-quality metrics across them.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `locate_test_blocks.ipynb` | Query ConsDB Visit1 for all observations in a date range, map BLOCK test case IDs to human-readable names via the Zephyr Scale API, and tabulate band counts by day_obs for selected test blocks. | 2026-01-27 | 2026-03-14 |
| `image_quality_trending.ipynb` | Trend delivered image quality (PSF FWHM, ellipticity, moment score, AOS contributions) for LSSTCam survey visits across a date range. Fetches per-CCD PSF moments and per-visit AOS / atmosphere metrics from ConsDB via a chunked, cached helper (`code/consdb_fetch.py`), aggregates to per-visit and per-night summaries, and produces histograms, time-series, and a 7-quantity per-visit corner plot. | 2026-05-04 | 2026-05-04 |

## Data dependencies

- **locate_test_blocks**: Requires ConsDB access and Zephyr Scale API credentials (run on RSP)
- **image_quality_trending**: Requires ConsDB access (run on RSP). Uses `blocks/code/consdb_fetch.py` for chunked queries with per-chunk parquet caching.
