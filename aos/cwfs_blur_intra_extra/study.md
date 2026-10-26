---
study: cwfs_blur_intra_extra
topic: aos
status: active
question: Does the intra - extra donut blur difference in the corner wavefront sensors, over the science visits, match the FAM difference, and which thermal telemetry drives it?
answer: ''
uses:
  cwfs_science: d13-blitz_rso1071@20261026
  value_added: v1@20261025b
started: 2026-10-26
---

# `cwfs_blur_intra_extra`

## Question

Does the intra - extra donut blur difference in the corner wavefront sensors, over the science visits, match the FAM difference, and which thermal telemetry drives it?

## Current answer

None yet.

## Log

### 2026-10-26 — started

Study created.

### 2026-10-26 — blur difference against time and thermal telemetry

`code/fetch_thermal.py` caches the thermal telemetry per visit; `code/blur_thermal.py`
writes `output/20261026_blur_thermal/blur_diff_science.pdf` (8 pages).

- Per visit: median over corner rafts of intra (SW1) − extra (SW0) blur, each half's median
  over quality-cut donuts (≥ 3 per half, ≥ 2 rafts). 84,301 visits, median +0.2602 arcsec,
  nMAD 0.2000 arcsec.
- Strongest single quantity: camera body − M1M3 air, Spearman rho −0.657 (dimensionless,
  n 80,654 visits), Huber slope −0.2246 arcsec/°C. Same leader as the FAM study.
- Forward selection stopped at 3 quantities: camera body − M1M3 air, M1M3
  quadratic-in-radius, camera air − M1M3 air. Robust scatter 0.1964 → 0.0871 arcsec
  (n 77,375 visits). Further steps each gained < 0.03 of the nMAD.
- Mean blur, TMA elevation and azimuth, and the corner-mean Z4, Z11, Z22 all have
  |rho| ≤ 0.150 (dimensionless, n 84,301 visits).
- The truss reads about 24 °C on 20251118 and 20251119, about 10 °C above the M1M3 air;
  masked as a sensor fault (`BAD_TRUSS_NIGHTS`).

### 2026-10-26 — wind and airflow

`code/fetch_wind.py` caches per visit the weather tower wind (`cdb_lsstcam.exposure`),
the ESS 301 outside airflow and the sonic anemometers 110 and 123–126 (all in the ConsDB
transformed EFD; 123–126 report from 2026-02-16). The PDF grows to 15 pages.

- Wind speed is the strongest wind quantity: tower speed Spearman rho −0.594, within
  night −0.225 (dimensionless, n 84,301 visits), Huber slope −0.0373 arcsec per m/s. It
  is largely seasonal: against the 3-quantity thermal residual, rho −0.199.
- Pointing into the wind matters little: cos(wind_dir − az) rho +0.068 (n 84,301 visits).
- In the forward selection over thermal and wind, tower wind speed enters at step 4 with
  gain 0.017 of the nMAD (dimensionless); residual nMAD 0.0873 → 0.0858 arcsec.
