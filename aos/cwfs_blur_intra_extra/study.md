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

### 2026-10-26 — boosted trees, scored on held-out nights

`code/blur_gbt.py` writes `output/20261026_blur_gbt/blur_gbt.pdf`. sklearn
`HistGradientBoostingRegressor` (absolute-error loss), 5 folds by whole night
(`GroupKFold` on `day_obs`), on the 76,963 visits with every thermal and wind-select input.
Held-out residual nMAD (arcsec): Huber 3-quantity 0.0875; trees on the same 3 inputs
0.0939; trees on all thermal 0.0813; trees on thermal, wind and pointing 0.0759 (0.0775
on all 84,301 visits, missing inputs allowed).

- Raft-scatter floor (nMAD of the visit median from raft-to-raft scatter alone) 0.0328
  arcsec, so about 0.07 arcsec of the residual is still unexplained.
- Permutation importance: camera body − M1M3 air dominates (0.066 arcsec rise in nMAD),
  then M1M3 quadratic-in-radius, M1M3 glass temperature, M1M3 y gradient, truss − M1M3
  air (each 0.012–0.016 arcsec).
- Nightly medians of the residual: nMAD 0.0632 (Huber) → 0.0460 arcsec (trees, all
  inputs). Most of what is left is a per-night offset.

### 2026-10-26 — thermal history and the 2026-04-06 camera-region cover

`code/fetch_history.py` reads EFD time series per night (ESS air 111/112/113/301, truss
ESS 122 items 6–7, camera body `AverageTemp`). It gives, per visit, the 30 and 60 min
rates, the value at 0 deg twilight, the change since twilight, and the hours after
twilight. Kept in this study's cache only, not in value_added. `code/blur_history.py`
writes `output/20261026_blur_history/blur_history.pdf` (13 pages).

- History adds little. Held-out residual nMAD (arcsec, 5 folds by night, n 76,963
  visits): trees on thermal, wind and pointing 0.0759; plus 28 history inputs 0.0753
  (std over folds 0.003). No history input raises the shuffled nMAD by more than 0.0016
  arcsec. The 60 min rates correlate with the Huber residual within a night (Spearman
  rho −0.14 to −0.21), but the trees gain little from them.
- Cover: the raw blur difference falls from +0.320 to +0.194 arcsec (median; ±0.023
  arcsec from night-to-night scatter; 113 nights before, 60 after). The telemetry
  models explain the drop: the held-out residual medians differ by ≤ 0.005 arcsec
  across the date, and a cover flag adds nothing. The camera body − M1M3 air median
  goes from about −1.5 to −1.0 deg C over the same months.
- Huber slope of camera body − M1M3 air: −0.262 ± 0.001 arcsec/°C before, −0.307 ± 0.001
  arcsec/°C after (n 42,834 and 37,820 visits). The intercept moves by −0.059 arcsec.
  The other two slopes do not change. The cover may have changed how the camera
  temperature couples to the blur, but a 0.04 arcsec/°C change is small next to the
  season-to-season changes, so this is not settled.
