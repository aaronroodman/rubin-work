---
study: elevation_pairs
topic: aos
status: active
question: How many same-night science visit pairs change elevation by at least 10 deg with azimuth and camera rotator angle each within 5 deg and the band unchanged, and do they give an elevation LUT sample to compare with the bounce tests?
answer: "85 consecutive science visit pairs over 52 nights within 2 min (|dEl| >= 10 deg, |dAz| and |dRot| < 5 deg, same band); 63 carry a recovered state. Relaxing to any pair within 60 min gives 58,740 pairs."
uses:
  value_added: v1@20261025b
started: 2026-10-29
---

# `elevation_pairs`

## Question

How many same-night science visit pairs change elevation by at least 10 deg with azimuth and camera rotator angle each within 5 deg and the band unchanged, and do they give an elevation LUT sample to compare with the bounce tests?

## Current answer

85 consecutive science visit pairs over 52 nights within 2 min (|dEl| >= 10 deg, |dAz| and |dRot| < 5 deg, same band); 63 carry a recovered state. Relaxing to any pair within 60 min gives 58,740 pairs. The per-pair open-loop hexapod change per degree of elevation is noisy: robust scatter (nMAD) 19 to 39 um/deg against medians of 3 to 22 um/deg.

## Log

### 2026-10-29 — started

Study created.

### 2026-10-29 — first pair search

Notebook `notebooks/elevation_pairs_v1.ipynb`, run `output/20261029_pairs_dt2min/` (`pairs.parquet`, `pairs.csv`, `hexapod_slopes.csv`, `elevation_pairs.pdf`). Band, image type and pointing come from ConsDB through `value_added.join_consdb`. Elevation is ConsDB `altitude`, rotator is `physical_rotator_angle`.

- Searched 95,938 science visits over 257 nights (20250415 to 20260713).
- Cuts: |dEl| >= 10 deg, |dAz| < 5 deg (wrapped), |dRot| < 5 deg, same band, same night.
- Pair counts by time limit (all pairs / consecutive science visits only):

| max dt [min] | all pairs | consecutive |
|---|---|---|
| 1 | 81 | 81 |
| 2 | 384 | 85 |
| 5 | 5,812 | 87 |
| 60 | 58,740 | 89 |
| none | 124,020 | 89 |

- Default sample, consecutive and dt <= 2 min: 85 pairs, 52 nights. Bands z 26, i 25, r 21, y 9, g 3, u 1. |dEl| is mostly 10–15 deg (74 pairs); only 3 exceed 20 deg.
- 63 of 85 pairs have a `v50_34__miw__consdb_v1` state on both visits. The 18 earliest pairs (before 20250724) have none.
- Open-loop hexapod change per degree of elevation (median over 63 pairs, nMAD in brackets, um/deg): M2 dy -21.7 (29.2), Cam dx +13.6 (39.1), Cam dy +14.6 (36.3). The scatter exceeds the median on every DOF, so a single pair says little; a fit over pairs is the next step.
