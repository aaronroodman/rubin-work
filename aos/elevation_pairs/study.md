---
study: elevation_pairs
topic: aos
status: active
question: How many same-night science visit pairs change elevation by at least 10 deg with azimuth and camera rotator angle each within 5 deg and the band unchanged, and do they give an elevation LUT sample to compare with the bounce tests?
answer: "Within 5 min (4,291 pairs, 130 nights, Danish 1.3, 22/12), the optical-state v-mode 1 changes by -2.47e-4 +- 0.28e-4 per deg of elevation (Pearson r = -0.132, Spearman rho = -0.126), about +0.27 um of equivalent hexapod dz per deg; v5, v3 and v8 carry stronger trends."
uses:
  value_added: v1@20261025b
  cwfs_science: d13-blitz_rso1071@20261026
started: 2026-10-29
---

# `elevation_pairs`

## Question

How many same-night science visit pairs change elevation by at least 10 deg with azimuth and camera rotator angle each within 5 deg and the band unchanged, and do they give an elevation LUT sample to compare with the bounce tests?

## Current answer

Within 5 min (4,291 pairs, 130 nights, Danish 1.3, 22/12), the optical-state v-mode 1 changes by -2.47e-4 +- 0.28e-4 per deg of elevation (Pearson r = -0.132, Spearman rho = -0.126), about +0.27 um of equivalent hexapod dz per deg; v5, v3 and v8 carry stronger trends. Search alone: 85 consecutive pairs within 2 min, 5,812 pairs within 5 min.

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

### 2026-10-29 — 22/12 optical state from Danish 1.3

Notebook `notebooks/elevation_pairs_state_v2.ipynb`, code `code/pairs_lib.py`, run `output/20261029_state_d13_22_12/` (`pairs_state.parquet`, `pairs_state.csv`, `slopes.csv`, `elevation_pairs_state.pdf`).

- Same 85 pairs (consecutive, dt <= 2 min). 62 pairs over 42 nights have Danish 1.3 Zernikes (`cwfs_science` `d13-blitz_rso1071`) on all four corners at both visits. That product starts on 20251021.
- Per-raft deviation: the mean of the SW0 and SW1 quality-cut medians, total minus intrinsic, in OCS. The per-raft MTAOS mean was not used: it is NaN on 0.449 of raft rows (fraction of all raft rows) in build 20261026, although `n_mtaos` > 0 there. That is a product bug to look at.
- 22/12 recovery with `aos_state.recover_optical_state`, then state = Trim - deviation. The hexapod LUT is not in the state, so its change is what the LUT leaves.
- Huber fit of the change in state v1 against the change in elevation: slope -3.2e-4 +- 1.7e-4 per deg (dimensionless v1 per deg; 1.9 sigma), Pearson r = -0.247, Spearman rho = -0.125, n = 62. Robust residual scatter nMAD 0.0158 (dimensionless v1). The Trim v1 hardly changes across a pair, so the signal is all in the deviation.
- At -9.0e-4 v1 per um of hexapod dz, the slope is about +0.36 um of equivalent dz per deg of elevation, 3.6 um over a 10 deg pair.
- Stronger slopes per deg of elevation: v5 +1.67e-3 (6.2 sigma), v8 -2.21e-3 (4.3 sigma), v3 -0.90e-3 (3.9 sigma); camera rx -1.1e-4 deg/deg (6.7 sigma), M2 dy +2.2 um/deg (5.4 sigma), M2 rx -4e-5 deg/deg (5.3 sigma).
- In 22/12 the recovered M2 dz and camera dz anticorrelate pair by pair, about +-100 um. Camera dx and dy changes reach 1000-2000 um. Both are weakly constrained combinations in 22/12, so read v-modes rather than single hexapod DOF.

### 2026-10-29 — widened to dt <= 5 min, non-consecutive pairs

Same notebook, now defaulting to `MAX_DT_MIN = 5.0` and `CONSECUTIVE_ONLY = False`. Run `output/20261029_state_d13_22_12_dt5min_all/`.

- 5,812 pairs over 175 nights; 4,291 pairs over 130 nights have a Danish 1.3 state at both visits. They use 5,394 distinct visits, so few pairs share a visit and the formal errors are only mildly optimistic.
- State v1 against elevation change, Huber fit: slope -2.47e-4 +- 0.28e-4 per deg (dimensionless v1 per deg; 8.8 sigma), Pearson r = -0.132, Spearman rho = -0.126, n = 4,291. nMAD residual 0.0213 (dimensionless v1). This agrees with the 2 min sample (-3.2e-4 +- 1.7e-4 per deg).
- In hexapod dz, at -9.0e-4 v1 per um, that is about +0.27 um of equivalent dz per deg of elevation.
- Within 5 min the Trim sometimes changes (nMAD of its v1 change 0.0075, dimensionless), but its slope against elevation is consistent with zero (+1.7e-5 +- 1.0e-5 per deg). The v1 trend stays in the deviation.
- Largest state slopes per deg of elevation: v5 +1.35e-3 (31 sigma), v3 -0.68e-3 (17 sigma), v8 -1.33e-3 (11 sigma), v11 +1.75e-3 (10 sigma); camera rx -8e-5 deg/deg (35 sigma), M2 dy +1.6 um/deg (29 sigma), M2 rx -3e-5 deg/deg (24 sigma).

### 2026-10-29 — v1 as DZ(1,4)

Converted with the ts_ofc v13 `lsst_sensitivity_dz_31_29_50.yaml`, which is `(k, j, DOF)`. In 22/12, v1 = 1 back-projects (`get_dofs_from_vmodes`) to M2 dz -463.94 um and camera dz -645.72 um, with every other DOF below 0.009 um. That gives DZ(1,4) = +17.418 um of wavefront per unit v1. The forward entries are -1.534e-2 (M2 dz) and -1.591e-2 (camera dz) um of wavefront per um.

- Optical state (Trim - deviation), 5 min sample: dDZ(1,4)/dEl = -4.31e-3 +- 0.49e-3 um of wavefront per deg of elevation, -0.043 um over a 10 deg pair. The 2 min sample gives -5.7e-3 +- 3.0e-3 um/deg.
- The measured deviation has the opposite sign, +4.64e-3 um/deg (deviation v1 slope +2.664e-4 per deg), since the Trim term is flat.
- DZ(1,4) computed from all 22 state DOF rather than from v1 alone: -2.34e-3 +- 0.36e-3 um/deg (Pearson r = -0.087, Spearman rho = -0.086, n = 4,291). The other v-modes add some DZ(1,4) of their own; that per-pair value correlates with the v1-only value at Pearson r = 0.884.
