---
study: elevation_pairs
topic: aos
status: active
question: How many same-night science visit pairs change elevation by at least 10 deg with azimuth and camera rotator angle each within 5 deg and the band unchanged, and do they give an elevation LUT sample to compare with the bounce tests?
answer: "After the last LUT change (66,306 pairs within 5 min with |dEl| >= 2.5 deg, 107 nights, Danish 1.3, 22/12), deviation - Trim has a DZ(1,4) slope of -0.0100 +- 0.0032 um of wavefront per deg at midpoint elevations 25-40 deg, +0.0033 +- 0.0011 at 40-55 deg and -0.0002 +- 0.0007 at 55-75 deg (night-clustered errors). The 22/12 bounce points (bounce_aba d13-batoid-22_12) between 30, 40 and 50 deg imply -0.043 and +0.024 um per deg: same signs, 4 and 7 times larger. In the same 22/12 basis, 36 of the 43 bounce secants above 3 sigma have the pair slope's sign; the median pair/bounce slope ratio is 0.14 (dimensionless). Camera dx, camera rx and M2 dy match the bounce best."
uses:
  value_added: v1@20261118
  cwfs_science: d13-blitz_rso1071@20261026
  bounce_aba: d13-batoid-22_12@20261121
started: 2026-10-29
---

# `elevation_pairs`

## Question

How many same-night science visit pairs change elevation by at least 10 deg with azimuth and camera rotator angle each within 5 deg and the band unchanged, and do they give an elevation LUT sample to compare with the bounce tests?

## Current answer

After the last LUT change (66,306 pairs within 5 min with |dEl| >= 2.5 deg, 107 nights, Danish 1.3, 22/12), deviation - Trim has a DZ(1,4) slope of -0.0100 +- 0.0032 um of wavefront per deg at midpoint elevations 25-40 deg, +0.0033 +- 0.0011 at 40-55 deg and -0.0002 +- 0.0007 at 55-75 deg (night-clustered errors). The 22/12 bounce points (bounce_aba d13-batoid-22_12) between 30, 40 and 50 deg imply -0.043 and +0.024 um per deg: same signs, 4 and 7 times larger. In the same 22/12 basis, 36 of the 43 bounce secants above 3 sigma have the pair slope's sign; the median pair/bounce slope ratio is 0.14 (dimensionless). Camera dx, camera rx and M2 dy match the bounce best. v1 per deg (dimensionless): -3.4e-4 +- 2.0e-4, +2.9e-4 +- 0.7e-4 and +0.4e-4 +- 0.5e-4 in the same bins.

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

### 2026-10-29 — deviation - Trim, the bounce-test sign

The bounce test reports the optical state as deviation - Trim, the negative of Trim - deviation used above. Huber fits of DZ(1,4) against elevation change, in um of wavefront per deg of elevation, 5 min sample (n = 4,291). "v1" uses +17.418 um per unit v1; "22 DOF" is the 22 DOF through the ts_ofc v13 DZ row (k=1, j=4).

| quantity | v1 slope | 22 DOF slope | v1, dEl = -30 deg | 22 DOF, dEl = -30 deg |
|---|---|---|---|---|
| deviation | +4.64e-3 +- 0.38e-3 | +3.01e-3 +- 0.29e-3 | -0.139 +- 0.011 | -0.090 +- 0.009 |
| Trim | +0.30e-3 +- 0.17e-3 | +0.54e-3 +- 0.14e-3 | -0.009 +- 0.005 | -0.016 +- 0.004 |
| deviation - Trim | +4.31e-3 +- 0.49e-3 | +2.34e-3 +- 0.36e-3 | -0.129 +- 0.015 | -0.070 +- 0.011 |

The 2 min sample (n = 62) gives deviation - Trim -0.170 +- 0.091 (v1) and -0.134 +- 0.066 (22 DOF) um over dEl = -30 deg.

### 2026-10-29: hexbin plots of deviation - Trim

`elevation_pairs_state_v2.ipynb` now plots deviation - Trim, the bounce-test sign, as
hexbin heatmaps with medians per 2 deg bin and the Huber line, four panels per page. The x
range is |dEl| <= 30 deg, and the y range spans the 0.1% to 99.9% quantiles. The
`dmt_*` rows of `slopes.csv` hold these fits. The v1 slope is +2.47e-4 +- 0.28e-4 per deg
of elevation (dimensionless v1), as before. Run `output/20261029_state_d13_22_12_dt5min_all/`,
rewritten.

### 2026-11-20: midpoint-elevation bins, any dEl, night bootstrap, bounce comparison

`elevation_pairs_bins_v1.ipynb`, run `output/20261120_bins_d13_22_12_dt5min_lut20251209/`. Pairs within 5 min at any dEl, day_obs >= 20251209 (the last LUT change, `thermal_focus.md`) less the `LUT_EPOCH_OFFSET_NIGHTS`: 175,399 pairs with a state, 126 nights. 87,888 of them have |dEl| < 1 deg. Bounce points are copied into `code/bounce_points_20261102.csv` by `code/extract_bounce_points.py`.

- The 1000-resample night bootstrap took 1,471 s. Bootstrap errors are 3-7 times the formal Huber errors (dimensionless ratio).
- A night-clustered sandwich error matches that bootstrap to within 5% for all 39 quantities and 4 samples, in 0.55 s. It replaces the bootstrap from the next run.
- The LUT page shows two more nights on a different M2 LUT, about +1,300 um in M2 dz: 20251214 and 20251216 (566 visits). The camera dz branches are per-band focus offsets.
- DZ(1,4) slope of deviation - Trim per midpoint bin [um of wavefront per deg]: 25-40 deg -0.0103 +- 0.0030, 40-55 deg +0.0032 +- 0.0010, 55-75 deg -0.0001 +- 0.0007. The signs follow the bounce points between 30, 40, 50 and 70 deg, but the slopes are 4-8 times smaller.

### 2026-11-21: night-clustered errors, two more LUT nights dropped

Rerun of `elevation_pairs_bins_v1.ipynb`, run `output/20261121_bins_d13_22_12_dt5min_lut20251209/` (`value_added` v1@20261118), 3.6 min in all.

- 20251214 and 20251216 dropped: 174,761 pairs with a state, 124 nights.
- Errors are night-clustered sandwich errors (0.48 s). A 200-resample night bootstrap of v1 and DZ(1,4) agrees to within 10% in all four samples. Clustered errors are 3-8 times the formal Huber errors (dimensionless ratio).
- DZ(1,4) slope of deviation - Trim [um of wavefront per deg]: 25-40 deg -0.0103 +- 0.0030, 40-55 deg +0.0032 +- 0.0010, 55-75 deg -0.0001 +- 0.0007. Between 30, 40 and 50 deg the bounce points imply -0.043 and +0.024: same signs, 4 and 7 times larger than the pair slopes.
- Linearity, DZ(1,4) refit by |dEl| range: in each bin the 0-5, 5-10 and >= 10 deg slopes agree within 2 sigma, except 55-75 deg, where |dEl| < 5 deg gives -0.0020 +- 0.0007 against +0.0004 +- 0.0007 at 5-10 deg.
- 109,000 of the 175,000 pairs are at 55-75 deg with |dEl| < 5 deg, so the all-pairs slope (DZ(1,4) +0.0002 +- 0.0007) mostly reflects that corner.
- The 22 DOF disagree with the bounce DOF by up to 10 times (M2 dy +500 to +900 um at 30-50 deg in the bounce, under 50 um implied here): the bases differ, 22/12 against RBR 50/50.
- Camera dx and rx agree with the bounce. Camera dx slopes are +9.7 to +12.2 um per deg in all three bins, against bounce secants of +9.7 to +13.8 um per deg. Chained from 0 at 70 deg, the pairs give camera dx -420 +- 53 um at 30 deg; the bounce gives -586 +- 110 um. Camera rx chained gives -0.0031 +- 0.0002 deg, against bounce -0.0040 +- 0.0003 deg. Camera ry has the right sign at 40% of the bounce (+0.0010 against +0.0025 deg). Camera dy and DZ(1,4) disagree; DZ(1,4) at 30 deg is +0.06 +- 0.04 um against bounce +0.37 +- 0.03 um.

### 2026-11-21: 22/12 bounce points from `bounce_aba`, |dEl| >= 2.5 deg

Rerun of `elevation_pairs_bins_v1.ipynb`, run `output/20261121_bins_d13_22_12_dt5min_del2p5_lut20251209/`. The bounce points now come from `bounce_aba` variant `d13-batoid-22_12` (22/12, batoid intrinsic, as the pairs), not from `code/bounce_points_20261102.csv` (RBR 50/50). Bounce DZ(1,j) is the 22/12 subspace `fit`, to match the pairs' S(1,j) x DOF; it differs from the raw delta by under 0.001 um of wavefront. v2..v12 now have bounce pages. Figures: `figures/bounce22_12_v1_dz14_dz15_dz16.png`, `figures/bounce22_12_v12_m2dz_m2dx_m2dy.png`.

- Cut |dEl| >= 2.5 deg: 66,306 pairs with a state (of 174,761), 107 nights. Clustered slope errors barely change, so the dropped pairs carried almost no slope information. At 55-75 deg, DZ(1,4) is still -0.0020 +- 0.0007 um per deg at |dEl| 2.5-5 deg against +0.0004 +- 0.0007 at 5-10 deg, so the cut does not remove the small-dEl anomaly there.
- DZ(1,4) slope [um of wavefront per deg]: 25-40 deg -0.0100 +- 0.0032, 40-55 deg +0.0033 +- 0.0011, 55-75 deg -0.0002 +- 0.0007. v1 and DZ(1,4) bounce points are within 0.0004 of the RBR ones (v1 dimensionless, DZ in um), so the v1 and DZ(1,4) comparison is unchanged: same signs, pair slopes 4-7 times smaller.
- Same-basis DOF: 36 of the 43 bounce secants above 3 sigma (30-40, 40-50, 60-70 deg) have the pair slope's sign. Median pair/bounce slope ratio 0.14 (dimensionless).
- Chained from 0 at 70 deg to 30 deg, pairs against bounce: camera dx -421 +- 54 against -525 +- 110 um; camera rx -0.0031 +- 0.0002 against -0.0044 +- 0.0002 deg; M2 dy +32 +- 5 against +58 +- 5 um; camera ry +0.0010 +- 0.0002 against +0.0023 +- 0.0002 deg. M2 dy, which disagreed by up to 10 times against RBR 50/50, now agrees in sign and to a factor 2. Camera dy (-194 +- 64 against -900 +- 99 um), v1 (-0.002 +- 0.002 against +0.019 +- 0.002, dimensionless) and DZ(1,4) (+0.05 +- 0.04 against +0.37 +- 0.03 um) disagree.
- Caveat: the bounce 40 deg legs are three April-May nights, the 50 and 60 deg legs single July nights, and only 20260713 has two legs (30 and 75 deg). A secant between legs mixes elevation with night-to-night differences; the zigzag at 40 deg in v1, DZ(1,4) and several DOF looks like that.
- `code/bounce_points_20261102.csv` and `code/extract_bounce_points.py` are now unused; kept pending Aaron's call.
- Rerun 2026-11-22 into the same run directory: bounce pages and per-quantity pages are grouped DZ, then v-modes, then DOF, each group on its own pages. Numbers unchanged.

