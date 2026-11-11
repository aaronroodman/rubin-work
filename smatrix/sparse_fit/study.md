---
study: sparse_fit
topic: smatrix
status: active
question: "If the donut fit fits only the primary aberrations and fixes secondary and tertiary at the design intrinsic, is the optical state still constrained, and how does the fixed content leak into the primaries?"
answer: "Piece 1 answered: nearly every DOF drives primary and secondary of a family with field correlation about ±1 (dimensionless). Primary-only observability loses nothing for 22 DOF / 12 v-modes (min 0.99, dimensionless) and degrades only high-order M2 bending modes for 50/34 (min 0.36). Pieces 2 (the leakage matrix) and 3 (the sparse MIW) are open."
uses: {}  # no rubin-work products; ts_ofc v13 sensitivity matrix
started: 2026-09-01
---

# `sparse_fit`: fitting only the primary aberrations

> **Code:** `code/analyze_sparse_fit.py` · **Write-up:** [`../vmode/vmode.md`](../vmode/vmode.md), "Sparse-fit findings" · **Output:** `smatrix/output/vmode/sparse_fit_study.pdf`

## Question

If the donut fit fits only the primary aberrations and fixes secondary and tertiary at the design intrinsic, is the optical state still constrained, and how does the fixed content leak into the primaries?

## Current answer

Piece 1 of 3 (2026-09-01):

- **Primary and secondary are near-degenerate per DOF.** Nearly every DOF, bending modes and hexapod tilts especially, drives the primary and secondary of a family with field correlation about ±1 (Pearson r, dimensionless). Astigmatism first-to-second order from the tilts is −0.44 to −0.95 (cos) and −0.40 to −0.87 (sin), bracketing the MIW's −0.28 and −0.62. A small fit bias can then move power between orders, one route to the MIW radial-order correlations (`optatmo/miw_radial_correlations`).
- **Primary-only observability** (primary-only over full matrix, dimensionless): 22 DOF / 12 v-modes, median 1.00 and min 0.99 at both corner sensors and field-complete sampling. 50 DOF / 34 v-modes, median 0.97 and min 0.36 field-complete, median 0.96 and min 0.44 at the corners; the degraded DOF are M2 bending b10, b11, b14, b15, b19, b20. This is the no-leakage upper bound.

Open: piece 2, the leakage matrix `L[j_prim, j_sec]` (needs the Danish forward model and injected-secondary simulations); piece 3, `A_sparse = A_prim + L @ A_sec` and a rebuilt sparse MIW.

## Log

### 2026-09-01 — backfilled from docs and history

Backfilled 2026-10-18 from [`../vmode/vmode.md`](../vmode/vmode.md), "Sparse-fit findings", and commit history; split out of the old `vmode` study in the phase 5 triage and converted as active. `analyze_sparse_fit.py` moved here from `smatrix/code/vmode/`; run it from `smatrix/` (`python sparse_fit/code/analyze_sparse_fit.py`), output to `smatrix/output/vmode/`.

Retracted on the way: "production zeroes coma2/tref2/astig3/tet2" (that was the non-loaded `new_measured` file), and an early observability run that used the default `zn_idx`.

Next: piece 2, then piece 3.

### 2026-11-06 — plan for pieces 2 and 3

The fit set is Noll 4 to 11, 14, 15, 20, 21, 22 and 27, 28, the `fam_tables` `d13-sparse` set plus Z22, which has a particular M1M3 thermal-gradient sensitivity; `d13-sparse` itself is run alongside for comparison. An orthogonal fit of the optical path difference (OPD) gives no leakage: on the annulus the annular Zernikes are orthogonal, so the sparse rows equal the dense rows. The leakage comes from fitting donut images.

`code/donut_leakage.py` follows ts_wep's Danish setup: noise-free model donuts from the defocused Batoid `zernikeTA` to Noll 78, binned ×2, fit with the sparse set free and every other term at the nominal Batoid model. Per point it gives the dense in-focus rows, the direct sparse sensitivity of all 50 DOF, and `L[j_fit, j_other]` for the unfitted terms Noll 4 to 28. The check is `A_direct = A_dense[fit] + L @ A_dense[other]`, at the 8 corner half-chips and at 9 science points (thx, thy each -0.75, 0, +0.75 deg; intra and extra).

Smoke test at (0.75, 0) deg extra-focal: the leakage pairs each secondary term with the primary term of the same symmetry, Z12/13 to Z6/5 and Z16/17 to Z8/7 at about 4.8 (dimensionless, µm fitted per µm injected), Z23/24 to Z5/6 at about 8. For M2 Rx the dense rows miss the direct result by 7.3% (relative norm); adding `L` brings that to 2.0%.


### 2026-11-09 — first full leakage run (job 44033445, run `20261109_donut_leakage`)

All 26 configurations and 50 DOF finished in 3 min. The nominal fit returns the intrinsic to < 4e-8 µm. Residual |A_direct − model| / |A_direct| (dimensionless, per DOF), median over configurations, fit set `prod_z22`:

| DOF group | model A_dense[fit] | model A_dense[fit] + L @ A_dense[other] |
|---|---|---|
| M2 hexapod | 0.072 | 0.020 |
| camera hexapod | 0.042 | 0.006 |
| M1M3 B1–B10 | 0.49 | 0.12 |
| M1M3 B11–B20 | 1.51 | 0.67 |
| M2 B1–B10 | 0.57 | 0.16 |
| M2 B11–B20 | 0.95 | 0.57 |

`prod` gives nearly the same numbers. The rigid-body DOF close with L; the bending modes do not. The likely cause is that the bending modes change Noll terms above 28, which leak into the fit and which L (injections in Noll 4–28 only) does not cover. Next: extend A_dense and L to Noll 78, the reference model's jmax.

### 2026-11-09 — leakage with dense rows and L to Noll 78 (job 44041372, run `20261109_donut_leakage_j78`)

Same residual as above, median over the 26 configurations, fit set `prod_z22` (`prod` within 0.01):

| DOF group | A_dense[fit] alone | with L, Noll 4–28 | with L, Noll 4–78 | with L, Noll 4–78, 90th percentile |
|---|---|---|---|---|
| M2 hexapod | 0.066 | 0.020 | 0.015 | 0.025 |
| camera hexapod | 0.042 | 0.006 | 0.006 | 0.023 |
| M1M3 B1–B10 | 0.49 | 0.12 | 0.030 | 0.31 |
| M1M3 B11–B20 | 1.51 | 0.67 | 0.18 | 0.44 |
| M2 B1–B10 | 0.56 | 0.16 | 0.054 | 0.64 |
| M2 B11–B20 | 0.95 | 0.57 | 0.19 | 0.58 |

Leakage from Noll 29–78 explains most of the bending-mode mismatch. The tails are not small-denominator cases: the worst are corner half-chips (R40_SW1 M1M3 B15, residual 5.0 against |A_direct| 2.9, µm per µm). Intra-focal configurations are worse than extra-focal (median per configuration 0.07–0.15 against 0.05–0.08). A likely cause is that the model uses in-focus Zernike changes while the donut sees the defocused (TA) wavefront; not yet tested. A_direct itself is the sparse sensitivity at these points and does not depend on the model.

### 2026-11-09 — step size, central differences, defocused model input

At the worst corner (R40_SW1) the 0.05 µm step is already nonlinear: the fit gives the same answer from different starting points, but |Δz|/step for M1M3 B15 is 1.79–1.83 µm per µm for steps up to 0.1 µm and jumps above that. `DZ_TARGET_UM` is now 0.01, and A_direct, L and the model inputs use central differences. `donut_leakage.py` also records the defocused changes per DOF: `dTA` (the `zernikeTA` fit, as the ts_wep reference) and `dOPD` (the defocused wavefront, `zernikeGQ`).

Residual / |A_direct| (dimensionless) at R40_SW1, fit set `prod_z22`:

| model input | M2 Rx | M1M3 B15 | M2 B7 |
|---|---|---|---|
| in-focus A_dense | 0.013 | 0.22 | 0.68 |
| defocused dOPD | 0.014 | 0.21 | 0.68 |
| defocused dTA | 0.015 | 0.053 | 0.18 |

Caveat: the test donuts are drawn by Danish from TA coefficients, so closure with dTA is partly by construction. The real test is a donut ray-traced by batoid, or drawn from the defocused OPD.

### 2026-11-09 — ray-traced and OPD-drawn test donuts

`donut_leakage.py` now fits three kinds of test donut: Danish drawn from the defocused TA fit (`A_direct`), Danish drawn from the defocused OPD (`A_direct_opd`), and batoid ray-traced (`A_direct_rt`). The ray-traced donut uses a fixed 1024² pupil grid (0.47M unvignetted rays, no photon noise), cloud-in-cell deposit on a 5× oversampled grid, a Kolmogorov 1.0 arcsec blur and 2× binning, centred on its own chief ray. Batoid focal-plane x, y map to Danish columns, rows with no flip: on difference images for M2 dx, dy and Rx at (0.9, 0.35) deg and R40_SW1, identity gives |ΔR − ΔD| / |ΔD| = 0.22–0.31 against ≥ 0.35 for any other orientation.

Smoke test (R40_SW1 and sci_p0_extra, M2 Rx, M1M3 B15, M2 B7), residual / |A_direct_rt| with L:

| model input | R40_SW1 | sci_p0_extra |
|---|---|---|
| in-focus A_dense | 0.033, 0.26, 0.85 | 0.053, 0.11, 0.42 |
| defocused dOPD | 0.034, 0.25, 0.85 | 0.053, 0.11, 0.42 |
| defocused dTA | 0.036, 0.040, 0.21 | 0.053, 0.10, 0.40 |

dTA is the right input for the ray-traced truth too, as ts_wep's choice of a TA reference implies. The ray-traced and TA-drawn A_direct differ by 3.9–7% for M2 Rx and B15 and by 29–37% for M2 B7 (|A| 0.2–0.4 µm per µm). For M2 Rx the ray-traced response is 2.3% (R40_SW1) and 3.7% (sci_p0_extra) smaller in norm.

### 2026-11-09 — full run with ray-traced truth (job 44049602, run `20261109_donut_leakage_rt`)

All 26 configurations, 50 DOF, 21 min. Residual / |A_direct_rt| (dimensionless), median over corners | science points, fit set `prod_z22`:

| DOF group | dTA[fit] alone | A_dense + L | dTA + L | TA-drawn vs ray-traced |
|---|---|---|---|---|
| M2 hexapod | 0.098, 0.093 | 0.049, 0.044 | 0.048, 0.044 | 0.059, 0.052 |
| camera hexapod | 0.066, 0.104 | 0.059, 0.045 | 0.059, 0.045 | 0.062, 0.046 |
| M1M3 B1–B10 | 0.48, 0.55 | 0.035, 0.039 | 0.033, 0.041 | 0.033, 0.038 |
| M1M3 B11–B20 | 1.59, 1.49 | 0.115, 0.085 | 0.065, 0.082 | 0.053, 0.062 |
| M2 B1–B10 | 0.57, 0.59 | 0.061, 0.039 | 0.033, 0.040 | 0.032, 0.039 |
| M2 B11–B20 | 0.94, 0.96 | 0.210, 0.075 | 0.071, 0.076 | 0.048, 0.048 |

With dTA + L the decomposition closes to 3–8% everywhere. That floor is the Danish-drawn against ray-traced difference, because L is measured with Danish-drawn injections. The worst single DOF are B6 and B7 of both mirrors (median 0.23–0.30), whose sparse response is small (|A_direct_rt| about 0.3 µm per µm).

**L depends only on field radius and defocus side.** Raw L differs by 35–85% (Frobenius, relative to the group mean) between points at the same radius. Rotated into a frame with the field point on +x, it agrees to 0.2–1.2%. The radial dependence is strong: 54% from 0 to 0.75 deg, 73% from 0.75 to 1.06 deg, 80% from 1.06 to 1.64 deg (extra-focal). So a full-field sparse matrix needs L on a radial grid per defocus side (L costs an estimated 2–4 min of one CPU per point, not yet timed) and dTA at every field point, which batoid gives quickly.

### 2026-11-10 — spiders, mask consistency, ray-traced L (smoke run `20261110_smoke_rt_l`)

- Danish fits already had spiders on (`spider_angle=0.0`; Danish skips them only for `None`).
  Batoid's design `LSST_r.yaml` has no spider, so the ray-traced donuts had none. The ray
  trace now masks the ts_wep `Spider_3D` vanes on the stop plane (z = 0.439 m) with Danish's
  own vane projection (`code/pupil_masks.py`), at the same `SPIDER_ANGLE` = 0 deg.
- Mask consistency: Danish's circular masks (ts_wep mask file) against batoid design
  vignetting, at all 26 configurations: rays clear in one and blocked in the other are
  ≤ 2e-4 of the rays inside the M1 edge (dimensionless). Both are the design, consistently.
- Spider shadow image (spider on minus off), Danish vs ray trace, 6 configurations:
  Pearson r = 0.93–0.96 (n = 6561 pixels); ray-traced shadow total |flux| 9–22% larger.
- Ray-traced L: a stop-plane `batoid.OPDScreen` with one annular Noll j (4..78), ±0.01 µm.
  It moves the defocused TA term j by −0.01 µm and every other term by < 1e-5 µm.
  `G_rt = F M^-1` (F: fit change per injection, M: ΔTA per injection) is the fit response
  per TA term; prediction `A_sparse = G_rt @ dTA`.
- Smoke, R44_SW0, residual / |A_direct_rt| (dimensionless), fit set prod:
  M2 dz 0.022, M2 Rx 0.017, M1M3 B6 0.061; with Danish-drawn L 0.040, 0.042, 0.211.
  max |G_rt[:, fit] − I| = 0.046 (prod), 0.20 (prod_z22). 794 s on 1 CPU for 3 DOF.

### 2026-11-10 — full run with ray-traced L (job 44167231, run `20261110_donut_leakage_rt_l`)

26 configurations, 31 min on 26 CPUs. Truth is the spider-masked ray-traced fit. Median
residual / |A_direct_rt| (dimensionless), corners (n=8) | science (n=18), fit set prod
(`code/analyze_rt_l.py`):

| DOF group | G_rt @ dTA | dTA + Danish-drawn L | dense rows only |
|---|---|---|---|
| M2 hexapod | 0.027 \| 0.051 | 0.037 \| 0.034 | 0.082 \| 0.080 |
| camera hexapod | 0.021 \| 0.030 | 0.041 \| 0.036 | 0.060 \| 0.121 |
| M1M3 B1–B10 | 0.027 \| 0.026 | 0.029 \| 0.030 | 0.48 \| 0.54 |
| M1M3 B11–B20 | 0.036 \| 0.049 | 0.052 \| 0.071 | 1.51 \| 1.52 |
| M2 B1–B10 | 0.033 \| 0.034 | 0.027 \| 0.031 | 0.56 \| 0.57 |
| M2 B11–B20 | 0.043 \| 0.054 | 0.056 \| 0.057 | 0.95 \| 0.98 |

- Worst DOF with G_rt (median over 26): M2 B7 0.215, M2 B6 0.198, M1M3 B6 0.154, M1M3 B7 0.141.
- Ray-traced L helps by about 1 point on most groups, not more. A 2–5% floor remains that a
  linear response to TA Noll 4..78 does not capture; not yet identified.
- G_rt on the fitted terms: diagonal 0.98–1.00 median (range 0.97–1.09 over 26); largest
  off-diagonal 0.08 (Z5 row, prod), 0.14 (Z4 row, prod_z22: Z4–Z22 mixing).
- G_rt field symmetry: rotated into the field-on-+x frame, equal-radius points agree to
  0.2–1.1% (RMS deviation over RMS mean, n = 4 per group); raw spread 6–76%.

### 2026-11-10 — the closure floor and B6/B7 (`code/diag_floor.py`, run `20261110_diag_floor_smoke`)

R44_SW0, residual / |A_rt| (dimensionless). "Screen to J" is a stop-plane OPD screen tuned
to reproduce the DOF's change in defocused TA, Noll 4..J.

| | M2 Rx | M2 B7 |
|---|---|---|
| real DOF vs screen to 78 | 0.013 | 0.052 |
| real DOF vs screen to 153 | 0.012 | 0.0014 |
| screen vs G_rt @ dTA | 0.010 | 0.074–0.108 |

- B7: TA above Noll 78 is 0.12% RMS of Noll 4..78 but moves the fit by 5%; to Noll 153 the
  screen reproduces the DOF to 0.14%.
- G_rt columns are noisy, not nonlinear: at injection amplitudes 0.0025, 0.01, 0.04 µm the
  column changes by 1.5–7% for strong columns and 23–37% for weak ones (Noll 66, 78),
  without growing with amplitude. Implied fit noise about 1e-4 µm per ray-traced fit.
  Summing 75 noisy columns gives the B7 miss.
- M2 Rx keeps 1.2% at any order: a vignetting change, which no screen represents.
- Conclusion: the L route would need Noll 153 (150 injections), much quieter G columns,
  and still leaves about 1% on rigid DOFs at corners. Switch to brute force.

### 2026-11-10 — dense 21-term fit against the dense Smatrix rows (job 44171999, run `20261110_bruteforce_check`)

Ray-traced donuts, no L; fit sets prod, prod_z22 and dense21 (Noll 4–19, 22–26); 25 min.
Median |A_rt − A_dense[fit]| / |A_rt| (dimensionless), corners (n=8) | science (n=18);
`A_dense` is the in-focus change the dense Smatrix holds (`code/analyze_bruteforce.py`):

| DOF group | dense21 | prod |
|---|---|---|
| M2 hexapod | 0.075 \| 0.050 | 0.082 \| 0.080 |
| camera hexapod | 0.069 \| 0.051 | 0.060 \| 0.121 |
| M1M3 B1–B10 | 0.126 \| 0.090 | 0.48 \| 0.54 |
| M1M3 B11–B20 | 0.96 \| 1.19 | 1.51 \| 1.52 |
| M2 B1–B10 | 0.136 \| 0.115 | 0.56 \| 0.57 |
| M2 B11–B20 | 0.94 \| 0.98 | 0.95 \| 0.98 |

- For B11–B20 the dense fit response points elsewhere: cosine(A_rt, A_dense[fit]) median
  0.42 (M1M3), 0.30 (M2), with |A_rt| / |A_dense[fit]| 0.92 and 1.56 (dimensionless).
- Mechanism, M1M3 B11 on axis: wavefront change per µm Z6 +2.55, Z12 −1.81, Z24 +0.45,
  Z38 −0.091 µm. The dense21 fit returns Z6 −0.20 µm (ray-traced; −0.02 Danish-drawn).
  A Danish-drawn Z38 injection moves the dense21 Z6 by 26 µm per µm (Z12 −5.4, Z24 +2.1).
  Danish fits ray slopes; slopes of Zernikes with the same m and different n are not
  orthogonal, so an unfitted high-n term leaks into fitted lower-n terms, with a
  coefficient that grows with n. In prod, Z12, Z24, Z38 → Z6: 5.0, 8.7, 17.6 per µm.
- Danish-drawn and ray-traced agree on this (B11 dense21 Z12 −1.29 vs −1.26 µm per µm),
  so it is the estimator, not the image model.
- Step check, |A(step) − A(2 step)| / |A| median: 1–2% for prod and prod_z22, 3–8% for
  dense21 (worst M2 B16 0.19).

### 2026-11-10 — fit convergence, linearity and step size (R44_SW0, `code/scan_test.py`)

- **Danish fit stopping.** `least_squares` stops on `xtol` (status 3) after ~17 evaluations;
  restarts land 1e-4–1e-3 µm apart at equal cost. Danish's analytic Jacobian differs from
  finite differences of its own model by 0.1–4.7% per parameter (high-order Zernikes worst).
  A second `least_squares` pass with a `3-point` Jacobian and explicit scaling is now in
  `fit()`; Gauss-Newton polishing with the analytic or a numerical Jacobian did not help.
- **Model image** is smooth in the Zernikes (second differences scale as h² down to 1e-6 µm,
  with a small kink near 1e-3 µm). Danish 1.4.0 gives identical images; draw 5 vs 6 ms,
  Jacobian 12 vs 13 ms (1.4 vs 1.3) for the 21-term model.
- **Linearity**, 5 amplitudes k = −2..2 × step, quadratic fit; quadratic term and RMS scatter
  about the quadratic, each over the linear change at the largest amplitude (dimensionless,
  vector norms over fitted terms), ray-traced:

| DOF, fit set | step ×1: quad, scatter | step ×4: quad, scatter | slope change ×1 → ×4 |
|---|---|---|---|
| M2 Rx, dense21 | 0.027, 0.021 | 0.026, 0.008 | 0.2% |
| M1M3 B14, prod | 0.032, 0.004 | 0.032, 0.015 | 1% |
| M1M3 B14, dense21 | 0.121, 0.100 | 0.276, 0.053 | 13% |

- B14 with the dense fit is nonlinear and noisy at any step: most of its wavefront is above
  Noll 26 and reaches the fitted terms through leak coefficients of 10–26. A linear Smatrix
  column describes it poorly. Earlier per-term deviation numbers (17-point scans) were
  normalized per term and overstated the problem; ignore them.

### 2026-11-10 — brute-force design and smoke runs

- Five-point mode: ray-traced fits at k = −2..2 × step (step ×2, 0.02 µm RMS in-focus Noll
  4..78 change per step); slope, quadratic term, curvature and scatter saved per DOF,
  configuration and fit set. Donuts saved as `donuts_<config>.fits` (NOMINAL; STEPS
  (ndof, 4, 81, 81) at k in header STEPS_K; DOFS table).
- The second fit pass (numerical Jacobian) changed no slope, curvature or scatter at the
  1e-3 level on R44_SW0 (M2 Rx, M1M3 B14, M2 B17) and costs 46% more; off by default.
- R44_SW0, step ×2, ray-traced, curvature | scatter (dimensionless, as above): prod M2 Rx
  0.013 | 0.010, B14 0.036 | 0.003, M2 B17 0.035 | 0.010; dense21 M2 Rx 0.033 | 0.009,
  B14 0.26 | 0.064, M2 B17 0.47 | 0.041.
- Field sets: `check` 26, `fam` 770 (385 Gauss-Legendre points × 2 sides, camera shift),
  `cwfs` 144 (6 × 3 per half-chip from cameraGeom, Detector shift); `all` 940.
- Cost about 28 s per DOF per configuration on one CPU: about 370 CPU-hours for 50 DOF.
  `bruteforce.sbatch`: one job per 5-DOF block, 48 CPUs, Danish 1.4.0 from
  `/sdf/group/rubin/u/roodman/LSST/packages/danish_1.4.0`.
