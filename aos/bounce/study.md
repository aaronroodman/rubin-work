---
study: bounce
topic: aos
status: done
question: "Do the elevation and rotator bounce tests show repeatable, gravity-driven changes in the optical state, for the look-up table?"
answer: "Yes, and the default recovery puts DOF out of range: 106 of 450 per-(leg, night) rows exceed their range, against 7 under Range-Bounded Recovery, at a median cost of +0.0027 arcsec FWHM."
uses: {}  # old aos/output tree
started: 2026-09-06
---

# `bounce`

> **Code:** `code/` (`run_bounce.py`, `bounce_lib.py`; the `bounce` rule); the reduction is `rubinwork.bounce`, the tables the `bounce_tables` product · **Write-up:** [`bounce.md`](bounce.md) · **Output:** `output/<P>_<M>/` (the study `output@`) · **Note:** `notes/aos-bounce-test-summary/`

## Question

Do the elevation and rotator bounce tests show repeatable, gravity-driven changes in the optical state, for the look-up table?

## Current answer

Results note delivered (`notes/aos-bounce-test-summary/`). Counting `kind = dof` rows, 106 of 450 per-(leg, night) rows exceed their range under the default recovery against 7 under RBR, at a median cost of +0.0027 arcsec FWHM (range −0.0002 to +0.0158 arcsec over nine (leg, night) points).

## Log

### 2026-10-12 — backfilled from docs and history

Moved from `aos/code/bounce/` and the study doc. `rule bounce` runs `bounce/code/run_bounce.py`. The bending-mode notebook is its own study, `bending_mode_test`. Converted 2026-10-23 in phase 5.

### 2026-10-28 — Danish 1.3, RBR 50/50 schemes and a first LUT fit

The input is the `miw` build `d13-A_50_50_i_rbr_5rot@20261028b`. Its `fits.parquet` already subtracts the 50/50 RBR MIW from each donut, so the bounce needed no new product. The output is in run `20261028_d13_miw_rbr5050_schemes_rbr5050_lut`. The earlier run, using the RBR 50/34 scheme, is kept in `output/danish_1_3_A_50_50_i_rbr_5rot/`.

- The RBR scheme is now 50/50 with full DOF. It uses the ts_ofc `dof_range_bounded` solve with kappa 4, power 3 and barrier weight 1.0 µm² of wavefront, the settings the MIW build uses.
- Default scheme per bounce: RBR 50/50 for elevation and 5/5 camera hexapod for the rotator. The scheme is listed on page 1 of each DOF PDF.
- The `bounce_fwhm_metric.pdf` bar chart is now page 1 of `bounce_fwhm_vs_bvalue.pdf`.
- Added `bounce_lut.pdf`. Each DOF is fitted with a Huber-weighted polynomial with no constant term in (x − ref). A DOF is kept at ≥ 3 sigma.
  - Elevation: 39 of 50 DOF kept.
  - Rotator: 4 of 5 DOF kept.
- **The per-DOF LUT does not correct the elevation sweep.**
  - Per-point FWHM after correction: 0.013–0.066 arcsec.
  - LUT FWHM after correction: 0.091–0.562 arcsec, which is worse than no correction (0.081–0.367 arcsec) at 5 of 7 points.
  - Cause: the per-DOF medians are not a consistent state, because the RBR DOF are strongly correlated. The per-DOF median DOF alone leaves 0.041–0.103 arcsec. Fitting each DOF with its own weights, then cutting DOF one at a time, breaks the cancellations further.
  - Rotator LUT: 0.054 and 0.036 arcsec, against 0.045 and 0.033 arcsec per point.
  - Open: fit the LUT in a space where the errors are independent (v-modes, or the wavefront, then solve).

### 2026-10-28 — Δ DZ, Δ v-mode and Δ DOF vs elevation

- New `bounce_trends.pdf`: paired Δ vs B-set position in three views, nights colored, with the LUT-form fit and the chi2/dof of the nights sharing a B position. Run `20261028_d13_miw_rbr5050_trends`.
- T720 elevation, median over quantities of chi2/dof (dimensionless), nights at one position about their weighted mean, then the LUT-form fit:
  - Δ DZ_kj (126): 3.96, 3.86.
  - Δ v-mode, RBR 50/50 decomposition (50): 4.98, 4.35.
  - Δ DOF, RBR 50/50 (50): 7.23, 10.07.
- T724 rotator, the same: DZ 0.59, 0.59; v-mode 2.20, 2.20; DOF 2.09, 2.09.
- The three 40 deg nights disagree beyond their errors in every view. DOF are the worst view; DZ and v-modes are comparable.
- Worst repeated-position spread: DZ(1,4) 131, DZ(2,5) 64; v4 299, v1 128; DOF 28 (B1_19) 830 (chi2/dof, dimensionless, 2 dof).

### 2026-10-28 — how the paired Δ is derived, DZ(1,4) test case

- New notebook `notebooks/bounce_delta_derivation_v1.ipynb`, parameters K, J and DAY_OBS. It rebuilds the per-night pairs with the library calls and checks that it reproduces the pipeline Δ and error. Run `20261028_delta_derivation_dz1_4`.
- Pairs: each B is paired with the A just before it. Mean B − A lag = 2.6–2.9 min (all 7 samples), always positive. So a drift biases Δ by drift × lag, with a sign that changes from night to night.
- DZ(1,4) drift, Huber fit of A and B visits, um of wavefront per hr: 0418 −0.83 ± 0.14, 0419 −0.51 ± 0.03, 0513 +0.48 ± 0.05. Lag bias: −0.039, −0.024, +0.023 um. The 0418 − 0513 Δ difference is 0.14 um; drift explains about 0.06 um of it.
- 40 deg nights, chi2/dof about their weighted mean (dimensionless, 2 dof): paired median 262; A–B–A interpolation 28; Huber step + linear drift 6.4. The Huber step with no drift term gives 2.3, but only because its errors (0.09–0.11 um) absorb the drift.
- 0419 also has a jump of about −0.4 um at 30–50 min, with residual structure of about 0.2 um before it, so a linear drift does not describe it.

### 2026-10-29 — DZ(1,4) error model: fit error plus coherent turbulence

New notebook `notebooks/bounce_delta_derivation_v2.ipynb`, run `20261029_delta_derivation_v2_dz1_4`. The per-visit error is the DZ fit error plus SIGMA_TURB_UM = 0.04 µm of wavefront in quadrature. The notebook propagates it through the four estimators by Monte Carlo (1000 realisations, white noise only).

- DZ(1,4) fit error: median 0.00044 µm (Huber RLM, 3300 donuts per visit), so 0.011 of SIGMA_TURB_UM (dimensionless). Turbulence sets the error.
- CWFS minus FAM Z4 at the corners (`d13-blitz_unpaired`): median robust covariance between sensors 0.0106 µm², which gives 0.073 µm per exposure if the CWFS and FAM terms are independent.
- Per-visit scatter from same-setting differences grows with time lag: 0.048 µm at 6 min, 0.17 µm at 22 min and 0.33 µm at 45 min of lag. This is red noise, not white. The B−A pairs (2.8 min lag) scatter by 0.022 µm per visit.
- Model errors on the paired median are 0.021–0.030 µm, against the data errors of 0.007–0.025 µm.
- With model errors, chi2/dof of the three 40 deg nights (2 dof) is 33 for the paired median, 60 for the step fit, 13 for step + drift and 16 for A–B–A.
- The extra error per night that brings chi2/dof to 1 is 0.090 µm for the paired median, 0.047 µm for step + drift and 0.061 µm for A–B–A.
