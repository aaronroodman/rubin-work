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
