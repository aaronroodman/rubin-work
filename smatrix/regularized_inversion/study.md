---
study: regularized_inversion
topic: smatrix
status: active
question: "What image quality does a feasible recovered optical state cost, one in which every DOF sits inside its allowed range, and how do the damped SVD, the range penalty and the ts_ofc OIC motion penalty compare?"
answer: "The range penalty B makes the recovery feasible for 0.0012 to 0.0052 arcsec of FWHM on 5 of 6 bounce legs (3% to 9% of the truncated residual); the OIC needs about 2.24 times more wavefront residual (dimensionless, OIC over RBR; median inflation 2.77 against 1.31) for the same range compliance."
uses: {}  # reads the bounce legs through rubinwork.bounce (the bounce_tables product); the --fits input is the old aos/output MIW fits
started: 2026-09-23
---

# `regularized_inversion`: bounding the recovered optical state to the allowed range

> **Write-up:** [`regularized_inversion.md`](regularized_inversion.md) · **Solvers:** `rubinwork.smatrix.regularized_inversion` (library) · **Runners:** `code/` · **Output:** `smatrix/output/regularized_inversion/<data>/`

## Question

What image quality does a feasible recovered optical state cost, one in which every DOF sits inside its allowed range, and how do the damped SVD, the range penalty and the ts_ofc OIC motion penalty compare?

## Current answer

- The truncated 50/34 recovery puts 1 to 14 of 50 DOF over range per bounce leg, up to 11.27 times the range (dimensionless, amplitude over range).
- **The range penalty B (RBR, `p = 3`, `kappa = 4`) buys feasibility for 0.0012 to 0.0052 arcsec of FWHM on five of six legs**, 3% to 9% of the truncated residual (fraction of FWHM). Elevation 30 deg is the exception at 0.0185 arcsec.
- **The ts_ofc OIC motion penalty is worse:** median residual inflation over truncated is 2.77 for the OIC against 1.31 for RBR (dimensionless), so the OIC gives up about 2.24 times more wavefront for the same nominal compliance. Its implied bending range is 132 to 280 times ours for M1M3, and a single `rho` cannot adapt per leg.

- **50/34 RBR pins the hexapods because truncation forbids pure hexapod motion** (principal angles to 31.8 deg between hexapod subspace and span(V_34)). Opening all 50 modes fixes it with either regularizer: at max `|d_j|/r_j` = 1, RBR-50 residual 0.0530 / 0.0553 µm of wavefront (Elev=40 / Rot=60) against 0.0583 / 0.0564 µm for a full-rank quadratic `H = diag(1/r_j²)`.

Details, tables and the rigid-body degeneracy caveat in [`regularized_inversion.md`](regularized_inversion.md).

## Log

### 2026-10-12 — backfilled from docs and history

Backfilled 2026-10-18 from [`regularized_inversion.md`](regularized_inversion.md) (formerly `smatrix/docs/studies/`) and the commits from 2026-09-23 to 2026-10-12. Converted as active in the phase 5 triage. The solvers are the library `rubinwork.smatrix.regularized_inversion` (shared with the aos bounce study through `rubinwork.bounce`); the two runners moved here from `smatrix/code/regularized_inversion/` and now import the library directly instead of the shim. They still reach `aos/code/aos_fwhm.py` by `sys.path` (a known cross-topic reach). The RBR studies stay separate (triage decision 7).

### 2026-10-19 — RBR against a conventional penalty matrix

`code/run_penalty_matrix_compare.py`, local, two cached bounce legs. Truncation's implied `H` (RSO-217 sec. 3.5) couples hexapod and bending; a pure M2 dz move at 0.2 of range is retained at 0.656 (dimensionless) by RBR-34 and at 1.000 by RBR-50 and full-rank quadratic. RBR-50 has the lowest residual at feasibility; the quadratic is 2% to 10% worse, linear, normalization-independent, and has 12% to 31% lower Cam dx noise. Write-up section "RBR against a conventional penalty matrix".

