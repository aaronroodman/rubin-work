---
study: momfit_accuracy
topic: optatmo
status: done
question: "Can the analytic phase-gradient moments of Zanmar Sanchez et al. (SPIE 2026) replace GalSim plus HSM moments inside an Optics+Atmosphere PSF fit, including the 3rd and 4th orders?"
answer: "Only at 2nd order, with a calibration slope (e0 slope 0.52, R^2 0.93, dimensionless). 3rd and 4th-order moments are dominated by diffraction (trefoil skew is 100% diffractive), so no geometric model reproduces them; a JAX Fraunhofer forward model was built instead."
uses: {}  # no rubin-work products; simulation
started: 2026-07-09
---

# `momfit_accuracy`: analytic moments against HSM weighted moments

> **Write-up:** [`findings.md`](findings.md)

## Question

Can the analytic phase-gradient moments of Zanmar Sanchez et al. (SPIE 2026) replace GalSim plus HSM moments inside an Optics+Atmosphere PSF fit, including the 3rd and 4th orders?

## Current answer

- **2nd order works with a calibration.** Analytic to HSM slope 0.52 (dimensionless) with R² = 0.93 for size e0; e1 and e2 slopes 0.37 and 0.32, degrading with aberration amplitude (e1 R² from 0.88 to 0.67). Gaussian weighting explains most of the slope.
- **3rd and 4th orders are diffractive.** Trefoil skew is zero geometrically and entirely diffractive; coma's 3rd moment flips sign and shrinks by 0.27× (dimensionless). Slopes are 0.02 to 0.21 (dimensionless) with R² ≤ 0.70.
- So the analytic method is only a fast 2nd-order initializer. The JAX `Fraunhofer × atmosphere × HSM` forward model built next reproduces galsim/PIFF to < 1% (relative, per moment).

Details in [`findings.md`](findings.md).

## Log

### 2026-07-09 — backfilled from docs and history

Backfilled 2026-10-18 from `findings.md` (formerly `optatmo/docs/findings.md`) and `optatmo/README.md`, phase 1. Done in the phase 5 triage. The simulation scripts named in `findings.md` (`moments_analytic.py`, `evaluate.py`, `render_geometric.py`, `decompose_modes.py`, `psf_model.py`) are not tracked in this repository.
