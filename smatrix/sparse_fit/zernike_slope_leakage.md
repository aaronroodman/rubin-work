# Unfitted high-order Zernikes leak into the Danish fit

> **Status:** current · **Last updated:** 2026-11-10 · **Kind:** finding

Danish is the donut wavefront estimator in ts_wep. It fits a donut image with a chosen
set of pupil Zernike terms (Noll index) free and all other terms held at the reference
model. The sensitivity matrix (Smatrix) gives the change of each pupil Zernike per unit
degree of freedom (DOF), as a projection of the wavefront. This note shows that the two
differ, using M1M3 bending mode B11 on axis (field angle 0, extra-focal, r band) with the
21-term dense fit set, Noll 4–19 and 22–26.

**M1M3 B11 on axis:**
- **Wavefront change:** per µm of B11, the wavefront changes by Z6 +2.55, Z12 −1.81,
  Z24 +0.45 and Z38 −0.09 µm. Only Z38 is outside the 21 fitted terms.
- **What the fit returns:** Z6 = −0.20 µm per µm, not +2.55.
- **Why:** a pure Z38 injection moves the fitted Z6 by 26 µm per µm. So B11's Z38 of
  −0.09 µm cancels nearly all of its Z6.

**Mechanism:** Danish fits ray slopes, the image of the pupil. Zernikes with the same
azimuthal order m and different radial order n are orthogonal over the pupil, but their
slopes are not. An unfitted high-n term therefore leaks into the fitted lower-n terms of
the same m. The leak grows with n. In the sparse production fit, which frees Z6 but not
Z12, Z24 or Z38, the leaks into Z6 are:

| unfitted term | Z12 | Z24 | Z38 |
|---|---|---|---|
| leak into fitted Z6 (µm per µm) | 5.0 | 8.7 | 17.6 |

**It is the estimator, not the image model.** Donuts drawn by Danish's own forward model
and donuts ray-traced with batoid give the same fit. For B11 with the 21-term fit, fitted
Z12 is −1.29 µm per µm (Danish-drawn) against −1.26 (ray-traced).

**Consequence:** an Smatrix built from wavefront projections does not predict what the
Danish fit measures for DOFs with wavefront outside the fit set. Over the 26 test
configurations, the median relative difference |A_rt − A_Smatrix| / |A_rt|
(dimensionless) is:

| DOF group | relative difference |
|---|---|
| rigid-body | 0.05–0.08 |
| B1–B10 | 0.09–0.14 |
| B11–B20 | about 1 |

A_rt is the ray-traced fit response and A_Smatrix the Smatrix row. The 26 configurations
are 8 corner half-chips and 9 science points, at both defocus sides. For B11–B20 the
response also points in a different direction: the median cosine is 0.3–0.4.

Runs and code: `20261110_bruteforce_check` and `20261110_diag_b11` under `output/`;
`code/donut_leakage.py` and `code/analyze_bruteforce.py`. Details are in the 2026-11-10
entries of the [study Log](study.md).
