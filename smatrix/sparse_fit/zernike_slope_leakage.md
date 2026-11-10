# The Smatrix does not predict the dense 21-term Danish fit

> **Status:** current · **Last updated:** 2026-11-10 · **Kind:** finding

The AOS (active optics system) estimates the optical state from donut wavefronts using a
sensitivity matrix (Smatrix). The Smatrix gives the change of each pupil Zernike (Noll
index) per unit degree of freedom (DOF), as a projection of the in-focus wavefront.
Danish, the ts_wep donut estimator, measures something else. It fits the donut image
with a set of terms free and every other term held at the reference model. This note
compares the two for the default dense fit, 21 terms: Noll 4–19 and 22–26.

## Size of the mismatch

The test set is 26 configurations: 8 corner half-chips and 9 science points (field angle
−0.75, 0 and +0.75 deg in each of x and y), at both defocus sides, in r band. At each,
batoid ray-traces the donut of the perturbed telescope (with spider vanes), Danish fits
it, and A_rt is the change in the fit per unit DOF. A_Smatrix is the Smatrix row, the
in-focus change on the same 21 terms.

All columns are dimensionless:
- relative difference is |A_rt − A_Smatrix| / |A_rt|, shown as corners (n = 8) |
  science points (n = 18);
- the magnitude ratio is |A_rt| / |A_Smatrix|;
- the cosine is the cosine of the angle between the two vectors.

| DOF group | relative difference | magnitude ratio | cosine |
|---|---|---|---|
| M2 hexapod | 0.075 \| 0.050 | 1.02 | 0.999 |
| camera hexapod | 0.069 \| 0.051 | 1.01 | 0.999 |
| M1M3 B1–B10 | 0.13 \| 0.09 | 0.94 | 0.998 |
| M1M3 B11–B20 | 0.96 \| 1.19 | 0.92 | 0.42 |
| M2 B1–B10 | 0.14 \| 0.12 | 0.93 | 0.998 |
| M2 B11–B20 | 0.94 \| 0.98 | 1.56 | 0.30 |

Values are medians over the configurations. The relative differences are medians over
the DOFs in each group; magnitude ratio and cosine are medians over all 26.

Group medians hide the spread. Median relative difference over the 26 configurations
(dimensionless), per bending mode:

| mode | B1–B5 | B6, B7 | B8, B9 | B10, B11 | B12, B13 | B14–B20 |
|---|---|---|---|---|---|---|
| M1M3 | 0.04–0.10 | 0.49, 0.50 | 0.08 | 1.90, 1.83 | 0.14, 1.01 | 0.94–1.87 |
| M2 | 0.04–0.12 | 0.78, 0.79 | 0.11 | 1.06, 1.07 | 0.56, 0.51 | 0.44–1.17 |

- **Good (5–14%):** rigid body, B1–B5 and B8–B9 of both mirrors, and M1M3 B12. The fit
  measures the Smatrix direction, but its size is off by up to 14%.
- **Bad (0.4–1.9):** B6, B7, B10, B11, B13–B20 of both mirrors, and M2 B12. The fit
  measures a substantially different vector.

## Mechanism, with M1M3 B11 on axis

- **Wavefront change:** per µm of B11 it is Z6 +2.55, Z12 −1.81, Z24 +0.45 and Z38
  −0.09 µm. Only Z38 is outside the 21 terms.
- **What the fit returns:** Z6 = −0.20 µm per µm, against +2.55 in the Smatrix.
- **Why:** a pure Z38 injection moves the fitted terms by Z6 +26, Z12 −5.4 and Z24 +2.1
  µm per µm. So B11's −0.09 µm of Z38 cancels nearly all of its Z6.

Danish fits ray slopes, which set where light lands in the donut. Zernikes with the same
azimuthal order m and different radial order n are orthogonal over the pupil, but their
slopes are not. An unfitted high-n term therefore leaks into the fitted low-n terms of
the same m, and the leak grows with n. Most bending modes put some wavefront above Noll
26, so their fitted response picks up these leaks.

Donuts drawn with Danish's own forward model show the same thing: for B11, fitted Z12 is
−1.29 µm per µm against −1.26 ray-traced. So this is how the estimator behaves, not an
artefact of the image model.

## Impact

The AOS treats the Danish Zernikes as if they were the Smatrix quantity.
- **Rigid body, B1–B5, B8–B9:** estimated DOFs carry gain errors of 5–14%.
- **B6, B7, B10–B20:** the Smatrix columns do not describe the measured response, so the
  estimates for those modes, and their mixing with others, are wrong.

How much reaches the correction depends on which modes the state estimator keeps after
its truncated singular value decomposition (SVD) and normalization; that has not been
evaluated here. The fix is an Smatrix built from Danish fits of ray-traced donuts, which
puts the leaks into the matrix.

**Precision:** responses at twice the DOF step agree to 3–8% (median, dense fit; worst
M2 B16 19%), so the smaller differences above carry an uncertainty of a few percent.

Runs and code: `20261110_bruteforce_check` and `20261110_diag_b11` under `output/`;
`code/donut_leakage.py` and `code/analyze_bruteforce.py`. Details are in the 2026-11-10
entries of the [study Log](study.md).
