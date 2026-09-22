# Rubin AOS bounce tests: elevation sweep 30–75 deg and rotator 0→60 deg

> **Status:** current · **Last updated:** 2026-09-22 · **Kind:** outward-facing note (results summary)

**Summary** A *bounce test* slews the telescope away from a reference position and back, so
that the difference between the two Full Array Mode (FAM) wavefront measurements isolates
gravity-driven flexure and hysteresis from measurement noise. Two BLOCKs supply the data:
**BLOCK-T720** moves in elevation, **BLOCK-T724** in camera rotator angle. Six BLOCK-T720
nights, three of them from July 2026, together make an elevation *sweep* against a fixed
reference at elevation 70 deg, with downward throws to 60, 50, 40 and 30 deg and one upward
throw to 75 deg. The wavefront change grows monotonically with throw: expressed as the
equivalent point-spread-function (PSF) full width at half maximum (FWHM), the elevation
throw produces 0.0915 arcsec FWHM at a 5 deg throw rising to 0.3958 arcsec FWHM at a 40 deg
throw. In every case the Optical Feedback Control (OFC) 50-degree-of-freedom /
34-v-mode correctable subspace removes most of it, leaving 0.0159 to 0.0674 arcsec FWHM of
uncorrectable residual. The rotator bounce gives 0.2082 arcsec FWHM before correction and
0.0191 arcsec FWHM after. So the bounce signal is large, highly significant, and almost
entirely correctable by the active optics system — which is what a Look-Up Table (LUT)
needs in order to remove it feed-forward.

## What is measured

Each FAM visit pairs intra- and extra-focal donuts on the same CCD and fits the wavefront
with Danish, giving annular Zernike coefficients across the focal plane. The field
dependence of each pupil coefficient is expanded in a **Double Zernike (DZ)** basis —
focal-plane Zernike order `k` times pupil Zernike Noll index `j` — over `k = 1…6` and the 21
pupil indices `j = 4…19, 22…26`, i.e. 126 DZ coefficients `(k, j)` in total, each in µm of
wavefront.

The statistic is a **time-ordered paired difference**

```
Δ(k, j) = median over pairs of [ DZ(k, j)_comparison − DZ(k, j)_reference ]
```

formed only between visits **within one night** (never across a `day_obs` boundary), which
cancels slowly-varying terms such as thermal drift. Its uncertainty is the standard error of
a median, `1.2533 × sigma_mad / sqrt(n_pairs)`, where `sigma_mad` is the median-absolute-
deviation scatter in µm of wavefront and `n_pairs` is the number of pairs.

The same paired Δ is also formed in two derived spaces:

- **OFC v-modes** — the 34 best-constrained left-singular vectors of the normalized AOS
  sensitivity matrix, spanning the optically correctable subspace. Dimensionless in the
  normalized basis.
- **Physical degrees of freedom (DOF)** — the 50-DOF vector of M2 hexapod, camera hexapod,
  M1M3 bending modes and M2 bending modes, in µm for translations and bending-mode
  amplitudes and arcsec for hexapod rotations.

A DZ coefficient is called **significant** (a "fail" of the null hypothesis that the bounce
produces no change) when either `|Δ| / error > 3.5` (dimensionless) **and** `|Δ| > 0.1` µm of
wavefront, or `|Δ| / error > 5.0` (dimensionless) on its own.

### The correctable-FWHM metric

The single most quotable number. The median Δ-DZ vector is converted to an equivalent PSF
FWHM in arcsec by root-sum-square over the DZ coefficients with the standard per-term
wavefront-to-FWHM weights; this is `fwhm_before`. The Δ is then projected onto the OFC
correctable subspace and the residual re-converted, giving `fwhm_after_50_34` — the part of
the bounce the active optics system could *not* remove. For the rotator bounce, where only
the camera hexapod physically moves, a 5-DOF / 5-v-mode camera-hexapod-only correction is
also evaluated as `fwhm_after_5_5`.

## The data

Six BLOCK-T720 nights and two BLOCK-T724 nights, all from the Danish 1.2 FAM processing.
Visit counts are after the quality cut of at least 160 CCDs with enough donuts.

| day_obs | BLOCK | reference leg | comparison legs present | n visits |
|---|---|---|---|---|
| 20260418 | T720 | elev 70 deg (6) | elev 40 deg (6) | 12 |
| 20260419 | T720 | elev 70 deg (9) | elev 40 deg (2 in the batoid fit, 12 in the MIW refit) | 11 (batoid), 24 (MIW refit) |
| 20260513 | T720 | elev 70 deg (8) | elev 40 deg (2 in the batoid fit, 8 in the MIW refit) | 10 (batoid), 16 (MIW refit) |
| 20260709 | T720 | elev 70 deg (6) | elev 60 deg (4) | 10 |
| 20260711 | T720 | elev 70 deg (8) | elev 50 deg (6) | 14 |
| 20260713 | T720 | elev 70 deg (12) | elev 30 deg (5), elev 75 deg (6) | 23 |
| 20260420 | T724 | rotator −1 deg (12) | rotator 59 deg (12) | 24 |
| 20260513 | T724 | rotator −1 deg (17 batoid, 19 MIW refit) | rotator 59 deg (16 batoid, 19 MIW refit) | 33 (batoid), 38 (MIW refit) |

Two features of this table drive the analysis:

- **The July nights are new elevations, not repeats.** No July night visits elevation 40 deg;
  each instead throws to a different elevation. Together with the April/May 40 deg nights they
  turn a single 30 deg throw into a sweep covering elevation 30–75 deg. Each comparison leg is
  a ±3 deg window about its nominal elevation (the 75 deg leg uses 73–78 deg so as not to
  overlap the 67–73 deg reference window).
- **BLOCK-T724 has no July nights**, so the rotator bounce is unchanged from the previous
  analysis. On all six BLOCK-T720 nights the camera rotator stays inside the reference window
  at −1.2 to −0.2 deg, so these are clean elevation-only bounces with no rotator confusion.

All wavefronts are in the Optical Coordinate System (OCS), the telescope-fixed frame.

## Results

### Correctable FWHM per leg

Elevation legs from the batoid-intrinsic fit, which is the only table covering July; the
rotator leg from the measured-intrinsic-wavefront (MIW) refit, which has more pairs. All
values are PSF FWHM in arcsec.

| bounce | leg | throw (deg) | n pairs | FWHM before | FWHM after 50/34 | FWHM after 5/5 |
|---|---|---|---|---|---|---|
| T720 elevation | elev 75 deg | +5 (upward) | 6 | 0.0915 | 0.0159 | — |
| T720 elevation | elev 60 deg | −10 | 4 | 0.1402 | 0.0317 | — |
| T720 elevation | elev 50 deg | −20 | 6 | 0.2524 | 0.0674 | — |
| T720 elevation | elev 40 deg | −30 | 10 | 0.2964 | 0.0456 | — |
| T720 elevation | elev 30 deg | −40 | 5 | 0.3958 | 0.0596 | — |
| T724 rotator | rotator 60 deg | 60 deg in rotator | 31 | 0.2082 | 0.0191 | 0.0482 |

The FWHM before correction rises monotonically with the magnitude of the elevation throw,
from 0.0915 arcsec FWHM at 5 deg to 0.3958 arcsec FWHM at 40 deg — the behaviour expected of
a gravity-driven flexure that grows with the change in the gravity vector. The correctable
subspace removes 73% to 85% of it in FWHM terms across the elevation legs (least on the
50 deg leg, most on the 30 and 40 deg legs), and 91% on the rotator leg.

The rotator bounce is instructive on the 5-DOF question: allowing all 50 DOF leaves 0.0191
arcsec FWHM, while restricting the correction to the five camera-hexapod DOF that actually
moved leaves 0.0482 arcsec FWHM. The rotator bounce therefore changes the wavefront in ways
the camera hexapod alone cannot undo, by a factor
`fwhm_after_5_5 / fwhm_after_50_34 = 2.53 (dimensionless; camera-hexapod-only residual over
full 50-DOF residual)`.

### Significance and the largest coefficients

Number of DZ coefficients failing the null out of 126, and the largest individual Δ, per leg
(batoid-intrinsic fit):

| leg | n fail / 126 | RMS(Δ) over (k, j) (µm) | largest Δ (µm of wavefront) |
|---|---|---|---|
| elev 75 deg | 11 | 0.0156 | k=1 j=6 (astigmatism) −0.1281 ± 0.0238, significance 5.4 |
| elev 60 deg | 10 | 0.0247 | k=1 j=6 (astigmatism) +0.1876 ± 0.0374, significance 5.0 |
| elev 50 deg | 14 | 0.0379 | k=1 j=4 (defocus) +0.1922 ± 0.0635, significance 3.0 |
| elev 40 deg | 26 | 0.0389 | k=1 j=7 (coma) +0.3088 ± 0.0155, significance 19.9 |
| elev 30 deg | 40 | 0.0498 | k=1 j=4 (defocus) +0.3134 ± 0.0160, significance 19.6 |
| rotator 60 deg | 31 | 0.0265 | k=1 j=8 (coma) −0.1805 ± 0.0068, significance 26.5 |

The number of significant coefficients and the RMS of Δ both grow with throw. The signal is
concentrated in the **field-constant term `k = 1`**, i.e. a change uniform across the focal
plane, and within it in defocus (`j = 4`), astigmatism (`j = 5, 6`) and coma (`j = 7, 8`) —
again the signature of bulk flexure and rigid-body motion rather than a high-order figure
change.

### Physical degrees of freedom

Δ in the 50-DOF space, restricted to entries with `|Δ| > 3 × error`, largest first. Labels
follow `lsst.ts.intrinsic.wavefront.ofc_svd.LABELS_50DOF`: `M2_*` and `Cam_*` are the M2 and
camera hexapod rigid-body axes (indices 0–9), `B1_*` are the 20 M1M3 bending modes (10–29) and
`B2_*` the 20 M2 bending modes (30–49). Hexapod translations and bending-mode amplitudes are
µm; hexapod rotations are arcsec.

| leg | dominant DOF changes |
|---|---|
| elev 30 deg | M2_dy +979 ± 60 µm, Cam_dy −808 ± 48 µm, Cam_dx −672 ± 40 µm |
| elev 40 deg | M2_dy +767 ± 73 µm, Cam_dy −616 ± 34 µm, Cam_dx −402 ± 64 µm |
| elev 50 deg | M2_dy +849 ± 134 µm, M2 bending B2_2 +0.120 ± 0.028 µm |
| elev 60 deg | M2_dy +342 ± 71 µm, M2 bending B2_12 −0.019 ± 0.005 µm |
| elev 75 deg | M2_dy −246 ± 27 µm, M2 bending B2_1 +0.046 ± 0.013 µm |

The recovered M2 and camera hexapod lateral decentres are the largest terms and **change sign
between the downward throws and the upward 75 deg throw**, as a gravity-driven decentre must.
Their magnitudes are large in µm but this is the *recovered optical state* of an open-loop
measurement, not a commanded motion; the correctable-FWHM numbers above are the statement of
how much image quality is at stake.

### Night-to-night repeatability

Only the elevation 40 deg leg is exercised on more than one night, and only in the MIW refit
(see the caveat below). Comparing the 126 per-(k, j) Δ values night against night:

| night pair | median difference (µm) | nmad (µm) | Pearson r | Spearman rho |
|---|---|---|---|---|
| 20260418 − 20260419 | +0.0001 | 0.0032 | 0.713 | 0.628 |
| 20260418 − 20260513 | −0.0002 | 0.0034 | 0.737 | 0.545 |
| 20260419 − 20260513 | −0.0001 | 0.0042 | 0.846 | 0.473 |

with `n = 126` DZ coefficients in each comparison, against a per-night Δ signal of
0.0370 / 0.0423 / 0.0451 µm RMS over (k, j) respectively. The night-to-night scatter is
0.0032–0.0042 µm of wavefront against a signal of about 0.04 µm, a ratio
`nmad(night difference)/RMS(Δ) ≈ 0.09 (dimensionless; night-to-night scatter over the
per-night signal amplitude)`, so the elevation bounce reproduces at roughly the 10% level in
amplitude. The two BLOCK-T724 nights agree comparably: median difference −0.0001 µm,
nmad 0.0013 µm, Pearson r = 0.851, Spearman rho = 0.646, `n = 126`, against per-night signals
of 0.0238 and 0.0264 µm RMS.

The Pearson r of 0.71–0.85 rather than near unity reflects that most of the 126 coefficients
carry little signal, so their night-to-night comparison is dominated by noise; the handful of
large coefficients repeat much better than the ensemble correlation suggests.

## Caveats

**The July numbers come from the batoid intrinsic, not the measured intrinsic.** The MIW
refit table is deliberately frozen at `day_obs` 20260513 and has no July rows. The July
results above therefore come from the Phase-1 DZ fit table, which fits against the *batoid*
design intrinsic. Any intrinsic that is fixed in the fitting frame cancels exactly in a
paired Δ, so the telescope-fixed **O** component of the intrinsic drops out and only the
camera-fixed **C** component, which rotates with the camera rotator, can bias a result. For
an elevation bounce at fixed rotator near 0 deg, C is essentially static within a pair, so the
choice should barely matter.

That argument is now a measurement. On 20260418 elevation 40 deg, where both tables select an
identical set of visits, the difference in Δ between the two intrinsics has median
−0.0001 µm of wavefront and nmad 0.0003 µm of wavefront, against a Δ signal of 0.0370 µm RMS
over (k, j) — a ratio of about
`nmad(intrinsic difference)/RMS(Δ) = 0.008 (dimensionless; intrinsic-choice scatter over
signal amplitude)`, i.e. roughly 120× smaller than the signal — with Pearson r = 1.000 and
Spearman rho = 0.983 between the two sets of Δ. On the BLOCK-T724 rotator bounce, where C
*does* rotate within a pair, the difference is larger but still small: nmad 0.0010–0.0015 µm
of wavefront against signals of 0.0238–0.0264 µm RMS, Pearson r ≈ 0.981. The intrinsic choice
is therefore not a limiting systematic for either bounce at present precision. When the MIW
refit is next carried past 20260513 for other reasons, re-running the elevation legs against
it would upgrade the July numbers; nothing in the conclusions above should change.

**The 75 deg leg is an upward throw and acts as a near-null control.** It moves only 5 deg
from the reference, and upward rather than downward. Its 0.0915 arcsec FWHM before correction
is the smallest of the five legs and its DOF changes carry the opposite sign, both as
expected. It is useful as a consistency check, not as a measurement of flexure at high
elevation.

**Two nights drop out of the per-night breakdown of the batoid-intrinsic 40 deg leg.** On
20260419 and 20260513 the Phase-1 table retains only 2 visits at elevation 40 deg (at 170–172
CCDs with enough donuts) where the MIW refit retains 12 and 8 respectively (down to 29 CCDs),
because the MIW refit is run with the quality cut relaxed while the Phase-1 table was cut
upstream. With only 2 visits those nights fall below the 3-visit-per-night floor and are
excluded from the batoid per-night breakdown. This does **not** affect July: all 47 July BLOCK-T720 visits sit at
176–180 CCDs with enough donuts and pass the quality cut, so no July visit is lost. It is the
reason the repeatability table above is quoted from the MIW refit.

**Pair counts are small on the new legs** — 4 to 6 pairs per July leg, against 22 pairs on the
pooled 40 deg leg. The errors quoted already reflect this, but a single night's leg is not a
repeatability measurement, and the monotonic throw trend is the stronger evidence than any one
leg taken alone.

## Provenance

| item | value |
|---|---|
| FAM processing | Danish 1.2, `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` |
| batoid-intrinsic fits | `aos/output/fam_processing/danish_1_2/fits.parquet` |
| MIW refit fits | `aos/output/miw/danish_1_2_A_50_34_i_5rot/fits.parquet` (MIW build `pathA_50_34_i_5rot`, frozen at `day_obs` 20260513) |
| bounce outputs | `aos/output/bounce/danish_1_2_batoid/`, `aos/output/bounce/danish_1_2_A_50_34_i_5rot/` |
| DZ grid | focal `k = 1…6`, pupil Noll `j = 4…19, 22…26`, fit prefix `z1toz6` |
| OFC subspace | 50 DOF, 34 v-modes kept; sensitivity matrix evaluated at camera rotator angle 0.0 deg |
| quality cut | at least 160 CCDs with enough donuts per visit |
| thresholds | significance 3.5 (dimensionless) with 0.1 µm of wavefront, or significance 5.0 alone |
| code | `aos/code/bounce/run_bounce.py`, `aos/code/bounce/bounce_lib.py`; config in `aos/analysis_config.yaml` under `bounce` |

See [`../../aos/docs/studies/bounce.md`](../../aos/docs/studies/bounce.md) for the study
reference, and `aos/output/bounce/*/plots/bounce_summary.pdf`,
`bounce_fwhm_metric.pdf`, `bounce_dof_night_scatter.pdf` and `bounce_dof_night_values.pdf`
for the figures behind these numbers.
