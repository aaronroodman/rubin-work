# Rubin AOS bounce tests: elevation sweep 30–75 deg and rotator 0→60 deg

> **Status:** current · **Last updated:** 2026-10-02 · **Kind:** outward-facing note (results summary)

**Summary** A *bounce test* slews the telescope away from a reference position and back, so
that the difference between the two Full Array Mode (FAM) wavefront measurements isolates
gravity-driven flexure and hysteresis from measurement noise. Two BLOCKs supply the data:
**BLOCK-T720** moves in elevation, **BLOCK-T724** in camera rotator angle. Six BLOCK-T720
nights, three of them from July 2026, together make an elevation *sweep* against a fixed
reference at elevation 70 deg, with downward throws to 60, 50, 40 and 30 deg and one upward
throw to 75 deg. The wavefront change grows monotonically with throw: expressed as the
equivalent point-spread-function (PSF) full width at half maximum (FWHM), the elevation
throw produces 0.0908 arcsec FWHM at a 5 deg throw rising to 0.3987 arcsec FWHM at a 40 deg
throw. In every case the Optical Feedback Control (OFC) 50-degree-of-freedom /
34-v-mode correctable subspace removes most of it, leaving 0.0153 to 0.0664 arcsec FWHM of
uncorrectable residual. The rotator bounce gives 0.2082 arcsec FWHM before correction and
0.0191 arcsec FWHM after. So the bounce signal is large, highly significant, and almost
entirely correctable by the active optics system — which is what a Look-Up Table (LUT)
needs in order to remove it feed-forward.

One qualification, which turns out not to weaken that conclusion. The default truncated recovery
returns mirror bending-mode amplitudes up to 11.9 times the range the mirrors can physically
apply (dimensionless, recovered amplitude over allowed range). Re-inverting the same data with a
penalty that bounds every degree of freedom to its range costs a median of only 0.0027 arcsec
FWHM, within a range of −0.0002 to +0.0158 arcsec over the nine (leg, night) measurements. The
unphysical amplitudes therefore carry almost none of the wavefront: the correction the bounce
calls for is not merely correctable in principle but **physically reachable**.

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

The single most quotable number. The median Δ-DZ vector is evaluated at field points across
the focal plane; at each point the resulting pupil-Zernike vector is converted to a
per-Zernike arcsec PSF FWHM contribution (Noll 4 and up) and those are quadrature-summed, and
the median over the focal plane is reported. Applied to the Δ itself this is `fwhm_before`.
Applied to the residual a correction leaves behind it gives `fwhm_after_50_34` — the part of
the bounce the active optics system could *not* remove. For the rotator bounce, where only the
camera hexapod physically moves, a 5-DOF / 5-v-mode camera-hexapod-only correction is also
evaluated as `fwhm_after_5_5`.

**Zero means no differential aberration**, not a perfect PSF and not the seeing floor: these
numbers are the AOS wavefront term only, and add in quadrature on top of the delivered image
quality. Full definition, including why the "after" series use the achieved residual
`dW − S·(d/w)` rather than the subspace projection, in
`aos/docs/studies/bounce.md`, section "How the correctable FWHM is computed, and what zero
means".

## The data

Six BLOCK-T720 nights and two BLOCK-T724 nights, all from the Danish 1.2 FAM processing.
Visit counts are from the MIW-referenced fit, after the per-visit quality cut: at least 160
CCDs carrying enough donuts, and `median_blur_arcsec` at most 1.2 arcsec.

| day_obs | BLOCK | reference leg | comparison legs present | n visits |
|---|---|---|---|---|
| 20260418 | T720 | elev 70 deg (6) | elev 40 deg (6) | 12 |
| 20260419 | T720 | elev 70 deg (9) | elev 40 deg (8) | 17 |
| 20260513 | T720 | elev 70 deg (8) | elev 40 deg (8) | 16 |
| 20260709 | T720 | elev 70 deg (6) | elev 60 deg (4) | 10 |
| 20260711 | T720 | elev 70 deg (8) | elev 50 deg (6) | 14 |
| 20260713 | T720 | elev 70 deg (12) | elev 30 deg (5), elev 75 deg (6) | 23 |
| 20260420 | T724 | rotator −1 deg (12) | rotator 59 deg (12) | 24 |
| 20260513 | T724 | rotator −1 deg (19) | rotator 59 deg (19) | 38 |

The elevation 30 deg leg is a single-night measurement on 20260713. 20260711 also pointed to
elevation 30 deg for two visits, but both fail the blur cut at `median_blur_arcsec` of 1.525
and 1.934 arcsec, so the 30 deg leg carries no 20260711 pair at all. That is atmospheric
seeing, not a donut-count problem: both visits have 178 CCDs above the donut threshold. One of
20260713's six 30 deg visits fails the same cut at 1.799 arcsec, which is why that leg shows 5
comparison visits rather than 6.

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

All legs from the measured-intrinsic-wavefront (MIW) referenced fit, which now covers the July
nights and has the most pairs on every leg. All values are PSF FWHM in arcsec. The
batoid-intrinsic values are within 0.002 arcsec FWHM on every leg except the 40 deg one, where
the two tables select different visit sets (see Caveats).

| bounce | leg | throw (deg) | n pairs | FWHM before | FWHM after 50/34 | FWHM after 5/5 |
|---|---|---|---|---|---|---|
| T720 elevation | elev 75 deg | +5 (upward) | 6 | 0.0908 | 0.0153 | — |
| T720 elevation | elev 60 deg | −10 | 4 | 0.1409 | 0.0326 | — |
| T720 elevation | elev 50 deg | −20 | 6 | 0.2516 | 0.0664 | — |
| T720 elevation | elev 40 deg | −30 | 22 | 0.2986 | 0.0381 | — |
| T720 elevation | elev 30 deg | −40 | 5 | 0.3987 | 0.0598 | — |
| T724 rotator | rotator 60 deg | 60 deg in rotator | 31 | 0.2082 | 0.0191 | 0.0482 |

The FWHM before correction rises monotonically with the magnitude of the elevation throw,
from 0.0908 arcsec FWHM at 5 deg to 0.3987 arcsec FWHM at 40 deg — the behaviour expected of
a gravity-driven flexure that grows with the change in the gravity vector. The correctable
subspace removes 74% to 87% of it in FWHM terms across the elevation legs (least on the
50 deg leg, most on the 40 and 30 deg legs), and 91% on the rotator leg.

The rotator bounce is instructive on the 5-DOF question: allowing all 50 DOF leaves 0.0191
arcsec FWHM, while restricting the correction to the five camera-hexapod DOF that actually
moved leaves 0.0482 arcsec FWHM. The rotator bounce therefore changes the wavefront in ways
the camera hexapod alone cannot undo, by a factor
`fwhm_after_5_5 / fwhm_after_50_34 = 2.53 (dimensionless; camera-hexapod-only residual over
full 50-DOF residual)`.

Achieved correctable FWHM in arcsec per leg, pooled over nights, for all the recovery schemes
evaluated — the 50-DOF / 34-v-mode default, Range-Bounded Recovery (RBR), the reduced
22-DOF / 12-v-mode set, the quadratic motion penalty `ts_ofc` ships (OIC), and the
camera-hexapod-only 5-DOF / 5-v-mode scheme on the rotator bounce:

| leg | before | 50/34 | RBR | 22/12 | OIC | 5/5 |
|---|---|---|---|---|---|---|
| elev 75 deg | 0.0908 | 0.0153 | 0.0151 | 0.0302 | 0.0457 | — |
| elev 60 deg | 0.1409 | 0.0326 | 0.0357 | 0.0563 | 0.0666 | — |
| elev 50 deg | 0.2516 | 0.0664 | 0.0715 | 0.1193 | 0.1347 | — |
| elev 40 deg | 0.2986 | 0.0381 | 0.0414 | 0.0933 | 0.1519 | — |
| elev 30 deg | 0.3987 | 0.0598 | 0.0756 | 0.1344 | 0.2169 | — |
| rotator 60 deg | 0.2082 | 0.0191 | 0.0203 | 0.0446 | 0.1417 | 0.0482 |

The ordering is the same on every leg: the unconstrained 50/34 recovery is best, RBR costs
almost nothing over it, and both restricting the DOF set (22/12) and the quadratic penalty
(OIC) cost substantially more. What each scheme costs and why is in
`aos/docs/studies/bounce.md`, section "Four recovery schemes at every bounce point".

### Significance and the largest coefficients

Number of DZ coefficients failing the null out of 126, and the largest individual Δ, per leg
(MIW-referenced fit). Significance is dimensionless, `|Δ| / error`.

| leg | n fail / 126 | RMS(Δ) over (k, j) (µm) | largest Δ (µm of wavefront) |
|---|---|---|---|
| elev 75 deg | 7 | 0.0155 | k=1 j=6 (astigmatism) −0.1273 ± 0.0256, significance 5.0 |
| elev 60 deg | 9 | 0.0247 | k=1 j=6 (astigmatism) +0.1897 ± 0.0368, significance 5.2 |
| elev 50 deg | 15 | 0.0378 | k=1 j=6 (astigmatism) +0.2482 ± 0.1380, significance 1.8 |
| elev 40 deg | 37 | 0.0410 | k=1 j=7 (coma) +0.3012 ± 0.0115, significance 26.3 |
| elev 30 deg | 42 | 0.0499 | k=1 j=4 (defocus) +0.3146 ± 0.0170, significance 18.5 |
| rotator 60 deg | 30 | 0.0260 | k=1 j=8 (coma) −0.1837 ± 0.0049, significance 37.3 |

The 50 deg leg is the one case where the largest Δ is not itself significant: its
astigmatism error of 0.1380 µm of wavefront is an order of magnitude larger than on the
other legs, from only 6 pairs on a single night with unusually large pair-to-pair scatter.
Its 15 failing coefficients come from smaller, better-determined terms.

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

| leg | n DOF over 3σ / 50 | three largest DOF changes |
|---|---|---|
| elev 30 deg | 27 | M2_dy +981 ± 65 µm, Cam_dy −802 ± 52 µm, Cam_dx −684 ± 42 µm |
| elev 40 deg | 29 | M2_dy +764 ± 44 µm, Cam_dy −561 ± 39 µm, Cam_dx −491 ± 34 µm |
| elev 50 deg | 10 | M2_dy +841 ± 149 µm, B2_2 +0.1201 ± 0.0243 µm, B2_4 +0.0527 ± 0.0082 µm |
| elev 60 deg | 7 | M2_dy +349 ± 65 µm, B2_17 +0.0223 ± 0.0063 µm, B2_12 −0.0187 ± 0.0043 µm |
| elev 75 deg | 11 | M2_dy −242 ± 23 µm, Cam_dy +236 ± 74 µm, B2_1 +0.0424 ± 0.0126 µm |
| rotator 60 deg | 31 | Cam_dx +951 ± 24 µm, Cam_dy +616 ± 35 µm, M2_dx +269 ± 24 µm |

The recovered M2 and camera hexapod lateral decentres are the largest terms on every elevation
leg, and **M2_dy changes sign between the downward throws and the upward 75 deg throw** —
+981 µm at elevation 30 deg against −242 µm at 75 deg, as a gravity-driven decentre must. Their
magnitudes are large in µm but this is the *recovered optical state* of an open-loop
measurement, not a commanded motion; the correctable-FWHM numbers above are the statement of
how much image quality is at stake.

For the rotator bounce the same Δ restricted to the 5-DOF / 5-v-mode camera-hexapod-only
scheme — the scheme in which this result is used, since only the camera rotator moved — gives
all five camera-hexapod axes significant:

| DOF | Δ | significance |
|---|---|---|
| Cam_dx | +998.1 ± 31.7 µm | 31.5 |
| Cam_dy | +648.6 ± 34.8 µm | 18.6 |
| Cam_dz | −7.352 ± 0.603 µm | 12.2 |
| Cam_ry | −0.003517 ± 0.000105 arcsec | 33.5 |
| Cam_rx | +0.000389 ± 0.000098 arcsec | 4.0 |

The lateral decentres Cam_dx and Cam_dy dominate, and the recovered values agree with the
full 50-DOF solution to within about 5% in amplitude (+951 versus +998 µm in Cam_dx), so the
camera-hexapod part of the rotator bounce is robust to how many DOF the recovery is allowed.
What the 5/5 scheme cannot capture is the rest — hence the larger `fwhm_after_5_5` residual
above.

### The recovered bending-mode amplitudes exceed what the mirrors can apply

Each degree of freedom has an allowed range `r_j` — the actuator-force-limited stroke for a
mirror bending mode, the `rb_stroke` limit for a hexapod axis — and it enters the recovery only
through the normalization weight `w_j = sqrt(r_j / f_j)`, where `f_j` is the full width at half
maximum (FWHM) response in arcsec per unit of that degree of freedom. That puts the range into
the *metric* of the fit but never into the *feasible set*: nothing in a truncated singular value
decomposition (SVD) stops it returning an amplitude a mirror physically cannot reach.

It does. Taking the ratio `abs(Δ)/r_j` (dimensionless, recovered amplitude over allowed range)
over all 50 degrees of freedom, per (leg, night):

| leg | night | largest ratio | DOF over range, of 50 |
|---|---|---|---|
| elev 30 deg | 20260713 | 11.70 | 15 |
| elev 40 deg | 20260418 | 11.93 | 16 |
| elev 40 deg | 20260419 | 10.63 | 14 |
| elev 40 deg | 20260513 | 5.95 | 14 |
| elev 50 deg | 20260711 | 5.46 | 13 |
| elev 60 deg | 20260709 | 3.15 | 8 |
| elev 75 deg | 20260713 | 1.40 | 3 |
| rotator 60 deg | 20260420 | 3.77 | 12 |
| rotator 60 deg | 20260513 | 3.15 | 11 |

The excursions sit in the high-order bending modes, not the rigid-body axes: B1_20 reaches
−0.0258 ± 0.0032 µm of mode amplitude against a range of 0.00221 µm at elevation 30 deg, at
significance 8.0, while the hexapod terms that dominate the wavefront stay far inside their
ranges. B1_20 is the largest excursion on the elevation 30, 40 and 75 deg legs; the rotator
bounce's is B1_11 at −0.0415 ± 0.0022 µm, significance 19.0, against a range of 0.01320 µm. The ratio grows
monotonically with throw and falls to about 1 on the near-null upward 75 deg leg, which is the
signature of an inversion artifact that scales with the signal rather than of a real mirror figure
change.

### Range-Bounded Recovery bounds them at a small cost in image quality

To settle whether those amplitudes carry any wavefront, the same Δ is inverted a second way.
**Range-Bounded Recovery (RBR)** keeps the least-squares fit but adds a per-degree-of-freedom
penalty that is negligible inside the allowed range and climbs steeply as an amplitude
approaches and passes it, with two dimensionless knobs set to `kappa = 4` and `p = 3`. The
penalty form, the choice of knobs and why RBR is applied per pair rather than to the median are
in `aos/docs/studies/bounce.md`, section "What RBR does".

The comparison is scored on the residual the correction **achieves**, `dW − S·(d/w)`, rather than
on the correctable-subspace projection used for `fwhm_after_50_34` in the table above — the
projection does not depend on the recovered amplitudes at all, which is why the over-range
amplitudes never showed up in the FWHM numbers. The two agree to 1.7e-16 arcsec FWHM for the
truncated solution, so the default column below reproduces the earlier result.

| leg | night | largest ratio, default → RBR | DOF over range, default → RBR | achieved FWHM, default → RBR (arcsec) | FWHM cost (arcsec) |
|---|---|---|---|---|---|
| elev 30 deg | 20260713 | 11.70 → 1.16 | 15 → 2 | 0.0598 → 0.0756 | +0.0158 |
| elev 40 deg | 20260418 | 11.93 → 1.04 | 16 → 1 | 0.0463 → 0.0489 | +0.0027 |
| elev 40 deg | 20260419 | 10.63 → 1.04 | 14 → 1 | 0.0487 → 0.0492 | +0.0005 |
| elev 40 deg | 20260513 | 5.95 → 1.03 | 14 → 1 | 0.0354 → 0.0459 | +0.0105 |
| elev 50 deg | 20260711 | 5.46 → 0.98 | 13 → 0 | 0.0664 → 0.0715 | +0.0052 |
| elev 60 deg | 20260709 | 3.15 → 0.76 | 8 → 0 | 0.0326 → 0.0357 | +0.0032 |
| elev 75 deg | 20260713 | 1.40 → 0.56 | 3 → 0 | 0.0153 → 0.0151 | −0.0002 |
| rotator 60 deg | 20260420 | 3.77 → 1.11 | 12 → 2 | 0.0247 → 0.0273 | +0.0026 |
| rotator 60 deg | 20260513 | 3.15 → 0.88 | 11 → 0 | 0.0185 → 0.0199 | +0.0014 |

Over the nine (leg, night) points the cost in achieved correctable FWHM has a median of
+0.0027 arcsec, within a range of −0.0002 to +0.0158 arcsec. Counting per-(leg, night) rows
across all legs, 106 of 450 degrees of freedom exceed their range under the default recovery
against 7 under RBR.

**So the over-range amplitudes carry almost none of the wavefront.** Removing them entirely costs
a few percent of the correction residual and well under 2% of the uncorrected FWHM the bounce
produces — they are an ill-conditioning artifact of the unconstrained inversion, not a real
high-order mirror figure change. A physically reachable correction for these bounce flexures
exists and delivers essentially the same image quality. The worst case is the largest throw
(elevation 30 deg, +0.0158 arcsec) and the near-null upward 75 deg leg is very slightly *better*
under RBR, which is what one expects when the discarded amplitude was noise.

Since the penalty is smooth rather than a hard bound, a few amplitudes still finish just outside
range — by at most a factor of 1.157. One caveat on reading individual axes: RBR biases toward
zero by construction and the rigid-body split is not uniquely pinned by these data (M2_dz can
flip sign between the two solutions, while the rigid-body *wavefront* is preserved to within a few
percent). **An RBR rigid-body amplitude is a constrained estimate and should not be quoted as a
measurement of hexapod motion.** For a look-up-table fit, which wants the amplitudes themselves,
the default recovery remains the estimator; RBR answers whether a reachable correction exists and
what image quality it delivers.

### Night-to-night repeatability

Only the elevation 40 deg leg is exercised on more than one night: each July night throws to a
different elevation, so the 30, 50, 60 and 75 deg legs are each a single-night measurement.
Comparing the 126 per-(k, j) Δ values night against night on the 40 deg leg:

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

**The intrinsic choice is measured on every leg, and is not a limiting systematic.** Two DZ
fit tables are available: the Phase-1 table, fit against the *batoid* design intrinsic, and a
MIW-referenced table. Any intrinsic that is fixed in the fitting frame cancels exactly in a
paired Δ, so the telescope-fixed **O** component of the intrinsic drops out and only the
camera-fixed **C** component, which rotates with the camera rotator, can bias a result. For
an elevation bounce at fixed rotator near 0 deg, C is essentially static within a pair, so the
choice should barely matter.

That argument is now a measurement on all six legs, the MIW-referenced fits having been
extended over the July nights (the MIW *build* stays frozen at `day_obs` 20260513; only the
fit that references it was run over the wider night range). Comparing the two tables' Δ per
(k, j), with `n = 126` DZ coefficients on each leg:

| leg | median difference (µm) | nmad (µm) | RMS(Δ), MIW (µm) | nmad/RMS | Pearson r | Spearman rho |
|---|---|---|---|---|---|---|
| elev 75 deg | −0.00002 | 0.00024 | 0.0155 | 0.016 | 0.9994 | 0.969 |
| elev 60 deg | +0.00006 | 0.00038 | 0.0247 | 0.015 | 0.9995 | 0.989 |
| elev 50 deg | −0.00003 | 0.00034 | 0.0378 | 0.009 | 0.9998 | 0.992 |
| elev 40 deg | +0.00012 | 0.00128 | 0.0410 | 0.031 | 0.9224 | 0.887 |
| elev 30 deg | +0.00002 | 0.00043 | 0.0499 | 0.009 | 0.9998 | 0.986 |
| rotator 60 deg | −0.00007 | 0.00135 | 0.0260 | 0.052 | 0.9811 | 0.797 |

The `nmad/RMS` column is
`nmad(intrinsic difference)/RMS(Δ) (dimensionless; intrinsic-choice scatter over per-leg
signal amplitude)`. On the four single-night elevation legs, where both tables select the same
visits, the intrinsic contributes 0.9% to 1.6% of the signal amplitude with Pearson
r ≥ 0.999. The 40 deg leg is larger at 3.1% only because the two tables select *different*
visit sets there — the MIW fit retains 22 visits against the Phase-1 table's 10 — so that row
mixes the intrinsic choice with a genuine change in sample. On the BLOCK-T724 rotator bounce,
where C *does* rotate within a pair, the difference is largest at 5.2%, as expected, and still
well below the signal. The intrinsic choice is therefore not a limiting systematic for either
bounce at present precision, and the elevation-sweep conclusions hold under both intrinsics.

**The 75 deg leg is an upward throw and acts as a near-null control.** It moves only 5 deg
from the reference, and upward rather than downward. Its 0.0908 arcsec FWHM before correction
is the smallest of the five legs and its DOF changes carry the opposite sign, both as
expected. It is useful as a consistency check, not as a measurement of flexure at high
elevation.

**The blur cut, not the donut count, is what removes visits.** The per-visit quality cut has two
active parts, and on these nights it is the seeing that bites. Every BLOCK-T720 visit on all six
nights has 176–180 CCDs carrying enough donuts, comfortably above the 160-CCD floor, so nothing
is lost to donut counts. What is lost is three visits above `median_blur_arcsec` of 1.2 arcsec:
both of 20260711's excursions to elevation 30 deg (1.525 and 1.934 arcsec) and one of
20260713's six (1.799 arcsec). The visible consequence is that the 30 deg leg is a single-night
measurement with 5 pairs rather than a two-night one with 8.

**Pair counts are small on the new legs** — 4 to 6 pairs per July leg, against 22 pairs on the
pooled 40 deg leg. The errors quoted already reflect this, but a single night's leg is not a
repeatability measurement, and the monotonic throw trend is the stronger evidence than any one
leg taken alone.

## Provenance

| item | value |
|---|---|
| FAM processing | Danish 1.2, `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` |
| batoid-intrinsic fits | `aos/output/fam_processing/danish_1_2/fits.parquet` |
| MIW-referenced fits (lead result) | `aos/output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet` — 3385 visits, `day_obs` 20250415–20260713, against MIW build `pathA_50_34_i_5rot` |
| MIW build | frozen at `day_obs` 20260513; the fit above references it over the wider night range without rebuilding it |
| bounce outputs, MIW | `aos/output/bounce/danish_1_2_A_50_34_i_5rot_july/` — every number in this note |
| bounce outputs, batoid | `aos/output/bounce/danish_1_2_batoid/` — the intrinsic-choice comparison |
| DZ grid | focal `k = 1…6`, pupil Noll `j = 4…19, 22…26`, fit prefix `z1toz6`, 126 coefficients |
| OFC subspace | 50 DOF, 34 v-modes kept; sensitivity matrix evaluated at camera rotator angle 0.0 deg |
| quality cut | at least 160 CCDs with enough donuts per visit, and `median_blur_arcsec` at most 1.2 arcsec |
| thresholds | significance 3.5 (dimensionless) with 0.1 µm of wavefront, or significance 5.0 alone |
| Range-Bounded Recovery | `kappa = 4.0`, `power = 3`, both dimensionless; solver in `smatrix/code/regularized_inversion/`, allowed range `r_j` back-derived from the shipped OFC normalization weights |
| code | `aos/code/bounce/run_bounce.py`, `aos/code/bounce/bounce_lib.py`; config in `aos/analysis_config.yaml` under `bounce` |

Two earlier bounce output directories are superseded and were moved to
`aos/output/archive/bounce/` on 2026-09-23: `danish_1_2_A_50_34_i_5rot` (the same MIW build over
April/May nights only) and `danish_1_2_A_50_34_i` (the earlier non-rotated MIW build). Note
`aos/output/miw/danish_1_2_A_50_34_i_5rot/` is a different path and is current — it is the MIW fit
table this note's lead result reads.

See [`../../aos/docs/studies/bounce.md`](../../aos/docs/studies/bounce.md) for the study
reference, and
[`../../smatrix/docs/studies/regularized_inversion.md`](../../smatrix/docs/studies/regularized_inversion.md)
for the derivation and validation of Range-Bounded Recovery. The tables behind these numbers are
`bounce_kj_stats.parquet` (per-(k, j) Δ), `bounce_dof_stats.parquet` (per-DOF and per-v-mode Δ in
all four schemes, with the RBR Δ, the allowed range and the ratios beside the default recovery),
`bounce_fwhm_metric.parquet` and `bounce_fwhm_vs_bvalue.parquet` (the three FWHM series per
(night, leg)); the figures are `bounce_summary.pdf`, `bounce_fwhm_metric.pdf`,
`bounce_fwhm_vs_bvalue.pdf`, `bounce_dof_night_scatter.pdf`, `bounce_dof_night_values.pdf`,
`bounce_dz_vs_ordinal.pdf`, `bounce_vmode_vs_ordinal.pdf`, `bounce_dof_vs_ordinal.pdf` and
`bounce_5x5_camera_hexapod.pdf`.
