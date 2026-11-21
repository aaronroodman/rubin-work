---
study: v4speed_mask_vs_batoid
topic: aos
status: answered
question: With the wavefront fixed at the Batoid v4_speed TA Zernikes (no fit), how well does the Danish forward model with the v4_speed obscuration file match Batoid-traced donuts, by defocus configuration (FAM, CWFS, giant), side and field position?
answer: Danish (v4_speed obscuration, spider_angle 0 deg) at the batoid TA wavefront misplaces a median 1.8% of the flux on FAM donuts, 2.2-2.5% on CWFS and 3.3% on giant donuts, flat to 1.4 deg and up to 6-8% at the outer CWFS corners; the giant pupil edges do not match (fitted blur 1.5 arcsec at a true 1.0 arcsec).
uses:
  batoid_intrinsic: v4_speed@20261121
  sim_donuts: v4_speed_mono@20261120
started: 2026-11-20
---

# `v4speed_mask_vs_batoid`: v4_speed: Danish obscuration file against the Batoid model

## Question

With the wavefront fixed at the Batoid v4_speed TA Zernikes (no fit), how well does the Danish forward model with the v4_speed obscuration file match Batoid-traced donuts, by defocus configuration (FAM, CWFS, giant), side and field position?

## Current answer

At the batoid TA wavefront, with no wavefront fit, Danish with the v4_speed obscuration
file misplaces a median 1.8% of the flux on FAM donuts, 2.2-2.5% on CWFS and 3.3% on
giant donuts (l1, dimensionless; spider_angle 0 deg). It is flat to 1.4 deg field
radius, then rises to 6-8% at the outer CWFS corners. The giant-donut pupil edges do not
match: a nuisance fit inflates the blur to 1.5 arcsec against a true 1.0 arcsec.

## Log

### 2026-11-21 — pass 1: Danish at the TA wavefront against the sims

Danish, with the v4_speed obscuration file, drawn at the `batoid_intrinsic` TA wavefront
(CCS, Noll 1-78) against all 998 `sim_donuts` stamps (`v4_speed_mono@20261120`: 770 FAM,
144 CWFS, 84 giant; r band, rotator 0 deg, 1.0 arcsec Kolmogorov). Code `90d0727`; runs
`20261121_ta_model` and `20261121_blur_check` (`20261121_smoke` superseded).

Metrics: l1 is the fraction of the flux in the wrong place, half the summed |sim - model|
of unit-sum images (dimensionless). "Fit" frees only flux, dx, dy and fwhm; the wavefront
stays at the reference.

| config | side | l1 no fit, spider 0 deg | l1 no fit, spider None | l1 fit (median / max) | fitted FWHM (arcsec) |
|---|---|---|---|---|---|
| fam | extra / intra | 0.0177 / 0.0177 | 0.0268 / 0.0269 | 0.016 / 0.024 | 0.915 / 0.912 |
| cwfs | extra / intra | 0.0225 / 0.0248 | 0.0324 / 0.0346 | 0.021-0.023 / 0.078 | 0.910 / 0.918 |
| giant | extra / intra | 0.0329 / 0.0331 | 0.0403 / 0.0407 | 0.036 / 0.052 | 1.524 / 1.532 |

- **Spider:** Danish `spider_angle=0` is needed; leaving it None adds 0.007-0.010 to l1
  everywhere.
- **Field:** l1 is flat to 1.4 deg field radius, then rises through the vignetted zone;
  worst at the outer R00/R04 SW1 corners (1.77-1.82 deg, l1 after fit 0.060-0.078).
  The vignetted area itself agrees: from the centre to 1.7 deg the batoid transmitted
  area falls by 0.915 and Danish's by 0.912 (ratios, dimensionless). The edge shapes
  differ, not the amount removed.
- **Giant edges do not match.** Residuals show a ring at both pupil edges and red/blue
  pairs along every vane ([figure](figures/giant_intra_edge.png)). The fit covers it by
  inflating the blur: fitted 1.36, 1.50 and 1.77-1.81 arcsec at a true 0.5, 1.0 and
  1.5 arcsec. Consistent with the obscuration file not matching the batoid pupil at the
  M2 + camera 4 mm configuration, as v4.02 needed a refit there.
- **FAM blur deficit, open:** the fit returns the true blur minus 0.06-0.09 arcsec
  (0.43, 0.91-0.93, 1.41-1.43 arcsec at 0.5, 1.0, 1.5), a near-constant offset. A test
  with a synthetic uniform annulus was inconclusive (the fit did not move from its
  start).
- **Conventions found on the way:** Danish centres an odd stamp at +0.5 px in x and y
  (the corner of the middle pixel), and its `dx`, `dy` are arcsec (1 arcsec = 5 px).
  With that, and with the TA tilts z2, z3 in the reference, the fitted shift is at most
  0.009 px. Without the tilts the giant donuts sat up to 3.2 px off: TA z2 reaches 14 µm
  at the field edge. `batoid_intrinsic` now stores Noll 1-78 (build `20261121`,
  `7943b45`), as ts_wep's Danish reference does.

[Agreement against field radius](figures/field_radius.png).

Next:
- refit the obscuration file at the giant configuration (`gen_mask_params.py`), or ask
  Josh, and rerun;
- pin down the FAM 0.07 arcsec blur offset (Danish's edge rendering against a sharp
  synthetic pupil);
- `sim_donut_fits` and `v4speed_danish_closure`.

### 2026-11-20 — started

Study created.
