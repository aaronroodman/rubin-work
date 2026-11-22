---
study: v4speed_mask_vs_batoid
topic: aos
status: answered
question: With the wavefront fixed at the Batoid v4_speed TA Zernikes (no fit), how well does the Danish forward model with the v4_speed obscuration file match Batoid-traced donuts, by defocus configuration (FAM, CWFS, giant), side and field position?
answer: With the v4_speed obscuration file (spider_angle 0 deg) Danish at the batoid TA wavefront misplaces a median 0.20% of the flux on FAM, 0.22-0.24% on CWFS and 0.15% on giant donuts, at most 0.27% within 1.725 deg; the giant residual is the +-4 mm M2 and camera shift (edges off 3-8 mm on the stop plane, sign reversing with side). Pass 1 used ts_wep's shipped mask by mistake.
uses:
  batoid_intrinsic: v4_speed@20261121, v4_speed@20261121b
  sim_donuts: v4_speed_mono@20261120, v4_speed_mono_edge@20261121
started: 2026-11-20
---

# `v4speed_mask_vs_batoid`: v4_speed: Danish obscuration file against the Batoid model

## Question

With the wavefront fixed at the Batoid v4_speed TA Zernikes (no fit), how well does the Danish forward model with the v4_speed obscuration file match Batoid-traced donuts, by defocus configuration (FAM, CWFS, giant), side and field position?

## Current answer

With the v4_speed obscuration file (spider_angle 0 deg), Danish at the batoid TA
wavefront misplaces a median 0.20% of the flux on FAM donuts, 0.22-0.24% on CWFS and
0.15% on giant donuts (l1, dimensionless), at most 0.27% within 1.725 deg, with the
fitted blur within 0.01 arcsec of the truth. The remaining giant mismatch is the ±4 mm
M2 and camera shift: M3, M2 and camera-body edges off by 3-8 mm on the stop plane, with
opposite signs on the two sides. Pass 1's larger numbers were ts_wep's shipped v3.14
mask, used by mistake.

## Log

### 2026-11-21 — correction and pass 2: the v4_speed file, edge grid, per-element edges

**Correction: pass 1 did not test the v4_speed obscuration file.** `rubinwork.donut_sim.optics.instrument`
passed `maskParams=None` to ts_wep's `Instrument`, and `mergeConfigWithFile` drops `None`
kwargs, so the 6-element v3.14 mask inline in `LsstCam.yaml` stayed in force. Fixed in
`90381bf` (clear with the setter, then check against the file). Runs `20261121_ta_model`
(rerun under `--mask lsstcam_shipped`, now titled for that mask) and the pass-1 Log numbers
describe **ts_wep's shipped mask against the v4_speed batoid model**, not Josh's file.
`20261121_blur_check` was redone with the right mask. The products are unaffected: neither
`batoid_intrinsic` nor `sim_donuts` uses the mask.

With the v4_speed file (run `20261121_ta_model_v4obsc`, all 998 `sim_donuts` stamps), l1 is
the fraction of the flux misplaced (dimensionless); medians, spider_angle 0 deg:

| config | l1 no fit | l1 fit, max within 1.725 deg | fitted FWHM (arcsec, true 1.0) |
|---|---|---|---|
| fam | 0.0020 | 0.0027 | 0.990 |
| cwfs | 0.0022-0.0024 | 0.0023 | 0.990 |
| giant | 0.0015 | 0.0024 | 0.992-0.993 |

Against the shipped mask the same numbers were 0.018, 0.022-0.025 and 0.033, with fitted
FWHM 0.91 and 1.53 arcsec. The FAM blur deficit and the giant edge mismatch of pass 1 were
both the shipped mask. Blur check with the v4_speed file: fitted 0.48-0.49, 0.99 and
1.49-1.50 arcsec at a true 0.5, 1.0 and 1.5 arcsec. Without the spider (spider_angle None)
l1 is 0.018-0.026: the vanes must be on.

**Edge grid** (run `20261121_edge_v4obsc`, `sim_donuts v4_speed_mono_edge@20261121`, 1344
stamps, 1.400-1.725 deg in 0.025 deg steps by 24 azimuths): l1 no fit 0.20-0.26% (FAM) and
0.17-0.21% (giant), rising smoothly with radius; after the fit at most 0.26% and 0.25%;
fitted FWHM 0.989-0.995 arcsec. No radius stands out where the eight camera edges switch on.

**Stop plane, per element** (`pupil_edges.py`, `0db135c`, run `20261121_pupil_edges`):
batoid vignetting against Danish's mask on a 512 x 512 stop-plane grid, 14 radii x 4
azimuths. Mismatched area over batoid clear area: FAM 0.02% median, 0.06% max; giant
0.19-0.21% median, 0.27% max. The giant offsets are nearly constant in field radius and
reverse with the side ([figure](figures/giant_edge_offsets.png)): extra M3 outer +8 mm,
M2 outer +5 mm, M2 inner -3 mm, camera body -6 mm; intra the opposite. That is the
+-4 mm M2 and camera shift, which one mask file cannot follow; for FAM the largest is the
camera body at 3.7 mm.

[Agreement against field radius](figures/field_radius.png) (now the v4_speed file).

Next:
- `sim_donut_fits` and `v4speed_danish_closure`: whether 0.2% of misplaced flux biases
  the fitted Zernikes;
- if the giant fits are biased: side-dependent giant masks from batoid edges, as
  `gen_mask_params.py` does.

### 2026-11-21 — pass 1 PDF: 1.725 deg limit, metric equations, edge examples

PDF of `20261121_ta_model` rewritten in place (code `30a9643`): metric equations
on page 1, a dashed line at the 1.725 deg field-radius limit on page 2, and example
donuts near the edge: FAM and giant 4 per side from 1.4 to 1.725 deg, CWFS 3 per side
from its innermost point to 1.725 deg. Numbers unchanged.

Within 1.725 deg, the largest l1 after the nuisance fit (fraction of flux misplaced,
dimensionless) is 0.023 (FAM extra), 0.024 (FAM intra, CWFS intra), 0.023 (CWFS extra),
0.045 (giant extra) and 0.047 (giant intra). The CWFS 0.060-0.078 stamps are all at
1.77-1.82 deg, outside it.

Blur and wavelength, checked in the code: both images use a galsim Kolmogorov of fixed
FWHM with no wavelength scaling. The sims convolve it without the pixel on a 3x
oversampled deposit and sum to 10 µm pixels; Danish draws it with galsim's default
pixel convolution (`donut_model.py:282`). The sims trace at 619.4 nm, which enters
through the glass indices, and the TA reference is batoid at 619.4 nm; Danish's model is
geometric given Zernikes in m, with no wavelength of its own (`bandpass_filter` None).

[Agreement against field radius](figures/field_radius.png) (replaced).

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
