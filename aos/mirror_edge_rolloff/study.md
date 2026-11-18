---
study: mirror_edge_rolloff
topic: aos
status: active
question: What M1 inner-edge roll-off (depth A, e-folding width w) and M3 outer-edge turn-up do the giant-donut rim profiles require, compared with the mirror-lab figure maps?
answer: ''
uses: {}
started: 2026-11-18
---

# `mirror_edge_rolloff`: M1 inner-edge roll-off and M3 outer-edge turn-up from giant-donut rims

## Question

What M1 inner-edge roll-off (depth A, e-folding width w) and M3 outer-edge turn-up do the giant-donut rim profiles require, compared with the mirror-lab figure maps?

## Current answer

None yet.

## Log

### 2026-11-18 — started; plan

Study created (Aaron). Background:
- **Mirror maps:** rubin-optical-model `studies/mirror_figure_errors/SUMMARY.md`. The M1
  inner edge is rolled down, h(r) = c − A exp(−(r − 2.558 m)/w), A = 76.4 ± 8.9 nm,
  w = 27.1 ± 2.8 mm; A and w vary by a factor of 3 (dimensionless) over twelve 30° sectors.
  The M3 outer edge is turned up by about +80 nm over its last 50 mm. `LSST_r_v4.03.yaml`
  is v4.02 plus the azimuthal-mean M1 roll-off.
- **Rims in the giant donuts:** `aos/giant_donut_pupil` Log, 2026-11-18 (rim analysis).
  Where M1 inner sets the inner edge, the R0 edge sits +0.9 to +1.4 pixels outside the
  danish model extra-focal and −0.3 to −0.4 pixels inside it intra-focal. The rim
  contrast is 1.36-1.48 extra-focal and 0.90-0.95 intra-focal (dimensionless).
  Batoid v4.03 − v4.02 gives +0.83 and −0.44 pixels, and a rim of 1.10. Where M3 outer
  sets the outer edge, there is an intra-focal rim of 1.33-1.38 and none extra-focal.
- **Related:** `aos/intra_extra` lists a turned-down M1 edge as its open candidate.

Plan:
1. **Input:** the `giant_donuts` product (exposures, sides, stamps, fit a and fit c). Its
   first build imports the `giant_donut_pupil` outputs: night 1 pass 6 `isr_phys_v402s`,
   and 20261107 and 20261108 `v402s`.
2. **Fast forward model:** trace Batoid v4.02 (no vanes) once per side, keeping each ray's
   radius and azimuth on M1 and M3 and its detector position. For a roll-off dz(r), move
   each ray in the focal plane by the extra deflection: lever L × 2 × 0.982 × dz′(r),
   with L about 10.31 m for M1. A check at r = 2.59 m gives −17.5 µm, against −17.6 µm
   traced through v4.03. Then re-bin and blur with Kolmogorov at the side's free-fit
   FWHM. About 1 s per model, instead of about 75 min for a v4.03 trace. The deflection
   direction and the M3 lever come from a Batoid finite-difference calibration.
3. **Fit:** A and w (with the inner clear aperture radius as a check on degeneracy) to the
   radial profiles of the bins where M1 inner sets the inner edge, intra and extra
   jointly, over all sides. Same for the M3 outer turn-up, on the bins where M3 outer sets
   the outer edge, with the M1Baffle1 bins as a control. Robust loss; ask Aaron which.
4. **Then:** per-sector A and w, once the map azimuth is tied to OCS. Compare with the
   mirror-lab sector table.
