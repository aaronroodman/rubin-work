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

### 2026-11-18 — input product built; fit decisions

- **Input:** `giant_donuts=v402s_fitc@20261118`, an import build. It holds 10 exposures and
  839 sides: night 1 has 173 science sides, 20261107 has 143 and 20261108 has 462, plus
  the corner sensors. Code is in `rubinwork/products/giant_donuts/`. The edge finder (R0)
  is now the library `rubinwork.giant_donuts.edge_profiles`, and the Batoid v4.02 helpers
  are `rubinwork.giant_donuts.batoid_model`. The pupil study now imports both; its R0
  edges and Batoid traces are unchanged (checked).
- **Fit design (Aaron):** shape only. Each bin gets a free radial shift and plateau scale,
  and the roll-off is fitted to the rim and toe shape. The implied edge shift is then
  checked against the measured data − danish offsets. Huber loss (scipy least_squares,
  f_scale = 2 sigma).
- **Interferometric map:** the mirror-lab M1 test map (`m1_figure_error.h5`, 48 averages,
  2019-02-21) is the source of the A, w fit and of v4.03. Besides the parametric fit (a),
  fit (b) uses the map itself near the inner edge: the ray deflections come from ∇h at
  each ray's M1 hit point. Its free parameters are an amplitude scale and the map's
  rotation (and possible flip) against the telescope frame. That tests the map's
  azimuthal structure (a factor 3 in depth between 30° sectors), and the donuts would fix
  the map-to-frame tie that SUMMARY.md lists as open.

### 2026-11-18 — model grids in batch; fit step

- **Model stage:** `code/model_profiles.py`, Slurm job 45765378, run `20261118_model_profiles`.
  176 science sides at 1.6-1.725 deg field radius. Grids: M1 A = 0-300 nm × w = 8-60 mm
  (48 points), M3 B = 0-320 nm × u = 10-50 mm (15 points), blur FWHM 0.8-2.0 arcsec (6).
  The one-side test matches the full v4.03 trace: inner-edge shift +0.79 pixels (v4.03
  +0.81), rim 1.09 (v4.03 1.10, dimensionless).
- **Fit stage:** `code/fit_rolloff.py`. Per bin, the model is shifted (±6 pixels) and
  scaled to the data at the data's own radii, with Huber loss at 2 sigma. A first version
  shifted the data instead; the interpolation halves the noise between samples and pulled
  the shifts to the 0.25-pixel midpoints, so it was dropped. The blur is free per side,
  shared by the M1 and M3 fits. Errors come from a bootstrap over sides. Controls: M1 grid
  on inner bins set by other elements, M3 grid on M1Baffle1 outer bins.
- **Test side** (extra-focal, 84 M1-inner bins): the data rim is 1.43; the nominal model
  gives 1.00 and the map roll-off (A = 76.4 nm, w = 27.1 mm) gives 1.09 (rim = peak within
  10 pixels over the 12-25 pixel plateau, dimensionless). The roll-off lowers the M1-inner
  Huber cost from 74 to 64 per bin; the inner-other control changes from 61 to 58.

### 2026-11-18 — first fit, 118 of 176 sides

- Model job 45765378 hit its 4 h limit with 118 sides done (60-74 min per side, not the
  30-40 min estimated); the 58 running sides were lost. Fit job 45767271 ran on the 118
  (64 extra, 54 intra): run `20261118_fit_rolloff`.
- **Intra and extra disagree.** Best grid points (spline-refined; bootstrap [16, 84] is
  under ±10 % and not the real error):

  | sides | A (nm) | w (mm) | B (nm) | u (mm) |
  |---|---|---|---|---|
  | extra | 300 (grid edge) | 37 | 78 | 10 (grid edge) |
  | intra | 86 | 60 (grid edge) | 283 | 22 |
  | mirror-lab map | 76.4 | 27.1 | about 80 | about 50 |

  The extra-focal M1 hot rim (data 1.33) needs the steep roll-off (model 1.25; the map
  gives 1.09). Intra-focal, that roll-off raises the M1-inner Huber cost from 270 to 371
  per bin and moves the edge +3.7 pixels; the intra data favour the map scale. One
  exponential roll-off does not fit both sides. The same holds for M3 with the roles
  swapped.
- **Control:** M1Baffle1 outer edges match the model on both sides (rim 1.00-1.01 data,
  0.98 model, dimensionless), with small ripples (+3 to +5 % at 5 pixels, then a dip),
  consistent with Fresnel fringes washed out by seeing. Inner edges set by other elements
  (M2, M2 baffle) are not a clean control: their rays also graze M1's inner edge, and
  extra-focal they show a rim of 1.26.
- The fits are poor everywhere (Huber cost 260-470 per bin over 76 points, dimensionless),
  so structure beyond the model remains.
