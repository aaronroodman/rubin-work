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

### 2026-11-19 — data against candidate roll-offs (118 sides)

`code/compare_profiles.py`, run `20261119_compare_profiles`. One data curve per bin set,
against five candidates in a common frame, with residuals, by psi and by night.
Rim = peak within 10 pixels over the 12-25 pixel plateau; cost = mean Huber cost per bin.
Both are dimensionless.

| edge, side | data rim | nominal rim (cost) | map rim (cost) | best-all rim (cost) |
|---|---|---|---|---|
| M1 inner, intra | 0.88 | 0.99 (270) | 0.97 (274) | 0.83 (308) |
| M1 inner, extra | 1.31 | 0.99 (469) | 1.03 (397) | 1.25 (304) |
| M3 outer, intra | 1.26 | 0.99 (425) | 1.11 (334) | 1.24 (293) |
| M3 outer, extra | 0.87 | 0.98 (430) | 0.94 (385) | 0.94 (336) |

Map = A 75 nm, w 27 mm; B 80 nm, u 25 mm. Best-all = A 300 nm, w 60 mm; B 80 nm, u 10 mm
(nearest grid points).

- **M3:** one turn-up fits both sides fairly well: B about 90 nm, close to the map's
  +80 nm, but over about 10 mm rather than 50 mm.
- **M1:** no single roll-off fits both sides. The map is a safe gain: extra-focal cost
  falls from 469 to 397 and intra-focal is unchanged. The best joint fit matches the
  extra-focal rim, but it tilts the intra-focal plateau (1.15 at +20 pixels).
- **Shared ripple:** both sides of the M1 inner edge, and M3 intra, show a peak at about
  +5 pixels and a dip at about +10 pixels. The pattern has the same sign intra and extra,
  so it is not a surface slope (that flips sign between the sides). Fresnel fringes are
  at about the right place: Fresnel scale sqrt(lambda d / 2) = 5.1 pixels for
  d = 8.4 mm, so maximum near 6 pixels and minimum near 9.5 pixels. But seeing at about
  7 pixels FWHM should damp them below 1 %. Not understood.
- **By psi:** the extra-focal M1 rim ranges from 1.17 (psi 180-240 deg) to 1.37
  (psi 300-360 deg).

### 2026-11-19 — all 176 sides

Fit run `20261119_fit_rolloff` (87 extra, 89 intra); comparison `20261119_compare_profiles`;
focal-plane gallery of the sample `20261119_sample_gallery` (one page per exposure,
stamps at true size). The 118-side picture holds: all sides A = 252 nm, w = 60 mm (grid
edge), B = 92 nm, u = 10 mm (grid edge); extra alone A = 300 nm (grid edge), w = 37 mm;
intra alone A = 88 nm, w = 60 mm. Rims (data; map; best all, dimensionless): M1 inner
intra 0.88; 0.97; 0.83, extra 1.27; 1.03; 1.26; M3 outer intra 1.27; 1.12; 1.24, extra
0.87; 0.94; 0.94. Median per-side blur 1.4 arcsec FWHM.

### 2026-11-19 — vane flags, trimmed windows, whole profiles by psi

Aaron asked for whole profiles (inner edge to outer edge) in 30 deg psi ranges, with no
vane shadows. Code `668891a` and this commit.

- **Vane flags:** `code/spider_bins.py`, run `20261119_spider_bins`. Batoid v4.02 with
  vanes on a 400 × 400 stop grid; a bin is dropped if any vane-blocked ray lands between
  3 pixels inside the inner edge and 3 pixels outside the outer edge, and so are its two
  neighbours. Taking the shadows from dips in the data fails: single 2 deg bins carry
  ±10–20 % texture across the flat top.
- **Truncated model windows:** the dense trace keeps only stop radii 2.30–3.10 and
  3.60–4.25 m, so in some windows the model runs out of rays. Bins where that reaches the
  fitted ±19 pixels are now dropped from the fit (`fit_rolloff.trim_window`): 0 % of
  m1_inner and baffle_outer, 7.4 % of m3_outer, 26.6 % of inner_other bins. Refit, Slurm
  job 46012866, run `20261119_fit_rolloff` (it overwrote the untrimmed run of the same
  name): the best fits do not change at the quoted precision.
- **Whole profiles:** run `20261119b_compare_profiles`, pages 6 (intra) and 8 (extra).
  Inner half aligned on each bin's inner edge, outer half on its outer edge placed at the
  median width; models drawn over the fitted ±19 pixels only.

Data peak over the flat top within 10 pixels of the edge (dimensionless), by 30 deg psi
range, 0–30 first:

| edge | side | psi 0–120 | 120–240 | 240–360 |
|---|---|---|---|---|
| inner | extra | 1.43, 1.46, 1.41, 1.38 | 1.11, 1.23, 1.24, 1.10 | 1.37, 1.43, 1.45, 1.41 |
| outer | intra | 1.03, 1.03, 1.03, 1.00 | 1.08, 1.16, 1.19, 1.09 | 1.00, 1.02, 0.99, 1.02 |

- The extra-focal inner rim is strong (1.37–1.46) wherever M1 or M2BaffleCAD sets the
  edge, and weaker (1.10–1.24) at psi 120–240, where M2 inner sets it in 79–100 % of the
  bins. Best all reaches about 1.3; the map and nominal about 1.15.
- The intra-focal outer rim is only where M3 outer sets the edge (psi 150–210, 1.16–1.19);
  no candidate reaches it.
- The fit wants w past the grid's 60 mm and u below its 10 mm (and extra A past 300 nm);
  a wider grid needs a new model batch run.
