---
study: giant_donut_pupil
topic: aos
status: active
question: "With a Danish model that includes the spider, what wavefront (coma, astigmatism) do giant donuts give across the full focal plane including the corner WFS, and does the v4.02 CAD-matched pupil fit better than v1000?"
answer: 'v4.02 fits better than v1000 (Z4-only fits: 15 of 16 clean donut sides, median residual RMS ratio 0.94, dimensionless). The v4.02 danish edges match full Batoid v4.02 to 0.2-1.6 pixels RMS inside 1.725 deg, but the data''s inner edge is decentred by about 2.5 pixels (25 um) from both.'
uses: {}  # none; raws, the_monster refcat and rubin-optical-model directly
started: 2026-10-18
---

# `giant_donut_pupil`: giant-donut wavefront over the full focal plane, v1000 against v4.02

> **Code:** `code/isr_cache.py` (binned ISR cache), `code/fp_mosaic.py` (focal-plane
> mosaic), `code/candidates.py` and `code/select_candidates.py` (candidate selection).
> Imports `wfs/code/giant_donuts/` (shared in the section 2 step).

## Question

With a Danish model that includes the spider, what wavefront do giant donuts give across
the full focal plane, including the corner wavefront sensors (WFS)? What do coma (Z7,
Z8) and astigmatism (Z5, Z6) look like? Does the v4.02 CAD-matched pupil (spider at the
CAD location) fit better than v1000?

Data: day_obs 20251023, BLOCK-T626, r band, 60 s; exposure 2025102300337 (extra-focal)
and 2025102300340 (intra-focal). Both sit at ±4000 µm on the camera hexapod and ±4000 µm
on M2, a 4 + 4 mm split, about 7.6 mm effective defocus. Rotator angle 0.012 deg and
0.149 deg. Background: `notes/status/item7_giant_donuts_handoff.md`,
`wfs/docs/giant_donuts_pupil_findings.md`.

Optical models: `~/notebooks/rubin-optical-model` at commit `e2a1e54` (read only).

## Current answer

v4.02 fits the giant donuts better than v1000. With Z4 + Z11 + Z22, its residual RMS is
lower on 16 of 16 clean donut sides; with the blitz set, on 12 of 16. The median ratio,
v4.02 over v1000 residual RMS, is 0.94 (Z4 + Z11 + Z22) and 0.95 (blitz set), both
dimensionless. Most of the gain is the spider: the v1000 vanes sit 50 mm (one vane width)
off the CAD shadow. Inside 1.725 deg the fitted Z5–Z8 deviations stay within ±3 µm of
wavefront and show no trend with field radius. Beyond 1.9 deg the fits break down.

Fitted with Z4 only (other Zernikes fixed at the batoid intrinsic, blur fixed at 1.0
arcsec), v4.02 still wins: 35 of 38 donut sides, 15 of 16 clean, median ratio 0.944
(dimensionless). Inside 1.725 deg the v4.02 danish edges match a Batoid ray trace of the
full v4.02 model to 0.2–1.6 pixels RMS. The data do not match either: on clean donuts
the inner edge sits 3.5–5.7 pixels outside Batoid over OCS azimuth 0–135 deg and within
1 pixel over 225–315 deg, on both sides of focus.

## Log

### 2026-10-20 — pupil pass 2

Run `20261019_pupil2`. Already there at the start of the day (code committed with this
line): `state336` (json, `_dof`, `_zk`), `trim_check`, `hexapod_actual`,
`hexapod_demand` (`giant_state.py`); `masks/` (6 v4.02g files), `v402g_validation`,
`v402g_edge_shift`, `v402g_silhouette`, `camaperture_cuts` (`obscuration_v402g.py`);
`batoid_images_v1000` and its G0515 extra test (`batoid_images.py --model v1000`);
`fits_v402g`, Z4 and Z4 + Z7 + Z8, 76 fits (`fits_pass2.py`); `coma_sensitivity`, 8
sides (`coma_sensitivity.py`); `element_tolerance` and `geom_coma_check.json`
(`geom_edges.py`); `images/` (38 each of `v4.02g_z4`, `v4.02g_z478`, `batoid_v1000`).
`robust_edges.py` is written but not yet run.

Found before today, against seq 336: Trim at 337 and 340 minus Trim at 336 is exactly
±4000 µm in M2 z and camera z and 0 elsewhere. EFD actual minus demand during 337 and
340 is at most 1.40 µm and 0.12 arcsec.

**Step 1, the state: seq 324.** value_added `optical_state`, variant
`v22_12__batoid__consdb_v1`, snapshot 20261016, has seq 324 (resid RMS 0.148 µm of
wavefront), so 324 replaces 336. `giant_state.py --seq 324` writes `state324*`,
`trim_check_324`, `hexapod_*_324`.
- Corner deviation from the batoid intrinsic (µm of wavefront, OCS):

  | corner | Z5 | Z6 | Z7 | Z8 |
  |---|---|---|---|---|
  | R00_SW0 | −0.57 | +0.24 | −0.09 | −0.10 |
  | R04_SW0 | +0.49 | +0.19 | +0.27 | −0.26 |
  | R40_SW0 | +0.24 | −0.07 | −0.26 | −0.12 |
  | R44_SW0 | −0.25 | −0.04 | +0.17 | −0.15 |
- DOF (optical_state; µm, arcsec): M2 dz +175.7, dx −22.7, dy −7.8, rx −0.05, ry
  −1.56; camera dz −172.4, dx +578.0, dy −160.7, rx 0.00, ry −3.63; bending modes
  within ±0.03 µm.
- Trim at 324 (µm, arcsec): M2 z −1230.19, rx +10.45, ry −86.58; camera z −1250.63,
  rx −27.44, ry +4.89; x, y within 0.07 µm. **The Trim at 324 does not equal that at
  337 and 340**: 337 and 340 carry the Trim of seq 328 (M2 rx +10.10, ry −84.97;
  camera rx −26.55, ry +6.89 arcsec), after a Tweak at 326 and one at 328 (M2 dz −53.5
  µm, camera dz +74.5 µm). Trim(337) − Trim(324): M2 z +3941.57 µm, rx −0.34, ry
  +1.61 arcsec; camera z +4081.76 µm, rx +0.89, ry +2.00 arcsec; x, y within 0.002
  µm. 340 is the same with z −4058.43 and −3918.24 µm. The EFD demand change from 324,
  less the elevation LUT change, matches these to 0.4 µm and 0.01 arcsec.
- EFD actual minus demand at 324: at most 1.73 µm and 0.05 arcsec.
- Batoid DOF for step 3b: the 324 optical_state plus the 337/340 Trim tilts (M2 rx
  +10.10, ry −84.97; camera rx −26.55, ry +6.89 arcsec), applied with batoid_rubin in
  the sensitivity-matrix convention (`giant_state.apply_dof`). In Batoid v4.02 the
  Trim tilts alone give Z8 −1.71 µm and Z7 −0.23 µm of wavefront on axis, nearly all
  from M2 ry; the 324 optical_state alone gives under 0.15 µm. The v4.02 file already
  holds an as-built alignment, so adding the Trim may count it twice; 3b runs both
  (`v4.02s` with the Trim tilts, `v4.02s0` without).

**Step 2, coma.** `pass2_report.py --steps coma` writes `coma_conversion` (per side) and
`coma_summary.csv`. Sensitivity (`coma_sensitivity`, 8 sides): +1 µm Z7 gives +4.45
pixels of inner-minus-outer centre shift along +y CCS, +1 µm Z8 gives +4.45 pixels
along +x CCS, the same sign intra and extra. Each side's pass-1 data − Batoid edges
are fitted with a circle per edge; the inner-minus-outer centre shift is solved for
Z7, Z8 with that side's sensitivity (or the 8-side median). Clean sides, median (nMAD),
µm of wavefront:

| side | n | Z7 from edges | Z8 from edges | Z7, danish Z4+Z7+Z8 fit | Z8, fit | Z7, 324 state | Z8, 324 state | Z7, 324 state + Trim tilts | Z8, same |
|---|---|---|---|---|---|---|---|---|---|
| extra | 7 | +0.71 (0.12) | −0.01 (0.07) | +0.02 (0.20) | +0.14 (0.22) | +0.02 (0.03) | −0.15 (0.03) | −0.21 (0.02) | −1.84 (0.03) |
| intra | 9 | +0.81 (0.17) | +0.05 (0.26) | +0.19 (0.08) | +0.07 (0.28) | +0.02 (0.03) | −0.15 (0.04) | −0.20 (0.03) | −1.83 (0.03) |

- The edges give one common coma, Z7 +0.7 to +0.8 µm, Z8 ≈ 0, on both sides of focus
  and at all field radii 0.17–1.70 deg (side range +0.63 to +1.03 µm; one exception,
  G0618 extra, −0.20 µm). That is what coma does and what a decentred obscuration does
  not, so the geometry supports the coma hypothesis.
- But the danish fits with Z7 and Z8 free do not find it: median Z7 +0.02 µm extra
  and +0.19 µm intra. Per side, edge-Z7 against fit-Z7: Pearson r +0.71 extra and
  −0.43 intra, Spearman rho +0.32 and −0.50 (n 7 and 9). The fit is driven by the
  whole flat-top pattern, not the edges, so a real 0.75 µm Z7 would show up there.
- The 324 state has no such coma: its Batoid Z7 at the donut fields is +0.02 µm. With
  the Trim tilts added it is −0.21 µm of Z7 and −1.84 µm of Z8, which the edges rule
  out (they give Z8 ≈ 0).
- So the inner-edge shift has the signature of coma but is not wavefront coma that
  danish or the state can see. Step 5's element tolerance and step 3b test what else
  could do it.

**Element tolerance** (`geom_edges.py`, `element_tolerance`). Geometric edges (no blur)
from a dense Batoid trace of v4.02 at both giant configurations and 4 fields (0, 0),
(0, 1.5), (1.2, 1.2), (1.5, 0) deg. Each non-optical group is moved 10 mm in x, y and
z (editing its yaml coordinates, optics untouched); the M2 and camera hexapods 100 µm.
Recorded: the edge change and the inner-minus-outer centre shift (pixels), and for the
hexapods the Batoid coma. To give the measured +3.3 pixels of shift along +y CCS, the
same sign on both sides of focus:
- the M1 inner hole, M1Baffle1, M2 and M3 apertures and the camera silhouette give
  0.3–0.9 pixels per 10 mm, but **their sign flips between intra and extra**, as any
  pupil-plane obscuration's does, so none of them can give the measured shift;
- the filter gives −0.35 pixels and L3 −5.7 to −6.3 pixels per 10 mm of dy, the same
  sign on both sides. L3 would need about 5 mm in −y, and that would move the outer
  edge by about 15 pixels RMS, which is not seen. The filter would need about 100 mm;
- the hexapods' 100 µm decentres give under 0.2 pixels and under 0.07 µm of coma;
- the M2 baffle, M2 cell and mid-level baffle never set an edge (0.00 pixels at 10
  mm): their lips sit inside the M1 hole on the incoming beam.

So no single plausible element shift produces the inner-edge offset. Of everything
tested, only something in the converging beam near focus (L3, the filter) has a
same-sign signature, and they would need implausible moves.

**Step 4, radius ratio** (`radius_ratio`): r_inner / r_outer (dimensionless) per bin
from the robust edges, bins usable on both edges. Per side: the median and the spread
over azimuth (nMAD, and 95th minus 5th percentile). Medians over the clean sides:

| image | extra: median | nMAD | p95 − p5 | intra: median | nMAD | p95 − p5 |
|---|---|---|---|---|---|---|
| data | 0.6279 | 0.0175 | 0.043 | 0.6306 | 0.0296 | 0.088 |
| Batoid v4.02 | 0.6159 | 0.0135 | 0.053 | 0.6173 | 0.0164 | 0.077 |
| danish v4.02 | 0.6158 | 0.0124 | 0.050 | 0.6156 | 0.0153 | 0.071 |
| danish v4.02g | 0.6160 | 0.0126 | 0.051 | 0.6163 | 0.0162 | 0.070 |
| danish v1000 | 0.6156 | 0.0128 | 0.050 | 0.6168 | 0.0163 | 0.079 |
| Batoid v1000 | 0.6158 | 0.0139 | 0.051 | 0.6176 | 0.0165 | 0.078 |

All five models agree to 0.002. The data ratio is 0.012–0.013 higher (2 %), on both
sides of focus: the inner shadow is larger relative to the outer edge than any model
has it. At r_outer ≈ 340 pixels, 0.012 is 4 pixels of inner radius. So the inner
shadow is both larger and shifted. Coma barely changes the radii (Z7 changes r_inner
by under 0.1 pixel per µm in `coma_sensitivity`), so coma cannot give the ratio
excess; a central obscuration about 1 % larger in radius than v4.02's could. The outer-edge
offset of pass 1 (−0.4 to −1.7 pixels) is a fraction −0.0033 extra and −0.0017 intra
of r_outer (median, clean sides). The spread over azimuth is set by field vignetting
and is the same in data and models.

**Step 5, Batoid against danish** (`pass2_report.py --steps cases`: `cases`,
`pattern_bins`, `pattern_table`). G0429 and G0492 are not among the 38 fitted sides:
both are `blended` candidates, never fitted. The table uses G0249 and G0592, the
fitted sides with the same digits; Aaron should confirm that is what he meant. Per case,
the surface is the Batoid v4.02 first blocker of the traced rays within 3 pixels
outside the Batoid edge; the offsets are medians over the range of raw 50 % crossings,
model minus Batoid v4.02 (pixels):

| case | surface | danish v4.02 | danish v4.02g | danish v1000 | Batoid v1000 | verdict |
|---|---|---|---|---|---|---|
| G0429→G0249 extra, outer 80–200 | M1Baffle1 | +0.25 | +0.22 | +0.24 | +0.22 | agree within 0.3 |
| G0249 intra | M1Baffle1 | +1.06 | +1.06 | +1.17 | +0.28 | danish +1 pixel: the outer circle is a polynomial fit; v4.02g no change |
| G0492→G0592 extra, inner 330–350 | M1 hole | +2.01 | +1.89 | +1.34 | +0.24 | danish 2 pixels off at 1.65 deg; v4.02g no fix |
| G0592 intra | M2 inner | −3.07 | −3.42 | −2.31 | −0.26 | as above |
| G0491 extra, inner 150–280 | M1 hole | +0.32 | +0.22 | +0.23 | +0.42 | agree; the "cut" is the M2 baffle CAD lip and spider |
| G0491 intra | M2 inner (0.51), M1 | +0.05 | +0.09 | +0.20 | +0.14 | agree |
| G0515 extra, inner 30–50 | M1 hole | +0.31 | +0.17 | +0.58 | — | agree |
| G0515 intra | M2 inner | −1.83 | −2.07 | +0.57 | — | danish v4.02 2 pixels inside; v4.02g no fix |
| G0618 extra, inner 125–145 | TopEndCentre plane | −0.42 | −0.64 | +0.24 | — | a top-end plane danish lacks; small |
| G0618 intra | inclined spider top | −3.48 | −3.88 | +0.70 | — | spider vane near the inner edge; danish's vane projection differs |
| G0663 extra, outer 220–240 | M3 outer | −0.14 | +0.18 | −10.43 | +0.65 | v1000 lacks the v4.02 M3; v4.02 and v4.02g fine |
| G0663 intra, outer 35–50 | Filter entrance | +24.4 | +3.8 | +5.3 | — | **v4.02g fixes it**: the filter circle's thetaMin moved |
| G0664 intra, outer 170–190 | Filter entrance | +8.0 | −8.4 | −7.2 | −14.4 | v4.02g overshoots |
| G0676 extra, outer 40–60 | M3 outer | −8.3 | −0.5 | −20.2 | +2.3 | **v4.02g fixes it**: M3 outer circle refitted at +4 mm |
| G0678 intra, outer 310–330 | Filter entrance | +17.4 | −2.5 | −1.9 | −18.9 | **v4.02g fixes it** |

- **Hypothesis (i), the mask fitted at nominal, holds** for the outer edges at
  1.73–1.75 deg set by M3 and the filter: v4.02g moves danish onto Batoid. But Batoid
  itself then misses the data there by 7–21 pixels: the data edge lies outside the
  filter cut Batoid applies (G0663, G0664, G0678 intra), as pass 1 found.
- **Hypothesis (ii), the dropped in-camera openings, is ruled out**: over the full
  field and all azimuths at the giant configurations, `CamAperture` is the unique
  blocker of no ray (fraction 0.000), so dropping it costs nothing. The camera
  silhouette uniquely blocks 0.2–2.5 % of rays at 1.5–1.95 deg and the filter 2–6 % at
  1.75–1.85 deg.
- The remaining danish v4.02 differences of 2–4 pixels (G0592, G0515 intra, G0618
  intra) are inner edges near 1.5–1.7 deg where the M2 inner circle, the spider and
  the top-end planes meet. That is the limit of a one-circle-per-edge fit, and v4.02g
  does not change it.

**Pattern over all bins** (8373 usable bins with all three models): Batoid v4.02 is
closest to the data in 37 % of bins, danish v1000 in 34 %, danish v4.02g in 29 %.
Median |model − data|: Batoid v4.02 1.63, danish v1000 1.70, danish v4.02g 1.75 pixels.
Batoid v4.02 wins on outer edges (50 % of bins), on intra-focal sides (47 %), and where
the camera silhouette (61 %), the filter (77 %) or M1Baffle1 (51 %) sets the edge.
danish v1000 and v4.02g win on inner edges set by the M1 hole (41–43 % each against
16 %). The inner edge is offset the same way in all three (median −2.3 pixels,
model − data, clean sides), so on inner edges the choice of model hardly matters.
No dependence on field radius or on azimuth from the radial direction stands out
(Batoid v4.02 closest in 31–44 % of bins in every band).

**Step 6, robust edges** (`robust_edges.py` through `pass2_report.py --steps edges`:
`edges_bins_pass2`, `robust_kept`, `edge_stats_robust`, `unmodelled_surfaces_robust`).
Every image type gets both the pass 1 50 % crossing and the robust edge; model images
use the data's noise for the thresholds.
- Kept fraction of usable bins (not spider, vignetted or masked), data, median over
  sides (dimensionless): clean 0.90 inner, 0.89 outer; flagged 0.89 inner, 0.85
  outer. Range 0.84–0.91 on clean sides. Lowest: G0592 extra outer 0.60, G0606 extra
  outer 0.60, G0635 extra outer 0.65 (neighbour-crowded corner sides). Model images:
  0.81–0.85 inner, 0.83–0.86 outer, lower than the data because their flat top has
  structure near the spider.
- The pass 1 crossing exists in 100 % of usable bins on most sides, so robust edges
  do not add bins; they remove wrong ones. The gain is in the RMS against Batoid v4.02
  (pixels, pass 1 → robust): G0592 extra inner 82.1 → 2.8 and outer 29.4 → 3.7;
  G0592 intra inner 26.1 → 1.5; G0606 extra inner 24.1 → 2.9; G0732 extra outer
  6.3 → 3.4; G0737 intra outer 4.1 → 2.6. Clean sides change by under 0.2.
- Unmodelled surfaces with robust edges: 188 runs (71 on clean sides), against 234
  (79) in pass 1. By surface: M1 hole, inner, 84 runs, 841 bins, median +3.9 pixels;
  M2 inner, 33 runs, 503 bins, +4.2; M1Baffle1, outer, 41 runs, 269 bins, −4.2;
  Filter entrance, outer, 9 runs, 82 bins, +11.0; M3, outer, 8 runs, 58 bins, −4.3.
  Same ranking as pass 1. The M1 and M2 inner runs are the inner-edge offset, now
  free of the bad crossings.

### 2026-10-19 — pupil model: Z4-only fits, Batoid images and edges

Code `93994b8`: `code/z4_fits.py`, `code/batoid_images.py`, `code/pupil_edges.py`.
Run `20261018_pupil`: tables `z4_fits`, `batoid_images`, `edges_bins`, `edge_stats`,
`unmodelled_surfaces` (parquet and csv). The PDF is `giant_donut_pupil_model.pdf`, 119
pages: three per donut side (step 1, 2, 3), then 5 summary pages.

**Aaron's read of the fits PDF.** v4.02 places the spider correctly; v1000's vanes are
one vane width (50 mm) off. On the intra-focal side, v4.02's vane widths fit the data.
The free fits push the blur FWHM to the 1.5 arcsec bound on 20 of 38 donut sides,
because the model has no thermal effects. Beyond 1.9 deg the fitted wavefront (20–44 µm)
is unphysical and is very likely pupil-model mismatch. The goal now is the pupil model.

Same 38 donut sides, flags, neighbour masks and full-resolution ISR stamps as
`20261018_fits` (read from its `images/`).

**Step 1, Z4-only danish fits.** Z4, flux and centre float. Noll 5–28 are fixed at each
pupil's own batoid intrinsic, at the donut field with the side's offsets. Blur is fixed
at 1.0 arcsec FWHM, with no background term. 76 of 76 fits converged.
- v4.02 has the lower residual RMS (inside 1.1 r_out, unmasked pixels) on 35 of 38 sides
  and 15 of 16 clean. Median v4.02/v1000 residual RMS (dimensionless): 0.944 all, 0.942
  clean.
- Z4 − intrinsic, v4.02 (µm of wavefront): +0.17 to +1.26 clean, −1.48 to +1.39 all.
  v1000: +0.19 to +1.23 clean.
- chi2/dof spans 1.16–199 (v4.02). It scales with brightness, as before.

**Step 2, Batoid images, v4.02.**
- Reduced model per side: the optical surfaces plus the 24–231 obscurations that
  vignette first at least one ray of a 500 × 500 traceFull grid. Agreement with the full
  model (dimensionless fraction of rays): 0.99996–1.00000. The reduced model passed on
  all 38 sides; the full model was not needed.
- Spider: the telescope (not the camera) is rotated by −rtp. Checked against the danish
  mask at a 10 deg test angle: agreement 0.99997 with that sign, 0.972 with the other.
- Camera piston that matches Batoid Z4 to the step 1 fit: −3 to +75 µm extra-focal
  (median +25 µm), −95 to +89 µm intra-focal (median +58 µm).
- 25–30 million rays per side at 620 nm; 103–116 rays per illuminated pixel (median per
  side). Kolmogorov blur, 1.0 arcsec FWHM, drawn as danish does.
- Cross-correlation shift after centring on the step 1 model (pixels): magnitude
  0.0–2.4 on clean sides, up to 18 on flagged sides (G0732 extra, G0714 extra).
- Batoid/danish residual RMS (dimensionless): median 0.977 all, 0.982 clean. Batoid is
  lower on 30 of 38 sides.

**Step 3, edges against azimuth.** 50 % of the flat top, 2 deg bins. Spider bins,
Batoid-vignetted bins and neighbour-masked crossings are marked and excluded.
Per-side RMS of the edge differences (pixels):

| field radius (deg) | subset | data − Batoid, outer | data − Batoid, inner | v4.02 model − Batoid, outer | v4.02 model − Batoid, inner |
|---|---|---|---|---|---|
| 0–1.0 | clean (4) | 1.2–1.5 | 2.6–4.8 | 0.19–0.65 | 0.17–0.29 |
| 1.0–1.6 | clean (8) | 1.2–2.2 | 2.8–4.3 | 0.19–0.81 | 0.22–1.09 |
| 1.6–1.725 | clean (4) | 1.3–2.1 | 2.6–12.4 | 0.66–0.79 | 0.83–1.57 |
| 1.6–1.725 | flagged (5) | 3.2–29 | 2.8–82 | 0.41–2.3 | 0.89–3.0 |
| 1.725–1.9 | flagged (13) | 1.2–8.8 | 2.1–41 | 0.47–7.2 | 1.2–4.1 |
| 1.9–2.0 | flagged (4) | 1.4–6.3 | 2.2–3.5 | 1.3–12.1 | 2.7–5.8 |

The danish v4.02 model tracks the full Batoid model to about 1 pixel inside 1.725 deg.
It departs beyond that, up to 12 pixels RMS at 1.9–2.0 deg. The data depart from both.
On clean sides, the median data − Batoid inner-edge offset per 45 deg OCS sector is
+3.5 to +5.7 pixels over 0–135 deg and −0.9 to +1.3 pixels over 225–315 deg (extra and
intra alike). The outer-edge offset is −0.4 to −1.7 pixels. Read plainly, the inner
shadow is decentred by about 2.5 pixels (25 µm in the focal plane) relative to the
model, the same way on both sides of focus. The cause is not established.

Unmodelled surfaces: 234 runs of 3 or more bins with |data − Batoid| > 3 pixels, 79 of
them on clean sides. By the first clipping surface just outside the Batoid edge:

| surface | edge | runs | bins | median offset (pixels) |
|---|---|---|---|---|
| M1 (inner hole) | inner | 101 | 964 | +3.7 |
| M2 | inner | 39 | 505 | +4.0 |
| M1Baffle1 | outer | 53 | 300 | −3.9 |
| M3 | outer | 13 | 109 | −3.5 |
| Filter entrance | outer | 11 | 108 | +12.5 |
| CameraSilhouette | inner | 8 | 53 | +3.7 |

The largest single runs are inner-edge runs at OCS 54–126 deg on clean donuts (G0005
extra, G0329, G0491, G0515 extra, G0160), each 30–37 bins at +4 to +6 pixels. Those are
the decentre above, not a missing surface. The Filter-entrance runs are all on corner or
edge-of-field sides at 1.74–1.86 deg (G0663, G0664, G0678, G0681, G0684 intra). There
the data edge lies 7–19 pixels outside Batoid's, so v4.02 clips at the filter where the
sky does not.

![Batoid comparison, G0515 extra](figures/batoid_G0515_extra.png)
![edges against azimuth, G0515 extra](figures/edges_G0515_extra.png)

Next:
- Fit the inner-obscuration decentre (M2 or M1 inner-hole shift) in Batoid against the
  clean inner edges.
- Check the filter aperture at 1.74–1.86 deg against the CAD.

### 2026-10-18 — spider check and fits: v4.02 against v1000

Code `1f09446`: `code/spider_validation.py`, `code/fit_donuts.py`, `code/plot_results.py`.
Runs: `20261018_spider_validation`, `20261018_fits`. The results PDF is
`output/20261018_fits/giant_donut_pupil_results.pdf`, 46 pages.

**Spider check** (`20261018_spider_validation`). Each file's `Spider_3D` shadow on the stop
plane is compared with the batoid-traced v4.02 CAD spider planes, rotator 0 deg, offsets
(0, 0), (+4, +4) and (−4, −4) mm. The plan called for each file against its own model,
but `Rubin_v1000_r.yaml` has no spider, so v1000 is checked against the CAD too. The
offset is the vane-shadow position, danish minus batoid, perpendicular to the arm. Strip
agreement is the fraction of rays within one vane width of the batoid shadow on which
the two agree (dimensionless).

| field radius (deg) | v4.02 max offset (mm) | v4.02 strip agreement | v1000 max offset (mm; vane widths) | v1000 strip agreement |
|---|---|---|---|---|
| 0 | 0.0 | 1.0000 | 50.0; 1.00 | 0.25 |
| 1.0 | 0.0 | 1.0000 | 55.6; 1.11 | 0.30 |
| 1.75 | 0.0 | 0.9995 | 59.9; 1.20 | 0.34 |

The defocus offsets change no number. v1000 places its vanes 0.40 m from the axis, the
CAD places them at 0.35 m, so every v1000 shadow sits about one vane width (50 mm) off.
The v4.02 stop rule (offset > 0.050 m) did not trip.

**Fits** (`20261018_fits`, table `fits.parquet`/`fits.csv`):
- 20 donuts on 38 donut sides: 18 on both visits, G0643 intra only, G0676 extra only.
- 4 fits per side, 152 fits in all. 152 of 152 converged, with no failures.
- Settings: full-resolution ISR on the 18 picked detectors; binning 1; spider on, with
  `spider_angle` = rtp (−0.199 deg and −0.046 deg); blur FWHM bounded to 0.1–1.5 arcsec;
  tolerance 1e-3.
- Coefficients are stored in CCS and in OCS (rotated by rtp).
- The neighbour mask (variance 1e20 electrons² within 1.05 r_out of a neighbour) was
  applied on 9 donut sides. It masked part of the target's ring on 4 of them. Fraction
  of the target's ring masked (dimensionless): G0592 extra 0.230, G0606 extra 0.487,
  G0606 intra 0.119, G0635 extra 0.344.
- Blur FWHM at the 1.5 arcsec bound on 20 of 38 donut sides in the blitz set, for both
  pupils, mostly intra-focal.

**Residuals.** Residual RMS is taken inside 1.1 r_out, over unmasked pixels, in
electrons per pixel. "v4.02 better" means lower residual RMS.

| Zernike set | v4.02 better, all | v4.02 better, clean | median v4.02/v1000 residual RMS, all | clean |
|---|---|---|---|---|
| Z4 + Z11 + Z22 | 35 of 38 | 16 of 16 | 0.939 | 0.938 |
| blitz (Noll 4–19, 22–26) | 28 of 38 | 12 of 16 | 0.948 | 0.948 |

The ratios are dimensionless. The per-donut residual images show the v1000 vane shadows
offset from the data, and the v4.02 shadows on it.

Residual RMS scales with brightness. The r = 8.9–10.3 mag donuts reach 540–2900
electrons per pixel, but as a fraction of the ring level (residual RMS over mean ring
level, dimensionless) they sit at 0.10–0.17, like the rest. The one fit that failed in
substance is G0592 intra (marginal, flagged): 1.92 of the ring level.

**Z5–Z8.** For the clean subset with the blitz set:
- The fitted deviations from the batoid reference are within ±3 µm of wavefront.
- No trend with field radius: |Pearson r| ≤ 0.29 and |Spearman rho| ≤ 0.35, against
  field radius (deg), n = 16 per pupil and term.
- Median |v4.02 − v1000| (µm of wavefront): Z5 0.61, Z6 0.19, Z7 0.20, Z8 0.10.
- Median |intra − extra| on the 7 clean donuts seen on both visits (µm of wavefront):
  0.30–0.70 for v1000 and 0.33–1.38 for v4.02.

Beyond 1.9 deg (G0732, G0737) the deviations reach 20–44 µm, with opposite signs intra
and extra. The model does not describe those donuts.

![residual RMS, v1000 against v4.02](figures/resid_rms_v1000_v402.png)
![spider overlay](figures/spider_overlay.png)

Next:
- A fixed-blur pass. The intra-focal blur sits at the 1.5 arcsec bound on most donuts.
- The TMA sections and the silhouette-circle error at > 1.75 deg are the known gaps of
  the v4.02 file.

### 2026-10-18 — plan

**Step 2, selection and display (done, below).**

**Step 3, Danish fits on the donuts Aaron picks.**
- Single-sided, one fit per donut, spider on (`spider_angle` from the rotator angle),
  through `wfs/code/giant_donuts/fit_giant_donut.py` (`fit_stamp`, `pupil_override`),
  with offsets (detector, camera, M2) = (0, ±4, ±4) mm. Corner sensors add their
  ±1.5 mm detector offset. Full-resolution ISR on the chosen detectors only.
- Two Zernike sets: Z4 + Z11 + Z22 only; the full blitz set (Noll 4–19, 22–26).
- Two pupils, so 4 fits per donut:
  - v1000: danish `RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml` and the v1000 batoid model.
  - v4.02: a RubinObsc-format obscuration file built from `LSST_r_v4.02.yaml`, with the
    batoid model swapped to match.
- Building the v4.02 file. ts_wep `utils/maskUtils.fitMaskModel` / `_fitEdges` exists:
  it fits a circle (cubic in field angle, centre and radius) per batoid obscuration.
  danish 1.3 ships only the v1000 and v3.14 files and no generator. v4.02 carries
  106 `ClearPolygon` camera openings, 688 `ObscPolygon` and 440 baffle planes, so a
  per-item fit is not usable as is. Plan:
  1. Fit M1, M2, M3, L1, L2, filter and L3 edges with `_fitEdges`, as v1000 was
     (`wfs/code/giant_donuts/gen_mask_params.py` handles the per-element frame).
  2. Replace `CameraBody` with one circle fitted to the union of the
     `CameraSilhouette_z*` planes, and drop the in-camera openings: they clip no ray at
     any field (`LSST_r_v4.02_changes.md`, validation). Check that.
  3. `Spider_3D` from the CAD: vanes at 0.350 m from the axis, 0.050 m wide, 2.77 m
     long, at the CAD heights (`spider_z.py`).
  4. Validate against batoid as the archived `wfs_batoid_pupil_compare` notebook did:
     agreement fraction of the danish mask against traced vignetting at 0, 1.0, 1.5,
     1.75 deg, with the 4 + 4 mm offsets applied, in the stop-plane frame.
  - Caveat: v4.02's camera openings come from the z-band CAD; these visits are r band.
- Output: one table, one row per (donut, pupil, Zernike set): coefficients (µm of
  wavefront), blur FWHM (arcsec), chi2/dof, residual image.

**Step 4, results.**
- Coma (Z7, Z8) and astigmatism (Z5, Z6) against field position for both pupils:
  focal-plane maps (OCS, deg) and against field radius (deg).
- Intra against extra for the same stars.
- Fit residual maps per donut, and v1000 minus v4.02 residual comparison: residual RMS
  per donut (electrons per pixel) and its difference.

### 2026-10-18 — candidate selection

ISR: overscan, assembly and gains (`donut_images.minimal_isr_config`) on all 197 detectors
of both visits, no failures; cached 4×4 binned. Run `output/20261018_candidates/`.

Measured donut radii, outer / inner half-maximum, full-resolution pixels:

| visit | science | SW0 | SW1 |
|---|---|---|---|
| 337 extra | 340 / 212 | 404 / 248 | 268 / 176 |
| 340 intra | 336 / 208 | 272 / 172 | 400 / 248 |

The corner sensors add their ±1.5 mm detector offset, so SW0 and SW1 swap sizes between
the visits. Raw-WCS refcat positions sit about 4.6 mm (x) and 1.7 mm (y) off the donut
centres in focal-plane coordinates. An affine fit per visit corrects them, with 0.08 mm
scatter (nMAD of the residual radius) over 271–273 matched stars.

Selection: the_monster stars at r = 7.0–14.5 AB mag, whole donut on silicon.
- (a) blitz `DonutSourceSelectorTask`: `unblendedSeparation` and `minBlendedSeparation`
  scaled from the FAM radius (66.5 px) to the giant one; `isolatedMagDiff` 2 mag.
- (b) Image check: mean flux in 1.05–1.5 r_out over the ring's mean (annulus_ratio,
  dimensionless), and the fraction of the ring under another donut (overlap_frac,
  dimensionless).
- Classes: **isolated** = (a) and (b) clean (annulus_ratio < 0.05, overlap_frac < 0.02);
  **marginal** = one of them; **blended** = neither; **saturated** = binned ring
  99th percentile > 1.2e5 electrons per pixel.

Candidate counts:

| region | visit | isolated | marginal | blended | saturated |
|---|---|---|---|---|---|
| science | 337 extra | 148 | 89 | 470 | 1 |
| science | 340 intra | 145 | 92 | 476 | 1 |
| corner WFS | 337 extra | 3 | 1 | 11 | 0 |
| corner WFS | 340 intra | 5 | 4 | 7 | 0 |

738 stars, 715 of them on both visits. 20 proposed (`proposed` column):
- 12 science donuts, isolated on both visits, field radius 0.17–1.98 deg.
- 8 corner WFS donuts, 2 per raft. Only G0618 (R40_SW0) and G0663 (R44_SW1) are isolated
  on both visits. G0643 and G0676 are on silicon in one visit only. G0592, G0606 and
  G0678 are marginal, and G0635 is isolated on intra only. The corner fields are crowded.

### 2026-10-18 — Aaron's picks; the v4.02 obscuration file

**Picks.** Aaron took all 20 proposed donuts (12 science, 8 corner WFS). The selection is
good enough for a first pass and is not re-tuned. Each donut is fitted on every visit
where it is on silicon. Two flags go on every table and plot:
- `beyond_fov`: field radius > 1.725 deg.
- `corner_not_isolated`: a corner donut with iso_class != isolated, overlap_frac > 0.02 or
  annulus_ratio > 0.05 on that visit.

The clean subset is the rows with neither flag.

**v4.02 file** (`code/obscuration_v402.py`, run `output/20261018_obscuration_v402/`,
`RubinObsc_v4.02_r_rtpp0_pp0d0.yaml`; not committed to rubin-optical-model).
- Circles from ts_wep `fitMaskModel` on a copy of v4.02 cut to M1 and its baffles, M2,
  M3, the lenses, the filter and the detector. ts_wep's `_fitCircle` fails under numpy 2
  (a 1-element array goes to `np.arange`). The study carries a fixed copy; ts_wep is
  not edited.
- In-camera openings dropped. Rays they alone block, over transmitted rays
  (dimensionless): 0 at 1.5–1.98 deg for every offset tried.
- Camera silhouette: one circle per field angle on the stop plane, then a cubic. It is
  centred at 0 m with radius 2.078 m at 0 deg, and at −0.795 m with radius 2.130 m at
  1.75 deg. Outside the M1 inner edge the shadow departs from the circle by up to 0.062 m
  (1.75 deg) and 0.081 m (2.0 deg).
- Not represented: the TMA sections. Rays they alone block, over transmitted rays
  (dimensionless): 0.0010 at 1.75 deg and 0.0046 at 1.98 deg.
- `Spider_3D`: the 48 CAD member edges (rubin-optical-model `spider_tilted`) as 50 mm
  vanes, heights from the stop plane. Against the batoid spider planes, the rays they
  classify differently, over transmitted rays (dimensionless), are below 0.0001 at 0 and
  1.75 deg.

**Validation against batoid.** Rays are traced on the stop plane out to 4.3 m. Each
file is compared with its own batoid model (v1000 with `Rubin_v1000_r.yaml`, v4.02 with
`LSST_r_v4.02.yaml`), both without the spider. The table gives the agreement fraction
(dimensionless): rays where danish and batoid agree on transmission, over all rays.
Offsets are (camera, M2) in mm, detector 0. The corner configurations add ±1.5 mm on the
detector, which changes no number in the table. At 1.65 deg the agreement fraction is
0.9988 for v1000 and 0.9963 for v4.02, with any corner offset.

| field radius (deg) | offsets (mm) | v1000 | v4.02 | v4.02 with spider |
|---|---|---|---|---|
| 0 | 0, 0 | 1.0000 | 1.0000 | 1.0000 |
| 0 | ±4, ±4 | 1.0000 | 1.0000 | 1.0000 |
| 1.0 | 0, 0 | 1.0000 | 1.0000 | 1.0000 |
| 1.0 | ±4, ±4 | 0.9998 | 0.9998 | 0.9998 |
| 1.5 | 0, 0 | 0.9999 | 0.9990 | 0.9990 |
| 1.5 | +4, +4 | 0.9990 | 0.9981 | 0.9981 |
| 1.5 | −4, −4 | 0.9992 | 0.9984 | 0.9984 |
| 1.75 | 0, 0 | 0.9990 | 0.9966 | 0.9967 |
| 1.75 | +4, +4 | 0.9848 | 0.9909 | 0.9911 |
| 1.75 | −4, −4 | 0.9838 | 0.9835 | 0.9837 |

The plan's stop rule was "stop if v4.02 agrees with its batoid model worse than v1000
does with its own". That rule tripped. Aaron's call: **continue**. The rule was too
strict. v4.02 is within 0.0025 (agreement fraction, dimensionless) of v1000 everywhere,
and better at 1.75 deg with +4/+4 mm. Both lose about 1.6% at 1.75 deg with −4/−4 mm.
That is the limit of the circle model, and it is the same for both files.
