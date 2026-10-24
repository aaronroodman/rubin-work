# Giant-donut pupil: findings ledger

> **Status:** current through pupil pass 6 and the night 2 review (2026-10-23); study parked.
> One row per finding, from the selection work, pupil passes 1–6, the model reviews and night 2. Superseded findings stay, with what replaced them.

Data: 20251023 seq 337 (extra-focal) and 340 (intra-focal), ±4 mm camera and M2. "Extra /
intra" pairs give the extra-focal value first. Pixels are 10 µm detector pixels; OCS image
means the image-plane axes in the telescope frame. Evidence cites `<run>/<file>.pdf p<N>`
(read by `code/figure_index.py`) and the tables in that run. Methods and the conventions in
force are in `method.md`; the full numbers are in the `study.md` Log entry of each pass.

Status: **settled** (measured and checked, no open alternative), **open** (measured, cause
or size not pinned), **superseded** (replaced; the row names by what).

## Selection

| # | claim | status | evidence | conventions |
|---|---|---|---|---|
| S1 | Measured donut radii, outer / inner half-maximum: science 340 / 212 pixels (337) and 336 / 208 (340); the corner sensors swap with focus (SW0 404 / 248 extra, 272 / 172 intra) | settled | 20261018_candidates/candidates_2025102300337_extra.pdf p1; Log 2026-10-18 | full-resolution pixels |
| S2 | Raw-WCS refcat positions sit about 4.6 mm (x) and 1.7 mm (y) off the donut centers; one affine per visit corrects them to 0.08 mm (nMAD, 271–273 stars) | settled | 20261018_candidates; Log 2026-10-18 | focal-plane mm, CCS |
| S3 | Final selection rule (^ with `overlap_frac_best < 0.02`, corner v from the sky-corrected metrics): science purity 0.976, completeness 0.965 (83 good, 2 bad, 3 missed); corner 1.000 and 0.875 (7 / 0 / 1), against Aaron's hand labels | settled | 20261020_candidates_v2/score_summary.pdf p1; 20261020_candidates_v2/tally.pdf p1; Log 2026-10-21 | purity and completeness dimensionless; `method.md` section 9 |
| S4 | The 5-nMAD peak threshold is not used: crowding inflates the score nMAD | settled | 20261020_selection_diag/diag_2025102300337_extra.pdf; Log 2026-10-20 (check B) | |

## Pupil model and wavefront

| # | claim | status | evidence | conventions |
|---|---|---|---|---|
| P1 | v4.02 fits better than v1000: lower residual RMS on 16 of 16 clean sides (Z4 + Z11 + Z22) and 12 of 16 (blitz set); median ratio v4.02 / v1000 0.94 and 0.95 (dimensionless). Most of the gain is the spider: the v1000 vanes sit 50 mm off the CAD shadow | settled | 20261018_fits/giant_donut_pupil_results.pdf p1-2; 20261018_spider_validation/spider_validation.pdf p1 | residual RMS inside 1.1 r_out, e per pixel |
| P2 | With Z4 only, v4.02 still wins on 35 of 38 sides (median ratio 0.944, dimensionless) | settled | 20261018_pupil/giant_donut_pupil_model.pdf p1-2; Log 2026-10-19 | |
| P3 | Inside 1.725 deg the fitted Z5–Z8 deviations stay within ±3 µm and show no trend with field; beyond 1.9 deg the fits break down | settled | 20261018_fits/giant_donut_pupil_results.pdf p5-7 | µm of wavefront, OCS |
| P4 | The 1–3 µm Z5–Z7 at large field in the per-side fits is wavefront | superseded (by P5: pupil bias) | 20261018_fits/giant_donut_pupil_results.pdf p5-7 | |
| P5 | The large outer-band Z5–Z7 is pupil bias, not wavefront: the central-band medians (Z5 −0.22, Z6 −0.13, Z7 +0.22, Z9 −0.30 µm) cost 6 % in chi2/dof against fit c's 1–3.9 µm on intra sides beyond 1.5 deg; the danish fits to Batoid design sims put up to −0.17 µm of Z5 into the v4.02g edge miss at 1.7 deg | settled | 20261021_pupil4/pupil_pass4.pdf p10-12; 20261023_pupil6/pupil_pass6.pdf p12 | µm of wavefront, OCS; chi2/dof dof about 7.4e5 |
| P6 | danish v4.02g (masks fitted at the giant configuration) fixes the outer edges v4.02 missed at 1.73–1.75 deg (M3, filter); the in-camera openings block no ray uniquely | settled | 20261019_pupil2/giant_donut_pupil_pass2.pdf p13-15; Log 2026-10-20 | |
| P7 | Inner-minus-outer edge-center shift of +3.3 pixels along +y CCS on both sides of focus at 0.17–1.70 deg reads as Z7 +0.71 / +0.81 µm, but danish fits with Z7 free find +0.02 / +0.19 µm: it has the signature of coma but is not wavefront coma | settled | 20261019_pupil2/giant_donut_pupil_pass2.pdf p1-6 | pixels CCS; µm of wavefront |
| P8 | The edge and spider shifts are coma | superseded (by P7 and P23: coma moves the inner edge and the spiders the same way relative to the outer edge, the data move them opposite; it needs about 0.7 µm of Z7 against 0.14–0.24 µm fitted) | 20261019_pupil2/giant_donut_pupil_pass2.pdf p2-4; Log 2026-10-22/23 | |
| P9 | The seq 324 optical state is the in-focus reference | superseded (by seq 336, the in-focus image just before the giant pair; Aaron 2026-10-20) | 20261019_pupil2/giant_donut_pupil_pass2.pdf p2 | |
| P10 | Not the telescope state: the seq 324 state moves the sector offsets by under 1 pixel; adding the Trim tilts makes 38 of 38 fits worse | settled | 20261019_pupil2/giant_donut_pupil_pass2.pdf p2-7 | |
| P11 | The data's r_inner / r_outer is 0.0104 above the model on science donuts (median, nMAD 0.0048, dimensionless), flat in field (Pearson r −0.15, Spearman rho −0.21, n 34) | settled | 20261020_pupil3/pupil_pass3.pdf p101-102 | intrinsicTA, v4.02g, fit a |
| P12 | Freeing Z11 and Z22 removes most of the ratio excess (to 0.003), but Z11 comes out −0.15 / +0.15 µm: a sign that flips with focus is the pupil, not the wavefront | settled | 20261020_pupil3/pupil_pass3.pdf p104-105 | µm of wavefront |
| P13 | Fit d (edges only) finds Z7 +0.76 µm, the whole-donut fits +0.14 to +0.24 µm, and fit d's whole-donut chi2/dof is 1.36 times fit a's: the coma fits the edges, not the donut | settled | 20261020_pupil3/pupil_pass3.pdf p106-107 | chi2/dof ratio dimensionless |
| P14 | The central edge mismatch is in the image frame, not the pupil: data − fit c agrees between extra and intra in OCS image azimuth (Pearson r +0.76 inner edge, +0.77 ratio), and anti-correlates in the pupil frame (−0.95, −0.91) | settled | 20261021_pupil4/pupil_pass4.pdf p3-4 | OCS image azimuth |
| P15 | Relative azimuth ψ stacks the outer band: outer-edge scatter between donuts 1.2–1.4 pixels in ψ against 7.1–7.5 in OCS | settled | 20261021_pupil4/pupil_pass4.pdf p5-9 | ψ as in `method.md` |
| P16 | Where the outer-band inner edge misses (ψ 330–30 deg intra, 160–200 deg extra) the camera silhouette sets it, and the v4.02g circle sits 3.3–3.9 pixels inside Batoid | settled | 20261022_pupil5/pupil_pass5.pdf p1-4 | danish − Batoid, pixels |
| P17 | v4.02s (CameraSilhouette refitted above 1.4 deg to the Batoid inner edge): danish − Batoid 0.0 pixels median in silhouette-set bins (worst −0.52), against −1.76 to −5.26 with v4.02g | settled | 20261023_pupil6/pupil_pass6.pdf p39-40 | |
| P18 | With v4.02s the data inner-edge miss in the silhouette range goes from +3.05 to −0.28 ± 0.57 pixels extra; intra from +5.51 to +1.93 ± 0.39. The intra remainder fits a silhouette radius +14.3 ± 4.4 mm and center −10.7 ± 4.6 mm (stop plane) | open | 20261023_pupil6/pupil_pass6.pdf p39 p41 | 17 science sides per side, 1.6–1.725 deg |
| P19 | danish reproduces its own Batoid design sim to the noise below 1.4 deg (chi2/dof 0.87); above, v4.02s gives 0.90 median, but 2.1–2.8 remains at 1.69–1.70 deg, on rays first clipped by TMASection and M1Baffle1, which the circle mask does not have | open | 20261023_pupil6/pupil_pass6.pdf p12-38 | chi2/dof dof 3.8e5 (corner), 7.3–7.4e5 (science) |
| P20 | Full ISR (bias, dark, flat, defects, linearity, crosstalk, CTI, PTC gains) changes no pupil result beyond the errors; fit c chi2/dof falls from 4.93 to 4.06 median | settled | 20261023_pupil6/pupil_pass6.pdf p2-3 | |

## Spider shadows

| # | claim | status | evidence | conventions |
|---|---|---|---|---|
| V1 | The data shadows sit about 1.8 pixels toward −y OCS image against fit c on both sides of focus: an m = 1 term of 0.36–0.41 deg in azimuth at r_half; per-vane residuals 0.11 deg at most, no field dependence | settled | 20261021_pupil4/pupil_pass4_spiders.pdf p32-38 | fit-a center, OCS image |
| V2 | The spider offset is a rotator offset | superseded (by V3: a rotation leaves the m = 1 term unchanged; a translation removes it) | 20261022_pupil5/pupil_pass5.pdf p52-53 | |
| V3 | The spider offset is a translation of the shadow pattern, dy −1.65 ± 0.05 / −2.07 ± 0.08 pixels OCS image, which removes the m = 1 term (0.35 / 0.42 deg to 0.003 deg) | settled | 20261022_pupil5/pupil_pass5.pdf p54-56; 20261023_pupil6/pupil_pass6.pdf p3 | fit-a center; rtp = physical_rotator_angle |
| V4 | The visitInfo rtp is the rotator angle | superseded (by physical_rotator_angle from ConsDB `cdb_lsstcam.visit1_quicklook`: +0.211 / +0.189 deg from visitInfo, matching the EFD) | 20261022_pupil5/pupil_pass5.pdf p53; 20261023_pupil6/pupil_pass6.pdf p4 | deg |
| V5 | With physical_rotator_angle the spider rotation δ is −0.020 / +0.075 deg (nMAD 0.050 / 0.066, error 0.007 / 0.008, n 91 / 97); corrected for the sim-measured recovery (0.76 / 0.86, dimensionless) −0.026 / +0.087 deg. Intra − extra 0.11 deg is unexplained | open | 20261023_pupil6/pupil_pass6.pdf p4-5 p10-11 | danish spider_angle = rtp + δ |
| V6 | danish has no spider modeling bias on the design pupil: sims give m = 1 0.001 / 0.006 deg, dy −0.01 / −0.05 pixels, per vane within 0.021 deg | settled | 20261023_pupil6/pupil_pass6.pdf p6-9 | 26 sim sides |
| V7 | The TEA decenter causes the spider translation | superseded (by V8: a TEA shift flips sign with focus; the data do not) | rubin-optical-model studies/tea_decenter | |
| V8 | A 10 mm TEA shift moves the shadows −0.83 / +0.83 pixels in the image (sims; Batoid 0.88 on axis), opposite signs; the data's flipping part (−0.13 ± 0.03 pixels y) puts the TEA within about 1.5–2 mm of center, so the common −1.9 pixel translation is not the TEA | settled | 20261023_pupil6/pupil_pass6.pdf p10-11 | about the image, not the fit-a center (the shade moves the center by −0.25 / +0.17 pixels) |
| V9 | Not the detector: the translation is the same in both amplifier halves and both vendors (extra dy −1.70 / −1.64 E2V, −1.75 / −1.57 ITL) | settled | 20261023_pupil6/pupil_pass6.pdf p44 | |
| V10 | The rotator motion during the 60 s exposure (+0.019 / +0.025 deg) is negligible: 0.1 pixels at r_half, at most 0.01 deg of δ | settled | Log 2026-10-22/23 | |
| V11 | The shadow width is not blur: with blur free, extra stays at 1.00 arcsec and a data / model width ratio of 1.74; intra fits 1.60 arcsec but then its shadows are too shallow | settled | 20261022_pupil5/pupil_pass5.pdf p57-58 | |
| V12 | A free vane width fits 102.8 mm extra (nMAD 3.6, n 26) and 61.3 mm intra (nMAD 2.0, n 27) against CAD 50 mm; sims return 49.7 / 49.9 mm. The shadow equivalent width is 10.0 / 5.4 pixels against 4.1 in the model: different on the two sides of focus, which an opaque vane cannot do | settled | 20261023_pupil6/pupil_pass6.pdf p42-43 | mm in the pupil; depth dimensionless |
| V13 | Diffraction (Batoid Huygens, 570–670 nm) moves the edges +0.31 / +0.26 pixels inner and −0.37 / −0.28 outer and gives ratio excess +0.002 / +0.001: about half the data's edge miss and ratio excess, same signs | settled | 20261023_pupil6/pupil_pass6.pdf p44-45 | v4.02s fit to the sims; 26 sides |
| V15 | A phase strip along the vanes of +0.10 to +0.15 µm of optical path (colder or denser air) gives the extra/intra shadow difference: equivalent width 8.8–9.6 / 3.6–4.5 pixels against the data's 10.0 / 5.7. Its physical cause (temperature of the vanes or the air) is not measured | open | 20261023_pupil6/pupil_pass6.pdf p44-45 | stop-plane Gaussian sigma 40–90 mm |
| V16 | Offset phase strips move the shadow pattern by 0.13–0.27 pixels, same sign on both sides, so they cannot give the data's 1.65–2.07 pixel translation | settled | 20261023_pupil6/pupil_pass6.pdf p44-45 | |
| V14 | Cause of the common translation (dy −1.65 / −2.07 pixels) and the inner-edge shift (+2.1 / +1.8 pixels) | open | 20261023_pupil6/pupil_pass6.pdf p7 p44 | not danish, TEA, coma, rotator, detector or ISR |

## Models and viewers

| # | claim | status | evidence | conventions |
|---|---|---|---|---|
| M1 | danish v1000's single `CameraBody` disk (Batoid radius 0.80469 m at z 3.500 m) is within 0.2–1.1 pixels of v4.02s at 1.4–1.75 deg (outermost stop-plane point −13.7 to −1.9 mm off); v4.02g's circle was 43–66 mm (3.5–5.4 pixels) short there. The 165 v4.02 slices add little over one disk for this edge | settled | camera_body_review.md section 3 | stop-plane mm along −x, extra-focal giant configuration; 12.3 mm is about 1 pixel |
| M2 | Two 3D viewers: `figures/pupil_mask_viewer.html` (the danish masks) and `figures/rubin_model_geometry.html` (every v1000 and v4.02 surface and obscuration, ray fans at 0 and 1.7 deg); build scripts in `code/viewers/` | settled | commit 9b554e0; camera_body_review.md section 2 | telescope frame |

## Night 2

| # | claim | status | evidence | conventions |
|---|---|---|---|---|
| N1 | Of 38 giant-donut sets on 15 nights other than 20251023: 22 extra-only, 8 perturbed, 3 other defocus, 3 not giant, 2 at 8 mm. No night has a usable intra-focal set. Chosen (Aaron 2026-10-23): 20260127 T680 r, 4 mm, RA 185 deg (16 exposures, galactic latitude +61.8 deg, rotator −47.9 to +59.0 deg), its 8 mm seq 220–222, and 20250920 z 8 mm seq 274 | settled | 20261023_night2_review/night2_review.pdf p1-4; choice.json | sets as `night2_review.classify` |
| N2 | Pass 6 analysis on other nights, extra-focal only: 20251027 i band (M2 dx 0 and ±1000 µm, 59–79 sides per group) gives spider dy −1.69 to −1.80 pixels and inner-edge dy +1.81 to +2.05 pixels OCS image, against night 1's −1.65 and +2.10; the free vane width is 76–92 mm against night 1's 103 mm. 20260127 is not yet fitted | open | 20261023_night2_20251027/night2_20251027.pdf p2-9; 20261023_night2_20250920/night2_20250920.pdf p2 | as pass 6; preliminary, run before the model rebuild |
