---
study: unpaired_intra_vs_extra
topic: aos
status: active
question: Does the wavefront measured from the intra-focal donut differ from the
  extra-focal donut of the same FAM pair, and is that difference thermally driven as the
  blur difference is?
answer: Yes to both, with a split. Z14 carries the largest difference
  (-0.5221 um of wavefront, 8.6x its own two-side mean) but only Z11 tracks the blur
  difference within a night and only Z11 responds to the pupil model. Z4 is the surprise:
  a modest median difference with the second-largest robust scatter over visit pairs
  (0.1710 um of wavefront), 25.3 per cent of which the thermal model removes.
uses:
  fam_tables: danish_1_3_v1000@20261013, danish_1_3_test@20261013
started: 2026-10-15
---

# Unpaired intra- vs extra-focal wavefront in the Danish 1.3 blitz FAM

## Question

The Danish 1.3 blitz FAM product is **unpaired**: one row per donut per exposure, each
its own Danish fit, joined across the two sides of focus afterward on
`(det_name, donut_id)`. So each FAM pair yields two independent wavefront measurements of
the same star, one from the intra-focal donut and one from the extra-focal donut.

The earlier blur study found the intra-focal donut is blurrier by a median
+0.3440 arcsec, that the difference is **not constant**, and that a Huber model on
thermal telemetry removes the residual correlation — so thermal effects drive the
asymmetry rather than the overall seeing.

Does the same hold for the Zernike coefficients?

## Current answer

Over **872 FAM visits**, 15 nights, `day_obs` 20260315–20260619,
`danish_1_3_v1000@20261013`, per-visit medians over ~3,400 donuts each (min 1,706,
max 3,455). 873 visits pass `visit_quality_pass`; one more is dropped as a clear
outlier — `day_obs` 20260327 `seq_num` 228, two-side mean Z4 = **+4.4153 µm of
wavefront**, a robust z of +11.8 (dimensionless), with Z6 at −11.7 and Z5 at −6.1. The
next largest |mean Z4| in the sample is +1.089 µm of wavefront, so it is isolated by a
factor of 4 rather than being the tail of a distribution.

**The two sides disagree, and the disagreement is a measurement, not a model artifact.**
Each side's intrinsic is evaluated at its own field angle (median 4.36 arcsec apart), and
the intrinsic difference is ≤0.0001 µm of wavefront in the median against signals of
0.1–0.5 µm — so the total and the intrinsic-subtracted deviation give the same difference
to four decimals. The paired wavefront is exactly the two-side mean (to
1.8e-15 µm of wavefront).

**Z14 is the largest difference, not Z11.** Ranked by
`|median(intra − extra)| / median|(intra + extra)/2|` (dimensionless):

| Noll | median(intra − extra) [µm wf] | nMAD over visit pairs [µm wf] | median abs two-side mean [µm wf] | ratio (dimensionless) |
|---|---|---|---|---|
| Z14 | −0.5221 | 0.0758 | 0.0609 | 8.57 |
| Z26 | +0.1331 | 0.0226 | 0.0215 | 6.18 |
| Z6 | −0.3276 | 0.2201 | 0.1899 | 1.73 |
| Z11 | +0.1831 | 0.0231 | 0.1116 | 1.64 |
| Z15 | −0.0605 | 0.0517 | 0.0404 | 1.50 |
| Z22 | −0.0265 | 0.0087 | 0.0201 | 1.31 |

Six terms exceed their own two-side mean. Z14's difference is 8.6× its mean with a
nMAD/median ratio of 0.145, so it is a clean systematic offset and not scatter. Z11 is
fourth by this measure, though it has the tightest scatter of the large terms (0.126).

**Large robust scatter is a second, independent way to be interesting**, and it is what
flags Z4. Ranked by the nMAD of the difference over visit pairs [µm of wavefront]: Z6
0.2201, **Z5 0.1711, Z4 0.1710**, Z7 0.0962, Z9 0.0841, Z10 0.0803, Z14 0.0758, Z8 0.0700.
Z4's median difference is modest (+0.0900 µm of wavefront, 0.33 of its own mean) but its
excursions over visits are among the largest in the sample — visible immediately in the
ordinal plots. So the follow-up set is chosen on **either** test: ratio ≥ 1 or nMAD in the
top 8, giving **12 Noll indices: 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 22, 26**. Z6 and Z14
qualify on both.

**But only Z11 behaves like the blur.** Correlation of the per-visit Zernike difference
against the per-visit blur difference (Spearman rho, n = 872 visits):

| Noll | rho, full sample | Huber slope [µm wf per arcsec] |
|---|---|---|
| Z11 | **+0.5191** | +0.0710 ± 0.0038 |
| Z14 | −0.4199 | −0.1812 ± 0.0135 |
| Z26 | +0.3365 | +0.0452 ± 0.0042 |
| Z7 | −0.2939 | −0.1750 ± 0.0174 |
| Z6 | −0.1493 | −0.2080 ± 0.0418 |
| Z4 | +0.1053 | +0.1077 ± 0.0323 |

Z11 has the strongest full-sample correlation and is the only term that survives
within-night de-meaning at any strength. Z14 and Z26 correlate in the full sample but
collapse within a night, so theirs is night-level covariance with the seeing rather than
a visit-by-visit link.

**Only Z11 responds to the pupil model.** `v1000` (as-built + M1M3 + M1 baffles) against
the blitz default, identical 872 visits, change in median(intra − extra): Z11
**−0.0752 µm of wavefront** (+0.2583 → +0.1831, a 29 per cent reduction). The next
largest change is Z22 at +0.0081 µm of wavefront, nine times smaller; Z14 moves by
+0.0003 µm of wavefront, which is nothing. So the pupil model is specific to Z11 — and
its 29 per cent reduction here is consistent in direction with the 8 mm giant-donut
result (0.81 → 0.37 µm of wavefront, 54 per cent) at a different defocus and sample.

**Thermal correction narrows every difference, but the blur most.** Greedy forward
selection on incremental gain, 22 candidate thermal quantities, Huber fits, 854 visits
with complete telemetry. All 12 follow-up terms plus the blur, ordered by the fractional
reduction in robust scatter, `n = 5` predictors maximum, stopping early when the next
candidate adds less than 1 per cent of the current scatter:

| target | leading predictor | its own rho | nMAD before | nMAD after | reduction |
|---|---|---|---|---|---|
| Δblur [arcsec] | `cam_body_minus_m1m3_air` | −0.7765 | 0.19967 | 0.08351 | **58.2 %** |
| ΔZ22 [µm wf] | `m1m3_z_gradient_c_per_m` | +0.5412 | 0.00866 | 0.00499 | 42.3 % |
| ΔZ11 [µm wf] | `m1m3_r2_coeff_c` | −0.3875 | 0.02314 | 0.01574 | 32.0 % |
| ΔZ14 [µm wf] | `cam_body_minus_m1m3_air` | +0.5433 | 0.07577 | 0.05552 | 26.7 % |
| ΔZ4 [µm wf] | `cam_air_minus_outside` | −0.4357 | 0.17098 | 0.12776 | 25.3 % |
| ΔZ7 [µm wf] | `m1m3_z_gradient_c_per_m` | −0.3758 | 0.09618 | 0.07701 | 19.9 % |
| ΔZ6 [µm wf] | `m1_mean_temp_c` | −0.2916 | 0.22014 | 0.18053 | 18.0 % |
| ΔZ26 [µm wf] | `cam_body_minus_m1m3_air` | −0.4441 | 0.02255 | 0.01887 | 16.3 % |
| ΔZ9 [µm wf] | `m1m3_x_gradient_c_per_m` | +0.0853 | 0.08413 | 0.08041 | 4.4 % |
| ΔZ15 [µm wf] | `m1m3_radial_gradient_c_per_m` | −0.0885 | 0.05171 | 0.05002 | 3.3 % |
| ΔZ10 [µm wf] | `m2_air_minus_m1m3_air` | −0.0403 | 0.08029 | 0.07871 | 2.0 % |
| ΔZ8 [µm wf] | `m1m3_radial_gradient_c_per_m` | −0.0426 | 0.07005 | 0.06872 | 1.9 % |
| ΔZ5 [µm wf] | `m1m3_z_gradient_c_per_m` | +0.0733 | 0.17108 | 0.16845 | 1.5 % |

**Four distinct thermal channels lead across the 12 terms**: camera-body-minus-M1M3-air
(Δblur, ΔZ14, ΔZ26), the M1M3 z gradient (ΔZ22, ΔZ7, ΔZ5), the M1M3 r² coefficient (ΔZ11)
and camera-air-minus-outside (ΔZ4). Z5, Z8, Z9, Z10 and Z15 have large raw scatter but are
barely thermal at all (≤ 4.4 per cent reduction) — so large scatter alone does not imply a
thermal origin, and those five are the ones whose scatter still needs an explanation.

**The blur and Z11 have different leading thermal channels.** Δblur's is the camera body
minus M1M3 air temperature (rho = −0.7764, reproducing the earlier study's −0.7828); it
alone cuts the blur nMAD by 44 per cent. ΔZ11's leading channel is the M1M3 r² thermal
coefficient instead, and camera-body-minus-M1M3-air does not lead for it. So the blur
asymmetry and the Z11 asymmetry are **both thermally sensitive but not through the same
channel** — they are related, not identical.

**The correction flattens the residual correlation**, repeating the blur study's test.
Huber slope of the difference against the extra-focal value, before → after:

| target | before | after | unit |
|---|---|---|---|
| Δblur | −0.5767 ± 0.0342 | −0.2058 ± 0.0180 | arcsec per arcsec |
| ΔZ11 | −0.1115 ± 0.0112 | −0.0563 ± 0.0074 | µm wf per µm wf |
| ΔZ7 | −0.2141 ± 0.0152 | −0.0907 ± 0.0145 | µm wf per µm wf |
| ΔZ4 | +0.0348 ± 0.0133 | +0.0021 ± 0.0112 | µm wf per µm wf |
| ΔZ22 | +0.1426 ± 0.0092 | −0.0197 ± 0.0057 | µm wf per µm wf |
| ΔZ14 | −0.0438 ± 0.0334 | −0.2432 ± 0.0246 | µm wf per µm wf |
| ΔZ26 | −0.0600 ± 0.0267 | −0.2214 ± 0.0219 | µm wf per µm wf |

Δblur and ΔZ11 both flatten by about half; **ΔZ4 and ΔZ22 flatten essentially to zero**
(+0.0348 → +0.0021 and +0.1426 → −0.0197 µm wf per µm wf), the cleanest results in the
set. Against the **two-side mean** the flattening is cleaner still for Z14 and Z26
(+0.3561 → +0.0389 and +0.2244 → −0.0010 µm wf per µm wf), so for those two the thermal
model removes essentially all of the dependence on the wavefront level. ΔZ14 and ΔZ26
*gain* slope against the extra-focal value, which the figures should be read against
before drawing any conclusion.

Open: Z14's difference is the largest, is thermally sensitive, but is untouched by the
pupil model and does not track the blur within a night. It is not explained here.

## Log

### 2026-10-15 — the Zernike extension, first pass

One notebook, one PDF (21 pages), four parquet summaries. Run directory
`output/20261015_visit_medians/`.

- `extract_visit_medians.py` reduces the 13 GB unpaired donut table to one row per visit
  per side: blur [arcsec] and the 21 Noll Zernikes [µm of wavefront], median over the
  visit's paired donuts inside 1.725 deg OCS field radius. One pass per variant, column
  projection, one row group (one visit) at a time, about 10 min each.
- One visit dropped as an outlier in the notebook, not in the extraction, so the cut is
  visible and reversible: `day_obs` 20260327 `seq_num` 228 (`EXCLUDED_VISITS`).
- Robust scatter is `scipy.stats.median_abs_deviation(..., scale='normal')` throughout,
  wrapped as `nmad` in `thermal_selection.py` rather than hand-rolled.
- `thermal_selection.py` holds the greedy forward selection described below. Developed
  here; intended for `rubinwork` once used on a second study.
- `notebooks/unpaired_intra_vs_extra.ipynb` produces every figure into the single PDF.
  Assembled by `notebooks/_build_nb.py` so it can be regenerated rather than hand-edited
  as JSON.

Reproduction checks: median(intra − extra) blur **+0.3441 arcsec** on `danish_1_3_test`
(+0.3449 on v1000), intra blurrier in **0.9954** (dimensionless) of 872. Before the
outlier cut the `test` value was +0.3440 arcsec, exactly the blur study's number; the cut
moves it by 0.0001 arcsec.
The donut-median route agrees with the `visits.parquet` columns to four decimals, and the
paired wavefront equals the two-side mean to 1.8e-15 µm of wavefront.

Results as in **Current answer** above.

**The follow-up set is chosen from the data.** A Noll index enters sections 3-5 on either
a difference at or above its own two-side mean (ratio ≥ 1) or a robust scatter in the top
8. That gives 12 terms rather than a list fixed in advance, and it is the scatter test that
brings in Z4 (Aaron, 2026-10-15).

**The thermal selection algorithm changed.** The earlier study ranked candidates by their
own correlation with the target and added them in that order; the thermal quantities are
themselves strongly correlated, so that double-counts. `select_predictors` now (a) ranks
by own Spearman rho and **fixes each sign before any orthogonalisation**, (b) takes the
leader, then re-ranks the remaining candidates by what each *adds* given those already
chosen — the reduction in Huber residual nMAD — and takes the largest, (c) continues to a
settable `n_max`. Each step reports the incremental reduction in the target's unit, the
partial correlation against the current residual, and a variance inflation factor. The
VIF earns its place: `truss_temp_mean_c` enters ΔZ11's fifth step at **VIF = 80.0**
(dimensionless), which the earlier ordering would have added silently.

Because the algorithm is deliberately different, the blur flattening does not reproduce
the earlier study's coefficients (−0.5788 → −0.0868 arcsec per arcsec there,
−0.5771 → −0.2058 here). The **conclusion** reproduces; the numbers do not, and should not
be quoted as if they did.

**Scope.** Only the earlier aos `blur_intra_extra` work is incorporated
(Aaron, 2026-10-15). `wfs/giant_donuts`, `optics/intra_extra_phase_strip` and the Z11/Z14
part of aos `cwfs` stay as they are and none of their work is included; the giant-donut
number above is cited for direction only. Triage question 6 is not answered here.

**Visit-level only** (Aaron, 2026-10-15). The pair-level (per-donut) analysis, which would
bound Danish's blur-against-Zernike fit degeneracy using within-visit variation, is not
done.

### 2026-10-06 — the blur study (carried over)

Predecessor, run on `danish_1_3_test`:
`aos/notebooks/fam_processing/blur_intra_extra_offset.ipynb`, output
`aos/output/fam_processing/blur_intra_extra/`, handoff
`aos/docs/status/blur_intra_extra_handoff.md` (not yet folded in here; still the full
record of the thermal-regression detail and the "tried and rejected" list).

873 quality triplets, 15 nights: median intra blur 1.0526 arcsec, extra 0.6636 arcsec,
difference **+0.3440 arcsec**, intra blurrier in 0.9954 (dimensionless). A quadrature
model `b_i² = b_e² + q` with **q = +0.5614 arcsec²** beat a constant offset and a
fractional scaling on robust residual scatter; the discriminator was a Huber slope of
(intra − extra) on extra blur of **−0.5764 ± 0.0344** (dimensionless), 17σ from zero.
Camera body minus M1M3 air was the strongest single thermal quantity (Spearman
rho = −0.7828 full sample, −0.2934 within night), and the fitted thermal model flattened
the residual slope from −0.5788 ± 0.0344 to −0.0868 ± 0.0171 arcsec per arcsec —
**thermal effects drive the difference, not the overall seeing**.

Also established there and relied on here: the two exposures of a pair are adjacent in
sequence number and see the **same thermal state** to a few parts in a thousand (nMAD of
the per-side difference over the nMAD of the quantity: 0.0013 mean truss temperature,
0.0016 M1M3 glass, 0.0020 M1M3 air). So one side's telemetry stands for the pair, and
there is no per-side thermal difference to find.

## Next

1. Explain Z14. Largest difference (8.6× its own mean), thermally sensitive, untouched by
   the pupil model, no within-night blur link. Nothing here accounts for it.
2. Explain Z4's scatter. Second-largest robust scatter over visit pairs
   (0.1710 µm of wavefront) on a modest median difference, so the two sides disagree about
   defocus by a varying amount visit to visit. Its leading thermal channel
   (`cam_air_minus_outside`) is distinct from the blur's, and the correction takes
   25.3 per cent of the scatter and flattens the residual against the extra-focal value to
   +0.0021 ± 0.0112 µm wf per µm wf. What the remaining 75 per cent is, is open.
3. Explain the five terms with large scatter but no thermal response — Z5, Z8, Z9, Z10,
   Z15, all at or below 4.4 per cent reduction. Large scatter without thermal sensitivity
   is a different kind of problem from the rest of this study.
4. Read the ΔZ14 / ΔZ26 "after" panels against the extra-focal value — their slopes grow
   under the correction while flattening against the two-side mean. Decide which
   reference is the right one before quoting either.
5. Field dependence, not started: per-detector maps of each difference in OCS and in CCS
   would separate a telescope-fixed from a camera-fixed pattern. The rotator spans enough
   angle over 15 nights.
6. Pair level, if wanted later: the within-visit slope of ΔZ11 on Δblur bounds the Danish
   fit degeneracy, since within one visit the wavefront is common and only the blur varies
   donut to donut.
7. Move `select_predictors` into `rubinwork` once a second study uses it.

## Files

| file | role |
|---|---|
| `extract_visit_medians.py` | the one pass over the 13 GB donut table; writes per-visit per-side medians |
| `thermal_selection.py` | greedy forward selection on incremental gain, Huber throughout |
| `notebooks/unpaired_intra_vs_extra.ipynb` | every figure and table, into one PDF |
| `notebooks/_build_nb.py` | assembles the notebook from cell sources |
| `output@/20261015_visit_medians/unpaired_intra_vs_extra.pdf` | 21 pages, all figures |
| `output@/20261015_visit_medians/*.parquet` | the two per-visit median tables and four summaries |
