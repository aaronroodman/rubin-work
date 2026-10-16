---
study: unpaired_intra_vs_extra
topic: aos
status: active
question: Does the wavefront measured from the intra-focal donut differ from the
  extra-focal donut of the same FAM pair, and is that difference thermally driven as the
  blur difference is?
answer: Yes to both, with a split. Z14 carries the largest difference
  (-0.5219 um of wavefront, 8.6x its own two-side mean) but only Z11 tracks the blur
  difference within a night and only Z11 responds to the pupil model.
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

Over **873 FAM visits passing `visit_quality_pass`, 15 nights, `day_obs`
20260315–20260619**, `danish_1_3_v1000@20261013`, per-visit medians over ~3,400 donuts
each (min 1,706, max 3,455):

**The two sides disagree, and the disagreement is a measurement, not a model artifact.**
Each side's intrinsic is evaluated at its own field angle (median 4.36 arcsec apart), and
the intrinsic difference is ≤0.0001 µm of wavefront in the median against signals of
0.1–0.5 µm — so the total and the intrinsic-subtracted deviation give the same difference
to four decimals. The paired wavefront is exactly the two-side mean (to
1.8e-15 µm of wavefront).

**Z14 is the largest difference, not Z11.** Ranked by
`|median(intra − extra)| / median|(intra + extra)/2|` (dimensionless):

| Noll | median(intra − extra) [µm wf] | nMAD over visits [µm wf] | median abs two-side mean [µm wf] | ratio (dimensionless) |
|---|---|---|---|---|
| Z14 | −0.5219 | 0.0756 | 0.0607 | 8.60 |
| Z26 | +0.1332 | 0.0225 | 0.0215 | 6.18 |
| Z6 | −0.3275 | 0.2206 | 0.1904 | 1.72 |
| Z11 | +0.1831 | 0.0232 | 0.1115 | 1.64 |
| Z15 | −0.0607 | 0.0518 | 0.0403 | 1.50 |
| Z22 | −0.0265 | 0.0087 | 0.0199 | 1.33 |

Six terms exceed their own two-side mean. Z14's difference is 8.6× its mean with a
nMAD/median ratio of 0.14, so it is a clean systematic offset and not scatter. Z11 is
fourth by this measure, though it has the tightest scatter of the large terms.

**But only Z11 behaves like the blur.** Correlation of the per-visit Zernike difference
against the per-visit blur difference (Spearman rho, n = 873 visits):

| Noll | rho, full sample | rho, within night | Huber slope [µm wf per arcsec] |
|---|---|---|---|
| Z11 | **+0.5195** | **+0.1863** | +0.0711 ± 0.0038 |
| Z14 | −0.4201 | −0.0404 | −0.1813 ± 0.0135 |
| Z26 | +0.3362 | +0.0588 | +0.0452 ± 0.0043 |
| Z6 | −0.1503 | −0.0314 | −0.2092 ± 0.0418 |

Z11 has the strongest full-sample correlation and is the only term that survives
within-night de-meaning at any strength. Z14 and Z26 correlate in the full sample but
collapse within a night, so theirs is night-level covariance with the seeing rather than
a visit-by-visit link.

**Only Z11 responds to the pupil model.** `v1000` (as-built + M1M3 + M1 baffles) against
the blitz default, identical 873 visits, change in median(intra − extra): Z11
**−0.0752 µm of wavefront** (+0.2583 → +0.1831, a 29 per cent reduction). The next
largest change is Z22 at +0.0081 µm of wavefront, nine times smaller; Z14 moves by
+0.0003 µm of wavefront, which is nothing. So the pupil model is specific to Z11 — and
its 29 per cent reduction here is consistent in direction with the 8 mm giant-donut
result (0.81 → 0.37 µm of wavefront, 54 per cent) at a different defocus and sample.

**Thermal correction narrows every difference, but the blur most.** Greedy forward
selection on incremental gain, 22 candidate thermal quantities, `n = 5` predictors,
Huber fits, 855 visits with complete telemetry:

| target | leading predictor | its own rho | nMAD before | nMAD after | reduction |
|---|---|---|---|---|---|
| Δblur [arcsec] | `cam_body_minus_m1m3_air` | −0.7764 | 0.19939 | 0.08374 | 58.0 % |
| ΔZ11 [µm wf] | `m1m3_r2_coeff_c` | −0.3872 | 0.02315 | 0.01570 | 32.2 % |
| ΔZ14 [µm wf] | `cam_body_minus_m1m3_air` | +0.5434 | 0.07557 | 0.05525 | 26.9 % |
| ΔZ26 [µm wf] | — | — | 0.02253 | 0.01884 | 16.4 % |
| ΔZ6 [µm wf] | — | — | 0.22055 | 0.18075 | 18.0 % |

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
| Δblur | −0.5771 ± 0.0341 | −0.2058 ± 0.0179 | arcsec per arcsec |
| ΔZ11 | −0.1115 ± 0.0112 | −0.0562 ± 0.0074 | µm wf per µm wf |
| ΔZ14 | −0.0441 ± 0.0334 | −0.2431 ± 0.0246 | µm wf per µm wf |
| ΔZ26 | −0.0599 ± 0.0266 | −0.2215 ± 0.0219 | µm wf per µm wf |

Δblur and ΔZ11 both flatten by about half. Against the **two-side mean** the flattening is
cleaner still for Z14 and Z26 (+0.3559 → +0.0388 and +0.2243 → −0.0010 µm wf per µm wf),
so for those two the thermal model removes essentially all of the dependence on the
wavefront level. ΔZ14 and ΔZ26 *gain* slope against the extra-focal value, which the
figures should be read against before drawing any conclusion.

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
- `thermal_selection.py` holds the greedy forward selection described below. Developed
  here; intended for `rubinwork` once used on a second study.
- `notebooks/unpaired_intra_vs_extra.ipynb` produces every figure into the single PDF.
  Assembled by `notebooks/_build_nb.py` so it can be regenerated rather than hand-edited
  as JSON.

Reproduction checks, both exact: median(intra − extra) blur **+0.3440 arcsec** on
`danish_1_3_test` (+0.3445 on v1000), intra blurrier in **0.9954** (dimensionless) of 873.
The donut-median route agrees with the `visits.parquet` columns to four decimals, and the
paired wavefront equals the two-side mean to 1.8e-15 µm of wavefront.

Results as in **Current answer** above.

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
2. Read the ΔZ14 / ΔZ26 "after" panels against the extra-focal value — their slopes grow
   under the correction while flattening against the two-side mean. Decide which
   reference is the right one before quoting either.
3. Field dependence, not started: per-detector maps of each difference in OCS and in CCS
   would separate a telescope-fixed from a camera-fixed pattern. The rotator spans enough
   angle over 15 nights.
4. Pair level, if wanted later: the within-visit slope of ΔZ11 on Δblur bounds the Danish
   fit degeneracy, since within one visit the wavefront is common and only the blur varies
   donut to donut.
5. Move `select_predictors` into `rubinwork` once a second study uses it.

## Files

| file | role |
|---|---|
| `extract_visit_medians.py` | the one pass over the 13 GB donut table; writes per-visit per-side medians |
| `thermal_selection.py` | greedy forward selection on incremental gain, Huber throughout |
| `notebooks/unpaired_intra_vs_extra.ipynb` | every figure and table, into one PDF |
| `notebooks/_build_nb.py` | assembles the notebook from cell sources |
| `output@/20261015_visit_medians/unpaired_intra_vs_extra.pdf` | 21 pages, all figures |
| `output@/20261015_visit_medians/*.parquet` | the two per-visit median tables and four summaries |
