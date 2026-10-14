# `miw_hull` study scope

> **Status:** scope for Aaron's review, not yet a converted study · **Last updated:**
> 2026-10-14 · **Kind:** working state (scope)

Scope for `miw_hull`, item 2 of the `aos` todo list and the **first phase 5 study
conversion** (Aaron, 2026-10-14). The triage row
(`notes/status/phase5_study_triage.md`) reads `parked` / "diagnosed: hull about 1.75 deg
vs 1.725 deg cut" / "waiting on Guillem's branch". **That branch has landed** —
`ts_intrinsic_wavefront` commit `3f66904`, 2026-10-05 — so the blocker is gone and the
status becomes `convert`.

Agreed with Aaron: **robust fits** (Huber RLM, `nmad` for scatter) and the **`danish_1_2`
variants only**.

**The study does not recommend an `r_max`** (Aaron, 2026-10-14). The donut selection for
wavefront has a settled field-radius limit of **1.725 deg of field angle**, and that limit
is not this study's to move. The question becomes: **are MIW values at r <= 1.725 deg
affected by the radial continuation of under-covered rim rings**, either through ring
interpolation (the polar route) or through the 71x71 map grid whose cells reach
**1.751 deg of field angle** (the Cartesian route)? Whether the rim ring itself carries
signal stays a secondary question. Analysis 5 is a summary table, not a recommendation.

## Question

The MIW is fit on polar rings whose azimuthal coverage thins toward the field edge.
Beyond the coverage hull the ring fit is **extrapolation**. Commit `3f66904` replaced
"null the under-determined rings" with "continue them radially from the nearest fitted
ring". So:

**Out to what field radius is the MIW trustworthy, and does the continued outer annulus
carry signal or artifact?** Concretely: can the 1.70-1.75 deg annulus stop being
excluded?

## Why this is worth doing, in one place

`aos/docs/studies/miw.md` already treats the outer annulus as a known defect and excludes
it from every number it reports. It calls it the **"convex-hull edge defect"**: across the
1.70 to 1.75 deg step the Z5 OCS difference root-mean-square rises from **0.0499 to
0.1109 µm of wavefront** and reaches **0.6441 µm**, over "240 of 3985 field points". The
`danish_1_3` pupil-model comparison likewise reports **0.0204 µm of wavefront** in the
1.70-1.75 deg ring as "the convex-hull edge defect present in both builds, reported as its
own annulus and excluded from every number above".

So the exclusion is a standing convention applied by hand across several studies, on a
quantity nobody has measured directly. This study measures it.

## One correction to the premise, carried from the 3c validation

The 3c validation (`$RUBINWORK_DATA/studies/phase3c_validation/miw/`) establishes that
**the map grid's field of view did not change** between the old and new builds:

| quantity | old 9-bin | new |
|---|---|---|
| half-width (deg of field angle) | 1.7260274 (`126/73`) | 1.7260000 |
| cell step (deg of field angle) | 0.04931507 | 0.04931429 |
| max cell radius (deg of field angle) | 1.7511779 | 1.7511780 |
| unique x values (dimensionless) | 71 | 71 |

The new lattice is very slightly **smaller**, by 2.7e-05 deg of field angle, and
old-only cells are **0**. The 154 extra cells are cells the old grid also had but left
NaN or dropped: old `_5rot` holds **124 NaN cells in `Z4_OCS`** (the "124 points" annulus
at `r >= 1.729 deg` that `3f66904` names) and the new build holds **0**.

So there is **no wider map hull** to study. The real hull is the **donut-coverage** hull,
and that is what this study is about. The 1.725 deg figure in the triage row is the grid
cut; the ~1.75 deg figure is where coverage actually dies.

## The coverage cliff, already visible

From `intrinsic_grid`'s `n_donuts` per cell, build `d12-A_50_34_i@20261014`:

| r bin (deg of field angle) | `rot_-3_3` cells | `rot_-3_3` median donuts/cell | `rot_40_50` cells | `rot_40_50` median donuts/cell |
|---|---|---|---|---|
| 0.000-1.000 | 1305 | 26.0 | 1293 | 11.0 |
| 1.000-1.500 | 1596 | 25.0 | 1576 | 11.0 |
| 1.500-1.650 | 560 | 27.0 | 564 | 11.0 |
| 1.650-1.700 | 187 | 39.0 | 179 | 17.0 |
| 1.700-1.729 | 116 | 18.5 | 117 | 9.0 |
| **1.729-1.800** | **67** | **6.0** | **51** | **2.0** |

Counts are donuts per grid cell (dimensionless). The drop at r = 1.729 deg is the cliff,
and it is **exactly where the old NaN annulus began**. The two bins also hold different
cell counts (3,831 vs 3,780) and different total donuts (104,194 vs 44,507), so the
rotator bins have genuinely different coverage — which analysis 3 turns into a test.

## Inputs, all already on disk

No new build is needed.

| input | what it gives |
|---|---|
| `miw/d12-A_50_34_i@20261014/build/rot_*/intrinsic_grid.parquet` | `n_donuts`, `zk_rms`, `zk`, `z4_optical_OCS` per cell, per rotator bin (9 bins) |
| `miw/d12-A_50_34_i@20261014/build/rot_*/intrinsic_cov_edge.parquet` | the 21x21 Zernike-Zernike covariance and correlation at the edge, per bin |
| `miw/d12-A_50_34_i_5rot@20261014/intrinsic_split_rms.parquet` | `n_continued_rings`, `min_ring_coverage_seen`, `n_null_rings`, `max_null_dim` per Zernike — the builder's own account |
| both variants' `intrinsic_split_maps.parquet` | the final OCS/CCS maps to judge |
| `fam_tables/danish_1_2@20261014/donuts.parquet` | donut field positions for the true coverage hull |
| `studies/phase3c_validation/miw/` | old-vs-new at the rim, already measured |

For `_5rot`: **1 continued ring per Zernike**, at `min_ring_coverage_seen = 0.1167`
(dimensionless azimuthal fraction) against the `min_ring_coverage = 0.5` threshold — so
the continued ring saw 11.67% azimuthal coverage.

## Five analyses

1. **The real coverage hull, per rotator bin.** Donut `(thx_OCS, thy_OCS)` convex hull and
   per-ring azimuthal coverage fraction against field radius, per bin, against the
   `min_ring_coverage = 0.5` threshold. Establishes where extrapolation begins **per bin**
   instead of assuming one radius. Robust trend by Huber RLM; scatter by `nmad`.
2. **Continued versus fitted rings.** Which radii were continued, at what coverage, per
   Zernike and per bin, from the four new `intrinsic_split_rms` columns. Pairs the
   builder's own bookkeeping against the geometry from analysis 1.
3. **Is the continued annulus signal? (the decisive test.)** The 9-bin and `_5rot` builds
   have **different** coverage, so they continue **different** rings. Where both ring-fit,
   the maps must agree; where one continues and the other fits, they need not. Agreement
   in the continued annulus argues signal; disagreement argues artifact. The 3c validation
   already shows max abs **7.601e-01 µm of wavefront** (9-bin) and **8.163e-01 µm**
   (`_5rot`) at `r >= 1.729 deg` against **7.179e-04** and **1.292e-03 µm** in the
   interior — suggestive, but not yet decomposed ring by ring, which is what settles it.
4. **Does it matter downstream?** Propagate with and without the annulus: wavefront RMS
   inside the hull versus including the rim, and the inferred FWHM difference in arcsec.
   Deliberately the same two quantities `aos/docs/studies/miw.md` reports (its legacy-vs-v1000
   table gives 0.0518 / 0.0528 µm of wavefront and 0.1645 / 0.1678 arcsec), so the numbers
   are directly comparable to existing work.
5. **The summary table.** Not a recommendation (Aaron, 2026-10-14: the 1.725 deg
   selection limit is settled and the study does not move it). One table other studies can
   read: per ring over the outer field, the coverage, the fitted/continued status, the
   measured old-versus-new difference in µm of wavefront, and whether anything at
   r <= 1.725 deg depends on it — plus a line on whether `miw.md`'s standing
   "exclude the 1.70-1.75 deg annulus" convention is wider than the measurement
   warrants.

## What this does NOT cover

- **Pupil-model dependence.** `danish_1_3_v1000` would test whether the edge behaviour is
  pupil-model specific. Out of scope per Aaron (2026-10-14); `danish_1_2` only.
- **The CWFS corners.** They sit near 1.19 deg of field angle, well inside every hull
  considered here, so nothing in this study bears on them. Aaron's planned TARTS-vs-Danish
  CWFS comparison is a separate study reading the `cwfs_tables` donut tables.
- **Changing the product.** If the answer is "move `r_max`", that is a `miw` `variants.yaml`
  decision for Aaron, not a change this study makes.

## Deliverable

`aos/miw_hull/`, the phase 5 `convert` layout from section 5 of the plan:

```
aos/miw_hull/
    study.md        front matter, Question, Current answer, Log
    code/           the analysis scripts
    figures/        a few committed PNGs cited by the log
    output@         gitignored symlink to studies/aos/miw_hull
```

Front matter:

```yaml
---
study: miw_hull
topic: aos
status: active
question: Out to what field radius is the MIW trustworthy, and does the continued outer annulus carry signal or artifact?
answer: (open)
uses:
  fam_tables: danish_1_2@20261014
  miw: d12-A_50_34_i@20261014, d12-A_50_34_i_5rot@20261014
started: 2026-10-14
---
```

Triage change: `miw_hull` **`parked` -> `convert`**, reason "Guillem's branch landed as
`3f66904`".

## Open scope questions for Aaron

1. **ANSWERED 2026-10-14: just the measurement.** The study does not recommend an `r_max`.
   The donut selection for wavefront has a settled limit of **1.725 deg of field angle**,
   so analysis 5 is a summary table (rewritten above) and the study's question is whether
   the continuation of the rim rings reaches inside that limit. **Measured in analysis 2:
   it does not** — the last fitted ring is `r = 1.7286 deg`, outside the limit, and 0 of
   the 3,833 map cells at r <= 1.725 deg lie outside it.
2. **The `_5rot` MIW is the calibration source.** `3f66904`'s own note calls
   `pathA_50_34_i_5rot` "the v2 calib source", and `aos/calibration/` carries a staged MIW.
   If the answer changes `r_max`, the staged calibration is downstream of that. Worth
   knowing before the recommendation lands, not something this study changes.
