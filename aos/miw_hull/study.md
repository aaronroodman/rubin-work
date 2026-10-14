---
study: miw_hull
topic: aos
status: active
question: Out to what field radius is the MIW trustworthy, and does the continued outer annulus carry signal or artifact?
answer: (open) Analysis 1 done — the coverage cliff is ONE ring wide, at r = 1.7500 deg of field angle, where union azimuthal coverage falls from 0.9333 to 0.1778 (9-bin) or 0.1167 (5-bin). Everything inside r <= 1.7286 deg is data-constrained. Whether that one ring carries signal is analysis 3.
uses:
  fam_tables: danish_1_2@20261014
  miw: d12-A_50_34_i@20261014, d12-A_50_34_i_5rot@20261014
started: 2026-10-14
---

# `miw_hull`: how far out is the measured intrinsic wavefront constrained?

## Question

The measured intrinsic wavefront (MIW) is decomposed on polar rings. Each ring's
telescope/camera (OCS/CCS) split is a least-squares fit over the azimuth nodes with real
donut support, so a thinly covered ring is poorly constrained, and one below
`min_ring_coverage` is not fit at all — since `ts_intrinsic_wavefront` commit `3f66904`
(2026-10-05) it is **continued radially** from the nearest fitted ring rather than nulled.

**Out to what field radius is the MIW trustworthy, and does the continued outer annulus
carry signal or artifact?** Concretely: can the 1.70–1.75 deg annulus stop being excluded?

The exclusion is currently a hand-applied convention. `aos/docs/studies/miw.md` calls it
the "convex-hull edge defect" and drops the annulus from every number it reports — across
the 1.70 to 1.75 deg step the Z5 OCS difference root-mean-square rises from **0.0499 to
0.1109 µm of wavefront** and reaches **0.6441 µm** over "240 of 3985 field points"; the
`danish_1_3` pupil comparison reports **0.0204 µm of wavefront** in that ring, likewise
excluded. No one had measured the coverage that justifies it.

Scope (Aaron, 2026-10-14): robust fits (Huber RLM, `nmad`), `danish_1_2` variants only.
Full scope in `aos/docs/status/miw_hull_scope.md`.

## Current answer

**Analysis 1 (coverage geometry) is done. The cliff is exactly one ring wide.**

| ring radius (deg of field angle) | union coverage, 9-bin | union coverage, 5-bin | per-bin floor | bins sampling |
|---|---|---|---|---|
| 1.6644 | 1.0000 | 1.0000 | 0.9111 | 9 / 5 |
| 1.6858 | 1.0000 | 1.0000 | 0.8889 | 9 / 5 |
| 1.7072 | 1.0000 | 1.0000 | 0.8500 | 9 / 5 |
| 1.7286 | 0.9333 | 0.9333 | 0.7111 | 9 / 5 |
| **1.7500** | **0.1778** | **0.1167** | **0.0056** | 9 / 5 |

Coverage is the fraction of the 180 azimuth nodes with real donut support
(dimensionless). The grid is the split's own: `r_min` 0.06, `r_max` 1.75 deg, `n_r` 80,
`n_az` 180, `hole_dist` 0.06 deg.

Three findings:

1. **Only the single outermost ring is under-covered.** 1 of 80 rings falls below
   `min_ring_coverage = 0.5`, at `r = 1.7500 deg`. The next ring in, `r = 1.7286 deg`,
   holds union coverage 0.9333 and a per-bin floor of 0.7111 — well constrained. So "the
   1.70–1.75 deg annulus" is too generous a description of the problem: the defect is one
   ring at the very rim, and the last fully covered ring (union >= 0.999) is
   `r = 1.7072 deg`.
2. **The coverage collapse is abrupt, not gradual** — union coverage 0.9333 → 0.1778
   across one 0.0214 deg radial step, a factor of 5.3. The per-bin floor falls to
   **0.0056**, a single azimuth node of 180. This is a hard edge of the donut
   distribution, not a taper.
3. **Both builds fail at the same ring, with different severity.** The 9-bin build pools
   more data (0.1778 union) than the 5-bin (0.1167), but neither reaches the 0.5
   threshold, and the per-bin floor is identical at 0.0056. More rotator bins do not
   rescue this ring — which is the useful negative result, and it sets up analysis 3.

**Independent check that this measures the right quantity.** The computed union coverage
at the outermost ring equals the builder's own `min_ring_coverage_seen` exactly — 0.1778
on the 9-bin build and 0.1167 on the 5-bin — and the builder reports 1 continued ring per
Zernike in both, matching the 1-of-80 count here. The script calls the package's
`make_polar_grid`, `sample_maps_polar` and `ring_coverage` rather than reimplementing the
geometry, so this agreement is a real cross-check of the pipeline, not a tautology.

**Still open:** whether the continued ring's values carry signal. Coverage says the ring
is not constrained by its own data; it does not say the radial continuation from
`r = 1.7286 deg` is wrong. Analysis 3 settles that by exploiting the two builds'
different coverage.

## Log

### 2026-10-14 — analysis 1: ring coverage geometry

`code/ring_coverage.py`, outputs `output/ring_coverage.parquet` and
`output/ring_coverage.pdf` (4 pages). Commit: see below.

Method: reuse the split's own polar grid and validity rule via
`intrinsic_split.make_polar_grid` / `sample_maps_polar` / `ring_coverage`, so the
coverage fraction is the quantity `min_ring_coverage` is compared against. Coverage is
computed from donut support, taking `z4_optical_OCS` as the finite-cell indicator, so it
is Zernike-independent. Two measures per ring: `coverage_union` pooled over rotator bins
(what the pooled design sees) and the per-bin fraction (what limits the O/C separation,
which needs several distinct rotator angles).

Results as in Current answer. Robust outward trend by Huber RLM: the fit converges to a
near-perfect weighted fit over the 80 rings because coverage is 1.0 across almost the
whole range, so the slope is not the informative number here — the one-ring step is. The
`ConvergenceWarning` on that fit is expected and harmless for that reason.

Per-bin totals (cells and donuts, both dimensionless counts) for the 9-bin build:

| rotator bin | cells | donuts |
|---|---|---|
| `rot_-65_-55` | 3846 | 102,175 |
| `rot_-50_-40` | 3820 | 67,596 |
| `rot_-35_-25` | 3826 | 68,077 |
| `rot_-20_-10` | 3856 | 131,361 |
| `rot_-3_3` | 3831 | 104,194 |
| `rot_10_20` | 3861 | 101,848 |
| `rot_25_35` | 3818 | 67,252 |
| `rot_40_50` | 3780 | 44,507 |
| `rot_55_65` | 3843 | 99,696 |

The 5-bin build reads `rot_-65_-55`, `rot_-20_-10`, `rot_-3_3`, `rot_10_20`,
`rot_55_65` — the five best-populated bins, 539,274 donuts of the 9-bin build's 786,706.

Note on the premise, carried from the phase 3c validation
(`$RUBINWORK_DATA/studies/phase3c_validation/miw/`): the map grid's **field of view did
not widen** between the old and new builds. Old-only cells are 0, the half-width went
1.7260274 → 1.7260000 deg of field angle (slightly *smaller*), and the 154 extra cells
are cells the old grid also had but left NaN or dropped. The hull that matters is the
donut-coverage hull measured here, not the map extent.

**Next:** analysis 2 (continued versus fitted rings, per Zernike, from the four
`intrinsic_split_rms` columns), then analysis 3 (the decisive signal-versus-artifact
test).
