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

**Analysis 1b (azimuthal probe at r = 1.725 deg) adds: the delivered maps are sound
there.** Walking azimuth 0 to 360 deg in 1 deg steps returns **no NaN and no defective
zeros** on any of the 21 OCS Zernikes or the Z4 CCS map, through either consumer route
(the written-maps interpolator and the rotator-dependent `reconstruct_at`). It also found
a consumer trap: on the **outermost** ring the polar grid's azimuthal node spacing is
**2.0 deg**, so a finer probe step manufactures NaN from complete data (180 of 360 at
r = 1.7500 deg) because there is no triangle beyond the last ring. See the log.

**Still open:** whether the continued ring's values carry signal. Coverage says the ring
is not constrained by its own data; it does not say the radial continuation from
`r = 1.7286 deg` is wrong. Analysis 3 settles that by exploiting the two builds'
different coverage.

## Log

### 2026-10-14 — analysis 1b: azimuthal probe at r = 1.725 deg (Aaron's request)

`code/azimuth_probe.py`, outputs `output/azimuth_probe.{parquet,pdf}`,
`output/azimuth_probe_summary.csv`, `output/azimuth_probe_radius_scan.csv`.

Aaron asked for: at radius exactly **1.725 deg of field angle**, step azimuth 0 to 360 deg
in 1 deg steps, check the returned value of the OCS map for every Zernike and the CCS map
for Z4, plot against azimuth, and flag any NaN or identically-zero return — exercising
**`ts_intrinsic_wavefront`'s own interpolator and setup**, not an ad-hoc method.

Both consumer routes are probed, each through its own package function:

| route | function | what it represents |
|---|---|---|
| `maps` | `intrinsic_split.polar_field_to_points` on the decomposition's own `X`/`Y` grid, `nearest_within` = the radial step | the call that **wrote** `intrinsic_split_maps.parquet`, rim fill included |
| `decomp` | `intrinsic_split.reconstruct_at` via `rubinwork.miw_corner.load_decomposition` | `value_added`'s `--intrinsic miw` path, at one camera rotator angle |

**Result: no NaN and no defective zeros anywhere.** 360 azimuth samples x 21 Zernikes x 2
components x 2 routes x 2 variants.

| variant | NaN returns | identically zero, by design | identically zero, defect | partially zero |
|---|---|---|---|---|
| `d12-A_50_34_i` | none | 0 | none | none |
| `d12-A_50_34_i_5rot` | none | 20 (CCS, `ocs_only`) | none | none |

The 20 zero CCS series in `_5rot` are **correct**: `intrinsic_split_rms.ocs_only` is true
for Z5-Z26 there, so the camera term is forced to 0 and **Z4 is the only meaningful CCS
map** — which is exactly why Aaron's request named Z4 only. `d12-A_50_34_i` splits all 21,
and none of its CCS columns is identically zero.

OCS values at this radius, `maps` route, µm of wavefront:

| variant | peak-to-peak over azimuth, min across Zernikes | max across Zernikes |
|---|---|---|
| `d12-A_50_34_i` | 1.4211e-02 | 4.6136e-01 |
| `d12-A_50_34_i_5rot` | 3.7804e-02 | 4.5382e-01 |

Every series varies smoothly. The spin-doublet pairs (5/6, 7/8, 9/10, 12/13, 14/15, 16/17,
18/19, 23/24, 25/26) return **identical** min, max and RMS, the correct signature of a
cos/sin doublet sampled around a full circle — an internal consistency check that came for
free.

**A trap worth knowing, found by the negative control.** A probe that reports "no NaN" is
only meaningful if it can produce one, so `radius_scan` sweeps the probe radius. At
**r = 1.7500 deg exactly, 180 of 360 azimuths return NaN** — and the rim fill is not the
cause (identical count with it off). The reason is **probe geometry, not missing data**:
the polar grid has `n_az = 180` nodes, i.e. **2.0 deg** azimuthal spacing, and on the
*outermost* ring there is no triangle farther out to interpolate within, so every azimuth
falling between two nodes is outside the convex hull. The stored polar field has **0 NaN**
on that ring. Even azimuths (on-node) return fine.

| r (deg of field angle) | NaN with rim fill | NaN without | polar field's own NaN on that ring |
|---|---|---|---|
| 1.7000 | 0 / 360 | 0 / 360 | 0 |
| 1.7250 | 0 / 360 | 0 / 360 | 0 |
| 1.7286 | 0 / 360 | 0 / 360 | 0 |
| 1.7500 | 180 / 360 | 180 / 360 | 0 |
| 1.7512 | 180 / 360 | 360 / 360 | 0 |
| 1.7600 | 180 / 360 | 360 / 360 | 0 |

Two things follow. **Any consumer sampling the MIW near the rim must not step finer than
2.0 deg in azimuth on the outermost ring**, or it will manufacture NaN from complete data.
And the rim fill is doing its job: at r = 1.7512 and 1.760 deg it recovers the 180 on-node
azimuths that would otherwise all be NaN.

Aaron's chosen r = 1.725 deg is inside the last interpolable band and returns 360/360
finite on every series — a well-chosen radius for this test.

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
