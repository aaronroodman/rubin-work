---
study: miw_hull
topic: aos
status: answered
question: Are MIW values inside the settled 1.725 deg of field angle donut-selection limit affected by the radial continuation of under-covered rim rings, and does the continued rim ring itself carry signal or artifact?
answer: No MIW value inside the 1.725 deg of field angle selection limit depends on a continued ring — the builder continues exactly ONE ring, at r = 1.7500 deg, the last fitted ring is r = 1.7286 deg, and 0 of the 3,833 map cells at r <= 1.725 deg lie outside it by either consumer route. Whether that one rim ring carries signal or artifact was NOT pursued: no consumer inside 1.725 deg depends on it, and both builds continue the same ring so analysis 3's two-build contrast is empty.
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

**Closed answered, 2026-10-14 (Aaron).** Analyses 1, 1b and 2 answer the question;
**analyses 3, 4 and 5 were not run** — see "Not pursued" below.

**The consumer trap, carried up from analysis 1b because it is the one thing a consumer
can get wrong.** On the **outermost** ring the polar grid's azimuthal node spacing is
**2.0 deg**, so a probe finer than that **manufactures NaN from complete data** — 180 of
360 azimuth samples at `r = 1.7500 deg` return NaN at a 1 deg step, because there is no
interpolation triangle beyond the last ring. The data are not missing. Probe the rim at
the grid's own 2.0 deg spacing, or accept NaN that means "between nodes", not "no data".

**Nothing inside the 1.725 deg selection limit depends on a continued ring** (analysis 2).
The builder continues exactly **one** ring, at `r = 1.7500 deg of field angle`, identically
for all 21 Zernikes in both builds. The last **fitted** ring is `r = 1.7286 deg`, already
outside the limit, so neither consumer route reaches the continuation: the polar route
interpolates r = 1.725 deg between the fitted rings 1.7072 and 1.7286 deg, and **0 of the
3,833 map cells at r <= 1.725 deg** lie outside the last fitted ring (the 152 cells that do
are all outside the limit). The scope decision (Aaron, 2026-10-14) is that the study does
not move the 1.725 deg limit, so **analyses 3 and 4 reduce to the rim only** — they no
longer have work inside the selection limit.

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

## Not pursued: is the rim ring signal or artifact?

**Analyses 3, 4 and 5 were not run** (Aaron, 2026-10-14). Coverage says the one continued
ring at `r = 1.7500 deg` is not constrained by its own data; it does not say the radial
continuation from `r = 1.7286 deg` is wrong. That remains unmeasured, deliberately, for
two reasons:

1. **No consumer depends on it.** Every MIW consumer works inside the settled 1.725 deg
   of field angle selection limit, and analysis 2 showed 0 of the 3,833 map cells at
   `r <= 1.725 deg` lie outside the last fitted ring. So the answer would change no
   number anyone reads.
2. **The decisive test has no data.** Analysis 3 was scoped on the 9-bin and `_5rot`
   builds continuing **different** rings. They do not — both fit the same 79 rings and
   continue the same one (analysis 2, finding B4), so the two-build contrast is **empty**.
   Settling the rim would need a different test (a held-out azimuth refit, or a
   pupil-model contrast against `danish_1_3_v1000`, which this study's scope excludes).

Analysis 4 (downstream propagation with and without the rim annulus) and analysis 5 (the
per-ring summary table) both reduce to the rim once analysis 2 ruled the interior clear,
so they inherit the same reason.

## Log

### 2026-10-14 — closed, status answered

Aaron closed the study on the strength of analyses 1, 1b and 2. The question as asked —
does the rim continuation reach inside the 1.725 deg of field angle selection limit — is
answered **no**, quantitatively: one continued ring at `r = 1.7500 deg`, last fitted ring
`r = 1.7286 deg`, **0 of 3,833 map cells** at `r <= 1.725 deg` outside it by either
consumer route.

**Analyses 3, 4 and 5 not run**, reason recorded in "Not pursued" above: no consumer
inside 1.725 deg depends on the rim ring, and analysis 3's two-build contrast is empty
because both builds continue the same ring.

**Downstream change:** `aos/docs/studies/miw.md` carried a standing convention to exclude
the 1.70-1.75 deg annulus as a "convex-hull edge defect". That convention is **wider than
the measurement warrants** — the defect is one ring at `r = 1.7500 deg`, not a 0.05 deg
annulus, and inside 1.725 deg the maps are fitted from data. Replaced there with a pointer
to this study.

### 2026-10-14 — analysis 2: continued versus fitted rings

`code/continued_rings.py`, outputs `output/continued_rings.parquet`,
`output/continued_rings_zernike.csv`, `output/continued_rings_summary.txt`,
`figures/continued_rings.png`. It imports analysis 1's `ring_table`, `builder_flags` and
`bin_dirs` rather than recomputing the geometry, so both analyses measure the same
coverage.

**Scope change recorded this day (Aaron):** the study does **not** recommend an `r_max`.
The donut selection for wavefront has a settled limit of **1.725 deg of field angle**. The
question is now whether the continuation of the rim rings reaches inside that limit.

**How the builder decides, read from the code and not assumed.** In
`ts_intrinsic_wavefront` (`3f66904` is an ancestor of the installed `a6394e7`),
`intrinsic_split.decompose_spin_lsq` computes the coverage **once per decomposition** at
line 453, `coverage = ring_coverage(valid)`; `ring_coverage` (line 331) is
`np.any(v, axis=0).mean(axis=1)` (line 338), and `np.any` over axis 0 pools the rotator
bins. The test is line 469, `if coverage[k] < min_coverage: continue`, and the skipped
rings with data are continued from the nearest fitted ring at lines 500-508
(`ring_filled[k] = True`). So the decision is on **union** coverage, not per bin, and it is
one decision for the whole decomposition — identical for every Zernike. Per-bin coverage
still limits the **rank** of a ring that is fit, which is why B2 reports it, but it does
not enter the fit/continue decision.

**B1 — per Zernike.** Uniform across all 21 Zernikes (Z4-Z19, Z22-Z26) in both builds, with
**no Zernike off the 1-continued-ring-at-r=1.7500-deg pattern**:

| build | rotator bins | n_continued_rings | continued ring radius (deg of field angle) | n_null_rings | min_ring_coverage_seen (fraction, dimensionless) | max_null_dim |
|---|---|---|---|---|---|---|
| `d12-A_50_34_i` | 9 | 1 | 1.7500 | 0 | 0.1778 | 0 |
| `d12-A_50_34_i_5rot` | 5 | 1 | 1.7500 | 0 | 0.1167 | 0 |

Uniformity is expected rather than surprising, given the decision rule above: coverage
comes from donut support, which is Zernike-independent. That the builder's `max_null_dim`
is **0** on every Zernike is the stronger statement — every ring it *did* fit was full
rank, so the rim is the only place the fit gave up.

**B2 — per rotator bin, outer 5 rings** (azimuthal coverage, fraction of the 180 nodes,
dimensionless; 9-bin build):

| r (deg of field angle) | union | per-bin min | per-bin median | bins | builder status |
|---|---|---|---|---|---|
| 1.6644 | 1.0000 | 0.9111 | 0.9222 | 9 | fitted |
| 1.6858 | 1.0000 | 0.8889 | 0.9111 | 9 | fitted |
| 1.7072 | 1.0000 | 0.8500 | 0.8889 | 9 | fitted |
| 1.7286 | 0.9333 | 0.7111 | 0.7778 | 9 | fitted |
| **1.7500** | **0.1778** | **0.0056** | **0.0278** | 9 | **continued** |

The 5-bin build is the same four fitted rings and the same continued ring, at union 0.1167
and an identical per-bin floor of 0.0056. Per-bin coverage at the rim is 0.0056 to 0.1000
across the nine bins — no single bin comes near 0.5, so **no choice of bin pooling rescues
this ring**. Full per-bin columns are in `continued_rings.parquet`.

**B3 — the cross-check passes on both builds.** Continued rings above the 0.5 threshold:
**0**. Rings below the threshold yet fitted: **0**. The builder's own reported counts equal
the geometry's (continued 1 vs 1, null 0 vs 0). So the coverage quantity analysis 1
measures is the quantity the builder decides on, and the two agree exactly.

**B4 — the ring set analysis 3 needs is EMPTY.** Both builds fit the same 79 rings and
continue the same one, at `r = 1.7500 deg`. **No ring is fitted in one build and continued
in the other.** The 9-bin build pools more data (union 0.1778 against 0.1167) but not
enough to cross 0.5, so the extra rotator bins change the coverage without changing any
ring's status. Analysis 3 as scoped — "where one continues and the other fits, compare the
maps" — therefore **has no ring to work on** and needs a different test for the rim.

**B5 — relative to the 1.725 deg selection limit.** The limit falls between the fitted
rings `r = 1.7072` and `r = 1.7286 deg`, and the single continued ring sits at
`r = 1.7500 deg`, outside it. Neither consumer route reaches the continuation:

| route | geometry checked | result |
|---|---|---|
| polar (ring interpolation) | the rings bracketing r = 1.725 deg | both **fitted** — does not draw on a continued ring |
| Cartesian (71x71 map grid) | map cells at r <= 1.725 deg lying outside the last fitted ring r = 1.7286 deg | **0 of 3,833** cells |

The 152 cells at r > 1.725 deg (of 3,985, the map reaching r = 1.7512 deg) are the only
ones that can draw on the continuation, and 108 of them sit strictly between the last
fitted ring and the continued ring.

**In one line: analyses 3 and 4 have no work left inside the 1.725 deg selection limit, and
reduce to the rim only.** Nothing a wavefront consumer reads at r <= 1.725 deg depends on
the radial continuation, by either route. No fit or scatter is quoted in this analysis, so
no robust estimator was needed; the one Huber RLM in the study is analysis 1's coverage
trend.

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

**Next:** analysis 2 — done the same day, see the entry above, which also records that its
two-build contrast left analysis 3 without a ring to work on.
