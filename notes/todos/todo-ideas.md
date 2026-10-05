# Rubin AOS — TODO / Ideas

> **Status:** current · **Last updated:** 2026-10-04 · **Kind:** working state (queue)

Open ideas and todo items for future Claude Code sessions to work on in
`~/notebooks/rubin-work` (mostly `aos/`). New and in-process items only — items that are
mostly or fully delivered move to [completed-todos.md](completed-todos.md) with their
original scope intact.

Every item follows [todo-style.md](todo-style.md): headline, status, the idea in one
paragraph, `**Goals:**`, then everything else collapsed. Edit the style file, and the items
follow.

## Table of contents

| # | item | status |
| --- | --- | --- |
| [1](#1-compare-different-wavefront-retrieval-methods-using-visits-from-several-nights) | Compare different wavefront retrieval methods using visits from several nights | not started, waiting for CWFS processing |
| [2](#2-extend-the-miw-grid-so-the-interpolation-hull-covers-the-full-field-of-view) | Extend the MIW grid so the interpolation hull covers the full field of view | diagnosed, Guillem has the fix |
| [3](#3-confluence-page-documenting-the-aos-production-runs-in-repomain) | Confluence page documenting the AOS production runs in `/repo/main` | probe done, T614 set added, page not written |
| [4](#4-pupil-measure-the-donut-pupil-geometry-data-against-model) | `pupil` — measure the donut pupil geometry, data against model | not started |
| [5](#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint) | Rebuild the MIW on the v1000 pupil model, then under the RBR constraint | in process, first build done |
| [6](#6-guider-star-second-moments-image-and-centroid-motion-into-the-value-added-db) | Guider star second moments, image and centroid motion, into the value-added DB | not started |
| [7](#7-giant-donuts-pupil-models-spiders-and-the-intraextra-z11-split) | Giant donuts: pupil models, spiders, and the intra/extra Z11 split | not started, supersedes the `wfs/` giant-donut work |

Recently closed and moved out: the Danish 1.3 blitz Full Array Mode (FAM) processing, the
July bounce test and its note for Guillem, the `thermal_focus` study, the `visit_telemetry`
backfill to 20250415, the shared regularized-inversion solvers, the four-scheme bounce test,
the three-scheme science-visit optical state, and the `thermal_focus` report reorganization.
See [completed-todos.md](completed-todos.md).

---

## 1. Compare different wavefront retrieval methods using visits from several nights

**Status:** not started · **Blocked on:** waiting for CWFS processing

Compare wavefront retrieval methods using regular FBS visits from several typical nights. Methods will include the current default Danish 1.2 paired, Danish 1.2 unpaired, Danish 1.3 unpaired, Danish 1.3 unpaired with an updated pupil model, TARTS and possibly
AIdonut.  The day_obs being processed are 20260512, 20260513 and 20260713.

**Goals:** Determine the consistency or lack thereof between methods and validate the new methods. 

<details>
<summary>Collections, existing machinery, scope and open questions</summary>


### Known collections

One row per method. All paths are in `/repo/main`; `wep` is ts_wep, `dv` is donut_viz.
`img_type` states which exposures the collection reduced. **Not yet probed** on the three
target nights — the day_obs ranges below are what the collection holds overall, from the
item 3 inventory, not a count of science and acq visits on 20260512, 20260513 and 20260713.

| method | collection | wep / dv | img_type | day_obs held |
| --- | --- | --- | --- | --- |
| Danish 1.2 paired | `LSSTCam/runs/aos/cwfs/danish_1_2_0/wep_17_9_0/dv_4_8_1/bin_x2/paired/refitWcs` | 17.9.0 / 4.8.1 | science | 20260512–20260713, 3 nights |
| Danish 1.2 unpaired | `LSSTCam/runs/aos/cwfs/danish_1_2_0/wep_17_9_0/dv_4_8_1/bin_x2/unpaired/refitWcs` | 17.9.0 / 4.8.1 | science | 20260512–20260713, 3 nights |
| Danish 1.3 unpaired (FAM) | `u/jmeyers3/t614_fam_unpaired` | blitz-prototype-v2 | cwfs, BLOCK-T614 only | 20260315–20260619, 966 visits |
| Danish 1.3 unpaired (corner) | `u/jmeyers3/t614_corner_unpaired` | blitz-prototype-v2 | acq, BLOCK-T614 only | 20260315–20260619, 961 visits |
| Danish 1.3, updated pupil model | does not exist yet — needs a rerun | — | — | — |
| AIdonut (corner) | `LSSTCam/runs/aos/cwfs/danish_1_2_0/wep_17_9_1/dv_4_8_5/binned_x2/refitWcsStamps/aidonut/` | 17.9.1 / 4.8.5 | science | 20260521–20260713, 6 nights |
| AIdonut (triplet, binned) | `LSSTCam/runs/aos/fam_cwfs_triplet/danish_1_2_0/wep_17_9_1/dv_4_8_5/refitWcsStamps/aidonut/binned_v1/` | 17.9.1 / 4.8.5 | FAM triplet | 20260315–20260513, 20 nights |
| AIdonut (triplet, unbinned) | `LSSTCam/runs/aos/fam_cwfs_triplet/danish_1_2_0/wep_17_10_0/dv_4_9_0/refitWcsStamps/aidonut/unbinned_v1/` | 17.10.0 / 4.9.0 | FAM triplet | 20260315–20260513, 20 nights |
| TARTS | personal only, `u/peterma2/*` | — | — | — |

The two Danish 1.3 blitz collections are **BLOCK-T614 test-block visits, not regular FBS
science visits**: `danish_1_3_test` in `aos/param_sets.yaml` carries
`fam_programs: [T614]`, the collections are named `t614_*`, and the `fam_unpaired` side
reduced 966 FAM `cwfs` exposures while `corner_unpaired` reduced the in-focus `acq` member
of the same triplets. So Danish 1.3 as it exists today does not cover the science and acq
visits this item asks for, on any night. The two AIdonut triplet collections are likewise
FAM triplets rather than science visits, and carry only `aggregateZernikesRaw` rather than
the joined `aggregateAOSVisitTable*`.

There is **no production `danish_1_3`** collection — the production tree has only
`danish_1_0`, `danish_1_1_1`, `danish_1_2_0` and `danish_1_2_0_alpha0`. Danish 1.3 is read
through `aos/code/fam_processing/blitz_reader.py`. Item 4 has the full production-run
inventory.

### Existing machinery to build on

| piece | path |
| --- | --- |
| per-donut A/B comparison | [aos/code/processing_compare/compare_donuts.py](../../aos/code/processing_compare/compare_donuts.py) |
| FAM-processing comparison | [aos/code/processing_compare/compare_fam_processings.py](../../aos/code/processing_compare/compare_fam_processings.py) |
| the single-night precedent | [aos/notebooks/processing_compare/aos_danish_tarts_compare_20260713.ipynb](../../aos/notebooks/processing_compare/aos_danish_tarts_compare_20260713.ipynb) |
| its output | `aos/output/danish_tarts_compare_20260713/` (13 PDFs) |
| notebook front-end | [aos/notebooks/processing_compare/study_compare_donuts.ipynb](../../aos/notebooks/processing_compare/study_compare_donuts.ipynb) |
| study doc | [aos/docs/studies/processing_compare.md](../../aos/docs/studies/processing_compare.md) |

`compare_donuts.py` matches donuts **positionally per CCD** — intra-focal centroids
within `tol_pix` on the same detector, via KDTree — and is written for exactly
**two** sides, A and B. It is numpy/scipy/pyarrow only, so it runs anywhere the parquets
exist, with no LSST stack needed — worth preserving in the generalization.

### Scope

- Probe the three nights for which methods actually have science and acq visits.
- Generalize the matcher from 2 sides to N methods.
- Add param sets for the methods that lack one, and a reader for the AIdonut
  `aggregateZernikesRaw` path.
- Rerun Danish 1.3 on science and acq visits, and again with the updated pupil model.
- Produce the comparison plots per night and pooled across the three nights.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. N-way join, or all pairwise?** One reference method with the others matched onto it
gives a single table but drops any donut missing from any method; all pairwise keeps every
donut but yields no common sample.

**A:** The main thing I would like is per cwfs averages per method shown vs. seq_num for each night, not donut to donut comparisons.  So then all methods which have data can be overlaid, and this is not an issue. 

**Q2. Does Danish 1.3 need a rerun on science and acq visits?** The blitz collections are
BLOCK-T614 test-block exposures, so Danish 1.3 cannot enter a science-visit comparison as
it stands. Either ask Josh for a rerun over the three nights' science and acq visits, run
the blitz ourselves, or drop Danish 1.3 from the first pass.

**A:** I will ask Josh to run these.

**Q3. How is partial coverage across methods handled?** Accept a ragged method-by-night
matrix, restrict the pooled comparison to the methods present on all three nights, or
report per night only.

**A:** See Q1

**Q4. What is the study named?** It is no longer `processing_compare`, which is a pairwise
regression utility.

**A:** Why not just use processing_compare, that sounds fine

**Q5. Is AIdonut in the first pass?** Its production collections split the three nights
across three different wep/dv and binning combinations, and the triplet ones need a
separate `aggregateZernikesRaw` reader.

**A:** Lets leave AIDonut for a second pass of this study, since yes, they will need their own reader

**Q6. How is the unpaired-versus-paired asymmetry matched?** Unpaired has one row per
single-position donut, paired one per pair, so "the same donut" is not a clean 1:1 concept
across those two.

**A:** see Q1

**Q7. Where do TARTS results come from?** There is no production collection, only
`u/peterma2/*`.

**A:** Peter is running that now

**Q8. Is version skew a confound to control or a result to report?** The methods being
compared do not share a ts_wep or donut_viz version.

**A:** The changes in ts_wep or donut_viz are small enough that those version numbers dont matter here

</details>

---

## 2. Extend the MIW grid so the interpolation hull covers the full field of view

**Status:** diagnosed, not fixed, Guillem has code to fix and he will send me his branch

Extend the grid of points used by the Measured Intrinsic Wavefront (MIW) in the external
`ts_intrinsic_wavefront` package, so that its convex hull reaches past the 1.725 deg field
radius cut that `ts_wep` applies. Donuts landing outside the hull currently receive NaN
intrinsic Zernikes. The fix lands in that external package, so it is a ticket and pull
request there rather than a `rubin-work` change.

**Goals:** Eliminate the NaN intrinsic Zernikes at the field edge, with enough margin that
a later increase in the field-radius cut does not reintroduce them.

<details>
<summary>The bug, prior diagnosis, scope and open questions</summary>


### The bug

The MIW interpolation hull is nominally 1.75 deg and `donut_viz`'s
`generateDonutFromRefitWcsTask.donutSelector.maxFieldDist` cut is 1.725 deg, so the two
nearly coincide and the failure bites only at the very edge. Runs that do not apply the
1.725 deg cut, such as `LSSTCam/runs/nightlyValidation/68`, push donuts out to about
1.84 deg field radius and see roughly 37% NaN (dimensionless, NaN rows over donut rows) at
the edge of the field of view.

### Prior work to start from

[aos/notebooks/cwfs/aos_miw_cwfs_intrinsic_check.ipynb](../../aos/notebooks/cwfs/aos_miw_cwfs_intrinsic_check.ipynb)
carries the diagnosis, in two sections.

- **Section 8, "NaN diagnosis — OCS hull vs CCS footprint"** attributes each donut's NaN
  to the rotated Optical Coordinate System (OCS) interpolator or to the per-detector Camera
  Coordinate System (CCS) interpolator, overlaying donut positions on the OCS sample
  coverage and the CCS footprints.
- **Section 9, "NaN vs field radius, and which term causes it"** gives per-half-sensor
  field-radius histograms with the NaN subset overplotted and lines at 1.725 deg and
  1.75 deg, plus a decomposition of which term produces the NaN.

The key relation encoded there: `getIntrinsicZernikes` is the OCS interpolator over all
Noll indices **plus** the per-detector CCS interpolator's Z4. A NaN therefore comes from
one of three causes, distinguishable by pattern — the OCS point outside the hull, where all
OCS Noll go NaN together as a flat bar; a single OCS Noll NaN in the source table, a spike;
or the CCS Z4 point outside that detector's footprint, Z4 only. The notebook verifies that
`(OCS-any | CCS-Z4)` reproduces the full-call NaN, so the attribution is complete. That
decomposition is what says whether extending the grid suffices or the per-detector CCS
footprints need extending too.

### Scope

- Extend the MIW grid so the hull exceeds 1.725 deg with margin.
- Check whether the per-detector CCS Z4 footprints also need extending.
- Regenerate the affected calibration products.
- Re-run the notebook's PASS/FAIL to confirm approximately 0% NaN on a run that does not
  apply the 1.725 deg cut.
- Open the ticket and pull request against `ts_intrinsic_wavefront`.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. How far does the grid extend?** Just past 1.725 deg, or out to the roughly 1.84 deg
field radius that uncut runs actually reach. The latter costs more grid points but removes
the dependence on the cut entirely.

**A:** _unanswered_

**Q2. Do the per-detector CCS Z4 footprints need extending as well,** or is the OCS hull
the only NaN source that matters in practice?

**A:** _unanswered_

**Q3. Does extending the grid require a Batoid recomputation of the intrinsic at the new
points,** and if so how long does that take?

**A:** _unanswered_

</details>

---

## 3. Confluence page documenting the AOS production runs in `/repo/main`

**Status:** Butler probe done, T614 set added 2026-10-04 — page not written ·
**Blocked on:** nothing; Q1 settled as cut-and-paste into an existing page

Write a comprehensive, up-to-date Confluence page describing the AOS production runs in
`/repo/main`: one section per track, each run with its Danish, ts_wep and donut_viz
versions, its binning, its pairing mode, whether it refit the World Coordinate System
(WCS), and its night coverage. There are 23 such runs.

**Goals:** Give the AOS group one current reference for which production processings exist
and which to use, and state the rule by which a collection counts as production so the page
can be regenerated rather than hand-maintained.

<details>
<summary>Known collections, the probe results, scope and open questions</summary>


### Known collections

Collection counts in `/repo/main`:

| set | count |
| --- | --- |
| all collections in `/repo/main` | 183,555 |
| matching `aos\|cwfs\|donut\|wep\|danish\|tarts\|fam` | about 2,935 |
| under `LSSTCam/runs/aos*` | 23 |
| everything else | about 2,912 |

All 23 are `CHAINED`. The roughly 2,912 excluded ones are of three kinds:

- **personal runs** — `u/brycek/*`, `u/jmeyers3/*`, `u/peterma2/*`, which is where TARTS and
  the Danish 1.3 blitz live. Josh's T614 set is the one the page most needs to mention by
  name even though it is excluded — see the table below;
- **bare aliases** — `aos_cwfs_aidonut`, `aos_cwfs_danish`,
  `aos_cwfs_danish_v1_aidonut`, `aos_cwfs_danish_v1_bin_2x`, `aos_cwfs_tie`,
  `aos_cwfs_unpaired_danish`, `aos_fam_danish`, `aos_fam_danish_triplets` and
  `aos_fam_tie`, which are pointers rather than runs;
- **timestamped RUN children** — for example `.../20260504T192032Z`, the output RUNs inside
  the CHAINED collections.

### The 23 production runs

Night counts and ranges below are from `aggregateAOSVisitTableRaw` (or
`aggregateZernikesRaw` where that's the only product), `findFirst=True`.
Paths are relative to the `LSSTCam/runs/aos/` prefix.

**`cwfs/` — corner wavefront sensors (8)**

| collection (under `cwfs/`) | danish | wep | dv | bin | mode | nights | day_obs range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2` | 1.0 | 17.3.0 | 4.2.0 | x2 | — | 18 | 20260315–20260428 |
| `danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/paired/refitWcs` | 1.2.0 | 17.7.0 | 4.7.0 | x2 | paired, refitWcs | 6 | 20260521–20260713 |
| `danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/paired/refitWcs/2025` | 1.2.0 | 17.7.0 | 4.7.0 | x2 | paired, refitWcs | **110** | **20250415–20251231** |
| `danish_1_2_0/wep_17_8_1/dv_4_7_2/bin_x2/paired/refitWcs/` | 1.2.0 | 17.8.1 | 4.7.2 | x2 | paired, refitWcs | 1 | 20260409 |
| `danish_1_2_0/wep_17_9_0/dv_4_8_1/bin_x2/paired/refitWcs` | 1.2.0 | 17.9.0 | 4.8.1 | x2 | paired, refitWcs | 3 | 20260512–20260713 |
| `danish_1_2_0/wep_17_9_0/dv_4_8_1/bin_x2/unpaired/refitWcs` | 1.2.0 | 17.9.0 | 4.8.1 | x2 | **unpaired**, refitWcs | 3 | 20260512–20260713 |
| `danish_1_2_0/wep_17_9_1/dv_4_8_5/binned_x2/refitWcsStamps/aidonut/` | 1.2.0 | 17.9.1 | 4.8.5 | x2 | **AIdonut**, refitWcsStamps | 6 | 20260521–20260713 |
| `danish_1_2_0/wep_blitz-prototype-v1/dv_blitz-prototype-v1/bin_x2/paired/donutBlitz` | 1.2.0 | blitz-v1 | blitz-v1 | x2 | paired, **blitz** | 3 | 20260512–20260713 |

**`fam/` — full array mode (7)**

| collection (under `fam/`) | danish | wep | dv | bin | mode | nights | day_obs range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `danish_1_0/wep_17_1_0/dv_4_1_0/bin_x1` | 1.0 | 17.1.0 | 4.1.0 | **x1** | — | 18 | 20260315–20260430 |
| `danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2` | 1.0 | 17.3.0 | 4.2.0 | x2 | — | 20 | 20260315–20260513 |
| `danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2/paired` | 1.0 | 17.3.0 | 4.2.0 | x2 | paired | 20 | 20260315–20260513 |
| `danish_1_0/wep_17_3_0/dv_4_2_0/bin_x2/unpaired` | 1.0 | 17.3.0 | 4.2.0 | x2 | **unpaired** | 20 | 20260315–20260513 |
| `danish_1_1_1/wep_17_3_0/dv_4_2_0/bin_x2/paired` | **1.1.1** | 17.3.0 | 4.2.0 | x2 | paired | 20 | 20260315–20260513 |
| `danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/paired/refitWcs` | 1.2.0 | 17.7.0 | 4.7.0 | x2 | paired, refitWcs | **115** | **20250415–20260711** |
| `danish_1_2_0_alpha0/wep_17_6_1/dv_4_5_0/bin_x2/paired/refitWcs` | 1.2.0α0 | 17.6.1 | 4.5.0 | x2 | paired, refitWcs | 20 | 20260315–20260513 |

**`fam_cwfs_triplet/` — FAM + corner triplets (5)**

| collection (under `fam_cwfs_triplet/`) | danish | wep | dv | bin | mode | nights | day_obs range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `danish_1_2_0/wep_17_10_0/dv_4_9_0/refitWcsStamps/aidonut/unbinned_v1/` | 1.2.0 | **17.10.0** | **4.9.0** | unbinned | AIdonut | 20 | 20260315–20260513 |
| `danish_1_2_0/wep_17_9_1/dv_4_8_5/refitWcsStamps/aidonut/binned_v1/` | 1.2.0 | 17.9.1 | 4.8.5 | binned | AIdonut | 20 | 20260315–20260513 |
| `danish_1_2_0_alpha0/wep_17_6_1/dv_4_5_0/ai_donut/binned` | 1.2.0α0 | 17.6.1 | 4.5.0 | binned | AIdonut † | 20 | 20260315–20260513 |
| `danish_1_2_0_alpha0/wep_17_6_1/dv_4_5_0/ai_donut/unbinned` | 1.2.0α0 | 17.6.1 | 4.5.0 | unbinned | AIdonut † | 20 | 20260315–20260513 |
| `danish_1_2_0_alpha0/wep_17_6_1/dv_4_5_0/bin_x2/refitWcs` | 1.2.0α0 | 17.6.1 | 4.5.0 | x2 | refitWcs | 20 | 20260315–20260513 |

† these two carry only `aggregateZernikesRaw`, not the joined `aggregateAOSVisitTable*`.

**`paired_cwfs_3mm_donuts/` — 3 mm defocus corner donuts (3)**

These three use an older naming convention — `wep_v17_6_1` with a `v`, `donut_viz_4_5_0`
spelled out, `danish_v1_2_0`, `bin2x` — with the version fields in the opposite order.

| collection (under `paired_cwfs_3mm_donuts/`) | danish | wep | donut_viz | bin | nights | day_obs range |
| --- | --- | --- | --- | --- | --- | --- |
| `wep_v17_10_0_alpha/donut_viz_4_9_0_alpha/danish_v1_2_0/bin2x` | 1.2.0 | 17.10.0α | 4.9.0α | 2x | 2 | 20260315–20260317 |
| `wep_v17_6_1/donut_viz_4_5_0/danish_v1_2_0/bin2x` | 1.2.0 | 17.6.1 | 4.5.0 | 2x | 2 | 20260315–20260317 |
| `wep_v17_8_0/donut_viz_4_7_0/danish_v1_2_0/bin2x` | 1.2.0 | 17.8.0 | 4.7.0 | 2x | 2 | 20260315–20260317 |

These are the **only** production collections carrying `intrinsicZernikes`.

### Josh's T614 runs — excluded as personal, but in use (2026-10-04)

Not production by the rule above (`u/jmeyers3/*`), but these are the collections the MIW
work actually builds on, so the page should name them and say why they are not production.
From `butler query-collections main "u/jmeyers3/t614*" --collection-type=chained`, 18
`CHAINED` collections: three corner tracks and three FAM tracks, crossed with a pupil-model
suffix.

| track | no suffix | `_legacy` | `_v1000` | `_v3.14` |
| --- | --- | --- | --- | --- |
| `t614_corner_paired` | yes | yes | yes | yes |
| `t614_corner_unpaired` | yes | yes | yes | yes |
| `t614_corner_full_detector` | yes | yes | yes | yes |
| `t614_fam_unpaired` | yes | yes | yes | yes |
| `t614_fam_paired` | yes | — | — | — |
| `t614_fam_full_detector` | yes | — | — | — |

The suffix is the **pupil model**, not a code version: `_legacy` is the pre-pupil-work
danish model, `_v3.14` the Batoid as-built model, `_v1000` as-built plus updated M1M3
measurements and the M1 baffles. The unsuffixed collections are `legacy` too — confirmed
2026-10-04 by flattening the chains, see Q7 of
[item 5](#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint).

`t614_fam_unpaired` is the Danish 1.3 blitz FAM processing the existing 50/34 MIW is built
on, and `t614_fam_unpaired_v1000` is what item 5 builds next.

### Identifying pairing mode from dataset types / tasks

Useful for the page because the collection *name* doesn't always say, and it's how
you'd verify a run rather than trust its path:

| signature | meaning |
| --- | --- |
| `aggregateAOSVisitTableCwfsTask`, `calcZernikesTask` | paired CWFS |
| `aggregateAOSVisitTableUnpairedTask`, `calcZernikesUnpairedTask`, `aggregateDonutTablesUnpairedTask` | unpaired |
| `aggregateAOSVisitTableTask`, `aggregateDonutTablesVisitTask` | FAM |
| `aggregateDonutTablesCwfsFamTask` | `paired_cwfs_3mm_donuts` |
| `donutBlitzResults`, `donutBlitzMonolithTask`, `formatBlitzTask` | blitz (and note the blitz production run has only `aggregateAOSVisitTableAvg`/`Raw`, **no** `zernikes`) |

### What the probe found

- **Coverage is lopsided.** Two runs have long baselines, 110 and 115 nights, both reaching
  back to 20250415, the start of LSSTCam images. The other 21 span 1 to 20 nights, and the
  20-night runs are all the same 20260315 to 20260513 window, the T614 test campaign.
- **The reference `wep_17_7_0` processing has the same 2026 gap in both tracks.** The CWFS
  side is split across two collections: the `/2025` child holds 110 nights, all within
  20250415 to 20251231 and no 2026 nights, while the parent holds 6 nights, 20260521 to
  20260713. The FAM side is one collection of the same shape: 115 nights, the same 2025 run
  plus 5 nights in 2026 (20260521, 20260522, 20260619, 20260709, 20260711). Neither track
  covers 20260101 to 20260520, and after 20260521 both are sparse at 6 and 5 nights rather
  than continuous.
- The FAM parent stops at 20260711 while CWFS reaches 20260713, so 20260713, one of item 1's
  target nights, has CWFS but no FAM in this processing.
- There is no production `danish_1_3`; the newest production Danish is `1_2_0`.
- Production AIdonut exists in three runs, but no single AIdonut run spans all three of
  item 1's target nights.
- Two long-baseline runs, `cwfs/.../wep_17_7_0/.../2025` at 110 nights and
  `fam/.../wep_17_7_0/...` at 115 nights, are the only ones with historical coverage;
  everything else is 20 nights or fewer.

### Scope

- Write the page from the tables above: one section per track
  (`cwfs`, `fam`, `fam_cwfs_triplet`, `paired_cwfs_3mm_donuts`), each run with its
  danish/wep/dv versions, binning, pairing mode, refitWcs status and night coverage.
- Document the naming convention as a taxonomy —
  `<track>/<danish>/<wep>/<dv>/<binning>/[pairing]/[refitWcs|aidonut|donutBlitz]` —
  and flag the deviations: the three `paired_cwfs_3mm_donuts` runs with their different
  field order and `v`-prefixed versions, the four collections with trailing slashes in their
  names, and `binned_x2` against `bin_x2` against `bin2x` against `binned` and `unbinned`.
- State the exclusion rule: production is `LSSTCam/runs/aos*`, with personal, alias and
  timestamped-RUN collections excluded.
- Add a short non-production section for Josh's 18 `u/jmeyers3/t614*` collections: the
  track-by-pupil-model grid, what each suffix means, and that the MIW builds use them. They
  are excluded by the rule but are the processing the AOS group is currently working from, so
  omitting them would make the page misleading rather than merely incomplete.
- Add the alias to current-target mapping for the nine bare `aos_*` aliases.
- Add a "which run should I use?" recommendation per track.
- Explain the 2026 coverage gap in the reference processing.
- Include the script or notebook that generated the tables, so the page can be regenerated
  rather than hand-maintained.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. How is the page published?** Writing the Confluence markup is one task; creating or
updating a page in the Rubin Confluence space is an outward-facing action. Either the
deliverable is a local file to paste in, or the page is created through the API — and if the
latter, which space and parent page. Nothing gets published without confirmation.

**A:** _I will cut and paste into an existing Confluence page_

**Q2. Does an existing Confluence page cover this and need updating** rather than a new one
created? "Up to date" in the request suggests there may be a stale one.

**A:** _yes_

**Q3. Is `/repo/embargo` in scope,** or is `/repo/main` the whole story for production?
Embargo may hold recent runs not yet in main.

**A:** _previously we were using both but right now we only have ouput in /repo/main_

**Q4. Which product do the night counts refer to?** They come from the aggregate visit
tables here. If a run's `zernikes` or donut-level products have different coverage, the page
should say so — worth one spot-check before publishing.

**A:** _just using the aggregate visit tables is fine, except for the blitz processing  which has different output_

</details>

---

## 4. `pupil` — measure the donut pupil geometry, data against model

**Status:** not started · **Blocked on:** Batoid 0.9.0, for the model variants only

New topic, `pupil`. Measure the 50% flux point of the inner and outer ring of each donut as
a function of angle about the donut center, and compare that radius-versus-angle curve
between the data image and the fitted model image. Run it on both the corner wavefront
sensor (CWFS) and the Full Array Mode (FAM) products, parameterized by Batoid optical model
so the as-built v3.14 and the v1000 variant can each be compared against the data.

**Goals:** Measure the donut pupil geometry directly, and determine whether that measurement
distinguishes the v3.14 and v1000 optical models.

<details>
<summary>Collections, the probe findings, the model variants, scope and open questions</summary>


### Known collections

Probed `u/jmeyers3/t614_corner_unpaired` / `donutBlitzResults`: 961 visits, about 124 rows
per visit, of which roughly half pass `group_fit_success & snr>50`. The image columns
present per row:

| column | shape | what it is |
| --- | --- | --- |
| `stamp` | (167, 167) | raw postage stamp, **unbinned** |
| `wf_img` | (83, 83) | the data image actually fitted, **binned ×2** |
| `model_img` | (83, 83) | the fitted model, same binning/grid as `wf_img` |

`wf_img` and `model_img` are the matched pair: same grid, sums agreeing to about 1%
(5.96e7 against 6.03e7 ADU or electrons on a test donut), so they difference with no
resampling. `stamp` is the wider unbinned cutout at 2× the scale, confirmed by half-flux
radii scaling as 39 and 65 unbinned pixels to 19 and 33 binned pixels.

Supporting columns present per row: `donut_radius` (62 to 68 unbinned stamp pixels),
`fit_dx` and `fit_dy` (the fitted center offset, in pixels), `bkg` and `bkg_std`,
`inner_frac`, `outer_frac`, `outer_sector_minmax_frac` (already a sector-based azimuthal
statistic), and `blend_frac`. Image metadata: `stamp_size = 167` pixels,
`stamp_frame = CCS`, pixel values in ADU or electrons.

The same image columns are present in the FAM product `u/jmeyers3/t614_fam_unpaired`. An
earlier probe of its parquet found none because `blitz_reader` does not write them:
167×167 float64 pixels per donut over 2.9M donuts is prohibitive, and the Danish 1.2 schema
has no counterpart columns. One visit of 124 rows is about 28 MB in `stamp` plus about
14 MB in the 83×83 pair.

The 2026-06 pupil-mask study
([wfs/docs/danish_pupil_mask_findings.md](../../wfs/docs/danish_pupil_mask_findings.md))
compared danish's analytic mask against the batoid vignetting boundary and never touched
donut images. Two of its findings apply here: danish and ts_wep fit against the design
model `LSST_r`, approximately v3.3, and against the as-built model the filter aperture
alone moves the outer pupil edge by about +16 mm.

### Existing machinery to build on

| what | where |
| --- | --- |
| scalar-column reader for the blitz products | `aos/code/fam_processing/blitz_reader.py` |
| the Danish 1.3 param set and its frozen provenance | `aos/param_sets.yaml` (`danish_1_3_test`), `aos/output/fam_processing/danish_1_3_test/provenance.yaml` |
| v3.14 and v1000 model YAMLs | [danish/danish/data](https://github.com/jmeyers314/danish/tree/main/danish/data) and [batoid `releases/0.9` data/LSST](https://github.com/jmeyers314/batoid/tree/releases/0.9/batoid/data/LSST) |

The scalar columns come through `blitz_reader` as normal; the images must be read directly
from the Butler for a selected subset of donuts, chosen by visit, detector or field radius,
never as a full-sample parquet join. The spider rotates with the camera, so in detector
coordinates its angle moves between visits — averaging over rotator angle averages the
spider away, which is useful rather than a nuisance.

### Two findings from the probe

**Azimuthally averaged radii agree.** Over 737 donuts (`snr>100`, 12 visits), half-flux
radii from radial profiles:

| ring | data (binned px) | model (binned px) | data − model (binned px) |
| --- | --- | --- | --- |
| inner | 19.983 | 19.970 | **+0.012 ± 0.326** |
| outer | 32.744 | 32.701 | **+0.043 ± 0.208** |

The mean pupil scale is therefore right to 0.012 to 0.043 binned pixels, which is why the
interesting content is azimuthal rather than radial-average: a plain radial profile shows
nothing.

**The model is azimuthally smooth and the data is not.** Median azimuthal profile in the
mid-annulus (22 to 30 binned pixels in radius, 10 deg bins, dimensionless — each profile
over its own median) on a high-SNR donut:

```text
data   0.75 0.66 0.88 0.76 0.73 ... 1.38 1.29 ... 0.74 0.94 0.93 0.91
model  0.75 0.76 0.79 0.84 0.89 ... 1.18 1.19 ... 0.93 0.87 0.81 0.77
```

Both share the same broad low-order trend, which is the wavefront, but the data carries
sharp localized dips — 0.66 at 10 deg, 0.70 at 90 deg, 0.76 at 190 deg, all dimensionless
ratios to the profile median — that the model has nowhere. 7 data bins fall below 0.8
against 4 for the model, and the model's are broad while the data's are narrow. That is the
unmodelled spider, and it confirms that a naive 50% crossing search latches onto spider
edges.

### The two optical-model variants

Both are Batoid optical models, not code in `rubin-work`.

| variant | what it is |
| --- | --- |
| v3.14 | the Rubin Batoid as-built model |
| v1000 | as-built plus updated M1M3 measurements, and the outer and inner baffles of M1, which are physically present at the telescope but in neither the default nor the as-built model |

The measurement — inner and outer half-flux radii of the donut — is the direct observable
for what v1000 changes, so the variant comparison is the primary signature rather than a
side effect.

M1 inner radius is 2.558 m in v3.14 and 2.5833 m in v1000, identically in the danish and
batoid repos. M3 changes too: outer 2.508 m to 2.48511 m, inner 0.55 m to 0.52735 m.
`RubinObsc.yaml` in danish is a symlink to the v1000 file, so danish's "default" is already
v1000.

M1's outer annulus radius is unchanged at 4.18 m. Instead v1000 adds three surfaces absent
from v3.14:

| new surface | type | radius (m) | z (m) |
| --- | --- | --- | --- |
| `M1Baffle1` | Baffle, `ClearCircle` | 4.165 | 0.48283, the entrance-pupil plane |
| `M1Baffle2` | Baffle, `ClearCircle` | 4.165 | 0.48283 |
| `CameraBody` | Baffle, `ObscCircle` | 0.80469 (camera body outer diameter 1.60938 m) | 3.5002 |

The outer edge is therefore set by a baffle 15 mm inside M1's 4.18 m rim while the inner
edge moves outward by 25.3 mm via M1's annulus, predicting an outer radius slightly smaller
and an inner radius slightly larger than the default model — a narrower annulus on both
sides, which the radius-versus-angle measurement tests directly.

`pupilSize` is 8.33 m in v1000 against 8.36 m in v3.14 and `LSST_r`, with
`pupilObscuration` 0.612 dimensionless (obscuration radius over pupil radius) in v1000. That
0.36% scale change shifts both radii together, so a common scale plus independent inner and
outer offsets separates it from the baffle effect.

No weekly ships v1000. `w_2026_33`, `w_2026_35` and `w_2026_38` — the newest on cvmfs as of
2026-09-22 — all carry Batoid 0.8.1 and danish 1.2.0, with no `Rubin_v1000_*` files and only
the v3.14 `RubinObsc` in danish's data directory. Batoid 0.9.0, which carries v1000, was
released 2026-09-16, days before that check.

### Scope

- Create the topic area `pupil/` with its own `Snakefile`, `config.yaml` and
  `docs/studies/pupil.md`, over both the corner and the FAM products.
- Write the edge finder: per-donut, per-angle 50% flux crossing for the inner and outer
  ring, measured from the fitted center `fit_dx` and `fit_dy`, on `wf_img` and `model_img`
  on their common grid, normalized per donut with `bkg` and `bkg_std` as the floor.
- Reject the spiders.
- Produce the radius-versus-angle product: data and model overlaid for the inner and outer
  ring, per donut and stacked — once in a pupil-fixed frame of field angle plus rotator so
  real pupil features add and spiders smear, once in a spider-fixed frame so the spider
  adds.
- Fit the amplitude and phase of the low-order azimuthal harmonics for data, model and
  residual, covering decenter (m=1), ellipticity (m=2) and m=3 or m=4 mount or vignetting
  terms.
- Measure whether the data-minus-model azimuthal residual grows toward the field edge.
- Parameterize the edge finder and the radius-versus-angle product by optical model from
  the start rather than retrofitting it.
- Compute the predicted inner and outer radii for v3.14 and v1000 from the model YAML
  aperture radii, and compare against the measured radii.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. How are the spiders rejected?** Three options, in increasing sophistication. Reject
angular sectors where the data-over-model mid-annulus ratio dips sharply, which isolates the
spider with no spider model. Or mask the expected sectors a priori from the known spider
geometry and rotator angle, then verify the dips land where predicted, which doubles as a
spider-position check. Or fit a smooth low-order function of angle and clip the outliers.
The first and second cross-check each other.

**A:** _unanswered_

**Q2. Which optical model did the Danish 1.3 blitz actually use?** "Default" is ambiguous
because `RubinObsc.yaml` in danish is a symlink to the v1000 file. This determines whether
there are two comparisons to make or one, and whether the data already carries the baffles.
The `provenance.yaml` beside `aos/output/fam_processing/danish_1_3_test/` records the
collection and calibration run and is the place to start.

**A:** _unanswered_

**Q3. Is the right target the current blitz output, or a rerun with the new pupil model?**
Josh's note says these runs are still on the old pupil model. Measuring against the old
model is a useful baseline, but the pupil-geometry conclusions would be about a superseded
model.

**A:** _unanswered_

**Q4. How is v1000 obtained?** It needs Batoid 0.9.0, released 2026-09-16, and no weekly
ships it yet. Either wait for a weekly carrying Batoid 0.9.0 or later, `pip install
batoid==0.9.0` into a user environment layered on the weekly and record it in the study
provenance, or read the model YAMLs directly from GitHub for a geometry-only comparison.
The third option is enough to test the central prediction and needs no new Batoid at all;
a full install is only needed to regenerate model images.

**A:** _unanswered_

**Q5. Are the model variants a config switch in the blitz processing, or a full rerun?**
The cost of the variant comparison hinges on this.

**A:** _unanswered_

**Q6. Binned pair or unbinned stamp?** The binned `wf_img` and `model_img` share a grid with
no resampling but are 2× coarser, and the 0.3 binned pixel scatter is already near the
measurement floor. The unbinned `stamp` is finer but the model must then be regenerated or
upsampled to match. Leaning binned for the first pass since the pair is already
co-registered.

**A:** _unanswered_

**Q7. Does the M1M3 surface update in v1000, beyond the baffles, shift the fitted Zernikes**
enough to change the model image shape independently of the aperture edges? If so the two
effects need separating.

**A: partial, from item 5 step A, 2026-10-05 — and the answer leans yes.** Rebuilding the
MIW on v1000 against the legacy pupil moved it by 0.1027 of amplitude, but the difference
is **astigmatism-led**: Noll 6 and 5 carry 0.4032 and 0.2969 of the difference power while
the spherical terms Noll 11 and 22 carry only 0.0143 (dimensionless) — less than a pure
retrieval change carries. An axisymmetric aperture-edge change should load spherical, so
whatever dominates that difference is **not** the baffle. A non-axisymmetric M1M3 surface
update is one candidate; the confounded `danish` code-version change
(`5037d9f3` to `ca41ae8c`) is the other, and the MIW products cannot separate them. Caveat
for reading this across: that measurement is at the **MIW** level, after a per-visit
optical-state fit that partly absorbs the change, not at the single-donut model-image level
this item works at. See [item 5](#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint)
and `aos/docs/studies/miw.md`.

**Q8. Does the inner ring behave differently from the outer?** The inner half-flux radius
scatter is 0.326 binned pixels against 0.208 for the outer, which may be intrinsic — fewer
pixels, lower contrast — or may be real M2-baffle structure.

**A:** _unanswered_

</details>

---

## 5. Rebuild the MIW on the v1000 pupil model, then under the RBR constraint

**Status:** step A complete 2026-10-05; step B's RBR implemented in ts_ofc, constrained
build not yet run · **Blocked on:** nothing

**Step A result, 2026-10-05.** The v1000 pupil moves the MIW by 0.0053 µm of wavefront
against the legacy build's own 0.0518 µm — 0.1027 of amplitude — and the inferred FWHM from
0.1645 to 0.1678 arcsec (+0.0033 arcsec). But the signature is **not** the one a baffle
predicts: the spherical terms Noll 11 and 22 carry only 0.0143 of the difference power
(dimensionless), *less* than the 0.0609 the Danish 1.2-vs-1.3 retrieval change carried, while
astigmatism Noll 6 and 5 carry 0.4032 and 0.2969. Unexplained, and confounded with the code
version — see [the handoff](../status/item5_miw_pupil_model_handoff.md) for the two readings
and why `_legacy` would be the clean discriminator.

**From item 7, 2026-10-05 — relevant if step A goes through ts_wep.** ts_wep's blitz builds
its donut pupil from `_INSTRUMENT.maskParams` (a singleton loaded from
`policy/instruments/LsstCam.yaml`), **not** from danish's pupil YAMLs. Those shipped
`maskParams` are v3.14 (`diameter: 8.36 m`, M1 inner `2.558 m`) and there is **no v1000
variant of them**, so selecting v1000 inside blitz is not a file swap — it needs v1000
`maskParams` polynomials generated from the v1000 batoid model. Confirm which code path this
item's rebuild actually takes before assuming the pupil model can be chosen by filename.

Two sequential steps on the Measured Intrinsic Wavefront (MIW), in this order (settled
2026-10-04). **Step A, next:** rebuild the 50 DOF / 34 v-mode (50/34) MIW on the
`v1000` pupil model and compare it against the existing Danish 1.3 blitz build, which
uses the current default pupil. **Step B, after that comparison:** apply the
Range-Bounded Recovery (RBR) constraint to the per-visit optical-state recovery of one
or more of these MIW builds, and compare constrained against unconstrained.

**Goals:** Step A — determine whether modelling the baffle just inside M1's outer radius
changes the measured intrinsic wavefront, and by how much. Step B — determine whether the
MIW build's recovered optical state can be made physically reachable, by RBR, without
degrading the MIW.

<details>
<summary>Known collections, existing machinery, scope and open questions</summary>

### Known collections — Josh's T614 processing runs

Josh Meyers' latest T614 runs in `/repo/main`, from
`butler query-collections main "u/jmeyers3/t614*" --collection-type=chained`, all
`CHAINED` (18). They are a grid of three tracks crossed with four pupil-model suffixes:

| track | no suffix | `_legacy` | `_v1000` | `_v3.14` |
| --- | --- | --- | --- | --- |
| `t614_corner_paired` | yes | yes | yes | yes |
| `t614_corner_unpaired` | yes | yes | yes | yes |
| `t614_corner_full_detector` | yes | yes | yes | yes |
| `t614_fam_unpaired` | yes | yes | yes | yes |
| `t614_fam_paired` | yes | — | — | — |
| `t614_fam_full_detector` | yes | — | — | — |

The full names are `u/jmeyers3/<track><suffix>`.

**The suffix is the pupil model.** `_legacy` is the model danish shipped before the pupil
work; `_v3.14` is the Batoid as-built model; `_v1000` is as-built plus the updated M1M3
measurements and the M1 outer and inner baffles — see
[item 4](#4-pupil-measure-the-donut-pupil-geometry-data-against-model) for what v1000
changes, including the `M1Baffle1`/`M1Baffle2` `ClearCircle` surfaces at radius 4.165 m
that sit 15 mm inside M1's 4.18 m rim. The unsuffixed collections are **`legacy` as well** —
confirmed 2026-10-04, not assumed: the unsuffixed and `_legacy` chains share 12 of their 13
flattened RUN children, differing only in their own output RUN. See Q7.

**What is already built:** the existing 50/34 MIW uses `t614_fam_unpaired`, which is the
same legacy configuration as `t614_fam_unpaired_legacy` (Q7, confirmed). **Step A builds the
same 50/34 MIW on `t614_fam_unpaired_v1000`** and compares the two — a pupil-model
comparison, legacy against v1000, with the scheme held fixed.

### Existing machinery to build on

| piece | path |
| --- | --- |
| the MIW build chain | `aos/Snakefile`, `rule build_intrinsic` through `rule refit_mi`, calling `run_build_intrinsic.py`, `run_intrinsic_split.py`, `run_dz_fit.py` in the installed `ts_intrinsic_wavefront` |
| the MIW build entries | `aos/mi_config.yaml`, `measured_intrinsics:` keyed by param set |
| two-build term-by-term comparison | [aos/code/miw/compare_miw_versions.py](../../aos/code/miw/compare_miw_versions.py) |
| how far the current build's states sit outside the allowed range | [aos/code/miw/check_dof_ranges.py](../../aos/code/miw/check_dof_ranges.py) |
| the RBR solver | [smatrix/code/regularized_inversion.py](../../smatrix/code/regularized_inversion.py), `dof_range_vector` and `invert_range_penalty` |
| the AOS-side RBR wrappers | `aos/code/bounce/bounce_lib.py`, `rbr_module`, `rbr_dof_per_pair`, `rbr_deltas` |
| residual wavefront to FWHM | [aos/code/aos_fwhm.py](../../aos/code/aos_fwhm.py), `residual_dW`, `zj_to_fwhm`, `fp_fwhm` |
| v-mode engine and DOF sets | `aos/code/aos_state.py`, `make_state_estimator`, `vmodes_from_dofs`, `recover_optical_state`, `DOF_SETS`, `N_MODES` |
| the equation-level residual derivation | `aos/docs/miw_coadd_equations.md`, section 4 |

The MIW library and its runners live in the installed `ts_intrinsic_wavefront` package;
`aos/` is a thin client, so the scheme enters as config rather than as a code change. The
(n_dof, n_keep) pair is passed to `build_ofc_svd(iZs, k_min, k_max, n_keep, n_dof=...)`
package-side. `aos_state.py` defines the DOF sets `hexapod_10`, `standard_22` and `all_50`
with default mode counts 10, 12 and 20, not the named scheme tuples.

The existing MIW builds are 50/34 and sit under `aos/output/miw/`, with the `_5rot` split
being the canonical product (`intrinsic_split_maps.parquet`). `mi_config.yaml` already
carries the `danish_1_3_test` param set with its `pathA_50_34_i` and `pathA_50_34_i_5rot`
entries, so the v1000 build is a sibling param set with the same two entries pointed at a
different collection. A `build_from` entry reuses its parent's per-rotator-bin grids and
re-runs only the split and downstream — so a **pupil-model change is a fresh build**, not a
`build_from`, because the wavefronts themselves differ.

Both the Danish 1.2 and the Danish 1.3 `mi_config` entries were written with matched knobs
so two MIW versions "differ only in the wavefronts"; that is exactly the property step A
needs, with the wavefront difference now being the pupil model rather than the retrieval.

RBR is implemented and in production use by the bounce study, with `kappa = 4` and
`power = 3` on those legs, and the penalty is smooth rather than a hard bound — a recovered
amplitude can finish outside its range, by up to a factor of 1.157 dimensionless (recovered
amplitude over allowed range) on the bounce legs. It has never been applied to the MIW
build: `check_dof_ranges.py` measures how far the existing build's recovered states fall
outside the allowed range `r_j` and changes nothing.

Step B also has a precedent now.
[Completed item 7](completed-todos.md#7-open-loop-and-deviation-recovered-optical-state-for-science-visits-three-schemes)
built the
`50_34_rbr` arm over 96,278 science visits and found unconstrained 50/34 asking a **median
33x the force-limited range, with a tail to 7613x** (dimensionless, recovered amplitude over
allowed range), while RBR gave back only about 0.025 arcsec of the 0.13 arcsec that 50/34
wins over 22/12 in residual wavefront FWHM. So on science visits the constraint is cheap
and the unconstrained state is unreachable; step B tests whether the same holds for the MIW
build, where the wavefront being inverted is a per-visit deviation against the grid rather
than a science-visit deviation. The corner-basis shim and the `50_34_rbr` solver path from
that item are reusable — see
[`notes/status/item2_optical_state_build_handoff.md`](../status/item2_optical_state_build_handoff.md).

### Scope

**Step A — the v1000 pupil model (do this next).**

- Optionally make Q7 airtight by reading the pupil-model task config out of both output RUNs.
  The chain comparison already shows the unsuffixed collection is legacy; this would confirm
  the two runs differ *only* in the pupil model.
- Add a `mi_config.yaml` param set for `u/jmeyers3/t614_fam_unpaired_v1000`, with
  `pathA_50_34_i` and `pathA_50_34_i_5rot` entries matching the `danish_1_3_test` knobs
  exactly, so the only difference from the existing build is the pupil model.
- Run the full MIW chain on it — a fresh build, not a `build_from`, because the wavefronts
  differ.
- Report how many visits the two builds have in common, and do every comparison on that
  common set rather than on each build's own sample.
- Compare the two MIW term by term and as field maps, with
  [aos/code/miw/compare_miw_versions.py](../../aos/code/miw/compare_miw_versions.py).
- Convert each MIW to an inferred full width at half maximum (FWHM) in arcsec over the focal
  plane and report both, so the pupil-model change is also stated as image quality.
- Report whether the difference concentrates at the field edge and at the pupil edge, which
  is where a baffle 15 mm inside M1's rim should act, rather than only reporting a global
  number.
- Report the recovered DOF and v-mode values of each build; a pupil-model change should move
  the MIW, and it should also be checked for whether it moves the subtracted optical state.

**Step B — RBR on the MIW (after the step A comparison).**

- Carry **`danish_1_3_v1000` alone** forward (Q8, answered 2026-10-05).
- **Done 2026-10-05:** RBR implemented in **`ts_ofc`** on `tickets/RSO-1007` (`979ec73`) —
  `range_bounded_recovery`, a `DoubleZernikeStateEstimator` so the DZ basis is a first-class
  ts_ofc capability, and `OFCData.dof_ranges`. `ts_intrinsic_wavefront` takes its SVD from
  ts_ofc on `tickets/RSO-809` (`957a54f`), bit-for-bit verified, so ts_ofc holds the only
  SVD and state estimator in the code base. `smatrix/code/regularized_inversion.py` is now
  the derivation and cross-check reference, not the implementation. Neither branch pushed.
- Apply the RBR penalty inside the MIW build's per-visit optical-state recovery: the runner
  calls `svd.dof(...)` and needs `dof_range_bounded(wavefront)`, plus an `mi_config.yaml`
  knob. **Still to write.**
- Compare constrained against unconstrained for each build carried forward, using the
  achieved residual `dW - S (d / w)` rather than the subspace projection (Q5) — now
  `lsst.ts.ofc.achieved_residual`.
- Report the MIW as an inferred FWHM in arcsec over the focal plane for each arm.
- Report the recovered DOF values against the allowed range `r_j`, in µm and arcsec as
  appropriate per DOF, and the corrected v-mode sizes — the comparison with the median 33x
  range excess that completed item 7 found on science visits is the point.
- Quantify how the subtracted optical-state correction changes between the arms, per DOF and
  per v-mode.

**Both steps.**

- Write the study up in `aos/docs/studies/miw.md`.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Which param set do the three builds run on?** The Danish 1.2 param set has a complete
50/34 build to compare against, while `danish_1_3_test` has entries whose build has not been
run. Using Danish 1.2 gives an existing baseline; using Danish 1.3 folds in the newer
retrieval but means building all three from scratch.

**A:** _lets move to Danish 1.3 for this study_

Still the answer after the 2026-10-04 reframing: Danish 1.3 blitz is `t614_fam_unpaired`,
which is the existing 50/34 build and the step A baseline. The v1000 build is the same
Danish 1.3 blitz retrieval on a different pupil model.

**Q2. What `kappa` and `power` does the MIW RBR use?** The bounce study uses `kappa = 4` and
`power = 3` dimensionless. Whether those transfer to the MIW recovery, where the wavefront
being inverted is a per-visit deviation rather than a bounce-leg difference, needs checking
rather than assuming.

**A:** _I want to use those same parameters to begin with_

**Q3. Is 22/12 built with `standard_22` at 12 modes, or at its `N_MODES` default?**
`aos_state.DOF_SETS["standard_22"]` is the 22-index set and `N_MODES` gives it 12, but the
mode count is set independently by `n_modes` and the scalar 22 silently picks DOF 0 to 21,
which is the wrong set.

**A:** _22/12 uses the standard_22 and not the first 22 DoF, those 22 Dof are the 10 hexapod, the first 7 M1M3 and the first 5 M2 _

**No longer binds this item** after the 2026-10-04 reframing: 22/12 is not one of the MIW
builds any more. The trap it names is real and still live wherever a DOF set is passed as a
scalar; the root `aos/CLAUDE.md` records the explicit index list.

**Q4. Does RBR belong in the MIW build, or only as a diagnostic on top of it?** The bounce
note records that an RBR amplitude is a constrained estimate rather than a measurement, and
that the default recovery remains the estimator. If that holds for the MIW, the RBR build is
a comparison arm rather than a candidate replacement.

**A:** _The RBR's constrained DoF results are a candidate replacement for unconstrained 50/34 and not merely a comparison, and the estimator should be the constrained DoF and constrained v-mode values. This is the achieved_residual. _

**Q5. Which residual is compared?** The subspace-projection residual from
`aos_fwhm.residual_dW` cannot see a regularizer that trades wavefront against amplitude, so
it will not show what RBR costs. The bounce study uses `achieved_residual` against the
applied DOF instead.

**A:** Use the achieved residual, `regularized_inversion.achieved_residual(dW, d, svd)`.

**Q6. Does the k-truncation of the MIW basis interact with the scheme change?** The build
runs `k_min = 1` to `k_max = 6`, and the existing leakage analysis
(`aos/calibration/miw/umode_k_leakage_50_34.npy`) is specific to 50/34.

**A:** _I do not need to repeat that leakage analysis and I expect the k=1..6 is still sufficient to capture the optical state_

**Q7. What pupil model did the unsuffixed `t614_fam_unpaired` use?** Aaron's reading is that
the unsuffixed collections are `legacy`, with Josh pointing at danish's current default
pupil, but that is not confirmed. It matters because the existing 50/34 MIW was built on the
unsuffixed collection: if it is `legacy` then step A is legacy-against-v1000, and if it is
already v1000 then the existing build is the v1000 build and step A is a different
comparison. Note `RubinObsc.yaml` in danish is a symlink to the v1000 file (item 4), so
"danish's default" is not obviously legacy. Resolvable by reading the task configuration out
of the collection, or by asking Josh.

**A:** **Legacy — Aaron's reading was right.** Measured 2026-10-04 from the Butler, not
assumed: `u/jmeyers3/t614_fam_unpaired` and `u/jmeyers3/t614_fam_unpaired_legacy` flatten to
13 RUN children each and **share 12 of them**. The 12 shared ones are inputs
(`LSSTCam/raw/all`, calibs, refcats); each chain then has exactly one output RUN of its own,
`t614_fam_unpaired/20260912T224304Z` against
`t614_fam_unpaired_legacy/20260930T165721Z`. So they are two separate runs of the same
legacy configuration, 18 days apart, and **step A is legacy-against-v1000** as planned. The
existing 50/34 MIW's baseline is legacy.

Worth keeping in mind as the reason this needed checking at all: in danish 1.3.0 as shipped
in `w_2026_39`, `RubinObsc.yaml` is **byte-identical** to
`RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml` by checksum, so danish's *own* default pupil is
v1000. "No suffix = legacy" is therefore true of Josh's collections but is **not** a general
rule and cannot be inferred from danish's default. See
[item 7](#7-giant-donuts-pupil-models-spiders-and-the-intraextra-z11-split).

**Now confirmed, 2026-10-04, and the answer is NO — it does not differ only in the pupil
model.** Read from the task configs on disk (the Butler `get` fails in `w_2026_39`, which
lacks `lsst.ts.wep.blitz`, so read `<run>/donutBlitzFam*_config/*.py` directly). The
unsuffixed baseline is an older code version as well as an older pupil model:

| | `t614_fam_unpaired` | `t614_fam_unpaired_v1000` |
| --- | --- | --- |
| pupil mask | no `maskModel` field — the older task predates it | `RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml` |
| `danish` | `5037d9f3` | `ca41ae8c` |
| `ts_wep` | `9651cd23` | `639a89d9` |
| task label | `donutBlitzFamTask` | `donutBlitzFam` |
| visits | 966 over 15 nights | 966 over the same 15 nights |

**`_legacy` against `_v1000` IS airtight**, and is the cleaner pair: same `danish` commit
`ca41ae8c`, same task label, and their config assignment lines differ in **exactly one
line** —

```
config.wavefrontFit.maskModel='policy:masks/LsstCamLegacy.yaml'           # _legacy
config.wavefrontFit.maskModel='RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml'  # _v1000
```

Aaron's call 2026-10-04: **use the existing build as the baseline anyway** — "I know that
the existing danish 1.3 used an older code version, but there were only small changes
since then. So v1000 against the existing build is fine." So the pupil model is the
leading term and the version change is a stated caveat, not a third build. `_legacy`
stays available if a result ever needs the airtight pair.

All three collections cover **identical visits** (966 `donutBlitzFamResults`, the same 15
nights), so `day_obs_min/max` copy across unchanged and the common-visit set is not
limited by coverage.

**Q8. Which build or builds does step B apply RBR to?** Aaron's phrasing is "one or more of
these MIW", to be decided after the step A comparison. If step A shows the pupil model barely
moves the MIW, one build suffices; if it moves it, RBR on both separates the pupil effect
from the constraint effect.

**A: `danish_1_3_v1000` only — Aaron, 2026-10-05.** Step A found the two MIW agree to
0.1027 of amplitude (difference RMS over build-A RMS), so maintaining two RBR arms buys
little. Step B runs on the v1000 build alone.

</details>

---

## 6. Guider star second moments, image and centroid motion, into the value-added DB

**Status:** not started · **Blocked on:** nothing

Calculate the guider star second moments — both of the images themselves and of the centroid
motion — run that over all guider stars in batch, and add a set of this information to the
value-added database. The per-exposure measurement code exists and produces both moment
kinds; what is missing is the full-survey batch run and a home for the results in the
database.

**Goals:** Have per-visit guider image-quality and image-motion measures available for every
guider exposure, joinable to the telemetry and recovered optical state already in the
database, so that atmospheric and tracking contributions to the point spread function (PSF)
can be separated from the optical ones.

<details>
<summary>What exists, the two moment kinds, scope and open questions</summary>

### Known collections

Guider raws come from `/repo/main`, collections `LSSTCam/raw/guider` and `LSSTCam/raw/all`,
as defaulted in `run_guider_moments.py`. The nights to process are discovered by
`list_guider_exposures.py` rather than listed by hand.

### Existing machinery to build on

| piece | path |
| --- | --- |
| the moment decomposition | [guider/code/guiderMoments.py](../../guider/code/guiderMoments.py) |
| the per-exposure driver | [guider/code/run_guider_moments.py](../../guider/code/run_guider_moments.py) |
| per-night combination | [guider/code/combine_moments.py](../../guider/code/combine_moments.py) |
| exposure discovery | [guider/code/list_guider_exposures.py](../../guider/code/list_guider_exposures.py) |
| the pipeline, rules `moments` and `combine` | [guider/Snakefile](../../guider/Snakefile), [guider/run_snake.sh](../../guider/run_snake.sh) |
| the weighting study | [guider/docs/moment_weighting_analysis.md](../../guider/docs/moment_weighting_analysis.md) |
| the session handoff | [guider/docs/status/handoff_guider_session_2026-09.md](../../guider/docs/status/handoff_guider_session_2026-09.md) |
| the database, registry pattern and readers | [value_added/code/efd_db.py](../../value_added/code/efd_db.py) |

**Both moment kinds already exist**, which is the main thing this item does not have to
build. `guiderMoments.decomposeDetector` returns a `DetectorMoments` holding, per detector:
the moments of the mean coadd centered on the mean weighted centroid; the flux-weighted
covariance of the per-stamp weighted centroids, which is the image motion, with the
centroid-noise floor `<err**2>` subtracted; and the per-stamp moments and centroids kept for
time-series and power-spectral-density work. All moments are in arcsec², in a `ShapeMoments`
symmetric second-moment matrix.

The measurement is a fixed-width Gaussian weighted centroid, iterated, then unweighted
second moments about that centroid within an aperture — deliberately a minimum-variance
centroid rather than an adaptive one. Hartmann-sensor-manager (HSM) moments are recorded
alongside under `hsm_*` in the PIFF naming (`e0 = M11 = T`, `e1 = M20 = Q1`,
`e2 = M02 = Q2`, normalized as `e1n`/`e2n`). A v6 config note in the module records that the
centroid variance is already split fast/slow by a Gaussian smooth at 1.6 s full width at half
maximum (FWHM), which is a ready-made decomposition of the motion into tracking and
atmospheric timescales.

The pipeline writes three parquets per exposure into
`output/night_<dayObs>/seq/<seqNum>_{moments,stars,metrics}.parquet` and combines them to
`guider_moments_<dayObs>.parquet` per night. So the per-night product exists; the gap is
running it over the whole survey and landing a summary in the database.

**The database has no guider table.** The eight tables are `visit_telemetry`,
`m1m3_thermal_r2`, `optical_state`, `fam_dz`, two registries, `column_coverage` and
`fetch_log`. A guider table is new, and the schema doc states the rule it has to satisfy: the
database holds only quantities expensive to fetch or intricate to compute, never a ConsDB
mirror. Guider moments qualify on the compute side.

The two schema idioms to choose between are documented and both have a precedent here:
`visit_telemetry` is wide, one row per exposure, where the quantity set is genuinely fixed;
`fam_dz` and `optical_state` are long, keyed `(visit_id, variant_id)` with a registry table,
where the quantity set is a family of variants along several axes. A guider measurement has
at least a config axis (`MomentConfig`, already versioned to v6) and a per-detector axis, so
the long idiom with a registry is the closer match — see Q1.

### Scope

- Run the existing moments pipeline over all guider exposures in the survey, as sharded
  batch jobs, which Aaron submits.
- Discover the nights to process rather than listing them, and record which nights have
  guider data at all, so coverage is known rather than assumed.
- Add a guider moments table to the value-added database, with its registry if the long
  idiom is chosen, following the existing table and registry pattern in `efd_db.py` rather
  than inventing a new one.
- Store both moment kinds: the image second moments of the coadd, and the second moments of
  the centroid motion, keeping the noise-floor subtraction and the fast/slow split, with
  every column in arcsec² and registered in `column_coverage` with its units.
- Decide and document the per-detector aggregation: whether the database holds one row per
  (visit, detector) or a per-visit summary over the guider detectors.
- Write the builder to the `build_*.py` plus `run_build.sh --what` pattern the other
  value-added builders use, so sharding, registration and merging work the same way.
- Verify a night in the database against its `guider_moments_<dayObs>.parquet` before
  declaring the build good.
- Update `value_added/docs/schema.md` and `status/build_progress.md` with the new table, its
  coverage and its sparse columns.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Wide or long?** Wide means one fixed set of guider moment columns per exposure, like
`visit_telemetry` — simplest to query, but a changed `MomentConfig` has nowhere to go except
new columns or an overwrite. Long means `(visit_id, guider_variant_id)` with a registry
naming the config version and the measurement choices, like `fam_dz` — a reprocessing becomes
rows rather than schema, at the cost of requiring a variant filter on every read.

**A:** _unanswered_

**Q2. One row per visit, or per visit and detector?** There are a handful of guider detectors
per exposure, each with its own moments and its own centroid motion. Per-(visit, detector)
keeps everything and lets a later analysis aggregate; per-visit is smaller and joins directly
to the other tables, but discards the per-detector spatial information that makes a guider
measurement useful for field-dependent PSF work.

**A:** _unanswered_

**Q3. Which subset of the per-stamp data, if any, goes in?** `DetectorMoments` keeps the full
per-stamp moments and centroids for time-series and PSD analysis. At 5 Hz over a full
exposure that is a large array per detector. Options: summary statistics only in the
database with the per-stamp series left in the night parquets; a stored power-spectral
summary such as a few band-integrated powers; or the full series as list columns.

**A:** _unanswered_

**Q4. What span does the batch run cover?** The whole guider era, or the span that matches
`optical_state` (`day_obs` 20251102 onward) so the guider moments join to a recovered optical
state on every row. Guider data likely predates that, and the earlier nights would have
moments but no optical state to join to.

**A:** _unanswered_

</details>

---

## 7. Giant donuts: pupil models, spiders, and the intra/extra Z11 split

**Status:** not started, scope settled 2026-10-04 · **Blocked on:** nothing. All 7 open
questions are answered. Scoped to one night, on-axis, unpaired; the thermal arm is a later
pass.

Fit the giant (8 mm defocus) donuts on both sides of focus with the `donut_blitz_v2` tag of
ts_wep and its blitz pipeline, on Danish 1.3. Compare the several available donut pupil
models and rank them on which delivers the closest agreement in spherical aberration
(Noll Z11) between the intra-focal and extra-focal fits. Separately, test whether the
spiders can be included in the fit with good fidelity. The intra/extra Z11 split is being
used as the **diagnostic of residual optical path difference (OPD)**, not merely as a
pupil-model scorecard — candidate causes include thermal effects and mirror figure roll-off
at the pupil edge, which would move flux differently on the two sides of focus.

**Goals:** Pick the pupil model that best describes real giant-donut data, learn whether
modelling the spider shadows improves the fit, and get at the sources of the intra/extra Z11
disagreement that are not diffraction. Supersedes the giant-donut and pupil-mask work in
`wfs/`, which was a geometry comparison rather than a fit to data — see below for what it
settled and what it did not.

<details>
<summary>What `wfs/` already settled, the pupil models, scope and open questions</summary>

### What this supersedes, and what of it still stands

This item supersedes the giant-donut and pupil-mask line in [wfs/](../../wfs/) — the
notebooks `wfs_giant_donut_fit.ipynb`, `wfs_batoid_pupil_compare.ipynb` and
`wfs_diffraction.ipynb`, and the writeup
[wfs/docs/danish_pupil_mask_findings.md](../../wfs/docs/danish_pupil_mask_findings.md).
Those were a **geometry comparison against batoid and a synthetic-donut study**, with no fit
to real giant-donut data through a production pipeline. **Do not re-derive the following —
they are settled and are inputs here:**

| settled in `wfs/` | the result |
| --- | --- |
| danish's circle mask against the true batoid boundary, matched model and configuration | 99.50% agreement; inner edge exact; the residual +3 mm outer term is a finite-ray-grid artifact converging to about +1.0 mm at `nrad = 1600`, not a mask error |
| the corner WFS case | already correct — the defocus is a detector piston, so the camera apertures do not move and the intra and extra masks are identical by construction |
| the giant case, 8 mm camera-hexapod defocus | this is where the fixed mask fails. Agreement 95.97% intra and 96.79% extra, with the intra outer edge over-extended by a mean +55 mm and up to +261 mm, dominated by the **filter** |
| why | the filter, L1 and L2 ride with the camera hexapod, so off-axis they clip the pupil differently intra against extra, while M1/M2/M3 do not move. A single defocus-independent mask sits between the two |
| a per-element ellipse edge model | tested and does **not** help; M1 and M2 already project as circles, and a closed ellipse over-clips the far camera-borne elements. Refitting the **circle** at the camera position recovers giant-intra from 96.0% to 99.5% |
| the design-against-as-built optical model | danish/ts_wep used the design `LSST_r` (about v3.3); against as-built the filter alone moves the outer edge about +16 mm |
| diffraction as a cause of the Z11 split | real but **partial**. On-axis Fraunhofer-FFT donuts fitted in the production config give about **+0.10 µm of wavefront** Z11 intra/extra split in separate (unpaired) fits, the same sign as data and about one third the magnitude, and it averages away to about +0.01 µm in the paired fit |
| chromaticity and static pupil/model mismatch as causes | ruled out — about 0.002 µm and about 0.003 µm of wavefront Z11 respectively, roughly 100x too small |

The observation that drives this item, from data: **intra and extra Z11 differ by about
0.3 µm of wavefront** when fitted separately, with rim residuals worse intra and struts
sharper extra.

**The question this item opens that `wfs/` did not close.** Diffraction accounts for roughly
a third of the 0.3 µm of wavefront. The rest is unexplained, and the `wfs/` work only ever
tested *static, perfect-optics* causes. Two candidate physical sources are now on the table
(Aaron, 2026-10-04) and neither has been looked at:

- **thermal effects** — a time-varying OPD, so a cause the static pupil comparison could not
  have seen;
- **mirror figure roll-off at the edges of the pupil** — a real OPD error concentrated
  exactly where the rim residuals appear, and exactly where intra and extra donuts weight
  the pupil differently.

Both would move flux differently on the two sides of focus, which is what the Z11 split
measures. Giant donuts are the probe because 8 mm of defocus spreads the pupil over far more
pixels than FAM's 1.5 mm, so a pupil-edge OPD term is resolved rather than buried in the rim.

Two further `wfs/` leads not yet followed: the pure **Fresnel near-field** term beyond the
Fraunhofer approximation, and that the diffraction test was run **on-axis** while the real
split is measured at off-axis WFS field positions where spherical aberration is larger.

### The pupil models available

danish 1.3.0 is in the current stack
(`w_2026_39`, `lsst-scipipe-13.1.0-exact`) and ships three pupil files in
`danish/data/`:

| file | what it is |
| --- | --- |
| `RubinObsc_v3.14_r_rtpp0_azp45_pp0d0.yaml` | the Batoid as-built model |
| `RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml` | as-built plus updated M1M3 measurements and the M1 outer and inner baffles |
| `RubinObsc.yaml` | **byte-identical to the v1000 file** (verified by checksum 2026-10-04), so danish 1.3's default *is* v1000 |

These are the same models as the suffixes on Josh's T614 collections, so this item and
[item 5](#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint) are
testing the same pupil question from two directions — item 5 through the MIW, this item
through single-donut fits. The checksum result also answers item 5's Q7 on danish's side:
danish's default is v1000, **not** legacy, so "no suffix = legacy" cannot be assumed from
danish's default alone.

**How to read a processed run's pupil model back** (found while doing item 5's step A, and
useful here): the blitz task records it as `config.wavefrontFit.maskModel` in its
`*_config` dataset — `'policy:masks/LsstCamLegacy.yaml'` for legacy against
`'RubinObsc_v1000_r_rtpp0_azp45_pp0d0.yaml'` for v1000. Read the file off disk at
`<run>/donutBlitzFam*_config/*.py` rather than through `butler.get`, which raises
`ModuleNotFoundError: lsst.ts.wep.blitz` in `w_2026_39`. Older runs predate the field
entirely: `u/jmeyers3/t614_fam_unpaired` has no `maskModel` line and nests its config
under `wfFittingTask` rather than `wavefrontFit`, so absence of the key means an older
task, not a default.

What v1000 changes, in full, is in
[item 4](#4-pupil-measure-the-donut-pupil-geometry-data-against-model): M1 inner radius
2.558 m to 2.5833 m, M3 outer 2.508 m to 2.48511 m, M3 inner 0.55 m to 0.52735 m, plus
`M1Baffle1` and `M1Baffle2` as `ClearCircle` surfaces at radius 4.165 m — 15 mm inside M1's
unchanged 4.18 m rim — and a `CameraBody` obscuration. `pupilSize` is 8.33 m in v1000
against 8.36 m in v3.14.

### Spiders are already modelled — the fit just does not switch them on

The pupil YAMLs carry a `Spider_3D` block of **12 vanes**, each with a 3D position `r0`,
direction `v0`, `width = 0.05 m`, `length` and `angle`, which `danish/factory.py` projects
onto the entrance pupil per field angle (`_project_spider_vane`, about line 524). This is
the real double-bladed LSST spider, not the single-blade radial approximation that
`wfs_diffraction.ipynb` had to fall back on.

`DonutFactory` and the model classes take `spider_angle`, documented as "additional rotation
for spider struts around the optic axis in degrees. **If None, spider shadows are not
modelled**". So including the spiders is **switching on existing capability and supplying
the right rotator angle**, not writing new pupil code. The angle is the camera rotator
angle, which in this repository comes from the ConsDB `physical_rotator_angle` and not
`boresightRotAngle`. The `rtpp0_azp45` in the filenames says the shipped files were built at
rotator-telescope-position 0 deg and azimuth 45 deg.

That the data shows **struts sharper extra-focally** is the observation to explain, and it is
the reason the spider arm is in this item rather than a separate one: a spider that is
modelled at the wrong effective width or angle would itself bias the intra/extra comparison.

### Existing machinery to build on

| piece | path |
| --- | --- |
| the interactive single giant-donut fitter — click a donut, single-sided Z4-only Danish fit, data/model/residual, row-slice and pie-slice grids, sky subtraction | `wfs/notebooks/wfs_giant_donut_fit.ipynb` |
| the batoid/danish pupil comparison and the synthetic-donut fit tooling | `wfs/notebooks/wfs_batoid_pupil_compare.ipynb`, sections 10, 13 |
| the diffraction donut generator and the production-config fit | `wfs/notebooks/wfs_diffraction.ipynb` |
| the ts_wep corner dataflow, and the selected/fit/used stage definitions | [wfs/docs/ts_wep_cwfs_dataflow.md](../../wfs/docs/ts_wep_cwfs_dataflow.md) |
| per-pixel focal-plane radius, for the radial sky model | `common/camera_utils.py`, `pixel_to_focal` |

Two practical findings from `wfs_giant_donut_fit.ipynb` that will bite again:

- The giant donuts in the existing data have **no donut tables and no pipeline fits** — that
  notebook runs minimal in-notebook ISR on the raw. Whether `donut_blitz_v2` changes this is
  Q1.
- The fit **must** pass bounds on the blur. Left unbounded the fitted full width at half
  maximum (FWHM) runs to about 3 arcsec, over-blurring to hide pupil mismatch, and a runaway
  FWHM blows up the galsim fast Fourier transform so that unbinned giant corner donuts fail
  outright. The notebook pins FWHM at 1 arcsec by default.
- The labelled 8 mm defocus is not the effective one: the donut in the default exposure sits
  at about **7.6 mm**, tuned by driving the fitted Z4 toward zero.

### Scope

Scoped down to **one night, on-axis, unpaired** by the answers of 2026-10-04. The
thermal-correlation arm needs many nights and so is explicitly a later pass, not part of
this first one.

- Establish what `donut_blitz_v2` produces for giant donuts: whether the blitz pipeline
  detects and fits them at 8 mm defocus at all, and what dataset types come out (Q1). This
  is the first thing to check, because it decides whether the rest is a pipeline run or a
  notebook study.
- Select the night by exploring ConsDB and the value-added DuckDB (Q2): filter on the
  program/block for giant donuts, require **both sides of focus present**, and check that
  crowding is not too bad and the seeing is decent. Report the per-candidate numbers the
  choice was made on, not just the chosen night.
- Read the camera and M2 hexapod dz Trim for the chosen exposures and record **how the 8 mm
  was apportioned** between the two (Q7). This is needed to model the pupil at all, and it
  doubles as the check that the selected exposures really are giant donuts.
- Prefer donuts **as close to on-axis as possible** to start (Q6), rather than the off-axis
  corner positions `wfs/` used.
- Fit every selected donut with each pupil model — v3.14 and v1000 at minimum — holding the
  pipeline tag, Danish version, binning and blur treatment fixed, so the pupil model is the
  only thing varying.
- Fit **unpaired** (single-sided), which is the only configuration in which the metric is
  visible at all (Q3), and run the paired fit alongside purely as the cross-check that the
  averaging to about +0.01 µm of wavefront still happens on real giant donuts.
- Score each pupil model on **intra minus extra Z11 in µm of wavefront**, as the primary
  metric, reported as a distribution over donuts and field positions rather than a single
  number. Report the other Zernike terms alongside, since a model that fixes Z11 by moving
  coma is not a better model.
- **Let the blur float** rather than pinning it (Q5), and report the fitted FWHM in arcsec
  per model and per side of focus. The `wfs/` diffraction work found danish absorbing a
  softened rim into blur plus spherical, so blur is part of the result and not a nuisance
  parameter. Keep a bound on it even when unpinned — see the warning above about the galsim
  FFT — and report the bound used alongside the fitted value, since a value sitting at its
  bound is not a measurement.
- Fit with spiders off and on — `spider_angle` unset against set from the ConsDB
  `physical_rotator_angle` — and report whether the residual at the strut shadows improves,
  whether the intra/extra Z11 split changes, and whether the fit stays stable.
- Check the spider fidelity directly against the data rather than only through Z11: the
  residual along the strut shadows, and whether the modelled width reproduces the observed
  one, given the shipped `width = 0.05 m` and the files being built at a fixed rotator and
  azimuth.
- Separate the pupil-geometry effect from an OPD effect. A pupil-model error is a **mask
  boundary** error and should show at the rim and scale with the camera-borne element
  clipping; a thermal or figure roll-off term is an **OPD** error and should show as a
  smooth phase term weighted toward the pupil edge. Report which of the two the residual
  looks like, rather than reporting Z11 alone.
- Test the mirror figure roll-off hypothesis: whether the intra/extra Z11 split correlates
  with the pupil-edge radius, and whether an edge-weighted OPD term absorbs it.
- **Later pass, not this one:** test the thermal hypothesis — whether the split correlates
  with the thermal telemetry already in the value-added database, the mean Telescope Mount
  Assembly truss temperature and the M1M3 bulk and quadratic radial gradients, read through
  `value_added/code/efd_db.py`. A single night cannot support this: within one night the
  truss temperature barely moves, and the whole point is a spread across nights. Q2 chose one
  night deliberately, so this arm waits. Record the thermal telemetry for the chosen night
  anyway, so the first night is already a usable point when the sample grows.
- Account for the known partial cause: state how much of the measured split the roughly
  0.10 µm of wavefront diffraction term explains at these field positions, so the remainder
  is what the thermal and figure arms are being asked to explain.
- Move [wfs/docs/danish_pupil_mask_findings.md](../../wfs/docs/danish_pupil_mask_findings.md)
  into the new study and keep updating it as more is learned, and move the three superseded
  notebooks into an archive area (Q4). Both are file moves, so confirm with Aaron before
  doing them; nothing gets deleted.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Does the `donut_blitz_v2` blitz pipeline detect and fit 8 mm giant donuts, or does this
item still have to cut its own stamps?** The existing giant-donut data carries no donut
tables and no fits, which is why `wfs_giant_donut_fit.ipynb` runs its own minimal ISR and
single-sided fit. Blitz is a monolithic fitter with its own detection, so it may handle them
— or its detection may be tuned for FAM and corner donut sizes and miss an 8 mm donut
entirely. This decides whether the item is a pipeline run or a notebook study, and it is the
first thing to check.

**A:** _I think so but this needs to be checked_

**Checked 2026-10-05 by reading the source — the answer is mostly yes, and the tag name in
this item is wrong.** The real tags are **`blitz-prototype-v1` and `blitz-prototype-v2`**;
there is no tag named `donut_blitz_v2` and no v3. v2 (`9651cd23`, 2026-09-11) is the newest
and is what this study uses.

- Detection is **size-agnostic**: `blindDetectTask` builds an annular template from
  `donutRadius` and scales `min_distance` / `exclude_border` by it, with an override
  argument, so an 8 mm donut is a parameter change rather than a code change.
- `cameraOffset` and `m2Offset` are separate config fields, in meters and signed per
  exposure, so the measured 4+4 mm split (Q7) is directly modellable.
- `modelSpiderShadows` is already a config field, defaulting False. It gates `rtp`, which
  gates danish's `spider_angle`, so the spider arm is a toggle as hoped.
- The fitted blur is bounded at `fwhm = [0.1, 5.0]` arcsec, **hardcoded in
  `wavefrontFittingTask` rather than exposed as config** — not the 0.5 to 1.5 arcsec of the
  production config assumed in Q5. The 5.0 arcsec ceiling is the runaway regime `wfs/`
  warned about, so bounding it means patching the task. `binning` defaults to 2.

Still to do: an actual end-to-end run on `seq_num` 337 and 340 to confirm the pipeline
detects and fits them. **But see the blocker below** — blitz builds its pupil from
ts_wep's `maskParams`, not from danish's pupil YAMLs, so the v3.14/v1000 comparison is not a
file swap.

**Blocker found 2026-10-05: ts_wep's blitz never reads danish's pupil YAMLs.** The donut
factory is built with `mask_params=_INSTRUMENT.maskParams`, where `_INSTRUMENT` is a
module-level singleton loaded from `policy/instruments/LsstCam.yaml`. That file carries
`diameter: 8.36 m` and M1 inner `2.558 m` — **v3.14 numbers** — and there is no v1000
variant of ts_wep's `maskParams`. So the pupil-model comparison requires generating v1000
`maskParams` polynomials (cubic-in-θ centre and radius per element edge) from the v1000
batoid model, which is the one place this item needs genuinely new code. The archived
`wfs_batoid_pupil_compare.ipynb` has the batoid-boundary tooling to build them from. The
alternative is to fit with danish directly, where the YAMLs *are* selectable, at the cost of
no longer being a blitz result. This also bears on
[item 5](#5-rebuild-the-miw-on-the-v1000-pupil-model-then-under-the-rbr-constraint) if it
goes through ts_wep. Note that danish 1.3's default being v1000 is still true but is
irrelevant inside blitz.

**Q2. How many giant donuts, over how many nights?** A pupil-model ranking needs enough
donuts to separate the models but could run on a few exposures. The thermal-correlation arm
needs a spread of truss temperature, so many nights. These may be two different samples
rather than one.

**A:** _lets start with just a single night with decent intra and extra focal giant donuts.  We need to do some exploration of the Consdb and DuckDb to select according to the program for blocks with Giant donuts and then make sure both sides of focus are present and also the crowding isn't too bad and that the seeing is decent. Bryce used 20250520_

**Resolved 2026-10-05: the night is `day_obs` 20251023, not 20250520.** Aaron corrected the
answer above — Bryce used 20251023, `seq_num` 337 and 340. The Trim settles it
label-independently (Q7): 20250520 has the camera dz Trim spanning 12000 µm with 15 DOF
moving, including bending modes 30–34, which is an active-optics night with no ±4000 µm
plateau and no giant donuts. The chosen sample is BLOCK-T626, `r` band, 60 s, **extra 337,
338 and intra 340, 341**.

Two exposures were rejected, and both show why selection must be Trim-first rather than
label-first:

- `seq_num` 339 sits at the giant intra Trim state but is 30 s with `science_program`
  "unknown", so it is not part of the T626 giant sequence.
- `seq_num` 351 and 352, labelled `intra_8mm_m1m3_b4`, show **no bending-mode motion in the
  Trim at all** — across `seq_num` 330–354 only `dof0` and `dof5` ever move. Either the b4
  mode was commanded to M1M3 as forces outside the OFC aggregated DOF, or it was never
  applied; the aggregated Trim cannot tell these apart, so they are excluded pending a check
  of the M1M3 applied forces.

The same night also carries ordinary FAM at camera-only ±1500 µm (`seq_num` 334/335 and
346/347), giving a built-in camera-only against 4+4 split contrast. Selection is implemented
in `wfs/code/giant_donuts/select_exposures.py`. The observation reason is free text and was
wrong in both directions here, so it is not a reliable filter on its own.

**Q3. Paired or unpaired fits?** The intra/extra Z11 split is only visible in **separate
(unpaired)** fits — the paired fit averages it to about +0.01 µm of wavefront, as `wfs/`
found. So the metric of record here requires unpaired fits, and the paired fit is the
cross-check that the averaging still happens on real giant donuts.

**A:** _unpaired_

**Q4. What happens to the superseded `wfs/` material?** The three notebooks and
`danish_pupil_mask_findings.md` hold results this item depends on and should not simply be
deleted. Options: leave them and add a status line pointing here; move the findings doc into
the new study and keep the notebooks as provenance; or keep `wfs/` as the geometry topic and
put only the data fits in the new study. Deleting any file needs Aaron's go-ahead regardless.

**A:** _move the findings doc into the new study to be updated as learn more and lets put the current notebooks into an archive area for the moment_

**Q5. Does the fitted blur stay bounded, or pinned?** `wfs_giant_donut_fit.ipynb` pins FWHM
at 1 arcsec by default because a floating blur runs to about 3 arcsec and hides pupil
mismatch. But blur is part of the result — the diffraction work found it absorbing the
softened rim. A pinned blur may force the mismatch into Z11, which is the metric; a floating
blur may absorb the very effect being measured. The production blitz config bounds it to
0.5 to 1.5 arcsec.

**A:** _We will want to unpin the blur_

Keep a **bound** on it even so. Unpinned is not unbounded: `wfs/` found a floating FWHM
running to about 3 arcsec, which both over-blurs to hide pupil mismatch and blows up the
galsim FFT so unbinned giant donuts fail outright. The production blitz config bounds it to
0.5 to 1.5 arcsec. Report the bound with the fitted value, since a fit sitting at its bound
is not a measurement. Note this interacts with Q1: if blitz does the fitting, the blur
treatment is whatever that pipeline's config does, so unpinning may mean overriding a
pipeline config rather than setting a notebook argument.

**Q6. Which field positions?** The `wfs/` diffraction test was on-axis while the measured
split is at off-axis WFS field positions, where spherical aberration is larger — a known gap
in the existing work. Giant donuts are FAM-style full-focal-plane images, so field position
is selectable over the whole focal plane rather than fixed at the four corners.

**A:** _start with donuts as close to on-axis as possible_

**Q7. Is the 8 mm defocus apportioned between the camera and M2 hexapods?** `wfs/` found that
camera-only 8 mm and a 4 mm + 4 mm camera-plus-M2 split give different pupils, donut span
7.0 mm against 6.7 mm, because M2 is powered — and that ts_wep could not apportion the offset
between the two. Which the data used has to be known to model it, and the effective defocus
is about 7.6 mm rather than the labelled 8 mm anyway.

**A:** _need to look at the Cam and M2 hexapod dz trim (in duckdb) to see what was used_

The columns are the Trim `dof0..49` in the value-added database, read through
`value_added/code/efd_db.py`. **`dof0` is M2 hexapod dz and `dof5` is camera hexapod dz**,
both in µm — that ordering is the opposite of what people assume, and it is documented in
`common/dof_telemetry.py`. So the apportionment is read off `dof0` against `dof5` directly.
This also serves as the label-independent test of which exposures are really giant donuts,
which is what Q2's 20250520 question needs.

**Answered 2026-10-05: a symmetric 4 + 4 mm split, not camera-only 8 mm.** Against the
bracketing in-focus exposure (`seq_num` 336, M2 dz Trim −1288.6 µm, camera dz Trim
−1168.9 µm), the giant exposures sit at **±4000 µm on the M2 hexapod and ±4000 µm on the
camera hexapod**, an 8000 µm throw on each. This is the configuration `wfs/` predicted gives
a different pupil — donut span 6.7 mm rather than 7.0 mm, because M2 is powered — so the
pupil model must carry both hexapod offsets. blitz v2 can: `cameraOffset` and `m2Offset` are
separate config fields in meters, which also retires the note that ts_wep could not
apportion the offset.

</details>
