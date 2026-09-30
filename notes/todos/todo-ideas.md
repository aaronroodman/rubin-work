# Rubin AOS — TODO / Ideas

> **Status:** current · **Last updated:** 2026-09-30 · **Kind:** working state (queue)

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
| [2](#2-assess-the-5034-correction-scheme-for-science-visits) | Assess the 50/34 correction scheme for science visits | not started |
| [3](#3-extend-the-miw-grid-so-the-interpolation-hull-covers-the-full-field-of-view) | Extend the MIW grid so the interpolation hull covers the full field of view | diagnosed, Guillem has the fix |
| [4](#4-confluence-page-documenting-the-aos-production-runs-in-repomain) | Confluence page documenting the AOS production runs in `/repo/main` | probe done, page not written |
| [5](#5-pupil-measure-the-donut-pupil-geometry-data-against-model) | `pupil` — measure the donut pupil geometry, data against model | not started |
| [6](#6-rebuild-the-miw-under-three-correction-schemes-and-compare) | Rebuild the MIW under three correction schemes and compare | not started |

Recently closed and moved out: the Danish 1.3 blitz Full Array Mode (FAM) processing, the
July bounce test and its note for Guillem, the `thermal_focus` study, and the
`visit_telemetry` backfill to 20250415. See [completed-todos.md](completed-todos.md).

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
item 4 inventory, not a count of science and acq visits on 20260512, 20260513 and 20260713.

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

## 2. Assess the 50/34 correction scheme for science visits

**Status:** not started · **Blocked on:** decide on approach and then implement the corresponding OLR 

Compare the current 22 degree-of-freedom / 12 v-mode (22/12) correction scheme against 50/34 for regular science images by using the Open Loop Reproduction (OLR). 

**Goals:** Determine how to implement a 50/34 scheme so that Mirror force limits are obeyed, and assess the IQ performance. 

<details>
<summary>Existing machinery, data selection, scope and open questions</summary>


### Existing machinery to build on

| piece | path |
| --- | --- |
| the OLR pipeline | [olr/code/run_olr.py](../../olr/code/run_olr.py) |
| its nightly table | [olr/code/nightly_table.py](../../olr/code/nightly_table.py) |
| parquet combine | [olr/code/combine_parquets.py](../../olr/code/combine_parquets.py) |
| topic Snakefile and config | `olr/Snakefile`, `olr/config.yaml` |
| the DuckDB machinery to copy | [value_added/code/](../../value_added/code/), read through `value_added/code/efd_db.py` |
| v-modes and DOF sets | `aos/code/aos_state.py`, imported by `olr/` |

`run_olr.py` writes `olr.parquet` per night, one row per usable seq, carrying the
open-loop-reproduced Optical Path Difference (OPD) and deviation Zernikes alongside the
original measured values and the intrinsic. `olr/` is its own top-level topic with its own
Snakefile and config.

### Data selection

- **After 20260419** — the Singular Value Decomposition (SVD) normalization fix.
- **Probably also after** the point where Danish 1.2 plus Refit WCS went online. Need to find what day_obs this occurred.

### Scope

- Write the OLR results for a large sample of science visits into a DuckDB table.
- Define the reduced-gain vector over the 34 kept v-modes, or implement and use the RBR.
- Run the 22/12 versus 50/34 comparison on the wavefront measurements in the ConsDB (or with reprocessed CWFS).

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. What does "lower gains for higher v-modes" mean concretely?** A gain vector over the
34 kept modes, a roll-off function of mode index, or a per-mode fit.

**A:** A vector of gains over the DoF.  Starting value might be a gain of 0.3 for the 22 DoF currently used and a lower value of 0.1 for the remaining mirror modes. 

**Q2. How large is a "large sample" of science visits,** and does the 20260419 cut leave
enough once the Danish 1.2 and refit WCS cut is also applied?

**A:** Yes, post 20260419 should be sufficient

**Q3. Does the OLR table live in the existing `aos_efd.duckdb` or its own database file?**

**A:** I am not sure, but this table will also need to be keyed off the nDof/nVmode scheme and perhaps also the wavefront retrieval

**Q4. Is 20260419 the right single cut?** It is the SVD normalization fix date and appears
to be the Danish 1.2 changeover date too, but refit WCS may have gone online on a
different day. Needs confirming against the online collection provenance.

**A:** The Danish 1.2 and Refit WCS was definitely later, but it may be ok to use the earlier processing

</details>

---

## 3. Extend the MIW grid so the interpolation hull covers the full field of view

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

## 4. Confluence page documenting the AOS production runs in `/repo/main`

**Status:** Butler probe done — page not written · **Blocked on:** how the page gets published

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
  the Danish 1.3 blitz live;
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

**A:** _unanswered_

**Q2. Does an existing Confluence page cover this and need updating** rather than a new one
created? "Up to date" in the request suggests there may be a stale one.

**A:** _unanswered_

**Q3. Is `/repo/embargo` in scope,** or is `/repo/main` the whole story for production?
Embargo may hold recent runs not yet in main.

**A:** _unanswered_

**Q4. Which product do the night counts refer to?** They come from the aggregate visit
tables here. If a run's `zernikes` or donut-level products have different coverage, the page
should say so — worth one spot-check before publishing.

**A:** _unanswered_

</details>

---

## 5. `pupil` — measure the donut pupil geometry, data against model

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

**A:** _unanswered_

**Q8. Does the inner ring behave differently from the outer?** The inner half-flux radius
scatter is 0.326 binned pixels against 0.208 for the outer, which may be intrinsic — fewer
pixels, lower contrast — or may be real M2-baffle structure.

**A:** _unanswered_

</details>

---

## 6. Rebuild the MIW under three correction schemes and compare

**Status:** not started · **Blocked on:** nothing

Rebuild the Measured Intrinsic Wavefront (MIW) under three degree-of-freedom (DOF) and
v-mode schemes — the regular 50 DOF / 34 v-mode (50/34) recovery, 50/34 with the
Range-Bounded Recovery (RBR) constraint applied, and the 22 DOF / 12 v-mode (22/12) scheme
alone — and compare the three builds against each other.

**Goals:** Determine how much the recovered optical state subtracted during the MIW build
depends on the scheme used to recover it, and whether the RBR constraint gives a physically
reachable state without degrading the MIW.

<details>
<summary>Existing machinery, scope and open questions</summary>

### Existing machinery to build on

| piece | path |
| --- | --- |
| the MIW build chain | `aos/Snakefile`, `rule build_intrinsic` through `rule refit_mi`, calling `run_build_intrinsic.py`, `run_intrinsic_split.py`, `run_dz_fit.py` in the installed `ts_intrinsic_wavefront` |
| the MIW build entries | `aos/mi_config.yaml`, `measured_intrinsics:` keyed by param set |
| two-build term-by-term comparison | [aos/code/miw/compare_miw_versions.py](../../aos/code/miw/compare_miw_versions.py) |
| how far the current build's states sit outside the allowed range | [aos/code/miw/check_dof_ranges.py](../../aos/code/miw/check_dof_ranges.py) |
| the RBR solver | [smatrix/code/regularized_inversion/regularized_inversion.py](../../smatrix/code/regularized_inversion/regularized_inversion.py), `dof_range_vector` and `invert_range_penalty` |
| the AOS-side RBR wrappers | `aos/code/bounce/bounce_lib.py`, `rbr_module`, `rbr_dof_per_pair`, `rbr_deltas` |
| residual wavefront to FWHM | [aos/code/aos_fwhm.py](../../aos/code/aos_fwhm.py), `residual_dW`, `zj_to_fwhm`, `fp_fwhm` |
| v-mode engine and DOF sets | `aos/code/aos_state.py`, `make_state_estimator`, `vmodes_from_dofs`, `recover_optical_state`, `DOF_SETS`, `N_MODES` |
| the equation-level residual derivation | `aos/docs/miw_coadd_equations.md`, section 4 |

The MIW library and its runners live in the installed `ts_intrinsic_wavefront` package;
`aos/` is a thin client, so the scheme enters as config rather than as a code change. The
(n_dof, n_keep) pair is passed to `build_ofc_svd(iZs, k_min, k_max, n_keep, n_dof=...)`
package-side. `aos_state.py` defines the DOF sets `hexapod_10`, `standard_22` and `all_50`
with default mode counts 10, 12 and 20, not the named scheme tuples.

Two existing MIW builds, both 50/34, sit at `aos/output/miw/danish_1_2_A_50_34_i/` and
`..._5rot/`, with the `_5rot` split being the canonical product
(`intrinsic_split_maps.parquet`). A `build_from` entry reuses its parent's per-rotator-bin
grids and re-runs only the split and downstream, so a scheme change is a fresh build rather
than a `build_from`.

RBR is implemented and in production use by the bounce study, with `kappa = 4` and
`power = 3` on those legs, and the penalty is smooth rather than a hard bound — a recovered
amplitude can finish outside its range, by up to a factor of 1.157 dimensionless (recovered
amplitude over allowed range) on the bounce legs. It has never been applied to the MIW
build: `check_dof_ranges.py` measures how far the existing build's recovered states fall
outside the allowed range `r_j` and changes nothing.

### Scope

- Add `mi_config.yaml` entries for the 50/34 with RBR and the 22/12 builds alongside the
  existing 50/34, and run all three over the same visits, rotator bins and filter.
- Apply the RBR penalty inside the MIW build's per-visit optical-state recovery, importing
  the solver from `smatrix/` rather than copying it.
- Compare the residual MIW between the three builds, term by term and as field maps.
- Convert each residual MIW to an inferred full width at half maximum (FWHM) in arcsec over
  the focal plane, and report the three.
- Report the size of the corrected v-modes in each build: all 34 in full 50/34, the 34 under
  the RBR constraint, and the 12 in 22/12.
- Report the recovered DOF values in each build, in µm and arcsec as appropriate per DOF,
  against the allowed range `r_j`.
- Quantify how the subtracted optical-state correction changes between the three builds, per
  DOF and per v-mode.
- Write the study up in `aos/docs/studies/miw.md`.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Which param set do the three builds run on?** The Danish 1.2 param set has a complete
50/34 build to compare against, while `danish_1_3_test` has entries whose build has not been
run. Using Danish 1.2 gives an existing baseline; using Danish 1.3 folds in the newer
retrieval but means building all three from scratch.

**A:** _unanswered_

**Q2. What `kappa` and `power` does the MIW RBR use?** The bounce study uses `kappa = 4` and
`power = 3` dimensionless. Whether those transfer to the MIW recovery, where the wavefront
being inverted is a per-visit deviation rather than a bounce-leg difference, needs checking
rather than assuming.

**A:** _unanswered_

**Q3. Is 22/12 built with `standard_22` at 12 modes, or at its `N_MODES` default?**
`aos_state.DOF_SETS["standard_22"]` is the 22-index set and `N_MODES` gives it 12, but the
mode count is set independently by `n_modes` and the scalar 22 silently picks DOF 0 to 21,
which is the wrong set.

**A:** _unanswered_

**Q4. Does RBR belong in the MIW build, or only as a diagnostic on top of it?** The bounce
note records that an RBR amplitude is a constrained estimate rather than a measurement, and
that the default recovery remains the estimator. If that holds for the MIW, the RBR build is
a comparison arm rather than a candidate replacement.

**A:** _unanswered_

**Q5. Which residual is compared?** The subspace-projection residual from
`aos_fwhm.residual_dW` cannot see a regularizer that trades wavefront against amplitude, so
it will not show what RBR costs. The bounce study uses `achieved_residual` against the
applied DOF instead.

**A:** _unanswered_

**Q6. Does the k-truncation of the MIW basis interact with the scheme change?** The build
runs `k_min = 1` to `k_max = 6`, and the existing leakage analysis
(`aos/calibration/miw/umode_k_leakage_50_34.npy`) is specific to 50/34.

**A:** _unanswered_

</details>
