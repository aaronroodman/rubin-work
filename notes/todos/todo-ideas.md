# Rubin AOS — TODO / Ideas

> **Status:** current · **Last updated:** 2026-10-02 · **Kind:** working state (queue)

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
| [2](#2-open-loop-and-deviation-recovered-optical-state-for-science-visits-three-schemes) | Open-loop and deviation-recovered optical state for science visits, three schemes | not started, scope settled and unblocked |
| [3](#3-extend-the-miw-grid-so-the-interpolation-hull-covers-the-full-field-of-view) | Extend the MIW grid so the interpolation hull covers the full field of view | diagnosed, Guillem has the fix |
| [4](#4-confluence-page-documenting-the-aos-production-runs-in-repomain) | Confluence page documenting the AOS production runs in `/repo/main` | probe done, page not written |
| [5](#5-pupil-measure-the-donut-pupil-geometry-data-against-model) | `pupil` — measure the donut pupil geometry, data against model | not started |
| [6](#6-rebuild-the-miw-under-three-correction-schemes-and-compare) | Rebuild the MIW under three correction schemes and compare | not started |
| [7](#7-reorganize-the-thermal_focus-analysis-and-its-pdf-report) | Reorganize the `thermal_focus` analysis and its PDF report | not started |
| [8](#8-guider-star-second-moments-image-and-centroid-motion-into-the-value-added-db) | Guider star second moments, image and centroid motion, into the value-added DB | not started |

Recently closed and moved out: the Danish 1.3 blitz Full Array Mode (FAM) processing, the
July bounce test and its note for Guillem, the `thermal_focus` study, the `visit_telemetry`
backfill to 20250415, the shared regularized-inversion solvers, and the four-scheme bounce
test. See [completed-todos.md](completed-todos.md).

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

## 2. Open-loop and deviation-recovered optical state for science visits, three schemes

**Status:** not started, scope settled 2026-10-02 · **Blocked on:** nothing. All 13 open
questions are answered, so the whole item — both arms, the RBR arm included — is ready to
build.

For all science and acquisition visits, compute and store two optical states per correction
scheme: the **open-loop** one, which is the Trim minus Deviation degree-of-freedom (DOF)
state with corresponding v-modes and Double Zernikes (DZ), where the Deviation is from the
corner wavefront sensor (CWFS) Zernikes — that is the Open Loop Reproduction (OLR) — and the
**deviation-recovered** one, the DOF and v-modes recovered from that visit's measured
deviation alone. Do both for the 22 DOF / 12 v-mode (22/12) scheme, the 50/34 scheme, and
50/34 with the Range-Bounded Recovery (RBR) constraint, from the Consolidated Database
(ConsDB) Zernike values, and land the results in the value-added database.

Collect the Trim-minus-Deviation code in one shared place first — pulled out of `olr/`, which
is then repurposed for analysis of the OLR as stored in the database.

**Goals:** Assess the 22/12 versus 50/34 correction schemes on real science visits, with a
50/34 implementation that obeys the mirror force limits, and make the per-visit open-loop and
deviation-recovered DOF and v-modes available to every later analysis as a join rather than a
recomputation.

<details>
<summary>What exists, what the OLR is, what is populated, scope and open questions</summary>

### Known collections

The measured CWFS Zernikes come from ConsDB as they are measured online in the AOS:
`aos_state.fetch_corner_zernikes_consdb` against `consdb_ccdvisit1_quicklook`, which is the
`opd_source` recorded on every variant. The Trim DOF come from the database's own
`visit_telemetry.trim` columns, 50 of them, from `MTAOS.logevent_degreeOfFreedom` as of
`obs_start` [µm, arcsec].

### Data selection

**No cut: fill every science and acquisition visit** (Q2, Q4, Q7). The subsample for a given
study is chosen later, at read time, which is what the variant-plus-join layout is for. So
the dates below are provenance to record, not filters to apply:

- **20260419** — the Singular Value Decomposition (SVD) normalization fix. Visits before it
  are still built; an analysis sensitive to it cuts on `day_obs` itself.
- Danish 1.2 plus Refit WCS went online later. That day_obs still needs finding, but it
  gates interpretation, not the build.

This extends the build back to where the ConsDB Zernikes start, 20250415 — about 2.4x the
span of the one populated variant today (see below), and it means the populated variant is
rerun complete over the wider span, not extended.

### The OLR is Trim minus Deviation

This is what collapses the two halves of this item into one build. The open-loop state is
just the Trim DOF; its v-modes are the forward projection `aos_state.vmodes_from_dofs(trim,
state_estimator, n_modes=...)`, which is `StateEstimator.get_vmodes_from_dofs` — the basis
the Main Telescope AOS reports on the summit. Its CWFS Zernikes are that state pushed
through the sensitivity matrix, which is the OLR. So OLR and deviation recovery are the
forward and inverse directions of one operator, computed per scheme in the same pass, and
the comparison between them is the thing worth looking at.

One sign convention to carry over exactly. In `olr/code/olr.py` the OLR **adds the applied
correction back** to the measured wavefront, recovering what would have been seen with the
loop open — `apply_trim(..., subtract=False)`, i.e. `olr_opd[c] = zk_opd[c] + z_change[c]`
with `z_change = sens_mat @ trim` reshaped to the four corners and Z20/Z21 zero-padded.
"Trim minus Deviation" is the DOF-space statement of the same operation; the Zernike-space
code adds. Only 22 of the 50 DOF enter `sens_mat` there, which matters for the 50/34 schemes
(see Q11).

`run_olr.py` already asserts the identity `olr_deviation == olr_opd - intrinsic` on every
corner row and prints the check into its log, so the forward direction has a verified
reference implementation to move and reuse.

**What is actually in `olr/code/` is narrower than "the OLR calculation".** It is Zernike
space only: `build_olr_sensitivity_matrix`, `apply_trim` and the corner stacking, plus the
pipeline around them. There is no DOF recovery, no v-mode projection and **no DZ code at
all** there. So the move is `build_olr_sensitivity_matrix` + `apply_trim` generalized over
the DOF set, and nothing more; the v-mode half already lives where it belongs (next
paragraph), and the DZ optical state Q10 asks for has to be **written**, not moved. Two
defects to fix in the same move: `build_olr_sensitivity_matrix` constructs a bare
`OFCData(name='lsst')` with **no normalization assertion**, which is exactly the obsolete-
normalization path `make_state_estimator` raises on and which *rotates* the v-mode basis
rather than rescaling it; and its field angles are named `field_angles_ccs` while
`aos_state` requires OCS at rotator zero, so the frame has to be settled explicitly rather
than inherited from the variable name.

**`build_optical_state.py` already stores the open-loop v-modes.** `make_commanded_projector`
projects both the hexapod lookup-table DOF and the Trim DOF through `vmodes_from_dofs`, and
`upsert_optical_state` writes them as `v_modes_lut` and `v_modes_trim` alongside the
deviation-recovered `v_modes`. The Trim DOF themselves are already in `visit_telemetry`
rather than in `optical_state`.

The CWFS Zernikes are **not stored** (Q6). They are already in ConsDB as OPD Zernikes for
all four corners, present for every science and acq visit whether the loop was open or
closed, so the build queries them live and the database holds no copy. What ConsDB does not
have is the **intrinsic** wavefront, which is why the intrinsic route — batoid or MIW — is a
variant axis: the intrinsic is what turns an OPD into the deviation that defines the optical
state. A Butler processing will eventually replace the ConsDB values, but not yet.

### Existing machinery to build on

| piece | path |
| --- | --- |
| the optical-state builder | [value_added/code/build_optical_state.py](../../value_added/code/build_optical_state.py) |
| its batch wrapper, sharded by night | [value_added/code/run_build.sh](../../value_added/code/run_build.sh) |
| the table, registry and readers | [value_added/code/efd_db.py](../../value_added/code/efd_db.py) |
| the schema reference | [value_added/docs/schema.md](../../value_added/docs/schema.md) |
| what is built and what is sparse | [value_added/docs/status/build_progress.md](../../value_added/docs/status/build_progress.md) |
| the OLR pipeline | [olr/code/run_olr.py](../../olr/code/run_olr.py) |
| its nightly table and parquet combine | [olr/code/nightly_table.py](../../olr/code/nightly_table.py), [olr/code/combine_parquets.py](../../olr/code/combine_parquets.py) |
| topic Snakefile and config | `olr/Snakefile`, `olr/config.yaml` |
| v-modes, DOF sets, per-corner recovery | `aos/code/aos_state.py`, imported by both `olr/` and `value_added/` |
| the solvers, shared code since [completed item 5](completed-todos.md#5-consolidate-the-regularized-inversions-as-shared-ofc-code-in-smatrixcode) | `smatrix/code/regularized_inversion.py` — the **module**, not the `smatrix/code/regularized_inversion/` directory of compare drivers next to it |

Most of this exists. `build_optical_state.py` takes `--scheme` (`22_12` or `50_34` in its
`SCHEMES` dict, mapping to the `ts_ofc` DOF-set names `standard_22` and `all_50`),
`--intrinsic`, `--opd-version` and `--img-type science,acq`; `run_build.sh --what state`
shards it by night, resolves each variant's defining flags out of the main database so every
shard registers the identical variant, and merges the shards. The 22/12 deviation build is
therefore a run, not new code.

`recover_night` calls `aos_state.recover_optical_state(row, state_estimator,
n_modes=n_modes)` per visit — the plain truncated recovery — and stores `dof` (50 elements,
µm and deg), `v_modes` (`n_modes` dimensionless amplitudes), `resid_rms_um` [µm of wavefront]
and `ok`.

**The RBR arm is the part that does not exist.** Three consequences:

- RBR rides in the `scheme` field as the pseudo-scheme `50_34_rbr`, giving the variant
  `v50_34_rbr__batoid__consdb_v1` (Q5). No schema change and no fourth axis, but it does
  mean `build_optical_state.SCHEMES` needs a third entry mapping `50_34_rbr` to the same
  `('all_50', 50, 34)` DOF set as `50_34`, so `scheme` no longer determines `n_dof` and
  `n_modes` uniquely — two schemes now share them and differ only by solver. Record the
  penalty and its parameters in `state_variant.notes`, since no column describes them.
  Adding the entry is enough to make `--scheme 50_34_rbr` selectable, because the argparse
  choices are `sorted(SCHEMES)`; `build_state_estimator` will then happily build the
  `all_50` estimator and `recover_night` will run the **truncated** solver under the RBR
  variant name. Guard explicitly: the builder must refuse `50_34_rbr` unless the RBR solver
  path is wired, or that variant silently becomes a duplicate of `50_34`.
- **The two solvers read different measurement spaces, so the RBR call cannot be made at
  all today.** `recover_optical_state` takes 84 corner values (4 corners x 21 Noll, µm of
  wavefront) and inverts the corner-evaluated, Zernike-selected SVD from
  `corner_recovery_basis`. `invert_range_penalty(dW, svd, ranges)` takes `dW` over
  `svd.kj_grid` — DZ coefficients over the full field, µm of wavefront — against an
  `OFCSvd` from `build_ofc_svd`. A science visit supplies the former. This is not a
  "how closely do the operators agree" tolerance to measure; there is no number to report
  until an adapter exists. Q13 settles the adapter: shim `corner_recovery_basis` into the
  solver's interface, which is the standard way the optical state is found from the CWFS.
- `resid_rms_um` as stored is `z_dev - zk_constrained`, the subspace residual. For the RBR
  variant that is the wrong metric, for the reason settled in item 6 and in the completed
  four-scheme bounce test ([completed item 6](completed-todos.md#6-extend-the-bounce-test-to-four-recovery-schemes)): it cannot see a
  regularizer trading wavefront for amplitude. The achieved residual `dW - S (d / w)` is what
  the RBR row should carry.

### What is populated today

| variant_id | rows | state |
| --- | --- | --- |
| `v50_34__batoid__consdb_v1` | 90,695 | built, `day_obs` 20251102 to 20260713, 181 nights |
| `v22_12__batoid__consdb_v1` | 0 | registered, never built |
| `v50_34__miw__consdb_v1` | 0 | registered, never built |

Row counts are from `build_progress.md`, read on 2026-09-24. `visit_telemetry` covers 366
nights and 213,704 exposures over `day_obs` 20250415 to 20260714, so the recovered optical
state covers a visibly narrower span than the telemetry it joins to. Closing that gap is now
in scope: the no-cut answer means every variant should reach the full telemetry span, which
is roughly 2.4x the nights and a complete rerun of the one populated variant.

The empty-but-registered variants are a live trap worth not reproducing:
`efd_db.optical_state('v50_34__miw__consdb_v1')` returns an empty DataFrame rather than
raising, so an analysis naming an unbuilt variant gets zero rows and no error.

### One caveat on comparing the schemes through v-modes

`recover_optical_state` is hybrid: it inverts in `corner_recovery_basis` and reports
v-modes in the `make_state_estimator` basis. Those are different bases, and the measured
principal angle between the retained DOF subspaces is 4.768 deg for `standard_22`/12 but
**89.951 deg for `all_50`/34** — effectively orthogonal. So the stored `v_modes` stand in a
radically different relation to the recovered DOF under 50/34 than under 22/12, and a
22/12-versus-50/34 comparison read off `v_modes` alone is not comparing like with like.
That the two schemes span different v-mode subspaces is expected, not a problem. It is the
reason the comparison is made elsewhere: **on recovered image quality first and on the DOF
values second** (decided 2026-10-02), with v-modes reported for continuity with what the
summit reports rather than as the metric of record. The angles above are quoted from the
`aos_state` docstring; they describe the situation and gate nothing.

### Scope

- **First, consolidate the Trim-minus-Deviation code in one place** (Q10), which is a
  smaller and differently shaped job than it first looks (see "What is actually in
  `olr/code/`" above). Concretely: move `build_olr_sensitivity_matrix` and `apply_trim`
  from `olr/code/olr.py` into `aos/code/` — where the v-modes, DOF sets and per-corner
  recovery already live — taking the state estimator as an argument so the DOF set decides
  the column count (Q11), asserting the required normalization, and naming the Zernike
  frame as OCS. The v-mode and DOF half needs no move: it is already in
  `build_optical_state.py` and `aos_state.py`. The DZ optical state is new code. Then
  delete the superseded functions from `olr/code/olr.py` so there is one implementation,
  not two, and repurpose `olr/` for analysis of the OLR as stored in the database.
  Deleting anything needs Aaron's go-ahead at the time.
- For each of the three schemes (22/12, 50/34, 50/34 plus RBR), compute and store both
  states per visit: the open-loop DOF and v-modes, and the DOF and v-modes recovered from
  the visit's deviation alone. The CWFS Zernikes stay in ConsDB and are queried live, not
  copied (Q6).
- Cover **all science and acq visits** (Q2, Q4, Q7) — no date cut. This extends back to
  20250415 and therefore includes rerunning the already-populated
  `v50_34__batoid__consdb_v1` over the wider span, not just building the empty variants.
- Build `v22_12__batoid__consdb_v1`, which is a run of the existing builder rather than new
  code.
- Add the RBR variant as the pseudo-scheme `50_34_rbr` (Q5), with `kappa = 4` and
  `power = 3`, both dimensionless, matching the bounce test (Q8). Call the shared solver in
  `smatrix/code/regularized_inversion.py` rather than copying it.
- **Batoid intrinsic only** (Q12). Three variants, all on the batoid route:
  `v22_12__batoid__consdb_v1`, `v50_34__batoid__consdb_v1` and
  `v50_34_rbr__batoid__consdb_v1`. The MIW route is deferred: the registered-but-empty
  `v50_34__miw__consdb_v1` stays empty here and is built by item 6, where the MIW builds are
  decided. When it is built it needs `--intrinsic miw --intrinsic-ref <the MIW build name>`
  plus a `MiwCornerLookup` from `aos/code/miw_corner_intrinsic.py`; the builder raises
  rather than guessing if the ref is missing.
- **Before any RBR build, write the corner-basis adapter (Q13, answered).** It is a
  small shim object built from `corner_recovery_basis` that presents the six attributes the
  solver module actually reads — `U_eff`, `Sigma`, `V`, `n_keep_eff`,
  `normalization_weights`, `dof_idx` — mapping one-to-one onto the basis dict's `U`, `s`,
  `V`, `n_modes`, `norm_vector`, `dof_indices`. With that, `invert_range_penalty`,
  `dof_range_vector` and `achieved_residual` all run against the corner problem unchanged,
  and the RBR DOF are comparable to the truncated DOF by construction rather than by
  measured agreement — which is what the old "check the forward operators agree" bullet was
  reaching for. Assert that the shim's weights are the `REQUIRED_NORM_YAML` ones.
- Store the achieved residual `dW - S (d / w)` for the RBR variant, not the subspace
  residual, and record on the variant which residual its `resid_rms_um` column holds.
- Carry `run_olr.py`'s identity check `olr_deviation == olr_opd - intrinsic` into the moved
  code and run it per visit at build time, so a sign or basis error in the forward direction
  fails loudly. Since the Zernikes are not stored, this is a build-time assertion in the
  log, not something a later query can re-derive from the database alone.
- Keep every scheme as rows under its own variant, never as new columns, so a comparison
  stays a self-join on `visit_id` through `efd_db.compare_variants`. Three schemes on the
  batoid route is **three variants** here (Q12); the MIW route would double that and is
  deferred to item 6.
- Run the builds as sharded batch jobs through `run_build.sh --what state --mode batch`,
  which Aaron submits. Size it honestly first: with Q12's batoid-only list this is **three
  full-span builds over 366 nights**, not "2.4x the nights" of a single build. Measure the
  per-night cost on one night before submitting the set.
- **Replace `v50_34__batoid__consdb_v1` with a complete rerun** (decided 2026-10-02) rather
  than extending it night-by-night. Its existing 90,695 rows over 181 nights are discarded
  and rebuilt over the full 366-night span, so the three variants are built by identical code
  against identical inputs and a scheme-to-scheme difference cannot be an artifact of build
  vintage. Dropping those rows needs Aaron's go-ahead at the time.
- Confirm the state estimators are held for the life of each shard. `corner_recovery_basis`
  caches on `id(state_estimator)`, and CPython reuses an `id` after garbage collection, so
  a short-lived estimator per scheme could in principle return another scheme's basis from
  the cache. With three schemes live in one build this is worth an explicit check rather
  than an assumption.
- Compare the schemes **on recovered image quality first, and on the DOF values second**,
  using the open-loop versus deviation-recovered difference per DOF over the science sample.
  Report v-modes alongside for continuity with the summit, but not as the metric of
  record — see the caveat above.
- Update `value_added/docs/schema.md` and `status/build_progress.md` with the new axis, the
  new columns, and the realized row counts and spans.
- Spot-check against a night already analysed elsewhere, so a build error shows up as a
  disagreement with a known result rather than passing silently.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. What does "lower gains for higher v-modes" mean concretely?** A gain vector over the
34 kept modes, a roll-off function of mode index, or a per-mode fit.

**A:** A vector of gains over the DoF.  Starting value might be a gain of 0.3 for the 22 DoF currently used and a lower value of 0.1 for the remaining mirror modes. 

**Q2. How large is a "large sample" of science visits,** and does the 20260419 cut leave
enough once the Danish 1.2 and refit WCS cut is also applied?

**A:** _Lets fill these tables for all science and acq visits, and I cut later on which
subsamples to use for various studies._

**Q3. Does the OLR table live in the existing `aos_efd.duckdb` or its own database file?**

**A:** I am not sure, but this table will also need to be keyed off the nDof/nVmode scheme and perhaps also the wavefront retrieval,so I guess it will want its own table

**Superseded 2026-10-02** by the merge with the value-added build. There is no separate OLR
table: the open-loop and deviation-recovered states are rows in the existing
`optical_state`, keyed by `variant_id`, with the scheme and the retrieval route carried as
variant axes exactly as this answer asked for. Kept as the record of the keying decision.

**Q4. Is 20260419 the right single cut?** It is the SVD normalization fix date and appears
to be the Danish 1.2 changeover date too, but refit WCS may have gone online on a
different day. Needs confirming against the online collection provenance.

**A:** Again for the value added duckdb lets just fill this for all visits

**Q5. How does RBR enter the variant name?** The name is three parts today
(`v50_34__batoid__consdb_v1`) and nothing in `state_variant` describes the solver. Either add
a fourth axis — say `v50_34__batoid__consdb_v1__rbr`, with the unregularized builds taking an
implicit or explicit `trunc` — or encode it in the `scheme` field as a pseudo-scheme like
`50_34_rbr`. The fourth axis is cleaner and matches how `fam_variant` already carries four;
the pseudo-scheme is less code but overloads a field that means DOF count and v-mode count
everywhere else. Q3's answer already says the keying must cover the scheme and perhaps the
retrieval, so this is the same decision made concrete.

**A:** _`v50_34_rbr` will encode the scheme._ So the pseudo-scheme option, not a fourth axis:
no schema change, and `SCHEMES` gains a `50_34_rbr` entry pointing at the same `all_50` DOF
set. Consequence to accept: `scheme` no longer implies `n_dof`/`n_modes` uniquely, and the
penalty parameters live only in `state_variant.notes`.

**Q6. Where do the open-loop CWFS Zernikes live?** The Trim DOF are already in
`visit_telemetry` and the Trim v-modes are already in `optical_state.v_modes_trim`, but the
implied Zernikes are 21 coefficients per corner per visit and exist nowhere. Options: list
columns on `optical_state` next to the v-modes; a separate long table keyed the same way; or
not stored at all, recomputed on read from the stored Trim v-modes, since the forward
projection is cheap.

**A:** _The CWFS Zernikes are currently in the ConsDB, with OPD Zernikes for all four
corners. Note that these are present for all science and acq visits independent of open or
closed loop. We can get these quantities as needed from the ConsDB. What isn't in the ConsDB
is the intrinsic wavefront, and so we need to use either Batoid intrinsic or MIW. Eventually
we'll have a processing in the Butler to replace the ConsDB values, but not yet._

**Q7. Which span does the build cover?** The existing 50/34 variant runs `day_obs` 20251102
to 20260713, and the data-selection cut above says after 20260419. Options: match the
existing variant's span so the three schemes join visit-for-visit; restrict to post-20260419
where the SVD normalization is fixed; or extend all three back to 20250415 where ConsDB
Zernikes exist, which means also rebuilding the populated 50/34 variant.

**A:** _See above, I want to fill all science and acq visits_

**Q8. What RBR `kappa` and `power`?** The bounce test used `kappa = 4` and `power = 3`, both
dimensionless, while `invert_range_penalty` defaults to `kappa = 0.5, power = 2`. Science
visits sit near the nominal optical state rather than at a deliberately bounced one, so the
penalty may rarely bind. Either adopt the bounce values for continuity, or sweep on a sample
of nights and pick for the science-visit regime.

**A:** _Use kappa=4 and power=3_

**Q9. Does the MIW intrinsic route come along?** Item 6 rebuilds the Measured Intrinsic
Wavefront (MIW) under three correction schemes, and `v50_34__miw__consdb_v1` is registered
but empty. Building the MIW route here would double the variant count; deferring keeps this
item to the batoid route and leaves the MIW variants to item 6, where the MIW builds are
decided.

**A:** _For now lets use the existing MIW_

**Q10. Does `olr/` stay a separate topic?** Its pipeline writes `olr.parquet` per night with
the open-loop OPD and deviation Zernikes. If the open-loop state is built into the
value-added database per visit, `olr/` either becomes the reference implementation this build
is verified against and is then left alone, or it is retired in favour of the database.

**A:** _Lets pull code from the olr topic or reproduce it in the aos/code area (or in the
rubin-work/common/code area) to calculate the OLR Trim-Deviation for DOF, v-modes and DZ
optical state. All of that code should move to one common place, and I will repurpose the
rubin-work/olr topic for analysis of the OLR in the duckdb. So please remove the code from
rubin-work/olr that moves over._

**Q11. Does the OLR sensitivity matrix cover 22 DOF or 50?** `olr/code/olr.py` builds
`sens_mat` with 22 columns and slices the Trim with `dof_state[indices]`, so the OLR Zernikes
it produces are the 22 DOF subset of the applied correction. For the 50/34 schemes the
open-loop Zernikes should arguably use all 50. Either extend the matrix to 50 columns for
those schemes, or keep 22 everywhere and accept that the open-loop Zernikes are a projection
of the correction rather than all of it. This decides whether the moved code is a copy or a
generalization.

**A:** _Use the scheme's own DOF set, so 50/34 gets 50 columns._ There is nothing to
generalize: `olr/code/olr.py` gets its 22 columns only by hand-masking `comp_dof_idx`
(`M1M3Bend[7:] = False`, `M2Bend[5:] = False`), which is precisely what
`make_state_estimator(dof_set=...)` already does via `_comp_dof_idx(DOF_SETS[dof_set])`. So
the moved function takes the state estimator as an argument and reads the column count off
it. `DEFAULT_DOF_INDICES` (`range(0,17) + range(30,35)`) goes away — it is a hand-written
duplicate of `DOF_SETS['standard_22']` that can drift from it silently. Keeping 22
everywhere is rejected on its merits, not on cost: it would put the 50/34 open-loop state
and its deviation-recovered state in different subspaces, which defeats the comparison this
item exists for.

**Q12. Which MIW build, and which of the six variants get built?** Q9 says "the existing
MIW", but `--intrinsic miw` needs `--intrinsic-ref` naming a specific build and a
`MiwCornerLookup`, and the builder raises rather than defaulting. Three schemes over two
intrinsic routes is six variants; the natural subset is the three batoid ones plus 50/34 MIW
(the one already registered), which is four. Worth fixing the list and the MIW build name
before any batch submission, since each variant is a full-span build.

**A:** _Batoid intrinsic to start._ So **three variants, not four**:
`v22_12__batoid__consdb_v1`, `v50_34__batoid__consdb_v1` and `v50_34_rbr__batoid__consdb_v1`.
No MIW build name is needed here, and `v50_34__miw__consdb_v1` stays registered-but-empty
until item 6 builds it — which leaves the empty-variant trap above live, so an analysis must
not name it meanwhile.

**Q13. How does the RBR solver reach a corner measurement?** This is the one thing blocking
the `50_34_rbr` variant. `invert_range_penalty` wants `dW` over `svd.kj_grid` — DZ
coefficients over the full field — while a science visit gives 84 corner Zernike values and
`recover_optical_state` inverts the corner-evaluated SVD. Three options:

1. **Shim `corner_recovery_basis` into the solver's interface.** The solver module reads
   only `U_eff`, `Sigma`, `V`, `n_keep_eff`, `normalization_weights` and `dof_idx` (and
   `kj_grid`, which the three functions needed here never touch). The basis dict already
   carries all six under the names `U`, `s`, `V`, `n_modes`, `norm_vector`,
   `dof_indices`, so this is a small dataclass in `aos/code/aos_state.py` and no solver
   change. `dW` becomes the 84-value `z_dev`. `dof_range_vector` still works, because its
   `f_j` is the field-averaged quadrature over the full 50 DOF and does not depend on which
   rows the sensitivity was evaluated at — provided `dof_idx` is the corner basis's
   `dof_indices`. **Preferred**; cheapest and keeps one solver.
2. **Project the corner measurement into DZ space,** fitting a DZ field to the four corner
   vectors and then running the existing path. Rejected: four field points cannot constrain
   the focal-plane DZ orders `build_ofc_svd` uses, so a regularized fit feeds a regularized
   solve and the RBR-versus-truncated DOF difference then has two inseparable causes.
3. **Write a corner-space range-penalty solver.** Duplicates the IRLS and contradicts the
   scope line about calling the shared solver rather than copying it. Only if option 1 needs
   real surgery.

If option 1 turns out not to work, drop `50_34_rbr` from this item, build the two real
schemes full-span, and move RBR to its own item with the shim as its first task — the
22/12-versus-50/34 comparison is the stated goal and does not need RBR.

**A:** _option 1 which is the standard approach for finding the optical state from the CWFS_

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
- Apply the RBR penalty inside the MIW build's per-visit optical-state recovery, calling the
  shared solver in `smatrix/code/regularized_inversion.py` rather than copying it.
- Compare the residual MIW between the three builds, term by term and as field maps, using
  the achieved residual `dW - S (d / w)` rather than the subspace projection.
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

**A:** _lets move to Danish 1.3 for this study_

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

</details>

---

## 7. Reorganize the `thermal_focus` analysis and its PDF report

**Status:** not started · **Blocked on:** nothing

Restructure the `thermal_focus` PDF so it opens with the study description and the summary
plots, then moves through the telemetry-term comparisons to the resulting trims. Settle the
model by evaluating the candidate telemetry terms in a controlled sequence against a mean
truss temperature baseline, and drop the material the study has outgrown.

**Goals:** Make the report read in the order a reader needs it, and establish which telemetry
terms the deliverable model carries.

<details>
<summary>The current report, the terms to evaluate, scope and open questions</summary>

### Existing machinery to build on

| piece | path |
| --- | --- |
| the report builder, one page per `figure_*` or `_text_page` call | [thermal_focus/code/run_thermal_focus_analysis.py](../../thermal_focus/code/run_thermal_focus_analysis.py), `main()` |
| the fitting engine, `huber_line`, `evaluate`, `model_comparison`, `nested_comparison`, `per_band_fit` | [thermal_focus/code/thermal_focus_fit.py](../../thermal_focus/code/thermal_focus_fit.py) |
| the feature groups and `resolve_features` | [thermal_focus/code/thermal_focus_lib.py](../../thermal_focus/code/thermal_focus_lib.py), `FEATURE_GROUPS`, `DELIVERABLE_GROUPS` |
| the standalone online calculator | [thermal_focus/code/trim_calculator.py](../../thermal_focus/code/trim_calculator.py) |
| the network and DuckDB stage | [thermal_focus/code/run_thermal_focus.py](../../thermal_focus/code/run_thermal_focus.py) |
| the study doc | [thermal_focus/docs/thermal_focus.md](../../thermal_focus/docs/thermal_focus.md) |

The report is 18 pages, built in `main()` inside `with PdfPages(pdf_path)`, currently ordered
as three parts: before the correction, training, then all the data. The pages to keep, and
where they are now:

| page | what it is |
| --- | --- |
| 2 | `figure_before` — focus error against truss temperature, by band, and the residual |
| 3 | `figure_sample` — the per-night medians |
| 7 | `figure_model` — before and after the correction, coefficient stability, residual by band |
| 12 | `figure_elevation` — elevation slopes and the hysteresis test |
| 14 | `_text_page`, "the correction as degrees of freedom" — the conversion explainer and the per-visit and start-of-night trim tables |
| 15 | `figure_dof` — the trim to command per visit |
| 16 | `figure_dof_start` — the trim at the start of each night |
| 18 | `figure_t539` — predicted trim against what the initial alignment block settled on |

`huber_line` already returns both `pearson_r` and `spearman_rho`, so both correlations are
computed wherever it is used; several display sites print only Pearson, among them the panel
titles at `figure_before` and the camera-temperature rows on page 8.

The axis clipping to the 1st and 99th percentiles is a single site in `figure_dof`, with the
percentiles also written into the legend string and the docstring.

The feature groups, with the column names as they appear in the code:

| group | columns | unit |
| --- | --- | --- |
| `truss` | `truss_temp_mean_c` | deg C |
| `grads` | `m1m3_z_gradient_c_per_m`, `m1m3_y_gradient_c_per_m`, `m1m3_radial_gradient_c_per_m`, `m1m3_x_gradient_c_per_m` | deg C per m |
| `r2grads` | `m1m3_r2_coeff_c`, `m1_r2_coeff_c`, `m3_r2_coeff_c` | deg C per unit norm r2 |
| `camtemp` | `cam_AverageTemp` | deg C |

`DELIVERABLE_GROUPS` is currently `('truss', 'grads')`. The r2 terms and camera temperature
are analysed but not in the deliverable set.

`truss_temp_mean_c` is not stored in the DuckDB. It is derived on the ConsDB join in
`value_added/code/efd_db.py` as the mean of the two ConsDB thermometers
`tma_truss_temp_pxpy` and `tma_truss_temp_mxmy`, then interpolated within each night, with a
companion `truss_temp_mean_c_interpolated` flag. The M1M3 gradients and `cam_AverageTemp`
are stored and come from `visit_telemetry`. `run_thermal_focus.py` is the only stage that
touches the network or the DuckDB; the analysis script reads only parquet.

There is no neural-network material in the topic or its doc. The lengthy explanation to
remove is the out-of-fold and train/test justification on pages 4 and 6, with the holdout
split coming from `section_holdout` and its figure being `figure_training` on page 5.

The hysteresis test currently concludes no consistent direction dependence, at a sign-test
p = 0.084 dimensionless, so keeping it retains a null result rather than a positive one.

### Scope

- Reorder the report to open with the study description and the summary plots, then the
  model comparisons, then the resulting trims.
- Write the opening study description: predict start-of-night focus, expressed as the degrees
  of freedom contributing to v-mode 1, from telemetry including the Telescope Mount Assembly
  (TMA) truss temperatures and the M1M3 thermal gradients, working in v-mode space.
- Explain the focus conversion in the opening: v-mode 1 to approximate equivalent hexapod dz
  in µm, via the factor relating it to camera or M2 defocus, stating that the conversion is
  not exact at the percent level but gives a physical sense of the focus change.
- State in the opening that the analysis uses ConsDB-derived Zernikes because they are the
  consistent and comprehensive data set, that the selected sample is the day_obs with a
  consistent look-up table, and what was excluded, including the hotter data.
- Rename "uncorrected response" to "open-loop focus" throughout, and label the error quantity
  "focus error".
- Keep and improve the opening summary plots: open-loop focus by band, open-loop focus
  against mean truss temperature, and before and after the linear correction, showing the
  residual after the Huber robust fit and the fitted coefficient.
- Show both the Pearson and the Spearman correlation coefficients at every display site,
  including the panel titles that currently print Pearson alone.
- Add a database-wide plot of mean TMA truss temperature over all data in the database.
- Keep the nightly-median plots: median open-loop focus error per night, median truss
  temperature per night, and the relation between them.
- Identify the outlier nights in the nightly medians and report where they fall in focus.
- Replace the training and validation discussion with a short statement: splitting by visit
  is inappropriate because images within a night are strongly correlated, so a split must be
  by night; and since the model is a low-dimensional linear Huber fit, a train/test or
  fold-based approach is not needed.
- Remove the fold analysis from the report: the out-of-fold and per-fold blocks on pages 4
  and 6, and the coefficient-stability panel in `figure_model`.
- Evaluate the candidate terms in sequence: mean truss temperature as the baseline, then
  truss temperature plus each of the four M1M3 gradients, the M1 and M3 r2 terms and camera
  temperature individually; rank the individual terms; then add them cumulatively, strongest
  first after truss temperature.
- Show each model with two plots: predicted against measured open-loop focus, and a
  one-dimensional residual histogram annotated with its NMAD in µm.
- Report the final model, expected to be truss temperature plus the four M1M3 gradients, with
  the remaining focus error by band.
- Keep the hysteresis study, comparing the rising and falling legs.
- Widen the `figure_dof` axis clipping from the 1st and 99th to the 0.25th and 99.75th
  percentiles, updating the legend string and the docstring with it.
- Show the applied-trim plots for all visits, and add the equivalent plots for the first
  visit of each night, selected on telemetry.
- Keep the four start-of-night comparison plots against the trim the closed-loop alignment
  blocks settled on, the correction size against Modified Julian Date (MJD) with its
  distribution, and the predicted against applied correction including its outliers.
- Identify the day_obs of the large outliers on the applied-correction axis of the final
  comparison.
- Update `thermal_focus/docs/thermal_focus.md` to match the new report order and the settled
  feature set.

### Open questions

Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Where does the database-wide truss-temperature plot get its data?** `truss_temp_mean_c`
is not stored in the DuckDB — it is derived on the ConsDB join from the two thermometers and
interpolated within each night. So a database-wide plot needs either a live ConsDB pass in
`run_thermal_focus.py`, which is the network stage, or a new `value_added` builder that
materializes the column into `aos_efd.duckdb`. The latter makes it available to every other
study; the former is a smaller change confined to this topic.

**A:** _Access of the ConsDB is fast enough that the existing code that gets the mean truss temp is fine and we don't need this ithe duckdb_

**Q2. Does the rename reach the dict keys, or only the display strings?** Roughly 15
user-visible strings carry "uncorrected", but so do about 8 dict keys and the module constant
`trim_calculator.UNCORRECTED_NMAD_UM`, which crosses into `thermal_focus_fit.py` and the
standalone calculator. Renaming only the display strings leaves the code and the report using
different vocabulary.

**A:** _Only need to change names in the PDF file not in the code or parquet files_

**Q3. Does the report keep reporting a cross-validated NMAD after the fold presentation is
removed?** `GroupKFold` in `thermal_focus_fit.evaluate` is what produces every NMAD the
report currently quotes, so dropping the fold *presentation* is separable from dropping the
mechanism. Either the quoted NMAD becomes an in-sample number, or the folds keep running
unseen.

**A:** _I still want the robust RMS (which I assume is what NMAD means here) for the residual of the Trim-Deviation v1's equivalent dz (v1_dz) around the prediction.  That doesn't need the KFold analysis I believe._

**Q4. Is the page-14 material to retain the DOF conversion explainer, or the fitted-model
summary?** Page 14 is the text page "the correction as degrees of freedom", holding the
conversion explainer and the per-visit and start-of-night trim tables. The fitted-model
summary is page 6, which is also where the per-fold table to be removed sits.

**A:** _page 14 is the 'correction as degrees of freedom'_

**Q5. Does the hysteresis test stay as a null result, or get a decision?** It currently
reports no consistent direction dependence at a sign-test p = 0.084 dimensionless. Keeping it
preserves the evidence; the alternative is to state the conclusion in the text and drop the
page.

**A:** _Keep the plots and as a null result we just show the plots, which I want to keep_

**Q6. What decides "useful" when adding terms cumulatively?** A reduction in residual NMAD in
µm by some threshold, coefficient sign stability, or physical interpretability. The r2 terms
were previously measured at a 4.8% reduction in robust residual scatter and their adoption was
left open.

**A:** _I will look by eye at the results, since I am weighing the NMAD residuals with the overhead of adding the r2 variables_

</details>

---

## 8. Guider star second moments, image and centroid motion, into the value-added DB

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
