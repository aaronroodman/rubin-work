# Phase 3 handoff: the other products — move, fix, rebuild

> **Status:** in progress · **Last updated:** 2026-10-10 · **Kind:** working state (handoff)

Phase 3 of `notes/status/organization_plan.md`: move every remaining product's code (3a),
apply the intended fixes (3b), then rebuild every product fresh into the new tree (3c).
Read the Phase 3 part of section 8 there for the step list; this file holds the state
between sessions.

**Resume protocol.** Each step writes its state here and then stops, so the session can be
cleared. A fresh session re-reads this file and the plan's sections 2–5, not a summary.
Phase 2's own state is in `notes/status/phase2_fam_tables_handoff.md`, which is complete
and is the worked example for every later product: reference build, move, zero-tolerance
comparison, manifest.

## Done and committed

| commit | what |
|---|---|
| `f7985db` | `rubinwork/products/fam_tables/refbuild/check_collections.py`, the 3c pre-flight |
| `9392f9b` | the 3c rebuild-feasibility results below |
| (step 1) | `cwfs_tables` reference build with the unmoved code, below |

## Next concrete action

**Phase 3a, `cwfs_tables`, step 2: the move.** Step 1 (the reference build) is done; see
"3a step 1" below for the command and the counts.

## 3a, `cwfs_tables` — the move

### Scope, agreed 2026-10-10

Moves: `aos/code/cwfs/run_wfs_mktable.py` only, to
`rubinwork/products/cwfs_tables/builders/`, with a shim at the old path. Snakefile rule
`wfs_mktable` only. The other five `aos/code/cwfs/` scripts (`corner_compare`,
`dof_compare`, `mimic`, `refit_ensemble`, `fam_compare`) are the CWFS-vs-FAM **study** and
stay; `wfs_intrinsic_sidecar` belongs to the `miw` product and is left alone.

Everything that reads the CWFS tables keeps reading the **old tree**
(`aos/output/wfs_ingest/<P>/<cwfs>/`) until phase 5. Each consumer reaches them through a
`--wfs-dir` argument or a literal old-tree path, and none of them imports
`run_wfs_mktable`, so the move cannot break them: `rule all`, `wfs_corner_compare`,
`wfs_dof_compare` and `wfs_intrinsic_sidecar` in `aos/Snakefile`; `run_wfs_fam_compare.py`;
`run_wfs_corner_compare.py` and `run_wfs_dof_compare.py` (both via `--wfs-dir`);
`aos/code/infra/migrate_output_layout.py`; `aos/snippets.ipynb`. As in phase 2, the three
table targets become inputs `aos/Snakefile` no longer builds.

### Variants: six entries, `aos/param_sets.yaml` stays byte-identical

`cwfs_tables/variants.yaml` owns them, each naming its `fam_variant` and carrying the
`collection`, `dataset_type`, `reader` and `seq_offset` fields that were in
`fam_tables/variants.yaml` `wfs_collections`. `fam_tables/variants.yaml` drops
`wfs_collections`; `gen_param_sets.py` merges both files back into `aos/param_sets.yaml`.

| variant | fam_variant | registered | fields beyond `collection` |
|---|---|---|---|
| `d12-refitWcs` | `danish_1_2` | true | — |
| `d12-refitWcs_2025` | `danish_1_2` | true | — |
| `d12-paired_3mm` | `danish_1_2` | true | `seq_offset: 0` |
| `d12-ai_donut` | `danish_1_2` | true | `dataset_type: aggregateZernikesRaw` |
| `d12-tarts` | `danish_1_2` | true | `reader: unpaired`, `dataset_type: aggregateAOSVisitTableAvg` |
| `d10-wep17_3_0` | `danish_1_0` | **false** | provenance only; never built |

`aos/param_sets.yaml` is md5 `0f72da651cf9319cc283754c614051fd` before the move and must
stay so — that is the acceptance test, checked with `--check` and a byte `diff`.

**Two decisions deferred to 3b**, both Aaron's calls on 2026-10-10:

1. **`d12-refitWcs_2025` folds into `d12-refitWcs`.** They are the same variant; the 2025
   visits just need a different collection, exactly as the FAM `danish_1_2` carries a
   per-chunk collection override for its 2025 chunks. The two CWFS collections are
   disjoint in `day_obs` (`refitWcs` 20260315..20260513, `refitWcs_2025`
   20250415..20251231), so a date-keyed collection list is the shape. Deferred because
   `run_wfs_mktable` opens **one** `Butler(repo, collections=...)` per `--wfs-name`, so
   merging them is a builder change, not an imports-and-paths move; 3a stays mechanical.
2. **`danish_1_0` is retired whole, both halves.** Dropping only the CWFS half would make
   the generated `aos/param_sets.yaml` lose two lines, and dropping the FAM half in 3a
   would break four live things: the staged-MIW provenance chain
   (`aos/calibration/miw/intrinsic_split_maps_v1.provenance.yaml`, `stage_miw.py`,
   `calibration/README.md`), the `aos-measured-intrinsics` tech note's
   `make_figures.py:23` and `provenance.md:5`, a **live** `overrides:` block at
   `aos/analysis_config.yaml:148` keyed on the long param_set, and
   `test_fam_tables.py:263`. That is a `fam_tables`/`miw` change with its own before/after
   check. So `d10-wep17_3_0` is carried as `registered: false` for now.

### 3a step 1 — reference build with the unmoved code: DONE

Built at commit `9392f9b` (clean tree), interactive on `sdfiana`, **65.3 s wall clock**.

Variant `refitWcs_2025` under param_set `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`, one
night, `day_obs = 20251126`. That variant has **no old-tree build**, so the reference
cannot touch one.

Input: one night of the phase-2 FAM reference, staged by
`rubinwork/products/cwfs_tables/refbuild/stage_fam_night.py` because `run_wfs_mktable` has
no `--day-obs` filter and would otherwise walk all 53 visits of the chunk:

```bash
python -m rubinwork.products.cwfs_tables.refbuild.stage_fam_night \
    --reference /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2 \
    --day-obs 20251126 \
    --out /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/cwfs_tables/_fam_20251126
```

Staged FAM slice: `visits.parquet` 6 rows x 600 columns, `fits.parquet` 5 rows x 653
columns, `donuts.parquet` 17,019 rows x 48 columns (all counts dimensionless).

**Night choice matters.** 20251116, the chunk's first night, has 6 FAM visits but **0**
`fits.parquet` rows — the DZ fit's `median_blur_arcsec <= 1.2 arcsec` quality cut drops all
of them — which makes the validation plot's FAM k=1 overlay all-NaN and the reference
degenerate. 20251126 is the smallest night that does not. The staging script warns on an
empty `fits.parquet`.

The build command, run from `aos/` (the unmoved script's own path):

```bash
cd ~/notebooks/rubin-work/aos && python code/cwfs/run_wfs_mktable.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --wfs-name refitWcs_2025 \
    --coord-sys OCS \
    --tables-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/cwfs_tables/_fam_20251126 \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/cwfs_tables/refitWcs_2025_20251126
```

Output, in `_phase2_refbuild/cwfs_tables/refitWcs_2025_20251126/`:

| file | rows | columns | bytes |
|---|---|---|---|
| `donuts.parquet` | 171 | 20 | 95,409 |
| `visits.parquet` | 6 | 8 | 6,000 |
| `wfs_mktable_validation.pdf` | — | — | 106,605 |

All 6 FAM visits found an in-focus CWFS table (0 missing). Per-visit donut counts
(count, dimensionless): 28, 28, 26, 36, 21, 32, band `i` throughout.
`donuts.meta['nollIndices']` is the 21-term Noll set
`[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26]`.

**Comparison key, verified unique.** `run_wfs_mktable` writes **no donut-id column**, so
the `fam_tables` key `(day_obs, seq_num, detector, extra_donut_id)` does not exist here.
The CWFS key is **`(day_obs, seq_num, detector, thx_OCS, thy_OCS)`**, where `thx/thy_OCS`
are field angles in radians (OCS). Uniqueness, checked on the reference and on all three
built old-tree variants (unique rows of total rows, counts dimensionless):

| table | `day_obs, seq_num, detector` | plus `thx_OCS, thy_OCS` |
|---|---|---|
| reference `refitWcs_2025` (171 rows) | — | **171 / 171 unique** |
| old-tree `refitWcs` (34,631 rows) | 4,485 — not unique | **34,631 / 34,631 unique** |
| old-tree `paired_3mm` (1,223 rows) | 515 — not unique | **1,223 / 1,223 unique** |
| old-tree `tarts` (7,813 rows) | 7,813 — unique | **7,813 / 7,813 unique** |

The detector-only key fails on the paired variants (several donuts per corner sensor per
exposure). Adding the centroids also works, but `tarts` has no centroid columns, so the
field-angle key is the only one that covers every variant. `visits.parquet` keys on
`(day_obs, seq_num)`. The comparison asserts uniqueness so a future variant that breaks it
fails loudly instead of mis-aligning.

The validation PDF is **not** byte-comparable (matplotlib embeds a creation timestamp), so
step 4 compares the two parquet tables at zero tolerance and checks only that the PDF is
produced at a comparable size.

`rotator_angle` and `alt` in the CWFS tables are **copied from the FAM `visits.parquet`**,
not re-derived, so the A1 `rotTelPos` non-reproducibility cannot affect this comparison.

## 3c rebuild feasibility, measured 2026-10-10

### Every collection resolves; the embargo chunk has not moved

`python -m rubinwork.products.fam_tables.refbuild.check_collections` probes each chunk
collection and each `wfs_collections` entry in its own Butler repo and counts datasets of
the aggregate dataset type. **All 15 collections resolve and all hold data.** Counts are
datasets of `aggregateAOSVisitTableRaw` (`donutBlitzFamResults` for the blitz variants),
one per FAM visit, in the chunk's `day_obs` range.

**The 20260713 chunk is still in `/repo/embargo` and is not in `/repo/main`**: 24
`aggregateAOSVisitTableRaw` datasets in `/repo/embargo`, **0** in `/repo/main` on the same
collection name. So the change the plan anticipated is **not needed** — `variants.yaml`
stays as it is, and 3c still needs a repo that can read `/repo/embargo`. Re-check before
the rebuild; the embargo period is what would move it.

`danish_1_2` chunks (FAM), `/repo/main` except where noted:

| chunk | Butler visits | nights | day_obs seen | built in old tree | gap |
|---|---|---|---|---|---|
| 20260315_20260327 | 467 | 8 | 20260315..20260327 | 427 | 40 |
| 20260331_20260409 | 480 | 5 | 20260331..20260409 | 480 | 0 |
| 20260418_20260513 | 219 | 7 | 20260418..20260513 | 219 | 0 |
| 20260514_20260731 | 90 | 5 | 20260521..20260711 | 54 | 36 |
| 20260713_20260713 (`/repo/embargo`) | 24 | 1 | 20260713 | 24 | 0 |
| 20250415_20250531 | 644 | 27 | 20250415..20250531 | 637 | 7 |
| 20250601_20250930 | 812 | 24 | 20250601..20250921 | 811 | 1 |
| 20251001_20251115 | 367 | 21 | 20251022..20251115 | 361 | 6 |
| 20251116_20251130 | 74 | 15 | 20251116..20251130 | 53 | 21 |
| 20251201_20251231 | 346 | 23 | 20251201..20251231 | 319 | 27 |
| **total** | **3,523** | | | **3,385** | **138** |

All counts are visits (count, dimensionless). The 138-visit gap is **expected, not a
defect**: the Butler probe applies no program filter and no quality cut, while the build
applies both. The reference chunk is the clearest case — 74 Butler visits, 53 built, and
the `median_blur_arcsec <= 1.2 arcsec` cut is what drops the rest (it dropped 5 of 5 on
20251116 alone in the timing run below). The two biggest gaps, 40 and 36 visits, are on
the 2026 chunks, which carry `fam_programs` while the probe does not.

`danish_1_2` corner-WFS collections (`wfs_collections`), all `/repo/main`, probed
**unbounded** in `day_obs` — the variant-level `day_obs_min/max` covers only its 2026
chunks, so bounding would have reported `refitWcs_2025` as 0:

| entry | datasets | nights | day_obs seen | dataset type |
|---|---|---|---|---|
| `refitWcs` | 1,183 | 20 | 20260315..20260513 | `aggregateAOSVisitTableRaw` |
| `refitWcs_2025` | 2,263 | 110 | 20250415..20251231 | `aggregateAOSVisitTableRaw` |
| `paired_3mm` | 134 | 2 | 20260315..20260317 | `aggregateAOSVisitTableRaw` |
| `ai_donut` | 1,184 | 20 | 20260315..20260513 | `aggregateZernikesRaw` |
| `tarts` | 1,029 | 17 | 20260316..20260428 | `aggregateAOSVisitTableAvg` |

The blitz variants, `/repo/main`, no `wfs_collections`:

| variant | Butler visits | nights | day_obs seen | built in old tree |
|---|---|---|---|---|
| `danish_1_3_test` | 966 | 15 | 20260315..20260619 | 966 |
| `danish_1_3_v1000` | 966 | 15 | 20260315..20260619 | 966 |

Both match their old-tree builds exactly — the blitz path applies no program filter, and
its quality cuts pass 12 of 12 on the reference night.

### Rebuild time, and the 19 s/visit figure in the plan is wrong

Measured on `sdfiana`, one night of the 20251116 chunk, `--workers 8`, by running
`run_mktable.py` three times:

| run | visits | wall clock |
|---|---|---|
| 20251116, `--no-thermal`, cold caches | 5 | 453 s |
| 20251116, with the thermal loop | 5 | 466 s |
| 20251116, `--no-thermal`, warm caches | 5 | 453 s |
| 20251117, `--no-thermal` | 15 | 845 s |

Cold and warm `--no-thermal` agree to 1 s on the 5-visit night, so there is no cache
effect to confuse the comparison, and the two no-thermal points give a clean linear cost:

- **fixed overhead 258 s per `mktable` invocation** (Butler init, the ConsDB query, the
  thermocouple table, the validation plot) — paid once per chunk, not per visit
- **marginal 39.1 s/visit** of donut reading, which is the dominant term
- **the EFD thermal loop adds 2.5 s/visit**, from 466 s against 453 s on the same 5 visits

**The plan's section 10 said the thermal loop costs about 19 s/visit and 17 min of the
reference's 54 visits. That does not hold:** at 2.5 s/visit it is about 3 min of a 74-visit
chunk, roughly 6% of the chunk's cost, not a third of it. The 39.1 s/visit donut read is
what dominates. Dropping the loop is still worth doing — it is what removes the EFD from
`mktable` and lets the chunk run in batch (plan 3b) — but **it is a batch-eligibility
change, not a speed change**. Section 10 of the plan is corrected to say so.

Cross-check: 39.1 s/visit plus the 258 s overhead predicts 52.6 min for the 74-visit
reference chunk, against the phase 2 reference build's 53 min for the whole chunk
including the DZ fit and the telemetry attachment. Consistent.

Per-variant rebuild estimate, from those rates:

| variant | visits | serial `mktable`, no thermal | with thermal | longest chunk |
|---|---|---|---|---|
| `danish_1_2` | 3,523 | 39.0 h | 41.5 h | 8.9 h (the 812-visit chunk) |
| `danish_1_3_test` | 966 | 1.0 h | 1.0 h | 1.0 h |
| `danish_1_3_v1000` | 966 | 1.0 h | 1.0 h | 1.0 h |

The blitz variants are an hour, not a day, because they do **not** use `mktable`: the A4
reference ran `run_blitz_mktable --fit` over 12 visits in 43 s, i.e. 3.6 s/visit including
the DZ fit, reading one pre-aggregated `donutBlitzFamResults` table per visit instead of
per-donut Butler datasets. 966 visits × 3.6 s/visit = 58 min.

`danish_1_2`'s 39 h is serial CPU across ten chunks. The chunks are independent, so with
`-j` the wall clock floor is the **longest chunk, 8.9 h** for the 812-visit
20250601_20250930 chunk. Memory: the `mktable` rule declares
`mem_mb = 3000 × workers = 24 GB` per chunk, so four concurrent chunks need 96 GB.

Downstream of `mktable`, measured in this session's step-5 rerun of the 53-visit reference
chunk (`-j4`, wall clock 2 min 40 s total):

| rule | wall clock |
|---|---|
| `fit` | 29 s |
| `combine_donuts` | 3 s |
| `combine_visits` | 1 s |
| `combine_fits` | 1 s |
| `attach_telemetry` | 157 s |
| `manifest` | 1 s |

`attach_telemetry` is the only expensive one and it scales per visit: 157 s for 53 visits
is 3.0 s/visit, so about 3 h for `danish_1_2`'s 3,385 built visits. It queries the EFD.

### What can run in batch

- **`mktable` with `--no-thermal`: yes**, once plan 3b lands. It is the long pole (39 h of
  the 42 h), it is embarrassingly parallel over chunks, and `--no-thermal` is what removes
  its EFD dependency. Ten chunks as ten Slurm jobs turns 39 h of CPU into an 8.9 h wall
  clock bounded by the largest chunk, or less if that chunk is split by night.
- **`mktable` with the thermal loop: no.** The EFD resolves only from interactive nodes.
- **`fit` and the three `combine_*` rules: yes.** No EFD, no ConsDB, minutes of CPU. Not
  worth a batch job on their own.
- **`attach_telemetry`: no**, for the same EFD reason, which is exactly why the Snakefile
  has `--config attach_telemetry=0` for batch mode. Run it afterwards interactively: about
  3 h for `danish_1_2`.
- **`manifest`: no**, 1 s, and it must run after `attach_telemetry` or the build is
  manifested without its telemetry columns. The rule already depends on the telemetry
  stamp when `attach_telemetry=1`, so a batch run (`attach_telemetry=0`) writes the
  manifest **without** it. For a two-stage batch-then-interactive rebuild, write the
  manifest only in the second stage.
- **The blitz variants: not worth batching**, an hour each, and they query ConsDB.

**No batch job has been submitted.** Doing so needs Aaron's OK on the exact submit
command, per `CLAUDE.md`.

## Open, not yet resolved

- **`rotTelPos` is not applied yet** (plan 3b, handoff section A1). Until it is, a rebuild
  of `danish_1_2` differs from the old-tree build in `rotator_angle` by up to 0.188 deg of
  camera rotator angle on the visits where ConsDB `physical_rotator_angle` is NULL — 19 of
  them in the reference chunk. Any 3c comparison must expect that.
- **A rebuild of `danish_1_2` loses 362 `cam_*` columns** from `fits.parquet` until the
  sidecar is reworked, because `combine_fits` drops them (phase 2, "Step 4 requirements").
  The plan lists it as an expected 3c difference.
- **`backfill_visit_sides.py` mutates a product table in place**, which rule 5 of the
  plan's section 2 forbids. Decide in 3a/3b whether it becomes a build step or is retired.
  It still targets the old tree, deliberately.
- The timing runs wrote throwaway output to
  `_phase2_refbuild/_timing_step5/{nothermal,thermal,n15}/`. Not deleted — file deletion
  needs Aaron's OK.
