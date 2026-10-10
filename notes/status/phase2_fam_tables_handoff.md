# Phase 2 handoff: the fam_tables pilot product

> **Status:** in progress · **Last updated:** 2026-10-10 · **Kind:** working state (handoff)

Phase 2 of `notes/status/organization_plan.md`: `fam_tables` end to end. Read the Phase 2
part of section 8 there for the step list; this file holds the state between sessions.

**Resume protocol.** Each step writes its state here and then stops, so the session can be
cleared. A fresh session re-reads this file and the plan's sections 2–5, not a summary.

## Done and committed

| commit | what |
|---|---|
| `58a6606` | `aos/code/infra/phase2_reference_build.sh`; the sidecar/EFD review item in the plan |
| `b355446` | Step 1 reference build recorded; `rubinwork/products/fam_tables/` with `variants.yaml`, `reader.py`, `compare_builds.py`, 18 tests |
| `d3f44a5` | This handoff, after step 1 |
| `cbe3b12` | **A1**: `rotator_angle` resolved; `rotTelPos` identified as the right source |
| `c757462` | **A2**: `danish_1_2` chunk narrowed to `20260418_20260513` in `variants.yaml` |
| `29dbebc` | **A3** decision: `variants.yaml` carries FAM+WFS and generates `param_sets.yaml` |

**Test status: 17 of 18 pass.** `test_chunks_match_snake_config` **fails on purpose** after
A2 — `variants.yaml` now says `[20260418, 20260513]` while `aos/snake_config.yaml` still
says `[20260418, 20260531]`. The failure is the open A2 decision made visible; do not
"fix" it by editing the test. It clears when `snake_config.yaml` is generated from the same
source (A3) or when Aaron picks an A2 option.

### Step 1 — reference build (done)

Baseline for the code move, built with the **unmoved** code at commit `cc500dc`.

- Variant `danish_1_2`, one chunk `20251116_20251130`: 6 nights (20251116, 17, 26, 27, 28,
  29), chosen because it is small and exercises the 2025 per-chunk overrides (own
  collection, `programs: [BLOCK, CWFS, AOSSEQUENCE]`).
- Command: `aos/code/infra/phase2_reference_build.sh <output-root>`, which pins the exact
  `mktable` / `fit` / `combine_*` / `attach_telemetry` commands from the Snakefile rules so
  the rerun cannot drift.
- Output: `/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2/`
  (outside both data trees, group-mount form so it resolves in batch too).
- Ran locally on `sdfiana032`, **53 min** (07:54:45Z to 08:47:52Z). No batch job.

Row counts, combined tables:

| table | rows | columns |
|---|---|---|
| `donuts.parquet` | 145,264 | 48 |
| `visits.parquet` | 53 | 600 |
| `fits.parquet` | 47 | 653 |

**Against the August build of the same chunk** (`compare_builds.py`): all 48 donut columns
identical on 145,264 rows. `visits`/`fits` differ only in telemetry — 196 `m1m3_dt_*`
thermocouple columns the August run lacks (its node could not reach the EFD:
`getSite() returned 'local'` in its log), the `tweak_dof*` / `*_gradient` /
`rotator_angle` values, and 412 `fits` columns at float round-off below 1e-12. **No DZ-fit
or wavefront column differs.**

### Step 5, partly done ahead of its turn

`reader.py` gives `load(variant, build="current")` returning `(donuts, visits, fits)`,
plus `load_table`, `variants` and `variant_config`. 18 tests pass; 44 with the catalog's
own. What remains for step 5 is switching the **build** code to the catalog.

## Next concrete action

Three things are **waiting on Aaron**, all recorded in full below:

1. **A4** — OK the proposed blitz reference build (`danish_1_3_v1000`, `day_obs=20260428`,
   ~25 s, no batch job) and its exact command, then run it.
2. **A3** — OK implementing the decided design: migrate the `wfs_collections` blocks and
   the 130 comment lines into `variants.yaml`, write the generator, regenerate
   `param_sets.yaml`, prove `snakemake -n` still gives 212 jobs.
3. **A2** — pick an option for `aos/snake_config.yaml` (or let A3 subsume it).

Then step 2, below. **Do not start step 2 before A4 is built** — it is the only
before-the-move baseline for the blitz path, and once the code moves it cannot be made.

**Step 2: move the FAM-table build code into `rubinwork/products/fam_tables/`.** Nothing
is moved yet. The move is four files, by `git mv` into a new `builders/` subdirectory:

```
aos/code/fam_processing/run_attach_telemetry.py   -> builders/run_attach_telemetry.py
aos/code/fam_processing/run_blitz_mktable.py      -> builders/run_blitz_mktable.py
aos/code/fam_processing/blitz_reader.py           -> builders/blitz_reader.py
aos/code/fam_processing/backfill_visit_sides.py   -> builders/backfill_visit_sides.py
```

Then: leave shims at the four old paths; move the Snakefile rules `mktable`, `fit`,
`combine_donuts`, `combine_fits`, `combine_visits`, `attach_telemetry` (lines 303–400) into
a Snakefile under the product, keeping the remaining `aos/Snakefile` rules working; and fix
`run_blitz_mktable.py`'s `from output_paths import study_dir` (its only use is the default
`--out-dir` at line 412, which becomes a catalog path).

Then step 3: rerun `fit` + `combine_*` + `attach_telemetry` from the existing chunk donuts
and compare with `compare_builds.py`. Aaron's decision: **do not** rerun `mktable` — it
stays in `ts_intrinsic_wavefront`, the move cannot touch it, and its EFD thermal loop is 47
of the 53 minutes.

## Decisions taken

- **Three variants registered**, the in-use ones: `danish_1_2` (build `20260920`),
  `danish_1_3_test` (`20261004`), `danish_1_3_v1000` (`20261005`), dated from each
  build's newest mtime. Not registered: `danish_1_0` (obsolete, disabled in
  `snake_config.yaml`, output in `aos/output/archive/`, but kept in `param_sets.yaml`
  because it is frozen MIW provenance), `danish_1_1_1` (gone from `param_sets.yaml`),
  `danish_1_2_0_wep17_7_0_2025` (not a param_set — a per-chunk `phrase` of the five 2025
  chunks that fold into `danish_1_2`).
- Variant names are today's `dir_name` values; each entry carries the long `param_set` key,
  which stays the identity for the value-added DB rows and the frozen provenance files.
- Copy budget for step 4: 24.78 + 18.36 + 13.30 = **56.44 GB** apparent. Free on
  `/sdf/group/rubin/u/roodman/LSST/rubin-work/products` before the copy: **727 GB of
  932 GB (22% used)**.

## Tried and rejected, and constraints found

- **`mktable` and `fit` are not `aos/` code.** They shell out to
  `$TS_INTRINSIC_WAVEFRONT_DIR/bin/run_mktable.py`, `run_dz_fit.py` and
  `combine_parquets.py` in the external `ts_intrinsic_wavefront` package. Only
  `attach_telemetry` calls into `aos/code/fam_processing/`. So phase 2 moves the Snakefile
  *rules* and the four builder files; it does **not** move the heavy builders, and the
  product calls the package's runners exactly as the Snakefile does today.
- **Naming a submodule `load.py` does not work** next to a re-exported `load` function:
  `__init__.py`'s `from .reader import load` rebinds the name, so both
  `from ...fam_tables import load` and `import ...fam_tables.load as x` resolve to the
  function and attribute access fails. The submodule is `reader.py`.
- **Comparing donut tables by position is meaningless.** Row order is not stable between
  runs; a positional compare reported 45 of 48 columns as differing, including `day_obs`.
  The unique key is `day_obs, seq_num, detector, extra_donut_id` (verified unique on both
  sides); with it, 0 of 48 differ. `compare_builds.py` encodes this — do not "simplify" it
  back to a positional compare.
- **The committed chunk directories predate a rename.** The August chunk logs write
  `output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/chunks/...`, i.e. the long param_set
  key, while today's tree uses `dir_name`. So chunk mtimes are not build evidence.
- **The August `fits.parquet` carries 362 `cam_*` columns that a fresh build does not.**
  That is the bug `run_attach_telemetry.py`'s own docstring describes: re-running
  `combine_fits` drops sidecar columns, because the combined file was the only place they
  lived. Not caused by anything in this phase.
- **`DatasetNotFoundError` for `day_obs=20251117, seq_num=39`** during `mktable` is
  expected and non-fatal: a visit in ConsDB but absent from the Butler collection, which
  is what `aos/code/fam_processing/check_chunk.py` exists to find.

### A1 — `rotator_angle`: resolved, and the right source is `rotTelPos` (2026-10-10)

The 0.188 deg discrepancy is **explained and is not a bug in either build**. Neither run
read the angle from the sidecar.

`intrinsics_lib.get_rotator_data` (lines 1421–1513) resolves `rotator_angle` (deg) in
three tiers: ConsDB `physical_rotator_angle`, else the **mean of EFD
`MTRotator.rotation.actualPosition` over the exposure window**, else
`calc_rotator_from_visitinfo(parallactic, boresightRot)` from Butler `raw.visitInfo`.

Reference chunk `20251116_20251130`, 53 visits joined on (day_obs, seq_num), reference vs
August build:

| quantity | value |
|---|---|
| visits compared | 53 |
| visits with \|difference\| > 1e-6 deg | 19 |
| median \|difference\|, all visits | 0.000000 deg |
| max \|difference\| | 0.188062 deg |

Per night, max \|difference\| (deg): 20251116 → 0.000000 (n=5), 20251117 → 0.188062
(n=15), 20251126 → 0.000000 (n=6), 20251127 → 0.000000 (n=8), 20251128 → 0.103675 (n=8),
20251129 → 0.000000 (n=11). **Not a constant offset** — it varies per visit, both signs.

Against ConsDB: 34 of 53 visits have a non-null `physical_rotator_angle`, and on those
**both builds match ConsDB to 0.000000000 deg** (max \|residual\|, n=34). The 19 differing
visits are *exactly* the 19 where ConsDB is NULL, so both runs fell through to the EFD
window-mean, which is not reproducible between runs. That is the whole discrepancy.

**The camera rotator angle is `rotTelPos`, and it is in both table families.** Confirmed
on day_obs=20260420, seq_num=89 (BLOCK-T724), ConsDB `physical_rotator_angle` =
59.427156 deg, cross-checked by Aaron against RubinTV (59 deg). Fitting the rotation that
maps (`thx_CCS`, `thy_CCS`) → (`thx_OCS`, `thy_OCS`) over 2972 used donuts gives
59.397229 deg; max \|residual\| in field angle per candidate angle:

| candidate | where | units | max \|residual\| vs the table's own rotation |
|---|---|---|---|
| ConsDB `physical_rotator_angle` | ConsDB `visit1_quicklook` | deg | 1.57e-05 deg |
| `rotTelPos` | `aggregateAOSVisitTableRaw.meta` | **rad** | 9.22e-05 deg |
| `rotAngle` | same meta | rad | 5.17e-03 deg — **wrong quantity** |

`meta['rotAngle']` is `visitInfo.boresightRotAngle` (`donut_viz/aggregate_visit.py:478`),
69.265555 deg on this visit, 9.84 deg from the rotator angle — excluded by 330× in the
residual. `rotTelPos` is exactly `parallacticAngle - rotAngle - 90 deg` (identity verified).

Blitz path (`donutBlitzFamResults`, `danish_1_3_v1000`): `rot_tel_pos` is in the meta **in
degrees** as an astropy `Quantity`, and reproduces the table's own CCS→OCS rotation to
**+0.0000 deg exactly** on 6 of 6 visits checked — it is the angle the pipeline used. On
those visits ConsDB sits **+0.30 deg** away (drifting +0.015 deg over 6 consecutive
visits, i.e. tracking during the sequence), so for the blitz path the meta value is better
than ConsDB.

Three traps recorded:

1. **Units differ between the two families** — `aggregateAOSVisitTableRaw.meta['rotTelPos']`
   is radians (bare float), `donutBlitzFamResults.meta['rot_tel_pos']` is degrees
   (`Quantity`). Convert explicitly; never sniff the magnitude.
2. **Reading blitz `.meta` needs `parameters={'strip_astropy_meta_yaml': False}`** or it
   returns 0 keys — already documented at `blitz_reader.py:81–84`.
3. Passing a `DatasetRef` to `butler.get` forbids extra dataId kwargs such as
   `instrument='LSSTCam'`.

**`skyAngle` is a mislabel.** `intrinsics_lib.py:641` sets
`'skyAngle': meta.get('rotAngle', ...)` and copies it into `visit_info` at lines 1110 and
1201. It is **never read** anywhere in the package — write-only. It is neither a sky angle
nor the rotator angle. `rotTelPos` is the established name across `donut_viz`, `ts_wep`
and `ts_intrinsic_wavefront/bin/ingest_calib_tables.py`; `intrinsics_lib.py` is the only
outlier.

**Proposed fix, deliberately NOT yet applied** (Aaron's call on sequencing): in
`ts_intrinsic_wavefront`, keep ConsDB as the primary source, and replace the EFD and
visitInfo fallbacks with `meta['rotTelPos']`; rename `skyAngle` to `rotTelPos` and read
the right key. That makes `rotator_angle` reproducible and would have made these two
builds agree exactly on all 53 visits. Recommended **after** phase 2 step 3, because step
3 expects exact matches and changing the angle mid-move makes a bad code move
indistinguishable from the intended new value. It needs `scons` re-run and updates the
convention line in `aos/CLAUDE.md` ("ConsDB `physical_rotator_angle`, not
`boresightRotAngle`") plus the `rotator_angle` docs.

### A2 — overlapping chunks: narrowed to 20260418–20260513 (2026-10-10)

`danish_1_2` had two chunks overlapping on 20260514–20260531. The built chunk
`20260418_20260531` holds **219 visits, min day_obs 20260418, max day_obs 20260513, and 0
visits with day_obs > 20260513**, so the overlap carries no data and the range is safe to
narrow. Done in `variants.yaml`: `[20260418, 20260531]` → `[20260418, 20260513]`.

`aos/snake_config.yaml`'s own comment on the next chunk already reads "Newer, DISJOINT
visits (post May 13)", so 20260513 is the intended boundary and the `20260531` end was
stale. The narrowing matches intent.

**Aaron's call, 2026-10-10: the directory name changes in the NEW output tree only.** The
old tree keeps `20260418_20260531` and is not touched. The new tree is written fresh in
step 4, so there is no directory to rename — the step-4 copy just lands under
`20260418_20260513`. No manifest carve-out for an old directory name is needed.

`aos/snake_config.yaml` is updated to match, so the two configs agree and
`test_chunks_match_snake_config` passes. Consequence to know: the **old** tree's chunk
directory no longer matches the config, so `snakemake -n` run against the old tree asks to
rebuild those **219 visits through `mktable`** (212 → 214 jobs, the extra being that
`mktable` plus a `combine_visits`). That is expected and harmless as long as the old tree
is not rebuilt — `mktable` is deliberately not re-triggered by code edits, and phase 5
retires the old tree. Do **not** run `snakemake` against the old tree to "fix" this.

### A3 — one source of truth for variants: DONE (2026-10-10)

Implemented as decided. `rubinwork/products/fam_tables/variants.yaml` is the source of
truth for both halves, and `aos/param_sets.yaml` is generated from it by
`rubinwork/products/fam_tables/gen_param_sets.py`:

```bash
python -m rubinwork.products.fam_tables.gen_param_sets          # write
python -m rubinwork.products.fam_tables.gen_param_sets --check   # exit 1 if stale
```

**Verification: `snakemake -n` from `aos/` is byte-identical before and after** — 214
jobs, 427 rule+output lines, `diff` empty on both the job-count table and the full
rule/output list. (The baseline is 214, not the 212 quoted earlier, because
`snake_config.yaml` now carries the A2-narrowed chunk.)

One intended data difference: `danish_1_2`'s `description` gains "plus the 2025
reprocessing", which `variants.yaml` already said and the old `param_sets.yaml` did not.
Safe — `intrinsics_lib.py:110` explicitly strips `description` as "not a pipeline
parameter", and no other code reads it. Every other field of all four param_sets is
unchanged, compared by parsing both files.

Checked against the real consumers, not just the tests: the external
`intrinsics_lib.load_param_sets()` returns all four param_sets and all five
`wfs_collections` entries, and `output_paths.ps_dir()` resolves every `dir_name`
unchanged.

What moved into `variants.yaml`: the five `wfs_collections` entries with their
`seq_offset` / `dataset_type` / `reader` fields, the `danish_1_0` entry (carrying
`registered: false`), and the prose — the frozen-provenance warning, the 2025 program
groups, the blitz recast explanation and the v1000-vs-legacy pupil-model comparison.
`variants.yaml` grew by 152 lines; the generated `param_sets.yaml` shrank from 191 to 99.

`registered: false` is the "emit but do not register" flag: `variants()` hides
`danish_1_0` while `variant_config('danish_1_0')` still resolves it, so the frozen MIW
provenance keeps working without `danish_1_0` appearing as a product build.
`variants(registered_only=False)` returns it.

Tests: **21 pass** (18 before, plus staleness of the generated file, the unregistered
variant, and the survival of the WFS half with its per-entry fields).

Note this subsumes the A2 `snake_config.yaml` question only partly: chunks still live in
`snake_config.yaml`, not in `variants.yaml`. Generating those too is a later step — the
`chunks` key in `variants.yaml` and the one in `snake_config.yaml` are kept in agreement
by `test_chunks_match_snake_config`.

### A3 — the original analysis, for the record

**Decision (Aaron, 2026-10-10): `rubinwork/products/fam_tables/variants.yaml` carries both
the FAM and the WFS/CWFS information, and `aos/param_sets.yaml` is generated from it.**

The reason the CWFS half cannot wait for phase 3: `param_sets.yaml` is what links the FAM
intra/extra pair to the in-focus corner-WFS member of the **same triplet** (FAM key = the
extra exposure; in-focus = FAM seq + 1). That pairing is one fact about the observation,
so it needs one home. Splitting it between `fam_tables/variants.yaml` and a phase-3
`cwfs_tables/variants.yaml` would leave the link with no single owner. The MIW is built
from the FAM tables alone, so `ts_intrinsic_wavefront` needs only the FAM half.

It lives in `fam_tables/variants.yaml` (not a new shared `products/variants.yaml`) because
the triplet is FAM-anchored; phase-3 `cwfs_tables` reads `wfs_collections` through
`fam_tables.variant_config()`. Promote to a shared observing-set file later only if
`cwfs_tables` turns out to need more than this.

Why `param_sets.yaml` must keep existing rather than be deleted, now or in phase 5:

- **An external package reads it by name from the current working directory.**
  `lsst.ts.intrinsic.wavefront.intrinsics_lib.load_param_sets()` resolves
  `Path('param_sets.yaml')` relative to the cwd (its only candidate path). Called by
  `aos/code/cwfs/run_wfs_mktable.py`, `run_wfs_fam_compare.py`,
  `run_wfs_refit_ensemble.py` and `aos/code/fam_processing/check_chunk.py`. We do not
  modify that package for this.

What the Snakefile actually takes from it — smaller than it looks. All four uses of
`PARAM_SETS`: `phrase()` (line 71), `ps_dir()` (line 92), and `wfs_variants()` /
`wfs_collection()` (lines 170–172). FAM **chunks come from `snake_config.yaml`**, not from
`param_sets.yaml`.

Two things the generator must handle, both found by reading the file rather than assumed:

1. **`wfs_collections` is not a plain name → collection map.** An entry is either a bare
   collection string or a dict carrying `seq_offset` (+1 in-focus default, 0 extra, -1
   intra), `dataset_type` (`aggregateZernikesRaw`, `aggregateAOSVisitTableAvg`) and
   `reader` (`unpaired`). Five such variants exist on `danish_1_2` alone: `refitWcs`,
   `refitWcs_2025`, `paired_3mm`, `ai_donut`, `tarts`.
2. **68% of the file is documentation.** 130 comment lines vs 55 data lines out of 191.
   `yaml.safe_load` + `safe_dump` **loses every comment**, and those comments encode real
   decisions (the TARTS Noll remap, the `ai_donut` reader path, why 2025 CWFS is a separate
   collection, why `danish_1_0` is frozen provenance). So the prose **moves into
   `variants.yaml`**, and the generated `param_sets.yaml` gets only a "GENERATED — do not
   edit" header plus data. Nothing is lost because `variants.yaml` becomes the documented
   file. (`ruamel.yaml` 0.19.1 is installed if comment-preserving round-trip is ever
   wanted; this split means it is not needed.)

Also keep in mind: `danish_1_0` stays in the generated output — it is frozen MIW
provenance, disabled in `snake_config.yaml` — while `variants.yaml` deliberately does not
*register* it as a buildable variant. The generator needs a flag for "emit but do not
register".

**Not implemented in this session.** Remaining work: migrate the 130 comment lines and the
`wfs_collections` blocks into `variants.yaml`; write the generator; regenerate
`param_sets.yaml`; verify `snakemake -n` from `aos/` gives a byte-identical job list
(baseline captured this session: **212 jobs**, with `mktable` showing no pending job).
This also subsumes the A2 `snake_config.yaml` question, since chunks can then be generated
into `snake_config.yaml` from the same source.

### A4 — blitz reference build: DONE (2026-10-10)

Built with the **unmoved** code at commit `5e793cb`, OK'd by Aaron. Output:
`/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_3_v1000/`.
Ran locally, **48 s** (predicted ~25 s; the `--fit` step is the rest). No batch job.

```bash
cd /sdf/home/r/roodman/notebooks/rubin-work/aos && \
python code/fam_processing/run_blitz_mktable.py \
  --param-set danish_1_3_v1000 \
  --day-obs 20260428 \
  --fit \
  --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_3_v1000
```

| table | rows | columns | vs existing build's 20260428 slice |
|---|---|---|---|
| `donuts.parquet` | 40,548 | 66 | rows and columns identical |
| `visits.parquet` | 12 | 23 | identical |
| `fits.parquet` | 12 | 448 | identical |

12 of 12 visits pass every quality cut; 0 of 12 flagged `bad_fit` on both `z1toz3` and
`z1toz6`. `provenance.yaml` written, recording calibration run
`LSSTCam/calib/DM-55048/intrinsicZernikes.v1.0/intrinsicsGen.20260528a`. No `donut_blur`
column, so the blur fit is skipped — same as the full build.

This is the baseline for the step-2/3 comparison of the blitz path. The step-1 reference
covers the Snakemake path; together they cover both builders.

### A4 — the original proposal, for the record

The blitz path (`run_blitz_mktable.py`, which builds both `danish_1_3` variants and writes
the combined tables directly, bypassing `mktable` and `combine_*`) is not covered by the
step-1 reference, and it moves in step 2. It needs its own before-the-move baseline.

**Proposed: `danish_1_3_v1000`, `day_obs=20260428`.** The collection
`u/jmeyers3/t614_fam_unpaired_v1000` holds 966 `donutBlitzFamResults` datasets over 15
nights; the three smallest are 20260326 (6), 20260428 (12), 20260619 (24).

| quantity | value |
|---|---|
| visits (`donutBlitzFamResults` datasets) | 12 |
| expected `donuts.parquet` rows | 40,548 |
| expected `visits.parquet` rows | 12 |
| expected `fits.parquet` rows | 12 (12 of 12 fit) |
| expected run time | ~25 s (2.0 s/visit) |

Runtime per visit is from the full v1000 build log
`aos/logs/blitz_mktable_v1000_20261005_000032.log`: 966 visits in ~33 min. Expected row
counts are the per-night slices of the existing `danish_1_3_v1000` build.

**Not 20260326**, the smallest night: 6 visits but only **1** produces a `fits.parquet`
row, so the fit and combine steps would barely be exercised. 20260428 fits all 12 for
~13 s more.

**No batch job** — ~25 s runs locally.

```bash
cd /sdf/home/r/roodman/notebooks/rubin-work/aos && \
python code/fam_processing/run_blitz_mktable.py \
  --param-set danish_1_3_v1000 \
  --day-obs 20260428 \
  --fit \
  --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_3_v1000
```

`--fit` is required: without it the run stops after donuts/visits and writes no
`fits.parquet`, leaving the fit path untested. All flags verified against the script's
argument parser. Command, commit, and the actual row/column counts go here once run.

### Step 4 requirements, to carry forward

Two things the manifests and READMEs must say when the existing builds are registered:

- **`danish_1_2` is a composite variant.** Its chunks use different Danish and wep
  versions and **two different Butler repos** (`/repo/main`, plus `/repo/embargo` for the
  20260713 chunk), so a single `collections` field in the manifest would be wrong. Its
  manifest **lists collections per chunk**. See the per-chunk `collection`, `phrase`,
  `butler_repo` and `programs` overrides in `variants.yaml`.
- **A rebuild of `danish_1_2` loses 362 columns.** The `20260920` `fits.parquet` carries
  362 `cam_*` columns that a fresh `combine_fits` drops (the combined file was the only
  place they lived — see the sidecar constraint above). Its README **must say** that
  rebuilding loses them until the sidecar is reworked in phase 3.
- The step-4 copy of `danish_1_2` writes the chunk as **`20260418_20260513`** (the
  narrowed A2 name) in the new tree. The old tree keeps `20260418_20260531`; the two
  differ by directory name only, over the same 219 visits.

## Open, not yet resolved

- The reference covers **one chunk of ten, one variant of three**. It does not exercise the
  blitz recast path (`run_blitz_mktable.py`, which builds both `danish_1_3` variants and
  writes the combined tables directly, bypassing `mktable` and `combine_*`), and
  `combine_*` only trivially, with a single chunk.
- `backfill_visit_sides.py` is a one-off migration that **mutates a product table in
  place**, which rule 5 of the plan's section 2 forbids. Moved with the builders for now;
  decide in phase 3 whether it becomes a build step or is retired.
- The sidecar/EFD review item is recorded in the plan's section 10 (carried-over open
  items), per Aaron on 2026-10-10: the MIW needs only elevation, camera rotator angle and
  filter band, all ConsDB; the sidecar's eight EFD groups exist for downstream studies that
  could read ConsDB and the value-added DuckDB instead.

## Spend note

Checked 2026-10-10: $695.44 of $2000 (34.8%), resets 2026-11-01. Pace $83.42/day would
reach ~125% of budget; about $60/day lands under. Keep reads narrow (`rg` first, `Read`
with `offset`/`limit`, never a whole `.ipynb`), and clear the session at these step
boundaries rather than carrying the transcript.
