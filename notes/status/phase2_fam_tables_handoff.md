# Phase 2 handoff: the fam_tables pilot product

> **Status:** complete · **Last updated:** 2026-10-10 · **Kind:** working state (handoff)

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
| `ed3417f` | **A3** implemented: `aos/param_sets.yaml` generated from `variants.yaml` |
| `26dd61c` | **A4** blitz reference built |
| `7137ea3` | **B0**: the old `aos/` tree keeps the old chunk range; test allows that one difference |
| `06039ab` | **Step 2**: the four builders and the six build rules moved into the product |
| `0781229` | **Step 3**: both reference builds reproduced exactly with the moved code |
| `f33b0ab` | **Step 5a**: catalog skips incomplete builds; `current` moves only by hand |
| `2b50ed0` | **Step 5b**: both builders write `manifest.json` as their last step |
| `04a8871` | **Step 5c**: both references rerun with the final code, still exact |

**Test status: 64 pass under `rubinwork/products/`** (28 fam_tables, 36 catalog/manifest).
The import smoke test is 63 of 63 modules ok (one fewer than phase 1's 64 because
`blitz_reader` became an intra-package import).

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

### Step 5's reader half, done ahead of its turn in step 2

`reader.py` gives `load(variant, build="current")` returning `(donuts, visits, fits)`,
plus `load_table`, `variants`, `variant_config` and — added in step 2 —
`variant_for_param_set` and `build_dir`. The **build** code now goes through the catalog:
the product Snakefile and `run_blitz_mktable`'s `--out-dir` default both resolve paths
through `build_dir`. What remains for step 5 is switching the repo's **readers** (the
study code that opens `output/fam_processing/<P>/...` by path) to `load()`.

## Next concrete action

**Phase 2 is complete.** Nothing is left in it. Continue in
`notes/status/phase3_products_handoff.md`, whose next action is phase 3a, `cwfs_tables`.

**Step 4 was dropped on 2026-10-10** by the revised route: no existing build is copied or
registered, every product is rebuilt fresh in phase 3c. The "Step 4 requirements" section
below is kept because two of its three points are **3c** requirements now — the composite
`danish_1_2` manifest (done: `write_manifest` records the per-chunk collections) and the
362 lost `cam_*` columns (still open). The copy budget in "Decisions taken" no longer
applies.

### Step 5 — manifests and the catalog rule: DONE

- **Both builders write `manifest.json` last.** New
  `rubinwork/products/fam_tables/write_manifest.py`, shared by the product Snakefile's
  final `manifest` rule and by `run_blitz_mktable`. `config` is the expanded
  `variants.yaml` entry, so `danish_1_2` records its **per-chunk** collections, Butler
  repos and program filters; row counts are read back off the parquet files, not taken
  from a builder's counter. Without `--fit` the blitz build gets status `tables-only` and
  the catalog skips it.
- **The catalog sees only complete builds.** No manifest, an unparseable manifest, or
  `status != "complete"` means the build is skipped when `current` is resolved and left
  out of `list_products()`; asking for it by name raises `ProductNotFound` saying which of
  those it was. A `current` symlink left pointing at a build that later failed is refused
  too.
- **No build moves `current`.** `manifest.write()` and `catalog.register()` no longer
  touch it. `python -m rubinwork.products.catalog set-current <product> <variant> <build>`
  does, and refuses a build that is not complete (exit status 1).
- **Both references rerun with the final code, both still exact at zero tolerance.** New
  `refbuild/stage_reference_chunk.py` makes the Snakemake-path rerun reproducible: it
  copies the reference chunk's `mktable` output and reconstructs the pre-merge
  `visits.parquet` (53 rows x 228 columns, down from 600, by dropping the merged sidecar
  while keeping the 13 columns `mktable` itself writes). The Snakefile gained
  `--config chunks=<dmin>_<dmax>` to build one chunk of ten.

  | path | build | donuts | visits | fits | differing columns |
  |---|---|---|---|---|---|
  | Snakemake | `_scratch_step5/products/fam_tables/danish_1_2/20261010` | 145,264 rows x 48 cols | 53 x 600 | 47 x 653 | **0 / 0 / 0** |
  | blitz | `_scratch_step5/danish_1_3_v1000_final` | 40,548 x 66 | 12 x 23 | 12 x 448 | **0 / 0 / 0** |

  Both manifests written, `status: complete`, `git_commit 2b50ed0`. The Snakemake one
  carries all ten chunks in `config.chunks` including the `/repo/embargo` override, and
  `build_options.chunks_built = 20251116_20251130` recording that this build is one chunk
  of ten. Tests: 64 under `rubinwork/products/`; import smoke test 63 of 63.
- **What is left for phase 5, as before:** switching the repo's *readers* — the study code
  that opens `output/fam_processing/<P>/...` by path — to `load()`.

The build directory layout step 2 settled:

```
products/fam_tables/<variant>/<build>/{donuts,visits,fits}.parquet   the tables
products/fam_tables/<variant>/<build>/manifest.json
products/fam_tables/<variant>/<build>/_work/chunks/<dmin>_<dmax>/    intermediates
products/fam_tables/<variant>/<build>/_work/logs/
```

`_work/` holds the per-chunk tables and the telemetry sidecars. `catalog._names()` hides
any entry starting with `_`, so it does not appear as a build or a variant.

### Step 2 — the move: DONE (commit `06039ab`)

The four builders moved by `git mv` into `rubinwork/products/fam_tables/builders/`, with
shims at the four old `aos/code/fam_processing/` paths. Each shim binds the real module's
namespace and sets `sys.modules[__name__] = _real`, so a bare-name import after a
`sys.path.insert` gives the **same module object**, not a copy — verified for all four.
The three runnable ones also still work when invoked by their old path.

Six Snakefile rules (`mktable`, `fit`, `combine_donuts`, `combine_fits`, `combine_visits`,
`attach_telemetry`) moved into **`rubinwork/products/fam_tables/Snakefile`**, which reads
`variants.yaml` and writes under the catalog's products root. It builds **one variant and
one build at a time**:

```bash
snakemake -s rubinwork/products/fam_tables/Snakefile -n \
    --config variant=danish_1_2 build=20261010
```

Run it from the **repository root**, because `mktable` runs with `aos/` as its cwd (see
below).

Four things found or decided during the move, each of which would otherwise be
rediscovered:

1. **`mktable` must run with `aos/` as its cwd.** `run_mktable.py --param-set` reaches the
   configuration through `intrinsics_lib.load_param_sets()`, which opens
   `Path('param_sets.yaml')` relative to the **current working directory** and has no
   other candidate path. So the product's `mktable` rule does
   `cd <repo>/aos && python $WF_BIN/run_mktable.py ...`. That is not a workaround to
   remove: the external package owns that lookup, and `aos/param_sets.yaml` is generated
   from `variants.yaml` (A3), so the two cannot disagree.
2. **Snakemake's own parser mishandles a multi-line f-string inside a rule.** Writing the
   `shell:` body as a parenthesized multi-line f-string raises
   `UnboundLocalError: cannot access local variable 't1'` from `snakemake/parser.py`. The
   product Snakefile therefore builds each shell command into a module-level variable
   (`MKTABLE_CMD`, `FIT_CMD`, `COMBINE_CMD`, `ATTACH_CMD`) and the rules say
   `shell: MKTABLE_CMD`. Do not "tidy" these back inline.
3. **`--config build=20261010` arrives as an `int`.** Snakemake parses a bare digit string
   as a number, so `BUILD` is wrapped in `str()`; without it `build_dir` raises
   `unsupported operand type(s) for /: 'PosixPath' and 'int'`.
4. **`run_attach_telemetry` needed a new `--chunks-dir`.** It located the per-chunk
   directories as `<out-dir>/chunks`, which forced the intermediates to sit beside the
   tables. The flag defaults to exactly that, so existing behavior is unchanged, and the
   product Snakefile passes `--chunks-dir <build>/_work/chunks`.

Two small additions to the product's reader, both needed by the moved builder:

- **`variant_for_param_set(name)`** accepts either the long `param_set` key or the short
  variant name and returns the variant. `run_blitz_mktable --param-set` historically takes
  the long key, and the two differ for `danish_1_2`.
- **`build_dir(variant, build)`** gives the write path for a build whether or not it
  exists; `catalog.path()` resolves only builds already on disk.

`run_blitz_mktable`'s `--out-dir` default is now that catalog build directory, with a new
`--build` flag naming the build (default: today's UTC date).

**What `aos/Snakefile` lost, and what it kept.** The six rules and the FAM-table targets
in `rule all` are gone. The surviving study rules read the FAM tables **at the paths they
always did** — `output/fam_processing/<P>/{donuts,visits,fits}.parquet` in the old tree —
so those three files are now inputs that Snakefile does not build. The config-reading
helpers (`chunks`, `chunk_files`, `coord`, `_chunk_entries`) stay: `rule residual_movies`
and others still use them. `ATTACH_TELEMETRY` is gone, since its only rule left;
`run_snake.sh --mode batch` still passes `--config attach_telemetry=0`, which snakemake
accepts and nothing reads.

**Verification.** `snakemake -n` from `aos/` went **212 -> 199 jobs**, and the drop is
exactly the four moved rules that had pending work: `fit` 10, `combine_donuts` 1,
`combine_fits` 1, `attach_telemetry` 1 (13 jobs, counts dimensionless). Every surviving
rule's count is identical, `diff` on the sorted job-stats table showing only those four
lines and the total. The product Snakefile plans **25 jobs** for `danish_1_2` — `mktable`
10, `fit` 10, `combine_*` 3, `attach_telemetry` 1, `all` 1 — over its ten chunks including
the A2-narrowed `20260418_20260513`, and raises a `ValueError` naming
`run_blitz_mktable` if asked for a `blitz` variant. The `mktable` commands it issues are
identical to the old ones apart from `--output-dir`: same `--param-set`, `--workers 8`,
`--overwrite` and the same per-chunk `--collections` / `--collection-phrase` /
`--programs` overrides.

### Step 3 — comparison against both references: DONE, exact match

Run with the moved code at commit `06039ab`. `compare_builds.py` runs at **zero
tolerance** (`rtol=0.0, atol=0.0`), so "identical" means bit-identical.

**Snakemake path**, reference `_phase2_refbuild/danish_1_2/` (chunk `20251116_20251130`).
`fit` + `combine_donuts` + `combine_fits` + `combine_visits` + `attach_telemetry` rerun
from the reference's existing chunk donuts; `mktable` deliberately **not** rerun. `fit`
28.7 s, `attach_telemetry` 2 min 39 s. Output under
`_phase2_refbuild/_scratch/products/fam_tables/danish_1_2/moved/`.

| table | rows | columns compared | columns only in one side | differing columns |
|---|---|---|---|---|
| `donuts.parquet` | 145,264 | 48 | 0 | **0** |
| `visits.parquet` | 53 | 600 | 0 | **0** |
| `fits.parquet` | 47 | 653 | 0 | **0** |

**Blitz path**, reference `_phase2_refbuild/danish_1_3_v1000/` (`day_obs=20260428`). The
whole A4 command rerun with the moved builder, 43 s (reference 48 s). Output
`_phase2_refbuild/danish_1_3_v1000_moved/`.

| table | rows | columns compared | columns only in one side | differing columns |
|---|---|---|---|---|
| `donuts.parquet` | 40,548 | 66 | 0 | **0** |
| `visits.parquet` | 12 | 23 | 0 | **0** |
| `fits.parquet` | 12 | 448 | 0 | **0** |

**Nothing differs — not one column in either build.** The two differences the step-2/3
plan predicted did not appear, and the reason matters for reading any future comparison:

- **Telemetry fetched at a different time came back the same.** The EFD and ConsDB queries
  are window means over fixed exposure windows on nights eight months past, so the values
  are settled; a later fetch returns the same numbers. All 385 telemetry columns over 53
  visits are bit-identical.
- **`rotator_angle` could not differ, because `mktable` was not rerun.** It is written by
  `mktable` (confirmed: `rotator_angle` is in the staged pre-merge chunk `visits.parquet`
  and **not** in `telemetry.parquet`), and the A1 non-reproducibility is in `mktable`'s
  EFD window-mean fallback on the 19 visits where ConsDB `physical_rotator_angle` is NULL.
  Step 3 reuses the reference's `mktable` output, so that code path never ran twice. A
  future comparison that **does** rerun `mktable` will see those 19 visits differ by up to
  0.188 deg of camera rotator angle until the `rotTelPos` fix lands.

**How the pre-merge chunk `visits.parquet` was reconstructed**, since this is not obvious
and a future rerun needs it. The reference's chunk `visits.parquet` was merged **in place**
by its own `attach_telemetry` (mtime 00:47:50, after `fits.parquet` at 00:45:10), so it is
not the file the reference's `fit` saw. Reconstruction: drop the sidecar columns the merge
added, but **keep the 13 that `mktable` itself writes** — the ones that also appear in the
pre-merge `fits.parquet` (`cam_air_temp`, `cam_m1m3_delta_t`, `dome_delta_t`,
`m1m3_air_temp`, `m2_air_temp`, `m2_delta_t`, `outside_temp` in deg C;
`x_gradient`, `y_gradient`, `z_gradient`, `radial_gradient` in deg C/m;
`tma_truss_temp_mxmy`, `tma_truss_temp_pxpy` in deg C). Those 13 were verified
bit-identical between the pre-merge `fits.parquet` and the post-merge `visits.parquet`
(max |difference| = 0.000e+00 on all 13, n=47 joined rows), so the merge did not change
them and dropping them would have been wrong. Result: 53 rows x 228 columns, down from 600.

**Leftover to clean up when convenient:** `_phase2_refbuild/danish_1_2_moved/` is an empty
directory from staging (`_work/logs/` only); the real output went to `_scratch/`. Not
deleted — file deletion needs Aaron's OK.

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

**Superseded by B0 (commit `7137ea3`).** A2 had also narrowed `aos/snake_config.yaml`,
which renamed the chunk directory the old tree already built and made `snakemake -n` in
`aos/` ask to rebuild those **219 visits through `mktable`** (212 → 214 jobs, the extra
being that `mktable` plus a `combine_visits`). Since the old study rules stay in use until
phase 5, any routine run would have triggered it.

B0 settled it: **`aos/snake_config.yaml` keeps `[20260418, 20260531]`** — the name of the
directory the old tree holds — and the narrowed `[20260418, 20260513]` lives only in
`variants.yaml`, which the product Snakefile uses. `test_chunks_match_snake_config` allows
exactly this one difference, with a comment naming it; the lists are otherwise compared
element by element, so a second divergence still fails the test. Both ranges cover the same
219 visits (max day_obs 20260513). With that, `snakemake -n` in `aos/` is back to 212 jobs
with **no `mktable` job**.

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

- **The blitz path IS covered** — A4 built its own reference (`danish_1_3_v1000`,
  `day_obs=20260428`) and step 3 reproduced it exactly, so the two references together
  cover both builders. What remains thin: the Snakemake reference is **one chunk of ten**,
  so `combine_*` is exercised only trivially, over a single chunk, and `mktable` was never
  rerun. The first real multi-chunk exercise is step 4's registration, or the first full
  `danish_1_2` build in the new tree.
- `backfill_visit_sides.py` is a one-off migration that **mutates a product table in
  place**, which rule 5 of the plan's section 2 forbids. Moved with the builders for now;
  decide in phase 3 whether it becomes a build step or is retired. It still targets the
  **old** `aos/output/fam_processing/<dir_name>/visits.parquet`, deliberately, since that
  is the tree it was written to repair.
- **The five `wfs_collections` entries in `variants.yaml` become `cwfs_tables` variants in
  phase 3.** They live in `fam_tables/variants.yaml` today because the FAM/CWFS triplet
  link needs one home (A3), and phase-3 `cwfs_tables` reads them through
  `fam_tables.variant_config()`. Deciding then whether they are promoted into a
  `cwfs_tables/variants.yaml` of their own, or stay here and are only *read* from there, is
  part of phase 3 — the triplet link must keep a single owner either way.
- The sidecar/EFD review item is recorded in the plan's section 10 (carried-over open
  items), per Aaron on 2026-10-10: the MIW needs only elevation, camera rotator angle and
  filter band, all ConsDB; the sidecar's eight EFD groups exist for downstream studies that
  could read ConsDB and the value-added DuckDB instead.

## Spend note

Checked 2026-10-10: $695.44 of $2000 (34.8%), resets 2026-11-01. Pace $83.42/day would
reach ~125% of budget; about $60/day lands under. Keep reads narrow (`rg` first, `Read`
with `offset`/`limit`, never a whole `.ipynb`), and clear the session at these step
boundaries rather than carrying the transcript.
