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

Working tree clean at `b355446`. Not yet pushed.

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
