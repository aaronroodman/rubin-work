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
| `71a345f` | 3a step 1: the `cwfs_tables` reference build with the unmoved code |
| `b880f2d` | 3a step 2: the move — builder, shim, `variants.yaml`, product Snakefile, tests |
| `160f8b7` | 3a steps 3 and 4 for `cwfs_tables`: verification and the zero-tolerance comparison |
| `34b9980` | `aos/Snakefile` reports the CWFS variants `wfs_variants()` skips |
| `d81cc3d` | 3a step 1: the `miw` reference build with the unmoved code |
| `b9756aa` | 3a step 2: the `miw` move — five rules, `variants.yaml` + generated `mi_config.yaml`, both readers, external registrations, tests |
| `48258b2` | 3a steps 3 and 4 for `miw`: the verification and the zero-tolerance comparison |
| `fec3741` | the resolver bridging the `aos/` study rules to the catalog |
| `3f607ab` | 3a step 1: the `coadds` reference build with the unmoved code |
| `d0e4182` | 3a step 2: the `coadds` move — builder, shim, `variants.yaml`, product Snakefile, two-upstream manifest, npz comparison, tests |
| `3d4f993` | 3a steps 3 and 4 for `coadds`: the verification and the zero-tolerance comparison |
| `ceca01e` | the coadd study scripts resolve the old tree before the catalog, as `aos/Snakefile` does |
| `d5eeba6` | `dof_lut`'s plain-FAM-DZ projection recorded as under review (Aaron's call) |
| (this file) | 3a step 1 for `dof_lut` and `bounce_tables`: both reference builds, scope and variants |

**Phase 3a, `cwfs_tables`, is DONE.** The moved code reproduces the reference exactly.

**Phase 3a, `miw`, is DONE.** The moved code reproduces the reference exactly — 0 differing
columns across 12 tables at zero tolerance.

**Phase 3a, `coadds`, is DONE.** The moved code reproduces the reference exactly — 0
differing columns across 2 tables (26 and 81 columns) and 0 differing arrays across 2 npz
files (42 and 6 arrays), at zero tolerance.

**The study rules now read the catalog too** (`fec3741`), so a product rebuilt into the new
tree is visible to them before phase 5.

## Next concrete action

**Phase 3a step 2 for `dof_lut` and `bounce_tables`: the move.** Step 1 is done — both
reference builds exist, both are the full production run, and every comparison key is
verified unique. Scope, variants and the three decisions Aaron made on 2026-10-11 are in
the `dof_lut` and `bounce_tables` section below; read it before moving anything, since two
of them supersede the plan.

What the move commit has to do, beyond the mechanical pattern the three earlier products
set:

1. **Split `run_bounce.py` at `:700`.** Everything to `:700` plus the four `to_parquet`
   calls is the builder; the nine `PdfPages` blocks stay as the study. The table data all
   comes from the `:479`-`:700` analysis block, so no physics moves.
2. **`rubinwork.bounce`, a new library**, so the four `smatrix` imports stop reaching into
   `aos/code/bounce/`. Three names are all they need. Fix `bounce_lib.rbr_module()`'s
   `sys.path` insert in the same commit.
3. **`bounce` keeps a rule in `aos/Snakefile`** — the plotting half. This is the first
   product whose rule survives the move, so the −3 has to be accounted against a rule that
   still exists.
4. **Declare every output.** `build_lut` writes 5 files against 3 declared, `bounce` 13
   against 2.
5. `bounce_tables/variants.yaml` must keep resolving the live `bounce:` override at
   `aos/analysis_config.yaml:180`, which is keyed on the long param_set and redefines the
   T720 legs as a five-point sweep.

Then steps 3 and 4 per product, as `coadds` did: `snakemake -n` with every job-count
change explained (expect 37 → 31), the catalog bridge extended to these products' study
readers, the import smoke test, all product tests, then the zero-tolerance comparison.

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

### 3a step 2 — the move: DONE (commit `b880f2d`)

`run_wfs_mktable.py` moved by `git mv` into
`rubinwork/products/cwfs_tables/builders/`, with a shim at
`aos/code/cwfs/run_wfs_mktable.py` that binds the real module's namespace and sets
`sys.modules[__name__] = _real`. Verified: a bare-name `import run_wfs_mktable` after a
`sys.path.insert` of `aos/code/cwfs` gives the **same module object**, and all seven
checked attributes (`main`, `get_aidonut_zernikes`, `get_unpaired_zernikes`, `_fam_noll`,
`_validation_plot`, `_wfs_shell`, `AIDONUT_SW0`) are the same objects, not copies.

New in the product: `__init__.py` (with `load`), `reader.py`, `variants.yaml`,
`Snakefile`, `write_manifest.py`, `compare_builds.py`, `tests/test_cwfs_tables.py` (26
tests) and `refbuild/stage_fam_night.py`.

**The one builder change, and why it is not a results change.** The builder gained a
`--variant` flag that resolves a `cwfs_tables` variant to its `(param_set, wfs_name)`
pair; the original `--param-set` plus `--wfs-name` route is untouched and still reads the
configuration through `intrinsics_lib.load_param_sets()`. Both routes end at the same
`pset['wfs_collections'][name]` lookup, so they cannot disagree — and step 4's comparison
is the unmoved `--param-set` route against the moved `--variant` route, which is the
stronger check.

Four things found or decided during the move:

1. **`wfs_mktable` must run with `aos/` as its cwd**, for the same reason `fam_tables`'
   `mktable` does: the external `intrinsics_lib.load_param_sets()` opens
   `Path('param_sets.yaml')` relative to the working directory. The product Snakefile does
   `cd <repo>/aos && python -m ...`.
2. **`reader` must be written before `dataset_type`** in a generated `wfs_collections`
   entry. `BUILDER_FIELDS` in `cwfs_tables/reader.py` fixes that order; with
   `dataset_type` first, `aos/param_sets.yaml` differed by the two swapped lines of the
   `tarts` entry and byte-identity failed.
3. **An entry with only a collection must be emitted as a bare string**, not a
   one-key dict, which is the shape `aos/param_sets.yaml` has today.
   `_builder_entry()` does that.
4. **`aos/Snakefile`'s `wfs_variants()` had to start filtering on what is on disk.** See
   step 3 — this is the one non-obvious consequence of the move.

The product Snakefile builds one variant and one build at a time, from the repository
root:

```bash
snakemake -s rubinwork/products/cwfs_tables/Snakefile -n \
    --config variant=d12-refitWcs build=20261010 fam_build=20260920
```

It takes `fam_build` (default `current`) and resolves it through the catalog, recording
the real build name under the manifest's `inputs` as
`{"fam_tables": "danish_1_2@20260920"}`. `manifest` depends on both tables **and** the
validation PDF, so an interrupted build leaves no manifest and the catalog does not see
it. A `registered: false` variant raises before any path resolves, naming the reason.

### 3a step 3 — verification: DONE

**`aos/param_sets.yaml` is byte-identical.** md5 `0f72da651cf9319cc283754c614051fd` before
and after; `gen_param_sets --check` passes, a byte `diff` against the pre-move copy is
empty, and `git diff 9392f9b -- aos/param_sets.yaml` is empty.

**`snakemake -n` in `aos/`: 199 -> 189 jobs, a drop of 10 — not 5.** The moved rule
accounts for 5; the other 5 are a real consequence that the next product's move will meet
again:

| rule | before | after | change | why |
|---|---|---|---|---|
| `wfs_mktable` | 5 | 0 | **-5** | moved to the product |
| `wfs_corner_compare` | 5 | 4 | -1 | `refitWcs_2025` lost its producer |
| `wfs_intrinsic_sidecar` | 10 | 8 | -2 | same |
| `wfs_dof_compare` | 10 | 8 | -2 | same |
| every other rule | | | **0** | unchanged |
| **total** | **199** | **189** | **-10** | |

All counts are jobs (dimensionless). **`refitWcs_2025` is the one CWFS variant with no
old-tree build** — `aos/output/wfs_ingest/danish_1_2/` holds `refitWcs`, `paired_3mm`,
`ai_donut` and `tarts`, but not it. Before the move, `wfs_mktable` was its producer, so
its four downstream study jobs were plannable. After the move there is no producer in
`aos/`, and a plain `snakemake -n` aborted the whole DAG with `MissingInputException`
rather than listing jobs.

So `wfs_variants()` now enumerates only the variants whose `donuts.parquet` exists. The
four built variants are completely unaffected — post-move they still plan 16, 16, 16 and
20 `wfs_corner_compare`/`wfs_dof_compare` output paths respectively — and zero
`refitWcs_2025` jobs remain. Building that variant with the product Snakefile makes its
study jobs reappear. **This is the step-3 criterion failing as literally written** ("the
same job list apart from the moved rule's jobs"), and it is reported rather than hidden:
the extra -5 is one never-built variant, enumerated exactly, not a side effect on anything
that was working.

**The product Snakefile plans the expected jobs**: 3 for one variant — `wfs_mktable` 1,
`manifest` 1, `all` 1 — verified against a staged `fam_tables` build in a scratch data
root (`_phase2_refbuild/_scratch_3a/`), never the real products tree. The `mktable`
command it issues is the moved builder with `--variant`, `--coord-sys OCS`,
`--dz-prefix z1toz6`, `--tables-dir <fam build>` and `--out-dir <cwfs build>`.

**Import smoke test: no regressions.** 63 of 63 modules import before and after
(`rubinwork/common/scripts/import_smoke_test.py --compare`), "all 63 module import
outcomes identical".

**Tests: 90 passed** across `rubinwork/products/` — the 26 new `cwfs_tables` tests plus
all 64 pre-existing `fam_tables` and `catalog` tests, none of which needed changing.
`test_generated_param_sets_carry_the_wfs_half` in `test_fam_tables.py` still passes
untouched, because it asserts on the generated output rather than on where the data lives.

### 3a step 4 — the comparison: DONE, exact match

The reference rerun with the moved code at commit `b880f2d`, **51.0 s wall clock**
(reference 65.3 s), same 6 in-focus exposures, same 171 donut rows, same 0 FAM visits
missing a CWFS table, and the same fitted validation offsets to the printed precision.
Output in `_phase2_refbuild/cwfs_tables/refitWcs_2025_20251126_moved/`.

`python -m rubinwork.products.cwfs_tables.compare_builds` at **zero tolerance**
(`rtol=0.0, atol=0.0`):

| table | rows | columns compared | columns only in one side | differing columns |
|---|---|---|---|---|
| `donuts.parquet` | 171 | 20 | 0 | **0** |
| `visits.parquet` | 6 | 8 | 0 | **0** |

The key `(day_obs, seq_num, detector, thx_OCS, thy_OCS)` is unique on both sides (171 of
171 rows), so the alignment is sound rather than accidentally agreeing. The validation PDF
came out at 106,605 bytes on both sides (0.00% difference), though it is deliberately not
byte-compared.

**Nothing differs — not one column in either table.** `RESULT: builds match`, exit 0.

## 3a, `miw` — the move

### Scope, agreed 2026-10-10

Moves **five Snakefile rules**, the measured-intrinsic configuration, and the two readers.
No builder code moves: the build runners are external, in `ts_intrinsic_wavefront/bin/`,
and were not touched.

| rule | jobs before | runner |
|---|---|---|
| `build_intrinsic` | 36 | `$WF_BIN/run_build_intrinsic.py` |
| `intrinsic_split` | 9 | `$WF_BIN/run_intrinsic_split.py` |
| `intrinsic_sidecar` | 12 | `$WF_BIN/run_make_intrinsic_sidecar.py` |
| `wfs_intrinsic_sidecar` | 8 | same, `--wfs-corner-height` |
| `refit_mi` | 12 | `$WF_BIN/run_dz_fit.py` |

All counts are jobs (dimensionless). **`refit_mi` belongs to `miw`**, Aaron's call
2026-10-10: its `fits.parquet` is a function of the MIW variant and nothing downstream
re-derives it, `fam_tables` already owns the plain `fit` rule calling the same
`run_dz_fit.py` without a sidecar, and nine `aos/` study rules read
`output/miw/<d>/fits.parquet` as product data. Owning it makes a `miw` build
self-contained: grids -> split -> sidecar -> refit, manifest last.

Everything in `aos/code/miw/` is **study** code and stayed: `compare_build_dof`,
`compare_miw_versions`, `compare_pupil_models`, `compare_rbr_arms`, `compare_schemes`,
`run_study_radialbins`, `check_dof_ranges`, `test_rbr_against_prototype`.

### `aos/mi_config.yaml` is now GENERATED, and its body is byte-identical

The same situation as `param_sets.yaml`:
`lsst.ts.intrinsic.wavefront.mi_config.default_config_path()` returns
`Path('mi_config.yaml')` relative to the working directory with no other candidate, and
**ten** things read it through that default — the three build runners plus
`aos/code/{coadd/run_coadd_blocks_miw, coadd/recompute_coadd_metrics, bounce/run_bounce,
lut/run_build_lut, miw/run_study_radialbins, correlations/run_dz_correlations,
output_paths}.py`. So it cannot be retired and must not change.

**The configuration is held in `variants.yaml` as literal text blocks, and the generator
concatenates them.** This is the one real design decision of the move, and it was reached
after both parsing routes failed:

- `yaml.safe_dump`, which `fam_tables.gen_param_sets` uses, **discards comments**.
  `mi_config.yaml` is hand-written and carries comments *inside* entries — the
  `day_obs_max` freeze note, the Range-Bounded Recovery rationale blocks, the per-param_set
  preambles — so this loses load-bearing prose.
- a **`ruamel.yaml` round-trip preserves comments but moves them** once the structure is
  reshaped. Measured: a straight round-trip of the file reproduces it byte-identically with
  `indent(mapping=2, sequence=4, offset=2)`, `width=70`, an explicit `null` representer and
  two fixups. But regrouping the entries from `param_set -> [entries]` into
  `variant -> entry` scatters them, because the tokens attach to **key names and sequence
  indices**: the commented-out `danish_1_0` block hangs off the `measured_intrinsics` key,
  the per-param_set preambles off the previous group's last entry, and the 22/12 note off
  sequence index 5. Reshaping drops or relocates all of them.

Holding each block as a `text:` scalar makes byte-identity true **by construction** and
keeps every comment where its author put it. What the generator still *checks*, rather than
merely copies: each entry's `variant`, `mi_name` and `dir_name` index fields must agree
with the `name:` and `dir_name:` in its own text block, and `variant` must be the group's
`short_code` plus the `dir_name`. A block edited without its index fields fails loudly.

Verified three ways:

- the generated **body** (everything from `defaults:` on) is byte-identical to the
  pre-move file, committed as `rubinwork/products/miw/tests/mi_config_premove_body.yaml`
  (13,197 bytes, md5 `ee466552be72a16cc312388fcb044f76`).
- the **resolved** config `load_mi_config` hands each runner is identical for **all 12**
  `(param_set, mi_name)` pairs, compared as canonical JSON. That is what the rules hash as
  their rerun trigger, so no cached output is invalidated.
- the whole-file md5 is `e0369b212b64057dec9aaa5997a679bb`, against the hand-written
  `580fcb974be53e3d43dafd401fde2637`. **The two differ only in the header**: the generated
  file replaces the hand-written preamble with the DO-NOT-EDIT notice, and that preamble's
  content moved into `variants.yaml`. The external package parses the body and ignores
  comments.

`aos/param_sets.yaml` is untouched: md5 `0f72da651cf9319cc283754c614051fd` before and
after. `fam_tables/variants.yaml` gained a `short_code` field (`d10`, `d12`, `d13t`,
`d13v1000`), which `gen_param_sets`'s `PASSTHROUGH` whitelist drops.

### Variants: twelve, named `<short_code>-<dir_name>`

| variant | param_set | mi_name | note |
|---|---|---|---|
| `d12-A_50_34_i` | `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` | `pathA_50_34_i` | `day_obs_max: 20260513` |
| `d12-A_50_34_i_5rot` | same | `pathA_50_34_i_5rot` | `build_from` |
| `d13t-A_50_34_i` | `danish_1_3_test` | `pathA_50_34_i` | |
| `d13t-A_50_34_i_5rot` | same | `pathA_50_34_i_5rot` | `build_from` |
| `d13v1000-A_50_34_i` | `danish_1_3_v1000` | `pathA_50_34_i` | |
| `d13v1000-A_50_34_i_rbr` | same | `pathA_50_34_i_rbr` | superseded, kept |
| `d13v1000-A_50_50_i_rbr` | same | `pathA_50_50_i_rbr` | |
| `d13v1000-A_50_34_i_5rot` | same | `pathA_50_34_i_5rot` | `build_from` |
| `d13v1000-A_50_50_i_rbr_5rot` | same | `pathA_50_50_i_rbr_5rot` | `build_from` |
| `d13v1000-A_22_12_i` | same | `pathA_22_12_i` | explicit 22-DOF list |
| `d13v1000-A_22_12_i_5rot` | same | `pathA_22_12_i_5rot` | `build_from` |
| `d13v1000-A_50_34_i_rbr_5rot` | same | `pathA_50_34_i_rbr_5rot` | `build_from` |

`build_from` names a **product variant** in `variants.yaml`; the generator writes the
parent's `mi_name`, which is what `--mi-name` expects. `reader.build_source()` resolves it
back and the tests assert it never crosses a `fam_variant`.

**Two names exceed the plan's 24-character variant limit** —
`d13v1000-A_50_34_i_rbr_5rot` and `d13v1000-A_50_50_i_rbr_5rot` are both 27. The limit in
section 5 of the plan is raised to 32 rather than mangling names that have to round-trip to
`dir_name`.

### The official MIW: three external variants, manifests only

Registered with `catalog.register(..., location=...)`, no files, build name `external`
(an external variant has one entry, not dated builds; a new release is a new variant).
Probed 2026-10-10 in `/repo/main`:

| variant | location | datasets |
|---|---|---|
| `official-20260528a` | `LSSTCam/calib/DM-55048/intrinsicZernikes.v1.0/intrinsicsGen.20260528a` | 1,230 `intrinsicZernikes` over bands u g r i z y |
| `official-gmegias-v3` | `u/gmegias/calib/DM-55048/intrinsicZernikes.v3` | 1,230, the same set |
| `staged-v1` | `aos/calibration/miw/intrinsic_split_maps_v1.parquet` | — |

All counts are datasets (dimensionless), 205 detectors x 6 bands. `official-gmegias-v3` is
the collection five `optatmo/code/` scripts read by default. `staged-v1` carries its
`provenance.yaml`'s `param_set`, `mi_name` and `git_sha` in the manifest `config`, and a
test asserts the three still agree with the YAML. **`aos/calibration/` is not touched** —
the manifest points at it. Nothing the official MIW needs comes from this repository.

### Readers: one product reader, one library

| old path | now | kind |
|---|---|---|
| `aos/code/miw_io.py` | `rubinwork/products/miw/reader.py` | **product reader** — takes a path, returns `(pts, zk, df)` from an `intrinsic_split_maps` parquet. `load_miw` kept as an alias of `load_maps`. |
| `aos/code/miw_corner_intrinsic.py` | `rubinwork/miw_corner.py` | **library** — evaluates a decomposition it is handed at the `ts_ofc` corner sample points, writes nothing, needs `lsst.ts.ofc` and `batoid_rubin`. A library may not import a product, so `MiwCornerLookup` takes `path=`. |

Shims at both old paths bind the real module's namespace and set
`sys.modules[__name__] = _real`. Verified: a bare-name import after a `sys.path.insert`
gives the **same module object**, and every checked attribute is the same object, not a
copy — `load_miw`/`JS_DEFAULT` for `miw_io`, and `MiwCornerLookup`, `decomp_path`,
`load_decomposition`, `corner_field_points`, `corner_z4_height_um`, `DEFAULT_PARAM_SET`,
`DEFAULT_MI_NAME` for `miw_corner_intrinsic`. The three real consumer routes were exercised
directly: `static_optics` (`from miw_io import load_miw`), `code/miw`
(`from miw_io import JS_DEFAULT`) and `value_added/code/build_optical_state.py`
(`from miw_corner_intrinsic import DEFAULT_PARAM_SET, MiwCornerLookup`).

**`decomp_path` was broken and is fixed — the one behaviour change in the move.** It built
`output/<param_set>/<mi_name>/`, joining the two **long** keys, a layout the tree has not
had since `dir_name` was introduced; the real layout is
`output/miw/<ps dir_name>_<mi dir_name>/`. So every default lookup missed.
`notes/status/vmode_thermal_and_lut_handoff.md` records `DEFAULT_PARAM_SET` and
`DEFAULT_MI_NAME` as "both still stale and name nothing on disk" — **that reading was
wrong**: the keys were always right and the path construction was not. With the path fixed
the default pair resolves to a real file
(`aos/output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_decomp.parquet`, confirmed on
disk). Still pass `--intrinsic-ref` and `--miw-param-set` explicitly: which MIW is
subtracted defines the optical state, so it belongs in the variant, not a module default.

### 3a step 1 — reference build with the unmoved code: DONE (commit `d81cc3d`)

Built at commit `34b9980` (clean tree), interactive on `sdfiana`, **454.2 s wall clock**
total across the six steps.

A throwaway `pathA_50_34_i_ref2bin` entry on `danish_1_2`, `n_dof 50 / n_keep 34`, in its
own `mi_config_ref.yaml` passed via `--config`, so **`aos/mi_config.yaml` was never
touched**. Two rotator bins, `[-3, 3]` and `[55, 65]`.

**Two bins, not one.** `run_intrinsic_split` forces OCS-only when it sees fewer than two
distinct rotator angles (`run_intrinsic_split.py:223`), so a one-bin reference would never
exercise the CCS half. With two the split saw thetas `[-0.01, 60.0] deg` (2 distinct) and
produced `|O|` telescope-fixed RMS 0.0542 µm and `|C|` camera-fixed RMS 0.0130 µm of
wavefront.

Input: a **three-night FAM slice**, 20260315/16/17, staged by
`rubinwork/products/miw/refbuild/stage_fam_nights.py`:

```bash
python -m rubinwork.products.miw.refbuild.stage_fam_nights \
    --reference aos/output/fam_processing/danish_1_2 \
    --day-obs 20260315 20260316 20260317 \
    --out /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/miw/_fam3n
```

Staged slice: `visits.parquet` 169 rows x 405 columns, `fits.parquet` 168 rows x 456
columns, `donuts.parquet` 475,128 rows x 48 columns in 169 row groups (507 MB, 4.1% of the
full 12.36 GB). 8.3 s.

**Why a slice.** `build_intrinsic` selects its own visits and is already small, but
`intrinsic_sidecar` and `refit_mi` read the **whole** FAM `donuts.parquet` — 9,083,526 rows
and 12.36 GB for `danish_1_2` — which would dominate the reference.

**Two things the staging had to get right, both found by failing first:**

1. **The donut slice must be copied ROW GROUP BY ROW GROUP.**
   `run_build_intrinsic.load_kept_donuts` selects donuts by row group, keying each group on
   its `day_obs`/`seq_num` column **statistics**, so a FAM `donuts.parquet` holds exactly
   one row group per visit. The first attempt streamed by row batch and rewrote the slice as
   one large group: every lookup missed and the build died on `RuntimeError: No donut row
   groups matched the kept visits`. Copying groups is also 16x faster — 8.3 s against 133 s.
2. **`visits.parquet` carries `alt` in RADIANS**, while `mi_config.yaml` states the
   elevation window in degrees (`alt_min_deg: 65.0`, `alt_max_deg: 75.0`). A first bin count
   filtering on `alt` directly returned **0 visits** of 3,385. `bin_visit_counts` converts.

Bin sizes under the build's own cuts (band `i`, `BLOCK-T614_triplets`, elevation 65-75 deg,
`day_obs <= 20260513`): `[-3, 3]` 37 visits and `[55, 65]` 36, both drawing only from those
three nights, so the slice does not change what the build sees in them. Counts are visits
(dimensionless).

The commands, run from `aos/` (the unmoved rules' own cwd), and the measured cost:

| step | wall clock | result |
|---|---|---|
| `build_intrinsic` rot_-3_3 | 145.1 s | 37 visits, 37 good, 104,194 donuts on a 73x73 grid |
| `build_intrinsic` rot_55_65 | 133.4 s | 36 visits, 36 good, 99,696 donuts |
| `intrinsic_split` | 58.4 s | 3,985 map rows, 21 (j, part) records, 64-page PDF |
| `intrinsic_sidecar` | 14.7 s | 475,128 of 475,128 donuts with a finite intrinsic, Z4_height mean -0.0091 µm of wavefront |
| `wfs_intrinsic_sidecar` (`refitWcs`) | 13.4 s | 34,631 of 34,631 corner donuts, Z4_height mean -0.0277 µm of wavefront |
| `refit_mi` | 88.9 s | 169 rows x 831 columns, 0 of 169 visits flagged bad_fit |

Output in `_phase2_refbuild/miw/ref/`, 21 files:

| file | rows | columns | bytes |
|---|---|---|---|
| `build/rot_-3_3/intrinsic_grid.parquet` | 3,831 | 9 | 2,488,564 |
| `build/rot_-3_3/dz_fits.parquet` | 37 | 683 | 494,718 |
| `build/rot_-3_3/intrinsic_cov_edge.parquet` | 21 | 4 | 7,823 |
| `build/rot_-3_3/mi_config.yaml` | — | — | 1,031 |
| `build/rot_55_65/intrinsic_grid.parquet` | 3,843 | 9 | 2,493,349 |
| `build/rot_55_65/dz_fits.parquet` | 36 | 683 | 489,296 |
| `build/rot_55_65/intrinsic_cov_edge.parquet` | 21 | 4 | 7,815 |
| `build/rot_55_65/mi_config.yaml` | — | — | 1,032 |
| `intrinsic_split_maps.parquet` | 3,985 | 44 | 1,610,291 |
| `intrinsic_split_decomp.parquet` | 21 | 10 | 9,602,937 |
| `intrinsic_split_rms.parquet` | 21 | 12 | 5,886 |
| `zk_intrinsic.parquet` | 475,128 | 8 | 96,589,099 |
| `wfs/refitWcs/zk_intrinsic.parquet` | 34,631 | 8 | 7,513,964 |
| `fits.parquet` | 169 | 831 | 1,414,885 |
| `intrinsic_split.pdf` | — | — | 4,458,423 |
| 6 per-bin PDFs | — | — | 810,920 to 2,421,301 each |

**Comparison keys, each verified unique on the reference** (unique rows of total, counts
dimensionless):

| table | key | unique |
|---|---|---|
| `intrinsic_split_maps` | `thx_deg, thy_deg` | 3,985 / 3,985 |
| `intrinsic_split_decomp` | `j, part` | 21 / 21 |
| `intrinsic_split_rms` | `j` | 21 / 21 |
| `fits` | `day_obs, seq_num` | 169 / 169 |
| `zk_intrinsic` | `day_obs, seq_num, detector, centroid_x_extra, centroid_y_extra` | 475,128 / 475,128 |
| `wfs/refitWcs/zk_intrinsic` | same | 34,631 / 34,631 |
| `build/rot_*/intrinsic_grid` | `thx_deg, thy_deg` | 3,831 / 3,831 and 3,843 / 3,843 |
| `build/rot_*/dz_fits` | `day_obs, seq_num` | 37 / 37 and 36 / 36 |
| `build/rot_*/intrinsic_cov_edge` | `j` | 21 / 21 |

`day_obs, seq_num, detector` alone is **not** unique on either sidecar — 30,352 of 475,128
on the FAM one and 4,485 of 34,631 on the corner one, since many donuts share a detector in
one exposure — so the extra-focal centroid is what separates the rows. The comparison
asserts uniqueness on both sides, so a build that breaks it fails loudly instead of
mis-aligning.

The PDFs are **not** byte-comparable (matplotlib embeds a creation timestamp), so step 4
compares the parquet tables at zero tolerance, byte-compares the per-bin `mi_config.yaml`,
and checks only that each PDF is produced at a comparable size.

### 3a step 2 — the move: DONE (commit `b9756aa`)

New in the product: `__init__.py` (with `load`), `reader.py`, `variants.yaml`,
`gen_mi_config.py`, `Snakefile`, `write_manifest.py`, `compare_builds.py`,
`refbuild/stage_fam_nights.py`, and `tests/test_miw.py` (56 tests) with
`tests/mi_config_premove_body.yaml`.

The product Snakefile builds one variant and one build at a time, from the repository root:

```bash
snakemake -s rubinwork/products/miw/Snakefile -n \
    --config variant=d12-A_50_34_i_5rot build=20261010 fam_build=20260920 \
             src_build=20261010
```

Four things it does that the `aos/` rules did not have to:

1. **`cd aos/` for every rule**, for the same reason `fam_tables` and `cwfs_tables` do: the
   external `mi_config.load_mi_config` and `intrinsics_lib.load_param_sets` open their
   config files relative to the working directory.
2. **`build_intrinsic` is not defined at all for a `build_from` variant.** Such an entry
   reuses its parent's grids, so they are inputs this Snakefile does not build, resolved
   from the parent's build through the catalog (`--config src_build=`, default `current`).
   Verified: `d12-A_50_34_i_5rot` plans **7** jobs with **zero** `build_intrinsic`, and its
   `intrinsic_split` reads exactly the five `rotator_select` bins.
3. **`wfs_intrinsic_sidecar` fans out over the `cwfs_tables` variants of the same FAM
   variant that have a build to read**, and reports every one it skips with the reason.
4. **An external variant raises before any path resolves**, naming the
   `--register-external` command instead.

Planned job counts, verified against staged `fam_tables`, `cwfs_tables` and parent-`miw`
builds in a scratch data root (`_phase2_refbuild/_scratch_3a/`), never the real products
tree:

| variant | jobs | breakdown |
|---|---|---|
| `d12-A_50_34_i` (self-building) | 16 | `build_intrinsic` 9, `intrinsic_split` 1, `intrinsic_sidecar` 1, `wfs_intrinsic_sidecar` 2, `refit_mi` 1, `manifest` 1, `all` 1 |
| `d12-A_50_34_i_5rot` (`build_from`) | 7 | the same without the 9 `build_intrinsic` |

`manifest` depends on every table, the split PDF and every CWFS sidecar, so an interrupted
build leaves no manifest and the catalog does not see it. It records
`{"fam_tables": "danish_1_2@20260920"}` and, for a `build_from` variant,
`{"miw": "d12-A_50_34_i"}` — without which the provenance of the maps is unrecoverable from
the manifest, since such a build runs only the split.

### 3a step 3 — verification: DONE

**`aos/mi_config.yaml`: body byte-identical**, `aos/param_sets.yaml` unchanged at md5
`0f72da651cf9319cc283754c614051fd`. Both generators pass `--check`. The resolved config is
identical for all 12 `(param_set, mi_name)` pairs.

**`snakemake -n` in `aos/`: 189 -> 37 jobs, a drop of 152.** 77 are the moved rules; the
other 75 are nine incomplete MIW builds losing their study jobs. Every rule accounted for:

| rule | before | after | change | why |
|---|---|---|---|---|
| `build_intrinsic` | 36 | 0 | **-36** | moved |
| `intrinsic_sidecar` | 12 | 0 | **-12** | moved |
| `refit_mi` | 12 | 0 | **-12** | moved |
| `intrinsic_split` | 9 | 0 | **-9** | moved |
| `wfs_intrinsic_sidecar` | 8 | 0 | **-8** | moved |
| `build_lut` | 12 | 3 | -9 | 9 incomplete MIW builds |
| `wfs_mimic` | 12 | 3 | -9 | same |
| `bounce` | 12 | 3 | -9 | same |
| `dz_correlations` | 12 | 3 | -9 | same |
| `dz_explained` | 12 | 3 | -9 | same |
| `thermal_correlations` | 12 | 3 | -9 | same |
| `vmode_correlations` | 12 | 3 | -9 | same |
| `study_radialbins` | 12 | 2 | -10 | 9 incomplete, plus 1 now up to date |
| `wfs_dof_compare` | 8 | 6 | -2 | `tarts` has no corner sidecar |
| `wfs_corner_compare` | 4 | 4 | 0 | unchanged |
| `aberration_pairs`, `dz_fit_check`, `plots`, `all` | 1 each | 1 each | 0 | unchanged |
| **total** | **189** | **37** | **-152** | |

All counts are jobs (dimensionless). **This is the step-3 criterion failing as literally
written**, by far the most of any product so far, and it is reported rather than hidden.
Three distinct causes:

1. **Only 3 of 12 MIW builds are complete on disk.** `danish_1_2_A_50_34_i`,
   `danish_1_2_A_50_34_i_5rot` and `danish_1_3_test_A_50_34_i_5rot` have both
   `fits.parquet` and `zk_intrinsic.parquet`; six more have the split only, and three have
   nothing. Before the move those nine were plannable because `build_intrinsic` through
   `refit_mi` were in this Snakefile and would have produced them. After it they have no
   producer here, and a plain `snakemake -n` aborted the whole DAG with
   `MissingInputException`. So `MI_PAIRS` now filters on `fits.parquet` **and**
   `zk_intrinsic.parquet` — every study rule reads one or the other — and prints each skip
   with what is missing. The 3 complete builds keep **every** study job.
2. **`tarts` has CWFS donuts but no MIW corner sidecar**, so its 2 `wfs_dof_compare` jobs
   lost their producer when `wfs_intrinsic_sidecar` moved. Filtered and reported the same
   way, on `output/wfs_dof_compare/<d>/<cwfs>/zk_intrinsic.parquet`.
3. **`study_radialbins` drops 10, not 9**, and the extra one is correct:
   `danish_1_3_test_A_50_34_i_5rot`'s PDF (2026-09-28T11:03:10) is **newer than all five of
   its source grids** (11:02:37 to 11:02:50), so it is genuinely up to date. It planned
   before the move only because `build_intrinsic` would have re-run and made the grids
   newer. Confirmed by mtime, not inferred.

**The product Snakefile plans the expected jobs**: 16 and 7, as tabulated above.

**Import smoke test: 63 of 63 import, no regressions.** Two entries change, both the move
itself: the test enumerates files matching `^\s*(from|import)\s+(common|aos_state|...)`,
and `aos/code/miw_corner_intrinsic.py` drops out because the shim no longer imports
`aos_state` while `rubinwork/miw_corner.py` appears because it does. Both shims were then
imported directly and both load cleanly.

**Tests: 146 passed** across `rubinwork/products/` — the 56 new `miw` tests plus all 90
pre-existing `fam_tables`, `cwfs_tables` and `catalog` tests, none of which needed changing.

### 3a step 4 — the comparison: DONE, exact match

The reference rerun with the moved code at commit `b9756aa`, **449.6 s wall clock**
(reference 454.2 s): 138.9 and 135.2 s for the two bins, 58.6 s split, 14.7 s sidecar,
13.4 s corner sidecar, 88.8 s refit. Same visit counts, same `|O|` 0.0542 and `|C|` 0.0130
µm of wavefront, same 475,128 and 34,631 donuts reconstructed, same 0 of 169 visits flagged
bad_fit. Output in `_phase2_refbuild/miw/moved/`.

`python -m rubinwork.products.miw.compare_builds` at **zero tolerance**
(`rtol=0.0, atol=0.0`), over **12 tables** discovered by walking both builds:

| table | rows | columns compared | only in one side | differing |
|---|---|---|---|---|
| `intrinsic_split_maps` | 3,985 | 44 | 0 | **0** |
| `intrinsic_split_decomp` | 21 | 10 | 0 | **0** |
| `intrinsic_split_rms` | 21 | 12 | 0 | **0** |
| `zk_intrinsic` | 475,128 | 8 | 0 | **0** |
| `wfs/refitWcs/zk_intrinsic` | 34,631 | 8 | 0 | **0** |
| `fits` | 169 | 831 | 0 | **0** |
| `build/rot_-3_3/intrinsic_grid` | 3,831 | 9 | 0 | **0** |
| `build/rot_-3_3/dz_fits` | 37 | 683 | 0 | **0** |
| `build/rot_-3_3/intrinsic_cov_edge` | 21 | 4 | 0 | **0** |
| `build/rot_55_65/intrinsic_grid` | 3,843 | 9 | 0 | **0** |
| `build/rot_55_65/dz_fits` | 36 | 683 | 0 | **0** |
| `build/rot_55_65/intrinsic_cov_edge` | 21 | 4 | 0 | **0** |

Every key is unique on both sides, so the alignment is sound rather than accidentally
agreeing. Both per-bin `mi_config.yaml` files are **byte-identical**. All 7 PDFs came out
at identical size (0.00% difference), though they are deliberately not byte-compared.

**Nothing differs — not one column in any of the twelve tables.** `RESULT: builds match`,
exit 0.


## Bridging the study rules to the catalog (commit `fec3741`)

Done before `coadds`, because after the `fam_tables`, `cwfs_tables` and `miw` moves the
`aos/` study rules read product data only from the old tree, so a product rebuilt into the
new tree stays invisible to them until phase 5 — and nine of the twelve MIW variants have
no complete old-tree build, so their study jobs were simply gone.

`aos/Snakefile` now has **one resolver**, used by every study-rule input that reads
`fam_tables`, `cwfs_tables` or `miw` data and by every `--tables-dir` / `--miw-dir` /
`--wfs-dir` argument those rules pass to a script. It returns a DIRECTORY, resolved:

1. the old-tree path, if its required files are all there;
2. else the catalog's `current` build of the matching product variant;
3. else the old-tree path anyway, so the existing gates (`_mi_complete`, `wfs_variants`,
   `_has_corner_sidecar`) see a missing file and skip the variant with a reported reason,
   exactly as before.

**No study output path changes** — outputs stay keyed on the old-tree dir names `{ps}`,
`{d}` and `{cwfs}`, so nothing already built is invalidated or relocated.

Three details that are not obvious:

- **The variant names come from the products' own index fields, not from string surgery on
  the directory names.** A `fam_tables` variant IS its `param_sets.yaml` `dir_name`; a
  `cwfs_tables` variant carries the `{cwfs}` wildcard as its `upstream_name`; a `miw`
  variant carries the second half of `{d}` as its `dir_name`. A renamed variant therefore
  stops resolving instead of silently resolving to a stale old-tree directory.
- **The MIW build grids resolve separately from the MIW tables.** A `build_from` parent
  legitimately has per-bin grids and nothing else — that is the only reason several of them
  exist — so gating the grids on `fits.parquet`/`zk_intrinsic.parquet` would send
  `src_miw_dir` back to the old tree for a parent only the new tree has. `src_miw_dir`
  resolves on the grid files themselves.
- **The corner sidecar changes directory across the trees.** The old tree writes it under
  the CONSUMER's study directory, `output/wfs_dof_compare/<d>/<cwfs>/zk_intrinsic.parquet`;
  the product writes it inside its own build as `wfs/<cwfs variant>/zk_intrinsic.parquet`,
  keyed on the full `cwfs_tables` variant name rather than the bare `wfs_name`.
  `miw_corner_sidecar()` maps between them.

`report_resolved()` prints which source every variant resolved to, once, at DAG time, after
a sweep that resolves every param_set and every planned pair so the report is complete
before the first rule is planned.

**Verified both ways.**

| | jobs | resolutions |
|---|---|---|
| `snakemake -n` in `aos/`, real data root | **37** — today's count, rule for rule | 21: 11 old tree, 0 catalog, 10 unresolved |
| the same with `RUBINWORK_DATA=.../_scratch_3a` and a staged complete `miw/d13v1000-A_50_34_i_5rot@20261010` | **44** | 23: 13 old tree, **1 catalog**, 9 unresolved |

All counts are jobs or variant resolutions (dimensionless). The +7 are exactly that
variant's study jobs reappearing — one each of `bounce`, `build_lut`, `dz_correlations`,
`dz_explained`, `thermal_correlations`, `vmode_correlations`, `wfs_mimic`.
`study_radialbins` stays at 2 because its grids come from the `build_from` parent
`d13v1000-A_50_34_i`, which has grids in neither tree. The planned `wfs_mimic` job reads
its MIW from the catalog and its FAM tables from the old tree — the intended mixed
resolution, since `danish_1_3_v1000`'s old-tree FAM tables are complete and win.

146 product tests pass; the import smoke test is 63 of 63 ok, the same set and the same
result as after the `miw` move.

`aos/code/infra/stage_scratch_product.py` stages a placeholder build (one-row tables plus a
complete manifest) into a scratch root, and **refuses any root not under a `_scratch_3a`
directory** so it can never touch the real products tree.

## 3a, `coadds` — the move

### Scope, agreed 2026-10-10

**The product is `run_coadd_blocks_miw.py` and its four data files.** Everything else in
`aos/code/coadd/` is the coadd-vs-MIW study and stays.

| moves to `rubinwork/products/coadds/builders/` | stays (study) |
|---|---|
| `run_coadd_blocks_miw.py` | `recompute_coadd_metrics`, `analyze_miw_bias_regression`, `analyze_miw_field_order`, `analyze_dz_goodness_of_fit`, `analyze_umode_null_coupling`, `analyze_miw_dz_full_k`, `check_k_truncation`, `run_fam_coadd_miw` |

Product files: `blocks_summary.parquet`, `coadd_metrics.parquet`, `block_grids.npz`,
`block_grids_ext.npz`, plus the two PDFs as build artifacts.

**Every reader, found by grep rather than taken from the plan. There is no code reader
outside `aos/code/coadd/` at all:**

| reader | reads | where |
|---|---|---|
| `recompute_coadd_metrics.py` | `block_grids.npz` | `:563` |
| same | `blocks_summary.parquet`, twice (`program`; then per-block camera temps) | `:264`, `:294` |
| `analyze_miw_bias_regression.py` | `block_grids.npz` | `:115` |
| `analyze_miw_field_order.py` | `block_grids.npz`, `coadd_metrics_rebin3.parquet` | `:43`-`:46`, **at import time** |
| `analyze_dz_goodness_of_fit.py` | `blocks_summary.parquet` | `:107`, default at `:294` |
| `analyze_umode_null_coupling.py` | `blocks_summary.parquet` via `assign_blocks` | `:69`, `:83` |
| `aos/moresnippets.ipynb:5393` | `block_grids.npz` | scratch snippets notebook |

`aos/Snakefile:922` declares `fam_coadd_miw_maps.pdf` as an **output**, not an input; no
rule consumes any coadd file. `aos/code/infra/migrate_output_layout.py:145` only rewrites
paths.

**`fam_coadd_miw` is NOT part of this move, and the next-action note above was wrong about
it.** That rule runs `run_fam_coadd_miw.py`, which writes only `fam_coadd_miw_maps.pdf` — a
plot no code reads back — so it is a study rule and stays. The script that moves,
`run_coadd_blocks_miw.py`, **has no Snakefile rule at all** today: it is run only from
`aos/run_coadd_blocks_miw.sbatch`. So `snakemake -n` in `aos/` must be **unchanged at 37**
across this move, and there is no `aos/`-side coadd target to plan.

Three findings worth carrying forward, all Aaron's calls on 2026-10-10:

1. **`block_grids_ext.npz` has no reader anywhere** — written at
   `run_coadd_blocks_miw.py:911`, read by nothing, 66 MB in the production build. **Kept**
   in the product: it is the extra-pupil-Zernike primitive the retrieval-bias work needs,
   and it is what distinguishes the two variants.
2. **`coadd_metrics.parquet` has no reader either.** The only metrics file anyone reads is
   `coadd_metrics_rebin3.parquet`, which `recompute_coadd_metrics.py:634` writes — a
   **study** output. **Kept**: it is the natural per-coadd table and the rebinned one is
   conceptually derived from it.
3. **Three study readers point at a stale path that does not exist on disk.**
   `recompute_coadd_metrics.py:562`, `analyze_miw_bias_regression.py:114` and
   `analyze_miw_field_order.py:36` build `output/<long param_set>/coadd_50_34/`, the layout
   the tree lost when `dir_name` arrived — the same bug class as `miw_corner.decomp_path`.
   `aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/` does not exist, so **all three
   are broken today**; two also hardwire `coadd_50_34` as a literal. **Fixed in the move
   commit**, as the one behaviour change.

### Variants: two, prefixed by the MIW upstream

A coadd depends on a `fam_tables` variant **and** a `miw` variant. The MIW is the main
upstream — the product is a per-block remnant *against* a given MIW — so the variant name
carries the MIW variant, not the FAM one.

| variant | fam_variant | miw_variant | `ext_terms` | old-tree dir | old-tree output? |
|---|---|---|---|---|---|
| `d12-A_50_34_i_5rot` | `danish_1_2` | `d12-A_50_34_i_5rot` | `secondary` | `coadd/danish_1_2/50_34/` | **yes**, 6 files, newest 2026-09-14 |
| `d12-A_50_34_i_5rot_v2` | `danish_1_2` | `d12-A_50_34_i_5rot` | `all` | `coadd/danish_1_2/50_34_v2/` | **no — `archive/` only, top level empty** |

**Checked before moving, as this handoff requires.** `aos/output/coadd/danish_1_2/50_34_v2/`
holds only `archive/20260903_iband/`; its top level is empty. Nothing is currently planned
for it (`run_coadd_blocks_miw` has no rule), so no job count changes — but the product
Snakefile reports it as unbuilt.

The plan's names `d12-50_34` and `d12-50_34_v2` are **not** used: `50_34` names the
`n_dof`/`n_keep` pair, not the MIW, so they would be ambiguous as soon as a second MIW
variant gets a coadd. The old-tree directory name (`50_34`, `50_34_v2`) is carried as a
`dir_name` field, exactly as `miw` does.

The two variants differ **only** in `ext_terms`, which changes nothing but
`block_grids_ext.npz` and the per-block PDF page count — the archive README records
`block_grids.npz` as byte-identical between the two 2026-09-03 runs.

**Where the block-selection settings live.** Today they are argparse defaults plus a
`DEFAULTS` array in `aos/run_coadd_blocks_miw.sbatch`; there is no config file, and
`analysis_config.yaml`'s `fam_coadd_miw` section belongs to the *other* script.
`coadds/variants.yaml` now owns them per variant (`bands`, `min_visits`, `max_blur`,
`pointing_tol`, `wide_tol`, `max_seq_span`, `bounce_round`, `bounce_programs`,
`wide_programs`, `ext_terms`, `cwfs_inner`/`cwfs_outer`, `n_bins`, `n_iter`, `pct`), with
the argparse defaults left intact so a hand run is unchanged. **No generated file is
needed** — unlike `param_sets.yaml` and `mi_config.yaml`, nothing external reads a coadd
config by a bare relative path.

### 3a step 1 — reference build with the unmoved code: DONE

Built at commit `fec3741` (clean tree), interactive on `sdfiana`, **76.7 s wall clock**.

**One night, 20260316, chosen to give three blocks.** `run_coadd_blocks_miw.py` has **no
block-count limit flag**, so the reference is sized by staging the FAM nights, the same
route `miw` took. Measured with the real `assign_blocks` on the three nights of the `miw`
reference slice (counts are blocks, dimensionless):

| night | FAM visits | blocks detected | built (`n_visits >= 6`) |
|---|---|---|---|
| 20260315 | 62 | 6 | 5 |
| **20260316** | **35** | **3** | **3** |
| 20260317 | 72 | 6 | 6 |
| all three | 169 | 15 | 14 |

Input, staged by the `miw` product's own script (it needed no change), 2.9 s:

```bash
python -m rubinwork.products.miw.refbuild.stage_fam_nights \
    --reference aos/output/fam_processing/danish_1_2 \
    --day-obs 20260316 \
    --out /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/coadds/_fam_20260316
```

Staged slice: `visits.parquet` 35 rows x 405 columns, `fits.parquet` 35 rows x 456 columns,
`donuts.parquet` 98,199 rows x 48 columns **in 35 row groups**. The row-group structure is
load-bearing again: `run_coadd_blocks_miw._visit_row_groups` keys each group on its
`day_obs`/`seq_num` column statistics, exactly as `run_build_intrinsic.load_kept_donuts`
does, so one row group per visit is required.

The MIW it compares against is the completed `miw` reference,
`_phase2_refbuild/miw/ref/zk_intrinsic.parquet` (475,128 donuts over the three nights). The
sidecar is indexed on `(day_obs, seq_num)`, not row position, so a three-night sidecar
covers a one-night block set.

The build command, run from `aos/` (the unmoved script's own path):

```bash
cd ~/notebooks/rubin-work/aos && python code/coadd/run_coadd_blocks_miw.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --mi-name pathA_50_34_i_5rot \
    --ext-terms secondary \
    --tables-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/coadds/_fam_20260316 \
    --miw-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/miw/ref \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/coadds/ref
```

Note `--out-dir` is **required**: without it the script builds
`output/<long param_set>/<out-name>/`, the stale layout of finding 3 above.

Output, in `_phase2_refbuild/coadds/ref/`, 6 files:

| file | contents | bytes |
|---|---|---|
| `blocks_summary.parquet` | 3 rows x 26 columns | 15,653 |
| `coadd_metrics.parquet` | 3 rows x 81 columns | 48,618 |
| `block_grids.npz` | 42 arrays | 835,382 |
| `block_grids_ext.npz` | 6 arrays | 943,315 |
| `coadd_blocks_miw_perblock.pdf` | 3 blocks x [coadd/MIW/diff] | 3,716,991 |
| `coadd_blocks_miw_timeseries.pdf` | metrics + corr-bars + scatter + 3 telemetry + u-mode heatmap, n=3 | 49,316 |

The three blocks, all `T614_triplets` on 20260316, band `i`, elevation 69.8 deg, all
in-family:

| block | ordinal | seq range | visits | donuts | rotator angle (deg, OCS) | FWHM (arcsec) |
|---|---|---|---|---|---|---|
| 0 | 0 | 647-680 | 12 | 33,496 | +60.12 | 1.13 |
| 1 | 1 | 693-726 | 12 | 33,762 | -59.89 | 1.00 |
| 2 | 2 | 737-767 | 11 | 30,941 | +0.15 | 0.97 |

Each block ran the full 3-iteration U-constrained build and converged: RMS change between
iterations 1.24e-03 then 7.09e-04 µm of wavefront on block 1, 1.25e-03 then 7.27e-04 on
block 2. Coadd-vs-MIW agreement, over the full focal plane:

| block | `resid_rms_combined` (µm of wavefront) | `corr_combined` (Pearson r, dimensionless) |
|---|---|---|
| 0 | 0.0351 | 0.946 |
| 1 | 0.0451 | 0.927 |
| 2 | 0.0496 | 0.946 |

So the reference exercises block detection, the iterative build, the MIW comparison, the
per-Zernike and CWFS-annulus metrics, the u-modes, both npz files and both PDFs.

**Comparison keys.** `block` is unique on both tables — 3 of 3 rows on
`blocks_summary.parquet` and on `coadd_metrics.parquet`. The two npz files are compared
array by array by name (42 and 6 arrays), with the per-block axis aligned on `block`.

The PDFs are **not** byte-comparable (matplotlib embeds a creation timestamp), so step 4
compares the two parquet tables and both npz files at zero tolerance and checks only that
each PDF is produced at a comparable size.

### 3a step 2 — the move: DONE (commit `d0e4182`)

`run_coadd_blocks_miw.py` moved to `rubinwork/products/coadds/builders/`, with a shim at
the old path. New in the product: `__init__.py` (with `load` and `load_grids`),
`reader.py`, `variants.yaml`, `Snakefile`, `write_manifest.py`, `compare_builds.py`, and
`tests/test_coadds.py` (40 tests).

The builder needed **no import changes at all** — it imports only stdlib, third-party and
the external package, with no `sys.path` work and no sibling imports. What was added is a
`--variant` route, as `run_wfs_mktable` has: it resolves the param_set, the `mi_name` and
the block-selection settings from `variants.yaml`, while `--param-set` plus `--mi-name`
still works. A flag given explicitly on the command line still wins over the variant's
setting, the sentinel for "not given" being the parser's own default.

**The shim gives module identity**, verified: a bare-name import after a `sys.path`
insert of `aos/code/coadd` returns the *same module object*, and `assign_blocks`,
`block_summary`, `main` and `THERMAL_VARS` are the same objects, not copies. Those are
what `analyze_miw_bias_regression.py`, `analyze_miw_field_order.py` and
`aos/moresnippets.ipynb` import that way.

Four things worth recording:

1. **`dir_name` must be quoted in `variants.yaml`.** An unquoted `50_34` parses as the
   **integer 5034** — YAML treats an underscore as a numeric digit separator. Caught by
   printing the parsed value, and now a test (`test_dir_name_is_a_string_not_a_yaml_integer`).
2. **`write_manifest` refuses to record only one upstream.** `--fam-build` and
   `--miw-build` must come together: the FAM build sets which visits exist and the MIW
   sets what is subtracted, so half the provenance recorded as though it were all of it
   would make a build look reproducible when it is not.
3. **The product Snakefile is one rule, not six.** The script does block detection, the
   per-block iterative build and both renders in one pass over the donut table; splitting
   it per output file would mean reading that multi-GB table more than once.
4. **`recompute_coadd_metrics.py` now requires `--out-dir` with `--variant`.** It writes
   `coadd_metrics_rebin<f>.parquet` and `coadd_timeseries.pdf` into the directory it read,
   which with a catalog build would edit a completed product build — rule 5 of the plan's
   section 2. Reading a build is fine; writing into it is not.

`aos/run_coadd_blocks_miw.sbatch` calls the module rather than the old path, and points at
the product Snakefile as the route that writes a manifest. The three stale-path study
readers were fixed in the same commit, per finding 3 above; `analyze_miw_field_order.py`
has no argument parser (110 lines, no functions, runs top to bottom), so its build is
named by the environment variables `COADD_VARIANT`, `COADD_MIW_VARIANT` and
`COADD_STUDY_DIR` instead.

Planned job counts, verified against staged `fam_tables` and `miw` builds in the scratch
data root (`_phase2_refbuild/_scratch_3a/`), never the real products tree:

| variant | jobs | breakdown |
|---|---|---|
| `d12-A_50_34_i_5rot` | 3 | `coadd_blocks` 1, `manifest` 1, `all` 1 |
| `d12-A_50_34_i_5rot_v2` | 3 | the same, with `--ext-terms all` |

All counts are jobs (dimensionless). A missing upstream build names itself:
`no build '29990101' of miw/d12-A_50_34_i_5rot; known builds: ['20261010', 'current']`.
An external MIW variant is refused before any path resolves, since it has no per-donut
sidecar to compare against.

### 3a step 3 — verification: DONE

**`snakemake -n` in `aos/`: 37 jobs, UNCHANGED**, rule for rule — `aberration_pairs` 1,
`bounce` 3, `build_lut` 3, `dz_correlations` 3, `dz_explained` 3, `dz_fit_check` 1,
`plots` 1, `study_radialbins` 2, `thermal_correlations` 3, `vmode_correlations` 3,
`wfs_corner_compare` 4, `wfs_dof_compare` 6, `wfs_mimic` 3, `all` 1. **This is the first
product whose move changes no job count**, and the reason is structural rather than
lucky: the moved script never had a rule in `aos/Snakefile`, so there was nothing there to
remove. `fam_coadd_miw`, the rule that *is* in that Snakefile, belongs to
`run_fam_coadd_miw.py` and did not move.

**The product Snakefile plans the expected jobs**: 3 and 3, as tabulated above.

**Import smoke test: 63 of 63 import, and `--compare` against the pre-move run reports
`no change: all 63 module import outcomes identical`.** No module's import regressed and
none dropped out of the match set, because the moved file was not in it — it imports no
moved library.

All eight remaining `aos/code/coadd/` scripts were then imported directly in script mode:
every one loads. `analyze_miw_field_order.py` is the exception by design — its reads are
at module level, so with no coadd build in the catalog it now stops with
`ProductNotFound: no variant 'd12-A_50_34_i_5rot' of product 'miw'`, a named error
instead of the previous silent miss on a path that never existed.

**Tests: 186 passed** across `rubinwork/products/` — the 40 new `coadds` tests plus all
146 pre-existing `fam_tables`, `cwfs_tables`, `miw` and `catalog` tests, none of which
needed changing.

### 3a step 4 — the comparison: DONE, exact match

The reference rerun with the moved code at commit `d0e4182`, **77.0 s wall clock**
(reference 76.7 s), through the new `--variant d12-A_50_34_i_5rot` route so that the
variant resolution itself is exercised rather than bypassed. Same 3 blocks, same visit and
donut counts, same convergence (1.24e-03 then 7.09e-04 µm of wavefront on block 1,
1.25e-03 then 7.27e-04 on block 2), same per-block `resid_rms` and Pearson r. Output in
`_phase2_refbuild/coadds/moved/`.

`python -m rubinwork.products.coadds.compare_builds` at **zero tolerance**
(`rtol=0.0, atol=0.0`):

| file | rows or blocks | columns or arrays compared | only in one side | differing |
|---|---|---|---|---|
| `blocks_summary.parquet` | 3 rows | 26 columns | 0 | **0** |
| `coadd_metrics.parquet` | 3 rows | 81 columns | 0 | **0** |
| `block_grids.npz` | 3 blocks | 42 arrays | 0 | **0** |
| `block_grids_ext.npz` | 3 blocks | 6 arrays | 0 | **0** |

`block` is unique on both sides of both tables (3 of 3), and both npz files hold the same
blocks in the same order, so the alignment is sound rather than accidentally agreeing.
Both PDFs came out at identical size (0.00% difference), though they are deliberately not
byte-compared.

**Nothing differs — not one column of either table, not one array of either npz.**
`RESULT: builds match`, exit 0.

The npz comparison is new and needed its own code: most of what this product carries is
arrays, not table rows. `compare_npz` compares array by array **by name**, so an array
added or removed is reported rather than skipped, and it **realigns the per-block axis on
the `block` array** rather than assuming stored order — a test builds two archives holding
the same blocks in different orders and asserts that only `block` itself differs.

### The coadd study scripts read the old tree again (2026-10-11)

The `coadds` move left the three study scripts resolving their build **only** through the
catalog, so until phase 3c they could not read the one complete coadd build that exists —
`aos/output/coadd/danish_1_2/50_34/`, six files, newest 2026-09-14. They now follow the
`aos/Snakefile` resolver's order instead, which is the order the study rules already use:

1. an explicit `--coadd-dir` (or `COADD_DIR` / `MIW_DIR` in the parser-less script);
2. else the old-tree directory built from the variant's **own index fields** —
   `aos/output/coadd/<fam_variant>/<dir_name>/` for a coadd,
   `aos/output/miw/<fam_variant>_<dir_name>/` for a MIW — if every required file is there;
3. else the catalog's build of that variant;
4. else `RuntimeError` naming both places tried.

`resolve_product_dir(product, variant, old_dir, required, build)` in
`recompute_coadd_metrics.py` is the one implementation; `resolve_coadd_dir` wraps it for
the two scripts with a parser and `analyze_miw_field_order.py` calls it directly, twice,
at module level. `COADD_REQUIRED` is `block_grids.npz` + `blocks_summary.parquet`; the MIW
side needs only `intrinsic_split_maps.parquet`.

**Unlike the Snakefile, a study script raises rather than falling back to the old-tree
path.** The Snakefile's fallback exists so a downstream gate sees a missing file and skips
the variant with a reason; a study script has no gate, so a silent bad path would surface
as a `FileNotFoundError` on a path nobody chose.

Two knock-on fixes the old-tree route forced, both in the same commit:

- **The "product builds are not written into" guard moved off `--variant`.** It is now
  `is_catalog_build(cdir)` — a `pathlib.relative_to` test against
  `catalog.products_root()` — so `--out-dir` (`recompute_coadd_metrics`) and `--out`
  (`analyze_miw_bias_regression`) are required only when the resolved directory really is
  a product build. Writing beside the data in the old tree is what a hand run always did
  and is allowed again. `analyze_miw_bias_regression` checks it immediately after
  resolving, not at render time, so it fails before the regression runs.
- **`resolve_visits_path` can no longer raise.** It called `catalog.manifest` whenever
  `--variant` was given, which throws with no catalog build, and its caller only *warns*
  on a missing file. It now takes the old `aos/output/fam_processing/<P>/visits.parquet`
  first and swallows the manifest lookup's failure.

The dead `--param-set` / `--out-name` / `--output-root` flags were dropped from both
parsers — nothing read them after the move.

**Verified on the real data root** (`/sdf/group/rubin/u/roodman/LSST/rubin-work`, which
holds only the `miw` product today):

| check | result |
|---|---|
| `resolve_coadd_dir('d12-A_50_34_i_5rot')` | `aos/output/coadd/danish_1_2/50_34`, `is_catalog_build` False |
| `resolve_visits_path` for it | `aos/output/fam_processing/danish_1_2/visits.parquet` |
| `resolve_coadd_dir('d12-A_50_34_i_5rot_v2')` | `RuntimeError` naming the absent old-tree dir **and** the absent catalog product — correct, it has a build in neither tree |
| MIW resolution | `aos/output/miw/danish_1_2_A_50_34_i_5rot` |
| `recompute_coadd_metrics.py --help`, `analyze_miw_bias_regression.py --help` | both exit 0 |
| `analyze_miw_field_order.py`, module level | `MIW` and `CO` both resolve to the old tree; it stops only on `coadd_metrics_rebin3.parquet`, which is a **study** output `recompute_coadd_metrics` has not been run to produce |
| `snakemake -n` in `aos/` | **37** jobs, unchanged |
| `pytest rubinwork/products/` | 186 passed |

All counts are jobs or tests (dimensionless).

## 3a, `dof_lut` and `bounce_tables` — the move

### Scope and decisions, agreed 2026-10-11

Aaron's three calls, in answer to the investigation below:

1. **`dof_lut` fits the plain FAM DZ, and that is recorded, not changed.** "I used the
   plain FAM DZ then but may want to switch to using the post MIW DZ." Recorded in
   `aos/docs/studies/lut.md` (commit `d5eeba6`); no code touched, no 3b fix applied.
2. **The `_july` bounce run is its own variant**, so it can be reviewed later — not a
   build of `d12-A_50_34_i_5rot` as the plan's section 9 says. "I don't recall what the
   _july variant changed."
3. **The bounce tables are promoted to a product, and the shared functions become a
   library, in the move commit** — "when it is most convenient", which is this commit
   because `run_bounce.DEFAULT` has to leave the builder either way.

### `dof_lut` does NOT subtract the MIW: it fits the plain FAM DZ

Established by reading `aos/code/lut/run_build_lut.py`, and the answer to the question
the previous "next concrete action" raised:

| file:line | what it does |
|---|---|
| `aos/Snakefile:794`-`:795`, `:405` | the rule's `fits`/`visits` inputs are `d_tables(w.d)` = `fam_dir(...)`, the **FAM** tables — not `output/miw/<d>/` |
| `run_build_lut.py:132`, `:152` | reads `visits.parquet` and `fits.parquet` from `--tables-dir` |
| `run_build_lut.py:165` | `osv.build_ofc_svd(iZs, k_min, k_max, n_keep_spec, n_dof=n_dof_spec, ...)` |
| `run_build_lut.py:170` | `osv.project_dz_table(fits, prefix, svd)`, whose `W` is `dz_table_to_W(fit_table, ...)` — the **raw** DZ matrix of that table |
| `run_build_lut.py:171`-`:172` | `W_fit = A @ U_eff.T`; `W_resid = W - W_fit`. The only subtraction is the SVD's own reconstruction |
| `run_build_lut.py:99`-`:121`, `:191` | `mi_config` is read for **parameters only**: `n_dof`, `n_keep`, `k_min`/`k_max`, `coord_sys`, `filter`, `programs`, `ofc_normalization_yaml`, `mc.rotator_bins(cfg)` |

No intrinsic map, no `zk_intrinsic`, no MIW path appears in the file. The docstring says
it at `:5`-`:8` ("the **Phase-1** per-visit Double-Zernike fits").

The reference build's own numbers show the consequence: `M2_dy = -2364 µm` of hexapod
decentre (`lut.parquet`, `value` column), far outside any physical stroke, which is what
fitting the **absolute** state without an intrinsic subtracted looks like.

Two side findings, neither fixed:

- The comment at `run_build_lut.py:180`-`:183` says "if MI subtraction leaves
  rotator-coupled residual it gets averaged away here". That describes a MIW build, not
  this script, and is misleading as written.
- `aos/code/processing_compare/compare_fam_processings.py:101` is the one reader outside
  `aos/code/lut/`, and it builds `<base>/<mi>/lut/lut.parquet` — the same dead-layout bug
  class as the `coadds` and `decomp_path` ones. Broken today. Not a 3b fix; flagged.

**Consequence for the variant keying.** `dof_lut`'s real upstream is a `fam_tables`
variant; the MIW variant contributes only SVD and selection settings. The variants are
still named with the MIW-prefixed form below, because a LUT is built per (param_set,
mi_name) pair and the dtag is what the old tree uses — but `variants.yaml` names the
`fam_tables` variant as the upstream the manifest records, with `mi_name` as a settings
key. If Aaron switches to the post-MIW DZ, the upstream becomes `miw` and that is a
variant-definition change, not just a builder change.

### `bounce_tables`: five tables are the product, nine PDFs are the study

`run_bounce.py` writes **5 parquet tables and 9 PDFs** into one directory (`tbl_dir` and
`out_dir` are both `--out-dir` when it is given, `run_bounce.py:272`).

Readers outside `aos/code/bounce/`, found with `rg`:

| file | read by | where |
|---|---|---|
| `bounce_dof_stats.parquet` | `aos/code/cwfs_lut/bounce_compare.py` — a **different study** | `:132`, hardwired `aos/output/bounce/<run>/`; `DEFAULT_BOUNCE_RUN = 'danish_1_2_A_50_34_i_5rot_july'` at `:60` |
| `bounce_fwhm_metric.parquet` | `smatrix/docs/studies/regularized_inversion.md:89` validates against it; the script **recomputes** rather than reads | docs only |
| `bounce_kj_stats.parquet` | the `bounce` rule's declared output and `rule all` (`aos/Snakefile:548`) | no code reader |
| `bounce_fwhm_vs_bvalue.parquet` | nothing | — |

So there is exactly **one** cross-study code reader, and it is the plan's rule 3 case
("a study never reads another study's output"). That is what justifies the product.
`aos/output/bounce/{bending_mode_test_meta,dof_normalization_split}.parquet` sit at the
tree's top level, come from other scripts, and have no code reader; they are not part of
this product.

**The split is inside one script, and the interleaving is worse than `coadds`'.** PDFs
are written at `:702`, `:729`, `:743`, `:775`, `:807`, `:840`, `:907`; then
`bounce_fwhm_metric.parquet` at `:955` with its PDF at `:973`,
`bounce_fwhm_vs_bvalue.parquet` at `:997` with its PDF at `:998`,
`bounce_dof_stats.parquet` at `:1162`, `bounce_kj_stats.parquet` at `:1201`. But **all
the table data comes from one shared analysis block, `:479`-`:700`** — `fwhm_rows`,
`fwhm_bvalue_rows`, the per-night deltas and `df_kj` accumulate there and the plot
sections only consume them. So the builder is `:218`-`:700` plus the four `to_parquet`
calls, the study is the nine `PdfPages` blocks, and no physics moves.

**`bounce` therefore keeps a rule in `aos/Snakefile`**, unlike the three earlier
products: `bounce_summary.pdf` is a study output, so the rule shrinks to its plotting
half and reads the tables from the product. None of `cwfs_tables`, `miw` or `coadds` had
that shape.

### The four wrong-way imports need only three names

`smatrix/code/regularized_inversion/` reaches into `aos/code/bounce/` in four places, and
between them uses **three** names:

| import | file:line | names used |
|---|---|---|
| `import bounce_lib as _bl` | `run_oic_compare.py:231` | `_bl.run_bounce` (`:273`) |
| `import bounce_lib as bl` | `run_oic_compare.py:419` | `bl.run_bounce` (`:461`) |
| `import bounce_lib as bl` | `run_regularized_compare.py:65` | `bl.run_bounce` (`:114`), `bl.paired_deltas_matrix` (`:120`) |
| `import run_bounce` | `run_regularized_compare.py:66` | `run_bounce.DEFAULT` (`:205`) |

**Fix: a library, `rubinwork.bounce`**, carrying the leg-definition and pairing layer
only — `DEFAULT`, `DEFAULT_BOUNCES`, `DOF22`, `CAM_HEX_DOF`, `run_bounce()`,
`form_pairs`, `paired_delta`, `paired_deltas_kj`, `paired_deltas_matrix`,
`filter_visits`, `bounce_program_mask`. The ~20 `plot_*` functions of the 86 KB
`bounce_lib.py` stay with the study.

Why a library and not the product reader: these three callers never read the bounce
*tables*. They recompute the Δ from a fits table to score alternative inversions against
the bounce's own selection, which is library use — pairing logic — not product use.
Routing it through the product reader would make a study import a product's internals to
reach a pure function.

Why not leave it behind the shim until phase 5: the shim preserves the *path*, not the
*direction*. `run_bounce.DEFAULT` would still be a study importing a product **builder**,
which is the thing `/review` is meant to catch, and `DEFAULT` has to leave the builder
anyway.

One wrinkle to fix in the same commit: `bounce_lib.rbr_module()` (`:942`) does a
`sys.path` insert into `smatrix/code` to reach `regularized_inversion`, which is now the
installed `rubinwork.smatrix.regularized_inversion`. The moved library imports it
directly instead.

### Variants: three `dof_lut`, four `bounce_tables`, checked for old-tree output first

Checked before anything moved, as this handoff requires — every one of the twelve MIW
variants, against both old trees:

| product | variants WITH old-tree output | variants with none |
|---|---|---|
| `dof_lut` | `d12-A_50_34_i` (5 files), `d12-A_50_34_i_5rot` (5 files) | the other ten |
| `bounce_tables` | **none, under any MIW-variant directory name** | all twelve |

**`bounce_tables`** — four variants, keyed on the MIW variant **and** the fits table:

| variant | miw_variant | `fits` | old-tree dir | old-tree output? |
|---|---|---|---|---|
| `d12-A_50_34_i_5rot_july` | `d12-A_50_34_i_5rot` | `fits_july.parquet` | `bounce/danish_1_2_A_50_34_i_5rot_july/` | **yes, 13 files** — the tech note's source |
| `d12-A_50_34_i_5rot` | `d12-A_50_34_i_5rot` | `fits.parquet` | `bounce/danish_1_2_A_50_34_i_5rot/` | no |
| `d12-A_50_34_i` | `d12-A_50_34_i` | `fits.parquet` | `bounce/danish_1_2_A_50_34_i/` | no |
| `d13t-A_50_34_i_5rot` | `d13t-A_50_34_i_5rot` | `fits.parquet` | `bounce/danish_1_3_test_A_50_34_i_5rot/` | no |

The last three are the three jobs `aos/Snakefile` plans today. The first is the only one
with data, and **it is a separate variant rather than a build**, Aaron's call: its input
is a different fits table, `output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet` at
3,385 visits over day_obs 20250415–20260713, against the MIW build's own `fits.parquet`
at 1,126 visits over 20260315–20260513. Section 5 says a changed configuration makes a
new variant, and `aos/docs/status/miw_danish_1_3_proposal.md:116` calls `fits_july.parquet`
a "hand-made extra" produced by no rule. The plan's section 9 maps it to
`bounce_tables/d12-A_50_34_i_5rot/`, treating it as a build; that mapping is superseded.

`notes/aos-bounce-test-summary/provenance.md:46` names
`aos/output/bounce/danish_1_2_batoid/` as the intrinsic-choice comparison, but **that
directory does not exist on disk** — so it gets no variant, and the provenance doc points
at a path that is gone.

**`dof_lut`** — three variants, matching the three planned jobs:

| variant | fam_variant | mi_name | old-tree dir | old-tree output? |
|---|---|---|---|---|
| `d12-A_50_34_i_5rot` | `danish_1_2` | `pathA_50_34_i_5rot` | `lut/danish_1_2_A_50_34_i_5rot/` | **yes, 5 files** |
| `d12-A_50_34_i` | `danish_1_2` | `pathA_50_34_i` | `lut/danish_1_2_A_50_34_i/` | **yes, 5 files** |
| `d13t-A_50_34_i_5rot` | `danish_1_3_test` | `pathA_50_34_i_5rot` | `lut/danish_1_3_test_A_50_34_i_5rot/` | no |

Both carry `dir_name` as a quoted string, as `coadds` does. The YAML-integer trap does
not bite here (every name starts with a letter), but the field is quoted anyway and the
`coadds` test is mirrored.

**The `bounce` legs come from an `overrides:` block, not from the code defaults.**
`aos/analysis_config.yaml:180` carries a live `bounce:` override for
`fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` that redefines `bounces` as an elevation
*sweep* (five T720 legs, not one). `DEFAULT_BOUNCES` in `run_bounce.py:65` is only the
fallback. So the product's settings must come through `mc.analysis_section`, which does
accept an explicit `config_path` — unlike `param_sets.yaml` and `mi_config.yaml`, no
generated file is forced, but the override keyed on the **long** param_set key has to keep
resolving.

**Undeclared outputs, both rules.** `build_lut` declares 3 outputs
(`aos/Snakefile:799`-`:801`) but the script writes **5** files: those plus
`lut_by_rotbin.parquet` (`:205`, conditional on a rotator column) and `lut_config.yaml`
(`:241`). `bounce` declares 2 (`:880`-`:881`) and writes **13**. Both get full output
lists in the product Snakefiles.

### 3a step 1 — reference builds with the unmoved code: DONE

Built at commit `d5eeba6` (clean tree), interactive on `sdfiana`. **Neither product needed
its input sliced** — unlike `miw` and `coadds`, the full production run IS the smallest
thing that exercises the product code, so both references are the real thing:

| product | wall clock | output |
|---|---|---|
| `dof_lut` | **8.99 s** | 5 files, in `_phase2_refbuild/dof_lut/ref/` |
| `bounce_tables` | **167.6 s** | 13 files, in `_phase2_refbuild/bounce_tables/ref/` |

```bash
cd ~/notebooks/rubin-work/aos && python code/lut/run_build_lut.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --mi-name pathA_50_34_i_5rot \
    --tables-dir output/fam_processing/danish_1_2 \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/dof_lut/ref
```

```bash
cd ~/notebooks/rubin-work/aos && python code/bounce/run_bounce.py \
    --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x \
    --mi-name pathA_50_34_i_5rot \
    --fits output/miw/danish_1_2_A_50_34_i_5rot/fits_july.parquet \
    --out-dir /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/bounce_tables/ref \
    --min-detectors 160
```

**`dof_lut` reference.** 970 of 3,385 visits pass the band/program cut (band `i`,
`BLOCK-T614_triplets`), 863 survive the good-fit cut (63 bad-fit dropped); SVD
`U_eff = (126, 34)`, `n_dof = 50`, `n_keep_eff = 34`, `k = 1..6`.

| file | rows x columns | note |
|---|---|---|
| `lut.parquet` | 50 x 7 | one row per DOF, plus 8 fields of parquet **schema metadata** |
| `lut_dz.parquet` | 126 x 6 | one row per (focal k, pupil j) |
| `lut_by_rotbin.parquet` | 450 x 7 | 9 rotator bins x 50 DOF |
| `lut_config.yaml` | — | frozen provenance |
| `lut.pdf` | 45,415 bytes | |

Averaged DZ: raw RMS 0.0329 µm of wavefront, residual RMS 0.0109 µm of wavefront after
the 34-mode reconstruction. Nine populated rotator bins, centres −60 to +60 deg (OCS) in
15 deg steps, 16 to 254 visits per bin.

**`bounce_tables` reference.** Six legs over seven nights — `T720_elevation` at Elev=30,
40, 50, 60, 75 and `T724_rotator` at Rot=60 — with 7 `kind` values in `bounce_dof_stats`
(`dof`, `dof22`, `dof5`, `dof_oic`, `vmode`, `vmode22`, `vmode5`), so all five recovery
schemes are exercised.

| file | rows x columns |
|---|---|
| `bounce_kj_stats.parquet` | 1,890 x 17 |
| `bounce_dof_stats.parquet` | 2,550 x 24 |
| `bounce_fwhm_metric.parquet` | 6 x 9 |
| `bounce_fwhm_vs_bvalue.parquet` | 9 x 11 |
| 9 PDFs | 84 kB to 4.6 MB |

The reference **independently reproduces the published result**: on the `T720_elevation`
`Elev=40` leg, `fwhm_before = 0.2986 arcsec` and `fwhm_after_50_34 = 0.0381 arcsec`,
which is what `smatrix/docs/studies/regularized_inversion.md:89` quotes ("elevation 40
deg at 0.2986 to 0.0381 arcsec"). A rerun against the committed `_july` build also gave
identical shapes and column lists on all four tables.

**Comparison keys, verified unique rather than assumed** — and the plan-shaped guess
would have been wrong on the first table:

| table | key | unique |
|---|---|---|
| `bounce_kj_stats` | `(bounce, night, comparison, k, j)` | **1,890 / 1,890** |
| the same, without `night` | `(bounce, comparison, k, j)` | 756 / 1,890 — **NOT unique** |
| `bounce_dof_stats` | `(bounce, comparison, night, kind, index)` | **2,550 / 2,550** |
| `bounce_fwhm_metric` | `(bounce, comparison)` | **6 / 6** |
| `bounce_fwhm_vs_bvalue` | `(bounce, comparison, night)` | **9 / 9** |
| `lut` | `(dof_index)` | **50 / 50** |
| `lut_dz` | `(k, j)` | **126 / 126** |
| `lut_by_rotbin` | `(rotbin_center, dof_index)` | **450 / 450** |

No key column holds a null on any table. Two schema facts the comparison has to carry:
`night` is a **string** on `bounce_kj_stats` and `bounce_dof_stats` — with a literal
`'all'` row for the pooled result — and an **int64** on `bounce_fwhm_vs_bvalue`; and
`lut.parquet`'s 8 parquet schema-metadata fields (`param_set`, `mi_name`, `reduce`,
`n_dof`, `n_keep`, `n_visits`, `k_min`, `k_max`) are part of the product and are compared
too. No earlier product's comparison checked schema metadata.

The PDFs are not byte-comparable (matplotlib embeds a creation timestamp), so step 4
compares the parquet tables at zero tolerance and checks only that each PDF is produced
at a comparable size.

**Expected `aos/` job count after the move: 37 → 31**, −3 for `build_lut` and −3 for
`bounce`, since both are in `rule all`. `rule all` loses `output/lut/{d}/lut.parquet` and
`output/bounce/{d}/bounce_kj_stats.parquet` as targets it builds; the `bounce` rule itself
survives as the plotting half, so the drop is accounted rule by rule in step 3.

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
- **3b: fold `d12-refitWcs_2025` into `d12-refitWcs`** as a date-keyed collection list.
  Aaron, 2026-10-10: they are one variant, differing only in the collection that covers
  2025. Needs a builder change — `run_wfs_mktable` opens one Butler per `--wfs-name` —
  so it was out of scope for 3a's mechanical move. The two collections are disjoint in
  `day_obs`, so the split is clean. Doing it also removes the never-built variant that
  forced the `wfs_variants()` filter in step 3.
- **3b: retire `danish_1_0` whole**, both the FAM variant and `d10-wep17_3_0`. Aaron,
  2026-10-10. Four live things resolve the long param_set key and must be handled in the
  same commit: the staged-MIW provenance chain
  (`aos/calibration/miw/intrinsic_split_maps_v1.provenance.yaml`, `stage_miw.py`,
  `calibration/README.md`), `notes/aos-measured-intrinsics/make_figures.py:23` and
  `provenance.md:5`, the **live** `overrides:` block at `aos/analysis_config.yaml:148`,
  and `test_fam_tables.py:263`. Both halves are carried as `registered: false` until then,
  which is what keeps `aos/param_sets.yaml` byte-identical.
- **An incomplete build breaks a plain `snakemake -n` in `aos/` once its producer moves.**
  Met twice now: `refitWcs_2025` cost 5 jobs in `cwfs_tables`, and nine incomplete MIW
  builds plus `tarts`'s missing corner sidecar cost 75 in `miw`. Both fixed by filtering on
  what is on disk and **reporting every skip**. `aos/Snakefile` has the idiom in three
  places to copy. `coadds` did not hit it — the moved script had no rule there — but
  `dof_lut` and `bounce_tables` both have rules in `rule all`, so they will.
  **Largely superseded by `fec3741`**: a variant whose build exists in the NEW tree now
  resolves through the catalog instead of being skipped, so the gates only drop a variant
  that is built in neither tree. The skip-and-report idiom still matters for that case.
- **Nine of the twelve MIW builds are incomplete in the old tree**, so the 3c rebuild has
  more to do for this product than a re-run: six have the split but no sidecar or refit,
  and three (`danish_1_3_test_A_50_34_i`, `danish_1_3_v1000_A_50_34_i`,
  `danish_1_3_v1000_A_50_34_i_rbr`) have nothing but their per-bin grids. Decide in 3c
  which are wanted — several are parents that exist only to be `build_from` sources, which
  is a legitimate reason to have grids and nothing else.
- **`refit_mi` reads the whole FAM `donuts.parquet`**, 12.36 GB and 9,083,526 rows for
  `danish_1_2`, at about 89 s per 475,128 donut rows measured on the reference — so roughly
  **28 min per MIW variant** for the full table, before `intrinsic_sidecar`'s share. Twelve
  variants is a 3c cost worth planning rather than discovering.
- **The plan's 24-character variant-name limit is raised to 32** (section 5), because
  `d13v1000-A_50_34_i_rbr_5rot` and `d13v1000-A_50_50_i_rbr_5rot` are 27 and have to
  round-trip to their `dir_name`.
- The timing runs wrote throwaway output to
  `_phase2_refbuild/_timing_step5/{nothermal,thermal,n15}/`. Not deleted — file deletion
  needs Aaron's OK.
- 3a left throwaway and reference directories under `_phase2_refbuild/`, none deleted
  (file deletion needs Aaron's OK):
  - `cwfs_tables/`: `_fam_20251126` the staged FAM night, `refitWcs_2025_20251126` the
    reference, `refitWcs_2025_20251126_moved` the rerun.
  - `miw/`: `_fam3n` the staged three-night FAM slice (507 MB), `ref` the reference, `moved`
    the rerun, `mi_config_ref.yaml` the throwaway config, and
    `_fam_20260315_20260317/` — the **first, unusable** staged slice, whose single-row-group
    donuts.parquet the builder cannot read. That one is pure waste; the rest should stay
    until 3c.
  - `coadds/`: `_fam_20260316` the staged one-night FAM slice (129 MB), `ref` the
    reference (5.4 MB), `moved` the rerun (5.4 MB).
  - `_scratch_3a/` the scratch data root, now holding placeholder `fam_tables`,
    `cwfs_tables`, `miw` and `coadds` manifests used only to plan the product Snakefiles
    and to show the `aos/Snakefile` resolver picking up a new-tree build. Written by
    `aos/code/infra/stage_scratch_product.py`, which refuses any root not under a
    `_scratch_3a` directory.
- `rubinwork/products/miw/tests/__init__.py` was added, which
  `rubinwork/products/{cwfs_tables,coadds}/tests/` do not have. Harmless — pytest collects
  either way — but inconsistent; remove it when a deletion is being approved anyway.
- **`block_grids_ext.npz` and `coadd_metrics.parquet` have no reader at all.** Kept in the
  product on Aaron's call 2026-10-10 — the former is the extra-pupil-Zernike primitive the
  retrieval-bias work needs and is what distinguishes the two coadd variants; the latter is
  the natural per-coadd table, and the only metrics file anything reads
  (`coadd_metrics_rebin3.parquet`) is a **study** output of `recompute_coadd_metrics.py`.
  Worth revisiting in 3c: a full build's `block_grids_ext.npz` is 66 MB that nothing opens.
- **`build_lut` reads its `fits.parquet` from the FAM tables, not from the MIW refit**
  (`aos/Snakefile`, `fits = d_tables(w.d)/fits.parquet`), even though the rule is keyed on
  the joined `{d}` and every other `{d}` rule reads `output/miw/<d>/fits.parquet`. Confirm
  that is intended before moving `dof_lut`, rather than discovering it in the comparison.
- **The coadd variants' `_v2` suffix is carried by `ext_terms` alone.** If
  `block_grids_ext.npz` is ever dropped the two variants become identical and should
  collapse into one.
- **Two coadd study scripts still hardwire `output/coadd/danish_1_2/50_34/`** as a
  `--blocks-summary` default (`analyze_dz_goodness_of_fit.py:293`,
  `analyze_umode_null_coupling.py:68`). Left alone deliberately: that path **exists** and
  works today, and both take the flag explicitly, so a catalog build is reachable by
  passing its path. They need the catalog only when phase 5 retires the old tree.
- **`analyze_miw_field_order.py` has no argument parser** — 110 lines, no functions, it
  runs top to bottom and reads at module level, so merely importing it needs the products
  on disk. Its builds are named by `COADD_VARIANT`, `COADD_MIW_VARIANT` and
  `COADD_STUDY_DIR` because there was nowhere to put a flag. Give it a `main()` when the
  coadd study is reshaped in phase 5.
