# rubin-work organization plan: products, libraries and studies

> **Status:** current · **Last updated:** 2026-10-09 · **Kind:** working state (plan)

The plan for reorganizing `rubin-work` around **products** (data that other work reads),
**libraries** (code that other work imports) and **studies** (work that asks a question,
mostly by making plots from products). It records the decisions Aaron made on 2026-10-07
to 2026-10-09, the target layout of the repository and of the data on S3DF (SLAC Shared
Scientific Data Facility), and the migration order. It replaces
`reorg_review_plan_2026-09.md`, `step1_structure_decisions.md` and `notes/status/todo.md`,
which remain readable at tag `pre-products-reorg-2026-10-09`:

```bash
git show pre-products-reorg-2026-10-09:notes/status/reorg_review_plan_2026-09.md
```

## Contents

1. [Decisions](#1-decisions)
2. [Concepts and rules](#2-concepts-and-rules)
3. [Target layout](#3-target-layout)
4. [Products](#4-products)
5. [Versions, manifests and the catalog](#5-versions-manifests-and-the-catalog)
6. [The value-added database](#6-the-value-added-database)
7. [Studies and the logbook skills](#7-studies-and-the-logbook-skills)
8. [Migration phases](#8-migration-phases)
9. [Output mapping, first draft](#9-output-mapping-first-draft)
10. [Carried-over open items](#10-carried-over-open-items)
11. [Open questions](#11-open-questions)

## 1. Decisions

| decision | outcome |
|---|---|
| unit of work | the study, below a topic; a topic is a folder of studies |
| products | 9: value-added database, FAM tables, CWFS tables, MIW and sidecar, coadds, guider moments, HSM moments, DOF LUT, bounce tables |
| libraries | `common`, `aos_state` (includes the v-modes), `smatrix` (`compute_smatrix`, `normalization_weights`, `regularized_inversion`), `open_loop` |
| package | one installable package `rubinwork/` at the repo root, `pip install -e .`, holding all libraries and all product code |
| data root | `/sdf/group/rubin/u/roodman/LSST/rubin-work/` with `products/` and `studies/`; not in git |
| laptop data | syncs chosen parts of `studies/` only, never `products/` |
| version naming | `<product>/<variant>/<build>`; variant = named configuration; build = date `YYYYMMDD` |
| derived products | variant name prefixed by the short code of its main upstream variant, e.g. `d12-A_50_34_i_5rot` |
| incremental products | value-added database and guider moments: `v<schema>`, provenance per night |
| official MIW | Guillem's `ts_intrinsic_wavefront` calibrations registered in the catalog as external variants |
| MIW sidecar | anything the official MIW needs stays in `ts_intrinsic_wavefront` and its sidecar, which cannot depend on the private database; per-donut data stays in parquet |
| old output | left untouched during the reorg and kept as the "before" for comparing rebuilds; Aaron zips it for temporary backup afterwards |
| populating the new tree | **revised 2026-10-10: rebuild, do not copy.** Code moves for all products first (mechanical, checked by small reference builds); then the intended fixes; then every product is rebuilt fresh into the new tree and compared with the old one. No existing build is copied or registered |
| obsolete builds | never rebuilt or registered ("virtual archive"): they stay only in the old tree. Includes FAM `danish_1_0`, `danish_1_1_1`, `danish_1_2_0_wep17_7_0_2025` |
| package name | `rubinwork`: personal use only; anything Rubin adopts officially moves to a `ts_*` package |
| HSM moments | product deferred until its study restarts |
| code review | after the reorg; a reference-run check guards each move |

Acronyms: Full Array Mode (FAM), Corner Wavefront Sensor (CWFS), Measured Intrinsic
Wavefront (MIW), Degree Of Freedom (DOF), Look-Up Table (LUT), Engineering Facility
Database (EFD), Consolidated Database (ConsDB), Double Zernike (DZ), Higher-order Shape
Moments as computed by galsim (HSM).

## 2. Concepts and rules

- **Product**: code that writes data other work reads. It has variants, builds and a
  manifest, and is read only through the catalog.
- **Library**: code that other work imports. It writes no data of its own.
- **Study**: answers a question, usually by plotting product data. It ends with an answer
  and a status, and nothing imports it.

Dependency rules, enforced by `/review`:

1. Studies may import libraries and read products. Products may import libraries.
2. Libraries never import products. Nothing imports a study.
3. A study never reads another study's output. If two studies need the same data, it
   becomes a product (or a small one: see the DOF LUT and bounce tables).
4. Code finds product data only through `rubinwork.products.catalog`, never by building a
   path.
5. A product build directory is never edited after it completes. A rerun makes a new
   build.

Wrong-way imports found on 2026-10-08, to remove during migration:

| importer | imports from | fix |
|---|---|---|
| `aos/code/psf_maps_lib.py` | `run_wfs_mimic._wedge_medians` (CWFS study) | move `_wedge_medians` into a library |
| `smatrix/code/regularized_inversion/` | `aos/code/bounce/` (4 imports) | bounce tables become a product; shared code into a library |
| `aos/code/closed_loop/`, `aos/code/psf/` | `aos/code/cwfs/` | shared code into a library |
| `optatmo` | reads `wfs_corner_compare.parquet` (CWFS study output) | promote to a small product or recompute |
| `blocks` | reads `t<N>_closedloop_aos_*.parquet` from `aos` | same |

## 3. Target layout

### Git repository

```
rubin-work/
  pyproject.toml
  CLAUDE.md  README.md
  STUDIES.md                    generated: every study, status, answer, products used
  PRODUCTS.md                   generated: every product, variants, latest build, size
  .claude/skills/               start-study, wrap, review, plus the style skills
  data@                         gitignored symlink to the data root
  rubinwork/
    common/                     library (today's common/)
    aos_state/  smatrix/  open_loop/          libraries
    products/
      catalog.py                load() / path() / register(); reads manifests
      manifest.py               writes manifest.json for a build
      fam_tables/
        __init__.py             load(variant, build="current")
        build_*.py  Snakefile   python -m rubinwork.products.fam_tables.build
        variants.yaml           short variant name -> full configuration
        tests/  README.md       schema and version history
      cwfs_tables/  miw/  coadds/  value_added/
      guider_moments/  hsm_moments/  dof_lut/  bounce_tables/
  aos/                          topic: studies only; README.md is a generated index
    coadd_vs_miw/
      study.md                  front matter, question, current answer, dated log
      code/  notebooks/
      figures/                  a few committed PNGs cited by the log
      output@                   gitignored symlink to studies/aos/coadd_vs_miw
    ...
  guider/  thermal_focus/  wfs/  optics/ ...      topics, studies only
  notes/                        outward-facing drafts (Slack posts, tech notes)
```

Imports become `from rubinwork.common import utils` and
`from rubinwork.products.fam_tables import load`. The `sys.path.insert` idiom and the
`parents[N]` rule in `CLAUDE.md` go away.

### Data root on S3DF

```
/sdf/group/rubin/u/roodman/LSST/rubin-work/
  products/
    fam_tables/danish_1_2/
      current@ -> 20261005
      20261005/  manifest.json  donuts.parquet  visits.parquet  fits.parquet  _work/
    miw/d12-A_50_34_i_5rot/20261007/   intrinsic_grid, intrinsic_split_*, zk_intrinsic
    miw/official-<release tag>/        manifest only, pointing at the official output
    value_added/v1/  live/  snapshots/<YYYYMMDD>/  current@  _work/
    guider_moments/v6/nights/<YYYYMMDD>/
    ...
  studies/<topic>/<study>/<YYYYMMDD>_<label>/   PDFs, PNGs, run.json, study-only scratch
```

This is a new tree, separate from today's `.../LSST/notebooks/rubin-work/<topic>/output/`
so the two layouts never mix.

## 4. Products

| product | kind | code today | output today | first variants |
|---|---|---|---|---|
| `value_added` | incremental | `value_added/code/` | `value_added/output/aos_efd.duckdb` (2.80 GB) + shards | `v1` |
| `fam_tables` | snapshot | `aos/code/fam_processing/`, Snakefile rules `mktable`, `fit`, `combine_*`, `attach_telemetry` | `aos/output/<param_set dir>/` | today's `param_sets.yaml` `dir_name`s: `danish_1_0`, `danish_1_2`, `danish_1_3_test`, `danish_1_3_v1000`, ... |
| `cwfs_tables` | snapshot | `aos/code/cwfs/run_wfs_mktable.py`, rule `wfs_mktable` | `.../wfs/<wep version>/` under each param_set | per param_set and CWFS collection |
| `miw` | snapshot + external | `aos/code/miw/`, rules `build_intrinsic`, `intrinsic_split`, `intrinsic_sidecar`; `ts_intrinsic_wavefront` | `aos/output/miw/<param_set>_<mi_name>/` | `mi_config.yaml` entries, prefixed: `d12-A_50_34_i_5rot`, `d13v1000-A_50_50_i_rbr`, ... |
| `coadds` | snapshot | `aos/code/coadd/` | `aos/output/coadd/<param_set>/50_34*/` | `d12-50_34`, `d12-50_34_v2` |
| `guider_moments` | incremental | `guider/code/`, `guider/Snakefile` | per-night `<seq>_moments`, `_stars`, `_metrics` parquet (4,763 files each) | `v6` (current `guiderMoments` schema) |
| `hsm_moments` | deferred | `optatmo/code/moments_hsm.py` | none yet: the study has not started | defined when the study restarts |
| `dof_lut` | snapshot | `aos/code/lut/` | `lut`, `lut_dz`, `lut_by_rotbin` parquet | per MIW variant |
| `bounce_tables` | snapshot | `aos/code/bounce/` (table-writing part) | `bounce_fwhm_metric`, `bounce_dof_stats` parquet | per MIW variant |

Snapshot products are built whole; incremental products grow night by night. External
products are written elsewhere and only registered.

## 5. Versions, manifests and the catalog

**Variant.** A named configuration, defined in the product's `variants.yaml`. Lowercase,
at most 32 characters, never reused for a different configuration. Changing the
configuration (collections, programs, rotator bins, correction scheme) makes a new variant.
(Raised from 24 on 2026-10-10: two `miw` variants are 27 —
`d13v1000-A_50_34_i_rbr_5rot` and `d13v1000-A_50_50_i_rbr_5rot` — and they have to
round-trip to their `mi_config.yaml` `dir_name`, so shortening them would break the
mapping.)

**Build.** One run of a variant, named by date (`20261007`, then `20261007b` for a second
run that day). A code change that alters results, run on the same configuration, makes a
new build; the two manifests show why the numbers differ. `current@` names the default
build.

**Derived products** prefix the variant with the short code of their main upstream variant
(`d12` for FAM `danish_1_2`). Every other input is named only in the manifest. This
replaces the 59-character joined `<param_set>_<mi_name>` directory names.

**Incremental products** carry a schema version, `v<N>`, instead of dated builds. Each
night's provenance is recorded per night. An incompatible schema change (a column renamed,
removed or redefined) starts `v<N+1>`; added columns, added variants and appended nights do
not.

**`manifest.json`**, written by `rubinwork.products.manifest` at the end of every build:

```json
{
  "product": "miw", "variant": "d12-A_50_34_i_5rot", "build": "20261007",
  "status": "complete", "created": "2026-10-07T14:02:11",
  "git_commit": "6377e5b", "git_dirty": false,
  "config": {"...": "the full variants.yaml entry, expanded"},
  "inputs": {"fam_tables": "danish_1_2@20261005"},
  "files": {"intrinsic_grid.parquet": {"rows": 1234, "bytes": 5678}}
}
```

**The catalog.** `rubinwork.products.catalog` reads the manifests:

- `load(product, variant, build="current")` returns the data (DataFrame, DuckDB connection
  or path, by product).
- `path(product, variant, build="current")` returns the build directory.
- `list()` lists everything; `PRODUCTS.md` is generated from it.
- The data root comes from one environment variable, `RUBINWORK_DATA`, defaulting to the
  S3DF path, so the same code runs on the laptop, the RSP (Rubin Science Platform) and
  batch nodes.
- External variants (`miw/official-<tag>`, Butler collections) are manifests with a
  `location` field and no files.
- Later, `load()` can return DuckDB views over the parquet files, so one connection
  reaches every product without copying. Not needed for the first version.

**Study runs.** Every study run writes `run.json` into its dated directory: git commit,
the product builds it read, and the command or notebook. `/wrap` copies the builds into
the study's `uses:` front matter.

## 6. The value-added database

An incremental product. Its builders already run per night, append with `--resume`,
record fetches in `fetch_log`, and keep variants as rows (`state_variant`, `fam_variant`).
That design stays.

```
products/value_added/v1/
  live/aos_efd.duckdb          only the builders and the nightly job write here
  snapshots/<YYYYMMDD>/aos_efd.duckdb
  current@ -> snapshots/<latest>
  _work/shards/
```

- DuckDB's file lock excludes readers while a writer is open (`value_added/code/run_build.sh`).
  So after each update the job copies `live` to a dated snapshot and moves `current`;
  studies and notebooks open `current` read-only.
- Retention: the last 7 nightly snapshots plus one per month, about 50 GB at 2.8 GB per
  snapshot. Pruning is listed for Aaron to approve, never automatic.
- Provenance: add the code git commit and build time to `fetch_log`.
- Registry rows (`state_variant.intrinsic_ref`, `fam_variant.param_set`,
  `fam_variant.fits_path`) refer to catalog names such as `miw:d12-A_50_34_i_5rot@20261007`
  instead of absolute paths and full `param_set` names. This removes the silent join
  breakage on renames that `value_added/docs/schema.md` warns about.
- An occasional full rebuild is built in `_work/`, compared with `live`, then swapped in.
  It is a new snapshot, not a new schema version, unless the schema changed.
- Nightly job, once observing resumes: process the previous `day_obs` with `--resume`,
  then redo the 2–3 nights before it to catch late ConsDB and quicklook data. The state
  builder can run under Slurm `scrontab`. The telemetry builder today reaches the EFD only
  from interactive nodes (`slacrd`, `sdfiana*`). Aaron has been told the EFD can be reached
  from batch nodes after some setup; if so, both builders run under `scrontab`. Otherwise
  the telemetry builder needs a scheduled job on an interactive node. To be checked.

## 7. Studies and the logbook skills

A study is a directory `<topic>/<study>/`. Small studies can be `study.md` plus one
notebook.

`study.md` front matter:

```yaml
---
study: coadd_vs_miw
topic: aos
status: active            # active | parked | done | dropped
question: Why do per-block FAM coadds disagree with the pooled MIW?
answer: Retrieval bias in the Danish fit redistributes power among radial orders.
uses:
  fam_tables: danish_1_2@20261005
  miw: d12-A_50_34_i_5rot@20261007
started: 2026-08-02
---
```

Then the sections **Question**, **Current answer** (kept true, with numbers and units),
and **Log** (dated entries, newest first: what was done, results with units, links to
`figures/` and to dated run directories, git commit). Handoffs become log entries. Longer
derivations sit beside `study.md` as their own `.md` files.

Skills:

| skill | does |
|---|---|
| `/start-study` | takes a short description; proposes topic, name and products used; on confirmation creates the directory, `study.md`, the `output@` symlink and the data directory, and adds it to `STUDIES.md`. Also handles "this is a product change" by pointing at the product's `variants.yaml`. Replaces `rubin-new-study` |
| `/wrap [study]` | from git log since the last entry plus new runs: appends a dated log entry, updates the answer, status and `uses:`, copies one or two key PNGs to `figures/`, commits. On an old study the first run rebuilds a starting entry from history and existing docs |
| `/review [topic]` | regenerates `STUDIES.md` and `PRODUCTS.md`; flags active studies untouched for more than 14 days, done studies without a conclusion, code not covered by any study, rule violations from section 2, studies pinned to superseded builds |

`rubin-output-layout` is replaced by sections 3 and 5; `rubin-notebooks` and
`rubin-doc-style` are updated for the study layout.

## 8. Migration phases

Each phase ends with a commit and a push. A topic being migrated is frozen for other
sessions until its phase is pushed (see `notes/status/parallel_claude_sessions.md`).

**Phase 0 — done 2026-10-09.** Tag `pre-products-reorg-2026-10-09`; old plan docs removed.

**Phase 1 — package skeleton and libraries. Done 2026-10-09**, commits `c932808`,
`ee89a96`, `30502ae`, `547fc63`, `2918a2b`, `cee055e`.

1. `pyproject.toml` — `rubinwork`, setuptools, `requires-python = ">=3.11"`, no
   dependencies (everything comes from the stack environment). `packages.find` includes
   only `rubinwork*` and excludes its `tests`, so the topic directories that carry an
   `__init__.py` (`aos/`) do not install. (`c932808`)
2. `common/` moved to `rubinwork/common/` (`ee89a96`); `aos_state`, `open_loop` and three
   `smatrix` modules moved into `rubinwork/` (`30502ae`). Shims at every old path.
3. `rubinwork/products/catalog.py` and `manifest.py`, 26 tests passing against a pytest
   `tmp_path` data root. (`547fc63`)
4. Data root created: `/sdf/group/rubin/u/roodman/LSST/rubin-work/{products,studies}`,
   both empty; `catalog.list_products()` reads them and returns `[]`.
5. Installed with `pip install --user -e .` in the stack environment (`w_2026_39`, Python
   3.13.15). `import rubinwork, rubinwork.common, rubinwork.products.catalog` verified in
   **three of the four** environments: the USDF terminal, an RSP notebook cell, and a
   Slurm batch node (`rubinwork/common/scripts/check_batch_import.sl`, commit `2918a2b`,
   job 40395908 on `sdfmilan257`, `RESULT: pass`, which also checked the `common` and
   `aos_state` shims there). Laptop: done, `/opt/local/bin/pip3 install --user -e .` into
   MacPorts Python 3.13; the imports pass from outside the repo and both shims give the
   same module objects. The laptop has no `pytest` or `rg`, so the catalog tests and the
   import smoke test run on S3DF only.

**Which `smatrix/code` modules are libraries.** The three that code outside the smatrix
study imports, and that are import-safe (constants and functions, work behind
`__main__`):

| module | imported by |
|---|---|
| `compute_smatrix.py` | `aos/code/static_optics/camera_gravity.py`, `smatrix/code/thermal_sensitivity.py`, three smatrix plot scripts (for `CONFIG_TAG`) |
| `normalization_weights.py` | `regularized_inversion.py`, `make_normalization.py`, `full_normalization.py`, `validate_normalization.py`, `vmode/plot_vmode_dof_matrix.py` |
| `regularized_inversion.py` | `value_added/code/build_optical_state.py` (and its test), `aos/code/test_open_loop.py`, `aos/code/bounce/bounce_lib.py`, `aos/code/miw/{check_dof_ranges,test_rbr_against_prototype}.py` |

Everything else in `smatrix/code/` is a study script and stayed: the `plot_*` and
`mode_gallery*` scripts, `compare_ofc.py`, `detailed_comparison.py`,
`demo_field_order.py`, `full_mode_analysis.py`, `miw_ocs_analysis.py`,
`pupil_zernike_study.py`, `thermal_sensitivity.py`, `thermal_study.py`,
`build_full_modes.py`, `validate_normalization.py`, and the `vmode/` and
`regularized_inversion/` subdirectories.

The **v-modes library named in section 3 does not exist in `smatrix/code`**:
`smatrix/code/vmode/` is a study (two analysis scripts, `docs/studies/vmode.md`), and the
v-mode code other topics actually import is `aos_state`, already moved. `optatmo` imports
`vmode_fit`, which is in `optatmo/code/`. Nothing extra to move.
`make_normalization.py` and `full_normalization.py` *write* `normalization_all.npz`, so
they are product builders for a later phase, not libraries.

**How the shims work.** `common/__init__.py` imports `rubinwork.common`, copies its
`__path__`, and registers each submodule in `sys.modules` under the old `common.*` name,
so `common.utils is rubinwork.common.utils`. The five bare-name libraries
(`aos/code/aos_state.py`, `aos/code/open_loop.py`, and three in `smatrix/code/`) are
module files that bind the real module's namespace and then set
`sys.modules[__name__] = _real`, since every caller imports them by bare name after a
`sys.path.insert`. No shim holds a copy of anything.

**Regression smoke test** (`rubinwork/common/scripts/import_smoke_test.py`, committed in
`c932808`): imports every repo module that touches a moved library, one per subprocess, in
script mode. Before: 72 modules, 69 ok. After: 64 modules, 64 ok. **No module's import
regressed.** Accounting for the renames, two outcomes changed, both improvements:
`aos/code/bounce/{bounce_lib,run_bounce}.py` went from timeout to ok (warm caches, not the
move). The 8 modules that dropped out of the match set are the moved files themselves,
whose imports became intra-package; each was verified to import directly.

**Fixed in passing, not planned.** `rubinwork/common/psf_moments_consdb.py` imported
`common.telemetry_clients` by absolute name, which failed before the move too
(`ModuleNotFoundError`); it is a relative import now and the module imports.
`rubinwork/common/FocalPlaneInterpolator.py` still fails on `numpy.lib.index_tricks`,
removed in numpy 2 — a pre-existing break, untouched, and the shim propagates it
unchanged.

**Tried and rejected.**
- *Making the shim a one-line `from rubinwork.common import *`.* It does not give module
  identity: `common.utils` would be a different object from `rubinwork.common.utils`, so
  monkeypatching or module-level state would silently split. The `sys.modules`
  registration is what keeps them the same object.
- *Keeping the `sys.path.insert` + `parents[1]` hack inside the moved `common` modules.*
  `parents[1]` is one level too shallow after the move. The three intra-package sibling
  imports (`consdb_efd`, `dof_telemetry`, `visit_telemetry`) became relative instead,
  which is correct inside an installed package and removes the hack.
- *Letting `packages.find` default.* It picks up `aos/`, which has an `__init__.py`, and
  would install a topic directory as a package. `include = ["rubinwork*"]` is required,
  not cosmetic.
- *Naming the catalog listing function `list`.* Section 5 calls it `list()`; it is
  defined as `list_products` and aliased, so the builtin is not shadowed for callers of
  the module's other functions.

**Left open.**
- `smatrix/code/regularized_inversion/` is a namespace package (no `__init__.py`) sitting
  next to `regularized_inversion.py`. The `.py` wins, so imports resolve to the module —
  but that was already true before the move and is worth removing in phase 5 when the
  study directories are reshaped.
- `common/notebook_template.ipynb` and `common/output/` stayed at the old path, since
  `CLAUDE.md` and the output-layout convention point at them. They move in phase 5.

**Phase 2 — pilot product: `fam_tables`, end to end. Done 2026-10-10**, commits `58a6606`,
`b355446`, `d3f44a5`, `cbe3b12`, `c757462`, `29dbebc`, `ed3417f`, `26dd61c`, `7137ea3`,
`06039ab`, `0781229`, `f33b0ab`, `2b50ed0`, `04a8871`. State in
`notes/status/phase2_fam_tables_handoff.md`.
1. **Done.** Two reference runs before the move, one per builder: `danish_1_2`, chunk
   `20251116_20251130`, 6 nights (Snakemake path), and `danish_1_3_v1000`,
   `day_obs=20260428` (blitz path). Both built with the unmoved code.
2. **Done.** The four builders moved into `rubinwork/products/fam_tables/builders/` with
   shims at the old paths, and the six build rules into the product's own `Snakefile`.
   `variants.yaml` carries both the FAM and the WFS/CWFS halves — the FAM/CWFS triplet
   link is one fact and needs one home — and `aos/param_sets.yaml` is a **generated**
   file. It cannot be deleted, even in phase 5: the external
   `intrinsics_lib.load_param_sets()` reads it by name from the cwd, which is also why the
   product's `mktable` rule runs with `aos/` as its cwd. `snakemake -n` in `aos/` went
   212 → 199 jobs, the drop being exactly the four moved rules' 13 pending jobs.
3. **Done, exact match.** Both references reproduced with the moved code and compared at
   zero tolerance: **0 differing columns** of 48/600/653 (Snakemake path) and 66/23/448
   (blitz path). `mktable` was deliberately **not** rerun (it stays in
   `ts_intrinsic_wavefront`, and its EFD thermal loop is 47 of the reference's 53 min), so
   the known `rotator_angle` non-reproducibility could not surface — a future comparison
   that does rerun it will see up to 0.188 deg of camera rotator angle differ on the 19
   visits where ConsDB `physical_rotator_angle` is NULL.
4. **Dropped 2026-10-10** (revised route): existing builds are not copied or registered;
   every product is rebuilt fresh in phase 3c.
5. **Done.** Both FAM builders write `manifest.json` (`status: complete`, expanded config
   with per-chunk collections) as their last step, through the shared
   `fam_tables/write_manifest.py`; the catalog skips builds without a complete manifest;
   `current` moves only through `python -m rubinwork.products.catalog set-current`, never
   as a side effect of a build. Both references rerun with the final code and still match
   at zero tolerance: **0 differing columns** on all three tables of both paths. Studies
   switch to `load()` in phase 5, once rebuilt products exist.

**Phase 3 — the other products: move, fix, rebuild.** Revised 2026-10-10 from "copy and
register" to three passes over all products. State in
`notes/status/phase3_products_handoff.md`, which holds the measured 3c feasibility numbers.

*3a. Move the code, mechanically.* In dependency order: `cwfs_tables` (**done**); `miw`
(**done**, plus the official-MIW registrations, which are manifests only); `coadds`;
`dof_lut` and `bounce_tables`; `value_added` (code only; the `v1/live` + snapshots layout
comes with the rebuild); `guider_moments`. Imports and paths only, no change in results. Each product is
checked by a small reference build made with the unmoved code and reproduced with the
moved code at zero tolerance, plus `snakemake -n` job-list identity and the import smoke
test. Every builder writes its manifest. Split the shared `aos/Snakefile` as each
product's rules leave it; what remains are study rules, which move with their studies in
phase 5. `hsm_moments` stays deferred.

**`cwfs_tables` done 2026-10-10**, commits `71a345f` (reference build) and `b880f2d` (the
move). `run_wfs_mktable.py` and the `wfs_mktable` rule moved; the other five
`aos/code/cwfs/` scripts are the CWFS-vs-FAM study and stayed, as did
`wfs_intrinsic_sidecar`, which is a `miw` rule. `cwfs_tables/variants.yaml` now owns the
FAM/CWFS triplet link with five registered variants plus one provenance-only entry;
`gen_param_sets.py` merges it with `fam_tables/variants.yaml` and `aos/param_sets.yaml` is
**byte-identical** (md5 `0f72da651cf9319cc283754c614051fd`). The reference reproduced
exactly: 0 differing columns of 20 in `donuts.parquet` and of 8 in `visits.parquet`, at
zero tolerance. `snakemake -n` in `aos/` went 199 -> 189 jobs; 5 of the 10 are the moved
rule and the other 5 are the never-built `refitWcs_2025` variant's study jobs, which lost
their producer — see the phase 3 handoff, which records the `wfs_variants()` filter that
fixes it and warns that `miw` and `coadds` should expect the same. Two items deferred to
3b: folding `d12-refitWcs_2025` into `d12-refitWcs`, and retiring `danish_1_0` whole.

**`miw` done 2026-10-10**, commits `d81cc3d` (reference build) and `b9756aa` (the move).
Five rules moved — `build_intrinsic`, `intrinsic_split`, `intrinsic_sidecar`,
`wfs_intrinsic_sidecar` and `refit_mi`, the last because its `fits.parquet` is a function
of the MIW variant alone while `fam_tables` owns the plain `fit`. No builder code moved:
the runners are external, in `ts_intrinsic_wavefront/bin/`. `miw/variants.yaml` owns the
twelve measured-intrinsic configurations and **generates `aos/mi_config.yaml`**, whose body
is byte-identical, because the external package opens `Path('mi_config.yaml')` relative to
the working directory and ten things read it that way. The configuration is held as
**literal text blocks** rather than parsed data: `safe_dump` discards the comments inside
entries and a `ruamel` round-trip through a reshaped structure moves them, so
concatenating text makes identity true by construction. `aos/code/miw_io.py` became the
product reader and `aos/code/miw_corner_intrinsic.py` the `rubinwork.miw_corner` library,
with shims at both old paths; `decomp_path` was **broken** before the move (it joined the
two long keys as a layout the tree lost when `dir_name` arrived) and is fixed. The official
MIW and the staged in-repo copy are three **external** variants — manifests with a
`location` and no files. The reference reproduced exactly: **0 differing columns across 12
tables** at zero tolerance. `snakemake -n` in `aos/` went 189 -> 37 jobs; 77 are the moved
rules and the other 75 are nine incomplete MIW builds and `tarts`'s missing corner sidecar
losing their study jobs, every one enumerated in the phase 3 handoff.

*3b. Apply the intended fixes*, each as its own commit with its own before/after check on
the reference builds:
- `rotator_angle`: ConsDB `physical_rotator_angle` first, then `meta['rotTelPos']`
  (radians in `aggregateAOSVisitTableRaw`, degrees in `donutBlitzFamResults`) instead of
  the EFD window mean and `visitInfo` fallbacks; rename `skyAngle` to `rotTelPos` in
  `intrinsics_lib.py`. In `ts_intrinsic_wavefront`, so `scons` and the `aos/CLAUDE.md`
  convention line follow.
- The FAM telemetry sidecar shrunk to what studies still read from `visits.parquet`
  (section 10), and `mktable` run with `--no-thermal` if the MIW needs only elevation,
  camera rotator angle and band. This removes the EFD from the FAM build, so it runs in
  batch.
- The `combine_fits` loss of the `cam_*` sidecar columns.
- `cwfs_tables`: fold `d12-refitWcs_2025` into `d12-refitWcs` as a date-keyed collection
  list, so one variant covers 2025 and 2026 as the FAM `danish_1_2` already does. Needs
  the builder to open a Butler per collection instead of one per variant.
- Retire `danish_1_0` whole — the FAM variant and the `d10-wep17_3_0` CWFS entry. Both are
  `registered: false` today, kept so `aos/param_sets.yaml` stays byte-identical. Four live
  references to the long param_set key must move in the same commit; the phase 3 handoff
  lists them.

*3c. Rebuild every product fresh into the new tree*, in dependency order: `fam_tables`
and `cwfs_tables` → `miw` and its sidecar → `coadds`, `dof_lut`, `bounce_tables` → the
`value_added` rows that depend on them (`fam_dz` and the MIW-route `optical_state`
variants), either rebuilt from the new FAM fits or checked to give the same results.
**Pre-flight done 2026-10-10** (`refbuild/check_collections.py`, commit `f7985db`): all 15
`fam_tables` collections resolve and hold data; the 20260713 chunk is **still only in
`/repo/embargo`** (24 datasets there, 0 in `/repo/main`), so no `variants.yaml` change.
Sizes: `danish_1_2` 3,523 Butler visits over ten chunks, both blitz variants 966 each.
`mktable` costs 258 s fixed per chunk plus 39.1 s/visit, so `danish_1_2` is 39 h of serial
CPU with an 8.9 h wall-clock floor from its largest chunk; the blitz variants are about an
hour each. The thermal loop costs **2.5 s/visit, not the 19 s/visit in section 10** — see
the phase 3 handoff. Each rebuilt product is compared with its old-tree
build, and every difference is explained (expected: `rotator_angle` on visits where
ConsDB is NULL, the dropped sidecar columns, the restored `cam_*` columns). Then
`set-current`.

Decided 2026-10-10: `danish_1_2` is rebuilt in full (option 1), with `mktable` run as
Slurm jobs **split by night** (`--no-thermal`, after 3b), then `fit`, `combine_*` and
`attach_telemetry` interactively, and the manifest written only after `attach_telemetry`.
Measured costs: 258 s fixed per `mktable` run, 39.1 s/visit donut reading, 3.0 s/visit
`attach_telemetry`; 3,523 Butler visits. Reusing the old chunk donuts was the rejected
alternative. Confirm `/repo/embargo` is readable from batch nodes before submitting.

**Phase 4 — skills and generators.** `/start-study`, `/wrap`, `/review`, the `STUDIES.md`
and `PRODUCTS.md` generators, updated style skills, and an updated `CLAUDE.md`.

**Phase 5 — studies, topic by topic.** Convert each study to `<topic>/<study>/` with
`study.md` (from `docs/studies/*.md`, handoffs in `docs/status/` and `notes/status/item*`
files, and `/wrap` backfill). Fix the wrong-way imports in section 2. Remove the shims.
Retire or document the empty topics (`camera`, `des`, `starcolor`, `survey`, `wcs`,
`alerts`, `scratch`).

**Phase 6 — memory into the repo.** Move the science facts in the laptop Claude memory
(MIW, guider, Z11 intra/extra, CCD height map, frame conventions) into the matching study
or product docs, so S3DF sessions see them.

**Phase 7 — cleanup, Aaron's call.** The old output trees are whole copies of the
pre-reorg state; Aaron zips them for temporary backup and removes them when satisfied.
The laptop's synced output copies (about 60 GB) can go.

**Phase 8 — code review.** Part C of the old plan (C1–C4, at the tag), applied to the new
layout. The reference runs from phases 2–3 are its first regression tests.

## 9. Output mapping, first draft

From the S3DF inventory of 2026-10-08 (`common/scripts/inventory_data_files.py`): 597
distinct file names, 110.2 GB in total. Since the revised route rebuilds rather than copies,
this table now says which old-tree build each rebuilt product is compared against, and
what is never rebuilt.

| today (under `aos/output/` unless noted) | becomes | note |
|---|---|---|
| `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/` donuts, visits, fits | `fam_tables/danish_1_2/<build>/` | build date from file mtime |
| `fam_danish_1_0_wep17_3_0_bin2x/`, `archive/fam_danish_1_0_*` | not copied | obsolete, virtual archive |
| `fam_danish_1_1_1_*`, `fam_danish_1_2_0_wep17_7_0_2025/` | not copied | obsolete, virtual archive |
| `fam_danish_v1_triplets_bin_1x/`, `_bin_2x/`, `output-archive-2026-06-11/` | stay, zipped later | read by no code |
| `.../wfs/<wep version>/` donuts, visits | `cwfs_tables/<variant>/<build>/` | |
| `miw/danish_1_2_A_50_34_i*/` intrinsic_grid, intrinsic_split_*, zk_intrinsic | `miw/d12-A_50_34_i*/<build>/` | sidecar stays with its MIW build |
| `coadd/danish_1_2/50_34*/` block_grids | `coadds/d12-50_34*/<build>/` | |
| `lut/danish_1_2_A_50_34_i/` | `dof_lut/d12-A_50_34_i/<build>/` | |
| `bounce/danish_1_2_A_50_34_i_5rot_july/` tables | `bounce_tables/d12-A_50_34_i_5rot/<build>/` | plots go to the bounce study |
| `value_added/output/aos_efd.duckdb`, `shards/` | `value_added/v1/live/`, `_work/shards/` | `aos_efd_archive_*_pre_vmode_rebuild.duckdb` (0.54 GB) not copied |
| guider per-night parquet | `guider_moments/v6/nights/<night>/` | `old/` night dirs not copied |
| every PDF and PNG | `studies/<topic>/<study>/<YYYYMMDD>_<label>/` | or left in the old tree |

505 file names (23.0 GB) are read by no code. Most are old FAM table formats in
`output-archive-2026-06-11` and `old/` directories.

## 10. Carried-over open items

From the removed `todo.md` and `reorg_review_plan_2026-09.md`:

| item | where it goes |
|---|---|
| outputs that predate a code change and need regenerating | `aos/docs/status/rerun_needed.md`; fold into the phase 2–3 rebuilds |
| `blocks/` rerun (rules `build_table`, `night_table`, `plots`) after the ESS and DOF telemetry moves | phase 5, `blocks` |
| `FocalPlaneInterpolator.py`: delete, or keep | kept in `rubinwork.common`, fixed for numpy 2 (`0aea305`) |
| empty topics: retire or document | phase 5 |
| output provenance helper (old A1) | replaced by `manifest.py` and `run.json` |
| per-study logbook (old B5) | replaced by `study.md` and `/wrap` |
| systematic code review (old Part C1–C4) | phase 8 |
| **The FAM sidecar carries telemetry the build does not need** — see below | phase 3, after `value_added` is registered |

**Shrink the FAM telemetry sidecar; let the downstream studies read ConsDB and DuckDB.**
Aaron's position, recorded 2026-10-10: building the measured intrinsic wavefront (MIW)
needs only **elevation (deg), camera rotator angle (deg) and filter band** — all ConsDB
quantities. So neither `ts_intrinsic_wavefront` nor the MIW build needs the EFD. Everything
else in the sidecar is there to serve *downstream* studies, and those studies could query
ConsDB and the value-added DuckDB themselves instead of having the telemetry baked into
`visits.parquet`.

What that means for the two build halves:

- `run_attach_telemetry.py` attaches eight groups — `thermal`, `gradients`, `wind`,
  `camera`, `lut`, `hexlut`, `trim`, `tweak` — which is why `danish_1_2`'s `visits.parquet`
  is 405 columns against 23 for the blitz variants that never had it run. None of those
  groups is needed to build the MIW. The question for phase 3 is which of them any study
  still reads from `visits.parquet` rather than from DuckDB, and the sidecar keeps only
  those.
- `run_mktable.py` takes `--no-thermal`, and with `no_thermal: false` (today's
  `snake_config.yaml`) its per-visit ESS-temperature loop queries the EFD. **Measured
  2026-10-10 at 2.5 s/visit**, from 466 s against 453 s over the same 5 visits — not the
  19 s/visit estimated earlier, which this supersedes. It is about 6% of a chunk's cost;
  the 39.1 s/visit donut read dominates. If the MIW needs only the three ConsDB columns,
  the loop is still a candidate for removal, but **because it is what keeps the EFD in
  `mktable` and so keeps the chunk out of batch**, not because it is slow.

Why it matters beyond width and speed: the EFD resolves only from interactive nodes, which
is why the Snakefile drops `attach_telemetry` under `--mode batch`
(`attach_telemetry=0`). A sidecar that needs no EFD is what would let the whole FAM build
run in batch. Blocked until `value_added` is a registered product (phase 3), so DuckDB
coverage can be checked per group rather than assumed, and so the studies have a sanctioned
path to read instead.

`notes/todos/todo-ideas.md` stays as Aaron's idea queue. Items there that become studies
get a `study.md`, and the item number is recorded in its front matter.

## 11. Open questions

Resolved 2026-10-09: HSM moments deferred; existing builds copied, not moved; obsolete FAM
variants left in the old tree only; package name `rubinwork`.

1. **Nightly job**: can the EFD be reached from S3DF batch nodes after setup? If not, is a
   scheduled job on `sdfiana`/`slacrd` allowed for the telemetry builder? Needed before
   observing resumes, not before phase 1.
