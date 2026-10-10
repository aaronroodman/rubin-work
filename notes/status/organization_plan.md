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
| old output | left untouched during the reorg; in-use builds are **copied** into the new tree, so the old tree stays whole; Aaron zips it for temporary backup afterwards |
| obsolete builds | not copied and not registered ("virtual archive"): they stay only in the old tree. Includes FAM `danish_1_0`, `danish_1_1_1`, `danish_1_2_0_wep17_7_0_2025` |
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
at most 24 characters, never reused for a different configuration. Changing the
configuration (collections, programs, rotator bins, correction scheme) makes a new variant.

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

**Phase 2 — pilot product: `fam_tables`, end to end.**
1. Reference run before the move: build one variant over a few nights and keep the output.
2. Move the code and the Snakefile rules into `rubinwork/products/fam_tables/`;
   `variants.yaml` from `param_sets.yaml`.
3. Rerun the reference build and compare row counts and column values with the reference.
4. Register the existing builds: copy each in-use build into the new tree and write its
   manifest. The old tree is not touched, so old code keeps working until phase 5. Obsolete
   builds are not copied. Copy cost: the in-use `danish_1_2` FAM tables are the bulk;
   `donuts.parquet` over all 24 copies on disk is 64.09 GB, and only the in-use ones are
   copied.
5. Update readers to use `load()`.

**Phase 3 — the other products**, same steps, in dependency order: `cwfs_tables`; `miw`
(plus the official-MIW registrations); `coadds`; `dof_lut` and `bounce_tables`;
`value_added` (switch to `v1/live` + snapshots, registry rows to catalog names);
`guider_moments` and `hsm_moments`. Split the shared `aos/Snakefile` as each product's
rules leave it; what remains are study rules, which move with their studies in phase 5.

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
distinct file names, 110.2 GB in total. To be checked by Aaron before phase 2.

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

`notes/todos/todo-ideas.md` stays as Aaron's idea queue. Items there that become studies
get a `study.md`, and the item number is recorded in its front matter.

## 11. Open questions

Resolved 2026-10-09: HSM moments deferred; existing builds copied, not moved; obsolete FAM
variants left in the old tree only; package name `rubinwork`.

1. **Nightly job**: can the EFD be reached from S3DF batch nodes after setup? If not, is a
   scheduled job on `sdfiana`/`slacrd` allowed for the telemetry builder? Needed before
   observing resumes, not before phase 1.
