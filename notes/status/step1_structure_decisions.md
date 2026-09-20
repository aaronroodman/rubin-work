# Step 1 — repository structure decisions

> **Status:** current — all ten queued items done · **Last updated:** 2026-09-20 · **Kind:** working state (plan)

Step 1 of the work in [`reorg_review_plan_2026-09.md`](reorg_review_plan_2026-09.md):
settle the directory structure of `rubin-work`, so that later choices follow from it.
Decided in discussion 2026-09-18; the queued items were executed between 2026-09-18 and
2026-09-20. The items still open below all belong to the main plan's Part A, not to step 1.

## Contents

- [The decisions](#the-decisions)
- [Queued work, in order](#queued-work-in-order)
- [What is blocked, and by what](#what-is-blocked-and-by-what)

---

## The decisions

### D1. Code organization is uniform; output organization depends on the study's data source

Code and output are indexed by different things, so forcing them into matching trees
makes one of them fit badly. Code is indexed by **what question is being asked** — the
study. Output is indexed by the question **and the data it was asked of**.

So:

- **Code** — uniform for every study: `<topic>/code/<study>/`, with genuinely shared
  modules flat at `<topic>/code/`.
- **Output** — the path names **the data the product depends on**, most-general axis
  first, then grouped by study.

In `aos/` the data dependence has **two** axes: `param_set` (a Butler collection paired
with a processing variant) and `mi_name` (which Measured Intrinsic Wavefront build was
used, e.g. `pathA_50_34_i_5rot`). Both are real — `mi_name` appears **54 times** across
`aos/code/` and is a Snakemake wildcard.

**The two axes are joined into one directory name** rather than nested, so there is
always exactly **one** data level:

| what the product depends on | output path |
|---|---|
| the processing run **and** the MIW build | `output/<study>/<param_set>_<mi_name>/` |
| the processing run only | `output/<study>/<param_set>/` |
| neither — the optical prescription, the S-matrix, or the value-added database | `output/<study>/` |

Joining rather than nesting is what makes D2 work: with a single data level the study can
be outermost without any study having to carry a nested subtree. The directory name still
says exactly what the product depends on, which was the point of the nesting.

**This supersedes an earlier A/B/C formulation in this document** that had a single data
axis and no `mi_name`.

### D2. Study outermost, one flattened data level — `output/<study>/<param_set>_<mi_name>/`

The study comes first, and the data axes are joined into a single directory name beneath
it. Aaron's resolution, and it is better than either of the two layouts previously
written here.

The objection to putting the study first was that with two nested data axes, each of the
six studies depending on both (`bounce`, `correlations`, `lut`, `wfs`, `psf`,
`closed_loop`) would have to carry its own `<param_set>/<mi_name>/` subtree — duplicating
the pair six times. **Joining the axes removes the objection entirely**: one level, so
there is no subtree to duplicate.

What this buys:

1. **The ragged axis becomes legible.** A study directory lists exactly the data sets it
   was actually run against, so "not run" and "not applicable" stop looking alike.
2. **A study's products are in one place**, which is how the work is actually read —
   a study is the unit of investigation.
3. **The directory name still states the dependence**, which was the only real merit of
   nesting. `bounce/fam_danish_1_2_0_..._pathA_50_34_i_5rot/` is self-describing.
4. **Kind-C studies need no exception** — they simply have no data level.

**Cost, accepted:** the products of one MIW build are no longer collected under a single
directory, so comparing two builds means reading the same subdirectory name across several
study directories, and deleting a superseded `param_set` becomes N removals rather than
one. Both are scriptable
(`find output -maxdepth 2 -type d -name '<param_set>_*'`), and deletion is a must-ask
one-off rather than daily friction.

**Prerequisite — shorten the names first (D9).** Joined naively, the current names give
`fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x_pathA_50_34_i_5rot`, which is 58 characters and
unreadable. The flattening should land *after* the renaming, not before, so the move
happens once.

### D3. `aos/` keeps its current substructure

`aos/` keeps `code/`, `notebooks/`, `docs/`, `output/`, `logs/`, `calibration/` at the
topic root, with per-study subdirectories inside `code/`, `notebooks/` and `output/`.
It does **not** become `aos/<study>/{code,notebooks,docs,output}`.

Evidence considered:

- **For per-study dirs:** cross-study coupling is almost nil — only 5 study→study imports
  across 67 files.
- **Against, and decisive:** three of the seven flat modules in `aos/code/`
  (`aos_trim.py`, `aos_state.py`, `aos_consdb_efd.py`) are imported **by bare module
  name from four sibling topics** — `blocks/`, `olr/`, `optatmo/`, `guider/` — through a
  hardcoded `sys.path.insert(.../aos/code)`, **39 references** in all. Moving them breaks
  four topics with no static-import warning. `aos/CLAUDE.md` already says "do not finish
  the job by moving the first three".
- Also against: the other four flat modules are used by more than one study
  (`aos_fwhm.py`, `fam_selection.py`, `miw_io.py`, `dz_plotting.py`, `psf_maps_lib.py`).
  Per-study directories give them no home except a pseudo-study, which is what
  `aos/code/` already is — relocating the problem, not solving it.
- Also against: the shared upstream products belong to no single study —
  `visits.parquet` is referenced 91 times, `fits.parquet` 62, `donuts.parquet` 41 (and is
  11.5 GB). Under `aos/<study>/output/` they have nowhere to live.

**What actually caused the confusion** was not the number of subdirectories but three
defects, which D1/D2 and the queued work below fix:

1. `smatrix_vmode` writes to **two** places at once — `output/smatrix_vmode/` and
   `output/<param_set>/smatrix_vmode/` — with duplicate filenames and no marker of which
   is live (D8). The three-level scheme itself is coherent; this study violates it.
2. Sparse subdirectories, so the tree cannot distinguish "this study has no notebooks"
   from "I have not found them yet" — 8 of 16 studies have no notebook dir, 7 have no
   output dir, and `notebooks/fam_focus/` is empty.
3. A study's material is 3–4 directories apart, with no single place that lists where its
   pieces are.

### D4. Each study gets one entry point, and that is where symmetry lives

The thing that should be symmetric across studies is **the description**, not the
directory shape. Each `docs/studies/<study>.md` gains a header block naming its kind and
its paths:

```markdown
> **Kind:** A (FAM-derived) · **Code:** `code/bounce/` · **Notebooks:** `notebooks/bounce/`
> **Output:** `output/bounce/<param_set>/` · **Key products:** `bounce_kj_stats.parquet`, `bounce_heatmaps.pdf`
```

One file to open per study, four paths listed. This resolves defect 3 above without
moving any code.

### D5. Reusability test — what belongs where

Three tiers, each with a test that can be applied to a single file:

| tier | location | test |
|---|---|---|
| **study** | `<topic>/code/<study>/` | answers one question about one data set |
| **topic-common** | `<topic>/code/` (flat) | encodes something true of the **instrument or convention**, not of one analysis |
| **repo-common** | `common/` | true across **topics** — used by two or more, and would be used by a third |
| **service** | its own topic | maintains **state or a product** that other topics consume |

The topic-common test is why `aos_state.py` (the DOF basis), `miw_io.py` (the MIW
loader) and `fam_selection.py` (the quality cut) are correctly flat in `aos/code/`: each
encodes a property of the AOS, not of an analysis.

The **service** tier is new, and is what `common/`-vs-topic previously had no slot for.
It is the basis for D6.

### D6. The value-added database becomes a topic: `value_added/`

It is a **service**, not shared code: it maintains a 2.6 GB database with seven tables,
consumed by `aos/` (4 files) and `common/scripts/` (4 files), with its documentation
currently scattered across `common/README.md`, `aos/README.md` and three
`aos/docs/studies/*.md`. Five of the six Python files already say "value-added" in their
own module docstring, so the name is the code's own.

```
value_added/
  README.md                       what it is, the seven tables, build cadence, consumers
  code/
    efd_db.py                     (1379 L) schema, upsert and read helpers — the library
    build_efd_db.py               ( 535 L) build visit_telemetry, night by night
    build_optical_state.py        ( 583 L) per-visit optical state -> optical_state
    build_fam_dz.py               ( 342 L) FAM DZ fits + v-modes -> fam_dz
    backfill_commanded_vmodes.py  ( 164 L) commanded hexapod LUT / Trim v-modes
    merge_db_shards.py            ( 213 L) merge parallel shards into the main DB
    run_build.sh                  ( 236 L) parallel shard driver, local or Slurm batch
  docs/
    schema.md                     seven tables, every column with units and provenance
    status/build_progress.md      nights built, columns known sparse
  output/
    aos_efd.duckdb                2.6 GB
    aos_efd_archive_20260916_pre_vmode_rebuild.duckdb
    shards/                       33 shard files
```

Tables: `visit_telemetry`, `state_variant`, `optical_state`, `fam_variant`, `fam_dz`,
`column_coverage`, `fetch_log`.

**Stays in `common/`:** `telemetry_clients.py` — raw EFD/ConsDB *client construction*,
used by `aos/` and `olr/` independently of the DB. Genuinely repo-common by the D5 test.

**Goes to `aos/code/` instead:** `common/miw_corner_intrinsic.py` — MIW physics (corner
field intrinsics vs rotator angle), not database plumbing. The DB is merely its only
current caller.

**Fixes three things beyond tidiness:**

1. The DB stops living in **repo-root `output/`**, which `.gitignore` guards as a path
   products should not use.
2. `efd_db.py` (1379 lines, imported only by `aos/` and the DB's own scripts) leaves
   `common/`, resolving the audit finding that `common/` holds code that is not common.
3. The documentation gets one home, and `docs/schema.md` with units per column is
   precisely the record-keeping gap identified in part (b) of the main plan.

### D7. `smatrix_vmode` moves from `aos/` to the `smatrix/` topic

It is already a satellite of `smatrix/` living in the wrong topic:
`aos/code/smatrix_vmode/plot_vmode_dof_matrix.py:45` does
`sys.path.insert(… 'smatrix' / 'code')`, and **all four** of its notebooks reach into
`smatrix/code`. Moving it **removes** a cross-topic dependency rather than adding one.

Destination — as a study inside `smatrix/`, which needs the study substructure anyway
(its `code/` has 21 flat files):

```
smatrix/code/vmode/        <- from aos/code/smatrix_vmode/       (3 .py)
smatrix/notebooks/vmode/   <- from aos/notebooks/smatrix_vmode/  (6 notebooks)
smatrix/output/vmode/      <- kind C, no param_set level
smatrix/docs/studies/vmode.md
```

`aos/code/static_optics/camera_gravity.py:51-52` also imports `compute_smatrix` from
`smatrix/code`. It stays in `aos/` — it is genuinely about camera gravity — and the
dependency is documented rather than removed.

### D8. Resolve the `smatrix_vmode` output collision

`vmode_dof_matrix_50_34.pdf` and `vmode_dof_matrix_22_12.pdf` currently exist in **two
places at once**, with nothing marking which is live:

| file | param_set-scoped copy (size, B; date) | study-root copy (size, B; date) |
|---|---|---|
| `vmode_dof_matrix_50_34.pdf` | 126,786; 2026-08-12 | 190,400; 2026-09-08 |
| `vmode_dof_matrix_22_12.pdf` | 69,870; 2026-08-12 | 92,361; 2026-09-08 |

Paths: `aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/smatrix_vmode/` and
`aos/output/smatrix_vmode/`.

**Different files.** The study's own doc settles which is live:
`smatrix/docs/studies/vmode.md` states that both scripts write to `output/vmode/` at the top
level, "not under a `param_set`", because the v-mode/DOF structure is a property of the OFC
sensitivity matrix and the DOF scheme alone. So the param_set-scoped copies are stale
leftovers from before that decision — consistent with being four weeks older and smaller.

**Resolved 2026-09-19.** Aaron deleted the 5 superseded files after they were verified
against the live copies with `cmp`. Exactly one copy of each product now exists, in
`smatrix/output/vmode/`:

| path | files | outcome |
|---|---|---|
| `smatrix/output/vmode/` | `sparse_fit_study.pdf` 98,665 B; `vmode_dof_matrix_22_12.pdf` 92,361 B; `vmode_dof_matrix_50_34.pdf` 190,400 B | **live** |
| `aos/output/smatrix_vmode/` | the same 3 files, 381,426 B total | deleted |
| `aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/smatrix_vmode/` | the 2 stale 2026-08-12 PDFs, 196,656 B total | deleted |

Of the 3 superseded copies, two were byte-identical to the live copy and
`vmode_dof_matrix_22_12.pdf` differed in 6 bytes of 92,361, all inside the PDF
`/CreationDate` string (2026-09-08 → 2026-09-19, from a regeneration while testing the moved
script) — same producer, same byte count, same plot content.

**The two orphans in `aos/output/` — resolved 2026-09-19, copied to `smatrix/output/vmode/`:**
`sensitivity_sparse_analysis.pdf` (65,949 B, 6 pages) and `sparse_observability.pdf`
(49,626 B, 4 pages), both 2026-09-07, written by `analyze_sensitivity_sparse.py` and
`analyze_sparse_observability.py`. Commit `f9f9ff1` combined those two scripts into
`analyze_sparse_fit.py`, so neither PDF is regenerable — but the combined 10-page
`sparse_fit_study.pdf` is 6 + 4 pages and reproduces their numbers exactly (retained-v-mode
observability, dimensionless, primary-only over full matrix: 0.44/0.96 and 0.36/0.97 for
50-DOF / 34-v-mode, 0.99/1.00 and 0.99/1.00 for 22-DOF / 12-v-mode, per `f9f9ff1`'s own
message). So they are superseded predecessors, not unique results. Copied with matching
md5 (`234d08b5…`, `aaa83266…`) and indexed in `smatrix/docs/plots.md` as not regenerable;
the `aos/output/` copies are Aaron's to delete, which is the last thing keeping this from
satisfying "one product, one path".

### D9. Shorten `param_set` and `mi_name`, and retire the obsolete param_sets

A prerequisite for D2, since the joined name is only readable if the parts are short.
Naively joined today:

```
fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x_pathA_50_34_i_5rot     (58 characters)
```

**Scope measured, not estimated.** The current param_set name appears **96 times across 71
files** — 27 `.yaml`, 23 `.py`, 14 `.md`, 7 `.ipynb`. The `.py` hits are hardcoded CLI
defaults and module constants, not config reads, so a rename is a code sweep and not a
one-line config edit. Seven of them are in **`optatmo/`**, reaching into `aos/output/` by
absolute-ish relative path — a sibling topic breaks if the rename misses them.

**The name is also inside the value-added database**, which is the part most likely to be
forgotten:

| table.column | stored value |
|---|---|
| `fam_variant.fam_variant_id` | `fam__fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x__batoid__z1toz6__50_34` |
| `fam_variant.param_set` | `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x` |
| `fam_variant.fits_path` | absolute path containing the param_set name |
| `state_variant.intrinsic_ref` | `pathA_50_34_i_5rot` — so `mi_name` is in there too |

A rename therefore needs an `UPDATE` pass over those rows, or a rebuild, or the database
stops joining to the files. `aos/code/fam_focus/run_fam_focus.py:101` hardcodes that
`fam_variant_id` as `DEFAULT_FAM_VARIANT`.

**What makes this cheap right now:** `mi_config.yaml` defines MI entries for only **one**
param_set, with only **two** entries (`pathA_50_34_i` and `pathA_50_34_i_5rot`, the second
reusing the first's grids via `build_from`). Five param_sets are defined in
`param_sets.yaml` but four are superseded. So the flattening touches far fewer directories
than the tree suggests — and the cost grows with every param_set and MI entry added.

**Order of operations.** Retire the obsolete param_sets *first* (they need no rename at
all), then rename, then flatten. Doing it in that order means the rename and the move each
touch the smallest possible set of paths.

Naming is Aaron's call. The constraint from D2 is only that
`<param_set>_<mi_name>` stay legible at a glance — roughly 30 characters or so for the
pair — and that the separator not be ambiguous, since both parts already contain `_`.

#### Decided 2026-09-19: retire three, rename nothing, shorten going forward

Aaron reviewed the five param_sets and settled it as follows.

**Retired** — `fam_danish_v1_triplets_bin_1x`, `fam_danish_v1_triplets_bin_2x`,
`fam_danish_1_1_1_wep17_3_0_bin2x` (`9958662`). All three had been superseded since
2026-09-06: disabled in `snake_config.yaml`, output in `output/archive/`, no live output
directory, and between 3 and 5 tracked references each.

**Kept, not renamed** — `fam_danish_1_0_wep17_3_0_bin2x`. Superseded as a working
param_set, but its name is the recorded provenance of the local MIW staged at
`aos/calibration/miw/intrinsic_split_maps_v1.parquet` and of the two-epochs tech note. It
still carries a live `analysis_config.yaml` override block. Renaming or dropping it would
dangle that provenance. **The official MIW is Guillem's**, built by `ts_intrinsic_wavefront`
and read from the Butler — the staged `v1` product is Aaron's own build, which is why that
provenance file names a Danish 1.0 param_set rather than 1.2.

**Kept, not renamed** — `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`. A rename to
`danish_1_2` was scoped and declined: **71 occurrences across 50 tracked files**, reaching
into `optatmo`, `value_added`, `blocks` and `smatrix`, plus an `UPDATE` pass over the
value-added registries. (The "96 times across 71 files" figure above counted gitignored
`snippets.ipynb` and `.ipynb_checkpoints`; the tracked-only count is the one that matters.)
It buys readability, not correctness, so the cost was not worth it on working code.

**`mi_name` unchanged** — `pathA_50_34_i` / `pathA_50_34_i_5rot`. Dropping the obsolete
`path` prefix was considered and declined on the same grounds: 4 characters against 87
occurrences and a DB `UPDATE`. Note 61 of those 87 are the substring inside
`pathA_50_34_i_5rot`, so any future rename must substitute longest-first, and
`intrinsic_split_maps_v1.parquet` contains the string as frozen binary provenance.

**The convention going forward** is recorded in `param_sets.yaml`: a new param_set gets a
short name — `danish_1_3`, not `fam_danish_1_3_0_wep17_8_0_refitWCS_bin2x` — with the wep /
donut_viz / bin detail in `description` and `fam_collections`, where it is actually read
from. This gets D2's legibility on everything new without a sweep over working code.

**Consequence for D2/item 7.** The joined name for the live pair would be
`fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x_pathA_50_34_i_5rot` (59 characters), well over
the ~30 target. D2's premise — that flattening needs short parts first — does not hold for
this param_set and will not be made to hold.

**Superseded by D10 on 2026-09-20.** The dilemma D9 posed — accept a long directory name or
wait for a short-named param_set — had a third answer: keep the long key as the identity and
give each entry a short `dir_name` used in paths only. Item 7 landed on that basis.

### D10. A `dir_name` translation layer separates the identity from the directory name

Each entry in `param_sets.yaml` and in the `measured_intrinsics` list of `mi_config.yaml`
carries a `dir_name` giving a short form used in `output/` paths **only**. The long key
remains the identity that `--param-set`, the value-added database rows, the LUT parquet
metadata and the frozen provenance resolve against. `dir_name` defaults to the key when
absent, so a future short-named param_set needs no entry.

```
fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x  ->  danish_1_2
pathA_50_34_i_5rot                         ->  A_50_34_i_5rot
output/correlations/danish_1_2_A_50_34_i_5rot/       (25 characters)
```

The repo already did this once: `collection_phrase` in `param_sets.yaml` maps a long Butler
collection name onto a short filename-safe string. `dir_name` follows that precedent.

The translation lives in the `Snakefile`, which owns the `output:` declarations. Each rule
resolves the short wildcard back to the long key through an injective map built at load
time, and passes both — `--param-set <long key>` for the config lookup and `--out-dir` for
the path. The scripts no longer re-derive their own output path, which removes the
duplicated path logic rather than adding a resolver to 27 call sites. Every semantic use of
`param_set` and `mi_name` inside the scripts was checked first and is either a config lookup
or a plot label; nothing derives physics from the string.

Two departures from the root `CLAUDE.md` rule, both now written there:

- **The corner wavefront sensor (CWFS) variant nests** one level below the data directory
  (`wfs_ingest/<P>/<cwfs>/`) instead of joining as a third axis, which would reach 36
  characters.
- **`fits.parquet` appears under two studies** — the phase-1 per-visit DZ fit under
  `fam_processing/` and the DZ refit referenced to the MIW under `miw/`. Different products
  under different studies is not a "one product, one path" violation.

---

## Queued work, in order

Each item is one commit. Verify imports after each before continuing.

Each item is one commit. Verify imports after each before continuing.

| # | work | touches | state |
|---|---|---|---|
| 1 | Write the structure rule into the root `CLAUDE.md` (D1, D2, D5) | `CLAUDE.md` | **done** — `3bc0134` |
| 2 | Add the D4 path header to the 16 `aos/docs/studies/*.md` | 16 docs | **done** — `3bc0134` |
| 8 | Prune the empty dirs — `aos/notebooks/fam_focus/`, `aos/code/output/` | dirs only | **done** — approved 2026-09-18 |
| 3 | Create `value_added/`, move the 7 files, fix the importing files, move the DB out of repo-root `output/` | `common/`, 11 files, `.gitignore` | **done** — `8f7ab33` |
| 4 | Move `miw_corner_intrinsic.py` → `aos/code/` | 2 files | **done** — `8f7ab33` |
| 5 | Move `smatrix_vmode` → `smatrix/code/vmode/` + `notebooks/vmode/` (D7) | 9 files + 15 cross-references | **done** — `8e5f76c` |
| 10 | Move `guider/output` and `optatmo/output` to group space and replace with symlinks | 6.4 GB / 28,596 files, `.gitignore` | **done** — `92aef81`; symlinked 2026-09-19, `/sdf/home` 87% → 66% |
| 6 | Resolve the output collision (D8) — delete the 2 stale param_set-scoped PDFs and the 3 superseded copies in `aos/output/smatrix_vmode/` | 5 files, 578,082 B | **done** — deleted 2026-09-19 |
| 9 | Retire the obsolete param_sets; renaming declined (D9) | 3 param_sets, 7 files | **done** — `9958662`; no rename, short names from `danish_1_3` on |
| 7 | Flatten `<param_set>/<mi_name>/<study>/` → `<study>/<param_set>_<mi_name>/` (D2, D10) | `Snakefile`, 2 config files, 16 scripts, 25 docs, 2 notebooks, 308 files moved | **done** — `d839399` + `6579204` + this commit; short `dir_name` per D10 |

Items 3–6 are unblocked as of 2026-09-18: the other session's
`aos/notebooks/smatrix_vmode/vmode_dof_ts_ofc-13Aug2026.ipynb` was deleted rather than
kept, which was the one file standing in the way of item 5.

Three corrections from doing items 3 and 4, worth carrying into the later items:

- The "13 importing files" count was wrong. Two of the thirteen
  (`run_thermal_model.py`, `run_science_lut_analysis.py`) mention `efd_db` only in
  comments, so there were **11** live imports; three further hits were in gitignored
  `.ipynb_checkpoints`. Count live imports, not grep hits, when sizing items 7 and 9.
- **Docs hold as many stale paths as code does.** A grep limited to `.py`/`.ipynb` came back
  clean while 13 stale references remained in tracked Markdown, including a 180-line section
  of `common/README.md` documenting the moved database. Item 7 rewrites output paths, so its
  verification grep must cover `*.md` too.
- `value_added/output/` is a **symlink** to group space, like `aos/output` and
  `blocks/output`, and needed its own `.gitignore` entry: `*/output/*` ignores the contents
  but not the symlink path itself, so without the entry it shows up as an untracked file on
  S3DF. Any future topic whose `output/` is a symlink needs the same line.

Four more from doing item 5, all bearing on items 7 and 9:

- **A move changes what a relative path means, in both directions.** Nine `git mv`s needed
  **15** cross-reference rewrites outside the moved files: five `aos/docs/studies/*.md`
  "See also" links, three `aos/docs/status/` docs, four Python docstrings, `aos/CLAUDE.md`,
  and the study tables in `aos/README.md` and `aos/docs/studies.md`. Two links *inside* the
  moved doc also broke — they resolved only from the old location. Check links in the moved
  file, not just links to it.
- **Verify relative links mechanically.** Resolving every non-anchor Markdown link against
  the filesystem found the two broken in-file links that reading had missed. Worth doing over
  the touched docs after item 7's path rewrite.
- **Counts in prose go stale with the tree.** `aos/docs/studies.md` opened with "sixteen
  studies" and "85 Python files"; both were wrong the moment the study left (now fifteen and
  78). Item 7 touches 7 study docs and item 9 touches 71 files — recount rather than assume.
- **Re-dumping a notebook through `json` rewrites the whole file.** `ensure_ascii=False`
  converted every stored `\uXXXX` escape and a per-line edit collapsed a multi-line source
  string, turning a 3-line fix into a 156-line diff. Editing the raw file as text, asserting
  each target string occurs exactly once, gave a 3-line diff. Do that for item 7's notebook
  path rewrites.

Item 7 is last on purpose. It is the only item that rewrites output paths in code, and
doing it before the renaming in item 9 would move every directory twice.

Four things from doing item 7 on 2026-09-20:

- **The scope estimate was low in code and high in notebooks.** The plan named ~20 docs and
  6 notebooks; the actual grep found 25 tracked `.md` files, only 3 notebooks with source
  hits (two of which the plan had not listed), and **13 hardcoded path defaults the plan
  never mentioned** — 8 of them in `optatmo/`, which reads `aos/output/` cross-topic.
- **A move into a sibling topic is repair, not refactor.** Fixing those 8 `optatmo/` paths
  looks like the side-effect refactoring the root `CLAUDE.md` forbids, but the migration had
  just broken them. Committed separately (`6579204`) so it is revertable on its own.
- **`find` will not traverse a symlinked start path** without `-L`, so `find output -type f`
  silently returns nothing for `aos/output`. Two counts were lost to this.
- **Compare rerun *reasons*, not job counts.** The post-migration dry run wanted 78 jobs,
  which looked like migration damage. Pre-migration it wanted 89, every rule reporting
  "Missing output files"; post-migration that phrase is gone from every rule whose product
  exists, and `mktable` × 10 vanished entirely. `os.rename` preserves mtimes, so the
  remaining `fit` × 10 is the same pre-existing staleness as before the move.

### Item 10 — `guider/output` and `optatmo/output` to group space

Aaron asked for these on 2026-09-18. `/sdf/home` has 4.0 GB free of 30 GB (87% used).

**Done 2026-09-19.** Both trees were copied to group space byte-for-byte, verified, then
the originals deleted and replaced with symlinks (Aaron ran the `rm`/`ln`, since
`Bash(rm:*)` is in the `.claude/settings.local.json` deny list, which conversational
approval does not override):

| topic | files | bytes | verified |
|---|---|---|---|
| `guider/output` | 23,354 | 4,485,731,571 | totals match; `rsync --dry-run` 0 files to transfer; 10 random files `cmp`-identical |
| `optatmo/output` | 5,242 | 2,158,645,519 | same, 10 random files `cmp`-identical |

After the switch both symlinks resolve and all 23,354 and 5,242 files are readable through
them, git reports the tree clean (the `.gitignore` entries work), and `/sdf/home` went from
27 GB used of 30 GB (87%) to 20 GB used (66%), recovering the expected 6.4 GB.

One caveat for any future check of this kind: once `<topic>/output` is a symlink into group
space, `df -h <topic>/output` reports the **group** filesystem (273 T), not `/sdf/home`. Run
`df -h /sdf/home/r/roodman` on a real path in home to see the quota that matters.

The git-side work is already committed in `92aef81`: `/guider/output` and `/optatmo/output`
added to the `.gitignore` symlink block, `guider/output/.gitkeep` removed from the index,
and the comment listing which `output/` dirs are symlinks brought up to date. `optatmo` was
already covered by its own topic-level `.gitignore`, so its root entry is redundant but
kept for an explicit list. Only `guider` carried a tracked `.gitkeep`.

The
pattern to follow is the one `aos/output`, `blocks/output` and now `value_added/output`
use: move the contents to
`/sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/<topic>/output/`, replace the
directory with a symlink, and add `/guider/output` and `/optatmo/output` to the
`.gitignore` symlink block — `*/output/*` ignores the contents but not the symlink path
itself.

Two things to check that did not arise for `value_added/`, whose files were a handful of
large database files:

- **A tracked `.gitkeep`.** `.gitignore` warns that a tracked `.gitkeep` inside a
  symlinked `output/` makes `git pull` abort on S3DF with "untracked working tree files
  would be overwritten by merge". Both topics are real directories today, so each may
  carry one that must be removed from the index as part of the move.
- **Whether the move is a rename or a copy.** Group space is a different filesystem from
  `/sdf/home`, so unlike the `value_added/` database move — which was a same-filesystem
  rename of 2.6 GB in 5.2 s — this is a genuine 6.4 GB copy across 28,596 files and will
  take real time. Verify the file count and total size on both sides before removing
  anything, and deleting the originals is a MUST-ASK.

The `**Output:**` headers added in item 2 originally described the nested
`<param_set>/<mi_name>/` form, so that a reader was never sent to a path that did not yet
exist. Item 7 rewrote all of them to the flattened layout as part of the move.

The 13 files needing the `efd_db` import edit in item 3:

```
aos/code/fam_focus/run_fam_focus.py
aos/code/science_lut/run_science_lut.py
aos/code/science_lut/run_science_lut_analysis.py
aos/code/science_lut/run_science_lut_report.py
aos/code/science_lut/run_thermal_model.py
aos/notebooks/bounce/bending_mode_test_lut_trim.ipynb
aos/notebooks/bounce/_exectest.ipynb
aos/notebooks/science_lut/science_lut_explore.ipynb
common/scripts/backfill_commanded_vmodes.py
common/scripts/build_efd_db.py
common/scripts/build_fam_dz.py
common/scripts/build_optical_state.py
common/scripts/merge_db_shards.py
```

After the move these import `value_added/code` via `sys.path.insert`, replacing an
undocumented `common/` import with a documented cross-topic dependency. All 13 change in
**one commit** so the tree is never broken in between.

Three further hits are inside `.ipynb_checkpoints/` (`run_science_lut-checkpoint.py`,
`run_science_lut_analysis-checkpoint.py`, `science_lut_explore-checkpoint.ipynb`). They
are gitignored local clutter, not tracked files — leave them alone.

Regenerate the list rather than trusting it, since the other session is still adding code:

```bash
cd ~/notebooks/rubin-work && grep -rl "efd_db" --include='*.py' --include='*.ipynb' . \
  | grep -v ipynb_checkpoints | grep -v '^./common/efd_db.py' | sort
```

## Where this stands

**The other-session blocker is cleared.** As of 2026-09-18 the only remaining uncommitted
work is two `aos/notebooks/correlations/` notebooks, which no queued item touches.

Tag **`pre-value-added-reorg-2026-09-18`** at `3bc0134` marks the tree before any file or
output moves — the point to return to if a move goes wrong.

**Check before starting any move:**

```bash
cd ~/notebooks/rubin-work && git status --porcelain
```

Proceed when nothing is modified under `common/`, `aos/code/`, or the directory being
moved.

All ten queued items are done. Both decisions this section previously held open are
settled: item 6's stale duplicate PDFs were deleted on 2026-09-19, and item 9 declined the
rename in favour of short names from `danish_1_3` on, which D10 then made sufficient for
item 7.

The migration left the emptied `output/<param_set>/` directories in place, per the
delete-asking rule. Removing them is a deletion and needs approval.

Still open from the main plan's Part A, unchanged by this document: deleting
`aos/output/archive/`, the `FocalPlaneInterpolator.py` delete-or-demonstrate call, study
splits for `smatrix`/`guider`/`optatmo`/`filters`, the seven empty topics, and the five
stray astrometry files in the repo-root `output/`. The archive deletion is a deletion, so
it needs explicit approval.
