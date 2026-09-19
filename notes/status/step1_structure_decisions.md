# Step 1 — repository structure decisions

> **Status:** agreed, queued — execution blocked on the other session finishing · **Last updated:** 2026-09-18 · **Kind:** working state (plan)

Step 1 of the work in [`reorg_review_plan_2026-09.md`](reorg_review_plan_2026-09.md):
settle the directory structure of `rubin-work`, so that later choices follow from it.
Decided in discussion 2026-09-18. **Nothing here has been executed yet.**

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

In `aos/` that data dependence has **two** axes, not one, and this is already the rule
`aos/README.md` states:

| what the product depends on | output path |
|---|---|
| the processing run **and** which MIW build was used | `output/<param_set>/<mi_name>/<study>/` |
| the processing run only | `output/<param_set>/<study>/` |
| neither — the optical prescription, the S-matrix, or the value-added database | `output/<study>/` |

`param_set` is a Butler collection paired with a processing variant; `mi_name` is a MIW
build (`pathA_50_34_i`, `pathA_50_34_i_5rot`, `coadd_50_34_v2`). Both are real axes —
`mi_name` appears **54 times** across `aos/code/`, and `pathA_50_34_i_5rot/` alone holds
six studies' output (`bounce`, `correlations`, `lut`, `wfs`, `psf`, `closedloop`).

**This supersedes an earlier A/B/C formulation in this document** that had a single
data axis and put the study level first. It was wrong on both counts: it missed
`mi_name`, and see D2.

### D2. Data axes stay outermost — `output/<param_set>/<mi_name>/<study>/` is kept

**Reversed from an earlier draft of this document, which proposed inverting to
`output/<study>/<param_set>/`. Do not invert.** That recommendation was made while
believing there was one data axis; there are two.

With two axes, study-first means every study that depends on both has to carry a
`<param_set>/<mi_name>/` subtree of its own. Six studies do
(`bounce`, `correlations`, `lut`, `wfs`, `psf`, `closed_loop`), so the pair of axes gets
duplicated six times over, and a MIW rebuild scatters its products across six trees
instead of landing in one directory.

Keeping the data axes outermost has three concrete properties the inversion loses:

1. **A MIW build is one directory.** `pathA_50_34_i_5rot/` is the complete set of
   products derived from that build — which is what makes it comparable against
   `pathA_50_34_i/`, and comparing MIW builds is the point of much of the work.
2. **Deleting a superseded `param_set` is one `rm -rf`**, which is how
   `output/archive/` already works.
3. **It is what the code does now.** `mi_name` is threaded through 54 call sites; the
   inversion would rewrite every one for no gain.

The ragged-axis objection that motivated the inversion is real but is a **documentation**
problem, not a layout problem — you cannot tell "not run" from "not applicable" by
looking at the tree. D4 fixes it by naming each study's actual output path in its study
doc, which costs nothing and does not move a single file.

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

**Different files.** Under D1 the study is kind C — v-mode/DOF matrices derive from the
S-matrix, not from a night's data — so the param_set-scoped copy is the stale leftover,
consistent with the other being newer and larger. It moves to `smatrix/output/vmode/`
and the stale copy is deleted (**must-ask**).

---

## Queued work, in order

Each item is one commit. Verify imports after each before continuing.

| # | work | touches | blocked? |
|---|---|---|---|
| 1 | Write the structure rule into the root `CLAUDE.md` (D1–D5) — the convention text, no file moves | `CLAUDE.md` | no |
| 2 | Add the D4 header block to the 16 `aos/docs/studies/*.md` | 16 docs | no |
| 3 | Create `value_added/`, move the 7 files, fix the **13** importing files, move the DB out of repo-root `output/` | `common/`, 13 files, `.gitignore` | **yes** |
| 4 | Move `miw_corner_intrinsic.py` → `aos/code/` | 2 files | yes (same commit as 3) |
| 5 | Move `smatrix_vmode` → `smatrix/code/vmode/` + `notebooks/vmode/` (D7) | ~9 files | **yes** |
| 6 | Resolve the output collision (D8) — needs deletion approval | 2 output paths | yes |
| 7 | ~~Invert the output tree~~ — **withdrawn**, see D2. No work to do. | — | — |
| 8 | Prune the sparse/empty dirs — `notebooks/fam_focus/` and any created-but-unused | dirs only | no |

Items 1, 2 and 8 are safe to start now — they touch no file the other session has open.

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

## What is blocked, and by what

**Items 3–7 wait for the other session to finish.** As of 2026-09-18 that session has
largely landed — from 16 modified files down to:

```
 M aos/notebooks/correlations/corner_z4_vs_temperature_science.ipynb
?? aos/notebooks/correlations/querying_efd_consdb.ipynb
?? aos/notebooks/smatrix_vmode/vmode_dof_ts_ofc-13Aug2026.ipynb
```

The third is inside `aos/notebooks/smatrix_vmode/`, which **item 5 moves wholesale** — so
item 5 in particular must not start while that notebook is uncommitted, or the move will
either miss it or conflict.

None of the 7 files moving to `value_added/`, and none of the 13 needing an import edit,
is currently modified. So **item 3 is unblocked the moment that notebook lands**; it does
not need the whole session's backlog cleared.

**Check before starting items 3–7:**

```bash
cd ~/notebooks/rubin-work && git status --porcelain
```

Proceed when it shows nothing under `aos/notebooks/smatrix_vmode/`, `common/` or
`aos/code/science_lut/`.

**Decision still outstanding:** item 6 deletes the stale duplicate PDFs, and item 7 of
the main plan's Part A proposes deleting the 27 GB `aos/output/archive/`. Both are
deletions, so both need explicit approval.
