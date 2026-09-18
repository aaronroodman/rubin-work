# Repository organization, documentation and code-review plan

> **Status:** proposed — awaiting Aaron's decisions on the marked choices · **Last updated:** 2026-09-18 · **Kind:** working state (plan)

Assessment of the current state of `rubin-work` (code, notebooks, output, documentation)
and a plan for (a) organization, (b) documentation and record keeping, (c) systematic
code review. Written to be executed in small, independent units that do not collide with
concurrent work in another session.

Scale of the repository as measured 2026-09-18: **57,179 lines** of Python across 20
topics, **60 notebooks**, **~65 GB** of output on disk, **112 tracked Markdown files**.

## Contents

- [Assessment](#assessment)
- [Part A — organization of code and output](#part-a--organization-of-code-and-output)
- [Part B — documentation and record keeping](#part-b--documentation-and-record-keeping)
- [Part C — systematic code review](#part-c--systematic-code-review)
- [Sequencing and conflict avoidance](#sequencing-and-conflict-avoidance)

---

## Assessment

### What the September reorganization actually achieved

The 2026-09-04 → 2026-09-08 effort was **scoped to `aos/` and was completed there**.
`notes/status/memory_cleanup_plan.md` records "aos study reorg complete (8 of 8)", and
the evidence agrees: `aos/code/` has 16 per-study subdirectories each with a matching
`aos/docs/studies/<study>.md`, `aos/notebooks/` has 9 study subdirectories with **zero**
loose notebooks, and no Python module name is duplicated anywhere in the repository. The
convention in `CLAUDE.md` is sound and `aos/` demonstrates it works.

The reason the repository still feels disorganized is not that the convention failed. It
is that **the convention was applied to one topic out of twenty, and the remaining
nineteen were never converted**. What follows is therefore mostly "finish the job",
not "rethink the design".

### The four structural gaps

**1. `notebooks/` exists in exactly one topic.** Of 60 notebooks, **35 sit loose in a
topic root** — `blocks/` 9, `guider/` 7, `wfs/` 6, `psf/` 4, `nightlyiq/` 4, `olr/` 3,
`astrometry/` 1, plus 4 untracked `Untitled.ipynb` and 3 untracked `*snippets.ipynb`.
The rule "nothing but `README.md` and `CLAUDE.md` belongs loose in a topic root" is
honoured only by `aos/` and `smatrix/`.

**2. Four topics have the flat `code/` root the convention warns against.** `optatmo/`
30 files, `guider/` 29, `smatrix/` 21, `filters/` 14 — all with visible study seams in
the filenames (`guider_atmo_*.py` is 11 files; `smatrix/` has distinct normalization,
thermal, full-mode and OFC-comparison clusters; `filters/` has a `design_*.py` family).
`CLAUDE.md` says "that is how a flat 56-file directory happens" — these are it.

**3. `common/` holds code that is not common.** Verified by grep across the repository:
- `FocalPlaneInterpolator.py` — **916 lines, zero importers anywhere**, yet root
  `CLAUDE.md` advertises it as one of three headline shared modules.
- `efd_db.py` — 1,379 lines, imported only by `aos/` (and by `common/scripts/`).
- `psf_render.py` — imported only by `aos/`.
- `psf_moments_consdb.py` — imported only by `psf/`.
- `miw_corner_intrinsic.py` — reached only through `common/scripts/build_optical_state.py`,
  itself an aos-only tool.
- 3 of 5 Python scripts in `common/scripts/` have no caller in the repository.

**4. Output has no provenance and no index.** 65 GB across ~30,000 files with **one**
`README.md` in the entire output tree (`aos/output/archive/`). `aos/output/` is
study-organized under the current `param_set`; every other topic's `output/` is a flat
pile (`psf/output/` is 403 loose PNG files, `optatmo/output/` is 3,244 files plus
directories literally named `old/` and `older/`). Nothing on disk records which code
version or which input produced a given plot. `aos/docs/status/rerun_needed.md` is the
only mechanism preventing a stale plot from being mistaken for a current result, and it
is maintained by hand.

Disk is concentrated, which makes it tractable: **27 GB in `aos/output/archive/`** (four
superseded `param_set`s, last built June 2026, "nothing current reads them") and **28 GB
in the current `param_set`**, of which a single `donuts.parquet` is 11.5 GB. Only ~850
of the files under `aos/output/` are actual result artifacts (353 PDF, 408 parquet, 18
PNG, 10 MP4).

### The code-quality hotspot

`aos/code/science_lut/` — 6 files, **9,476 lines** (30% of all `aos/code/`), created
2026-09-14, four days before this assessment. It was built by **forking rather than
extracting**:

| pair | identical non-blank lines |
|---|---|
| `run_science_lut_analysis.py` ↔ `run_thermal_model.py` | 675 |
| `run_science_lut_analysis.py` ↔ `run_science_lut_report.py` | ~308 |

`run_science_lut_analysis.py` alone is **3,164 lines** and is the source of 30 of the 44
duplicate function names in `aos/code/`. This is the single highest-value refactor in
the repository, and it is recent enough that the context is still fresh.

Separately, and more dangerous than duplication: **three different functions named
`assign_blocks`** with three different behaviours —
`coadd/run_coadd_blocks_miw.py:181` (greedy pointing-set walk, 9 parameters),
`coadd/analyze_dz_goodness_of_fit.py:100` (table lookup by sequence range, 2 parameters),
and `fam_focus/run_fam_focus.py:109`, whose own docstring says it re-implements "the
`coadd` study's greedy pointing-set walk" with its own defaults. Same name, three
semantics. This is the same failure mode as the already-resolved `quality_cut` and
`load_miw` items in `aos/docs/status/code_review_backlog.md`, and it should be recorded
there.

### What is already good — and should not be disturbed

- **The study partition in `aos/`**, and the `docs/` vs `docs/status/` split. Both work.
- **Review discipline.** `code_review_backlog.md` records *why* each resolved item was
  resolved so a later pass does not re-litigate it, and `code_review_findings.md`
  honestly marks itself `Status: stale — verify before acting` with the reason. This is
  better record keeping than most research repositories have.
- **No hardcoded laptop paths in any tracked `.py` file** — verified. The only surviving
  `/home/r/roodman` literals are in the untracked `aos/*snippets.ipynb` scratch notebooks
  and in the review documents that describe the problem.
- **`notes/`** matches its documented convention exactly.
- **Every one of the 67 `aos/code/` files has a module docstring.**

---

## Part A — organization of code and output

Ordered by benefit per unit of disruption. A1–A3 are mechanical and safe; A4–A6 are
judgement calls; A7 needs Aaron's decisions.

### A1. Reclaim 27 GB and make output provenance automatic — **needs Aaron's approval**

`aos/output/archive/` holds four superseded `param_set`s, 27 GB, last built June 2026,
documented as "nothing current reads them". Deleting files is a hard must-ask, so this
is a decision, not an action. The commands are prepared but **not run**:

```bash
du -sh /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/archive
rm -rf /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/archive/fam_danish_1_0_wep17_3_0_bin2x
rm -rf /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/archive/fam_danish_1_1_1_wep17_3_0_bin2x
rm -rf /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/archive/fam_danish_v1_triplets_bin_1x
rm -rf /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/archive/fam_danish_v1_triplets_bin_2x
rm -rf /sdf/group/rubin/u/roodman/LSST/notebooks/rubin-work/aos/output/archive/fam_danish_1_2_0_wep17_7_0_2025
```

A cheaper middle option: delete only the `donuts.parquet` inside each archived
`param_set` (the 3.5 + 3.5 + 3.1 + 1.6 GB per-donut tables, ~12 GB) and keep the small
derived products, which are what one would actually want to consult.

**The durable fix** is a provenance sidecar, written by the code rather than by hand.
Add to `common/utils.py` a `write_provenance(outdir, **fields)` that drops a
`_provenance.yaml` next to every product recording: git commit SHA, `param_set`, input
paths with mtimes, the full command line, UTC timestamp, and hostname. Call it from the
Snakemake rules and from each `run_*.py` `main()`. This replaces hand-maintained
`rerun_needed.md` entries with something a script can check, and it is what makes the
other 19 topics' flat `output/` piles interpretable without reorganizing them.

### A2. Give every topic a `notebooks/` directory (9 topics, 35 notebooks)

Pure `git mv`, no code changes *except* the `sys.path` bootstrap cell. `CLAUDE.md`
already specifies the notebook idiom that survives the move (the `_TOPIC` walk-up), and
`aos/`'s 15 notebooks already use it — so the pattern is proven and copy-pasteable.

Per topic, lowest-risk first:

| topic | notebooks | proposed subdirs |
|---|---|---|
| `astrometry/` | 1 | flat `notebooks/` |
| `olr/` | 3 | flat `notebooks/` |
| `nightlyiq/` | 4 | flat `notebooks/` |
| `psf/` | 3 | flat `notebooks/` |
| `wfs/` | 6 | flat `notebooks/` (pure-notebook topic) |
| `guider/` | 6 | `notebooks/{atmo,moments,stars}/` |
| `blocks/` | 8 | `notebooks/{rotator,t539,trending}/` |

Do one topic per commit, verify the moved notebook still imports, then move on. A topic
with a single study keeps a flat `notebooks/` per the convention.

Also: rename `blocks/ImageCount.ipynb` to `image_count.ipynb` (the only CamelCase
notebook), and decide on the 4 untracked `Untitled.ipynb` and 3 `aos/*snippets.ipynb`
(20 MB of scratch, one of which imports a module that no longer exists). They are
already gitignored, so this is local hygiene only.

### A3. Return topic-private code from `common/` to its owner

Four verified moves, each a `git mv` plus an import fix in the importing topic:

- `common/efd_db.py` → `aos/code/` (1,379 lines, aos-only)
- `common/psf_render.py` → `aos/code/` (aos-only)
- `common/miw_corner_intrinsic.py` → `aos/code/` (aos-only, via `common/scripts/`)
- `common/psf_moments_consdb.py` → `psf/code/` (psf-only — and this fills `psf/code/`,
  which is currently an empty `.gitkeep`)

**`common/FocalPlaneInterpolator.py` needs a decision**: 916 lines, zero importers,
advertised in root `CLAUDE.md`. Either it is dead (delete — must-ask) or it is a tool
you intend to use and it should be *demonstrated* in a notebook. Leaving it advertised
but unused is the worst of the three states.

`common/scripts/{merge_db_shards,backfill_commanded_vmodes,check_undefined}.py` have no
callers — but two of those are from your current session's work, so leave them alone
until that work lands.

### A4. Split the four overloaded `code/` roots into studies

Per `CLAUDE.md`, each new study needs a README entry, a `docs/studies/<study>.md`, and a
`code/<study>/` directory. **Agree the study list with Aaron before moving anything** —
the convention says to ask rather than guess, and the seams below are inferred from
filenames only.

Proposed, for discussion:

- **`smatrix/`** (21 files) → `normalization/`, `thermal/`, `full_modes/`,
  `ofc_compare/`, plus existing `zemax_build/`
- **`guider/`** (29) → `atmo/` (the 11 `guider_atmo_*.py`), `moments/`, `pipeline/`
- **`optatmo/`** (30) → `extract/`, `fit/`, `plot/`, `pipeline/`
- **`filters/`** (14) → `designs/` (the 7 `design_*.py`), `lib/` (`thinfilm.py`,
  `filterstack.py`), plus resolve the `_v2` forks

`guider/` and `optatmo/` each need a `docs/studies/` directory too; `filters/` and
`blocks/` have no `docs/` at all.

### A5. Resolve the `miw` / `coadd` / `static_optics` ambiguity in `aos/`

MIW-labelled work is spread across three `aos/code/` studies: `coadd/` holds
`analyze_miw_bias_regression.py`, `analyze_miw_dz_full_k.py`,
`analyze_miw_field_order.py` and `check_k_truncation.py`; `static_optics/` holds
`run_m3_backprojection_miw.py`, `run_miw_backprojection_surfaces.py` and
`run_miw_joint_fit.py` (all three loading `miw_io.load_miw`); while `miw/` itself holds a
single file. `static_optics/` is really two topics — camera gravity and M3
back-projection / MIW joint fit. This is a decision about what the studies *mean*, so it
is yours to make; it is listed here because it is the one place the `aos/` study
partition genuinely drifted.

### A6. Standardize the `sys.path` bootstrap

There are **six distinct idioms** across 86 `sys.path.insert` calls in `aos/` — inline
`Path(__file__)`, module-qualified `pathlib.Path`, `os.path` with a string `_HERE`,
`_HERE` as a `Path` with a *different* index base, a named `_ROOT`, and two files
reaching into `smatrix/code`. None hardcode absolute paths, so this is a readability and
foot-gun issue rather than a portability bug: `psf_maps_lib.py` uses `parents[2]` while
study files use `parents[3]`, both commented `# repo root`, so a file moved between
levels silently inserts the wrong directory.

Fix by adding one helper — `common/utils.repo_root()` already exists for notebooks — and
converting files to the single `CLAUDE.md` idiom as they are touched during Part C,
rather than in a dedicated sweep.

### A7. Retire or document the empty topics

`camera/`, `des/`, `starcolor/`, `survey/`, `wcs/` are byte-for-byte identical
scaffolding: an `__init__.py` whose docstring says "see README.md", and **no README.md
exists in any of them**. `alerts/` is empty. `scratch/` has been empty since
2026-03-13. Root `CLAUDE.md` lists several as real topics, which overstates the state.
Either give each a one-paragraph README or remove the scaffolding — a decision for you.

---

## Part B — documentation and record keeping

### B1. Fix the root `README.md` — highest priority in Part B

It is actively misleading, which is worse than missing. It currently states:

- "`*.ipynb` — Notebooks live directly in topic dir" — **contradicts the convention** in
  `CLAUDE.md`, and describes the problem A2 exists to fix.
- "`output/` — Small curated outputs (**git-tracked**)" — **the opposite of `.gitignore`**,
  which ignores `*/output/*` entirely.
- Lists **11 topics; there are 20.** Missing: `smatrix`, `optatmo`, `olr`, `wfs`,
  `filters`, `optics`, `nightlyiq`, `notes`, `alerts`, `astrometry`.
- Recommends `~/notebooks/rubin-data/<topic>/` for large outputs; that tree contains
  only an empty `aos/` directory and is unused in practice.

Rewrite as a genuine index: one line per topic linking to its README, in the prose
register `CLAUDE.md` prescribes, with the output conventions stated as they actually are.

### B2. Add the required status header to all 12 topic READMEs

`CLAUDE.md` says every `.md` opens with
`> **Status:** … · **Last updated:** … · **Kind:** …` directly under its H1.
**Zero of 12 topic READMEs have it** — including `aos/README.md`. Only files under
`docs/` and `notes/status/` comply. This is a mechanical fix and it is what makes the
convention self-enforcing: a reader can see at a glance whether to trust the file.

### B3. Refresh the two stale topic READMEs

- **`guider/README.md`** — its notebook table lists **1 of 6** notebooks.
- **`blocks/README.md`** — lists 3 of 8 notebooks, and 7 of 8 `code/*.py` are
  unmentioned, including the whole T539 group.

These are the two most actively developed non-aos topics, so their indexes matter most.
`smatrix/README.md` (15 of 21 code files unmentioned) and `optatmo/README.md` (19 of 30)
are next. `wfs/`, `nightlyiq/` and `optics/README.md` are current and are the models to
copy.

### B4. Document the undocumented cross-topic couplings

Root `CLAUDE.md` documents two couplings (`guider→optatmo`, `guider→aos`) as "real and
intentional". Two more exist and are not documented:
`blocks/code/telemetry_pipeline.py:39-40` → `aos/code` **and** `olr/code`, and
`blocks/code/find_long_exposures.py:25` → `aos/code`. Also
`aos/code/smatrix_vmode/plot_vmode_dof_matrix.py:45` and
`aos/code/static_optics/camera_gravity.py:51` reach into `smatrix/code`. All go through
`sys.path.insert`, so no static check will find them. Add them to the same list.

### B5. Establish a lightweight per-study logbook

This is the record-keeping gap that matters most for a fast-moving analysis repository,
and it is not solved by any existing document. `docs/studies/<study>.md` describes *what
a study is*; `rerun_needed.md` tracks *stale output*; neither records **what was learned
and when**.

Add to each `docs/studies/<study>.md` a dated `## Findings` section, appended to
newest-first, one short entry per result with units per the reporting standard:

```markdown
## Findings

### 2026-09-18 — focus drift within a FAM block
Median intra-block Z4 drift = 0.12 ± 0.03 µm of wavefront (n = 47 blocks);
Spearman rho = 0.31 vs elevation in deg (OCS). Output:
`output/<param_set>/fam_focus/fam_focus_20260918.pdf`, commit ec23101.
```

Three lines per finding, written when the plot is made. It costs almost nothing and it
is the difference between a 65 GB output tree you can interrogate and one you cannot.

### B6. Keep the memory snapshot fresh, and fix the small doc defects

- `notes/claude-memory/` was snapshotted 2026-09-04 and warns it will drift; it is now
  two weeks stale. Re-snapshot, and add it to a monthly cadence.
- `aos/docs/status/code_review_findings.md` is correctly marked stale — once Part C
  starts, either re-anchor its findings or retire it, so there is one live review
  document rather than two.
- `notes/aos-measured-intrinsics/` lacks the `<YYYYMMDD>-` prefix its own
  `notes/README.md` documents.
- `aos/calibration/` is a non-standard directory in a topic root (it holds tracked
  calibration parquet, deliberately negated in `.gitignore`). It is justified, but
  `CLAUDE.md`'s "nothing loose in a topic root" rule should name it as an exception —
  along with `__init__.py`, `Snakefile`, `run_snake.sh`, `snake_config.yaml` and
  `config.yaml`, which are present in several topic roots and are sanctioned in practice
  but not in the text.
- Delete the local `.ipynb_checkpoints/` clutter (7 shadow `.py` copies under `aos/`,
  3 under `common/`). Confirmed **not tracked** in git — `.gitignore` already handles
  them, so this is local-only tidying.

---

## Part C — systematic code review

56 files exceed no threshold worth reviewing blind; 57,179 lines is too much to review
uniformly. Triage by risk, and **review by study, not by file**, so one unit of review
produces one commit and one updated document.

### C1. Fix the review-infrastructure duplication first

There are currently two review documents for `aos/`: `code_review_backlog.md` (current,
records resolved items with reasoning) and `code_review_findings.md` (2026-06-27, marked
stale, line anchors untrustworthy). Before adding to either, re-anchor the still-real
findings from the stale document into the live backlog and retire the stale one. Its
named defects — the paired-Δ SEM missing the 1.2533 median factor in `bounce_lib.py`,
and the silent `nan_to_num` in the u-mode projection — are exactly the kind that
silently bias published numbers, so they should not be lost to a stale anchor.

### C2. Review order, by risk × rate of change

**Tier 1 — highest risk, review now**

1. **`aos/code/science_lut/`** (9,476 lines, 6 files, 4 days old). ~675 verbatim
   duplicated lines between two files; 30 of 44 duplicate function names in the topic.
   Extract the shared `build_pdf`, `page_*` family, `load_target`, `evaluate`/`fit_full`/
   `make_model`, `robust_poly`, `day_obs_to_date` into one module. Do this while the
   context is fresh. **This is the single highest-value item in the whole plan.**
2. **The three `assign_blocks`.** Same name, three semantics, one self-documented as a
   re-implementation. Record in the backlog, then unify or rename so the divergence is
   visible at the call site.
3. **`aos/code/cwfs/run_wfs_dof_compare.py`.** Re-implements `zj_to_fwhm` and `fp_grid`
   (both in `aos/code/aos_fwhm.py`) and `corner_matrix_at` (in `psf_maps_lib.py`) — three
   functions already on its own `sys.path`. Smallest-effort, cleanest win; do it first as
   a warm-up.
4. **The `nan_to_num` / silent-zero NaN handling** from the stale findings document.
   It biases every "corrected" map and SVD projection toward zero and logs nothing.

**Tier 2 — large files that carry results**

`dz_plotting.py` (1,243 lines, 11 importers — highest payoff, highest risk),
`run_dz14_truss.py` (993, imported by 3 `science_lut/` scripts across a study boundary),
`recompute_coadd_metrics.py` (972, 23 `add_argument`), `run_coadd_blocks_miw.py` (939),
`bounce_lib.py` (892). **15 of 67 `aos/code/` files exceed 600 lines and hold 55% of the
topic's code.**

**Tier 3 — the non-aos topics**, reviewed as each is reorganized in A4. Fold the review
into the move: you are already touching every file.

### C3. Method — what each review unit does

Per study, in one sitting, producing one commit:

1. Run `/code-review` scoped to the study's directory. Effort `high` for Tier 1.
2. Verify each finding against the source before acting — the stale-anchor lesson.
3. Fix mechanically-safe items; record behaviour-changing ones in the backlog **with the
   reasoning**, as the existing backlog already does well.
4. Bring docstrings to the Rubin DM standard for files touched, including units on every
   physical parameter. 13 of 54 runnable `aos/` scripts lack the invocation line their
   docstring should carry (concentrated in `coadd/` and `static_optics/`).
5. If a fix changes results, add the affected product to `rerun_needed.md`.
6. Append what was learned to the study's `## Findings` section (B5).

### C4. Add the regression safety net the repository lacks

There are **no tests anywhere** — the only `test_`-named file, `aos/code/test_m1m3.py`,
is a manual EFD probe that pytest would try to collect and that belongs in
`aos/code/infra/`. Refactoring 9,476 lines of `science_lut/` without any test is how a
reorganization silently changes a published number.

Before C2 item 1, add a minimal `aos/tests/` with characterization tests — not unit
tests of intent, but tests that pin *current* behaviour on a small fixture: run the
fitter on ~20 saved visits and assert the DZ coefficients match a stored reference to
within a tolerance. Three or four such tests make every later refactor verifiable. This
is the one piece of genuinely new infrastructure in this plan, and it is what makes
Part C safe rather than hopeful.

---

## Sequencing and conflict avoidance

Another session is actively working (modified: `aos/code/aos_state.py`,
`aos/code/dzfit/run_dz_plots.py`, `aos/code/smatrix_vmode/plot_vmode_dof_matrix.py`,
`common/efd_db.py`, `common/scripts/build_optical_state.py`, `olr/code/nightly_table.py`,
`guider/guider_moments_ensemble.ipynb`, and several docs; untracked: `aos/notebooks/bounce/`,
`common/scripts/{backfill_commanded_vmodes,merge_db_shards,run_build}.*`).

**Rule: this plan touches none of those paths until that work is committed.** That rules
out, for now: A3 (moves `common/efd_db.py`), the `guider/` half of A2, and anything under
`aos/code/smatrix_vmode/` or `aos/code/dzfit/`.

**Safe to start immediately** — disjoint from every modified path:

| unit | touches | risk |
|---|---|---|
| B1 root README rewrite | `README.md` | none |
| B2 status headers | 12 `*/README.md` | none |
| B4 document couplings | `CLAUDE.md` | none |
| A2 for `wfs/`, `psf/`, `nightlyiq/`, `astrometry/` | 14 notebooks in quiet topics | low |
| B3 for `blocks/`, `smatrix/` | 2 READMEs | none |
| C2 item 3 (`run_wfs_dof_compare.py`) | 1 file in a quiet study | low |
| C4 characterization tests | new `aos/tests/` | none — new files only |
| A1 provenance helper | `common/utils.py` (not currently modified) | low |

**Deferred until the other session lands:** A3, A4 for `guider/`, A2 for `guider/`,
C2 items 1–2 (`science_lut/` is adjacent to `build_optical_state.py` work).

**Decisions needed from Aaron before the relevant unit can start:**

1. **A1** — delete the 27 GB archive, delete only the ~12 GB of archived `donuts.parquet`,
   or keep all of it?
2. **A3** — `FocalPlaneInterpolator.py`: delete, or keep and demonstrate?
3. **A4** — approve the proposed study splits for `smatrix/`, `guider/`, `optatmo/`,
   `filters/`, or amend the study lists.
4. **A5** — how should `miw` / `coadd` / `static_optics` divide in `aos/`?
5. **A7** — retire `camera/`, `des/`, `starcolor/`, `survey/`, `wcs/`, `alerts/`,
   `scratch/`, or give each a README?
</content>
