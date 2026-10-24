# Phase 4 handoff: skills and generators

> **Status:** done; open items answered 2026-10-18 ·
> **Last updated:** 2026-10-18 · **Kind:** working state (handoff)

Phase 4 of `notes/status/organization_plan.md` (section 8): the `/start-study`, `/wrap` and
`/review` skills, the `STUDIES.md` and `PRODUCTS.md` generators, the updated style skills
and an updated `CLAUDE.md`. The skill specifications are in section 7; the dependency
rules `/review` enforces are in section 2; the target layout is section 3; `run.json` is
in section 5. This file holds the step list, the decisions taken and what was built.

## Decisions taken

Aaron, 2026-10-16:

1. `answered` is added to the status set: `active | answered | parked | done | dropped`.
2. Generated files are committed. `STUDIES.md` and `PRODUCTS.md` are at the repo root.
   Topic `README.md` generation waits for phase 5, after the hand-written topic READMEs
   are merged. Phase 4 overwrites no existing `README.md`.
3. `PRODUCTS.md` size: the manifest records total bytes at write time. No tree walk.
4. `run.json` is written by the library helper `rubinwork.common.study_run(study,
   label)`: dated run directory, git commit, command, and the product builds read as
   recorded by the catalog's `load()`/`path()`.
5. The section 2 rules are report-only in `/review`, with an allow-list of the known
   wrong-way cases. No failing test until phase 5.
6. Skills in `.claude/skills/<name>/SKILL.md`; the logic in `rubinwork/` modules with
   tests.
7. `/wrap` commits, never pushes.
8. The 14-day staleness flag counts from the later of the last Log entry date and the
   last git commit touching the study directory.

Deletion approved: `.claude/skills/rubin-new-study/` and `.claude/skills/rubin-output-layout/`,
once replaced (done in step 7). Nothing else.

Aaron, 2026-10-18, on the open items:

9. The local `/wrap aos/miw_hull` commit is pushed (`8c9b798`).
10. The 5 new rule violations of the dry-run `/review` go into `ALLOWED`, each to be fixed
    in phase 5.
11. The `blocks/` allow-list entry is removed. The `blocks` Snakefile's reference to
    `../aos/code/aos_state.py` is a known dependency; the checker does not scan
    Snakefiles.
12. Deletion approved: `rubinwork/common/tests_placeholder` and
    `rubinwork/studies/tests/__init__.py`. Nothing else. pytest collects the studies
    tests without the `__init__.py` (431 tests passed either way), so it is removed.
    `tests_placeholder` is untracked; the session's `rm` was blocked by the permission
    settings, so it is still on disk for Aaron to remove.
13. `blitz_cwfs_vs_danish12_20260315.ipynb`: repair only if mechanical. The JSON was
    valid; the failure was in `/review`, which dropped every line starting with `%` as
    a magic, including a `% (...)` string-format continuation. Fixed in `review.py`
    (a cell that parses whole keeps its `%` lines); the notebook is unchanged.

## Step list

1. **Front-matter schema and parser.** A small library module (proposed
   `rubinwork/studies.py`) that finds every `<topic>/<study>/study.md`, parses the YAML
   front matter, and validates it: required keys, the `status` set (decision 1), `uses:`
   entries resolving through the catalog. Tests against the three existing `study.md`
   files. Everything after this step reads through it.
2. **The generators.** `python -m rubinwork.studies index` writes `STUDIES.md` (study,
   topic, status, question, answer, products used); `python -m rubinwork.products.catalog
   index` writes `PRODUCTS.md` from `list_products()` (product, variant, current build,
   latest build, created, size per decision 3). Both deterministic, so a rerun with no
   change gives no diff. Decision 2 settles whether they are committed.
3. **`run.json` helper** (decision 4): a dated run directory under
   `catalog.studies_root()` and a `run.json` with git commit, command and the product
   builds read.
4. **`/start-study`.** Replaces `rubin-new-study`. Proposes topic, name and the products
   used; on confirmation makes `<topic>/<study>/` with `study.md`, `code/`,
   `notebooks/`, `figures/`, the `output@` symlink to `studies/<topic>/<study>/` and the
   data directory, then reruns the `STUDIES.md` generator. Detects "this is a product
   change" and points at the product's `variants.yaml` instead.
5. **`/wrap [study]`.** Reads git log since the last Log entry and any new `run.json`;
   appends a dated Log entry, updates Current answer, `status` and `uses:`, copies one or
   two key PNGs to `figures/`, commits (decision 7). On a study with no `study.md` it
   builds a starting entry from history and the existing docs — this is what phase 5
   backfill uses.
6. **`/review [topic]`.** Reruns both generators; flags active studies untouched for more
   than 14 days (decision 8), done studies with no answer, code under a topic not covered
   by any study, studies pinned to a build that is no longer `current`, and the section 2
   rule violations (decision 5).
7. **Style skills.** Retire `rubin-output-layout` (replaced by sections 3 and 5); update
   `rubin-notebooks` and `rubin-doc-style` for the study layout (`study.md` sections,
   `figures/`, `output@`). `rubin-docstrings` is unchanged.
8. **`CLAUDE.md`.** Replace the conventions table and the `sys.path` / `parents[N]`
   section with the study layout and the three skills; keep the old idiom described only
   until the shims go in phase 5.
9. **Dry run on the three existing studies.** Run `/review`, then `/wrap` on
   `aos/miw_hull` and `camera/itl_dip`, and check the result by eye before phase 5 uses
   the skills on every topic.

## Order, and why

Steps 1 to 3 first: they are plain library code with tests, and every skill depends on
them. The parser and the two generators are also the cheapest useful output — `STUDIES.md`
and `PRODUCTS.md` are readable the day they exist, with no skill behaviour to get right.

Then `/start-study`, because it is the simplest skill and fixes the directory shape the
other two read. `/wrap` comes before `/review` because phase 5 depends on `/wrap`'s
backfill mode for every converted study, while `/review` is a report. The style skills and
`CLAUDE.md` come last, once the layout they describe is real, so they describe what the
skills do rather than what the plan says.

## Constraints already known

- Three `study.md` files exist (`aos/miw_hull`, `aos/unpaired_intra_vs_extra`,
  `camera/itl_dip`) and match section 7's front matter apart from decision 1. The parser
  has to accept them as they are.
- The other 18 study docs are under `<topic>/docs/studies/*.md`, mostly in `aos/`.
  Converting them is phase 5, not phase 4.
- `catalog.studies_root()` already exists and gives `<data root>/studies`.
- Phase 4 changes no product and no science result, so section 12's "continue without
  asking" covers it once the decisions above are answered.

## Done (2026-10-16)

| step | commit | what |
|---|---|---|
| 1 | `f516212` | `rubinwork/studies/` (`core.py`): find, parse, validate `study.md` |
| 2 | `60ed760` | `python -m rubinwork.studies index` → `STUDIES.md`; `python -m rubinwork.products.catalog index` → `PRODUCTS.md`; manifests record `bytes` |
| 3 | `a8b39a5` | `rubinwork/common/runs.py`: `study_run`; the catalog calls `record_read` on every resolved build |
| 4 | `fb8ea8d` | `/start-study` skill; `python -m rubinwork.studies new` and `variants` (`new.py`) |
| 5 | `e40e1f7` | `/wrap` skill; `wrap-gather` and `wrap-apply` (`wrap.py`) |
| 6 | `9dc924c` | `/review` skill; `python -m rubinwork.studies review` (`review.py`) |
| 7 | `c236fba` | `rubin-notebooks` and `rubin-doc-style` updated; the two retired skills removed |
| 8 | `9fa422a` | `CLAUDE.md`: study layout and the skill table |
| 9 | local only | dry run, below |

Both generators rerun with no git diff. Full `pytest rubinwork`: see the closing report.

**Dry run.** `/review` found 3 studies, 2 schema warnings (both aos `answer:` lines not
valid YAML), no stale or superseded study, 323 tracked code files not under any study,
5 new rule violations and known ones in 9 files (count). `/wrap aos/miw_hull`: no work since its last
Log entry, so the only change is the `answer:` line re-quoted; that commit is local and
unpushed for Aaron's review. `/wrap camera/itl_dip`: nothing new, no commit.

## Tried and rejected, and non-obvious findings

- **Strict YAML rejects two of the three existing `study.md` files.** `aos/miw_hull` and
  `aos/unpaired_intra_vs_extra` have `answer:` values with an unquoted `: `. The parser
  parses each top-level key on its own and falls back to plain text with a warning;
  `/wrap` re-quotes on its next write. Rejecting the files was not an option (they must
  parse as they are).
- **Decision 3 needed less than it said.** Every existing manifest already has per-file
  `bytes`, so old builds are sized from those, not shown blank. New manifests also carry
  a `bytes` total.
- **Sizes and dates in the generated files come from the manifests and Log entries, not
  today**, so a rerun gives no diff; the "Last updated" date is the newest build or Log
  entry.
- **`data_root()` moved to `rubinwork.common.runs`**, re-exported by the catalog
  unchanged. The catalog has to report reads to the run recorder, and a library must not
  import a product, so the dependency runs product → library.
- **`/wrap` "since the last entry" is "after the commit that last wrote `study.md`"**,
  not a date or hash search: Log entries cite commits inconsistently (`miw_hull` cites
  none of its four), and a same-day date cut picks up the commit that wrote the entry.
- **`/start-study` pins `current`** to the build it names, so `uses:` never says
  `@current`.
- **Rule checks parse the code with `ast`**, notebooks included; a grep hit every
  docstring that names a path. A loose notebook in a flat `<topic>/notebooks/` is not
  treated as a study, so the `optics/` fresnel notebooks importing `rubinwork/fresnel/`
  are not flagged.
- **`blocks/` allow-list entry matched nothing.** `blocks` reads its own
  `output/t539_closedloop_aos_*.parquet`; the section 2 row may be stale. Its Snakefile
  does reference `../aos/code/aos_state.py`, which the checker does not scan (only `.py`
  and `.ipynb`).
- `rubinwork.studies` is tooling, outside the library/product split; the rule checks
  skip it.

## Open, for Aaron

All answered 2026-10-18 (decisions 9 to 13).

## Next

Phase 5: convert each topic's studies with `/wrap` backfill (`wrap-gather` on a
directory with no `study.md` lists its docs and commits), fix the section 2 violations,
generate topic READMEs after merging the hand-written ones, remove the shims.
