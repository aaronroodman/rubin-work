# Phase 4 handoff: skills and generators

> **Status:** planned, nothing built · **Last updated:** 2026-10-16 · **Kind:** working
> state (handoff)

Phase 4 of `notes/status/organization_plan.md` (section 8): the `/start-study`, `/wrap` and
`/review` skills, the `STUDIES.md` and `PRODUCTS.md` generators, the updated style skills
and an updated `CLAUDE.md`. The skill specifications are in section 7; the dependency
rules `/review` enforces are in section 2; the target layout is section 3; `run.json` is
in section 5. This file holds the step list and the decisions Aaron has to make.

## Decisions for Aaron

1. **The `status` values.** Section 7 says `active | parked | done | dropped`. Two of the
   three `study.md` files already written use `active`, and `aos/miw_hull/study.md` uses
   `answered`. Options: (a) add `answered` (the question is answered but the study may
   continue) to the set; (b) map `answered` to `done` and edit `miw_hull`.
   Recommendation: (a). Blocks step 2.
2. **Generated files in git.** Commit `STUDIES.md`, `PRODUCTS.md` and the topic
   `README.md` indexes (plan section 3 says the topic README "is a generated index"), or
   generate them on demand only. Committing makes them readable on GitHub and in a fresh
   session, at the cost of a diff on every `/review`. Recommendation: commit them. Also
   blocks step 2: if topic READMEs are generated, the hand-written ones (`aos/README.md`
   and others) must be moved or merged first, which is phase 5 work.
3. **`PRODUCTS.md` size column.** Section 3 lists "size". `catalog.list_products()` gives
   `n_files` but not bytes, and summing bytes over every build walks about 3 GB of
   `value_added` alone, plus every `fam_tables` build. Options: (a) record bytes in the
   manifest at write time and read it from there; (b) walk the tree in the generator;
   (c) drop the column. Recommendation: (a), with older manifests showing blank.
4. **Who writes `run.json`.** Section 5 says every study run writes it; nothing does
   today, and `aos/miw_hull`'s output directory is flat, not in dated run directories.
   Options: (a) a small library helper, `rubinwork.common.study_run(study, label)`, that
   makes the dated directory and writes `run.json` (git commit, product builds read,
   command); (b) leave it to `/wrap` to reconstruct from git and the catalog. `/wrap`
   reads `run.json`, so without (a) its `uses:` update is a guess. Recommendation: (a),
   built in step 3, with the catalog's `load()`/`path()` recording what each process read.
5. **`/review` dependency checks: report only, or also fail a test?** The section 2 rules
   (no product import from a library, no study import, no study reading another study's
   output, no hand-built product paths) can be a `pytest` check that fails, or only a
   flag in the `/review` report. The known wrong-way imports in section 2 would fail it
   today until phase 5 fixes them. Recommendation: report only in phase 4, with an
   allow-list of the known cases; turn it into a test in phase 5 once they are fixed.
6. **Skill form.** Project skills (`.claude/skills/<name>/SKILL.md`, as today, invoked
   with `/name` and auto-loaded by description) or plain slash commands. Skills can carry
   a helper script beside `SKILL.md`. Recommendation: skills, each with its generator or
   helper as a Python module in `rubinwork/` so it is testable, and the `SKILL.md` saying
   when to run it.
7. **`/wrap` commits.** Section 7 says `/wrap` commits. Commit only, or commit and push?
   Recommendation: commit only; push stays an explicit step.
8. **The 14-day staleness flag in `/review`** — from the last `Log` entry date, or from
   the last git commit touching the study directory? Recommendation: the later of the two.

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
