# Claude context cleanup + code review plan

> **Status:** in progress — Phases 1–4 done, Phase 5 next · **Last updated:** 2026-09-04 · **Kind:** working state (plan)

Plan for rationalizing the `CLAUDE.md` / memory files in this repo, and for the code
review that follows. Written 2026-09-04. Supersedes
`~/Claude/laptop_memory_export_request.md` (the laptop-export request, now complete).

Work from the repo root: `cd ~/notebooks/rubin-work && claude`

## Background

Context for this repo had fragmented across four locations — two disjoint snapshots
from two different laptop Claude accounts (`.claude-memory/` and
`notes/claude-memory/`, 0 index entries in common), plus untracked S3DF memory in
`.claude/`, plus a stranded memory dir under the wrong project bucket on S3DF. Of 38
indexed memories, 29 pointed at files that existed only on the laptop.

Memory dirs under `~/.claude/` are keyed to one machine + one account and do not
transfer. `CLAUDE.md` and in-repo files do. So the durable home for anything that
matters is the repo.

## Phase 1 — consolidate memory into notes/claude-memory/ — DONE

Done by the laptop account, commit `bee5e1d`, pulled to S3DF and verified 2026-09-04:

- 34 memory files, 34 index entries, **0 dead links** (was 29 of 38 dead)
- `.claude-memory/` deleted; `notes/claude-memory/` is now the single location
- Only `MEMORY.md` and `README.md` are unindexed, which is correct

Nothing further needed here.

## Phase 2 — promote standing instructions into CLAUDE.md — DONE

Done 2026-09-04. A "Working with Aaron" section now sits in the root `CLAUDE.md`
between "Repository Structure" and "Conventions", with six subsections: autonomy and
hard stops, commands handed to Aaron, paths across environments, reporting numbers,
fitting AOS data. Each rule names its source memory file in parentheses.

Several memories are behavioral rules, not project facts. Memory is the wrong home:
it is per-machine and per-account, so these silently fail to apply elsewhere.
`CLAUDE.md` loads every session on every machine.

Rules covered:

- Ask before SLAC / USDF access (`ask-before-slac`, `usdf-access-slacrd`)
- Ask before submitting batch jobs — Slurm via `run_snake.sh --mode batch`
  (`s3df-batch-job-rules`)
- Ask before deleting files (`work-autonomy-and-guardrails`); otherwise act without
  asking
- Give full copy-paste-ready commands, no ellipsis (`full-commands-no-ellipsis`)
- Every number cited carries its quantity name and units, or "dimensionless" with the
  ratio spelled out (`reporting-units-standard`)
- Prefer robust fits (Huber); **ask which robust method** before implementing; show
  both Pearson r and Spearman rho (`robust-fits-aos`)
- Sync repos with Aaron's `gitpull` script, not raw `git pull` (`gitpull-script`)
- Python interpreter differs by environment: MacPorts `python3` on the laptop, bare
  `python` on RSP/USDF (`python-interpreter-by-env`)
- Use `/sdf/group/rubin/u/roodman/LSST/...` paths everywhere; `/home/r/roodman/u/...`
  is RSP-only and breaks batch (`usdf-mount-paths`)

Keep the memory files as the detailed record — `CLAUDE.md` carries the rule, the
memory carries the reasoning.

## Phase 3 — track the S3DF memory — DONE

Done 2026-09-04. All three files copied into `notes/claude-memory/` and tracked;
index and README updated. `notes/claude-memory/` is now 37 files / 37 entries, still
0 dead links in either direction (nothing indexed-but-missing, nothing
file-but-unindexed).

**Open question resolved: copy, do not move.** The in-repo
`.claude/projects/-sdf-home-r-roodman-notebooks-rubin-work/memory/` copies stay in
place and stay untracked. That is the path Claude Code reads natively on this
machine, so moving the files out would drop those rules from live S3DF sessions —
and Phase 2 only promoted the *rules* into `CLAUDE.md`, not the reasoning. This also
matches the pattern `notes/claude-memory/README.md` already documents: that directory
is a versioned **snapshot**, the live originals stay under `.claude/` and will drift.
Re-copy to refresh.

**Correction found while doing Phase 2:** the 3 files are split across two
directories, not one. The plan above assumed all three were in the repo.

`rubin-work/.claude/projects/-sdf-home-r-roodman-notebooks-rubin-work/memory/` is
untracked (`??` in git status) and holds only 2 of them, plus its own `MEMORY.md`:

- `work-autonomy-and-guardrails.md`
- `s3df-batch-job-rules.md`

`robust-fits-aos.md` is **not** there — it lives outside the repo, at
`~/.claude/projects/-sdf-home-r-roodman-notebooks-rubin-work/memory/`, which is that
dir's only file. The in-repo `MEMORY.md` indexes it anyway, so that index entry is a
dead link from the repo's point of view.

Track all three. Keep `.claude/settings.local.json` ignored — it is machine-local.

Open question: whether to keep them at that path (which Claude Code reads natively on
this machine, but the slug is machine-specific) or move them into
`notes/claude-memory/` with everything else. Leaning toward moving them, for one
location, since Phase 2 promotes their content into `CLAUDE.md` anyway.

## Phase 4 — per-subproject CLAUDE.md — DONE

Done 2026-09-04. `aos/CLAUDE.md` and `guider/CLAUDE.md` written, plus a "Topic
independence and shared code" subsection in the root `CLAUDE.md`.

Deliberately **behavioral, not descriptive** — `aos/README.md` is already 397 lines of
pipeline reference and `guider/docs/status/handoff_guider_session_2026-09.md` is 199 lines of current
state, so both new files point at those and carry only what is easy to get wrong: which
code lives outside the repo, frame/unit traps, hard-stop reminders, and which results are
still open questions rather than settled facts.

Corrections found while writing them (all verified against the source, not the docs):

- `dz_fitting.py` is in the **external** `ts_intrinsic_wavefront` package, not
  `aos/code/` — the `robust-fits-aos` memory says "as used in `code/dz_fitting.py`",
  which reads as in-repo. The in-repo Huber uses are
  `run_wfs_corner_compare.py` and `run_wfs_dof_compare.robust_fit`.
- The guider `guider_*` names in the HANDOFF §6 are per-night **output files**, not
  Snakefile rule names — the rules are `moments`, `combine`, `psfmoments`, `movie`,
  `movies`, `rtvplot`, `rtvplots`, `plots`.
- `guider/` has real `sys.path` couplings into `optatmo/code`
  (`moments_hsm.measure_hsm_moments`) and `aos/code`
  (`aos_trim.make_consdb_client`) — invisible to static import checks, so the root
  CLAUDE.md now names them before anyone refactors.
- In `aos/code/`, `common` in an import almost always means the external
  `lsst.ts.intrinsic.wavefront.common`, not this repo's `common/`. Only
  `plot_visits_summary.py` uses the repo's.
- Guider `--mode batch` submits **one job per night**, so `--day-obs A,B,C` is three
  submissions — differs from `aos/`, and from the single-job description in
  `s3df-batch-job-rules`.

### Original plan

The repo has ~20 topic subdirectories that are mostly independent work but share
`common/` and one git history. Launch from the repo root (not a subdirectory) so
shared-code and git context stay visible; get per-topic scoping from nested
`CLAUDE.md` files, which load when files in that subtree are touched.

Start with the two biggest:

- `aos/CLAUDE.md` — MIW, FAM coadds, DZ fitting, sensitivity/v-modes. Existing docs
  to reference rather than duplicate: `aos/docs/miw_coadd_equations.md`,
  `aos/docs/status/miw_investigation_handoff.md`, `aos/docs/camera_gravity.md`
- `guider/CLAUDE.md` — pipeline commands, `summit_utils` fork state, bias/streak work

The root `CLAUDE.md` should name which subprojects are independent and where the
shared code lives, so a session working on `aos/` does not wander into `guider/`.

## Phase 5 — reorganization and cleanup of aos/code

**Do this before any new code review.** The point is to get the tree into the shape
it should have before spending effort on a review whose line references would then
immediately go stale.

Scope to be defined at the time, but expected to include:

- Factor genuinely shared helpers out of `aos/code/` into `common/` — the old review
  flagged cross-file duplication as a candidate for a shared utils module
- Resolve the angle-unit heuristic that the old review found duplicated across files
  ("fix once at source")
- Retire dead code and superseded scripts
- Settle path conventions (see `usdf-mount-paths`)

## Phase 6 — fresh code review of aos/code

Only after Phase 5. A new review from scratch, not a continuation of the old one.

### The old review is superseded — treat as a cross-check, not a backlog

`aos_code_review_2026-08.md` (2026-08-09, 14 KB) reviewed all 35 files / ~13K lines
of `aos/code/` via 6 parallel reviewers, and is a severity-ordered checklist with
`file:line` anchors. Every item is still unchecked. It currently sits in the wrong
project bucket on S3DF, at
`~/.claude/projects/-sdf-home-r-roodman-notebooks/memory/aos_code_review_2026-08.md`.

It is **obsolete as a task list** — the code has moved since, and Phase 5 will move it
much more, so the line numbers cannot be trusted. It is **not worthless**: the
specific defects it names are likely still real. Examples worth re-checking:

- NaN is truthy, so `np.nanpercentile(...) or 0.01` propagates NaN into vmin/vmax
  (`run_psf_fp_maps.py`)
- Undocumented row-order invariants between sidecar parquet and `donuts.parquet`
  (count mismatch caught, order mismatch not)
- Use of the private `svd._keep()` from ts_ofc
- Several O(N²) loops and a quadratic full-parquet-rewrite checkpoint
- Bare `except: continue` hiding schema drift

Plan: before deleting it, move it into `aos/` as a dated, clearly-superseded
reference (e.g. `aos/notes/code_review_2026-08_superseded.md`), then after the new
review, diff old against new — items that recur are confirmed real and were never
fixed; items that vanished were either fixed by Phase 5 or were false positives.

## Conventions settled along the way

- Launch Claude from the repo root, always: `cd ~/notebooks/rubin-work && claude`.
  Memory is keyed by working directory, so a consistent launch point keeps it in one
  bucket; the repo root also gives git context and access to `common/`.
- User-level, cross-project preferences go in `~/.claude/CLAUDE.md` (loads everywhere,
  regardless of directory). Project facts go in the repo's `CLAUDE.md`.
- `/sessions` (user-level slash command, `~/.claude/commands/sessions.md`, backed by
  `~/.claude/scripts/list_sessions.py`) lists past sessions across all project
  directories with their first prompts, for finding earlier work.

Source session for this plan: `a8bc4e71-405f-4249-81f6-4c1f69944554` (run from
`~/Claude`). Resume with
`claude --resume a8bc4e71-405f-4249-81f6-4c1f69944554`.
