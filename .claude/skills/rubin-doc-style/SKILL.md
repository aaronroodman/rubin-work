---
name: rubin-doc-style
description: Writing or revising any README.md or Markdown doc in rubin-work: prose register, acronyms, status header line, and which directory each kind of .md belongs in.
---

# Documentation style and placement for rubin-work

The Writing rules in the root `CLAUDE.md` (plain, short, direct) apply here too.

## Writing style for READMEs and docs
A `README.md` opens with a **high-level description of the content**, in plain prose,
with every acronym defined on first use. It says what the work *is* — not how the code
got that way. Aaron's model for `aos/README.md`:

> Analysis and Development for the Rubin Observatory Active Optics System (AOS). This
> directory contains multiple studies of AOS engineering data and development of
> calibrations and methods for AOS operation. These include the construction of the
> Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) data, study of
> correlations between Double Zernike (DZ), v-modes and Rubin telemetry, analysis of
> bounce test data for Look-Up-Tables (LUT), and comparisons between FAM and Corner
> Wavefront Sensor (CWFS) data.

Follow that register throughout. Specifically:

- **Describe content, not development history.** No lists of renamed or removed files,
  no "consolidates the former X and Y", no "ported from notebook Z". That is what git
  history is for.
- **Define acronyms on first use** — AOS, FAM, MIW, DZ, CWFS, LUT, OFC, DOF, PSF, FWHM,
  EFD, ConsDB. Assume a competent reader who does not know this project's shorthand.
- **No meta-commentary about the repository or the documentation itself.** Not "this is
  the largest topic", not "this file is the map", not "Phase 7 will fix this", not "do
  not re-derive these".
- **Describe a study by its content, not as a rhetorical question.** "Comparison of the
  optical state recovered from the CWFS against the FAM measurement", not "Does the
  corner WFS recover the same optical state as FAM?"
- Outstanding work is stated plainly as a fact about the current state ("splitting these
  per study is outstanding work"), not as a plan or a scolding.

Status headers, units, and the file-location rules below still apply.

## Markdown docs
Every `.md` file has exactly one home, by kind:

| kind | location | example |
|---|---|---|
| topic scope + index | `<topic>/README.md` | `aos/README.md` |
| Claude scoping rules | `<topic>/CLAUDE.md` | `aos/CLAUDE.md` |
| durable reference, conventions, investigation writeups | `<topic>/docs/` | `smatrix/docs/conventions.md` |
| transient working state — handoffs, todo lists, review backlogs, plans | `<topic>/docs/status/` | `aos/docs/status/code_review_findings.md` |
| repo-wide working state (belongs to no one topic) | `notes/status/` | `notes/status/memory_cleanup_plan.md` |
| outward-facing deliverable (Slack post → `lsstdoc` tech note) | `notes/<slug>/` | `notes/aos-measured-intrinsics/` |

Rules:
- **Filenames are snake_case**, like notebooks and Python modules. `README.md` and
  `CLAUDE.md` are the only uppercase names.
- **Nothing loose in a topic root** except `README.md` and `CLAUDE.md`.
- **Every doc opens with a status line** directly under its H1, so a reader knows
  immediately whether to trust it:
  `> **Status:** current · **Last updated:** YYYY-MM-DD · **Kind:** reference (conventions)`
  Use `Status: stale — verify before acting` once the content has been outrun by the
  code, and say what specifically went stale.
- **The `docs/` vs `docs/status/` split is the point**: a stale handoff must never be
  mistaken for a live convention. If a doc records *where the work is*, it is status; if
  it records *how something is defined or what was concluded*, it is reference.
- **Index every doc in the topic's `README.md`** — a doc nothing links to is a doc
  nobody finds.
