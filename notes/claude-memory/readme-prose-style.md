---
name: readme-prose-style
description: "How Aaron wants README/doc prose written — content summary, acronyms defined, no development history"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a5ccd75-0c0d-40e5-a0c8-b305bc328359
  modified: 2026-09-06T16:29:37.759Z
---

READMEs and docs open with a **high-level description of the content**, in plain prose,
with every acronym defined on first use. They describe what the work **is**, not how the
code got that way.

**Why:** Aaron rewrote my `aos/README.md` opening (2026-09-06) to show the target
register. Mine led with implementation detail ("per-donut wavefront tables... a Snakemake
pipeline runs the MIW chain per `param_set`") and then editorialized about the repo
("this is the largest topic in the repo — 56 Python files"). His version is a general,
high-level summary of content with acronyms spelled out, and it does not dwell on the
history of his code development.

His model text for `aos/README.md`:

> Analysis and Development for the Rubin Observatory Active Optics System (AOS). This
> directory contains multiple studies of AOS engineering data and development of
> calibrations and methods for AOS operation. These include the construction of the
> Measured Intrinsic Wavefront (MIW) from Full Array Mode (FAM) data, study of
> correlations between Double Zernike (DZ), v-modes and Rubin telemetry, analysis of
> bounce test data for Look-Up-Tables (LUT), and comparisons between FAM and Corner
> Wavefront Sensor (CWFS) data.

**How to apply:**
- **No development history.** No "consolidates the former X and Y", no lists of removed
  or renamed notebooks, no "ported from notebook Z". Git history holds that.
- **Define acronyms on first use** — AOS, FAM, MIW, DZ, CWFS, LUT, OFC, DOF, PSF, FWHM,
  EFD, ConsDB. Write for a competent reader who does not know the project shorthand.
- **No meta-commentary about the repo or the doc.** Not "this is the largest topic", not
  "this file is the map", not "Phase 7 will fix this", not "do not re-derive these".
- **Describe a study/section by content, not as a rhetorical question.** "Comparison of
  the optical state recovered from the CWFS against the FAM measurement", not "Does the
  corner WFS recover the same optical state as FAM?"
- State outstanding work as a plain fact about current state, not as a plan.
- Applies to all of `rubin-work`, and to the per-study docs as well as READMEs.

The rule is also in the root `CLAUDE.md` under "Writing style for READMEs and docs", so
it loads every session. Related: [[reporting-units-standard]] (numbers still carry
quantity name and units, everywhere including docs).
