---
study: processing_compare
topic: aos
status: dropped
question: "Do two reductions of the same donut data agree across code versions, binnings and fitting algorithms?"
answer: "Dropped as a study: it is a regression harness. compare_donuts.py became the library rubinwork.compare_donuts; the TARTS notebook belongs to todo item 1."
uses: {}  # old aos/output tree
started: 2026-09-06
---

# `processing_compare`

> **Code:** `code/compare_fam_processings.py`; the matching machinery is the library `rubinwork.compare_donuts` · **Notebooks:** `notebooks/` · **Write-up:** [`processing_compare.md`](processing_compare.md)

## Question

Do two reductions of the same donut data agree across code versions, binnings and fitting algorithms?

## Current answer

Dropped as a study (triage): a regression harness, not a question. `compare_donuts.py` became `rubinwork.compare_donuts`; the Danish against TARTS notebook (20260713) goes with todo item 1.

## Log

### 2026-10-12 — backfilled from docs and history

Moved from `aos/code/processing_compare/` and `aos/notebooks/processing_compare/`. The TARTS notebook now imports `rubinwork.aos_state`, `rubinwork.common` and the `cwfs_tables` builder directly. Converted 2026-10-23 in phase 5.
