---
study: closed_loop
topic: aos
status: parked
question: "What delivered PSF does AOS closed-loop control give over a FAM visit sequence, given the corner-sensor recovery and a proportional controller?"
answer: ""
uses: {}  # old aos/output tree
started: 2026-09-07
---

# `closed_loop`

> **Code:** `code/run_closed_loop.py` (with `aos/code/psf_maps_lib.py`) · **Write-up:** [`closed_loop.md`](closed_loop.md) · **Output:** `aos/output/closed_loop/<P>_<M>/`

## Question

What delivered PSF does AOS closed-loop control give over a FAM visit sequence, given the corner-sensor recovery and a proportional controller?

## Current answer

None current: the eight PDFs in `output/` predate the split from `psf` and the single-`<mi>` change.

## Log

### 2026-09-07 — backfilled from docs and history

Moved from `aos/code/closed_loop/`. It imports the WFS-mimic defaults from `rubinwork.wfs_mimic`. Converted 2026-10-23 in phase 5.
