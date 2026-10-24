---
study: psf_fp_maps
topic: aos
status: parked
question: "What focal-plane PSF FWHM, ellipticity and shape does a given optical wavefront deliver, measured the way the survey measures them?"
answer: ""
uses: {}  # old aos/output tree
started: 2026-09-06
---

# `psf_fp_maps`

> **Code:** `code/run_psf_fp_maps.py` (with `aos/code/psf_maps_lib.py`; Slurm `aos/run_psf_fp_maps.sbatch`) · **Write-up:** [`psf.md`](psf.md) · **Output:** `aos/output/psf/<P>_<M>/`

## Question

What focal-plane PSF FWHM, ellipticity and shape does a given optical wavefront deliver, measured the way the survey measures them?

## Current answer

A forward tool, no current result: its six PDFs predate the 2026-09-07 split and the single-`<mi>` change (`../docs/status/rerun_needed.md`).

## Log

### 2026-09-20 — backfilled from docs and history

Moved from `aos/code/psf/`, renamed `psf_fp_maps` because `psf` is a topic. It imports the WFS-mimic wedges from `rubinwork.wfs_mimic`. Converted 2026-10-23 in phase 5.
