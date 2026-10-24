---
study: lut
topic: aos
status: dropped
question: "What averaged degree-of-freedom look-up table do the FAM Double Zernike fits give, collapsed over elevation and rotator angle?"
answer: "Dropped as a study: the LUT is the dof_lut product. The open question, plain against post-MIW input, stays with the product; the switch to post-MIW did not make the recovered DOF plausible."
uses: {}  # old aos/output tree
started: 2026-09-08
---

# `lut`

> **Code:** `rubinwork/products/dof_lut/` (the product; a shim remains at `aos/code/lut/run_build_lut.py`) · **Write-up:** [`lut.md`](lut.md)

## Question

What averaged degree-of-freedom look-up table do the FAM Double Zernike fits give, collapsed over elevation and rotator angle?

## Current answer

Dropped as a study (triage): the averaged-DOF LUT is the `dof_lut` product. On the `miw` 3a reference (169 visits) the post-MIW input moved `M2_dy` from −2407.11 to −2407.34 µm, so the implausible magnitudes are an ill-conditioned inversion, not a missing subtraction.

## Log

### 2026-10-13 — backfilled from docs and history

The study doc moved from `aos/docs/studies/lut.md`. The code had already moved to the product in phase 3a. Converted 2026-10-23 in phase 5.
