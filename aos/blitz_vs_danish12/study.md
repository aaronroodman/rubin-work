---
study: blitz_vs_danish12
topic: aos
status: done
question: "Do the Danish 1.3 blitz unpaired wavefronts agree with Danish 1.2 on the same FAM and corner-sensor visits, and where do they differ?"
answer: "Same convention; the DZ offsets Z7 +0.1388 and Z6 -0.1116 um of wavefront (mean offsets) come from unpairing, not from the code version."
uses: {}  # old aos/output tree
started: 2026-09-22
---

# `blitz_vs_danish12`

> **Notebooks:** `notebooks/blitz_vs_danish12_20260315.ipynb` (FAM), `notebooks/blitz_cwfs_vs_danish12_20260315.ipynb` (corner sensors) · **Write-up:** [`../../rubinwork/products/fam_tables/qa/fam_processing.md`](../../rubinwork/products/fam_tables/qa/fam_processing.md)

## Question

Do the Danish 1.3 blitz unpaired wavefronts agree with Danish 1.2 on the same FAM and corner-sensor visits, and where do they differ?

## Current answer

Same convention. The DZ fits differ by mean offsets on Z7 Coma_y (+0.1388 µm of wavefront, nMAD 0.0553) and Z6 Astig0 (−0.1116, nMAD 0.0462), attributed to unpairing: a paired fit imposes one solution on two donuts that differ in blur and Z11.

## Log

### 2026-09-22 — backfilled from docs and history

Moved from `aos/notebooks/fam_processing/`. The FAM notebook imports `rubinwork.compare_donuts`. Notebooks marked complete (triage: done). Converted 2026-10-23 in phase 5.
