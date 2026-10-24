---
study: miw_radial_correlations
topic: optatmo
status: done
question: "Do the MIW terms of the same azimuthal family but different radial order correlate across the field, as a leakage between radial orders would make them?"
answer: "Yes: MIW radial orders of one family correlate across the field (Z5/6 with Z12/13, Z7/8 with Z16/17). This feeds the smatrix sparse_fit study."
uses: {}  # reads the danish_1_2_A_50_34_i_5rot MIW from the old aos/output tree
started: 2026-09-01
---

# `miw_radial_correlations`: cross-correlation of MIW radial orders

> **Code:** `code/plot_miw_radial_correlations.py` (run from the repo root; writes `output/miw_radial_correlations.pdf`)

## Question

Do the MIW terms of the same azimuthal family but different radial order correlate across the field, as a leakage between radial orders would make them?

## Current answer

Yes: MIW radial orders of one family correlate across the field (Z5/6 with Z12/13, Z7/8 with Z16/17). This feeds the smatrix sparse_fit study.

## Log

### 2026-10-24 — input moved to the miw product

`plot_miw_radial_correlations.py` reads `intrinsic_split_maps.parquet` from the `miw`
product, variant `d12-A_50_34_i_5rot`, instead of
`aos/output/miw/danish_1_2_A_50_34_i_5rot/`. The product's Z7_OCS map differs from the old
one by 0.0316 µm RMS of wavefront, against a map RMS of 0.118 µm. Recorded results were not
rerun; Aaron accepted the change (2026-10-24) on his 1 deg rotator-angle tolerance. A check
run to `/tmp` completes (5 pages).

### 2026-09-01 — backfilled from docs and history

Backfilled 2026-10-18 from the script docstring, commits `0ebf23c` and `a047ce5`, and the phase 5 triage (done; feeds `sparse_fit`). The script scatters the cos and sin components of each family at the native MIW field bins, with the Pearson r and slope per panel. Pearson r values were not recorded in a doc. It moved here from `optatmo/code/`; its default MIW path still reads `aos/output/miw/...` by hand (allow-listed under `optatmo/`).
