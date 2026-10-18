---
study: zernike_opd_vs_consdb
topic: olr
status: done
question: "Are the ConsDB corner Zernikes the total OPD, the deviation or the intrinsic, and are they identical to the Butler ones?"
answer: "They are the total OPD: ConsDB and Butler zk_opd agree to a median 0.16 nm of wavefront over 2,651 (seq, corner) pairs on day_obs 20260420. They differ only in non-axisymmetric terms, by a pupil rotation of about 0.1 deg."
uses: {}  # old-tree olr/output/20260420/nightly_aos_table.parquet and blocks/output/t539 table
started: 2026-07-08
---

# `zernike_opd_vs_consdb`: ConsDB against Butler corner Zernikes

> **Write-up:** [`zernike_opd_vs_consdb.md`](zernike_opd_vs_consdb.md)

## Question

Are the ConsDB corner Zernikes the total OPD, the deviation or the intrinsic, and are they identical to the Butler ones?

## Current answer

They are the total OPD: ConsDB and Butler zk_opd agree to a median 0.16 nm of wavefront over 2,651 (seq, corner) pairs on day_obs 20260420. They differ only in non-axisymmetric terms, by a pupil rotation of about 0.1 deg.

## Log

### 2026-07-08 — backfilled from docs and history

Backfilled 2026-10-18 from [`zernike_opd_vs_consdb.md`](zernike_opd_vs_consdb.md) (formerly `olr/docs/`). Done in the phase 5 triage. Mean |Δ| against Butler OPD 0.0005 µm (max 0.027 µm); against Butler deviation 0.11 µm, so the ConsDB values are the OPD. Z4, Z11 and Z22 agree exactly; the residual sits in m ≠ 0 terms, consistent with a pupil rotation of about 0.1 deg.
