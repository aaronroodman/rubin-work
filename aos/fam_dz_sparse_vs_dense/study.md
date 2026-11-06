---
study: fam_dz_sparse_vs_dense
topic: aos
status: active
question: Do the per-visit FAM double-Zernike coefficients DZ(k,j) from the Danish 1.3 sparse fit (d13-sparse) agree with the Danish 1.3 dense fit (danish_1_3) on the same visits, for every (k,j) both fit?
answer: ''
uses:
  fam_tables: d13-sparse@20261106, danish_1_3@20261028b
started: 2026-11-06
---

# `fam_dz_sparse_vs_dense`

## Question

Do the per-visit FAM double-Zernike coefficients DZ(k,j) from the Danish 1.3 sparse fit (d13-sparse) agree with the Danish 1.3 dense fit (danish_1_3) on the same visits, for every (k,j) both fit?

## Current answer

None yet.

## Log

### 2026-11-06 — started

Study created.

### 2026-11-06 — first pass, `code/compare_dz.py`

Run `20261106_sparse_vs_dense`: d13-sparse@20261106 against danish_1_3@20261028b, 1041
common visits, none flagged bad_fit in either. Common pupil terms Z4-Z11, Z14, Z15.
k=1..6 fit, sparse minus dense, µm of wavefront OCS:

- Z4 and Z11 agree: Spearman rho 0.87-0.998 (dimensionless) at every k, Huber slope
  0.67-1.41 (dimensionless, sparse over dense).
- Large offsets in the low-order astigmatism and coma: DZ(1,7) median +0.464, DZ(1,6)
  -0.304, DZ(3,6) -0.296, DZ(2,5) +0.300, DZ(1,5) -0.141.
- Elsewhere the Huber slope is mostly 0.05-0.75 and rho 0.05-0.83: the sparse
  coefficients vary less from visit to visit than the dense ones. Z14 is weakest (rho
  0.095 at k=1).

The two builds also differ in danish version (1.3.0 against 1.2.1.dev5), batoid (0.9.0
against 0.8.1.dev26) and possibly pupil model (not in the d13-sparse provenance), so
these differences are not yet attributable to the sparse fit alone.
