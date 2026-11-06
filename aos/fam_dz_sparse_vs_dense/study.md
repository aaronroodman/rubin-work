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

### 2026-11-06 — sparse term against its dense family

Added pages comparing each sparse DZ(k,j) with dense terms of the same m and parity
(`FAMILIES` in `code/compare_dz.py`: 5/13/23, 6/12/24, 7/17, 8/16, 9/19, 10/18, 11/22,
14/26, 15/25), and a Huber regression of sparse on the family. k=1..6 fit, 1041 visits,
µm of wavefront OCS.

- The unit-weight sum helps but does not close the gap: DZ(1,7) median difference +0.464
  -> +0.363, nMAD 0.185 -> 0.148.
- Free weights do. At k=1 the regression gives about 1.0 on the fitted term and 2.4-4.9
  on the omitted term (Z7 1.025, Z17 4.864; dimensionless, sparse over dense term), and
  cuts the nMAD of the residual by factors 2-15 (DZ(1,7) 0.185 -> 0.012, DZ(1,5) 0.158
  -> 0.015). Z11/Z22 is the exception: weight on Z22 0.34.
- A least-squares projection of the omitted term onto the fitted one over the annular
  pupil (eps 0.612) gives 0 for annular Zernikes and 0.03-0.67 for circular ones, so the
  weights of 2-5 are not a pupil projection; they come from the image fit.
- Dense Z_j and its partner are anticorrelated over visits (Pearson r -0.23 to -0.93),
  which is why the unit sum partly cancels.

### 2026-11-06 — optimal linear combination of the dense family

Each family page now has a bottom row: sparse DZ(k,j) against the Huber multi-regression
combination offset + sum c_t Z_t of its dense family, with the coefficients and formal
errors. A new page plots the coefficients against k for every family. Coefficients are
dimensionless (sparse over dense term); `dz_family_fit.csv` has them per (j, k).
k=1..6 fit, 1041 visits, µm of wavefront OCS.

- The weight on the fitted term is 0.5-1.1 and falls with k. At k=1 it is 0.90-1.03 for
  every family.
- The weight on the omitted second term is 2-5 for most (j, k): Z7/Z17 at k=1 is
  1.025 ± 0.003 on Z7 and 4.864 ± 0.013 on Z17; Z5, Z6 and Z8 are 4.1-4.8 at k=1.
  Exceptions: Z11/Z22 (-0.27 to 0.65), Z7/Z17 at k=3 (0.80), Z8/Z16 at k=2 (0.40),
  Z6/Z12 at k=6 (-0.13), Z5/Z13 at k=5 (1.2). Those are also the cases with the lowest
  rho.
- Third terms (Z23, Z24) take weights of about 3-4, except Z5 k=5 (-2.7) and Z6 k=6
  (-5.1).
- With the fitted combination, Spearman rho (sparse against combination,
  dimensionless) is 0.99-0.998 at k=1 for Z5-Z11, and the residual nMAD at k=1 is
  0.001-0.030 µm, against 0.020-0.148 µm for the unit-weight sum.
