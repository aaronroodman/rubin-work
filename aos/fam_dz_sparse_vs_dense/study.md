---
study: fam_dz_sparse_vs_dense
topic: aos
status: active
question: Do the per-visit FAM double-Zernike coefficients DZ(k,j) from the Danish 1.3 sparse fit (d13-sparse) agree with the Danish 1.3 dense fit (danish_1_3) on the same visits, for every (k,j) both fit?
answer: ''
uses:
  fam_tables: d13-sparse@20261106, danish_1_3@20261028b
  miw: d13-A_50_50_i_rbr@20261027
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

### 2026-11-06 — is dense DZ(1,17) a physical optical state?

`code/dz17_correctable.py`, on the 301 visits (5 nights, T614, 9 rotator bins) of
`miw` d13-A_50_50_i_rbr@20261027, built from the danish_1_3 dense fits: per-visit DZ with
the build's 50-DoF state (S·DoF) and residual, and the sensitivity matrix S the build
used. µm of wavefront, OCS, k=1.

- Yes, mostly. With DoF recovered from every DZ term except Z17 (truncated SVD of the
  normalized S, 30 modes), the predicted DZ(1,17) follows the measured one: Pearson
  r 0.873, Spearman rho 0.812 (dimensionless, n=301); mean predicted +0.103 against
  measured +0.124 µm; std over visits 0.0383 µm measured, 0.0189 µm residual, so about
  75% of the variance (residual over measured, squared). 20 modes gives the same; 50
  modes is worse (r 0.632), from the near-degenerate modes.
- Measured DZ(1,17) correlates with DZ(1,7) (r -0.827), DZ(4,7) (r +0.758), DZ(2,5)
  (r -0.713) and DZ(3,6) (r +0.679), the pattern of the state the other terms give.
- The build's full 50-DoF state reproduces 95% of the DZ(1,17) variance over visits
  (1 - var(resid)/var(raw), dimensionless).
- What makes it: M2 decenter with compensating M2 tilt. Per-visit std of the S·DoF
  contribution to DZ(1,17): M2_dy 0.051, M2_rx 0.011, Cam_dy 0.005 µm; bending modes
  under 0.004 µm. Mean contributions: M2_dy +0.143, M2_rx -0.029 µm. For DZ(1,7) the
  same two cancel (-2.06 and +1.95 µm): this is the coma-free M2 decenter-plus-tilt
  combination, and DZ(1,17) is the term that breaks the degeneracy. Recovered M2_dy is
  -3245 ± 1154 µm (mean ± std over visits).
- A pure DZ(1,17), with nothing else, is not reachable: by least squares it needs B1_16
  at 250 times its range.
- So the sparse fit folding 4.9 x DZ(1,17) into DZ(1,7) mixes a real secondary-coma
  signal of coma-free M2 motion into coma. With dense DZ(1,17)/DZ(1,7) Huber slope
  -0.162 (dimensionless) over these visits, sparse DZ(1,7) would respond to this state
  with 1.025 - 4.86 x 0.162 = 0.24 of the dense coma (dimensionless). Inferred, not
  measured directly.

### 2026-11-06 — DZ(1,17) state coverage beyond the MIW sample

`code/dz17_correctable.py` now also predicts DZ(1,17) from the DoF fitted without Z17
(30 modes) on every good visit of the `miw` d13-A_50_50_i_rbr@20261027 `fits` table: the
danish_1_3 dense DZ refitted with this MIW subtracted, 1821 visits on 45 nights. That
table agrees with the build's own per-visit DZ(1,17) to Pearson r 0.992 (dimensionless,
n=301). µm of wavefront, OCS, k=1. Statistics are measured against predicted.

The `_5rot` variants drop rotator bins, not nights. In this sample the four dropped bins
(-50..-40, -35..-25, 25..35, 40..50 deg) are exactly the night 20260409. The five kept
bins are the other four nights.

| group | visits | std measured | std residual | Pearson r | Spearman rho |
|---|---|---|---|---|---|
| MIW sample, 5rot bins (20260315-0324) | 205 | 0.031 | 0.020 | 0.826 | 0.804 |
| MIW sample, other 4 bins (20260409) | 96 | 0.013 | 0.018 | 0.181 | 0.218 |
| T614 i, outside the MIW sample (8 nights) | 605 | 0.031 | 0.034 | 0.532 | 0.356 |
| other programs or bands (2025-2026) | 915 | 0.084 | 0.034 | 0.920 | 0.934 |

- The residual has a floor of about 0.015-0.035 µm (std over visits) in every group and
  on most nights.
- Where the state moves DZ(1,17) by more than that floor, the prediction holds. Over the
  other programs it gives Pearson r 0.92: T606 r band 0.93 (n=158), the M1M3 bending
  tests T378_M1M3B12 and B20 in i 0.95 and 0.98, and T598 r band 0.95.
- 20260409 spans only 0.013 µm, under the floor, so it gives no test. Its mean agrees:
  predicted +0.135 µm, measured +0.158 µm.
- T614 i outside the MIW sample is mixed. Pearson r is 0.78 on 20260327 (n=121). It is
  about 0 or negative on 20260325 and 20260331. On 20260331 the measured value is 0.060 µm
  below the prediction.
- The median of measured minus predicted is +0.02 to +0.03 µm in three of the four
  groups. So part of the mean DZ(1,17) is not explained by the state.

### 2026-11-06 — sparse residual median closer to Batoid (Aaron's observation)

Aaron's observation: after the k=1..6 fit, the median over FAM visits of the Z5-Z8
residual wavefront across the focal plane agrees much better with the Batoid intrinsic
for d13-sparse than for dense danish_1_3. That suggests sparse may give a more physical
MIW, and he will pursue it. This is not yet quantified here; the dense trio PDF comes from
job 43538638.

This does not conflict with the DZ(1,17) result above. That result is about the
per-visit variation, which follows the optical state. The median is the intrinsic part.
The +0.02 to +0.03 µm median of DZ(1,17) that the state does not explain is the kind of
term the two modes would put into the MIW differently.
