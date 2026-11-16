---
study: miw_smatrix_consistency
topic: aos
status: active
question: Does a Danish-fitted sensitivity matrix (sparse_no22 for the sparse fit, dense21 for the dense fit) make the nine per-rotator-bin MIWs agree better than the v13 matrix?
answer: ''
uses:
  fam_tables: d13-sparse@20261106, danish_1_3@20261028b
  fitted_smatrix: dense21@20261114, sparse_no22@20261114
  miw: d13-A_50_34_i@20261025
started: 2026-11-16
---

# `miw_smatrix_consistency`

## Question

Does a Danish-fitted sensitivity matrix (sparse_no22 for the sparse fit, dense21 for the dense fit) make the nine per-rotator-bin MIWs agree better than the v13 matrix?

## Current answer

None yet.

## Log

### 2026-11-16 — started

Study created.

### 2026-11-16 — plan and the two MIW variants

Two new `miw` variants, nominal 50 DoF / 34 v-modes over the nine rotator bins (T614
triplets, i band, alt 65-75 deg, 301 visits on 5 nights; the same visits in both FAM
variants):

- `d13s-A_50_34_i_fsm_sparse_no22`: fam_tables d13-sparse (Noll 4-11, 14, 15, 20, 21,
  27, 28; no Z22, checked in `nollIndices`) on `fitted_smatrix` sparse_no22@20261114.
- `d13-A_50_34_i_fsm_dense21`: fam_tables danish_1_3 (Noll 4-19, 22-26) on
  `fitted_smatrix` dense21@20261114.

Reference: `miw` d13-A_50_34_i@20261025, the same scheme on the v13 matrix. No sparse
build on the v13 matrix: v13 is not appropriate for the sparse fit (Aaron). The
normalization weights stay at v13, so only the matrix changes.

The variant's `fitted_smatrix:` field selects the matrix. The Snakefile passes that
build's OFC config directory to `run_build_intrinsic.py --ofc-config-dir`
(ts_intrinsic_wavefront 787c778). Without the field the SVD is unchanged, checked
equal.

Metrics, once built:
- the per-Zernike RMS over the focal plane of each bin's grid minus the OCS/CCS split
  model, over the nine bins;
- the same without 20260409 (the ±30 and ±45 deg bins);
- Z5-Z8 against the Batoid intrinsic;
- the recovered DoF amplitudes against their ranges. Large ones mean the 50/50 RBR
  scheme is worth a build.
