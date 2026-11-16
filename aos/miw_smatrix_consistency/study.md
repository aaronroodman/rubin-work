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

### 2026-11-16 — first comparison of the three MIWs

Built `miw` d13s-A_50_34_i_fsm_sparse_no22@20261116 and d13-A_50_34_i_fsm_dense21@20261116
(jobs 45347848 and 45347849). Their manifests record `fitted_smatrix`, and the bin logs
show the fitted OFC config. The sparse split needed its own `noll_list` and Z27, Z28 in
the split's Noll table (rubin-work 8eb3764, ts_intrinsic_wavefront 181a51e). The first
job hung because `sbatch` exported an older stack; submit with `--export=NONE`.

A few grid cells hold 1 or 2 donuts (median 27 donuts per cell, count). They are off by up
to 1409 µm (sparse) and 54 µm (dense, v13 and dense21 alike) and dominate any RMS over
cells. `compare_miw.py` now drops cells with fewer than 5 donuts. The split step has no
such cut, so the product's split residual RMS for sparse (Z10 5.88 µm, Z14 5.37 µm) is
driven by them; dense21 and v13 are unchanged (Z10 0.226 µm).

Bin-to-bin scatter (RMS over cells of the std over bins, µm, OCS), cells >= 5 donuts:

| Noll j | v13 9 bins | dense21 9 bins | sparse 9 bins | v13 5rot | dense21 5rot | sparse 5rot |
|---|---|---|---|---|---|---|
| 5 | 0.062 | 0.063 | 0.052 | 0.042 | 0.044 | 0.038 |
| 6 | 0.063 | 0.066 | 0.045 | 0.041 | 0.042 | 0.039 |
| 7 | 0.049 | 0.045 | 0.015 | 0.026 | 0.024 | 0.011 |
| 8 | 0.066 | 0.069 | 0.019 | 0.033 | 0.034 | 0.013 |
| 11 | 0.018 | 0.018 | 0.007 | 0.011 | 0.012 | 0.005 |

- Sparse is the most consistent over the 9 bins: coma scatter is a third of the dense
  value. Dense21 is no better than v13.
- Bin-mean MIW minus Batoid, RMS over field (µm): Z5 0.123 / 0.136 / 0.064, Z7 0.082 /
  0.078 / 0.033, Z8 0.071 / 0.085 / 0.023 (v13 / dense21 / sparse). Sparse is closer
  to Batoid for Z5-Z8, as seen in the FAM fits; Z4 is worse (0.026 / 0.019 / 0.072).
- Per-visit DoF are large in all three: p95 over visits of |DoF|/range up to 59 (v13),
  37 (dense21), 51 (sparse) (dimensionless), on M1M3 bending modes B1_12, B1_16, B1_19,
  B1_20; 31-34 of 50 DoF have p95 above 1. The S-matrix choice does not change this,
  so the 50/50 RBR build is worth checking (Aaron's criterion).

Next: a `min_donuts` cell cut in the split step (product change, then rebuild sparse),
and a 50/50 RBR sparse variant.

### 2026-11-16 — do low-donut cells bias the OCS/CCS split?

Yes. I reran the split with grid cells under 5 donuts removed (28-127 of about 3850 cells
per bin, count), into /lscratch/roodman/split_test*. RMS over the split-map points of
(all cells - cut), µm:

- v13 nominal (d13-A_50_34_i@20261025): Z4 OCS 0.009 (map 0.031), Z5 OCS 0.053 (map
  0.089), Z5 CCS 0.044 (map 0.028), Z14 CCS 0.054 (map 0.007). The split residual RMS
  drops from 0.268 to 0.051 (Z5) and from 0.249 to 0.014 (Z14).
- sparse: the change is larger than the maps for most j, for example Z10 OCS 1.13 (map
  0.055) and Z4 OCS 0.062 (map 0.087). The residual RMS drops from 5.88 to 0.012 (Z10).

The 9-bin variants split every j; `split_js: [4]` is set only in the `_5rot` variants,
which this test did not cover.
