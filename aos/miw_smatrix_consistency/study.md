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
- v13 5rot (d13-A_50_34_i_5rot@20261025, split Z4 only): Z4 changes by 0.0017 µm RMS
  (OCS) and 0.0016 µm (CCS), at most 0.031 µm, against maps of 0.030 and 0.029 µm.
  Negligible. Z5, which is OCS-only there, changes by 0.014 µm RMS, at most 0.24 µm
  (map 0.110 µm), at a few cells.

### 2026-11-16 — cause of the out-of-family Z5 region in sparse rot_-50_-40

The region at thx −0.5°, thy −1.33° (OCS) is one grid cell holding one donut, Z5 438 µm
against −0.07 to +0.01 µm in the neighbouring cells. The donut is R30_S11, 20260409 seq
491. This unpaired processing reports `zk_OCS` = (intra + extra)/2, checked exactly. For
this donut Z5 is 0.33 µm intra and 876 µm extra: the extra Danish fit diverged. The dense
`danish_1_3` table has the same failure on the same donut (Z5 extra −125.7 µm).

In the 24 visits of this bin (81776 donuts, count), 91 donuts have any |Zk| > 5 µm and 25
have > 20 µm, up to 4845 µm (sparse) and 7093 µm (dense). All are `used = True`. The
quality cut in `fam_tables` is per visit; nothing rejects single donuts.
`lstsq_status` is −1 for every donut in both tables, so it carries no information.
Per-side `lstsq_optimality` and blur (several bad sides at the 5.0 arcsec bound) do not
separate the bad fits cleanly.

Intra and extra agreement does. The cut is any Noll with
|(intra − extra) − median| > 15 nMAD, where nMAD is the robust scatter of
(intra − extra) per Noll over the bin. It flags 0.59% (sparse) and 0.83% (dense) of the
donuts. It catches every donut with |Zk| > 5 µm, and the largest |Zk| left is 1.93 µm
(sparse) and 2.04 µm (dense).

### 2026-11-16 — MIWs rebuilt on the pair-cut fam_tables builds

The pair cut drops a donut when any Noll has |(intra − extra) − median| > 15 nMAD,
per visit (`refit_build --pair-cut-nmad 15`, rubin-work 2ce3a02). New builds:
`fam_tables` d13-sparse@20261116 (0.68% of donuts dropped) and danish_1_3@20261116
(1.11%). The visit quality cut is unchanged: 1130 and 1988 visits pass. `miw`
d13s-A_50_34_i_fsm_sparse_no22@20261116b and d13-A_50_34_i_fsm_dense21@20261116b are
built on them. Run `20261116_paircut`, all cells kept (`--min-donuts 1`).

The largest |bin − 5rot median| over all bins, cells and j drops from 1409 to 0.75 µm
(sparse) and from 63 to 0.95 µm (dense21). The Z5 cell at (−0.49°, −1.33°) in
rot_-50_-40 is empty after the cut: its one donut was dropped.

| µm | v13 | dense21 | dense21 cut | sparse | sparse cut |
|---|---|---|---|---|---|
| split residual RMS, Z5 | 0.268 | 0.268 | 0.052 | 1.830 | 0.043 |
| split residual RMS, Z10 | 0.226 | 0.226 | 0.020 | 5.878 | 0.012 |
| 9-bin scatter, Z5 | 0.374 | 0.374 | 0.063 | 2.572 | 0.052 |
| 9-bin scatter, Z8 | 0.152 | 0.153 | 0.068 | 1.759 | 0.019 |
| 9-bin scatter, Z14 | 0.351 | 0.351 | 0.020 | 7.553 | 0.015 |
| MIW − Batoid, Z5 | 0.166 | 0.176 | 0.135 | 0.776 | 0.064 |
| MIW − Batoid, Z8 | 0.088 | 0.100 | 0.087 | 0.529 | 0.023 |

- The 5rot scatter does not change: the bad donuts were in 20260409.
- After the cut, sparse is still the most consistent MIW and the closest to Batoid on
  Z5–Z8, Z11. Z4 against Batoid is worse for sparse (0.072 against 0.019 µm).
- Large DoF are unchanged: max p95 |DoF|/range 36.6 (dense21) and 51.2 (sparse).
- The v13 nominal MIW has the same bad donuts and needs the same rebuild.
- Night 20250519 loses 27–37% of donuts in four visits (dense). That night is not in
  the MIW sample; not investigated.

### 2026-11-16 — bin-mean MIW maps beside Batoid

`code/mean_miw_maps.py`, run `20261116_mean_miw_maps`: per Zernike, the mean over all 9
bins, the 5rot bins and the other 4 bins for both @20261116b MIWs, beside Batoid, on the
cells present in all 9 bins (3224 cells). RMS over field of (mean − Batoid), Z5: sparse
0.053 / 0.072 / 0.055 µm, dense21 0.133 / 0.145 / 0.136 µm (all 9 / 5rot / other 4).
Dense21 Z5 has out-of-family values at the field corners in all three means.

### 2026-11-16 — 50/50 RBR on the fitted S-matrices

New variants `d13-A_50_50_i_rbr_fsm_dense21@20261116` and
`d13s-A_50_50_i_rbr_fsm_sparse_no22@20261116`, on the pair-cut fam_tables builds
@20261116 (4f100d0). Runs `20261116_rbr` (compare_miw, `--min-donuts 1`) and
`20261116_mean_miw_maps_rbr`, with the two 50/34 builds @20261116b.

- RBR bounds the DoF: max over DoF of p95 |DoF|/range (dimensionless) 36.6 → 1.42
  (dense21), 51.2 → 1.08 (sparse). DoF with p95 over range: 31 → 15 (dense21, the
  bending modes B1/B2), 34 → 2 (sparse).
- 9-bin scatter and split residual are about the same as 50/34, slightly worse.
- MIW − Batoid RMS over field, Z5: sparse 0.064 → 0.137 µm, dense21 0.135 → 0.178 µm;
  Z8: sparse 0.023 → 0.050 µm, dense21 0.087 → 0.049 µm. Z4 improves for sparse
  (0.072 → 0.038 µm).
