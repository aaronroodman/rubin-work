---
study: miw_compare_rbr
topic: aos
status: active
question: "How do the MIW and its recovered optical state compare between 50/34 without RBR, 50/50 without RBR and 50/50 with Range-Bounded Recovery (RBR), and which keeps every DOF inside its allowed range?"
answer: "On Danish 1.3, RBR cuts the fraction of (visit, DOF) pairs out of range from 0.4625 to 0.0800 (dimensionless) but grows the MIW RMS from 0.2594 to 0.3148 um of wavefront."
uses:
  miw: d13-A_50_34_i@20261025, d13-A_50_50_i_rbr@20261027
started: 2026-09-28
---

# `miw_compare_rbr`

> **Code:** `code/` (`check_dof_ranges.py`, `compare_rbr_arms.py`, `compare_schemes.py`, `compare_build_dof.py`, `test_rbr_against_prototype.py`) · **Write-up:** [`../docs/studies/miw.md`](../docs/studies/miw.md) sections on reachability and RBR · **Depends on:** unpushed `ts_ofc` and `ts_intrinsic_wavefront` branches

## Question

How do the MIW and its recovered optical state compare between 50/34 without RBR, 50/50 without RBR and 50/50 with Range-Bounded Recovery (RBR), and which keeps every DOF inside its allowed range?

## Current answer

On Danish 1.3 (`danish_1_3` FAM tables, 9 rotator bins, 301 visits):

- 50/34 without RBR puts 17 of 20 M1M3 and 16 of 20 M2 bending modes out of range on at
  least 5 % of visits. 50/50 with RBR cuts that to 10 and 5, and the fraction of
  (visit, DOF) pairs out of range from 0.4625 to 0.0800 (dimensionless).
- RBR grows the MIW: on `rot_-3_3`, RMS over the field 0.2594 → 0.3148 µm of wavefront,
  inferred FWHM 0.1748 → 0.2097 arcsec.
- The 50/50 arm without RBR is not built yet.

## Log

### 2026-10-08 — backfilled from docs and history

Split from aos `miw` (triage row `miw_compare_rbr`). The four scripts moved from `aos/code/miw/`; the MIW build itself is the `miw` product and `ts_intrinsic_wavefront`. Converted 2026-10-23 in phase 5.

### 2026-10-27 — Danish 1.3 test and v1000 MIW builds removed

The `miw` builds this study used, `d13t-A_50_34_i@old-tree-20260928`, `d13v1000-A_50_34_i@old-tree-20261006`,
`d13v1000-A_50_34_i_rbr@old-tree-20261006`, `d13v1000-A_22_12_i@old-tree-20261007` and
`d13v1000-A_50_50_i_rbr@old-tree-20261007`, were deleted on 2026-10-27 in the phase 7
cleanup, and dropped from `uses:`, with their variants dropped from `rubinwork/products/miw/variants.yaml`.
The originals are in `/sdf/data/rubin/user/roodman/LSST/backup/pre_reorg_outputs_20261024/aos_output.tar.zst`,
under `aos/output/miw/`. The results above stand; rerunning the scripts needs a new build.
The 50/50 RBR arm is redefined on the `danish_1_3` FAM tables as `d13-A_50_50_i_rbr` and
`d13-A_50_50_i_rbr_5rot`, same settings, not yet built.

### 2026-10-27 — renamed; Danish 1.3 50/34 against 50/50 RBR

Renamed from `miw_recovery_constraint`; the question now names the three arms: 50/34
without RBR, 50/50 without RBR and 50/50 with RBR. Built `d13-A_50_50_i_rbr@20261027`
and `_5rot` (job 42123807, `n_iter` 3, RBR kappa 4.0, power 3), set current. Ran
`check_dof_ranges.py` and `compare_rbr_arms.py` against `d13-A_50_34_i@20261025`, all 9
rotator bins, 301 visits common to both. Output `output/20261027_d13_rbr_arms/`.

- DOF flagged (at least 5 % of visits outside r_j): 50/34 has 17 of 20 M1M3 and 16 of
  20 M2 bending modes; RBR has 10 of 20 and 5 of 20. Neither flags a hexapod DOF.
- Fraction of (visit, DOF) pairs outside range: 50/34 0.4625, RBR 0.0800
  (dimensionless). Worst |d_j|/r_j: 66.24 against 1.667 (dimensionless).
- Over-range wavefront RMS over full-state wavefront RMS (amplitude, median over
  visits): 0.548 against 0.065 (dimensionless).
- Fit residual ||dW − S d||, median RMS over the (k, j) grid: 0.00621 against 0.00838 µm
  of wavefront.
- MIW on `rot_-3_3`: RMS over the field 0.2594 µm (50/34) against 0.3148 µm (RBR) of
  wavefront, difference RMS 0.2116 µm. Inferred FWHM, median over field: 0.1748 against
  0.2097 arcsec. Noll 6 and 5 carry 0.23 and 0.19 of the difference power.
- Convergence: the RBR bins move by 3.0e-03 to 4.1e-03 µm RMS of wavefront at the last
  of 3 iterations. Aaron takes about 5e-3 µm as converged; `n_iter` stays at 3 because
  more iterations did not converge before.

Next:
- a 50/50 build without RBR, the third arm (new variant and batch job);
- `compare_schemes.py` over the three arms;
- Aaron to review `rbr_arms/rbr_arms_plots.pdf` and iterate on the plots.
