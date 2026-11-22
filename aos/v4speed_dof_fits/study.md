---
study: v4speed_dof_fits
topic: aos
status: active
question: With one of the 22 DOF set at +2 steps of the smatrix/sparse_fit scan (about 0.04 um RMS in-focus change), do Danish fits against the nominal v4_speed intrinsicTA recover the Zernike deviation (perturbed OPD minus intrinsic) and still fit the donuts well, for FAM, CWFS and giant donuts, with the sparse (spider, bin 2) and dense production (no spider, bin 2) fits?
answer: At +2 sparse_fit steps (about 0.34 µm of expected Zernike deviation) sparse fits recover rigid body to 0.01-0.08 µm but only 27% of M1M3 B6 and B7 (error 0.25 µm, FAM), from leakage of unfitted terms (Z12, Z13, Z16); dense production fits recover the change to 0.007-0.03 µm except B6, B7 (0.12 µm), on top of a constant 0.2-0.8 µm no-spider bias.
uses:
  batoid_intrinsic: v4_speed@20261121b
  sim_donut_fits: v4_speed_dense_prod@20261122, v4_speed_dof_p2_dense_prod@20261122, v4_speed_dof_p2_sparse@20261122, v4_speed_sparse@20261121
  sim_donuts: v4_speed_mono_dof_p2@20261122
started: 2026-11-22
---

# `v4speed_dof_fits`: v4_speed: Danish fits of single-DOF perturbed donuts

## Question

With one of the 22 DOF set at +2 steps of the smatrix/sparse_fit scan (about 0.04 um RMS in-focus change), do Danish fits against the nominal v4_speed intrinsicTA recover the Zernike deviation (perturbed OPD minus intrinsic) and still fit the donuts well, for FAM, CWFS and giant donuts, with the sparse (spider, bin 2) and dense production (no spider, bin 2) fits?

## Current answer

At +2 sparse_fit steps (about 0.34 µm of expected Zernike deviation, quadrature sum over
the fitted terms), sparse fits (spider, bin 2) recover rigid-body changes to 0.01-0.08 µm
but only 27% of M1M3 B6 and B7 (error 0.25 µm on FAM): unfitted terms (Z12, Z13, Z16)
leak into the fitted ones. Dense production fits (no spider, bin 2) recover the change to
0.007-0.03 µm except B6 and B7 (0.12 µm), on top of a constant 0.2-0.8 µm bias from the
missing spider.

## Log

### 2026-11-22 — pass 1: 22 DOF at +2 steps, sparse and dense production fits

First pass (`dof_fits.py`, `5f68faa`; run `20261122_dof_fits`). Products: `sim_donuts
v4_speed_mono_dof_p2@20261122` (22 DOF, each alone at +2 steps of the sparse_fit scan,
about 0.04 µm RMS in-focus Noll 4-78 change, 998 donuts per DOF; code `dd904cf`) and its
fits `sim_donut_fits v4_speed_dof_p2_sparse` and `v4_speed_dof_p2_dense_prod`, both
against the nominal intrinsicTA. Jobs: sims array 46442649, fits arrays 46442672 and
46442673, all completed.

Per donut, quadrature sums over the fitted terms (µm of wavefront): expected = perturbed
OPD - nominal intrinsic, about 0.34 µm for most DOF (0.6-1.1 µm for Cam Rx, Ry);
"error of the change" = (fit - unperturbed fit at the same point) - expected. Medians
within 1.725 deg; gain = fitted change projected on the expected over |expected|²
(dimensionless):

| DOF | sparse FAM: error of change (µm) / gain | dense_prod FAM: error of change (µm) / gain |
|---|---|---|
| M2 dz, Cam dz | 0.009 / 1.00 | 0.011 / 1.00 |
| M2 dx, dy | 0.077 / 0.94 | 0.008 / 1.01 |
| Cam dx, dy | 0.042 / 1.11 | 0.009 / 1.00 |
| M1M3 B1, B2 | 0.060 / 0.83 | 0.010 / 0.97 |
| M1M3 B4, B5 | 0.091 / 0.74 | 0.023 / 0.94 |
| M1M3 B6, B7 | 0.25 / 0.27 | 0.12 / 0.66 |
| M2 B3, B4 | 0.11 / 0.69 | 0.033 / 0.91 |

CWFS and giant follow FAM closely (sparse M1M3 B6, B7 gain: CWFS 0.29, giant 0.44).

- **Sparse fits lose signal to leakage from the unfitted terms.** For M1M3 B6 at FAM
  r05_a04 extra the expected z8 is +0.30 µm and the fit gives +0.08 µm. The expected is
  right: the defocused TA change equals the in-focus OPD change to 0.005 µm. With the
  full perturbed intrinsicTA as reference the fit is clean (l1 0.0031, deviations
  <= 0.003 µm); with the nominal reference plus the true fitted terms it moves away
  (z8 -0.22 µm, l1 0.0064). The unfitted change is 0.064 µm (Z16 -0.050, Z12 -0.028,
  Z13 -0.023 µm) and it pulls the fitted terms by about 4 times its size. This is the
  sparse_fit study's leakage, now in ray-traced v4_speed donuts.
- **Fit quality drops where leakage is large:** sparse FAM l1 0.0030 unperturbed,
  0.0049-0.0065 for M1M3 B4-B7 and M2 B3, B4; the residual is the unfitted wavefront.
  Dense_prod l1 stays at its no-spider level, 0.015.
- **Dense production fits** carry a large constant bias (0.22 µm FAM, 0.33 µm CWFS,
  0.78 µm giant, quadrature sum) from the missing spider, but it cancels in the change:
  error of the change 0.007-0.03 µm except M1M3 B6, B7 (0.12 µm, gain 0.66).

[Sparse medians per DOF](figures/sparse_medians_per_dof.png). Per-DOF examples:
`dof_examples_<variant>.pdf`, one page per DOF.

Next:
- leakage per Noll term (which unfitted terms drive which fitted ones), against the
  sparse_fit L matrix;
- larger amplitudes (the v13 DOF range) and combined DOF states.

### 2026-11-22 — started

Study created.
