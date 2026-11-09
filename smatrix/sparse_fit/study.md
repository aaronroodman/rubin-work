---
study: sparse_fit
topic: smatrix
status: active
question: "If the donut fit fits only the primary aberrations and fixes secondary and tertiary at the design intrinsic, is the optical state still constrained, and how does the fixed content leak into the primaries?"
answer: "Piece 1 answered: nearly every DOF drives primary and secondary of a family with field correlation about ±1 (dimensionless). Primary-only observability loses nothing for 22 DOF / 12 v-modes (min 0.99, dimensionless) and degrades only high-order M2 bending modes for 50/34 (min 0.36). Pieces 2 (the leakage matrix) and 3 (the sparse MIW) are open."
uses: {}  # no rubin-work products; ts_ofc v13 sensitivity matrix
started: 2026-09-01
---

# `sparse_fit`: fitting only the primary aberrations

> **Code:** `code/analyze_sparse_fit.py` · **Write-up:** [`../vmode/vmode.md`](../vmode/vmode.md), "Sparse-fit findings" · **Output:** `smatrix/output/vmode/sparse_fit_study.pdf`

## Question

If the donut fit fits only the primary aberrations and fixes secondary and tertiary at the design intrinsic, is the optical state still constrained, and how does the fixed content leak into the primaries?

## Current answer

Piece 1 of 3 (2026-09-01):

- **Primary and secondary are near-degenerate per DOF.** Nearly every DOF, bending modes and hexapod tilts especially, drives the primary and secondary of a family with field correlation about ±1 (Pearson r, dimensionless). Astigmatism first-to-second order from the tilts is −0.44 to −0.95 (cos) and −0.40 to −0.87 (sin), bracketing the MIW's −0.28 and −0.62. A small fit bias can then move power between orders, one route to the MIW radial-order correlations (`optatmo/miw_radial_correlations`).
- **Primary-only observability** (primary-only over full matrix, dimensionless): 22 DOF / 12 v-modes, median 1.00 and min 0.99 at both corner sensors and field-complete sampling. 50 DOF / 34 v-modes, median 0.97 and min 0.36 field-complete, median 0.96 and min 0.44 at the corners; the degraded DOF are M2 bending b10, b11, b14, b15, b19, b20. This is the no-leakage upper bound.

Open: piece 2, the leakage matrix `L[j_prim, j_sec]` (needs the Danish forward model and injected-secondary simulations); piece 3, `A_sparse = A_prim + L @ A_sec` and a rebuilt sparse MIW.

## Log

### 2026-09-01 — backfilled from docs and history

Backfilled 2026-10-18 from [`../vmode/vmode.md`](../vmode/vmode.md), "Sparse-fit findings", and commit history; split out of the old `vmode` study in the phase 5 triage and converted as active. `analyze_sparse_fit.py` moved here from `smatrix/code/vmode/`; run it from `smatrix/` (`python sparse_fit/code/analyze_sparse_fit.py`), output to `smatrix/output/vmode/`.

Retracted on the way: "production zeroes coma2/tref2/astig3/tet2" (that was the non-loaded `new_measured` file), and an early observability run that used the default `zn_idx`.

Next: piece 2, then piece 3.

### 2026-11-06 — plan for pieces 2 and 3

The fit set is Noll 4 to 11, 14, 15, 20, 21, 22 and 27, 28, the `fam_tables` `d13-sparse` set plus Z22, which has a particular M1M3 thermal-gradient sensitivity; `d13-sparse` itself is run alongside for comparison. An orthogonal fit of the optical path difference (OPD) gives no leakage: on the annulus the annular Zernikes are orthogonal, so the sparse rows equal the dense rows. The leakage comes from fitting donut images.

`code/donut_leakage.py` follows ts_wep's Danish setup: noise-free model donuts from the defocused Batoid `zernikeTA` to Noll 78, binned ×2, fit with the sparse set free and every other term at the nominal Batoid model. Per point it gives the dense in-focus rows, the direct sparse sensitivity of all 50 DOF, and `L[j_fit, j_other]` for the unfitted terms Noll 4 to 28. The check is `A_direct = A_dense[fit] + L @ A_dense[other]`, at the 8 corner half-chips and at 9 science points (thx, thy each -0.75, 0, +0.75 deg; intra and extra).

Smoke test at (0.75, 0) deg extra-focal: the leakage pairs each secondary term with the primary term of the same symmetry, Z12/13 to Z6/5 and Z16/17 to Z8/7 at about 4.8 (dimensionless, µm fitted per µm injected), Z23/24 to Z5/6 at about 8. For M2 Rx the dense rows miss the direct result by 7.3% (relative norm); adding `L` brings that to 2.0%.


### 2026-11-09 — first full leakage run (job 44033445, run `20261109_donut_leakage`)

All 26 configurations and 50 DOF finished in 3 min. The nominal fit returns the intrinsic to < 4e-8 µm. Residual |A_direct − model| / |A_direct| (dimensionless, per DOF), median over configurations, fit set `prod_z22`:

| DOF group | model A_dense[fit] | model A_dense[fit] + L @ A_dense[other] |
|---|---|---|
| M2 hexapod | 0.072 | 0.020 |
| camera hexapod | 0.042 | 0.006 |
| M1M3 B1–B10 | 0.49 | 0.12 |
| M1M3 B11–B20 | 1.51 | 0.67 |
| M2 B1–B10 | 0.57 | 0.16 |
| M2 B11–B20 | 0.95 | 0.57 |

`prod` gives nearly the same numbers. The rigid-body DOF close with L; the bending modes do not. The likely cause is that the bending modes change Noll terms above 28, which leak into the fit and which L (injections in Noll 4–28 only) does not cover. Next: extend A_dense and L to Noll 78, the reference model's jmax.

### 2026-11-09 — leakage with dense rows and L to Noll 78 (job 44041372, run `20261109_donut_leakage_j78`)

Same residual as above, median over the 26 configurations, fit set `prod_z22` (`prod` within 0.01):

| DOF group | A_dense[fit] alone | with L, Noll 4–28 | with L, Noll 4–78 | with L, Noll 4–78, 90th percentile |
|---|---|---|---|---|
| M2 hexapod | 0.066 | 0.020 | 0.015 | 0.025 |
| camera hexapod | 0.042 | 0.006 | 0.006 | 0.023 |
| M1M3 B1–B10 | 0.49 | 0.12 | 0.030 | 0.31 |
| M1M3 B11–B20 | 1.51 | 0.67 | 0.18 | 0.44 |
| M2 B1–B10 | 0.56 | 0.16 | 0.054 | 0.64 |
| M2 B11–B20 | 0.95 | 0.57 | 0.19 | 0.58 |

Leakage from Noll 29–78 explains most of the bending-mode mismatch. The tails are not small-denominator cases: the worst are corner half-chips (R40_SW1 M1M3 B15, residual 5.0 against |A_direct| 2.9, µm per µm). Intra-focal configurations are worse than extra-focal (median per configuration 0.07–0.15 against 0.05–0.08). A likely cause is that the model uses in-focus Zernike changes while the donut sees the defocused (TA) wavefront; not yet tested. A_direct itself is the sparse sensitivity at these points and does not depend on the model.
