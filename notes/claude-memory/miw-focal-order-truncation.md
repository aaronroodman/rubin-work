---
name: miw-focal-order-truncation
description: 83% of MIW power sits above the k<=6 focal orders the build fits — reframes every DZ-subspace analysis
metadata: 
  node_type: memory
  type: project
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-09-04T18:16:49.720Z
---

**The single most important structural fact found 2026-09-03.** The per-visit DZ fit
uses only focal-plane Zernike orders **k=1..6** (`k_min`/`k_max` in
ts_intrinsic_wavefront `pipelines/mi_config.yaml`), because that is all the
optical-state determination needs. But the MIW's structure is overwhelmingly ABOVE
that range, so any subspace built on the 126-coefficient (21 pupil x 6 focal) vector
is blind to most of it.

`aos/code/analyze_miw_dz_full_k.py` -> `output/miw_dz_full_k.pdf` +
`miw_dz_full_k_coeffs.parquet` (w_kj, 21 x 45, in um). Fits each MIW pupil map
Z_j(theta) [um] (OCS) to focal Noll k=1..45 on the 3985 finite cells, fp_radius
1.8 deg. Capture measured as 1 - residual_power/total_power because the focal basis
is NOT orthonormal on the masked grid (summing squared coefficients is wrong).

**Fractions of MIW POWER (dimensionless, power-weighted over the 21 pupil Zernikes):**
| k <= 6 (fitted) | 7..30 (in ts_ofc, unfitted) | k > 30 (outside ts_ofc) | unfit at k<=45 |
|---|---|---|---|
| **0.169** | **0.675** | 0.076 | 0.079 |

**83% of MIW power is at focal orders the fit never touches.** Worst exactly where
the interesting structure is — fraction of that Z_j's power captured by k<=6:
Z5 **0.041**, Z7 **0.051**, Z8 **0.080**, Z6 **0.186**; vs Z11 0.856, Z4 0.629
(the low-order terms the state fit actually needs). MIW RMS [um]: Z5 0.113,
Z6 0.114, Z7 0.118, Z8 0.112, Z11 0.031, Z4 0.039.

**Consequences:**
- Any "reachable vs null" DZ split done on k<=6 (126 coefficients, 50-dim reachable,
  76-dim null) covers only ~17% of the MIW. See [[miw-coadd-retrieval-bias]] — the
  n_null coupling result is real but confined to that corner.
- The right construction is a **k=1..30 refit**: 630 DZ coefficients, 50-dim
  reachable, **580-dim null**, covering 84.4% of MIW power. Needs a build-side
  refit (cheap: per-pupil-j least squares goes from 6 to 30 basis columns).
- Above k=30 ts_ofc tabulates NOTHING (its DZ field axis is length 31 with index 0
  unused, so k=1..30 usable), so no reachable/null split is definable there at all —
  that 7.6% of MIW power can be neither attributed to nor excluded from the DOF.
- k=30 cuts focal radial order n=7 (k=29..36) in half — slightly asymmetric boundary.
- The 7.9% not captured even by k<=45 is probably NOT smooth field structure
  (a 45-term basis over 3985 cells would find it); more likely per-sensor /
  CCD-height-scale effects.

Related but DISTINCT: the *correction* truncation. The build subtracts only the
k<=6 part of the estimated state even though ts_ofc gives all 31 field orders, so a
mode's k>6 response is never removed (0.086-0.179 of power for modes 28-34, ~0 for
1-7). Fixable — recover the DOF and subtract S_full.d; note S_{k<=6}.d == P.w
exactly, so it is purely additive. Details in `aos/MIW_COADD_EQUATIONS.md` §4.3a.
