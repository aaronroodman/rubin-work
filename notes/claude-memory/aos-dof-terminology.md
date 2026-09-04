---
name: aos-dof-terminology
description: "Aaron's terminology for AOS DOF quantities (optical_state, Tweak, Trim) and how they relate"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9325d2f9-106f-4f9c-9a8a-9af33e47c0f5
---

Aaron's terminology for the AOS degree-of-freedom quantities (his informal names, not EFD/ConsDB/ts_ofc terms):

- **optical_state** — the DoF obtained by running the measured Zernike *deviations* (Zⱼ, OPD − intrinsic) through the sensitivity-matrix SVD → v-modes → DoF, for a given (NDoF, n_keep) choice. The current best estimate of the optical state from the wavefront measurement. Has no standard name.
- **Tweak** = `PID(optical_state)` — the per-iteration correction the AOS controller emits.
- **Trim** — the accumulated offset from the LUT: `Trim_(i+1) = Trim_i + Tweak`. This is what the EFD `lsst.sal.MTAOS.logevent_degreeOfFreedom` `aggregatedDoF0..49` reports.

Implication for reconstructions: the "controllable wavefront" (`zk_constrained` in olr/code/nightly_table.py) should be reconstructed from the **optical_state**, NOT the Trim. The OLR nightly_table currently (mistakenly) feeds it the Trim (aggregatedDoF).

Related: [[aos-vmode-normalization]] (geom weights, StateEstimator vs build_ofc_svd, 4-CWFS vs double-Zernike).
