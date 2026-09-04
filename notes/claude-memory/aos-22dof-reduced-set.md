---
name: aos-22dof-reduced-set
description: "Definition of the AOS \"22 DOF\" reduced degree-of-freedom set used in v-Mode fits"
metadata: 
  node_type: memory
  type: reference
  originSessionId: cc746a2f-9746-41cd-90db-f7d33ae6f183
---

The AOS "22 DOF" reduced set (as in the 22-DOF/12-v-Mode case) is:
**10 rigid-body + first 7 M1M3 bending modes + first 5 M2 bending modes**.

In the standard ts_ofc 50-DOF ordering [0–4 M2 rigid, 5–9 Camera rigid,
10–29 M1M3 bending 1–20, 30–49 M2 bending 1–20], this is DOF indices
`list(range(0,10)) + list(range(10,17)) + list(range(30,35))`.

NOT the first-22-contiguous indices (that would be 10 rigid + first 12 M1M3).
Pass as an explicit list to `ofc_svd.build_ofc_svd(..., n_dof=DOF_IDX)`.
See [[optatmo-moments-project]].
