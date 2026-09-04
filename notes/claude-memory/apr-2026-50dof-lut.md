---
name: apr-2026-50dof-lut
description: A fixed 50-DOF/21-v-mode single-value LUT was on-sky ~Apr 24-28 2026 (explains the 20260424/20260428 anomaly); NOT active Apr 9
metadata: 
  node_type: memory
  type: reference
  originSessionId: d6ed390e-ee48-44ee-8c0a-a6efca6a5fdf
  modified: 2026-07-27T23:27:51.283Z
---

On-sky, a fixed **50-DOF / 21-v-mode LUT** was installed ~**2026-04-24** and used
through ~**2026-04-28**. It applied a SINGLE optical-state offset at ALL
elevations and rotator angles (not an elevation/rotator-dependent lookup).

- **Explains the 20260424 / 20260428 anomaly** seen in the T539 closed-loop
  v-mode analysis: the "camera-focus-baseline" nights with Cam hex dz ~ -2400 µm
  (vs normal ~ -240), distributed across Cam piston + M1M3 B3 + M2 B5; recovered
  by ~seq 397. See [[optatmo-moments-project]].
- It was **NOT active on 2026-04-09**, so it does NOT explain the distinct
  wavefront-map population at the ±30°/±45° camera-rotator bins on Apr 9 (user
  confirmed 2026-07-27). Those 4 rotator subsets — [-50,-40],[-35,-25],[25,35],
  [40,50] — are the ones dropped from the `pathA_50_34_i_5rot` MIW build in
  `fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x`; they come EXCLUSIVELY from the one
  night 2026-04-09 (kept 5 bins are all March: 03-15/16/17/24). Cause of the
  Apr-9 population difference is still OPEN.
