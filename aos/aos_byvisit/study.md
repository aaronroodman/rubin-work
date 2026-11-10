---
study: aos_byvisit
topic: aos
status: active
question: 'For one visit, what corner Zernikes (OPD and deviation), optical state (v-modes and DOF), PID input and output, and tweak and trim did MTAOS produce, and can we reproduce the tweak? First case: 20261108 seq 468, whose large tweak should have come from seq 466, which had no recovered donuts.'
answer: 'Reproduced. The Tweak in effect during 20261108 seq 468 came from seq 466, which had Zernikes on only two corners (199, 203); MTAOS counts tables, not finite rows, so its fewer-than-3 skip did not fire, and the 2-corner solve gave camera dx -1380 um and dz -1153 um of state. With kp = 0.3 and ki = kd = 0 (dimensionless gains) the Tweak is kp times the unclipped error (camera dx -414 um); the +-100/+-500 um clip acts only on the unused integral term. The summit ts_ofc v4.8.1 and config v13 reproduce the logged PID error to 9.5e-4 relative (dimensionless).'
uses: {}
started: 2026-11-09
---

# `aos_byvisit`

## Question

For one visit, what corner Zernikes (OPD and deviation), optical state (v-modes and DOF), PID input and output, and tweak and trim did MTAOS produce, and can we reproduce the tweak? First case: 20261108 seq 468, whose large tweak should have come from seq 466, which had no recovered donuts.

## Current answer

Reproduced. The Tweak in effect during 20261108 seq 468 came from seq 466, which had Zernikes on only two corners (199, 203); MTAOS counts tables, not finite rows, so its fewer-than-3 skip did not fire, and the 2-corner solve gave camera dx -1380 um and dz -1153 um of state. With kp = 0.3 and ki = kd = 0 (dimensionless gains) the Tweak is kp times the unclipped error (camera dx -414 um); the +-100/+-500 um clip acts only on the unused integral term. The summit ts_ofc v4.8.1 and config v13 reproduce the logged PID error to 9.5e-4 relative (dimensionless).

## Log

### 2026-11-09 — started

Study created.

### 2026-11-09 — seq 468 of 20261108

Notebook `notebooks/mtaos_visit_step_v1.ipynb`, run `output/20261109_visit_20261108_00468/`
(PDF, summary and 22-DOF CSVs). Code in `code/byvisit.py`.

- The OFC step in effect during seq 468 used seq 466's wavefront: computed 07:16:02 UTC,
  issued 07:16:09 UTC, and seq 468 opened 07:16:17 UTC.
- Seq 466: `wavefrontError` is NaN on 191 and 195. MTAOS logged "Found 4 non-empty tables"
  and "4/4 wavefront error measurements", then `OFC.calculate_corrections` dropped the NaN
  rows and solved 22 DOF from 2 corners. The WEP donut sizes it logged are wrong on those
  corners (191 intra 224 px, extra 80 px).
- PID: kp = 0.3, ki = kd = 0 (dimensionless). The Tweak equals -kp x error with the
  **unclipped** error. `max_integral` (100 um for dx/dy, 500 um for dz, 0.01 deg for camera
  tilts) clips only the integral term, which ki = 0 discards, so the clip never limits a
  P-only step. Had it acted on P, the seq 466 camera dx Tweak would have been -30 um, not
  -414 um.
- The seq 466 step also tilted the camera (rx +0.0199 deg, ry -0.0609 deg). Seq 468 then
  measured Z4 = -7.99 and -9.22 um on 191 and 195, with 199 and 203 under 1 um, and
  its own step (applied to seq 470) moved camera dx +886 um and M2 dz +318 um. That the
  466 tilt caused 468's bad corners is a hypothesis, not tested.
- Reproduction: ts_ofc v4.8.1 and ts_config_mttcs v0.22.12 (the versions MTAOS reported),
  exported with `git archive`. 22 DOF, 12 v-modes, Z4-Z19 and Z22-Z26,
  normalization `range0.5_fwhm-0.15.yaml` (`default.yaml` does not match), and the
  exposure-mean MTRotator angle passed to `dof_state` without the +90 deg `rotation_offset`.
  Max |repro - logged| / max |logged| PID error: 8.0e-4 (seq 466), 9.5e-4 (seq 468),
  dimensionless. The stack's ts_ofc (branch tickets/RSO-1007) with the local v13 config
  misses by 4-28 % relative, so `aos_state.make_state_estimator` is not used here; the
  v-modes are in the summit basis.
- `MTAOS.logevent_degreeOfFreedom.opticalState` is published as all zeros; the state comes
  from the PID log message.
- Open: ConsDB `ccdvisit1_quicklook` z4-z28 differ from the EFD deviation (191, seq 468:
  z7 -1.017 against 0.004 um of wavefront); whether ConsDB holds OPD or another frame is
  not settled. The summit quickLook `zernikes` tables are not in the USDF embargo Butler.
