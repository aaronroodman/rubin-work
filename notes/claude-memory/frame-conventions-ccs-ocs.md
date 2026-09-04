---
name: frame-conventions-ccs-ocs
description: "Rubin frame conventions (CCS/OCS) and that static optical PSF variation moves with the mirrors (OCS), not fixed in the instrument frame"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-08-20T14:59:37.995Z
---

In the user's rubin-work/aos and MIW work:
- **CCS** = Camera Coordinate System (the instrument/camera frame).
- **OCS** = Observatory Coordinate System (the mirror/telescope-structure frame).
- The **camera rotator angle** relates CCS↔OCS↔sky; the canonical value is ConsDB
  `cdb_lsstcam.visit1_quicklook.physical_rotator_angle` (degrees), surfaced as the
  `rotator_angle` column in AOS visits tables. `rotTelPos` (radians) in
  `aggregateZernikesRaw.meta` is the equivalent for AOS Zernike datasets.
- `visitInfo.boresightRotAngle` is the **sky position angle of the boresight — NOT**
  the mechanical camera rotator angle. Don't use it as the rotator.
- **Compute the physical camera rotator from visitInfo (no ConsDB needed):**
  `rotTelPos = parallacticAngle − boresightRotAngle − 90°`
  i.e. `vi.boresightParAngle.asDegrees() − vi.boresightRotAngle.asDegrees() − 90`,
  wrapped to (−180,180]. **Verified vs ConsDB `physical_rotator_angle` = 0.2° ±
  0.26°** over a ~90° rotator sweep (day_obs 20260709). Same form used in
  `aos/code/run_wfs_fam_compare.py` / `run_wfs_refit_ensemble.py` (there via the
  stamp-metadata keys `BORESIGHT_PAR_ANGLE_RAD` / `BORESIGHT_ROT_ANGLE_RAD`,
  with `−π/2`). Frame-rotation machinery: `optatmo/code/frames.py`
  (`rotate_moments`, `rotate_field`, `R(rotTelPos)`; note DVCS→CCS swap first).

**Correction to a wrong assumption I made:** the static optical PSF/focal-plane
variation is **NOT** fixed in the instrument (CCS) frame. The MIW evidence is that
the static focal-plane variation **moves with the mirrors (OCS)**. So the
static-vs-dynamic frame separation must reference the OCS/mirror frame, not assume
the optics are camera-fixed.

Relates to [[z11-intra-extra-mystery]] and the guider static-vs-dynamic moment work.
