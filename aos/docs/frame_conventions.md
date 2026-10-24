# Frame conventions: CCS, OCS and the camera rotator angle

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (conventions)

The coordinate frames used for field angles, Zernike coefficients and PSF moments in the
Active Optics System (AOS) work, which angle relates them, and where to read that angle.

## Frames

| frame | name | fixed to | rotates with the camera rotator |
|---|---|---|---|
| DVCS | Data Visualization Coordinate System | detector pixels, as cameraGeom `FIELD_ANGLE` returns them | yes |
| CCS | Camera Coordinate System | the camera body | yes |
| OCS | Optical Coordinate System (`ts_wep` / `ts_ofc` name; some docs here say Observatory Coordinate System — the same frame) | the mirrors and telescope structure; constant in altitude/azimuth | no |

- **DVCS → CCS is an x↔y swap**, a reflection, applied *before* any rotation
  (`optatmo/code/frames.py`: `dvcs_to_ccs_field`, `dvcs_to_ccs_moments`). Skipping it gives
  a mirror image of OCS across field_x = field_y: moment fits that look good with a wrong,
  reflected wavefront.
- **CCS → OCS is a rotation by the camera rotator angle** `rotTelPos`:
  OCS = R(rotTelPos) · CCS (`frames.py`: `rotate_field`, `rotate_moments`). For Zernikes,
  `ts_wep` applies `galsim.zernike.zernikeRotMatrix(jmax, -rotTelPos)`
  ([`ts_wep_zernike_intrinsics.md`](ts_wep_zernike_intrinsics.md)).

## Static optics move with OCS, not CCS

The static optical wavefront across the focal plane is **not** fixed in the camera frame.
The Measured Intrinsic Wavefront (MIW) shows the static focal-plane variation moving with
the mirrors (OCS) as the rotator turns. A static-versus-dynamic separation must therefore
reference OCS; assuming the optics are camera-fixed is wrong. Gravity-induced camera-lens
deformation also follows OCS, because the gravity vector is fixed relative to the mirrors
([`../miw_origin/miw_astig_coma_investigation.md`](../miw_origin/miw_astig_coma_investigation.md)).
What is genuinely camera-fixed (CCS) is anything bolted to the camera body: lens figure,
mount stress, and the detector heights.

## The camera rotator angle

### `boresightRotAngle` is not the rotator

`visitInfo.boresightRotAngle` is the sky position angle of the boresight. It is also what
`aggregateAOSVisitTableRaw.meta['rotAngle']` holds. On day_obs 20260420, seq_num 89 it is
69.27 deg (angle), 9.84 deg (angle) from the rotator angle. Never use it as the rotator.
The AOS visits tables used to carry it as a `skyAngle` column, which was neither a sky
angle nor the rotator angle and was never read; phase 3b replaced that column with
`rotTelPos` in rad.

### Identity from `visitInfo`

  rotTelPos = parallacticAngle − boresightRotAngle − 90 deg,

wrapped to (−180, 180] deg. In code: `vi.boresightParAngle.asDegrees() −
vi.boresightRotAngle.asDegrees() − 90`, or from stamp metadata
`BORESIGHT_PAR_ANGLE_RAD − BORESIGHT_ROT_ANGLE_RAD − π/2` (rad), as in
`code/cwfs/run_wfs_refit_ensemble.py` and `code/cwfs/run_wfs_fam_compare.py`. The identity
holds exactly against the `rotTelPos` stored in the AOS tables. Against the Consolidated Database (ConsDB)
`physical_rotator_angle` it agrees to a mean of 0.2 deg (angle) with 0.26 deg (angle) scatter
over a ~90 deg rotator sweep on day_obs 20260709.

### Which source to read (refined 2026-10-10)

The angle that matters is the one the pipeline used for its own CCS → OCS rotation. That is
**`rotTelPos` from the table metadata**, not ConsDB
([`../../notes/status/phase2_fam_tables_handoff.md`](../../notes/status/phase2_fam_tables_handoff.md),
section "A1 — `rotator_angle`").

| source | where | units | agreement with the table's own CCS → OCS rotation |
|---|---|---|---|
| `rotTelPos` | `aggregateAOSVisitTableRaw.meta` | **rad**, bare float | max residual 9.2e-05 deg (angle), 2972 donuts on one visit |
| `rot_tel_pos` | `donutBlitzFamResults.meta` | **deg**, astropy `Quantity` | 0.0000 deg (angle) on 6 of 6 visits |
| ConsDB `physical_rotator_angle` | `cdb_lsstcam.visit1_quicklook` | deg | **0.1 to 0.5 deg (angle)** off in general: over the 53 visits of chunk `20251116_20251130`, median 0.129 deg and max 0.456 deg, with 0 of 53 agreeing to 1e-5 deg. The 1.6e-05 deg agreement reported earlier was **one** visit (day_obs 20260420, seq_num 89) and does not generalize. On the blitz visits it sits **+0.30 deg (angle)** off, drifting +0.015 deg over 6 consecutive visits |

### Why the sources disagree: the rotator moves during the exposure

The camera rotator turns throughout an exposure to hold the sky stationary on the focal
plane. So there is no single rotator angle for a visit — each source samples the angle at
a different moment of the exposure, and the spread between them is the rotator's own
travel over that window. A few tenths of a degree (angle) between `rotTelPos`, ConsDB
`physical_rotator_angle` and an EFD window-mean is **expected tracking motion, not a
pipeline error**.

This is why the rule is to read the angle the pipeline's own CCS → OCS rotation used
rather than the "most accurate" angle: consistency with the rotated coordinates matters,
not which instant was sampled. It is also why the size of the disagreement does not
threaten the downstream use — `rotator_angle` selects FAM visits near the nominal rotator
angles (0, ±15, ±30 deg (angle) and similar), with bins a degree or two wide, so a
fraction of a degree cannot move a visit out of its bin except at an edge (Aaron,
2026-10-12).

Two things make ConsDB alone insufficient:

- **ConsDB is often NULL.** In chunk `20251116_20251130`, 19 of 53 visits have no
  `physical_rotator_angle`. `ts_intrinsic_wavefront`'s `intrinsics_lib.get_rotator_data`
  then falls back to the mean of Engineering Facility Database (EFD) `MTRotator.rotation.actualPosition` over the exposure
  window, which is not reproducible between runs: two builds of the same chunk differ by up
  to 0.188 deg (angle) on exactly those 19 visits and agree to 0 deg on the other 34.
  (Those 34 agreed with *each other* because both runs read the same ConsDB value — not
  because ConsDB matches `rotTelPos`, which it does not; see the table above.)
- **On the blitz path ConsDB is not the angle used.** `rot_tel_pos` reproduces the blitz
  table's rotation exactly; ConsDB sits 0.30 deg (angle) away.

This corrects the earlier rule that ConsDB `physical_rotator_angle` is the canonical
rotator angle. It remains a good cross-check where it is non-NULL, and it is what the
value-added database stores as `optical_state.rotator_angle_deg`
([`../../value_added/docs/schema.md`](../../value_added/docs/schema.md)).

**Landed in phase 3b** of
[`../../notes/status/organization_plan.md`](../../notes/status/organization_plan.md): the
FAM tables' `rotator_angle` column (deg) is now `rotTelPos` from the donut metadata,
because it is the angle the CCS → OCS rotation used, with ConsDB
`physical_rotator_angle` only as the fallback. The EFD window-mean and `visitInfo`
fallbacks are gone — the window-mean was not reproducible between two runs of the same
chunk — so `rotator_angle` is NaN where neither source has a value, and the visit is
reported. A `rotator_angle_source` column records which source each visit used:
`rotTelPos`, `consdb` or `missing`. Both builders write it: the aggregate path in
`ts_intrinsic_wavefront` (`intrinsics_lib.resolve_rotator_angle`) and the blitz path in
`rubinwork/products/fam_tables/builders/run_blitz_mktable.py`
(`resolve_rotator_angle`).

Traps:

1. **Units differ between the two table families** — radians in
   `aggregateAOSVisitTableRaw.meta['rotTelPos']`, degrees in
   `donutBlitzFamResults.meta['rot_tel_pos']`. Convert explicitly; never infer the unit
   from the magnitude.
2. Reading blitz `.meta` needs `parameters={'strip_astropy_meta_yaml': False}`, or it
   returns no keys.
