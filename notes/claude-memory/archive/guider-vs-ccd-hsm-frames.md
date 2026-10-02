---
name: guider-vs-ccd-hsm-frames
description: How the guider-vs-adjacent-CCD 2nd-moment (matched HSM) comparison is computed and in which frames
metadata: 
  node_type: memory
  type: reference
  originSessionId: 086f02ad-edae-43f2-b884-836dc3cd9f2c
  modified: 2026-09-04T18:21:56.349Z
---

The ensemble-notebook "Guider vs adjacent-CCD 2nd moments (matched HSM)" plot (`rubin-work/guider/guider_moments_ensemble.ipynb`) merges guider moments (`df`, `hsm_*`) with adjacent-CCD PSF moments (`psf`, `*_ccd`) on (expId, detector==guider).

**Both sides use the SAME estimator:** optatmo galsim-HSM (`measure_hsm_moments`, PIFF naming e0=T, e1=Q1, e2=Q2). Matched-algorithm, no unweighted-vs-weighted confound.

**Guider side (`hsm_Q1/hsm_Q2/hsm_T`):** from Aaron's own `guiderMoments` pipeline (NOT summit_utils moments — summit_utils only supplies raw stamps + the star tracker). HSM is applied to the guide-star **coadd** in the **guider detector pixel frame, `view='dvcs'`**, unrotated. `run_guider_moments.py` injects `hsmFunc=measure_hsm_moments`; `guiderMoments.decomposeDetector` builds the coadd and calls it. arcsec^2 at 0.2"/px.

**Science-CCD side (`Q1_ccd/Q2_ccd/T_ccd`):** NOT ConsDB. From `extract_adjacent_psf_moments.py` (Snakefile `psfmoments` rule, `/repo/main`): `single_visit_star` (SNR-selected, cleanStars) list, stamps read from **`preliminary_visit_image`** on each guider's adjacent science CCD (`guider_adjacent_ccds.yaml`, collections `guider_science_collections.yaml`). Same HSM per star, then **median (+RMS) over that CCD's stars** -> `*_ccd`/`*_ccd_rms`. In the **science-CCD pixel (DVCS) frame**, unrotated.

**Frames caveat:** both sides are in their own detector's DVCS pixel frame, NEITHER rotated to OCS. So **T** (trace) is a genuine 1:1 test; **Q1/Q2 are frame-dependent** and only line up after an OCS rotation (each side by its recorded `rot_deg` = physical rotator = parallacticAngle - boresightRotAngle - 90, + cameraGeom detector yaw). The plot shows Q1/Q2 unrotated "for scale/pattern only." See [[frame-conventions-ccs-ocs]].
