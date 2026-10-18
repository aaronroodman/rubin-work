# Guider vs adjacent-CCD second moments

> **Status:** current · **Last updated:** 2026-10-11 · **Kind:** reference (conventions)

Definition of the comparison between guide-star second moments and the point spread
function (PSF) second moments on the adjacent science CCD, in
`guider/guider_ellipticity_ensemble/notebooks/guider_moments_ensemble.ipynb` ("Guider vs adjacent-CCD 2nd moments (Q1, Q2, T)
-- matched HSM"). The trace `T` is a 1:1 comparison; `Q1` and `Q2` are not, because
neither side is rotated to the Optical Coordinate System (OCS).

## Estimator

Both sides use the same estimator: the galsim Hirata–Seljak–Mandelbaum (HSM) adaptive
moments in `rubinwork.hsm_moments` (`measure_hsm_moments`), with PIFF naming
e0 = `T`, e1 = `Q1`, e2 = `Q2`. Units: arcsec² at 0.2 arcsec/pixel. Using the same
estimator on both sides removes any weighted-vs-unweighted difference; a change to
`rubinwork/hsm_moments.py` changes both sides.

## Inputs

The notebook merges guider moments (`df`, columns `hsm_*`) with adjacent-CCD moments
(`psf`, columns `*_ccd`) on (`expId`, `detector` == guider detector).

| side | columns | source | frame |
|---|---|---|---|
| guider | `hsm_T`, `hsm_Q1`, `hsm_Q2` | `guider/code/guiderMoments.py` (`decomposeDetector` builds the guide-star coadd and applies HSM to it); `run_guider_moments.py` passes `hsmFunc=measure_hsm_moments`. `summit_utils` supplies only the raw stamps and the star tracker, not the moments. | guider detector pixels, `view='dvcs'`, unrotated |
| science CCD | `T_ccd`, `Q1_ccd`, `Q2_ccd`, `*_ccd_rms` | `guider/code/extract_adjacent_psf_moments.py` (Snakefile rule `psfmoments`, butler repo `/repo/main`). Stars from `single_visit_star` (signal-to-noise selected, clean stars); stamps cut from `preliminary_visit_image` on the guider's adjacent CCD (`code/guider_adjacent_ccds.yaml`, collections in `code/guider_science_collections.yaml`). HSM per star, then median and RMS over the CCD's stars. Not from the Consolidated Database (ConsDB). | science-CCD pixels (DVCS), unrotated |

## Frames

Each side is in its own detector's Data Visualization Coordinate System (DVCS) pixel
frame. Consequences:

- `T` is rotation-invariant, so guider `T` vs CCD `T` is a genuine 1:1 test.
- `Q1`, `Q2` are frame-dependent. They agree only after each side is rotated to OCS by
  its recorded `rot_deg` (the physical rotator angle, `parallacticAngle -
  boresightRotAngle - 90`, in degrees) plus the cameraGeom detector yaw.
- The notebook plots `Q1`, `Q2` unrotated, for scale and pattern only. An OCS-rotated
  `Q1`, `Q2` panel does not exist yet.
