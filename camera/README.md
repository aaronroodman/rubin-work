# camera

Studies of LSSTCam sensor and detector behavior using on-sky images from the Vera C.
Rubin Observatory. The work here concerns features of the raw and Instrument Signature
Removal (ISR) corrected pixel data that are properties of the sensors themselves rather
than of the sky or the optics.

## Studies

- [`itl_dip`](itl_dip/study.md) — the sky background is depressed in the columns
  containing a saturated star on Imaging Technology Laboratory (ITL) Charge-Coupled
  Devices (CCDs), and not on the e2v CCDs. Measures the depth of the deficit against the
  sky level, which requires running ISR with the pipeline's dip and saturation masking
  disabled. Notebook only.

## Data access

Images come from the `/repo/main` Butler repository: raw frames from `LSSTCam/raw/all`
and calibrations from `LSSTCam/calib`. Visit selection uses the Consolidated Database
(ConsDB) through its REST interface.

Preliminary Visit Images (PVIs) for the 2026 nights are in the numbered
`LSSTCam/runs/nightlyValidation/<n>` collections; the collections named by `day_obs` end
in September 2025. PVIs are background subtracted, with the background model stored as a
separate dataset, so they are not usable directly for sky-level studies — but they
supply the World Coordinate System (WCS) and photometric calibration.
