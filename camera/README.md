# camera

Studies of LSSTCam sensor and detector behavior using on-sky images from the Vera C.
Rubin Observatory. The work here concerns features of the raw and Instrument Signature
Removal (ISR) corrected pixel data that are properties of the sensors themselves rather
than of the sky or the optics.

<!-- BEGIN generated: python -m rubinwork.studies index -->

## Studies

Generated from each `<study>/study.md`; edit those, not this table.

| study | status | last log | answer |
|---|---|---|---|
| [cosmic_diffusion](cosmic_diffusion/study.md) | active | 2026-11-04 | Full night 20261102 (406 darks), sigma0 prior 2.89 +- 0.5 um: median per-track k = 0.0692 and 0.0686 on R21_S01 and S02 at 8.8 V, 0.0415 to 0.0420 on REB1 at 50 V (dimensionless); R21_S00 is the outlier at 0.0885. All exceed the uniform-field sqrt(2kT/eV), by 1.2x at 8.8 V and 1.7x at 50 V. |
| [itl_dip](itl_dip/study.md) | active | 2026-10-16 | About 3 to 4% of sky deep (50 to 72 counts against 1630 to 1735 counts, i band, 30 s) and 10 to 19 pixels wide on ITL detectors; an e2v control shows 1.3 counts. Measured on one exposure only. |

<!-- END generated -->

## Data access

Images come from the `/repo/main` Butler repository: raw frames from `LSSTCam/raw/all`
and calibrations from `LSSTCam/calib`. Visit selection uses the Consolidated Database
(ConsDB) through its REST interface.

Preliminary Visit Images (PVIs) for the 2026 nights are in the numbered
`LSSTCam/runs/nightlyValidation/<n>` collections; the collections named by `day_obs` end
in September 2025. PVIs are background subtracted, with the background model stored as a
separate dataset, so they are not usable directly for sky-level studies — but they
supply the World Coordinate System (WCS) and photometric calibration.
