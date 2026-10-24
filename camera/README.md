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
