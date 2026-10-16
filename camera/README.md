# camera

Studies of LSSTCam sensor and detector behavior using on-sky images from the Vera C.
Rubin Observatory. The work here looks at features of the raw and Instrument Signature
Removal (ISR) corrected pixel data that are properties of the sensors themselves rather
than of the sky or the optics.

## Studies

### itl_dip

The sky background along the columns through a saturated star is depressed on ITL
Imaging Technology Laboratory (ITL) Charge-Coupled Devices (CCDs). The effect is not
seen on the e2v CCDs that make up the rest of the focal plane. This study displays the
feature on low galactic latitude images, which are rich in saturated stars, and measures
the depth of the sky deficit against the sky level.

The images are produced by running ISR in the notebook with dip masking disabled. The
Science Pipelines flag these pixels `SUSPECT` and `ITL_DIP` by default, which is correct
for science but removes the pixels the study is about. Twenty-five ITL CCDs carry
production dip-correction settings; no e2v CCD does.

Notebook: `notebooks/itl_dip/itl_column_dip_v1.ipynb`. It depends only on the LSST
Science Pipelines (`lsst_distrib`) and standard scientific Python, so it can be run
outside this repository.

Documentation: `docs/itl_dip.md` describes the feature, the ISR configuration that
preserves it, which CCDs carry production dip-correction settings, how to select
saturated stars by magnitude, and the open problem of measuring the depth reliably in
crowded fields.

## Data access

Images come from the `/repo/main` Butler repository: raw frames from `LSSTCam/raw/all`
and calibrations from `LSSTCam/calib`. Visit selection uses the Consolidated Database
(ConsDB) through its REST interface.

Preliminary Visit Images (PVIs) for the 2026 nights are in the numbered
`LSSTCam/runs/nightlyValidation/<n>` collections; the collections named by `day_obs` end
in September 2025. PVIs are background subtracted, with the background model stored as a
separate dataset, so they are not usable directly for sky-level studies.
