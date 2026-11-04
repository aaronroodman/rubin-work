---
study: psf_shape_frame
topic: value_added
status: active
question: Do the ConsDB per-CCD PSF moments, rotated to Az/El, reproduce the RubinTV psf_shape_azel whiskers, and do the ConsDB sonic-anemometer values agree with the raw EFD?
answer: 'Yes to both. ConsDB per-CCD moments rotated to Az/El match the RubinTV whiskers on 4 visits (rotator -45.7 to +50.9 deg), with |e| quartiles within 0.004 of RubinTV and a WCS check within 0.0038. The AOS OCS is Az/El mirrored in elevation (e2_ocs = -e2_azel). Raw-EFD sonic means equal ConsDB to 0.0000 m/s (median, 2 nights).'
uses:
  value_added: v1@20261025b
started: 2026-11-04
---

# `psf_shape_frame`

## Question

Do the ConsDB per-CCD PSF moments, rotated to Az/El, reproduce the RubinTV psf_shape_azel whiskers, and do the ConsDB sonic-anemometer values agree with the raw EFD?

## Current answer

Yes to both.

- ConsDB `ccdvisit1_quicklook` per-CCD moments, rotated by `rubinwork.psf_shape.azel_rotation`, give the same whiskers, e1 and e2 maps as the RubinTV `psf_shape_azel` plots for `day_obs` 20261027 seq 109, 111, 119 and 131 (rotator angle -45.7 to +50.9 deg). The |e| quartiles over CCDs are within 0.004 (dimensionless) of the RubinTV quartiles over stars.
- A boresight WCS gives the same Az/El directions to 0.0038 (dimensionless, largest element difference of unit vectors).
- The AOS OCS is not the Az/El frame: +x_OCS = +Az, +y_OCS = -El. So `e1_ocs = e1_azel` and `e2_ocs = -e2_azel`.
- Raw-EFD sonic means (value-added `sonic` group) equal ConsDB `mt_salindex<idx>_wind_speed_<a>_mean` to a median 0.0000 m/s for salIndex 110 and 123-126, on 20260410 and 20261027 (723 and 1,502 visits).

## Log

### 2026-11-04 — started

Study created.

`notebooks/psf_shape_azel_check_v1.ipynb`, PDF in `output/20261104_azel_vs_rubintv/`. Library `rubinwork/psf_shape.py`; value-added groups `psf_shape` and `sonic` added to `build_efd_db`.

- Per-CCD whiskers match the four RubinTV plots. Rotator angle from ConsDB `physical_rotator_angle`, 0.2 to 0.4 deg from rotTelPos.
- |e| p25/p50/p75 over CCDs against RubinTV over stars: seq 111 (0.148, 0.193, 0.243) against (0.148, 0.190, 0.242).
- WCS test and OCS relation as in the answer.
- Raw ξ at 2 and 6 CCDs: seq 111, the most coherent, has xi_e1e1 0.0257 and 0.0160, xi_e2e2 0.0123 and 0.0116 (dimensionless; 904 and 1,904 CCD pairs).
- Sonic: EFD means equal ConsDB. The EFD has 107 visits for 110 and 25 for 123-126 that ConsDB lacks (2 nights).
- The EFD answers from milano batch nodes once `EXTERNAL_INSTANCE_URL` is set (`code/efd_batch_probe.py`, job 43082426). `make_efd_client` now does that.
