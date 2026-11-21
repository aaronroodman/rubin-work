# v4_speed products and studies: plan

> **Status:** in progress: batoid_intrinsic built · **Last updated:** 2026-11-20 · **Kind:** plan

v4_speed is Josh Meyers' batoid model (v3.14 optics, re-measured baffles without the
45-vane TMA (Telescope Mount Assembly) baffle, CAD camera apertures) and its danish
obscuration file (the danish `mask_params`). The files are in `rubin-optical-model`
(commit 7785f3c) until they go into batoid and danish:
`Rubin_v4_speed_cadaper_{u,g,r,i,z,y}.yaml`,
`RubinObsc_v4_speed_cadaper_r_rtpp0_azp45_pp0d0.yaml`. Separate from Aaron's v4.0x series.

Software: batoid 0.9.0 and danish 1.4.0 from the `w_2026_45` stack, the latest releases
on 2026-11-20.

## Decisions (2026-11-20)

- r band only to start.
- One obscuration file for every band and every defocus configuration. Josh is checking
  it; a change is a new variant. Refit per configuration (`gen_mask_params.py`) only if
  the fit-quality study needs it.
- Rotator angle 0 deg, to match the obscuration file.
- Studies in `aos`.

## Products

Defocus configurations, both sides throughout:

| config | shifted | step (mm) | field |
|---|---|---|---|
| `fam` | camera | ±1.5 | science field, 11 rings × 35 spokes to 1.75 deg |
| `cwfs` | detector | ±1.5 | 8 corner half-chips, 6 × 3 each, physical side only |
| `giant` | camera and M2 | ±4 | science field, 5 rings × 12 spokes |

1. **`batoid_intrinsic`**, variant `v4_speed`. In-focus OPD (optical path difference)
   Zernikes (`zernikeGQ`, Noll 4–78, µm) on a field grid to 1.75 deg plus corners, with
   a double-Zernike fit for interpolation; defocused TA (transverse-aberration)
   Zernikes (`zernikeTA`, Noll 4–78, µm) per configuration and side, as ts_wep builds
   its donut model. Parquet plus FITS; the manifest records the yaml sha256 values.
2. **`sim_donuts`**, variant `v4_speed_mono`. Noise-free donuts ray-traced through the
   full v4_speed model (vignetting and spider from the model), monochromatic, Kolmogorov
   1.0 arcsec FWHM. Multi-HDU FITS per configuration and side. Noise is added at fit
   time.
3. **`sim_donut_fits`**, variants `v4_speed_sparse` and `v4_speed_dense`. Danish fits
   with the v4_speed obscuration file and the TA reference from product 1. Sparse: the
   d13-sparse production Noll set for `fam` and `cwfs`, the fit-c set for `giant`;
   dense: Noll 4–28. Per donut: fitted Zernikes, deviation from the reference, blur,
   centroid, chi2/dof, residual RMS; model and residual stamps.

Code to reuse: `smatrix/sparse_fit/code/donut_leakage.py` (`raytrace`, `zk_ta`,
`factory_for`) and `rubinwork/giant_donuts/`. Study code cannot be imported, so the
shared parts move to a library (`rubinwork/donut_sim.py`). Sims and fits run as an
sbatch array; ask before submitting.

## Studies

- `aos/v4speed_mask_vs_batoid`: Danish model drawn at the Batoid TA wavefront, no
  fit, against the Batoid-traced donuts. Tests only the obscuration file against the
  Batoid pupil. Reads products 1 and 2.
- `aos/v4speed_danish_closure`: fit quality, and fitted Zernikes against the Batoid
  wavefront. The reference is the truth, so a nonzero fitted deviation is danish
  model bias. Reads products 1–3.

## Done

- 2026-11-20: `batoid_intrinsic` `v4_speed` build `20261120` (current), r band, built
  in 53 s on one CPU. TA matches ts_wep's `_getIntrinsicZernikesTACached` exactly. The
  k <= 36 double-Zernike fits leave a largest residual of 1e-4 µm. In-focus v4_speed
  minus ts_wep `LSST_r`: Noll 4-78 quadrature sum median 0.098 µm, maximum 0.20 µm over
  385 points, mostly z5 and z6. v4_speed carries the v3.14 decenters and tilts; `LSST_r`
  is an older prescription.
- 2026-11-20: frames. Build `20261120` rewritten in place with CCS and OCS columns
  (`thx_CCS_deg`, `z4_OCS`, ...) and a `rotator_angle_deg` variant field (telescope
  rotated by minus it; 0 for `v4_speed`). Double-Zernike fits in OCS, as the
  optical-state fits. Shared conversions in
  `rubinwork.common.frames`; rule "no DVCS in AOS work" in `aos/CLAUDE.md`.

## Findings to keep

- ts_wep `Instrument` takes the v4_speed yaml (`batoidModelName` as a path without
  `.yaml`) and the obscuration file (`maskParamsFile`, absolute path) directly. Pass
  `maskParams=None`: `LsstCam.yaml` carries inline `maskParams`, which win over the file.
- ts_wep refuses a configuration with a 0 mm offset (`defocalOffset` derivation fails).
- The corner grid swaps cameraGeom `FIELD_ANGLE` (DVCS) x and y to get batoid's CCS.
  `smatrix/sparse_fit/code/donut_leakage.py` uses DVCS unswapped; noted in that study's
  Log, to fix before its rerun.
- ts_wep `CalcZernikesNeuralTask` stores DVCS field angles as `thx_ccs` (no swap), so its
  OCS and N/W columns are x/y swapped: `wfs/docs/calc_zernikes_neural_frames_bug.md`.
- batoid projection: ts_wep and this product use the default (postel);
  `ts_intrinsic_wavefront` `batoid_intrinsic.py` uses gnomonic. Largest difference
  0.0004 µm over Noll 4-28 at (1.2, 1.2) deg; ignore.

## sim_donuts (2026-11-20)

Code, tests and `run_build.sbatch` written. A smoke build (every 40th donut, status
`partial`) sits at `sim_donuts/v4_speed_mono/20261120`; the batch run overwrites it.
Full set: 770 FAM, 144 CWFS, 84 giant donuts, about 100 CPU minutes.

- Ray spacing on the stop plane: 8.0 mm (FAM, CWFS; 0.72 M rays, 4.2 s each) and 3.5 mm
  (giant; about 4 M rays). Giant donuts are about 690 px across, FAM 136 px.
- Stamps are odd (181, 801 px): Danish asserts odd `npix`. The chief ray is on the centre
  pixel.
- Orientation checked against Danish at the TA wavefront: the stored stamp matches best
  (half L1 difference, fraction of flux: CWFS R04_SW0 0.036 vs 0.107-0.137 transposed or
  flipped; FAM r05_a25 0.019 vs 0.034-0.091). Danish `spider_angle=0` matches better
  than `None`: first item for `v4speed_mask_vs_batoid`.
- `flux_m2` is the stamp sum; Kolmogorov wings past the stamp edge are not in it.

## v4speed_mask_vs_batoid pass 1 (2026-11-21)

`sim_donuts` `v4_speed_mono@20261120` built (job 46234293, 7.5 min) and current.
`batoid_intrinsic` build `20261121` stores Noll 1-78: ts_wep's Danish reference includes
the TA tilts, and giant-donut z2 reaches 14 µm (3 px). Study result in
`aos/v4speed_mask_vs_batoid/study.md`: FAM l1 0.018, CWFS 0.022-0.025, giant 0.033
(fraction of flux, dimensionless); giant pupil edges do not match.

Conventions to keep: Danish centres an odd stamp at +0.5 px in x and y; Danish `dx`, `dy`
are arcsec; `spider_angle=0` for the v4_speed obscuration file.

## Next action

Aaron to choose: refit the obscuration file at the giant configuration (or ask Josh)
before the fits, or go on to `sim_donut_fits`, whose Danish fits need batch (ask first).
