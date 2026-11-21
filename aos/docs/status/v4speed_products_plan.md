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

## Findings to keep

- ts_wep `Instrument` takes the v4_speed yaml (`batoidModelName` as a path without
  `.yaml`) and the obscuration file (`maskParamsFile`, absolute path) directly. Pass
  `maskParams=None`: `LsstCam.yaml` carries inline `maskParams`, which win over the file.
- ts_wep refuses a configuration with a 0 mm offset (`defocalOffset` derivation fails).
- The corner grid swaps cameraGeom `FIELD_ANGLE` (DVCS) x and y to get batoid's CCS.
  `smatrix/sparse_fit/code/donut_leakage.py` uses DVCS unswapped (not changed; other
  topic).

## Next action

Build `sim_donuts`: move `raytrace` from `donut_leakage.py` into
`rubinwork.donut_sim`, trace the full v4_speed model (its own vignetting and spider)
at rotator 0 deg, on the `batoid_intrinsic` grid points (giant: a subset of rings and
spokes), and write multi-HDU FITS. Then the sbatch ask for the fits.
