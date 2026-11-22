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
  `.yaml`) and the obscuration file (`maskParamsFile`, absolute path). **Passing
  `maskParams=None` to the constructor does nothing** (`mergeConfigWithFile` drops `None`
  kwargs), so `LsstCam.yaml`'s inline v3.14 `maskParams` win over the file. Clear them
  with the setter after construction, `instr.maskParams = None`, as
  `rubinwork.donut_sim.optics.instrument` now does and checks. Before 2026-11-21 it did
  not: every Danish model drawn until then used the shipped v3.14 mask.
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

## sim_donuts build and TA tilts (2026-11-21)

`sim_donuts` `v4_speed_mono@20261120` built (job 46234293, 7.5 min) and current.
`batoid_intrinsic` build `20261121` stores Noll 1-78: ts_wep's Danish reference includes
the TA tilts, and giant-donut z2 reaches 14 µm (3 px). The first study pass ran with
ts_wep's shipped mask by mistake; its outputs were deleted (see "Mask fix" below).

Conventions to keep: Danish centres an odd stamp at +0.5 px in x and y; Danish `dx`, `dy`
are arcsec; `spider_angle=0` for the v4_speed obscuration file (None leaves the vanes
out, l1 0.018-0.026).

## Edge grid (2026-11-21, Aaron approved)

Eight obscuration-file edges switch on between 1.4 and 1.725 deg (L1 aperture ring
1.537 to filter rear retainer 1.714 deg); giant donuts sampled only 1.394 and 1.628 deg.
(Motivated by the shipped-mask pass; with the v4_speed file the zone shows no extra
mismatch, but the grid stays as the dense check.)
`batoid_intrinsic` build `20261121b` (current) adds an `edge` grid, 1.400-1.725 deg in
0.025 deg steps by 24 azimuths, for `fam` and `giant`; its other rows equal `20261121`
exactly. `sim_donuts` variant `v4_speed_mono_edge`: 1344 donuts (FAM and giant, both
sides), about 5 CPU hours.

## Mask fix and pass 2 (2026-11-21)

`optics.instrument` had been using ts_wep's shipped v3.14 mask (fixed `90381bf`). With
the v4_speed file Danish and batoid agree to about 0.2% of the flux (l1) for FAM, CWFS
and giant, edge grid included; `sim_donuts v4_speed_mono_edge@20261121` built (job
46362870) and current. Giant edges are off 3-8 mm on the stop plane from the ±4 mm M2
and camera shift, sign reversing with side. Details in the study Log.

## sim_donut_fits (2026-11-21)

Code `6d381db`: `rubinwork.donut_sim.fit` mirrors ts_wep `DanishAlgorithm._estimateSingleZk`
(block-mean binning, start at the TA reference with fwhm 1.0 arcsec, fwhm bounds 0.1-5.0
arcsec, sky variance over binning², model added as Poisson variance). Noise-free stamps
scaled to 1e6 electrons, sky 1000 electrons per 10 µm pixel. Four variants: `v4_speed_sparse`
(d13-sparse set for FAM and CWFS, fit c for giant, bin 2, spider on, run to convergence),
`v4_speed_dense` (Noll 4-19, 22-26), `v4_speed_sparse_prod` (no spider and the donut_viz
tolerances, as production), `v4_speed_sparse_bin1`. 2342 stamps each, both sim variants.

- Danish's model centre is +0.5 px from the middle pixel only when `(n - 1) // 2` is even
  (n the stamp side): the Kolmogorov kernel is drawn on a `(n - 1) // 2` pixel galsim
  image. Binned giant stamps (399 px) are centred on the middle pixel; 89, 181 and 801 px
  stamps at +0.5. `fit.centre_offset_px` handles it.
- Smoke (every 60th stamp, `20261121_smoke`, partial): binned x2 the fitted FWHM is
  0.96-0.97 arcsec (true 1.0) and FAM l1 0.0035-0.0042, against 0.99 arcsec and 0.0016
  unbinned. Fitted deviations, quadrature sum over the fit set: FAM median 0.018 µm, giant
  0.19-0.23 µm.

## sim_donut_fits builds (2026-11-22)

All five unperturbed variants built and current (`20261121`; `v4_speed_dense_prod`
`20261122`). The first bin1 job hung: danish caches up to 1000 kernels per method on the
class, across models (5 GB per process at 801 px); `fit.clear_danish_caches` after each
fit (`c70245d`). Examples PDF in `v4speed_danish_closure` (`fit_examples.py`): data,
model, residual and OPD against the intrinsic with photon-noise 1-sigma from the Jacobian
(`fit.param_errors`).

Terminology (Aaron): OPD = the fitted wavefront; intrinsic = batoid in-focus wavefront
(`gq_infocus`); Zernike deviation = OPD - intrinsic; intrinsicTA = the defocused TA
Zernikes Danish's model starts from. ts_wep hands Danish intrinsicTA - intrinsic +
zkStart (= intrinsic) for the fitted terms. The product's `dz<j>` is the deviation.

## DOF scan (2026-11-22, Aaron approved all jobs in advance)

`sim_donuts v4_speed_mono_dof_p2`: the v4_speed_mono points, once per DOF of the 22-DOF
set, each DOF alone at +2 steps of the sparse_fit scan (twice the FAM-mean step of run
`20261111_dz_scan_ofcsign`, about 0.04 µm RMS in-focus Noll 4-78 change; z4 -0.35 µm for
Cam dz +21.85 µm). `optics.perturb` (batoid_rubin, ts_ofc sign). The index adds the
perturbed in-focus OPD. Fit variants `v4_speed_dof_p2_sparse` and
`v4_speed_dof_p2_dense_prod` against the nominal intrinsicTA; they store the expected
deviation `dz<j>_true_CCS_um`. Each product builds one partition per DOF
(`--dof-part`, an sbatch array) then `--finalize`. Jobs: sims 46442649 (array), finalize
46442671, fits 46442672 and 46442673 (arrays), finalize 46442674 and 46442675; logs
`aos/logs/<product>_<array>_<task>.log`. Code `dd904cf`.

Study `aos/v4speed_dof_fits`, script `code/dof_fits.py` (`5f68faa`): summary PDF of
expected deviation, fit error and error of the change (unperturbed fit removed) per DOF
and configuration, and per variant an examples PDF, one page per DOF.

A smoke partition (DOF 5, every 100th donut) sits at
`products/sim_donuts/v4_speed_mono_dof_p2/20261122_smoke`; delete after asking.

## Next action

When the chain ends: run `aos/v4speed_dof_fits/code/dof_fits.py`, Log entry, report.
Then `v4speed_danish_closure` field maps of the deviation per Noll term.
