---
name: guider-atmo-sim
description: "imSim-based simulator for atmospheric PSF on the 8 guide sensors, in rubin-work/guider/code"
metadata: 
  node_type: memory
  type: project
  originSessionId: 493209c9-433d-40d1-8407-7d862d0df97c
  modified: 2026-07-30T21:48:41.108Z
---

Guider atmospheric-PSF simulator lives in `rubin-work/guider/code/`:
`guider_atmo_sim.py` (sim) + `guider_atmo_stripcharts.py` (centroid/moment time series) +
`guider_atmo_movie.py` (summit_utils-style grayscale mosaic movie: Greys/origin=lower/nearest,
per-frame percentile scaling, FuncAnimation, gif=pillow/mp4=ffmpeg) + `guider_atmo_fovmap.py`
(focal-plane FWHM/ellipticity-whisker/3rd-moment-spin3 maps). Outputs -> `guider/output/atmo_sim/`.

Modes (`--mode`): `guiders` (one fixed star/guide CCD, 50 ms stamps @5 Hz, 150/30 s visit ->
`[n_stamp,n_guider,200,200]` .npy + moments parquet) and `fov` (one star per SCIENCE CCD, single
long exposure -> per-CCD FWHM/e1/e2/3rd-moment parquet for focal-plane maps vs data).
Atmosphere source (`--atmo-source`): `imsim` (Ellerbroek profile replicated inline) or `psfws`
(psf-weather-station). Render (`--render`): `phot` (geometric+SecondKick) or `fft` (full spectrum,
no second kick) as a cross-check. `--doOpt` adds imSim OptWF optics.

Key design decisions (non-obvious):
- REFACTORED (2026-08) to a layer-parameter model: `imsim_layer_params`/`psfws_layer_params` ->
  `make_atmo` builds a raw `galsim.Atmosphere` directly (NOT via imsim.AtmosphericPSF), so render
  mode controls instantiation (phot: kmax=kcrit/r0 + SecondKick; fft: full spectrum instantiate()).
  Seeing pinned to rawSeeing via von Karman r0 solve (`_r0_500_for_target`, copied from imSim/psfws).
  Draws via `atm.makePSF(theta, t0=k*cadence, exptime)`; t0 walks the same frozen screens.
- Only `lsst.obs.lsst` (LsstCam geometry) is a hard dep now; imsim import only needed for --doOpt.
- psfws: bundled data is Cerro Pachon 2019-05..10. `--psfws-month 7` samples July/winter climatology.
  For ACTUAL 2026 weather: `fetch_cp_weather.py` (wraps psfws.get_ecmwf_data via sys.path) downloads
  ERA5 model-level winds (lat=-30.25 lon=-70.75) -> `ecmwf_..._YYYYMMDD_YYYYMMDD.p`; feed via
  `--psfws-forecast <file> --psfws-data-dir <dir> --psfws-date "YYYY-MM-DD HH:MM"` (nearest datapoint,
  times are 00/06/18 UTC). NO summit telemetry needed: custom forecast -> forecast-only (telemetry=None),
  ground layer interpolated from ECMWF. Prereqs (checked by fetch preflight): `pip3 install xarray cfgrib`
  (cfgrib done 2026-08; xarray STILL missing on macports py; cdsapi 0.7.7 + eccodes present) + a
  `~/.cdsapirc` CDS token + accept ERA5-complete licence (NOT set up yet). Caveat: psfws hardcodes
  expver='1'; very recent dates are ERA5T (expver 5) and may return empty until final ERA5 (~2-3 mo lag).
  User accepts 2x jet-layer screen wrap. jet layer can exceed screen_size cap -> report_atmo warns "WRAPS".
- FoV mode `--fov-dets` = science|guiders|**all** (default all = 189 science + 8 guide CCDs). NOW also
  `--also-fov` in GUIDERS mode: draws one long-exposure star per science+guider CCD from the SAME
  calibrated guider atmosphere (same r0/screens/realization) -> guider_atmo_fov_moments.parquet +
  guider_atmo_fov_stamps + guider_atmo_fov_meta.json. run_fov saves fov stamps too now. Use --also-fov
  (not a separate fov run) to guarantee the FoV shares the guider atmosphere (a separate --mode fov run
  re-seeds and the FWHM calibration changes r0). guider_atmo_fovmap.py makes FWHM/ellipticity/spin-3 maps
  + a PSF-stamp focal-plane MOSAIC (make_psf_mosaic). Atmosphere-only FoV: FWHM ~flat, ellipticity coherent
  but small (~0.02); the big optical FoV pattern in data is absent (sim has no optics) -> that gap IS the
  result. 837 exposures on 20260706 -> night batch is a batch-node job (not built yet; hold per user).
- DVCS orientation (Option A, chosen): `_det_entry` builds each stamp WCS from FIELD_ANGLE<->FOCAL_PLANE
  (FOCAL_PLANE = DVCS view; summit_utils `transformation.roiImageToDvcs` = np.rot90(-getNQuarter()),
  pure rotation no flip). Stamps drawn directly in DVCS at 0.2"/pix -> no per-CCD remap; moments/centroids
  in DVCS axes; movie octagon needs no rotation. NEEDS stack verification of axis SIGN vs a real RubinTV
  frame.
- DONE: `guider_atmo_to_guiderdata.py` adapter feeds sim DVCS stamps into the REAL summit_utils
  GuiderPlotter. `SimGuiderData` duck-types the GuiderData surface GuiderPlotter uses (len, [det,i],
  getStampArrayCoadd, guiderNames, view, camRotAngle, alt, az, expid, guiderDurationSec) + builds
  starsDf (detector/stamp/xroi/yroi/xroi_ref/yroi_ref) from moments for the tracking centroid. Sim writes
  `guider_atmo_meta.json` sidecar (alt/az/cadence/dayObs/seqNum/roi/...). RSP: `python
  guider_atmo_to_guiderdata.py --outdir <d> --save m.mp4`. VALIDATED on the RSP stack (2026-08): drives
  the real GuiderPlotter.makeAnimation end-to-end on sim data. Gotchas fixed along the way: adapter must
  tolerate missing meta.json (older output); starsDf must carry expid + dalt/daz (+ dx/dy/magoffset) or
  GuiderPlotter.__init__/getStdCentroid KeyError.
- Noise/flux/FWHM (2026-08): stamps now in ADU counts with shot+read noise. draw_stamp shoots in
  electrons (flux*gain, gain=1.6 e-/count), adds Gaussian read noise (default 6 e-), /gain -> counts.
  `--flux-counts` total ADU per star; `--flux-file` JSON {detName:counts} for per-guider observed flux
  (get from summing each stacked-coadd stamp; NOT in the moments summary parquet). `--target-fwhm F`
  calibrates the atmospheric von Karman FWHM so the median per-stamp delivered FWHM = F (2-iter same-seed
  rebuild); FoV sets it directly. Observed for 20260705/688: fwhm_med ~1.18-1.34" (median ~1.23", R00_SG1
  outlier 1.83"), jitter_altaz=0.218", alt=55.05 az=-75.88 camRot=-8.13 (in synced moments summary; NO
  flux column). Real cadence there ~133 stamps/30s ~4.4 Hz (vs 5Hz/150 spec).
- RubinTV PSF-SHAPE focal-plane plot: code is in summit_extras (NOT summit_utils):
  packages/summit_extras/.../plotting/psfPlotting.py -> makeFigureAndAxes(nrows=3) + makeAzElPlot(fig,axs,
  table, camera, physicalFilter). Table cols (makeTableFromSourceCatalogs): detector, x,y (FP mm), Ixx/Iyy/
  Ixy (arcsec^2), T, FWHM=sqrt(T/2*ln256), e1=(Ixx-Iyy)/T, e2=2Ixy/T, e, coma1=HOM30+HOM12, coma2=HOM21+HOM03,
  trefoil1=HOM30-3HOM12, trefoil2=3HOM21-HOM03, kurtosis=HOM40+HOM04+2HOM22; +extendTable(aaRot,"aa") &
  meta(aaRot,nwRot,rotTelPos,rotSkyPos,az,el,"LSST BUTLER DATAID VISIT"). higher moments are HSM-STANDARDIZED
  (whitened by adaptive 2nd moments -> dimensionless, kurtosis=2 for Gaussian). Sim now emits exactly these
  (measure_third rewritten to whitened form; sim kurtosis 2.12 vs data 2.11). BRIDGE: guider_atmo_psfshape.py
  builds the astropy Table from the sim FoV output + looks up FP mm from LsstCam, calls the REAL makeAzElPlot
  -> identical plot for sim & data. VALIDATED on RSP (20260706/688: plot renders, FWHM 0.71, e~0.02,
  kurtosis 2.120 vs data 2.11 -- atmosphere-only is very uniform, confirming optics dominate FoV variation).
  psfshape --marker-size (default 40) enlarges the FWHM/e1/e2 scatter (summit_extras s=1 too small for
  189 pts). rtp from --rot-tel-pos = camRotAngle (approx; exact needs parAngle), rsp defaults 0.
  One-star-per-CCD=189 is --fov-dets science.
- NIGHT BATCH: guider_atmo_night.py loops a night's seqNums (from the on-sky summary), per visit pulls
  alt/az + mjd->psfws-date + field-median fwhm_med + camRotAngle, builds one atmosphere, draws guider
  stamps, decomposes centroid-MOTION moments (guiderMoments.measureStampMoments) and tabulates/plots sim
  vs data (Q1/Q2/T motion). --max/--stride to test; 837 exposures = batch-node job. Structure validated on
  laptop (fallback geom). Guider movie for a run: guider_atmo_to_guiderdata.py (cutout default 25 = RubinTV).
- Movie zoom: RubinTV guider movies use GuiderPlotter cutoutSize ~25-26 px (measured from the 2" circle);
  adapter --cutout default now 25 (was 31) so sim/real movies overlay. Both movies show "Center Stdev."
  = per-axis centroid jitter (sim 0.09" vs data 0.07" for 20260706/688).
- Moment comparison (`guider_atmo_compare.py`): reuses guiderMoments.measureStampMoments (importable w/o
  stack — lsst import is TYPE_CHECKING-guarded) to reproduce decomposeDetector on the sim stamps: static
  = flux-weighted mean per-stamp moments; motion = flux-weighted centroid covariance; noise-floor
  subtracted; arcsec^2. Diffs vs the night summary (Mxx_stamp/T_stamp static, Mxx_motion/T_motion motion).
  20260706/688 result: static T_stamp matches DATA to 0.5% (FWHM cal works). STATIC ellipticity (Q1/Q2)
  is optics-dominated and the sim has no optics (doOpt is only a mock, won't reproduce the real telescope
  aberrations) -> STATIC shown for reference only. The CLEAN comparison is the CENTROID-MOTION 2nd moments
  (purely atmospheric; optics don't move the centroid): script now reports Ixx/Iyy/Ixy/Q1/Q2/T of the
  motion sim-vs-data with a SIM/DATA field-median ratio. 20260706/688: sim over-predicts motion ~1.5-1.6x
  (T_motion 0.0078 vs 0.0050 arcsec^2, Ixx worst at 1.6x) -> psfws winds/ground-layer give too much image
  motion (or the placeholder psfws-date/exact obs time). RSP run w/ real geometry+exact UTC for the definitive number.
- Validation/astrometry/tracker additions (2026-08): (#4 confirmed: render=phot ray tracing, doOpt=False no
  optics, night batch atmo_source=psfws). (#1) `guider_atmo_atmoinfo.py` per-visit atmosphere validation:
  phase-screen images (atmo.atm[i]._tab2d.f), wind profile, Cn2 fraction vs altitude, FWHM-calibration
  record -> guider_atmo_atmosphere.json + 2 PNGs; --atmo-plots (on in night batch); build_calibrated_atmo
  records _calib. (#2) astrometric shifts: FoV table shift_x/shift_y = centroid - no-atmo Airy reference
  (reference_offsets; bias <0.1mas, negligible); guider_atmo_fovmap.make_astrometry quiver (~10mas rms,
  coherent). (#3) Cn2 in atmosphere.json; guider_atmo_tracker.py (Option A) reproduces GuiderStarTracker
  DEFAULT_COLUMNS schema by running the REAL summit_utils helpers (makeInitGuiderWcs/getCamRotAngle/
  getVisitInfo from butler + convertToFocalPlane/convertToAltaz/computeOffsets/computeRotatorAngle/
  convertEllipticity) on sim centroids (xccd from FIELD_ANGLE->PIXELS of field-angle+DVCS-shift, no rot
  needed). For Jackie's tomography. RSP-only, expect stack iteration. Night batch --repo emits it per visit.
- Per-night pointing (`resolve_pointing`): priority explicit `--alt`/`--az` > `--obs-time` (or --psfws-date)
  + boresight_ra/dec via astropy AltAz @ Cerro Pachon > airmass fallback. Resolved alt/az feeds psfws
  `skycoord=True` so winds rotate into the correct sky frame (verified: winds rotate with az); resolved
  airmass (secz) overrides cfg for the seeing scaling. Warns if boresight alt<10 deg.
- CDS token: register/login cds.climate.copernicus.eu -> profile page has Personal Access Token ->
  ~/.cdsapirc (url: https://cds.climate.copernicus.eu/api + key: <token>) -> accept ERA5-complete licence.
- ERA5T GOTCHA (hit 2026-08): a recent-date ECMWF fetch returns BOTH final ERA5 (expver=1) and
  preliminary ERA5T (expver=5) for the overlap, so _process_grib writes DUPLICATE timestamps (identical
  values). psfws get_parameters then passes a length-2 Series to its spline -> "x and y same length".
  Fixed: `_psfws_generator` drops duplicate timestamps (keep=first); fetch_cp_weather de-dupes on write.
  The delivered 2026-07 file was 132 rows / 93 unique (07-01..07-13 duplicated); cleaned in place.
- resolve_pointing raises if boresight below horizon (was -> negative airmass -> complex seeing solve);
  default boresight (60,-30) is NOT up at arbitrary UTC times, so set a real boresight or --alt/--az.
- Fallback if 2026 July ERA5 is empty (ERA5T): fetch_cp_weather validates row count and suggests
  (1) last year (2025 July, final ERA5) or (2) bundled 2019 climatology (--psfws-month 7, no download).
  User OK using 2019 as plausible or 2025 if 2026 unavailable. xarray now pip-installed (2026-08);
  only ~/.cdsapirc + licence remain before a real fetch.
- `choose_screen_size` auto-sizes from actual drawn winds/FoV/exposure, capped at screen_size_max
  (default 1638.4 m ~13 GB/6 layers, 2^14 @0.1 m).
- Phase-screen size: imSim default 819.2 m is TOO SMALL for full-FoV multi-guider over 30 s.
  Need >= 2*theta_max*H_top(15.46km) + v_max(20)*30s + D(8.36) ~ 1559 m -> auto-sized to
  1638.4 m (2^14 @ 0.1 m) ~ 13 GB for 6 layers. 819 m is ok only if guiders analyzed
  independently (per-guider temporal uniqueness holds; cross-guider aliasing does not).
- Guide-sensor field angles + focal-plane WCS come from real cameraGeom
  (`LsstCam().getCamera()`, FIELD_ANGLE<->PIXELS); needs the lsst stack. `--selftest` runs the
  draw/moments/IO plumbing with pure GalSim (no imSim/stack).

Distinct from [[guider-summit-utils-fixes]] (that is on-sky guider moment analysis).
