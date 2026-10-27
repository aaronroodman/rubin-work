# Atmospheric PSF simulation for the Rubin guide sensors and science CCDs

> **Status:** current · **Last updated:** 2026-10-27 · **Kind:** reference (method)

The simulation draws stars through a model atmosphere for one Rubin visit. It covers two
sets of detectors with the same atmosphere:

- the 8 guide sensors (corner CCDs read out in a 200x200 pixel region of interest, ROI),
  as a train of 50 ms stamps at about 5 Hz through the visit;
- the 189 science CCDs, one star per CCD, integrated over the full visit exposure.

It contains **atmosphere only**, with no telescope optics and no tracking errors. Its
purpose is to predict two things for comparison with data:

1. the amount and pattern across the field of the point-spread function (PSF) full width
   at half maximum (FWHM) and ellipticity that turbulence alone produces;
2. the size of the second moments of the guide-star centroid motion.

The code is `code/guider_atmo_sim.py`. It is built on GalSim (version 2.8)
phase-screen PSFs.

## 1. Turbulence profile

The atmosphere has 6 frozen-flow von Kármán layers. Each layer has an altitude, a
fractional turbulence weight (its share of Cn², the refractive-index structure constant
integrated through the layer), a wind speed and a wind direction. These come from one of
two sources.

**psf-weather-station (psfws).** This is the default for comparisons with data. psfws
(Hébert et al.) builds the profile for Cerro Pachón from European Centre for
Medium-Range Weather Forecasts (ECMWF) ERA5 reanalysis winds and temperatures:

- free-atmosphere layers from the ERA5 model levels;
- a ground layer interpolated from the same data. The summit telemetry is not used when a
  custom forecast is given.

For a given visit we fetch ERA5 for the night (`code/fetch_cp_weather.py`, latitude
−30.25 deg, longitude −70.75 deg). We then take the ERA5 time step nearest the visit;
ERA5 has 00, 06, 12 and 18 UTC. psfws gives the layer altitudes, the turbulence weights
and the winds. With `skycoord=True` and the telescope altitude and azimuth, it rotates
the winds into the frame of the sky as the telescope sees it.

**imSim profile.** This is the alternative: the Ellerbroek six-layer profile used in
imSim production.
- Altitudes 0.2, 2.58, 5.16, 7.73, 12.89 and 15.46 km.
- Nominal weights 0.652, 0.172, 0.055, 0.025, 0.074 and 0.022 (dimensionless, fraction
  of total Cn²), each jittered by 10%.
- Random wind speeds from 0 to 20 m/s and random directions.

**Outer scale and seeing.** The outer scale L0 is drawn log-normally about 25 m and
limited to 10 to 100 m; all layers share it. The overall turbulence strength, r0 at
500 nm, is set by solving the von Kármán FWHM formula (Tokovinin 2002) for a target
FWHM. psfws and the imSim profile set only the *relative* layer weights, never the
absolute seeing.

## 2. Matching the seeing to the visit

The target is the measured guider FWHM for the visit: `fwhm_med`, the median over guide
sensors from the guider moment pipeline. A short calibration loop makes the simulated
FWHM match it:

1. Build the atmosphere with the current guess for the atmospheric FWHM.
2. Draw 6 stamps on axis and measure the median FWHM.
3. Scale the guess by the target over the measured FWHM, and repeat until they agree to
   2% (at most 4 passes).

The FWHM depends only on the turbulence the pupil sees, not on the size of the screen.
So calibration uses a small 204.8 m screen with the same random seed and winds as the
final build. The full-size atmosphere is then built once.

## 3. Phase screens

Each layer is a periodic phase screen with 0.1 m sampling, well below r0/2. The screen
size is chosen so that no star's pupil footprint wraps around the screen during the
exposure:

    size >= 2 * theta_max * h_top + v_max * t_exp + D

- `theta_max` is the largest field angle drawn, about 1.75 deg for the full focal plane.
- `h_top` is the highest layer altitude.
- `v_max` is the fastest layer wind.
- `t_exp` is the exposure time.
- `D` = 8.36 m is the telescope diameter.

This is rounded up to a power of two in pixels and capped at 1638.4 m (16384 pixels,
about 13 GB for 6 layers). A fast jet-stream layer can still exceed the cap. The run then
reports the wrap. The imSim default of 819.2 m is too small when all guiders and the full
field share one atmosphere over a 30 s visit.

Stars at different field angles look through different but overlapping parts of each
screen. So the field pattern and the correlation between guiders are both physical.

## 4. Drawing a star

The aperture is the Rubin pupil: 8.36 m diameter, central obscuration 0.61
(dimensionless, inner over outer diameter). The wavelength is the band's effective
wavelength, for example 754 nm in i band.

The standard render is **photon shooting**, split in GalSim's usual way:
- the low spatial frequencies of the screens are ray-traced as a geometric deflection of
  each photon, so the image moves and distorts coherently over time;
- the high frequencies, above k_crit = 0.2/r0, are added as a static `SecondKick`
  convolution.

An FFT render of the full screen spectrum is also available, with no noise, as a check
on that split.

Each star is drawn at its field angle. The time window [t0, t0 + exposure] walks along
the frozen screens as the winds carry them past.

**Noise (guide stamps).**
- Photon shooting gives Poisson shot noise in electrons, with gain 1.6 e-/ADU.
- Gaussian read noise of 6 e- rms is added.
- The image is converted to ADU.
- Sky is negligible in 50 ms and is set to zero by default.

The total star flux per stamp can be set per guider from the observed flux.

## 5. Guide sensors

Each guide sensor sees one star. The stamp train runs t0 = 0, 0.2, 0.4 s and so on, with
a 50 ms exposure per stamp, for the length of the visit. All stamps and all 8 sensors
share one atmosphere, so the centroid motion between stamps and the correlation between
sensors come only from the screens moving.

From each stamp we measure an adaptive-moment centroid and second moments, using GalSim
HSM (`FindAdaptiveMom`). The visit is then split, as in the data pipeline
(`rubinwork.products.guider_moments`), into:

- **static** moments: the flux-weighted mean of the per-stamp moments;
- **motion** moments: the flux-weighted covariance of the centroids over the visit, with
  the noise floor subtracted, in arcsec².

The motion moments are the cleanest test of the atmosphere, because optics do not move
the centroid.

## 6. Science CCDs

In the same run and from the same atmosphere, one star is drawn at the centre of each of
the 189 science CCDs. Each is a single integration over the full visit exposure (30 s),
with 1e6 ADU of flux.

For each star we measure:
- the FWHM;
- the second moments and the ellipticity e1, e2;
- the higher moments, standardized as in the Rubin quick-look PSF plots: coma, trefoil
  and kurtosis, with kurtosis = 2 for a Gaussian.

We also measure the atmospheric astrometric shift of each star. This is its centroid
minus that of a no-atmosphere (Airy) reference star, about 10 mas rms (milliarcseconds)
for one 30 s visit.

With no optics, the simulated field is nearly uniform: on day_obs 20260706 seq 688, FWHM
is about 0.71 arcsec and ellipticity e about 0.02 (dimensionless). The difference from
the field pattern seen in data is the part due to the optics.

## 7. Geometry

Field angles and pixel scales come from the LSSTCam camera geometry in the Rubin
Science Pipelines, through the cameraGeom `PIXELS` and `FIELD_ANGLE` transforms. They
include the optical distortion. The pixel scale is about 0.2 arcsec/pix.

In the current code the guide stamps are drawn in the Data Visualization Coordinate
System (DVCS) orientation, the focal-plane view that the guider display tools use.

## 8. Planned changes to the output

The output is being changed so the simulation feeds the same code that runs on data:

- **Guide stamps as real `guider_raw` data.** Each stamp is drawn in the ROI the real
  visit read out (from its headers) and converted to the amplifier readout frame. The
  real headers are copied: ROI position, per-stamp times, pointing, rotator angle.
  The stamps are then written to a Butler collection. The guider moment pipeline,
  movies and star tracker then run on the simulation unchanged.
- **Science-CCD stars** go to one multi-HDU FITS file per visit. Their moments are
  measured in each CCD's pixel frame, with the same columns as the Rubin Consolidated
  Database (ConsDB) per-CCD PSF summary (`psf_sigma`, `psf_ixx`, `psf_iyy`, `psf_ixy`),
  so sim and data are read the same way.
- The pointing, rotator angle and time come from the real visit. The first comparison
  (seq 688) used a rotator angle of 0 deg against the real −13.1 deg. Its ERA5 time step,
  06:00 UTC, was the nearest to the 07:41 UTC visit; ERA5 time steps are 6 hours apart.

## 9. Not included

- Telescope optics: an imSim mock optical screen exists (`--doOpt`), but it does not
  represent the real telescope.
- Tracking and wind-shake telescope motion, dome seeing, and chromatic effects within
  the band.
- Detector effects beyond shot and read noise. The real guider bias structure is
  replaced by a flat pedestal.
