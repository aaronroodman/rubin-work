# Atmospheric PSF simulation for the Rubin guide sensors and science CCDs

> **Status:** current · **Last updated:** 2026-10-28 · **Kind:** reference (method)

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

The code is the `atmo_sim` product (`rubinwork/products/atmo_sim/`, `atmosphere.py` and
`simulate.py`). It is built on GalSim (version 2.8) phase-screen PSFs. The August version,
`code/guider_atmo_sim.py`, also has the imSim profile and an FFT render.

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
the fetched file has 00, 06 and 18 UTC. psfws gives the layer altitudes, the turbulence
weights and the winds. With `skycoord=True` and the telescope altitude and azimuth, it
rotates the winds into the sky frame (East, North) as the telescope sees it. Layer
distances are line of sight, above the site, with the ground layer raised to 0.2 km.

psfws draws the split of turbulence between the ground layer and the free atmosphere at
random, so the seed matters. For 20260706 seq 688 the ground-layer weight ranges from
0.05 to 0.93 (fraction of total Cn²) over 9 seeds; v1 drew 0.93.

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

Each layer is a periodic phase screen with 0.2 m sampling. Photon shooting puts only
spatial frequencies below k_crit = 0.2/r0 (about 2 rad/m) in the screens, so 0.2 m is
ample. The screen size is chosen so that no star's pupil footprint wraps around the screen during the
exposure:

    size >= 2 * theta_max * h_top + v_max * t_exp + D

- `theta_max` is the largest field angle drawn, about 1.75 deg for the full focal plane.
- `h_top` is the highest layer altitude.
- `v_max` is the fastest layer wind.
- `t_exp` is the exposure time.
- `D` = 8.36 m is the telescope diameter.

This is rounded up to a power of two in pixels and capped at 3276.8 m (16384 pixels,
26 GB of memory for 6 layers). For seq 688, 2144 m is needed and 3276.8 m is built, so
no layer wraps. The imSim default of 819.2 m is too small when all guiders and the full
field share one atmosphere over a 30 s visit. One visit takes about 11 min on one core.

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

Each star is drawn at its field angle: the tangent-plane offset from the boresight,
in arcsec East and North, from the visit's initial `SkyWcs` (the one `summit_utils`
builds from the boresight and rotator angle). The image WCS is the local Jacobian of
that `SkyWcs` at the star, so a stamp comes out directly in the CCD pixel frame. The
time window [t0, t0 + exposure] walks along the frozen screens as the winds carry them
past.

**Noise (guide stamps).**
- Photon shooting gives Poisson shot noise in electrons, with gain 1.6 e-/ADU.
- Gaussian read noise is added at each sensor's measured level, the robust rms of
  stamp-to-stamp differences over √2: 4.2 to 7.3 ADU for seq 688.
- The image is converted to ADU and the sensor's real median pedestal is added
  (13378 to 15443 ADU). Pixels are rounded and clipped at 65535 ADU.
- Sky is negligible in 50 ms and is left out.

## 5. Guide sensors

Each guide sensor sees one star, at the median CCD position and flux of the star that
`guider_moments` tracked in the real visit. Each stamp is drawn in the sensor's real
ROI, ending at that stamp's real `STMPTMJD` and lasting `ROIMILLI` (50 ms); seq 688 has
135 stamps at 0.222 s cadence. All stamps and all 8 sensors
share one atmosphere, so the centroid motion between stamps and the correlation between
sensors come only from the screens moving.

Each stamp is converted from the CCD frame to the amplifier readout frame
(`rubinwork.guider_roi.ccd_to_amp`). The 8 files are written in the real `guider_raw`
layout, with the real primary header and per-stamp time keys, and ingested into the
Butler. The `guider_moments` builder then runs on them unchanged (product
`atmo_sim_moments`) and splits the visit, as for data, into:

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

Both the guide stamps and the science stamps are drawn in their CCD pixel frame. The
guider code turns the guide stamps into the Data Visualization Coordinate System (DVCS)
view, as for data, so sim and data guider moments share a frame, Q1 and Q2 included.

## 8. Taken from the real visit

- Boresight, rotator angle, altitude, azimuth, time and exposure time: `raw.visitInfo`.
- Band: the exposure's `physical_filter`.
- Guider ROIs, stamp count, stamp times, pedestals and read noise: the real `guider_raw`.
- Guide-star positions and fluxes, and the target FWHM (median tracked FWHM over
  sensors): `guider_moments` v4.

## 9. Not included

- Telescope optics: an imSim mock optical screen exists (`--doOpt`), but it does not
  represent the real telescope.
- Tracking and wind-shake telescope motion, dome seeing, and chromatic effects within
  the band.
- Detector effects beyond shot and read noise. The real guider bias structure is
  replaced by a flat pedestal.
- Other stars in the guider ROIs; only the tracked star is drawn.
