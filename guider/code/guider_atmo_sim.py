#!/usr/bin/env python
"""
Rubin guide-sensor atmospheric-PSF simulator.

Two modes:
  * guiders (default): one bright star per guide CCD at a FIXED focal-plane
    location, imaged in a train of 50 ms "stamps" at 5 Hz (150/30 s visit),
    through a SINGLE frozen-flow atmosphere per visit. Per stamp it draws a
    200x200 ROI (focal-plane oriented) and measures centroid + 2nd moments.
  * fov: one star per SCIENCE CCD across the full field of view, a single
    long exposure, measuring FWHM / ellipticity / 3rd-order moments -> a
    focal-plane map to compare against typical on-sky results.

Atmosphere layer profile (atmo_source):
  * imsim : the imSim production profile (Ellerbroek layers), replicated inline.
  * psfws : realistic winds/turbulence from psf-weather-station (Hebert et al.),
            Cerro Pachon telemetry+reanalysis. Bundled data is 2019-05..10, so
            --psfws-month 7 samples the JULY (Chilean winter) climatology.
In both cases the absolute seeing is pinned to rawSeeing via the same von Karman
r0 solve; psfws sets only the relative layer weights, altitudes, and winds.

Rendering (render): photon shooting (geometric + second kick) or fft (full
phase-screen spectrum). Running both is an approximation cross-check.

Companion scripts: guider_atmo_stripcharts.py (centroid/moment time series),
guider_atmo_movie.py (summit_utils-style stamp movie), guider_atmo_fovmap.py
(focal-plane maps).

  python guider_atmo_sim.py                                  # guiders, imsim, phot
  python guider_atmo_sim.py --atmo-source psfws --psfws-month 7
  python guider_atmo_sim.py --mode fov --outdir output/fov
  python guider_atmo_sim.py --render fft
  python guider_atmo_sim.py --selftest                       # pure-GalSim plumbing
"""
import argparse
import numpy as np
import galsim
from scipy.optimize import bisect

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CFG = dict(
    band="r",                # u g r i z y
    rawSeeing=1.0,           # 500 nm, zenith FWHM [arcsec] -> "typical 1 arcsec"
    airmass=1.015,           # fallback if no pointing is resolved (see obs_time/alt/az)
    boresight_ra=60.0,       # deg
    boresight_dec=-30.0,     # deg

    # Telescope pointing -> alt/az (for psfws wind->sky rotation) and airmass.
    # Priority: explicit alt+az; else obs_time+boresight via astropy; else airmass.
    obs_time=None,           # UTC "YYYY-MM-DD HH:MM" (defaults to psfws_date)
    alt=None, az=None,       # explicit pointing [deg]

    atmo_source="imsim",     # "imsim" | "psfws"
    render="phot",           # "phot" | "fft"
    nlayers=6,               # layers (psfws) / fixed at 6 for imsim
    kcrit=0.2,               # 1/r0 split between geometric kick and SecondKick

    # psfws (psf-weather-station) options
    psfws_month=7,           # restrict datapoints to this month (None=all)
    psfws_location="com",    # layer centroid: "com" | "mean"
    psfws_forecast_file=None,   # None=bundled 2019; else ECMWF pickle from fetch_cp_weather.py
    psfws_telemetry_file="auto",  # "auto"|None|filename; auto=None when custom forecast set
    psfws_data_dir=None,     # dir holding the forecast/telemetry pickles (None=psfws pkg data)
    psfws_date=None,         # UTC "YYYY-MM-DD HH:MM" to pick nearest datapoint (overrides month)

    # guiders mode
    n_visits=1,
    stamps_per_visit=150,    # 5 Hz x 30 s
    stamp_rate_hz=5.0,       # -> 0.20 s between stamp starts
    exptime=0.05,            # 50 ms integration per stamp
    roi=200,                 # ROI size [pix]
    flux_counts=2.0e5,       # total star flux per stamp [ADU counts]

    # detector noise (guiders): shot noise via photon shooting in electrons,
    # then Gaussian read noise; gain converts counts<->electrons.
    gain=1.6,                # e-/count (guider gain, for shot-noise scaling)
    read_noise=6.0,          # read noise [e- rms] (ITL guider ~5-7)
    sky_counts=0.0,          # sky background [ADU/pixel] (negligible in 50 ms)

    # delivered-FWHM matching (item 3): if set, calibrate the atmosphere so the
    # median per-stamp FWHM equals this [arcsec] (matches observed fwhm_med).
    target_fwhm=None,
    fwhm_cal=True,           # auto-calibrate when target_fwhm is set
    fwhm_cal_stamps=6,       # stamps used per calibration measurement

    # fov mode
    fov_dets="all",          # "science" | "guiders" | "all" (science + 8 guiders)
    fov_exptime=30.0,        # single long exposure per CCD [s]
    fov_roi=64,              # ROI per CCD [pix]
    fov_flux_counts=1.0e6,   # total star flux per CCD [ADU counts]

    day_obs=0, seq_num=0,    # labels only (for the GuiderData adapter / movie titles)
    doOpt=False,             # add imSim field-dependent optical phase screen
    seed=57721,

    screen_scale=0.10,       # [m] (< r0/2; r0~0.15-0.2 m)
    screen_size=None,        # [m] None -> auto from winds/FoV/exposure
    screen_size_max=1638.4,  # [m] cap on auto size (~13 GB at 0.1 m, 6 layers)
)

D_APER = 8.36
OBSC = 0.61
PIXEL_SCALE = 0.2            # arcsec/pixel, nominal LSSTCam DVCS grid
WLEN_EFF = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)
CP_LAT, CP_LON, CP_HEIGHT = -30.2446, -70.7494, 2715.0   # Cerro Pachon (Rubin)
GUIDE_DETS = ["R00_SG0", "R00_SG1", "R04_SG0", "R04_SG1",
              "R40_SG0", "R40_SG1", "R44_SG0", "R44_SG1"]


# --------------------------------------------------------------------------
# Detector geometry: field-angle centers + local focal-plane pixel WCS
# --------------------------------------------------------------------------
def _det_entry(det):
    """(name, theta_x_asec, theta_y_asec, galsim_wcs) for one afw detector.

    Option A (draw directly in DVCS): the stamp axes are aligned to the camera
    FOCAL_PLANE frame -- which IS the DVCS view (summit_utils' np.rot90(-nQuarter)
    is exactly what aligns a CCD's pixels to FOCAL_PLANE). So the drawn stamps come
    out already in the real-data DVCS orientation and need no per-CCD remap; all
    downstream code (moments, movie) is identical to the on-sky path.

    The WCS maps stamp pixels -> field-angle arcsec (what makePSF's theta uses),
    with +x = FOCAL_PLANE/DVCS X, +y = DVCS Y, at PIXEL_SCALE arcsec/pix.
    """
    import lsst.afw.cameraGeom as cg
    import lsst.geom as geom
    R2A = 206265.0
    t = det.getTransform(cg.FOCAL_PLANE, cg.FIELD_ANGLE)   # mm -> radians
    c = det.getCenter(cg.FOCAL_PLANE)
    fa0 = t.applyForward(c)
    eps = 0.1                                              # mm step
    fax = t.applyForward(geom.Point2D(c.getX() + eps, c.getY()))
    fay = t.applyForward(geom.Point2D(c.getX(), c.getY() + eps))
    # field-angle directions [arcsec/mm] of the FOCAL_PLANE (DVCS) axes
    ex = np.array([fax.getX() - fa0.getX(), fax.getY() - fa0.getY()]) * R2A / eps
    ey = np.array([fay.getX() - fa0.getX(), fay.getY() - fa0.getY()]) * R2A / eps
    ex = ex / np.linalg.norm(ex)                          # unit DVCS-X in field angle
    ey = ey / np.linalg.norm(ey)                          # unit DVCS-Y in field angle
    s = PIXEL_SCALE
    wcs = galsim.JacobianWCS(s * ex[0], s * ey[0], s * ex[1], s * ey[1])
    return (det.getName(), fa0.getX() * R2A, fa0.getY() * R2A, wcs)


def get_guiders_afw(camera_name="LsstCam"):
    """The 8 guide CCDs, from the real LsstCam geometry."""
    from lsst.obs.lsst import LsstCam
    camera = LsstCam().getCamera()
    return [_det_entry(camera[name]) for name in GUIDE_DETS]


def get_science_afw(camera_name="LsstCam"):
    """All science CCDs (189), from the real LsstCam geometry."""
    import lsst.afw.cameraGeom as cg
    from lsst.obs.lsst import LsstCam
    camera = LsstCam().getCamera()
    dets = [d for d in camera if d.getType() == cg.DetectorType.SCIENCE]
    return [_det_entry(d) for d in dets]


def get_guiders_fallback(pixel_scale=0.2):
    """Approximate guide positions when the lsst stack is absent (plumbing only)."""
    R, d = 1.76 * 3600.0, 0.12 * 3600.0
    wcs = galsim.PixelScale(pixel_scale)
    corners = {"R00": (-1, -1), "R04": (-1, +1), "R40": (+1, -1), "R44": (+1, +1)}
    out = []
    for raft, (sx, sy) in corners.items():
        cx, cy = sx * R / np.sqrt(2), sy * R / np.sqrt(2)
        out.append((f"{raft}_SG0", cx - sy * d / 2, cy + sx * d / 2, wcs))
        out.append((f"{raft}_SG1", cx + sy * d / 2, cy - sx * d / 2, wcs))
    return out


# --------------------------------------------------------------------------
# Seeing / r0 solve  (von Karman, identical to imSim atmPSF + psfws)
# --------------------------------------------------------------------------
def resolve_pointing(cfg):
    """Return (alt_deg, az_deg, airmass, source). Priority: explicit alt+az;
    else obs_time (or psfws_date) + boresight via astropy; else airmass-derived."""
    if cfg.get("alt") is not None and cfg.get("az") is not None:
        alt, az = float(cfg["alt"]), float(cfg["az"])
        return alt, az, 1.0 / np.cos(np.radians(90.0 - alt)), "explicit"
    t = cfg.get("obs_time") or cfg.get("psfws_date")
    if t is not None:
        import pandas as pd
        from astropy.coordinates import EarthLocation, AltAz, SkyCoord
        from astropy.time import Time
        import astropy.units as u
        site = EarthLocation(lat=CP_LAT * u.deg, lon=CP_LON * u.deg,
                             height=CP_HEIGHT * u.m)
        time = Time(pd.Timestamp(t, tz="UTC").to_pydatetime())
        coord = SkyCoord(cfg["boresight_ra"] * u.deg, cfg["boresight_dec"] * u.deg)
        aa = coord.transform_to(AltAz(obstime=time, location=site))
        alt, az, am = aa.alt.deg, aa.az.deg, float(aa.secz)
        if alt <= 0.0:
            raise ValueError(
                f"boresight_ra/dec ({cfg['boresight_ra']},{cfg['boresight_dec']}) is "
                f"below the horizon (alt={alt:.1f} deg) at {t}. Set a boresight that is "
                f"up at that UTC time, or override with --alt/--az.")
        if alt < 20.0:
            print(f"  WARNING: boresight at alt={alt:.1f} deg (airmass {am:.2f}) at {t} "
                  f"-- high airmass; check boresight_ra/dec + obs_time.")
        return alt, az, am, f"astropy@{t}"
    alt = 90.0 - np.degrees(np.arccos(1.0 / cfg["airmass"]))
    return alt, (cfg.get("az") or 0.0), cfg["airmass"], "airmass"


def target_fwhm(cfg):
    # Calibrated atmospheric von Karman FWHM override (set by FWHM matching), else
    # the rawSeeing -> delivered scaling.
    if cfg.get("_atm_fwhm") is not None:
        return cfg["_atm_fwhm"]
    wlen = WLEN_EFF[cfg["band"]]
    return cfg["rawSeeing"] * cfg["airmass"]**0.6 * (wlen / 500.0)**(-0.3)


def _vk_seeing(r0_500, wavelength, L0):
    # Tokovinin 2002 eqn 19 fitting formula for von Karman FWHM.
    kolm = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength / 500.0)**(6.0 / 5.0)
    arg = 1.0 - 2.183 * (r0 / L0)**0.356
    return kolm * (np.sqrt(arg) if arg > 0.0 else 0.0)


def _r0_500_for_target(wavelength, L0, target):
    hi = min(1.0, L0 * (1.0 / 2.183)**(-0.356) * (wavelength / 500.0)**(6.0 / 5.0))
    return bisect(lambda r: _vk_seeing(r, wavelength, L0) - target, 0.01, hi,
                  args=())


def _draw_L0(rng):
    gd = galsim.GaussianDeviate(rng)
    L0 = 0.0
    while L0 < 10.0 or L0 > 100.0:
        L0 = np.exp(gd() * 0.6 + np.log(25.0))
    return L0


# --------------------------------------------------------------------------
# Layer-parameter sources -> dict(speed, direction, altitude, weights, L0, r0_500, wlen_eff)
# --------------------------------------------------------------------------
def imsim_layer_params(cfg, rng):
    """imSim production profile (Ellerbroek), replicated from imsim.atmPSF._getAtmKwargs."""
    wlen = WLEN_EFF[cfg["band"]]
    ud, gd = galsim.UniformDeviate(rng), galsim.GaussianDeviate(rng)
    altitudes = [0.2, 2.58, 5.16, 7.73, 12.89, 15.46]       # km (ground elevated to 0.2)
    weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    weights = [np.abs(w * (1.0 + 0.1 * gd())) for w in weights]
    weights = np.clip(weights, 0.01, 0.8)
    weights = list(weights / np.sum(weights))
    L0 = _draw_L0(rng)
    r0_500 = _r0_500_for_target(wlen, L0, target_fwhm(cfg))
    speeds = [ud() * 20.0 for _ in range(6)]
    directions = [ud() * 360.0 * galsim.degrees for _ in range(6)]
    return dict(speed=speeds, direction=directions, altitude=altitudes,
                weights=weights, L0=[L0] * 6, r0_500=r0_500, wlen_eff=wlen)


def _psfws_generator(cfg, rng):
    """Build a psfws.ParameterGenerator, optionally from a custom ECMWF forecast
    file (e.g. actual 2026 weather from fetch_cp_weather.py)."""
    import pathlib
    import psfws
    kw = dict(seed=int(rng.raw()) % (2**31))
    if cfg["psfws_forecast_file"]:
        kw["forecast_file"] = cfg["psfws_forecast_file"]
    tel = cfg["psfws_telemetry_file"]
    if tel == "auto":
        # bundled telemetry only matches the bundled 2019 forecast; a custom
        # forecast has no matching telemetry, so run forecast-only.
        if cfg["psfws_forecast_file"]:
            kw["telemetry_file"] = None
    else:
        kw["telemetry_file"] = tel
    if cfg["psfws_data_dir"]:
        kw["data_dir"] = pathlib.Path(cfg["psfws_data_dir"])
    pg = psfws.ParameterGenerator(**kw)
    # ERA5/ERA5T (expver 1 vs 5) overlap duplicates timestamps in recent-date
    # forecast files; keep one row per timestamp so get_parameters is well-posed.
    if pg.data_fa.index.has_duplicates:
        dup = pg.data_fa.index.duplicated(keep="first")
        n = int(dup.sum())
        pg.data_fa = pg.data_fa[~dup]
        if getattr(pg, "data_gl", None) is not None:
            pg.data_gl = pg.data_gl[~pg.data_gl.index.duplicated(keep="first")]
        pg.N = len(pg.data_fa)
        print(f"  psfws: dropped {n} duplicate timestamps (ERA5/ERA5T overlap)")
    return pg


def _psfws_pick(pg, cfg, rng):
    """Select a datapoint: nearest to psfws_date, else random within psfws_month."""
    import numpy as _np
    import pandas as pd
    idx = pg.data_fa.index
    if cfg["psfws_date"] is not None:
        target = pd.Timestamp(cfg["psfws_date"], tz="UTC")
        return idx[int(_np.argmin(_np.abs(idx - target)))]
    if cfg["psfws_month"] is not None:
        idx = idx[idx.month == cfg["psfws_month"]]
        if len(idx) == 0:
            raise ValueError(f"no psfws datapoints in month {cfg['psfws_month']}")
    return idx[int(galsim.UniformDeviate(rng)() * len(idx))]


def psfws_layer_params(cfg, rng):
    """Realistic winds/turbulence from psf-weather-station (Cerro Pachon)."""
    wlen = WLEN_EFF[cfg["band"]]
    nl = cfg["nlayers"]
    pg = _psfws_generator(cfg, rng)
    pt = _psfws_pick(pg, cfg, rng)
    # Use the resolved telescope pointing so winds rotate into the correct sky frame.
    p = pg.get_parameters(pt, nl=nl, skycoord=True, alt=cfg["_alt"], az=cfg["_az"],
                          location=cfg["psfws_location"])
    altitudes = [float(h - pg.h0 + 0.2) for h in p["h"]]     # km above ground, GL at 0.2
    directions = [float(a) * galsim.degrees for a in p["phi"]]
    L0 = _draw_L0(rng)
    r0_500 = _r0_500_for_target(wlen, L0, target_fwhm(cfg))
    return dict(speed=list(np.asarray(p["speed"], float)), direction=directions,
                altitude=altitudes, weights=list(np.asarray(p["j"], float)),
                L0=[L0] * nl, r0_500=r0_500, wlen_eff=wlen, _pt=str(pt))


def selftest_layer_params(cfg, rng):
    """Tiny 3-layer profile for pure-GalSim plumbing tests."""
    wlen = WLEN_EFF[cfg["band"]]
    L0 = 25.0
    r0_500 = _r0_500_for_target(wlen, L0, target_fwhm(cfg))
    return dict(speed=[3.0, 10.0, 20.0],
                direction=[0 * galsim.degrees, 45 * galsim.degrees, 90 * galsim.degrees],
                altitude=[0.2, 5.16, 15.46], weights=[0.7, 0.1, 0.2],
                L0=[L0] * 3, r0_500=r0_500, wlen_eff=wlen)


LAYER_SOURCES = dict(imsim=imsim_layer_params, psfws=psfws_layer_params,
                     selftest=selftest_layer_params)


# --------------------------------------------------------------------------
# Screen sizing
# --------------------------------------------------------------------------
def next_fft_size(size_m, scale_m):
    npix = 1 << int(np.ceil(np.log2(size_m / scale_m)))
    return npix * scale_m


def choose_screen_size(params, theta_max_deg, exposure_s, cfg):
    """Pick screen_size [m] to keep footprints on one non-wrapping screen:
    field span (top layer) + wind drift over the exposure + pupil. Capped for memory."""
    if cfg["screen_size"] is not None:
        return float(cfg["screen_size"]), None
    h_max = max(params["altitude"]) * 1e3
    v_max = max(params["speed"])
    need = 2.0 * np.radians(theta_max_deg) * h_max + v_max * exposure_s + D_APER
    ss = next_fft_size(min(need, cfg["screen_size_max"]), cfg["screen_scale"])
    return ss, need


def report_atmo(params, screen_size, need_m, scale_m):
    npix = int(round(screen_size / scale_m))
    gb = npix**2 * 8 / 1e9 * len(params["speed"])
    print(f"  layers={len(params['speed'])}  r0_500={params['r0_500']:.3f} m  "
          f"L0={params['L0'][0]:.1f} m")
    print(f"  altitudes[km]={np.round(params['altitude'],2).tolist()}")
    print(f"  speeds[m/s]  ={np.round(params['speed'],1).tolist()}")
    print(f"  weights      ={np.round(np.asarray(params['weights'])/np.sum(params['weights']),3).tolist()}")
    if params.get("_pt"):
        print(f"  psfws datapoint: {params['_pt']}")
    msg = f"  screen_size={screen_size:.1f} m ({npix} pix, ~{gb:.1f} GB)"
    if need_m is not None:
        ok = screen_size >= need_m
        msg += f"  required>={need_m:.0f} m  {'OK' if ok else '*** WRAPS (jet layer) ***'}"
    print(msg)


# --------------------------------------------------------------------------
# Atmosphere build + stamp draw
# --------------------------------------------------------------------------
class Atmo:
    def __init__(self, atm, aper, second_kick, opt, wlen_eff, render):
        self.atm, self.aper = atm, aper
        self.second_kick, self.opt = second_kick, opt
        self.wlen_eff, self.render = wlen_eff, render


def make_atmo(params, screen_size, cfg, rng):
    wlen, render = params["wlen_eff"], cfg["render"]
    atm = galsim.Atmosphere(
        r0_500=params["r0_500"], L0=params["L0"], speed=params["speed"],
        direction=params["direction"], altitude=params["altitude"],
        r0_weights=params["weights"], screen_size=screen_size,
        screen_scale=cfg["screen_scale"], rng=rng)
    aper = galsim.Aperture(diam=D_APER, obscuration=OBSC, lam=wlen, screen_list=atm)
    r0 = atm.r0_500_effective * (wlen / 500.0)**(6.0 / 5.0)
    if render == "fft":
        atm.instantiate()                              # full spectrum, FFT-ready
        second_kick = None
    else:
        atm.instantiate(kmax=cfg["kcrit"] / r0, check="phot")
        second_kick = galsim.SecondKick(wlen, r0, aper.diam, aper.obscuration,
                                        kcrit=cfg["kcrit"])
    opt = None
    if cfg["doOpt"]:
        from imsim.atmPSF import OptWF
        opt = galsim.PhaseScreenList(OptWF(rng, wlen))
    return Atmo(atm, aper, second_kick, opt, wlen, render)


def draw_stamp(atmo, theta_xy_asec, t0, exptime, roi, wcs, rng, flux_counts, cfg):
    """Draw one stamp in ADU counts with realistic noise. Flux is total star
    counts; shot noise is Poisson in electrons (flux*gain) via photon shooting,
    then Gaussian read noise (+ optional sky) is added, then converted to counts.
    The fft render is the noiseless PSF cross-check."""
    theta = (theta_xy_asec[0] * galsim.arcsec, theta_xy_asec[1] * galsim.arcsec)
    if atmo.render == "fft":
        psf = atmo.atm.makePSF(atmo.wlen_eff, aper=atmo.aper, theta=theta,
                               t0=t0, exptime=exptime)
        comps = [psf]
    else:
        psf = atmo.atm.makePSF(atmo.wlen_eff, aper=atmo.aper, theta=theta,
                               t0=t0, exptime=exptime, second_kick=False)
        comps = [psf, atmo.second_kick]
    if atmo.opt is not None:
        comps.append(atmo.opt.makePSF(atmo.wlen_eff, aper=atmo.aper, theta=theta,
                                      t0=t0, exptime=exptime, second_kick=False))
    gain = cfg["gain"]
    prof = galsim.Convolve(comps).withFlux(flux_counts * gain)   # electrons
    img = galsim.Image(roi, roi, wcs=wcs)
    if atmo.render == "fft":
        prof.drawImage(image=img, center=img.true_center, method="fft")  # noiseless e-
    else:
        # photon shooting -> Poisson shot noise in electrons
        prof.drawImage(image=img, center=img.true_center, method="phot",
                       rng=rng, poisson_flux=True)
        sky_e = cfg["sky_counts"] * gain
        if sky_e > 0:
            img += sky_e
        var = cfg["read_noise"]**2 + sky_e          # read noise + sky shot noise
        if var > 0:
            img.addNoise(galsim.GaussianNoise(rng, sigma=np.sqrt(var)))
    img /= gain                                     # electrons -> ADU counts
    return img


# --------------------------------------------------------------------------
# Moment measurement
# --------------------------------------------------------------------------
def measure(img):
    """Adaptive-moment centroid offset (pix, from center) + 2nd moments."""
    scale = np.sqrt(img.wcs.pixelArea(image_pos=img.true_center))
    try:
        m = galsim.hsm.FindAdaptiveMom(img)
        cx = m.moments_centroid.x - img.true_center.x
        cy = m.moments_centroid.y - img.true_center.y
        e1, e2, sig = m.observed_shape.e1, m.observed_shape.e2, m.moments_sigma
        T = 2.0 * sig**2 / np.sqrt(max(1e-9, 1.0 - e1**2 - e2**2))
        out = dict(flux=float(m.moments_amp), cen_x=cx, cen_y=cy,
                   cen_x_asec=cx * scale, cen_y_asec=cy * scale, sigma_pix=sig,
                   e1=e1, e2=e2, fwhm_asec=2.3548 * sig * scale,
                   Ixx=T * (1 + e1) / 2, Iyy=T * (1 - e1) / 2, Ixy=T * e2 / 2,
                   T_asec2=T * scale**2, ok=True)
        out["_cen_pix"] = (m.moments_centroid.x, m.moments_centroid.y)
        out["_sig_pix"] = sig
        out["_scale"] = scale
        return out
    except Exception as exc:                                   # noqa: BLE001
        return dict(flux=np.nan, cen_x=np.nan, cen_y=np.nan, cen_x_asec=np.nan,
                    cen_y_asec=np.nan, sigma_pix=np.nan, e1=np.nan, e2=np.nan,
                    fwhm_asec=np.nan, Ixx=np.nan, Iyy=np.nan, Ixy=np.nan,
                    T_asec2=np.nan, ok=False, err=str(exc))


def measure_third(img, m2):
    """Gaussian-weighted 3rd central moments [arcsec^3] about the adaptive centroid.
    Returns M30, M21, M12, M03 and the spin-1 / spin-3 combinations."""
    if not m2.get("ok"):
        return dict(M30=np.nan, M21=np.nan, M12=np.nan, M03=np.nan,
                    M_spin1_x=np.nan, M_spin1_y=np.nan,
                    M_spin3_x=np.nan, M_spin3_y=np.nan)
    arr = img.array.astype(float)
    ny, nx = arr.shape
    cx, cy = m2["_cen_pix"]
    sig, scale = m2["_sig_pix"], m2["_scale"]
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx + img.bounds.xmin - cx) * scale        # arcsec, centred
    y = (yy + img.bounds.ymin - cy) * scale
    w = np.exp(-(x**2 + y**2) / (2.0 * (sig * scale)**2))
    wi = w * np.clip(arr, 0, None)
    norm = wi.sum()
    if norm <= 0:
        return dict(M30=np.nan, M21=np.nan, M12=np.nan, M03=np.nan,
                    M_spin1_x=np.nan, M_spin1_y=np.nan,
                    M_spin3_x=np.nan, M_spin3_y=np.nan)
    M30 = (wi * x**3).sum() / norm
    M21 = (wi * x**2 * y).sum() / norm
    M12 = (wi * x * y**2).sum() / norm
    M03 = (wi * y**3).sum() / norm
    return dict(M30=M30, M21=M21, M12=M12, M03=M03,
                M_spin1_x=M30 + M12, M_spin1_y=M03 + M21,       # spin-1 (coma-like)
                M_spin3_x=M30 - 3 * M12, M_spin3_y=3 * M21 - M03)  # spin-3 (trefoil)


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def _resolve_geometry(cfg, selftest):
    """Return (geom_list, source_key). geom_list = [(name, tx, ty, wcs), ...]."""
    if selftest:
        return get_guiders_fallback()[:2], "selftest"
    try:
        if cfg["mode"] == "fov":
            which = cfg.get("fov_dets", "all")
            geom = []
            if which in ("science", "all"):
                geom += get_science_afw()
            if which in ("guiders", "all"):
                geom += get_guiders_afw()
        else:
            geom = get_guiders_afw()
    except Exception as exc:                                   # noqa: BLE001
        print(f"lsst stack unavailable ({exc}); using approximate guider geometry.")
        geom = get_guiders_fallback()
    return geom, cfg["atmo_source"]


def _median_fwhm(atmo, geom, cfg, rng, nstamps, exptime, roi, flux):
    """Median measured FWHM [arcsec] over a few (stamp, guider) draws."""
    cadence = 1.0 / cfg["stamp_rate_hz"]
    vals = []
    for k in range(nstamps):
        for (name, gx, gy, wcs) in geom:
            img = draw_stamp(atmo, (gx, gy), k * cadence, exptime, roi, wcs, rng, flux, cfg)
            m = measure(img)
            if m["ok"]:
                vals.append(m["fwhm_asec"])
    return float(np.median(vals)) if vals else float("nan")


def build_calibrated_atmo(cfg, source_key, geom, theta_max, exposure_s, seed):
    """Build the atmosphere; if target_fwhm is set, first calibrate the atmospheric
    von Karman FWHM so the median per-stamp delivered FWHM matches it. The FWHM is
    independent of screen size (the PSF samples only an aperture-sized footprint),
    so calibration uses a small cheap screen at theta=0 (same seed/winds as the
    final), and the full-size atmosphere is built ONCE at the end. Returns
    (atmo, cfg, rng, params, ss)."""
    import gc
    if cfg.get("target_fwhm") and cfg.get("fwhm_cal", True):
        target = cfg["target_fwhm"]
        cfg = dict(cfg, _atm_fwhm=target_fwhm(cfg))          # seed atmospheric target
        cal_geom = [("cal", 0.0, 0.0, geom[0][3])]           # on-axis, reuse a WCS
        for it in range(4):
            cal_cfg = dict(cfg, screen_size=204.8, screen_scale=0.1)
            rng = galsim.BaseDeviate(seed)                   # same winds as final
            params = LAYER_SOURCES[source_key](cal_cfg, rng)
            atmo = make_atmo(params, 204.8, cal_cfg, rng)
            m = _median_fwhm(atmo, cal_geom, cfg, galsim.BaseDeviate(seed + 7),
                             cfg["fwhm_cal_stamps"], cfg["exptime"], cfg["roi"],
                             cfg["flux_counts"])
            del atmo, params
            gc.collect()
            print(f"  [fwhm cal {it}] on-axis FWHM={m:.3f}\" "
                  f"(target {target:.3f}\", atm_fwhm={cfg['_atm_fwhm']:.3f}\")")
            if not np.isfinite(m) or abs(m - target) / target < 0.02:
                break
            cfg = dict(cfg, _atm_fwhm=cfg["_atm_fwhm"] * target / m)

    rng = galsim.BaseDeviate(seed)                           # final full-size build
    params = LAYER_SOURCES[source_key](cfg, rng)
    ss, need = choose_screen_size(params, theta_max, exposure_s, cfg)
    atmo = make_atmo(params, ss, cfg, rng)
    report_atmo(params, ss, need, cfg["screen_scale"])
    return atmo, cfg, rng, params, ss


def run_guiders(cfg, geom, source_key, outdir, os, pd):
    cadence = 1.0 / cfg["stamp_rate_hz"]
    visit_time = cfg["stamps_per_visit"] * cadence
    theta_max = max(np.hypot(gx, gy) for _, gx, gy, _ in geom) / 3600.0
    rows, ng, roi = [], len(geom), cfg["roi"]
    for v in range(cfg["n_visits"]):
        print(f"\n=== visit {v} ({source_key}, {cfg['render']}): building atmosphere ===")
        atmo, cfgv, rng, params, ss = build_calibrated_atmo(
            cfg, source_key, geom, theta_max, visit_time, cfg["seed"] + v)
        fmap = cfgv.get("_flux_map") or {}   # optional per-guider observed counts
        stamps = np.empty((cfg["stamps_per_visit"], ng, roi, roi), dtype=np.float32)
        for k in range(cfg["stamps_per_visit"]):
            t0 = k * cadence
            for gi, (name, gx, gy, wcs) in enumerate(geom):
                flux = fmap.get(name, cfgv["flux_counts"])
                img = draw_stamp(atmo, (gx, gy), t0, cfgv["exptime"], roi, wcs, rng,
                                 flux, cfgv)
                stamps[k, gi] = img.array.astype(np.float32)
                m = measure(img)
                m = {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
                m.update(visit=v, stamp=k, t0=t0, guider=name,
                         theta_x_asec=gx, theta_y_asec=gy)
                rows.append(m)
            if (k + 1) % 25 == 0 or k == 0:
                print(f"  stamp {k+1}/{cfg['stamps_per_visit']} (t0={t0:.2f}s)")
        out = os.path.join(outdir, f"guider_atmo_stamps_visit{v:02d}.npy")
        np.save(out, stamps)
        print(f"  wrote {out}  {stamps.shape}  ({stamps.nbytes/1e6:.0f} MB)")
    df = pd.DataFrame(rows)
    cols = ["visit", "stamp", "t0", "guider", "theta_x_asec", "theta_y_asec",
            "cen_x", "cen_y", "cen_x_asec", "cen_y_asec", "sigma_pix", "fwhm_asec",
            "e1", "e2", "Ixx", "Iyy", "Ixy", "T_asec2", "flux", "ok"]
    df = df[[c for c in cols if c in df.columns]]
    pq = os.path.join(outdir, "guider_atmo_moments.parquet")
    df.to_parquet(pq, index=False)
    med_fwhm = float(df.loc[df.ok, "fwhm_asec"].median())
    # Sidecar metadata for the GuiderData adapter / movie (alt/az/cadence/etc.)
    import json
    meta = dict(dayObs=int(cfg["day_obs"]), seqNum=int(cfg["seq_num"]),
                alt=float(cfg["_alt"]), az=float(cfg["_az"]),
                airmass=float(cfg["airmass"]), camRotAngle=0.0,
                cadence=float(cadence), exptime=float(cfg["exptime"]),
                band=cfg["band"], source=source_key,
                psfws_date=cfg.get("psfws_date"), roi=int(roi),
                pixel_scale=PIXEL_SCALE, gain=float(cfg["gain"]),
                read_noise=float(cfg["read_noise"]), flux_counts=float(cfg["flux_counts"]),
                target_fwhm=cfg.get("target_fwhm"), median_fwhm=med_fwhm,
                guider_order=[g[0] for g in geom])
    with open(os.path.join(outdir, "guider_atmo_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {pq}  ({len(df)} rows) + guider_atmo_meta.json")
    tgt = f" (target {cfg['target_fwhm']:.3f}\")" if cfg.get("target_fwhm") else ""
    print(f"delivered median per-stamp FWHM = {med_fwhm:.3f}\"{tgt}")
    print(df.head(6).to_string(index=False))


def run_fov(cfg, geom, source_key, outdir, os, pd):
    theta_max = max(np.hypot(gx, gy) for _, gx, gy, _ in geom) / 3600.0
    roi, exptime = cfg["fov_roi"], cfg["fov_exptime"]
    rows = []
    # long exposure -> delivered FWHM ~ atmospheric von Karman FWHM; set it directly
    cfgf = dict(cfg, _atm_fwhm=cfg["target_fwhm"]) if cfg.get("target_fwhm") else cfg
    for v in range(cfg["n_visits"]):
        print(f"\n=== visit {v} ({source_key}, {cfg['render']}): FoV, "
              f"{len(geom)} CCDs, {exptime:.0f}s exposure ===")
        rng = galsim.BaseDeviate(cfgf["seed"] + v)
        params = LAYER_SOURCES[source_key](cfgf, rng)
        ss, need = choose_screen_size(params, theta_max, exptime, cfgf)
        report_atmo(params, ss, need, cfgf["screen_scale"])
        atmo = make_atmo(params, ss, cfgf, rng)
        for di, (name, gx, gy, wcs) in enumerate(geom):
            img = draw_stamp(atmo, (gx, gy), 0.0, exptime, roi, wcs, rng,
                             cfgf["fov_flux_counts"], cfgf)
            m2 = measure(img)
            m3 = measure_third(img, m2)
            row = {kk: vv for kk, vv in m2.items() if not kk.startswith("_")}
            row.update(m3)
            row.update(visit=v, det=name, field_x_asec=gx, field_y_asec=gy,
                       field_x_deg=gx / 3600.0, field_y_deg=gy / 3600.0)
            rows.append(row)
            if (di + 1) % 40 == 0 or di == 0:
                print(f"  ccd {di+1}/{len(geom)}  {name}  fwhm={row['fwhm_asec']:.3f}\"")
    df = pd.DataFrame(rows)
    lead = ["visit", "det", "field_x_asec", "field_y_asec", "field_x_deg",
            "field_y_deg", "fwhm_asec", "e1", "e2", "T_asec2",
            "M_spin1_x", "M_spin1_y", "M_spin3_x", "M_spin3_y",
            "M30", "M21", "M12", "M03", "flux", "ok"]
    df = df[[c for c in lead if c in df.columns]]
    pq = os.path.join(outdir, "guider_atmo_fov_moments.parquet")
    df.to_parquet(pq, index=False)
    print(f"\nwrote {pq}  ({len(df)} rows)")
    print(df[["det", "field_x_deg", "field_y_deg", "fwhm_asec", "e1", "e2"]]
          .head(8).to_string(index=False))


def main(cfg, selftest=False, outdir="output/atmo_sim"):
    import os
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    if selftest:
        cfg = dict(cfg, n_visits=1, stamps_per_visit=8, screen_size=102.4,
                   screen_scale=0.1, mode="guiders")
    # Resolve telescope pointing once; sets alt/az (psfws wind rotation) + airmass.
    alt, az, airmass, psrc = resolve_pointing(cfg)
    cfg = dict(cfg, _alt=alt, _az=az, airmass=airmass)
    geom, source_key = _resolve_geometry(cfg, selftest)
    print(f"band={cfg['band']} rawSeeing={cfg['rawSeeing']}\" "
          f"| pointing({psrc}): alt={alt:.1f} az={az:.1f} airmass={airmass:.3f}")
    print(f"mode={cfg['mode']} source={source_key} render={cfg['render']}")
    if cfg["mode"] == "fov":
        run_fov(cfg, geom, source_key, outdir, os, pd)
    else:
        run_guiders(cfg, geom, source_key, outdir, os, pd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/atmo_sim")
    ap.add_argument("--mode", choices=["guiders", "fov"], help="override CFG mode")
    ap.add_argument("--fov-dets", choices=["science", "guiders", "all"], dest="fov_dets",
                    help="fov mode: which CCDs get a star (default all = science + guiders)")
    ap.add_argument("--atmo-source", choices=["imsim", "psfws"], dest="atmo_source")
    ap.add_argument("--render", choices=["phot", "fft"])
    ap.add_argument("--psfws-month", type=int, dest="psfws_month",
                    help="restrict psfws to this month (e.g. 7=July); -1 for all")
    ap.add_argument("--psfws-forecast", dest="psfws_forecast_file",
                    help="ECMWF forecast pickle from fetch_cp_weather.py (actual weather)")
    ap.add_argument("--psfws-data-dir", dest="psfws_data_dir",
                    help="dir holding the forecast/telemetry pickles")
    ap.add_argument("--psfws-date", dest="psfws_date",
                    help="UTC 'YYYY-MM-DD HH:MM' to pick the nearest datapoint")
    ap.add_argument("--obs-time", dest="obs_time",
                    help="UTC 'YYYY-MM-DD HH:MM' for alt/az (default: --psfws-date)")
    ap.add_argument("--alt", type=float, help="explicit pointing altitude [deg]")
    ap.add_argument("--az", type=float, help="explicit pointing azimuth [deg]")
    ap.add_argument("--n-visits", type=int, dest="n_visits")
    ap.add_argument("--day-obs", type=int, dest="day_obs", help="label for outputs")
    ap.add_argument("--seq-num", type=int, dest="seq_num", help="label for outputs")
    ap.add_argument("--flux-counts", type=float, dest="flux_counts",
                    help="total star flux per stamp [ADU counts]")
    ap.add_argument("--flux-file", dest="flux_file",
                    help="JSON {detName: counts} of observed per-guider fluxes "
                         "(e.g. sum of each stacked-coadd stamp); overrides --flux-counts per CCD")
    ap.add_argument("--gain", type=float, help="guider gain [e-/count] (default 1.6)")
    ap.add_argument("--read-noise", type=float, dest="read_noise", help="read noise [e- rms]")
    ap.add_argument("--sky-counts", type=float, dest="sky_counts", help="sky [ADU/pix]")
    ap.add_argument("--target-fwhm", type=float, dest="target_fwhm",
                    help="match delivered median per-stamp FWHM to this [arcsec]")
    ap.add_argument("--no-fwhm-cal", action="store_true",
                    help="with --target-fwhm, set the atmospheric FWHM directly (no calibration)")
    ap.add_argument("--screen-size", type=float, dest="screen_size",
                    help="phase-screen size [m] (default: auto from winds/FoV)")
    ap.add_argument("--screen-scale", type=float, dest="screen_scale",
                    help="phase-screen pixel [m] (default 0.10 -> 16384px/12.9GB; "
                         "0.2 -> 8192px/3.2GB, same FoV coverage)")
    ap.add_argument("--doOpt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    cfg = dict(CFG)
    cfg.setdefault("mode", "guiders")
    for key in ("mode", "fov_dets", "atmo_source", "render", "n_visits",
                "psfws_forecast_file", "psfws_data_dir", "psfws_date",
                "obs_time", "alt", "az", "screen_size", "screen_scale",
                "day_obs", "seq_num", "flux_counts", "gain", "read_noise",
                "sky_counts", "target_fwhm"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_fwhm_cal:
        cfg["fwhm_cal"] = False
    if args.flux_file:
        import json
        with open(args.flux_file) as fh:
            cfg["_flux_map"] = json.load(fh)
    if args.psfws_month is not None:
        cfg["psfws_month"] = None if args.psfws_month < 0 else args.psfws_month
    if args.doOpt:
        cfg["doOpt"] = True
    main(cfg, selftest=args.selftest, outdir=args.outdir)
