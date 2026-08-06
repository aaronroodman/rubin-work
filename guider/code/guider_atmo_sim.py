#!/usr/bin/env python
"""
Rubin guide-sensor atmospheric-PSF stamp simulator.

Simulates one bright star per guide CCD, at a FIXED focal-plane location, imaged
in a train of 50 ms "stamps" at 5 Hz (150 stamps per 30 s visit), through a SINGLE
frozen-flow atmosphere per visit (imSim's validated AtmosphericPSF model).

For every stamp it draws a 200x200 ROI oriented in the detector (focal-plane) pixel
frame and measures the adaptive-moment centroid + second moments.

Outputs per visit:
  guider_stamps_visit{v:02d}.npy   float32 array [n_stamp, n_guider, ROI, ROI]
and one combined table:
  guider_moments.parquet           one row per (visit, stamp, guider)

Run in an environment with the LSST Science Pipelines + imSim + GalSim.
A self-test that uses only GalSim (no imSim / no lsst stack) is available via
`--selftest`, to exercise the drawing / moments / I/O plumbing.

  python guider_atmo_sim.py                 # full run (needs imSim + lsst.obs.lsst)
  python guider_atmo_sim.py --selftest      # plumbing check with pure GalSim
"""
import argparse
import numpy as np
import galsim

# --------------------------------------------------------------------------
# Observation / cadence configuration
# --------------------------------------------------------------------------
CFG = dict(
    band="r",                 # u g r i z y
    rawSeeing=1.0,            # 500 nm, zenith FWHM [arcsec]  -> "typical 1 arcsec"
    airmass=1.015,           # ~zenith (altitude ~80 deg)
    boresight_ra=60.0,       # deg
    boresight_dec=-30.0,     # deg

    n_visits=1,
    stamps_per_visit=150,    # 5 Hz x 30 s
    stamp_rate_hz=5.0,       # -> cadence 0.20 s between stamp starts
    exptime=0.05,            # 50 ms integration per stamp

    roi=200,                 # ROI size [pix]
    flux_photons=1.0e5,      # photons per stamp (bright guide star, ~50 ms)
    doOpt=False,             # add imSim field-dependent optical phase screen
    seed=57721,

    # Phase-screen sizing (see size_report()); screen_size auto-computed if None.
    screen_scale=0.10,       # [m]  (< r0/2; r0~0.15-0.2 m)
    screen_size=None,        # [m]  None -> auto (safe for full FoV over the visit)
)

# imSim atmosphere constants (from imsim.atmPSF._getAtmKwargs), used for sizing.
H_MAX_KM = 15.46             # highest turbulent layer
V_MAX = 20.0                 # max layer wind speed [m/s]
D_APER = 8.36                # pupil diameter [m]
WLEN_EFF = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)

GUIDE_DETS = ["R00_SG0", "R00_SG1", "R04_SG0", "R04_SG1",
              "R40_SG0", "R40_SG1", "R44_SG0", "R44_SG1"]


# --------------------------------------------------------------------------
# Phase-screen size check  (answers requirement #8)
# --------------------------------------------------------------------------
def required_screen_size(theta_max_deg, visit_time_s):
    """Screen size [m] so a single shared atmosphere gives every guide star a
    unique (non-wrapping) turbulence footprint over the whole visit, AND avoids
    aliasing between the most-separated guide stars at the highest layer.

    extent = (field span across the FoV at top layer)
             + (wind drift over the visit) + (pupil diameter)
    """
    theta_max = np.radians(theta_max_deg)
    field_span = 2.0 * theta_max * (H_MAX_KM * 1e3)   # opposite corners, top layer
    wind_drift = V_MAX * visit_time_s
    return field_span + wind_drift + D_APER


def next_fft_size(size_m, scale_m):
    """Round screen_size up so npix = size/scale is a power of two (fast FFTs)."""
    npix = size_m / scale_m
    npix2 = 1 << (int(np.ceil(np.log2(npix))))
    return npix2 * scale_m


def size_report(screen_size, theta_max_deg, visit_time_s, scale_m, n_layers=6):
    need = required_screen_size(theta_max_deg, visit_time_s)
    npix = int(round(screen_size / scale_m))
    gb = npix**2 * 8 / 1e9 * n_layers
    ok = screen_size >= need
    print("  [screen-size check / req #8]")
    print(f"    field span (2*theta_max*H_top) = "
          f"{2*np.radians(theta_max_deg)*H_MAX_KM*1e3:7.1f} m")
    print(f"    wind drift (V_max*t_visit)      = {V_MAX*visit_time_s:7.1f} m")
    print(f"    pupil diameter                  = {D_APER:7.1f} m")
    print(f"    -> required screen_size        >= {need:7.1f} m")
    print(f"    using screen_size               = {screen_size:7.1f} m  "
          f"({npix} pix, ~{gb:.1f} GB for {n_layers} layers)  "
          f"{'OK' if ok else '*** TOO SMALL ***'}")
    if not ok:
        print("    WARNING: footprints will wrap the periodic screen -> "
              "spurious correlations. Increase screen_size.")
    return ok


# --------------------------------------------------------------------------
# Guide-sensor geometry: field-angle centers + local focal-plane pixel WCS
# --------------------------------------------------------------------------
def get_guiders_afw(camera_name="LsstCam"):
    """Return list of (det_name, theta_x_asec, theta_y_asec, galsim_wcs) using the
    real LSST camera geometry. WCS maps stamp pixels -> field-angle arcsec in the
    detector's own pixel frame, so stamps come out in focal-plane orientation."""
    import lsst.afw.cameraGeom as cg
    import lsst.geom as geom
    from lsst.obs.lsst import LsstCam
    camera = LsstCam().getCamera()
    R2A = 206265.0
    out = []
    for name in GUIDE_DETS:
        det = camera[name]
        t = det.getTransform(cg.PIXELS, cg.FIELD_ANGLE)   # pixels -> radians
        c = det.getCenter(cg.PIXELS)
        fa0 = t.applyForward(c)
        fax = t.applyForward(geom.Point2D(c.getX() + 1.0, c.getY()))
        fay = t.applyForward(geom.Point2D(c.getX(), c.getY() + 1.0))
        dudx = (fax.getX() - fa0.getX()) * R2A
        dvdx = (fax.getY() - fa0.getY()) * R2A
        dudy = (fay.getX() - fa0.getX()) * R2A
        dvdy = (fay.getY() - fa0.getY()) * R2A
        wcs = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
        out.append((name, fa0.getX() * R2A, fa0.getY() * R2A, wcs))
    return out


def get_guiders_fallback(pixel_scale=0.2):
    """Approximate guide-sensor field positions when the lsst stack is absent.
    Guide CCDs sit in the four corner rafts at ~1.76 deg radius. Orientation is a
    plain pixel scale (no per-detector parity/rotation). Prefer get_guiders_afw."""
    R = 1.76 * 3600.0           # arcsec, approximate corner-raft radius
    d = 0.12 * 3600.0           # arcsec, split of the two guide CCDs within a raft
    wcs = galsim.PixelScale(pixel_scale)
    corners = {"R00": (-1, -1), "R04": (-1, +1), "R40": (+1, -1), "R44": (+1, +1)}
    out = []
    for raft, (sx, sy) in corners.items():
        cx, cy = sx * R / np.sqrt(2), sy * R / np.sqrt(2)
        out.append((f"{raft}_SG0", cx - sy * d / 2, cy + sx * d / 2, wcs))
        out.append((f"{raft}_SG1", cx + sy * d / 2, cy - sx * d / 2, wcs))
    return out


# --------------------------------------------------------------------------
# Atmosphere builders (return a small container with the pieces we draw with)
# --------------------------------------------------------------------------
class Atmo:
    def __init__(self, atm, aper, second_kick, opt, wlen_eff):
        self.atm = atm
        self.aper = aper
        self.second_kick = second_kick
        self.opt = opt
        self.wlen_eff = wlen_eff


def build_atmo_imsim(cfg, screen_size, rng):
    """Build imSim's AtmosphericPSF (the production model) and expose its pieces."""
    from imsim.atmPSF import AtmosphericPSF
    boresight = galsim.CelestialCoord(cfg["boresight_ra"] * galsim.degrees,
                                      cfg["boresight_dec"] * galsim.degrees)
    a = AtmosphericPSF(
        airmass=cfg["airmass"], rawSeeing=cfg["rawSeeing"], band=cfg["band"],
        boresight=boresight, rng=rng, t0=0.0, exptime=cfg["exptime"], kcrit=0.2,
        screen_size=screen_size, screen_scale=cfg["screen_scale"],
        doOpt=cfg["doOpt"],
    )
    return Atmo(a.atm, a.aper, a.second_kick, a.opt, a.wlen_eff)


def build_atmo_selftest(cfg, screen_size, rng):
    """Tiny pure-GalSim atmosphere (3 layers, small screen) for plumbing tests."""
    wlen = WLEN_EFF[cfg["band"]]
    target = cfg["rawSeeing"] * cfg["airmass"]**0.6 * (wlen / 500.0)**(-0.3)
    r0_500 = 0.976 * 0.5e-6 / (target / 206265.0) * (500e-9 / 500e-9)  # rough
    r0_500 = float(np.clip(r0_500, 0.05, 0.3))
    alt = [0.2, 5.16, 15.46]
    atm = galsim.Atmosphere(r0_500=r0_500, L0=[25.0] * 3,
                            speed=[3.0, 10.0, 20.0],
                            direction=[0 * galsim.degrees, 45 * galsim.degrees,
                                       90 * galsim.degrees],
                            altitude=alt, r0_weights=[0.7, 0.1, 0.2], rng=rng,
                            screen_size=screen_size, screen_scale=cfg["screen_scale"])
    aper = galsim.Aperture(diam=D_APER, obscuration=0.61, lam=wlen, screen_list=atm)
    r0 = r0_500 * (wlen / 500.0)**1.2
    atm.instantiate(kmax=0.2 / r0, check="phot")
    sk = galsim.SecondKick(wlen, r0, aper.diam, aper.obscuration, kcrit=0.2)
    return Atmo(atm, aper, sk, None, wlen)


# --------------------------------------------------------------------------
# Draw one stamp + measure moments
# --------------------------------------------------------------------------
def draw_stamp(atmo, theta_xy_asec, t0, cfg, wcs, rng):
    theta = (theta_xy_asec[0] * galsim.arcsec, theta_xy_asec[1] * galsim.arcsec)
    psf = atmo.atm.makePSF(atmo.wlen_eff, aper=atmo.aper, theta=theta,
                           t0=t0, exptime=cfg["exptime"], second_kick=False)
    comps = [psf, atmo.second_kick]
    if atmo.opt is not None:
        comps.append(atmo.opt.makePSF(atmo.wlen_eff, aper=atmo.aper, theta=theta,
                                      t0=t0, exptime=cfg["exptime"], second_kick=False))
    prof = galsim.Convolve(comps).withFlux(cfg["flux_photons"])
    roi = cfg["roi"]
    img = galsim.Image(roi, roi, wcs=wcs)
    prof.drawImage(image=img, center=img.true_center, method="phot", rng=rng)
    return img


def measure(img):
    """Adaptive-moment centroid offset (pix, from stamp center) + second moments."""
    scale = np.sqrt(img.wcs.pixelArea(image_pos=img.true_center))  # arcsec/pix
    try:
        m = galsim.hsm.FindAdaptiveMom(img)
        cx = m.moments_centroid.x - img.true_center.x
        cy = m.moments_centroid.y - img.true_center.y
        e1, e2 = m.observed_shape.e1, m.observed_shape.e2
        sig = m.moments_sigma
        T = 2.0 * sig**2 / np.sqrt(max(1e-9, 1.0 - e1**2 - e2**2))  # pix^2
        Ixx, Iyy, Ixy = T * (1 + e1) / 2, T * (1 - e1) / 2, T * e2 / 2
        return dict(flux=float(m.moments_amp), cen_x=cx, cen_y=cy,
                    cen_x_asec=cx * scale, cen_y_asec=cy * scale,
                    sigma_pix=sig, e1=e1, e2=e2, fwhm_asec=2.3548 * sig * scale,
                    Ixx=Ixx, Iyy=Iyy, Ixy=Ixy, T_asec2=T * scale**2, ok=True)
    except Exception as exc:                                   # noqa: BLE001
        return dict(flux=np.nan, cen_x=np.nan, cen_y=np.nan, cen_x_asec=np.nan,
                    cen_y_asec=np.nan, sigma_pix=np.nan, e1=np.nan, e2=np.nan,
                    fwhm_asec=np.nan, Ixx=np.nan, Iyy=np.nan, Ixy=np.nan,
                    T_asec2=np.nan, ok=False, err=str(exc))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(cfg, selftest=False, outdir="output/atmo_sim"):
    import os
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)

    cadence = 1.0 / cfg["stamp_rate_hz"]
    visit_time = cfg["stamps_per_visit"] * cadence

    if selftest:
        guiders = get_guiders_fallback()[:2]
        cfg = dict(cfg, stamps_per_visit=8, n_visits=1, screen_scale=0.1)
        build_atmo = build_atmo_selftest
    else:
        try:
            guiders = get_guiders_afw()
            build_atmo = build_atmo_imsim
        except Exception as exc:                               # noqa: BLE001
            print(f"lsst stack unavailable ({exc}); using approximate guider "
                  "positions + plain WCS. Install lsst.obs.lsst for exact FP frame.")
            guiders = get_guiders_fallback()
            build_atmo = build_atmo_imsim

    theta_max = max(np.hypot(gx, gy) for _, gx, gy, _ in guiders) / 3600.0  # deg
    ss = cfg["screen_size"]
    if ss is None:
        ss = next_fft_size(required_screen_size(theta_max, visit_time),
                           cfg["screen_scale"])

    print(f"band={cfg['band']}  rawSeeing={cfg['rawSeeing']}\"  airmass={cfg['airmass']}")
    print(f"{cfg['stamps_per_visit']} stamps @ {cfg['stamp_rate_hz']} Hz "
          f"(cadence {cadence*1e3:.0f} ms, exptime {cfg['exptime']*1e3:.0f} ms, "
          f"visit {visit_time:.1f} s)  x {cfg['n_visits']} visit(s)")
    print(f"guide sensors: {[g[0] for g in guiders]}")
    size_report(ss, theta_max, visit_time, cfg["screen_scale"],
                n_layers=(3 if selftest else 6))

    rows, ng, roi = [], len(guiders), cfg["roi"]
    for v in range(cfg["n_visits"]):
        print(f"\n=== visit {v}: building atmosphere (seed {cfg['seed']+v}) ===")
        rng = galsim.BaseDeviate(cfg["seed"] + v)
        atmo = build_atmo(cfg, ss, rng)
        stamps = np.empty((cfg["stamps_per_visit"], ng, roi, roi), dtype=np.float32)
        for k in range(cfg["stamps_per_visit"]):
            t0 = k * cadence
            for gi, (name, gx, gy, wcs) in enumerate(guiders):
                img = draw_stamp(atmo, (gx, gy), t0, cfg, wcs, rng)
                stamps[k, gi] = img.array.astype(np.float32)
                m = measure(img)
                m.update(visit=v, stamp=k, t0=t0, guider=name,
                         theta_x_asec=gx, theta_y_asec=gy)
                rows.append(m)
            if (k + 1) % 25 == 0 or k == 0:
                print(f"  stamp {k+1}/{cfg['stamps_per_visit']}  (t0={t0:.2f}s)")
        out = os.path.join(outdir, f"guider_atmo_stamps_visit{v:02d}.npy")
        np.save(out, stamps)
        print(f"  wrote {out}  shape={stamps.shape}  ({stamps.nbytes/1e6:.0f} MB)")

    df = pd.DataFrame(rows)
    cols = ["visit", "stamp", "t0", "guider", "theta_x_asec", "theta_y_asec",
            "cen_x", "cen_y", "cen_x_asec", "cen_y_asec", "sigma_pix", "fwhm_asec",
            "e1", "e2", "Ixx", "Iyy", "Ixy", "T_asec2", "flux", "ok"]
    df = df[[c for c in cols if c in df.columns]]
    pq = os.path.join(outdir, "guider_atmo_moments.parquet")
    df.to_parquet(pq, index=False)
    print(f"\nwrote {pq}  ({len(df)} rows)")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/atmo_sim",
                    help="Directory for stamp .npy + moments parquet.")
    ap.add_argument("--n-visits", type=int, help="Override CFG n_visits.")
    ap.add_argument("--doOpt", action="store_true",
                    help="Add imSim field-dependent optical phase screen.")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-GalSim plumbing test (no imSim / no lsst stack)")
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.n_visits is not None:
        cfg["n_visits"] = args.n_visits
    if args.doOpt:
        cfg["doOpt"] = True
    main(cfg, selftest=args.selftest, outdir=args.outdir)
