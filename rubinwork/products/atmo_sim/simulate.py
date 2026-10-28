"""Simulate one visit: guide stamps as ``guider_raw`` files and one star per science CCD.

Everything geometric comes from the real visit. The boresight, rotator angle and time
come from ``raw.visitInfo``. The guider regions of interest (ROIs), stamp times and stamp
count come from the real ``guider_raw`` headers. The guide-star positions and fluxes come
from the ``guider_moments`` star table. Field angles and the local pixel-to-sky Jacobian
come from the same initial `SkyWcs` that ``summit_utils`` builds, so the rotator angle
enters through the WCS.

Command line::

    python -m rubinwork.products.atmo_sim.simulate --day-obs 20260706 --seq-num 688

It writes the files only. `rubinwork.products.atmo_sim.write_manifest` then ingests the
guider files and writes the manifest.
"""

import argparse
import json
import pathlib
import sys
import time

import galsim
import numpy as np
import pandas as pd

from . import atmosphere, reader

__all__ = ["visit_inputs", "simulate_visit", "main"]

PER_STAMP_KEYS = ("TIMESYS", "STMPTIME", "STMPTMJD", "DATE-END", "MJD-END", "DAQSTAMP")
"""Per-stamp header keys copied from the real stamps."""

SATURATION_ADU = 65535
"""The guider readout is 16 bit."""

_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "COMMENT", "CHECKSUM", "DATASUM",
               "HAS_ARCHIVE", "HAS_MASK", "HAS_VARIANCE", "N_STAMPS", "STAMPCLS"}


def _tangent_offset_asec(boresight, sky):
    """Tangent-plane offset (arcsec, East and North) of `sky` from `boresight`."""
    off = boresight.getTangentPlaneOffset(sky)
    return off[0].asArcseconds(), off[1].asArcseconds()


def field_and_jacobian(wcs, boresight, x, y, step=1.0):
    """Field angle and local Jacobian of a CCD pixel position.

    Returns
    -------
    theta : `tuple` [`float`, `float`]
        Field angle (arcsec, East and North of the boresight).
    jac : `numpy.ndarray`
        2x2 ``[[du/dx, du/dy], [dv/dx, dv/dy]]`` (arcsec/pix), ``u`` East, ``v`` North.
    """
    import lsst.geom as geom

    def th(px, py):
        return np.array(_tangent_offset_asec(boresight, wcs.pixelToSky(geom.Point2D(px, py))))

    jac = np.column_stack([(th(x + step, y) - th(x - step, y)) / (2 * step),
                           (th(x, y + step) - th(x, y - step)) / (2 * step)])
    return tuple(th(x, y)), jac


def _galsim_wcs(jac):
    return galsim.JacobianWCS(jac[0, 0], jac[0, 1], jac[1, 0], jac[1, 1])


def visit_inputs(day_obs, seq_num, config):
    """Everything taken from the real visit.

    Returns
    -------
    inputs : `dict`
        ``visit_info``, ``guiders`` (one `dict` per guide sensor with its template
        ``Stamps``, ROI box, amplifier, star position, flux, pedestal, read noise and
        stamp times), ``science`` (detector list), ``target_fwhm`` (arcsec), ``t_open``
        (shutter-open MJD, TAI) and ``guider_moments_build``.
    """
    from astropy.stats import mad_std
    from lsst.afw.cameraGeom import DetectorType
    from lsst.daf.butler import Butler
    from lsst.obs.base import createInitialSkyWcsFromBoresight
    from lsst.obs.lsst import LsstCam

    from rubinwork import guider_roi
    from rubinwork.products import guider_moments

    camera = LsstCam.getCamera()
    data = config["data"]
    butler = Butler(data["repo"], collections=data["collections"])
    vi = butler.get("raw.visitInfo", instrument="LSSTCam", day_obs=day_obs,
                    seq_num=seq_num, detector=94)
    boresight, rot = vi.getBoresightRaDec(), vi.getBoresightRotAngle()

    gm_variant = data["guider_moments_variant"]
    stars = guider_moments.load_table("stars", day_obs, variant=gm_variant)
    stars = stars[stars.seqNum == seq_num]
    if stars.empty:
        raise RuntimeError(f"no guider_moments stars for {day_obs} seq {seq_num}")
    per_det = stars.groupby("detector").agg(
        xccd=("xccd_ref", "median"), yccd=("yccd_ref", "median"),
        flux=("flux", "median"), fwhm=("fwhm", "median"))

    guiders, t_open = [], None
    for det in camera:
        if det.getType() != DetectorType.GUIDER or det.getName() not in per_det.index:
            continue
        stamps = butler.get("guider_raw", instrument="LSSTCam", day_obs=day_obs,
                            seq_num=seq_num, detector=det.getId())
        md = stamps.metadata
        t_open = md.get("SHUTTER OPEN STARTTIME TAI MJD")
        raw = np.array([s.stamp_im.image.array for s in stamps])
        star = per_det.loc[det.getName()]
        wcs = createInitialSkyWcsFromBoresight(boresight, rot, det, flipX=False)
        theta, jac = field_and_jacobian(wcs, boresight, star.xccd, star.yccd)
        guiders.append(dict(
            detector=det, name=det.getName(), stamps=stamps,
            bbox=guider_roi.roi_ccd_bbox(md, camera), amp=guider_roi.amp_name(md.toDict()),
            xccd=float(star.xccd), yccd=float(star.yccd), flux=float(star.flux),
            theta=theta, jac=jac,
            pedestal=float(np.median(raw)),
            read_noise=float(mad_std(np.diff(raw, axis=0)) / np.sqrt(2.0)),
            t_end=[(s.metadata.get("STMPTMJD") - t_open) * 86400.0 for s in stamps],
            exptime=float(md.get("ROIMILLI")) / 1000.0))

    science = []
    for det in camera:
        if det.getType() != DetectorType.SCIENCE:
            continue
        c = det.getBBox().getCenter()
        wcs = createInitialSkyWcsFromBoresight(boresight, rot, det, flipX=False)
        theta, jac = field_and_jacobian(wcs, boresight, c.getX(), c.getY())
        science.append(dict(detector=det, name=det.getName(), xccd=c.getX(),
                            yccd=c.getY(), theta=theta, jac=jac))

    return dict(visit_info=vi, guiders=guiders, science=science, t_open=t_open,
                target_fwhm=float(np.median(per_det.fwhm)),
                guider_moments_build=f"{gm_variant}@nights/{int(day_obs)}")


def _write_guider_file(path, g, ccd_stamps):
    """One ``guider_raw`` file in the real layout: primary header, then one compressed
    image HDU per stamp carrying that stamp's time keys."""
    from astropy.io import fits

    from rubinwork import guider_roi

    md = g["stamps"].metadata
    primary = fits.Header()
    for key in md.names():
        if key in _STRUCTURAL:
            continue
        value = md.get(key)
        if value is None or isinstance(value, (list, tuple)):
            continue
        primary["HIERARCH " + key if len(key) > 8 or " " in key else key] = value
    primary["HAS_ARCHIVE"] = False
    primary["HAS_MASK"] = False
    primary["HAS_VARIANCE"] = False
    primary["N_STAMPS"] = len(ccd_stamps)
    primary["STAMPCLS"] = "lsst.meas.algorithms.Stamps"
    primary["ATMOSIM"] = (True, "pixels simulated by rubinwork atmo_sim")
    hdus = [fits.PrimaryHDU(header=primary)]
    for k, (stamp, ccd) in enumerate(zip(g["stamps"], ccd_stamps)):
        raw = guider_roi.ccd_to_amp(ccd, g["detector"], g["amp"])
        hdr = fits.Header()
        for key in PER_STAMP_KEYS:
            if stamp.metadata.exists(key):
                hdr[key] = stamp.metadata.get(key)
        hdr["EXTNAME"] = "IMAGE"
        hdr["EXTVER"] = k + 1
        pixels = np.clip(np.rint(raw), 0, SATURATION_ADU).astype(np.int32)
        hdus.append(fits.CompImageHDU(pixels, header=hdr, compression_type="RICE_1"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def simulate_visit(day_obs, seq_num, config, outdir, log=print, max_stamps=None):
    """Simulate one visit into `outdir`.

    Parameters
    ----------
    day_obs, seq_num : `int`
        The real visit.
    config : `dict`
        Variant entry from ``variants.yaml``.
    outdir : `pathlib.Path`
        Visit directory; written into.
    log : callable, optional
    max_stamps : `int`, optional
        Keep only the first `max_stamps` stamps per guider; for a smoke test.

    Returns
    -------
    summary : `dict`
        The atmosphere, calibration and timing record, also written as
        ``atmosphere.json``.
    """
    outdir = pathlib.Path(outdir)
    (outdir / "guider_raw").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    inp = visit_inputs(day_obs, seq_num, config)
    if max_stamps:
        for g in inp["guiders"]:
            g["t_end"] = g["t_end"][:max_stamps]
    vi = inp["visit_info"]
    alt = vi.getBoresightAzAlt().getLatitude().asDegrees()
    az = vi.getBoresightAzAlt().getLongitude().asDegrees()
    when = pd.Timestamp(vi.getDate().toPython(), tz="UTC")
    band = config["band"] if config.get("band") else None
    if band is None:
        from lsst.daf.butler import Butler

        butler = Butler(config["data"]["repo"])
        rec = next(iter(butler.query_dimension_records(
            "exposure", instrument="LSSTCam",
            where=f"exposure.day_obs={day_obs} and exposure.seq_num={seq_num}")))
        band = rec.physical_filter[0]
    wlen = atmosphere.WLEN_EFF[band]
    exptime_sci = float(vi.getExposureTime())

    atm_cfg = config["atmosphere"]
    seed = int(atm_cfg["seed"]) + int(day_obs) % 10000 * 10000 + int(seq_num)
    layers = atmosphere.psfws_layers(
        atm_cfg["forecast_file"], atm_cfg["data_dir"], when, alt, az,
        atm_cfg["nlayers"], seed)
    L0 = atmosphere.draw_L0(galsim.BaseDeviate(seed))
    log(f"{day_obs} seq {seq_num}: band {band}, alt {alt:.2f} deg, az {az:.2f} deg, "
        f"rotator {vi.getBoresightRotAngle().asDegrees():.2f} deg, ERA5 {layers['era5_time']}")

    gcfg = config["guider"]
    g0 = inp["guiders"][0]
    cadence = float(np.median(np.diff(g0["t_end"])))
    atm_fwhm, calib = atmosphere.calibrate_atm_fwhm(
        layers, inp["target_fwhm"], wlen, L0, atm_cfg["screen_scale_m"], atm_cfg["kcrit"],
        seed, _galsim_wcs(g0["jac"]), gcfg["calibration_stamps"], g0["exptime"], cadence,
        float(np.median([g["flux"] for g in inp["guiders"]])), gcfg["gain"])
    for a, m in calib:
        log(f"  FWHM calibration: atmospheric FWHM {a:.3f} arcsec -> on-axis stamp FWHM "
            f"{m:.3f} arcsec (target {inp['target_fwhm']:.3f} arcsec)")

    theta_max = max(np.hypot(*s["theta"]) for s in inp["guiders"] + inp["science"]) / 3600.0
    t_lo = min(min(g["t_end"]) - g["exptime"] for g in inp["guiders"])
    t_hi = max(max(max(g["t_end"]) for g in inp["guiders"]), exptime_sci)
    size_m, need_m = atmosphere.screen_size(layers, theta_max, t_hi - min(t_lo, 0.0),
                                            atm_cfg["screen_scale_m"],
                                            atm_cfg["screen_size_max_m"])
    log(f"  screens: {size_m:.1f} m built, {need_m:.0f} m needed to avoid wrapping"
        + ("" if size_m >= need_m else " (top layers wrap)"))
    atmo = atmosphere.make_atmo(layers, atm_fwhm, wlen, L0, size_m,
                                atm_cfg["screen_scale_m"], atm_cfg["kcrit"], seed)
    log(f"  r0_500 {atmo.r0_500:.3f} m, L0 {L0:.1f} m; built in "
        f"{time.time() - t_start:.0f} s")

    rng = galsim.BaseDeviate(seed + 1)
    truth = []
    for g in inp["guiders"]:
        bbox = g["bbox"]
        x0, y0 = bbox.getMinX(), bbox.getMinY()
        img = galsim.ImageF(bbox.getWidth(), bbox.getHeight(), wcs=_galsim_wcs(g["jac"]))
        center = galsim.PositionD(g["xccd"] - x0 + 1.0, g["yccd"] - y0 + 1.0)
        ccd_stamps = []
        for t_end in g["t_end"]:
            atmosphere.draw_star(atmo, g["theta"], t_end - g["exptime"], g["exptime"], img,
                                 center, g["flux"], gcfg["gain"], g["read_noise"], rng)
            ccd_stamps.append(img.array + g["pedestal"])
        name = f"MC_O_{day_obs}_{seq_num:06d}_{g['name']}_guider.fits"
        _write_guider_file(outdir / "guider_raw" / name, g, ccd_stamps)
        truth.append(dict(kind="guider", detector=g["detector"].getId(), det_name=g["name"],
                          xccd=g["xccd"], yccd=g["yccd"], theta_e_asec=g["theta"][0],
                          theta_n_asec=g["theta"][1], flux_adu=g["flux"],
                          pedestal_adu=g["pedestal"], read_noise_adu=g["read_noise"],
                          n_stamps=len(g["t_end"]), jac=g["jac"].ravel().tolist()))
        log(f"  {g['name']}: {len(g['t_end'])} stamps, {time.time() - t_start:.0f} s")

    scfg = config["science"]
    from astropy.io import fits

    n = scfg["stamp_pix"]
    hdus = [fits.PrimaryHDU(header=fits.Header([
        ("DAYOBS", int(day_obs)), ("SEQNUM", int(seq_num)), ("BAND", band),
        ("EXPTIME", exptime_sci, "s"), ("ATMOSIM", True)]))]
    for s in inp["science"]:
        img = galsim.ImageF(n, n, wcs=_galsim_wcs(s["jac"]))
        x0, y0 = int(np.floor(s["xccd"])) - n // 2, int(np.floor(s["yccd"])) - n // 2
        center = galsim.PositionD(s["xccd"] - x0 + 1.0, s["yccd"] - y0 + 1.0)
        atmosphere.draw_star(atmo, s["theta"], 0.0, exptime_sci, img, center,
                             float(scfg["flux_adu"]), float(scfg["gain"]),
                             float(scfg["read_noise_adu"]), rng)
        hdr = fits.Header([
            ("EXTNAME", s["name"]), ("DETECTOR", s["detector"].getId()),
            ("X0", x0, "CCD pixel x of array column 0"),
            ("Y0", y0, "CCD pixel y of array row 0"),
            ("XSTAR", s["xccd"], "star CCD pixel x"), ("YSTAR", s["yccd"], "star CCD pixel y"),
            ("THETA_E", s["theta"][0], "field angle East of boresight, arcsec"),
            ("THETA_N", s["theta"][1], "field angle North of boresight, arcsec")])
        hdus.append(fits.ImageHDU(img.array.astype(np.float32), header=hdr))
        truth.append(dict(kind="science", detector=s["detector"].getId(), det_name=s["name"],
                          xccd=s["xccd"], yccd=s["yccd"], theta_e_asec=s["theta"][0],
                          theta_n_asec=s["theta"][1], flux_adu=float(scfg["flux_adu"]),
                          pedestal_adu=0.0, read_noise_adu=float(scfg["read_noise_adu"]),
                          n_stamps=1, jac=s["jac"].ravel().tolist()))
    fits.HDUList(hdus).writeto(outdir / "science_stars.fits", overwrite=True)
    log(f"  {len(inp['science'])} science CCDs, {time.time() - t_start:.0f} s")

    pd.DataFrame(truth).to_parquet(outdir / "truth.parquet", index=False)
    summary = dict(
        day_obs=int(day_obs), seq_num=int(seq_num), band=band, wavelength_nm=wlen,
        alt_deg=alt, az_deg=az, rot_deg=vi.getBoresightRotAngle().asDegrees(),
        date=str(when), era5_time=layers["era5_time"], seed=seed,
        altitude_km=layers["altitude"], weights=layers["weights"],
        speed_mps=layers["speed"],
        direction_deg=[d / galsim.degrees for d in layers["direction"]],
        L0_m=L0, r0_500_m=atmo.r0_500, target_fwhm_asec=inp["target_fwhm"],
        atm_fwhm_asec=atm_fwhm, calibration=calib, screen_size_m=size_m,
        screen_need_m=need_m, science_exptime_s=exptime_sci, runtime_s=time.time() - t_start,
        guider_moments=inp["guider_moments_build"])
    with open(outdir / "atmosphere.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m rubinwork.products.atmo_sim.simulate",
                                description=__doc__.splitlines()[0])
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, required=True)
    p.add_argument("--variant", default=reader.PRODUCT_VARIANT)
    p.add_argument("--visit-root", default=None,
                   help="Write here instead of the catalog partition")
    p.add_argument("--max-stamps", type=int, default=None,
                   help="Smoke test: first N stamps per guider; needs --visit-root")
    args = p.parse_args(argv)
    if args.max_stamps and not args.visit_root:
        p.error("--max-stamps writes a partial visit; give --visit-root")
    outdir = (pathlib.Path(args.visit_root) if args.visit_root
              else reader.visit_dir(args.day_obs, args.seq_num, args.variant))
    simulate_visit(args.day_obs, args.seq_num, reader.variant_config(args.variant), outdir,
                   max_stamps=args.max_stamps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
