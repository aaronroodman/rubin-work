"""Find straight cosmic-ray muon tracks in LSSTCam dark exposures.

For each visit and detector: run a minimal ISR (overscan, bias, defects, PTC gains;
no dark, flat, crosstalk, CTI, brighter-fatter or linearity), detect sources by a
hysteresis threshold, measure each one's line fit, length and peak, and reject hot
columns and anything near saturation. Writes, per visit:

- ``cosmics_<day_obs>_<seq>.fits``: one image extension per long straight track,
  in electrons, with defect pixels and other sources set to NaN; the header records
  the detector, raft, sensor, REB (readout electronics board), back-bias voltage
  and the track measurements;
- ``tracks_<day_obs>_<seq>.parquet``: every detected source with its measurements,
  for the energy-loss (dE/dx) spectrum.

Run from the repo root::

    python camera/cosmic_diffusion/code/find_cosmics.py --day-obs 20261102 --seq-num 533
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cosmic_tools import (PIXEL_PITCH_UM, THICKNESS_UM, line_fit, nmad,  # noqa: E402
                          project)

R21 = ["R21_S00", "R21_S01", "R21_S02", "R21_S10", "R21_S11", "R21_S12"]


def isr_config():
    """Minimal ISR: overscan, bias, defects and gains only."""
    from lsst.ip.isr import IsrTaskLSSTConfig
    cfg = IsrTaskLSSTConfig()
    for name in ["doBrighterFatter", "doDeferredCharge", "doCrosstalk", "doLinearize",
                 "doFlat", "doDark", "doITLDipMask", "doITLSatSagMask",
                 "doInterpolate", "doAmpOffset", "doCorrectGains"]:
        setattr(cfg, name, False)
    cfg.doBias = True
    cfg.doDefect = True
    return cfg


def amp_noise(img, det):
    """Per-amp robust noise (nMAD), electrons, and an image of it."""
    noise = np.zeros_like(img)
    per_amp = {}
    for amp in det:
        bb = amp.getBBox()
        sl = (slice(bb.getMinY(), bb.getMaxY() + 1), slice(bb.getMinX(), bb.getMaxX() + 1))
        s = nmad(img[sl])
        per_amp[amp.getName()] = s
        noise[sl] = s
    return noise, per_amp


def hot_columns(img, det, src_mask, nsig=5.0, floor_e=1.0):
    """Boolean image marking columns, per amp, whose median is high.

    A column is hot when its median over the amp's rows, sources excluded, exceeds
    ``max(floor_e, nsig * nMAD)`` of all column medians in that amp.
    """
    hot = np.zeros(img.shape, bool)
    work = np.where(src_mask, np.nan, img)
    for amp in det:
        bb = amp.getBBox()
        sl = (slice(bb.getMinY(), bb.getMaxY() + 1), slice(bb.getMinX(), bb.getMaxX() + 1))
        colmed = np.nanmedian(work[sl], axis=0)
        cut = max(floor_e, np.nanmedian(colmed) + nsig * nmad(colmed))
        cols = np.where(colmed > cut)[0]
        if cols.size:
            hot[sl][:, cols] = True
    return hot


def measure(k, lbl_slice, lab, img, noise_img, gain_img, det_shape, amp_index,
            high_sigma, bad_dist):
    """Measurements for one labelled source.

    Returns
    -------
    row : `dict`
        Line fit, length (pixels), straightness, peak and sum (electrons), dE/dx
        (electrons per um of path in silicon) and flags.
    """
    sy, sx = lbl_slice
    sub = lab[sy, sx]
    yy, xx = np.nonzero(sub == k)
    xx = xx + sx.start
    yy = yy + sy.start
    v = img[yy, xx]
    fit = line_fit(xx.astype(float), yy.astype(float), v)
    t, d = project(xx, yy, fit)
    length = float(t.max() - t.min()) if xx.size > 1 else 0.0
    ipk = int(np.argmax(v))
    peak_e = float(v[ipk])
    gain = float(gain_img[yy[ipk], xx[ipk]])
    path_um = np.hypot(length * PIXEL_PITCH_UM, THICKNESS_UM)
    ny, nx = det_shape
    amps = np.unique(amp_index[yy, xx])
    return dict(
        npix=int(xx.size), n_hi=int((v > high_sigma * noise_img[yy, xx]).sum()),
        xc=fit["xc"], yc=fit["yc"], ux=fit["ux"], uy=fit["uy"],
        tmin=float(t.min()), tmax=float(t.max()), length_px=length,
        perp_rms_px=float(np.sqrt(fit["lam2"])),
        elong=float(1 - fit["lam2"] / fit["lam1"]) if fit["lam1"] > 0 else 0.0,
        angle_deg=float(np.degrees(np.arctan2(fit["uy"], fit["ux"])) % 180),
        n_cols=int(np.unique(xx).size), n_rows=int(np.unique(yy).size),
        peak_e=peak_e, gain=gain, peak_adu=peak_e / gain, sum_e=float(v.sum()),
        dedx_e_per_um=float(v.sum() / path_um),
        noise_e=float(noise_img[yy[ipk], xx[ipk]]),
        end_dist=float(min(bad_dist[yy[t <= t.min() + 1], xx[t <= t.min() + 1]].min(),
                           bad_dist[yy[t >= t.max() - 1], xx[t >= t.max() - 1]].min())),
        cross_amp=bool(amps.size > 1),
    ), (xx, yy)



def process_detector(butler, exposure, det_name, args, cfg):
    """ISR plus source finding for one detector.

    Returns
    -------
    rows : `list` [`dict`]
        One per source.
    hdus : `list` [`astropy.io.fits.ImageHDU`]
        One per selected long straight track.
    """
    from lsst.ip.isr import IsrTaskLSST
    from lsst.obs.lsst import LsstCam
    det = LsstCam.getCamera()[det_name]
    did = dict(instrument="LSSTCam", exposure=exposure, detector=det.getId())
    raw = butler.get("raw", **did)
    md = raw.getMetadata()
    calibs = {k: butler.get(k, **did) for k in ["bias", "defects", "ptc"]}
    exp = IsrTaskLSST(config=cfg).run(raw, **calibs).exposure
    img = exp.image.array.astype(float)
    mask = exp.mask
    bad = (mask.array & mask.getPlaneBitMask(["BAD", "SAT", "NO_DATA"])) != 0
    img[bad] = np.nan

    noise_img, noise_amp = amp_noise(img, det)
    # A track ending at the detector edge or a defect has an unknown exit depth.
    bad_dist = ndimage.distance_transform_edt(
        np.pad(~bad, 1, constant_values=False))[1:-1, 1:-1]
    gain_img = np.zeros_like(img)
    amp_index = np.zeros(img.shape, int)
    for i, amp in enumerate(det):
        bb = amp.getBBox()
        sl = (slice(bb.getMinY(), bb.getMaxY() + 1), slice(bb.getMinX(), bb.getMaxX() + 1))
        gain_img[sl] = calibs["ptc"].gain[amp.getName()]
        amp_index[sl] = i

    with np.errstate(invalid="ignore"):
        low = img > args.low_sigma * noise_img
        high = img > args.high_sigma * noise_img
    lab, nlab = ndimage.label(low, structure=np.ones((3, 3)))
    keep = np.unique(lab[high])
    keep = keep[keep > 0]
    hot = hot_columns(img, det, ndimage.binary_dilation(low, iterations=2))
    slices = ndimage.find_objects(lab)

    reb = int(det_name[5])
    meta = dict(DETECTOR=det.getId(), DETNAME=det_name, RAFT=det_name[:3],
                SENSOR=det_name[4:], REB=reb, CCDTYPE=det.getPhysicalType(),
                SERIAL=det.getSerial(), BSSVBS=md.get("BSSVBS", np.nan),
                CCDTEMP=md.get("CCDTEMP", np.nan))
    rows, hdus = [], []
    for k in keep:
        row, (xx, yy) = measure(k, slices[k - 1], lab, img, noise_img, gain_img,
                                img.shape, amp_index, args.high_sigma, bad_dist)
        if row["npix"] < args.min_npix:
            continue
        row["hot_col"] = bool(hot[yy, xx].any())
        row["columnar"] = bool(row["n_cols"] <= 2 and row["n_rows"] >= args.min_length)
        row["straight"] = bool(row["perp_rms_px"] <= args.max_perp_rms)
        row["unsat"] = bool(row["peak_adu"] < args.max_peak_adu)
        row["long"] = bool(row["length_px"] >= args.min_length)
        row["selected"] = (row["long"] and row["straight"] and row["unsat"]
                           and not row["hot_col"] and not row["columnar"])
        row.update(meta)
        row["track_id"] = int(k)
        rows.append(row)
        if not row["selected"]:
            continue
        sy, sx = slices[k - 1]
        g = args.margin
        y0, y1 = max(sy.start - g, 0), min(sy.stop + g, img.shape[0])
        x0, x1 = max(sx.start - g, 0), min(sx.stop + g, img.shape[1])
        stamp = img[y0:y1, x0:x1].copy()
        other = (lab[y0:y1, x0:x1] > 0) & (lab[y0:y1, x0:x1] != k)
        stamp[ndimage.binary_dilation(other, iterations=1)] = np.nan
        h = fits.ImageHDU(stamp.astype(np.float32), name=f"{det_name}_{k}")
        hdr = h.header
        for key, val in meta.items():
            hdr[key] = val
        hdr["BUNIT"] = "electron"
        hdr["X0"] = (x0, "stamp origin x, detector pixels")
        hdr["Y0"] = (y0, "stamp origin y, detector pixels")
        for key, col, com in [
                ("TRACKID", "track_id", "label within detector"),
                ("XC", "xc", "centroid x, detector pixels"),
                ("YC", "yc", "centroid y, detector pixels"),
                ("UX", "ux", "unit vector along track, x"),
                ("UY", "uy", "unit vector along track, y"),
                ("TMIN", "tmin", "track start along line, pixels from centroid"),
                ("TMAX", "tmax", "track end along line, pixels from centroid"),
                ("LENGTH", "length_px", "projected length, pixels"),
                ("ANGLE", "angle_deg", "track angle from +x, deg, CCS pixel frame"),
                ("PERPRMS", "perp_rms_px", "weighted rms across track, pixels"),
                ("PEAK_E", "peak_e", "peak pixel, electrons"),
                ("PEAKADU", "peak_adu", "peak pixel, ADU"),
                ("SUM_E", "sum_e", "total charge, electrons"),
                ("DEDX", "dedx_e_per_um", "charge per um of path in Si, e/um"),
                ("GAIN", "gain", "PTC gain at peak, e/ADU"),
                ("NOISE_E", "noise_e", "amp nMAD noise, electrons"),
                ("ENDDIST", "end_dist", "track end to nearest edge or defect, pixels"),
                ("CROSSAMP", "cross_amp", "spans more than one amp")]:
            val = row[col]
            hdr[key] = (bool(val) if isinstance(val, (bool, np.bool_)) else val, com)
        hdus.append(h)
    return rows, hdus


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, nargs="+")
    p.add_argument("--seq-file", help="text file of seq_num, one per line")
    p.add_argument("--chunk", type=int, nargs=2, metavar=("I", "N"),
                   help="process only the I-th of N interleaved slices (0-based)")
    p.add_argument("--run-date", help="YYYYMMDD of the run directory; default today")
    p.add_argument("--detectors", nargs="+", default=R21,
                   help="detector names, or 'science' for all 189 science CCDs")
    p.add_argument("--label", default=None, help="run label; default find_<day_obs>")
    p.add_argument("--low-sigma", type=float, default=3.0)
    p.add_argument("--high-sigma", type=float, default=8.0)
    p.add_argument("--min-npix", type=int, default=5)
    p.add_argument("--min-length", type=float, default=10.0, help="pixels")
    p.add_argument("--max-perp-rms", type=float, default=2.0,
                   help="pixels; loose, to drop blobs. Diffusion widens tracks, "
                        "so straightness proper is tested in the analysis")
    p.add_argument("--max-peak-adu", type=float, default=50000.0,
                   help="half of the ~100k ADU full well")
    p.add_argument("--margin", type=int, default=6, help="stamp margin, pixels")
    args = p.parse_args()

    from lsst.daf.butler import Butler
    from rubinwork.common import study_run

    dets = args.detectors
    if dets == ["science"]:
        from lsst.obs.lsst import LsstCam
        from lsst.afw.cameraGeom import DetectorType
        dets = [d.getName() for d in LsstCam.getCamera()
                if d.getType() == DetectorType.SCIENCE]
    seqs = list(args.seq_num or [])
    if args.seq_file:
        seqs += [int(x) for x in open(args.seq_file).read().split()]
    if args.chunk:
        seqs = seqs[args.chunk[0]::args.chunk[1]]
    label = args.label or f"find_{args.day_obs}"
    if args.chunk and args.chunk[0] != 0:
        # Array tasks share one run directory; only task 0 writes run.json, as
        # concurrent writers collide on its temporary file.
        import datetime
        from rubinwork.common.runs import data_root
        date = args.run_date or datetime.date.today().strftime("%Y%m%d")
        run_path = data_root() / "studies" / "camera" / "cosmic_diffusion" / f"{date}_{label}"
    else:
        run_path = study_run("camera/cosmic_diffusion", label, date=args.run_date).path
    outdir = run_path / "visits"
    outdir.mkdir(parents=True, exist_ok=True)
    butler = Butler("/repo/main", collections=["LSSTCam/raw/all", "LSSTCam/calib"])
    cfg = isr_config()

    for seq in seqs:
        exposure = args.day_obs * 100000 + seq
        rec = list(butler.query_dimension_records(
            "exposure", instrument="LSSTCam", exposure=exposure))[0]
        if rec.observation_type != "dark":
            print(f"skip {exposure}: observation_type {rec.observation_type}")
            continue
        all_rows, hdus = [], []
        for name in dets:
            t0 = time.time()
            rows, h = process_detector(butler, exposure, name, args, cfg)
            all_rows += rows
            hdus += h
            print(f"{exposure} {name}: {len(rows)} sources, {len(h)} long straight, "
                  f"{time.time() - t0:.1f} s", flush=True)
        prim = fits.PrimaryHDU()
        ph = prim.header
        ph["DAY_OBS"] = args.day_obs
        ph["SEQNUM"] = seq
        ph["EXPOSURE"] = exposure
        ph["EXPTIME"] = (rec.exposure_time, "s")
        ph["DARKTIME"] = (rec.dark_time, "s")
        ph["NCOSMIC"] = len(hdus)
        ph["MINLEN"] = (args.min_length, "pixels")
        ph["MAXPKADU"] = (args.max_peak_adu, "ADU")
        ph["MAXPERP"] = (args.max_perp_rms, "pixels")
        stem = f"{args.day_obs}_{seq:05d}"
        fits.HDUList([prim] + hdus).writeto(outdir / f"cosmics_{stem}.fits",
                                            overwrite=True)
        df = pd.DataFrame(all_rows)
        df["exposure"] = exposure
        df.to_parquet(outdir / f"tracks_{stem}.parquet")
        print(f"wrote {outdir}/cosmics_{stem}.fits ({len(hdus)} tracks)", flush=True)


if __name__ == "__main__":
    main()
