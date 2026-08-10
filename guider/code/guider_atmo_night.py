#!/usr/bin/env python
"""
Batch: simulate the guider stars for many exposures on a night and compare the
centroid-MOTION 2nd moments (Ixx/Iyy/Ixy -> Q1/Q2/T) sim-vs-data, per visit.

For each seqNum in the on-sky summary it pulls the matched pointing (alt/az),
weather time (mjd -> psfws datapoint) and delivered seeing (field-median fwhm_med),
builds one atmosphere, draws the guider stamps, and decomposes the image motion with
the SAME guiderMoments.measureStampMoments used on data. Writes a per-(seqNum,detector)
sim-vs-data table + summary scatter plots.

This is expensive (one atmosphere build per visit) -- a batch-node job. Test first
with --max / --stride on a handful of visits.

  python guider_atmo_night.py --data-summary output/night_20260706/moments/dayObs=20260706 \
      --psfws-forecast ecmwf_-30.25_-70.75_20260701_20260713.p \
      --psfws-data-dir /sdf/.../psf-weather-station/data \
      --outdir output/night_sim_20260706 --max 5      # drop --max for the full night

Needs the LSST stack (LsstCam geometry) + psfws + summit_extras' guiderMoments.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_summary(path, seq_min, seq_max):
    files = glob.glob(os.path.join(path, "*.parquet")) if os.path.isdir(path) else [path]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if seq_min is not None:
        df = df[df.seqNum >= seq_min]
    if seq_max is not None:
        df = df[df.seqNum <= seq_max]
    return df


def _motion_moments(stamps_g, cen_list, fwhm_arcsec, pixscale, gain, config, mm):
    """Flux-weighted centroid-covariance (image motion) [arcsec^2] for one guider,
    using guiderMoments.measureStampMoments (same as the data pipeline)."""
    sigma_pix = fwhm_arcsec / (mm._FWHM_PER_SIGMA * pixscale)
    r_aper = config.apNSigma * sigma_pix
    half = int(np.ceil(2 * r_aper)) + 2
    xc, yc, f = [], [], []
    roi = stamps_g.shape[-1]
    for k in range(stamps_g.shape[0]):
        ref = (roi / 2.0 + cen_list[k][0], roi / 2.0 + cen_list[k][1])
        m = mm.measureStampMoments(stamps_g[k].astype(float), ref, sigma_pix,
                                   r_aper, half, config.cenNIter)
        if m is not None:
            xc.append(m["xc"]); yc.append(m["yc"]); f.append(m["flux"])
    if len(f) < 3:
        return None
    xc, yc, f = np.array(xc), np.array(yc), np.array(f)
    w = f.sum()
    nx = np.mean(sigma_pix**2 / np.clip(f * gain, 1, None)) if config.subtractNoiseFloor else 0.0
    xm, ym = (f * xc).sum() / w, (f * yc).sum() / w
    s2 = pixscale**2
    vxx = ((f * (xc - xm) ** 2).sum() / w - nx) * s2
    vyy = ((f * (yc - ym) ** 2).sum() / w - nx) * s2
    vxy = ((f * (xc - xm) * (yc - ym)).sum() / w) * s2
    return dict(Mxx_motion=vxx, Myy_motion=vyy, Mxy_motion=vxy,
                Q1_motion=vxx - vyy, Q2_motion=2 * vxy, T_motion=vxx + vyy)


def simulate_visit(gas, mm, geom, source_key, drow, cfg0, seed):
    """Return per-detector sim motion moments for one visit (dict det->moments)."""
    from astropy.time import Time
    alt = float(drow["alt"].iloc[0]); az = float(drow["az"].iloc[0])
    mjd = float(drow["mjd"].iloc[0])
    fwhm_field = float(np.median(drow["fwhm_med"]))
    airmass = 1.0 / np.sin(np.radians(alt)) if alt > 0 else 1.0
    psdate = Time(mjd, format="mjd").strftime("%Y-%m-%d %H:%M")
    cfg = dict(cfg0, _alt=alt, _az=az, airmass=airmass, target_fwhm=fwhm_field,
               psfws_date=psdate, rot_tel_pos=float(drow["camRotAngle"].iloc[0]))
    cadence = 1.0 / cfg["stamp_rate_hz"]
    visit_time = cfg["stamps_per_visit"] * cadence
    theta_max = max(np.hypot(gx, gy) for _, gx, gy, _ in geom) / 3600.0
    atmo, cfgv, rng, params, ss = gas.build_calibrated_atmo(
        cfg, source_key, geom, theta_max, visit_time, seed)
    roi, config = cfg["roi"], mm.MomentConfig()
    out = {}
    for gi, (name, gx, gy, wcs) in enumerate(geom):
        stamps = np.empty((cfg["stamps_per_visit"], roi, roi), dtype=np.float32)
        cen = []
        for k in range(cfg["stamps_per_visit"]):
            img = gas.draw_stamp(atmo, (gx, gy), k * cadence, cfgv["exptime"], roi, wcs,
                                 rng, cfgv["flux_counts"], cfgv)
            stamps[k] = img.array.astype(np.float32)
            m = gas.measure(img)
            cen.append((m["cen_x"], m["cen_y"]) if m["ok"] else (0.0, 0.0))
        mo = _motion_moments(stamps, cen, fwhm_field, gas.PIXEL_SCALE, cfg["gain"], config, mm)
        if mo is not None:
            out[name] = mo
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-summary", required=True)
    ap.add_argument("--psfws-forecast", required=True)
    ap.add_argument("--psfws-data-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--atmo-source", default="psfws")
    ap.add_argument("--screen-scale", type=float, default=0.2)
    ap.add_argument("--max", type=int, default=None, help="only the first N visits")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth visit")
    ap.add_argument("--seq-min", type=int, default=None)
    ap.add_argument("--seq-max", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("gas", os.path.join(here, "guider_atmo_sim.py"))
    gas = importlib.util.module_from_spec(spec); spec.loader.exec_module(gas)
    import guiderMoments as mm

    data = _load_summary(args.data_summary, args.seq_min, args.seq_max)
    seqs = sorted(data.seqNum.unique())[:: args.stride]
    if args.max:
        seqs = seqs[: args.max]
    geom = gas.get_guiders_afw()
    cfg0 = dict(gas.CFG, atmo_source=args.atmo_source, screen_scale=args.screen_scale,
                psfws_forecast_file=args.psfws_forecast, psfws_data_dir=args.psfws_data_dir)

    rows = []
    for i, seq in enumerate(seqs):
        drow = data[data.seqNum == seq]
        print(f"[{i+1}/{len(seqs)}] seqNum {seq}  ({len(drow)} guiders)")
        try:
            sim = simulate_visit(gas, mm, geom, args.atmo_source, drow, cfg0,
                                 gas.CFG["seed"] + int(seq))
        except Exception as exc:                              # noqa: BLE001
            print(f"   FAILED: {type(exc).__name__}: {exc}")
            continue
        for name, mo in sim.items():
            d = drow[drow.detector == name]
            row = dict(seqNum=int(seq), detector=name)
            for k, v in mo.items():
                row["sim_" + k] = v
            for k in ("Mxx_motion", "Myy_motion", "Q1_motion", "Q2_motion"):
                if k in d and len(d):
                    row["data_" + k] = float(d[k].iloc[0])
            if {"Mxx_motion", "Myy_motion"} <= set(d):
                row["data_T_motion"] = float(d["Mxx_motion"].iloc[0] + d["Myy_motion"].iloc[0])
            rows.append(row)

    out = pd.DataFrame(rows)
    pq = os.path.join(args.outdir, "guider_atmo_night_motion.parquet")
    out.to_parquet(pq, index=False)
    print(f"\nwrote {pq}  ({len(out)} rows, {out.seqNum.nunique()} visits)")
    _plots(out, args.outdir)


def _plots(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    quants = [("T_motion", "T"), ("Q1_motion", "Q1"), ("Q2_motion", "Q2")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, lab) in zip(axes, quants):
        sc, dc = "sim_" + col, "data_" + col
        if sc in df and dc in df:
            ax.scatter(df[dc], df[sc], s=10, alpha=0.5)
            lim = np.nanpercentile(np.abs(np.r_[df[dc], df[sc]]), 99)
            lo = 0 if col == "T_motion" else -lim
            ax.plot([lo, lim], [lo, lim], "k--", lw=0.8)
            ax.set_xlabel(f"data {lab}_motion [arcsec$^2$]")
            ax.set_ylabel(f"sim {lab}_motion [arcsec$^2$]")
            ax.set_title(f"{lab}_motion  (median sim/data = "
                         f"{np.nanmedian(df[sc])/np.nanmedian(df[dc]):.2f})" if col == "T_motion"
                         else f"{lab}_motion")
            ax.set_aspect("equal")
    fig.suptitle(f"Guider centroid-motion moments: sim vs data "
                 f"({df.seqNum.nunique()} visits)")
    fig.tight_layout()
    save = os.path.join(outdir, "guider_atmo_night_motion.png")
    fig.savefig(save, dpi=120)
    print(f"wrote {save}")


if __name__ == "__main__":
    main()
