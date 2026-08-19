#!/usr/bin/env python
"""
Batch a whole night: for every exposure in the on-sky summary, simulate the
guiders + full-FoV stars on ONE matched atmosphere and produce, per visit:
  - the guider star movie            (summit_utils GuiderPlotter, cutout 20)
  - the psf-shape focal-plane plot    (summit_extras makeAzElPlot, 189 CCDs)
  - the centroid-MOTION moment comparison vs data (Q1/Q2/T motion)

Each visit gets its own subdir output/<night>/seq_NNNNN/ with the movie + png +
moment tables + meta. The big raw stamp arrays (~192 MB/visit) are deleted after
the plots are made unless --keep-stamps (837 visits would otherwise be ~160 GB).
Per-visit checkpointing makes the run restartable: rerun the same command and
finished visits are skipped.

Matched per visit from the summary: alt/az, mjd -> psfws weather datapoint,
field-median fwhm_med (delivered FWHM), camRotAngle, band.

  python guider_atmo_night.py \
      --data-summary output/night_20260706/moments/dayObs=20260706 \
      --psfws-forecast ecmwf_-30.25_-70.75_20260701_20260713.p \
      --psfws-data-dir /sdf/.../psf-weather-station/data \
      --band i --outdir output/night_sim_20260706 --max 5     # drop --max for all

Needs the LSST stack + psfws + summit_utils + summit_extras. Expensive (one
atmosphere build + guider + FoV draws + two renders per visit) -> run overnight.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _mods():
    here = os.path.dirname(os.path.abspath(__file__))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, name + ".py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m
    return (load("guider_atmo_sim"), load("guider_atmo_compare"),
            load("guider_atmo_to_guiderdata"), load("guider_atmo_psfshape"),
            load("guider_atmo_fovmap"))


def _load_summary(path, seq_min, seq_max):
    files = glob.glob(os.path.join(path, "*.parquet")) if os.path.isdir(path) else [path]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if seq_min is not None:
        df = df[df.seqNum >= seq_min]
    if seq_max is not None:
        df = df[df.seqNum <= seq_max]
    return df


def _make_movie(ga, subdir, cutout):
    from lsst.summit.utils.guiders.plotting import GuiderPlotter
    gd, starsDf = ga.load_sim_guiderdata(subdir, 0)
    save = os.path.join(subdir, "guider_movie.mp4")
    GuiderPlotter(gd, starsDf).makeAnimation(cutoutSize=cutout, saveAs=save)


def _make_psfshape(gp, subdir, band, marker_size):
    import matplotlib.collections as mcoll
    from lsst.summit.extras.plotting.psfPlotting import makeFigureAndAxes, makeAzElPlot
    with open(os.path.join(subdir, "guider_atmo_fov_meta.json")) as fh:
        meta = json.load(fh)
    t, camera = gp.build_table(subdir, 0, meta)
    fig, axs = makeFigureAndAxes(nrows=3)
    makeAzElPlot(fig, axs, t, camera, physicalFilter=band or meta.get("band", "i"), saveAs="")
    for ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
        for coll in ax.collections:
            if isinstance(coll, mcoll.PathCollection):
                coll.set_sizes([marker_size])
    fig.savefig(os.path.join(subdir, "guider_psfshape.png"))


def _visit_cfg(gas, drow, cfg0, band):
    from astropy.time import Time
    alt = float(drow["alt"].iloc[0]); az = float(drow["az"].iloc[0])
    mjd = float(drow["mjd"].iloc[0]); fwhm = float(np.median(drow["fwhm_med"]))
    airmass = 1.0 / np.sin(np.radians(alt)) if alt > 0 else 1.0
    return dict(cfg0, _alt=alt, _az=az, airmass=airmass, target_fwhm=fwhm,
                psfws_date=Time(mjd, format="mjd").strftime("%Y-%m-%d %H:%M"),
                rot_tel_pos=float(drow["camRotAngle"].iloc[0]),
                day_obs=int(drow["expId"].iloc[0]) // 100000,
                seq_num=int(drow["seqNum"].iloc[0]), band=band or cfg0["band"])


def _motion_rows(gc, subdir, seq, drow):
    """sim-vs-data centroid-motion moment rows for one visit."""
    sim, _ = gc.decompose_sim(subdir, visit=0)
    rows = []
    for r in sim.itertuples():
        d = drow[drow.detector == r.detector]
        row = dict(seqNum=int(seq), detector=r.detector,
                   sim_Mxx_motion=r.Mxx_motion, sim_Myy_motion=r.Myy_motion,
                   sim_Mxy_motion=r.Mxy_motion, sim_Q1_motion=r.Q1_motion,
                   sim_Q2_motion=r.Q2_motion, sim_T_motion=r.T_motion)
        if len(d):
            dd = d.iloc[0]
            for k in ("Mxx_motion", "Myy_motion", "Q1_motion", "Q2_motion"):
                if k in d.columns:
                    row["data_" + k] = float(dd[k])
            if {"Mxx_motion", "Myy_motion"} <= set(d.columns):
                row["data_T_motion"] = float(dd["Mxx_motion"] + dd["Myy_motion"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-summary", required=True)
    ap.add_argument("--psfws-forecast", required=True)
    ap.add_argument("--psfws-data-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--band", default=None, help="filter band (else CFG default)")
    ap.add_argument("--atmo-source", default="psfws")
    ap.add_argument("--screen-scale", type=float, default=0.2)
    ap.add_argument("--roi", type=int, default=64, help="guider stamp ROI [pix]")
    ap.add_argument("--cutout", type=int, default=20, help="movie cutout [pix]")
    ap.add_argument("--marker-size", type=float, default=40.0)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seq-min", type=int, default=None)
    ap.add_argument("--seq-max", type=int, default=None)
    ap.add_argument("--keep-stamps", action="store_true",
                    help="retain the raw stamp .npy files (large!)")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--no-psfshape", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gas, gc, ga, gp, gf = _mods()
    data = _load_summary(args.data_summary, args.seq_min, args.seq_max)
    seqs = sorted(data.seqNum.unique())[:: args.stride]
    if args.max:
        seqs = seqs[: args.max]
    guider_geom = gas.get_guiders_afw()
    science_geom = gas.get_science_afw()               # 189, for the FoV / psf-shape
    cfg0 = dict(gas.CFG, atmo_source=args.atmo_source, screen_scale=args.screen_scale,
                roi=args.roi, fov_dets="science", fov_roi=args.roi, atmo_plots=True,
                psfws_forecast_file=args.psfws_forecast, psfws_data_dir=args.psfws_data_dir)

    ckptdir = os.path.join(args.outdir, "visits"); os.makedirs(ckptdir, exist_ok=True)
    for i, seq in enumerate(seqs):
        ckpt = os.path.join(ckptdir, f"seq_{seq:05d}.parquet")
        if os.path.exists(ckpt) and not args.overwrite:
            print(f"[{i+1}/{len(seqs)}] seqNum {seq}: checkpoint exists, skipping")
            continue
        drow = data[data.seqNum == seq]
        subdir = os.path.join(args.outdir, f"seq_{seq:05d}"); os.makedirs(subdir, exist_ok=True)
        print(f"\n########## [{i+1}/{len(seqs)}] seqNum {seq}  ({len(drow)} guiders in data) ##########")
        try:
            cfg = _visit_cfg(gas, drow, cfg0, args.band)
            gas.run_guiders(cfg, guider_geom, args.atmo_source, subdir, os, pd,
                            fov_geom=science_geom)
            if not args.no_movie:
                try:
                    _make_movie(ga, subdir, args.cutout)
                except Exception as exc:                 # noqa: BLE001
                    print(f"   movie failed: {type(exc).__name__}: {exc}")
            if not args.no_psfshape:
                try:
                    _make_psfshape(gp, subdir, args.band, args.marker_size)
                except Exception as exc:                 # noqa: BLE001
                    print(f"   psfshape failed: {type(exc).__name__}: {exc}")
            try:
                gf.make_astrometry(subdir, 0)            # atmospheric shift quiver
            except Exception as exc:                     # noqa: BLE001
                print(f"   astrometry failed: {type(exc).__name__}: {exc}")
            rows = _motion_rows(gc, subdir, seq, drow)
            pd.DataFrame(rows).to_parquet(ckpt, index=False)   # checkpoint (also = 'done')
        except Exception as exc:                          # noqa: BLE001
            print(f"   VISIT FAILED: {type(exc).__name__}: {exc}")
            continue
        finally:
            if not args.keep_stamps:                      # reclaim disk
                for f in glob.glob(os.path.join(subdir, "*stamps_visit*.npy")):
                    os.remove(f)

    files = sorted(glob.glob(os.path.join(ckptdir, "seq_*.parquet")))
    out = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
           if files else pd.DataFrame())
    pq = os.path.join(args.outdir, "guider_atmo_night_motion.parquet")
    out.to_parquet(pq, index=False)
    print(f"\nwrote {pq}  ({len(out)} rows, "
          f"{out.seqNum.nunique() if len(out) else 0} visits from {len(files)} checkpoints)")
    if len(out):
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
            m = np.isfinite(df[sc]) & np.isfinite(df[dc])
            ax.scatter(df[dc][m], df[sc][m], s=8, alpha=0.4)
            lim = np.nanpercentile(np.abs(np.r_[df[dc][m], df[sc][m]]), 99) if m.any() else 1
            lo = 0 if col == "T_motion" else -lim
            ax.plot([lo, lim], [lo, lim], "k--", lw=0.8)
            ax.set_xlabel(f"data {lab}_motion [arcsec$^2$]")
            ax.set_ylabel(f"sim {lab}_motion [arcsec$^2$]")
            r = (np.nanmedian(df[sc][m]) / np.nanmedian(df[dc][m])) if m.any() else np.nan
            ax.set_title(f"{lab}_motion  (median sim/data={r:.2f})")
            ax.set_aspect("equal")
    fig.suptitle(f"Guider centroid-motion moments: sim vs data ({df.seqNum.nunique()} visits)")
    fig.tight_layout()
    save = os.path.join(outdir, "guider_atmo_night_motion.png")
    fig.savefig(save, dpi=120)
    print(f"wrote {save}")


if __name__ == "__main__":
    main()
