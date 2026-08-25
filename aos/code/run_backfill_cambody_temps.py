#!/usr/bin/env python
"""Backfill per-coadd-block CAMERA-BODY temperatures from the EFD and write a
sidecar keyed by block, so they can be correlated against the coadd-vs-MIW
spatial r (recompute_coadd_metrics.py --cambody-parquet ...).

These temperatures are NOT in ConsDB.  They come from the camera utility-trunk
housekeeping topic

    lsst.sal.MTCamera.utiltrunk_body      (Chronograf: "lsst.MTCamera"."autogen".
                                           "lsst.MTCamera.utiltrunk_body")

which carries ~25 structural temperatures -- CamBody{X+,Y+,Y-}Temp,
CamHous{X+,X-,Y+,Y-}Temp, BackFlng*, ShrdRng*, L1/L2 lens temps, and the
aggregate `AverageTemp` (the single most useful scalar).  Field names verified
from ts_xml MTCamera/utiltrunk_body.avsc.

RSP-ONLY: needs `lsst_efd_client` + EFD access.  The camera housekeeping lives in
its own InfluxDB database (`lsst.MTCamera`), NOT the main `efd` DB, so we point
the EFD client at --db (default lsst.MTCamera).  If that DB is not reachable from
your EFD instance, try --db efd with --topic lsst.sal.MTCamera.utiltrunk_body
(in case it is bridged into the observatory EFD).

Per block: one mean per field over the block's visit time window (min..max visit
mjd, padded by --pad-sec), using only samples that are finite and within
[--tmin,--tmax] degC.  Time window from visits.parquet ((day_obs,seq_num)->mjd)
joined to the coadd blocks_summary.parquet ((block,day_obs,seq_min,seq_max)).

Writes <coadd_dir>/cambody_temps.parquet : columns
  block, n_samp, <field>_mean for each requested field
(prefixed `cam_` so they read clearly in the correlation/ML pages, e.g.
cam_AverageTemp, cam_CamBodyXPlusTemp).
"""
import argparse
import asyncio
import os

import numpy as np
import pandas as pd
from astropy.time import Time

TOPIC_DEFAULT = "lsst.sal.MTCamera.utiltrunk_body"
DB_DEFAULT = "lsst.MTCamera"

# camera-body-relevant defaults (AverageTemp first); --all-fields adds the rest
CAMBODY_FIELDS = ["AverageTemp",
                  "CamBodyXPlusTemp", "CamBodyYPlusTemp", "CamBodyYMinusTemp",
                  "CamHousXPlusTemp", "CamHousXMinusTemp",
                  "CamHousYPlusTemp", "CamHousYMinusTemp"]
ALL_FIELDS = CAMBODY_FIELDS + [
    "BackFlngXMinusTemp", "BackFlngYMinusTemp",
    "ShrdRngXPlusTemp", "ShrdRngXMinusTemp", "ShrdRngYPlusTemp",
    "L1XMinusTemp", "L1YMinusTemp", "L2XPlusTemp", "L2XMinusTemp", "L2YPlusTemp",
    "DomeXMinusTemp", "DomeYMinusTemp", "VPPlenumInTemp",
    "ShtrEboxRtnAirTemp", "ShtrMtrRtnAirTemp", "ChgrYMinusRtnAirTemp", "AmbAirtemp"]


def block_windows(coadd_dir, param_root, pad_days):
    """Per-block [t0, t1] astropy Time window from the block's visit mjds."""
    bs = pd.read_parquet(os.path.join(coadd_dir, "blocks_summary.parquet"),
                         columns=["block", "day_obs", "seq_min", "seq_max"])
    v = pd.read_parquet(os.path.join(param_root, "visits.parquet"),
                        columns=["day_obs", "seq_num", "mjd"])
    rows = []
    for _, b in bs.iterrows():
        sub = v[(v.day_obs == b.day_obs) & (v.seq_num >= b.seq_min)
                & (v.seq_num <= b.seq_max)]
        if len(sub) == 0 or not np.isfinite(sub.mjd).any():
            rows.append((b["block"], int(b.day_obs), np.nan, np.nan))
            continue
        rows.append((b["block"], int(b.day_obs),
                     float(np.nanmin(sub.mjd)) - pad_days,
                     float(np.nanmax(sub.mjd)) + pad_days))
    return pd.DataFrame(rows, columns=["block", "day_obs", "mjd0", "mjd1"])


async def run(args):
    from lsst_efd_client import EfdClient
    coadd_dir = os.path.join(args.output_root, args.param_set, args.out_name)
    param_root = os.path.join(args.output_root, args.param_set)
    fields = ALL_FIELDS if args.all_fields else (args.fields or CAMBODY_FIELDS)
    win = block_windows(coadd_dir, param_root, args.pad_sec / 86400.0)
    print(f"{len(win)} blocks; {int(win.mjd0.notna().sum())} with a time window; "
          f"{len(fields)} fields from {args.topic} (db={args.db})")

    efd = EfdClient(args.efd, db_name=args.db)
    cols = fields + [f + "_state" for f in fields]
    per_block = {f: [] for f in fields}
    nsamp = []
    # one EFD query per night, then slice per block
    for day, grp in win.groupby("day_obs"):
        ok = grp.dropna(subset=["mjd0", "mjd1"])
        if len(ok) == 0:
            for _ in range(len(grp)):
                nsamp.append(0)
                for f in fields:
                    per_block[f].append(np.nan)
            continue
        t0 = Time(ok.mjd0.min(), format="mjd", scale="utc")
        t1 = Time(ok.mjd1.max(), format="mjd", scale="utc")
        try:
            df = await efd.select_time_series(args.topic, cols, t0, t1)
        except Exception as e:
            print(f"  {day}: EFD query failed ({e}); NaN for {len(grp)} blocks")
            df = pd.DataFrame()
        if len(df):
            df = df.copy()
            df["mjd"] = Time(df.index).utc.mjd
        for _, b in grp.iterrows():
            if not np.isfinite(b.mjd0) or len(df) == 0:
                nsamp.append(0)
                for f in fields:
                    per_block[f].append(np.nan)
                continue
            m = (df.mjd >= b.mjd0) & (df.mjd <= b.mjd1)
            sl = df[m]
            nsamp.append(int(len(sl)))
            for f in fields:
                if f not in sl or len(sl) == 0:
                    per_block[f].append(np.nan); continue
                x = pd.to_numeric(sl[f], errors="coerce").to_numpy(float)
                good = np.isfinite(x) & (x >= args.tmin) & (x <= args.tmax)
                if args.require_state and (f + "_state") in sl:
                    st = pd.to_numeric(sl[f + "_state"], errors="coerce").to_numpy(float)
                    good &= (st == args.ok_state)
                per_block[f].append(float(np.mean(x[good])) if good.any() else np.nan)
        print(f"  {day}: {len(df)} EFD rows over the night, {len(grp)} blocks")

    out = pd.DataFrame({"block": win["block"].to_numpy(), "n_samp": nsamp})
    for f in fields:
        out["cam_" + f] = per_block[f]
    outpath = args.out or os.path.join(coadd_dir, "cambody_temps.parquet")
    out.to_parquet(outpath, index=False)
    filled = int(out["cam_AverageTemp"].notna().sum()) if "cam_AverageTemp" in out else -1
    print(f"wrote {outpath}  ({len(out)} blocks; cam_AverageTemp finite on {filled})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-set", default="fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x")
    ap.add_argument("--out-name", default="coadd_50_34")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--efd", default="usdf_efd", help="EfdClient instance name")
    ap.add_argument("--db", default=DB_DEFAULT, help="InfluxDB database (camera HK "
                    "lives in lsst.MTCamera, not the main efd DB)")
    ap.add_argument("--topic", default=TOPIC_DEFAULT)
    ap.add_argument("--fields", nargs="+", default=None,
                    help="override field list (default: camera-body set)")
    ap.add_argument("--all-fields", action="store_true", help="pull all ~25 temps")
    ap.add_argument("--pad-sec", type=float, default=120.0,
                    help="pad each block window by this many seconds")
    ap.add_argument("--tmin", type=float, default=-40.0, help="reject samples below")
    ap.add_argument("--tmax", type=float, default=50.0, help="reject samples above")
    ap.add_argument("--require-state", action="store_true",
                    help="keep only samples whose <field>_state == --ok-state")
    ap.add_argument("--ok-state", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
