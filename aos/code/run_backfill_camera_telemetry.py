#!/usr/bin/env python3
"""Backfill CAMERA-BODY temperatures as a per-chunk, visit-keyed parquet, then
(optionally) merge them into the COMBINED visits.parquet.

Architecture (parallel to the per-chunk {donuts,fits,visits}.parquet products):

  output/<ps>/chunks/<dmin>_<dmax>/camera_telemetry.parquet   <- this script
      one row per visit: day_obs, seq_num, cam_n_samp, cam_<field>...

`--merge` gathers ALL chunks' camera_telemetry.parquet and joins the cam_<field>
columns (on day_obs, seq_num) into the COMBINED output/<ps>/visits.parquet --
NOT into the per-chunk visits.parquet, which are left pristine (they are the
expensive mktable outputs; the combined file is a cheap concat, safe to augment
in place).  From there the camera telemetry flows into the coadd / time-series
study exactly like the ESS thermal columns.  (Re-running combine_visits would
drop the cam columns from the combined file -- just re-run --merge --skip-produce
to restore them from the per-chunk camera_telemetry.parquet, the source of truth.)

Camera-body temperatures are NOT in ConsDB; they come from the camera
utility-trunk housekeeping EFD topic

    measurement lsst.MTCamera.utiltrunk_body  in InfluxDB database lsst.MTCamera
    (Chronograf: "lsst.MTCamera"."autogen"."lsst.MTCamera.utiltrunk_body")

which carries ~25 structural temps -- CamBody{X+,Y+,Y-}Temp, CamHous*, BackFlng*,
ShrdRng*, L1/L2 lens temps, and the aggregate `AverageTemp` (the single most
useful scalar).  Field names verified from ts_xml MTCamera/utiltrunk_body.avsc.

The camera housekeeping lives in its OWN InfluxDB database (`lsst.MTCamera`), not
the main `efd` DB, and its measurement name is `lsst.MTCamera.utiltrunk_body`
(NOT the SAL name lsst.sal.MTCamera.*).  We point the EFD client at --db (default
lsst.MTCamera) and query --topic (default lsst.MTCamera.utiltrunk_body).  NOTE:
EfdClient.select_time_series only flags "Topic ... not in EFD schema" when the
query comes back EMPTY -- i.e. it means the measurement/db/time-range yielded no
rows, usually a wrong --topic/--db, not a hard schema error.

RSP-ONLY (needs lsst_efd_client + EFD access; run on an interactive node).

  # 1. produce per-chunk camera_telemetry.parquet (verify before merging):
  python run_backfill_camera_telemetry.py --param-set <ps> --all-fields
  # or a single chunk:
  python run_backfill_camera_telemetry.py --visits output/<ps>/chunks/<d>_<d>/visits.parquet
  # 2. merge the per-chunk tables into the COMBINED visits (no re-fetch):
  python run_backfill_camera_telemetry.py --param-set <ps> --merge --skip-produce

Per visit: mean of each field over [visit_mjd - pad, visit_mjd + pad], keeping
only finite samples within [--tmin,--tmax] degC (and, with --require-state, only
samples whose <field>_state == --ok-state).  One EFD query per night.
"""
import argparse
import asyncio
import glob
import os
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time

# Harmless: QTable.read warns "No table::len::<col> found" for the combined
# visits.parquet (written by combine_parquets.py via pandas, without astropy's
# fixed-string-length metadata).  astropy falls back to the longest string, so
# band/science_program are read losslessly (verified: values + dtypes preserved,
# coadd filters match exactly), and our QTable.write adds the metadata so it does
# not recur.  Silence just this one message.
warnings.filterwarnings("ignore", message="No table::len::")

TOPIC_DEFAULT = "lsst.MTCamera.utiltrunk_body"   # camera-DB measurement (no `sal`)
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


def _read_visit_times(visits_path):
    """(day_obs, seq_num, mjd) per visit, from a chunk visits.parquet."""
    v = pd.read_parquet(visits_path, columns=["day_obs", "seq_num", "mjd"])
    v = v.dropna(subset=["mjd"]).copy()
    v["day_obs"] = v["day_obs"].astype(int)
    v["seq_num"] = v["seq_num"].astype(int)
    return v


async def fetch_visit_means(efd, topic, v, fields, pad_days, tmin, tmax,
                            require_state, ok_state):
    """Return a DataFrame [day_obs, seq_num, cam_n_samp, cam_<field>...] with one
    row per visit, meaning each field over the visit's padded window."""
    cols = list(fields) + ([f + "_state" for f in fields] if require_state else [])
    out = {"day_obs": [], "seq_num": [], "cam_n_samp": []}
    for f in fields:
        out["cam_" + f] = []
    for day, g in v.groupby("day_obs"):
        t0 = Time(g.mjd.min() - pad_days, format="mjd", scale="utc")
        t1 = Time(g.mjd.max() + pad_days, format="mjd", scale="utc")
        try:
            df = await efd.select_time_series(topic, cols, t0, t1)
        except Exception as e:
            print(f"  {day}: EFD query failed ({e}); NaN for {len(g)} visits")
            df = pd.DataFrame()
        if len(df):
            df = df.copy()
            df["mjd"] = Time(df.index).utc.mjd
        for _, r in g.iterrows():
            out["day_obs"].append(int(r.day_obs)); out["seq_num"].append(int(r.seq_num))
            if len(df) == 0:
                out["cam_n_samp"].append(0)
                for f in fields:
                    out["cam_" + f].append(np.nan)
                continue
            sl = df[(df.mjd >= r.mjd - pad_days) & (df.mjd <= r.mjd + pad_days)]
            out["cam_n_samp"].append(int(len(sl)))
            for f in fields:
                if f not in sl or len(sl) == 0:
                    out["cam_" + f].append(np.nan); continue
                x = pd.to_numeric(sl[f], errors="coerce").to_numpy(float)
                good = np.isfinite(x) & (x >= tmin) & (x <= tmax)
                if require_state and (f + "_state") in sl:
                    st = pd.to_numeric(sl[f + "_state"], errors="coerce").to_numpy(float)
                    good &= (st == ok_state)
                out["cam_" + f].append(float(np.mean(x[good])) if good.any() else np.nan)
        print(f"  {day}: {len(df)} EFD rows, {len(g)} visits")
    return pd.DataFrame(out)


def gather_chunk_camera(output_root, param_set):
    """Concatenate all per-chunk camera_telemetry.parquet for a param_set.
    Chunks are date-disjoint, so (day_obs, seq_num) is unique; keep first if not."""
    paths = sorted(glob.glob(os.path.join(
        output_root, param_set, "chunks", "*", "camera_telemetry.parquet")))
    if not paths:
        return None, paths
    cam = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    cam = cam.drop_duplicates(subset=["day_obs", "seq_num"], keep="first")
    return cam, paths


def merge_camera_into_visits(visits_path, cam_df, out_path):
    """Add the cam_<field> columns from cam_df to the visits table at
    visits_path (aligned by day_obs, seq_num) and write to out_path, preserving
    the QTable (units on existing columns).  Never modifies the per-chunk files;
    intended for the COMBINED output/<ps>/visits.parquet."""
    from astropy.table import QTable
    vi = QTable.read(str(visits_path))
    key = list(zip(np.asarray(vi["day_obs"], int), np.asarray(vi["seq_num"], int)))
    idx = {(int(r.day_obs), int(r.seq_num)): r for r in cam_df.itertuples()}
    camcols = [c for c in cam_df.columns if c.startswith("cam_")]
    for c in camcols:
        vi[c] = np.array([getattr(idx.get(k), c, np.nan) if k in idx else np.nan
                          for k in key], float)
    vi.write(str(out_path), overwrite=True)
    nfin = int(np.isfinite(np.asarray(vi["cam_AverageTemp"], float)).sum()) \
        if "cam_AverageTemp" in vi.colnames else -1
    print(f"merged {len(camcols)} cam cols into {out_path} "
          f"(matched {sum(k in idx for k in key)}/{len(vi)} visits; "
          f"cam_AverageTemp finite {nfin}/{len(vi)})")


async def process_one(efd, visits_path, args, fields):
    print(f"[camera_telemetry] {visits_path}")
    v = _read_visit_times(visits_path)
    print(f"  {len(v)} visits with mjd; {len(fields)} fields from {args.topic} "
          f"(db={args.db})")
    cam_df = await fetch_visit_means(
        efd, args.topic, v, fields, args.pad_sec / 86400.0,
        args.tmin, args.tmax, args.require_state, args.ok_state)
    out = args.output or os.path.join(os.path.dirname(visits_path),
                                      "camera_telemetry.parquet")
    cam_df.to_parquet(out, index=False)
    nfin = int(cam_df["cam_AverageTemp"].notna().sum()) if "cam_AverageTemp" in cam_df else -1
    print(f"  wrote {out} ({len(cam_df)} visits; cam_AverageTemp finite {nfin})")


async def run(args):
    fields = ALL_FIELDS if args.all_fields else (args.fields or CAMBODY_FIELDS)

    # --- produce per-chunk camera_telemetry.parquet (unless --skip-produce) ---
    if not args.skip_produce:
        from lsst_efd_client import EfdClient
        if args.visits:
            targets = [args.visits]
        else:
            targets = sorted(glob.glob(os.path.join(
                args.output_root, args.param_set, "chunks", "*", "visits.parquet")))
            if not targets:
                print(f"no chunk visits.parquet under "
                      f"{args.output_root}/{args.param_set}/chunks/")
                return
            print(f"{len(targets)} chunk(s) for {args.param_set}")
        efd = EfdClient(args.efd, db_name=args.db)
        for t in targets:
            await process_one(efd, t, args, fields)

    # --- merge all chunks' camera_telemetry into the COMBINED visits.parquet ---
    # (leaves the per-chunk visits.parquet untouched; the combined file is a
    #  cheap-to-regenerate concat, so it is safe to augment in place.)
    if args.merge:
        cam, paths = gather_chunk_camera(args.output_root, args.param_set)
        if cam is None:
            print("  --merge: no per-chunk camera_telemetry.parquet found "
                  "(run the produce step first)")
            return
        combined = os.path.join(args.output_root, args.param_set, "visits.parquet")
        out = args.merge_output or combined
        print(f"  merging {len(paths)} chunk camera tables ({len(cam)} visit rows) "
              f"-> {out}")
        merge_camera_into_visits(combined, cam, out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--visits", help="a single chunk visits.parquet")
    src.add_argument("--param-set", help="loop over all chunks of this param_set")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--output", default=None,
                    help="output parquet path (only with --visits; default: sibling "
                    "camera_telemetry.parquet)")
    ap.add_argument("--efd", default="usdf_efd", help="EfdClient instance name")
    ap.add_argument("--db", default=DB_DEFAULT,
                    help="InfluxDB database (camera HK is in lsst.MTCamera, not efd)")
    ap.add_argument("--topic", default=TOPIC_DEFAULT)
    ap.add_argument("--fields", nargs="+", default=None,
                    help="override field list (default: camera-body set)")
    ap.add_argument("--all-fields", action="store_true", help="pull all ~25 temps")
    ap.add_argument("--merge", action="store_true",
                    help="after producing, merge ALL chunks' camera_telemetry into "
                    "the COMBINED output/<ps>/visits.parquet (per-chunk visits.parquet "
                    "are left untouched); requires --param-set")
    ap.add_argument("--merge-output", default=None,
                    help="write the camera-augmented combined visits here instead of "
                    "overwriting output/<ps>/visits.parquet")
    ap.add_argument("--skip-produce", action="store_true",
                    help="skip the EFD fetch; only merge existing per-chunk "
                    "camera_telemetry.parquet into the combined visits (use with --merge)")
    ap.add_argument("--pad-sec", type=float, default=120.0,
                    help="pad each visit window by this many seconds")
    ap.add_argument("--tmin", type=float, default=-40.0, help="reject samples below")
    ap.add_argument("--tmax", type=float, default=50.0, help="reject samples above")
    ap.add_argument("--require-state", action="store_true",
                    help="keep only samples whose <field>_state == --ok-state")
    ap.add_argument("--ok-state", type=float, default=1.0)
    args = ap.parse_args()
    if args.output and not args.visits:
        ap.error("--output only applies with --visits")
    if args.merge and not args.param_set:
        ap.error("--merge merges into the combined visits, so it requires --param-set")
    if args.skip_produce and not args.merge:
        ap.error("--skip-produce only makes sense with --merge")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
