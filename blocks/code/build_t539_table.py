#!/usr/bin/env python
"""Build the BLOCK-T539 closed-loop AOS state table (parquet).

CLI port of ``blocks/t539_closedloop_aos.ipynb`` for the Snakemake pipeline.
For each converged closed-loop in-focus image (the last of each contiguous run)
it collects: ConsDB image quality, the aggregated DOF trim, the geom v-modes,
per-corner retrieved wavefront Zernikes, and thermal + wind telemetry (via the
shared ``olr/code/telemetry.py`` helper).

Runs on the RSP (ConsDB + EFD access), Summit or USDF.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

# Shared helpers: aos/code (DOF, v-modes, zernikes) + olr/code (telemetry).
REPO = Path(__file__).resolve().parents[2]          # -> rubin-work/
sys.path.insert(0, str(REPO / "aos" / "code"))
sys.path.insert(0, str(REPO / "olr" / "code"))
import aos_trim       # noqa: E402
import aos_state      # noqa: E402
import telemetry      # noqa: E402


def build(args):
    os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",.consdb"
    corners = {191: "R00_SW0", 195: "R04_SW0", 199: "R40_SW0", 203: "R44_SW0"}
    zk_noll = [z for z in range(4, 27) if z not in (20, 21)]

    # 1. Select converged closed-loop images from ConsDB ----------------------
    client = aos_trim.make_consdb_client(args.consdb_url)
    q = f"""
        SELECT v1.*, ql.physical_rotator_angle, ql.psf_sigma_median,
               ql.seeing_zenith_500nm_median
        FROM cdb_{args.instrument}.visit1 v1
        LEFT JOIN cdb_{args.instrument}.visit1_quicklook ql
          ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs >= {args.day_obs_min} AND v1.day_obs <= {args.day_obs_max}
    """
    allvisits = client.query(q).to_pandas()
    print(f"Fetched {len(allvisits)} visits in day_obs range", flush=True)

    sp = allvisits["science_program"].fillna("")
    note = allvisits["scheduler_note"].fillna("")
    cand = (allvisits[
        sp.str.startswith(args.science_program_prefix)
        & (allvisits["observation_reason"] == args.observation_reason)
        & note.isin(args.annotations)
    ].copy().sort_values(["day_obs", "seq_num"]).reset_index(drop=True))
    print(f"Candidate infocus images: {len(cand)}", flush=True)

    grp = (
        (cand["day_obs"] != cand["day_obs"].shift())
        | (cand["scheduler_note"] != cand["scheduler_note"].shift())
        | ((cand["seq_num"] - cand["seq_num"].shift()) > args.seq_gap_max)
    ).cumsum()
    cand["group_id"] = grp
    grp_size = cand.groupby("group_id").size()
    visits = cand.groupby("group_id").tail(1).copy()
    visits["n_in_group"] = visits["group_id"].map(grp_size).values
    visits = visits.sort_values(["day_obs", "seq_num"]).reset_index(drop=True)
    visits["psf_fwhm_median"] = 2.355 * pd.to_numeric(
        visits["psf_sigma_median"], errors="coerce") * 0.2
    print(f"Converged images (last per group): {len(visits)}", flush=True)
    if len(visits) == 0:
        raise SystemExit("no converged T539 images in range -- nothing to build")

    ids = ",".join(str(v) for v in visits["visit_id"])
    for col in ("donut_blur_fwhm", "aos_fwhm"):
        try:
            extra = client.query(
                f"SELECT visit_id, {col} FROM cdb_{args.instrument}.visit1_quicklook "
                f"WHERE visit_id IN ({ids})").to_pandas()
            visits = visits.merge(extra, on="visit_id", how="left")
        except Exception as e:
            print(f"{col} not available: {type(e).__name__}", flush=True)

    # 2. Aggregated DOF (trim) from the EFD -----------------------------------
    from lsst_efd_client import EfdClient
    efd = EfdClient(args.efd, output_mode="dataframe")
    fit_table = Table.from_pandas(visits[["day_obs", "seq_num"]].astype(int))
    trim, dof_info = aos_trim.fetch_aggregated_dof_for_visits(
        fit_table, efd_client=efd, consdb_client=client)
    for i in range(aos_trim.N_DOF):
        visits[f"dof{i}"] = trim[:, i]
    print(f"DOF finite: {dof_info['n_dof']}/{len(visits)}", flush=True)

    # 3. Geom v-modes from the DOF trim ---------------------------------------
    se = aos_state.make_state_estimator(config_dir=args.ofc_config_dir,
                                        dof_set="standard_22")
    dof_mat = visits[[f"dof{i}" for i in range(aos_trim.N_DOF)]].to_numpy(dtype=float)
    vmodes = aos_state.vmodes_from_dofs(dof_mat, se, n_modes=args.n_vmode)
    for j in range(args.n_vmode):
        visits[f"v{j + 1}"] = vmodes[:, j]
    print(f"v-modes finite: {int(np.isfinite(vmodes).all(axis=1).sum())}/{len(visits)}",
          flush=True)

    # 4. Per-corner retrieved wavefront (OPD) ---------------------------------
    zk_df = aos_state.fetch_corner_zernikes_consdb(
        client, visits["visit_id"].values, instrument=args.instrument,
        zk_noll=zk_noll, corners=corners)
    visits = visits.merge(zk_df, left_on="visit_id", right_index=True, how="left")

    # 5. Thermal + wind telemetry (shared helper; per-night, robust) ----------
    day_seq = (visits[["day_obs", "seq_num", "obs_start", "obs_end"]]
               .drop_duplicates().rename(columns={"seq_num": "seq"})
               .reset_index(drop=True))
    thermal = telemetry.fetch_thermal_telemetry_sync(efd, day_seq, progress=True)
    wind = telemetry.fetch_dome_wind_sync(efd, day_seq, progress=True)
    for tbl in (thermal, wind):
        t = tbl.rename(columns={"seq": "seq_num"})
        newcols = [c for c in t.columns if c not in ("day_obs", "seq_num")]
        visits = visits.merge(t[["day_obs", "seq_num"] + newcols],
                              on=["day_obs", "seq_num"], how="left")
    if "z_gradient" in visits:
        print(f"z_gradient finite: {int(visits['z_gradient'].notna().sum())}/{len(visits)}",
              flush=True)

    # 6. Save -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    visits.to_parquet(args.out, index=False)
    print(f"Wrote {len(visits)} rows x {visits.shape[1]} cols to {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day-obs-min", type=int, required=True, dest="day_obs_min")
    ap.add_argument("--day-obs-max", type=int, required=True, dest="day_obs_max")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instrument", default="lsstcam")
    ap.add_argument("--consdb-url", default="http://consdb-pq.consdb:8080/consdb",
                    dest="consdb_url")
    ap.add_argument("--efd", default="usdf_efd")
    ap.add_argument("--ofc-config-dir", required=True, dest="ofc_config_dir")
    ap.add_argument("--science-program-prefix", default="BLOCK-T539",
                    dest="science_program_prefix")
    ap.add_argument("--observation-reason", default="infocus_initial_alignment",
                    dest="observation_reason")
    ap.add_argument("--annotations", nargs="+",
                    default=["closed_loop_hexapods_10dof_trunc5",
                             "closed_loop_22dof_trunc12"])
    ap.add_argument("--seq-gap-max", type=int, default=10, dest="seq_gap_max")
    ap.add_argument("--n-vmode", type=int, default=12, dest="n_vmode")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
