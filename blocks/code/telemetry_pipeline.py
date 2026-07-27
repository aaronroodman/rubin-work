"""Shared AOS telemetry collection for the blocks build scripts.

Given a ``visits`` DataFrame (with ``day_obs``, ``seq_num``, ``obs_start``,
``obs_end``, ``visit_id``), attach the full AOS telemetry set:
  - aggregated DOF Trim (dof0-49),
  - hexapod LUT (lut_dof0-9) + mirror LUT (lut_dof10-49) via ts_ofc,
  - geom v-modes (v1..N),
  - per-corner retrieved Zernikes,
  - thermal + wind telemetry (ESS temps, delta-Ts, M1M3 gradients, TMA truss,
    dome wind) via olr/code/telemetry.py.

Used by both ``build_t539_table.py`` (converged-image selection) and
``build_night_table.py`` (all visits in a night).  ConsDB + EFD only -- no
Butler -- so it runs on the USDF or Summit RSP.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

# Shared helpers: aos/code (DOF, v-modes, Zernikes, LUT) + olr/code (telemetry).
REPO = Path(__file__).resolve().parents[2]          # -> rubin-work/
sys.path.insert(0, str(REPO / "aos" / "code"))
sys.path.insert(0, str(REPO / "olr" / "code"))
import aos_trim       # noqa: E402
import aos_state      # noqa: E402
import telemetry      # noqa: E402

CORNERS = {191: "R00_SW0", 195: "R04_SW0", 199: "R40_SW0", 203: "R44_SW0"}
ZK_NOLL = [z for z in range(4, 27) if z not in (20, 21)]


def collect_telemetry(visits, args, client, efd=None, with_zernikes=True):
    """Attach DOF/LUT/v-mode/Zernike/thermal/wind telemetry to ``visits``.

    ``args`` supplies ``efd``, ``ofc_config_dir``, ``instrument`` and ``n_vmode``.
    ``client`` is a ConsDB client (for the obs_start anchoring).  Returns the
    enriched DataFrame.
    """
    from lsst_efd_client import EfdClient
    if efd is None:
        efd = EfdClient(args.efd, output_mode="dataframe")
    fit_table = Table.from_pandas(visits[["day_obs", "seq_num"]].astype(int))

    # --- aggregated DOF (Trim) ---
    trim, dof_info = aos_trim.fetch_aggregated_dof_for_visits(
        fit_table, efd_client=efd, consdb_client=client)
    visits = pd.concat([visits, pd.DataFrame(
        trim, columns=[f"dof{i}" for i in range(aos_trim.N_DOF)],
        index=visits.index)], axis=1)
    print(f"DOF finite: {dof_info['n_dof']}/{len(visits)}", flush=True)

    # --- hexapod LUT (compensationOffset) -> lut_dof0-9 ---
    lut, lut_info = aos_trim.fetch_hexapod_lut_for_visits(
        fit_table, efd_client=efd, consdb_client=client)
    visits = pd.concat([visits, pd.DataFrame(
        lut, columns=[f"lut_dof{i}" for i in range(10)],
        index=visits.index)], axis=1)
    print(f"hexapod LUT finite: {lut_info['n_lut']}/{len(visits)}", flush=True)

    # --- mirror LUT (M1M3 elevation + M2 gravity forces -> bending modes) ---
    try:
        mlut, mlut_info = aos_trim.fetch_mirror_lut_for_visits(
            fit_table, config_dir=args.ofc_config_dir, efd_client=efd,
            consdb_client=client)
        visits = pd.concat([visits, pd.DataFrame(
            mlut, columns=[f"lut_dof{10 + i}" for i in range(40)],
            index=visits.index)], axis=1)
        print(f"mirror LUT finite: {mlut_info['n_lut']}/{len(visits)}", flush=True)
    except Exception as e:
        print(f"mirror LUT skipped ({type(e).__name__}: {e})", flush=True)

    # --- geom v-modes from the DOF Trim ---
    se = aos_state.make_state_estimator(config_dir=args.ofc_config_dir,
                                        dof_set="standard_22")
    dof_mat = visits[[f"dof{i}" for i in range(aos_trim.N_DOF)]].to_numpy(dtype=float)
    vmodes = aos_state.vmodes_from_dofs(dof_mat, se, n_modes=args.n_vmode)
    visits = pd.concat([visits, pd.DataFrame(
        vmodes, columns=[f"v{j + 1}" for j in range(args.n_vmode)],
        index=visits.index)], axis=1)
    print(f"v-modes finite: {int(np.isfinite(vmodes).all(axis=1).sum())}/{len(visits)}",
          flush=True)

    # --- per-corner retrieved wavefront Zernikes (mostly N/A for science) ---
    if with_zernikes:
        try:
            zk_df = aos_state.fetch_corner_zernikes_consdb(
                client, visits["visit_id"].values, instrument=args.instrument,
                zk_noll=ZK_NOLL, corners=CORNERS)
            visits = visits.merge(zk_df, left_on="visit_id", right_index=True, how="left")
        except Exception as e:
            print(f"corner Zernikes skipped ({type(e).__name__}: {e})", flush=True)

    # --- thermal + wind telemetry (shared helper; per-night, robust) ---
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
    return visits
