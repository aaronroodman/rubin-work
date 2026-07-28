"""Shared AOS telemetry collection for the blocks build scripts.

Given a ``visits`` DataFrame (with ``day_obs``, ``seq_num``, ``obs_start``,
``obs_end``, ``visit_id``), attach the full AOS telemetry set: aggregated DOF
Trim (dof0-49), hexapod + mirror LUT (lut_dof0-49), geom v-modes (v1..N),
per-corner Zernikes, and thermal + wind.

Two interchangeable telemetry sources (same output schema), chosen by
``args.telemetry_source`` / the ``source=`` arg:

  * ``"consdb"`` (default) -- per-exposure MEANS from the ConsDB Consolidated
    (transformed) EFD via ``aos_consdb_efd`` (~2 queries; no per-visit raw EFD).
    Fast, works on RSP + slaciana.  M1M3 spatial gradients aren't in the
    transform, so they are still pulled from the raw EFD (per night) unless
    ``args.gradients_from_efd`` is False.  Adds the separate M2-temperature
    bending modes (m2temp_dof0-19) and hexapod Trim (trim_hex_dof0-9).
  * ``"efd"`` -- the original raw-EFD path (``aos_trim`` per-visit/-night +
    ``olr/telemetry``).  Kept as a cross-check / for time ranges the ConsDB
    transform doesn't cover.

Used by ``build_t539_table.py`` and ``build_night_table.py``.  ConsDB + EFD
only (no Butler).
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
import aos_trim         # noqa: E402
import aos_state        # noqa: E402
import aos_consdb_efd   # noqa: E402
import telemetry        # noqa: E402

CORNERS = {191: "R00_SW0", 195: "R04_SW0", 199: "R40_SW0", 203: "R44_SW0"}
ZK_NOLL = [z for z in range(4, 27) if z not in (20, 21)]


def collect_telemetry(visits, args, client, efd=None, with_zernikes=True, source=None):
    """Attach DOF/LUT/v-mode/Zernike/thermal/wind telemetry to ``visits``.

    ``source`` overrides ``args.telemetry_source`` (default ``"consdb"``).
    ``args`` supplies ``efd``, ``ofc_config_dir``, ``instrument``, ``n_vmode``.
    """
    source = source or getattr(args, "telemetry_source", "consdb")
    if source == "consdb":
        return _collect_consdb(visits, args, client, efd, with_zernikes)
    return _collect_efd(visits, args, client, efd, with_zernikes)


# ----------------------------------------------------------------------------
# shared pieces
# ----------------------------------------------------------------------------
def _add_vmodes(visits, args):
    se = aos_state.make_state_estimator(config_dir=args.ofc_config_dir,
                                        dof_set="standard_22")
    dof_mat = visits[[f"dof{i}" for i in range(aos_trim.N_DOF)]].to_numpy(dtype=float)
    vmodes = aos_state.vmodes_from_dofs(dof_mat, se, n_modes=args.n_vmode)
    add = pd.DataFrame(vmodes, columns=[f"v{j + 1}" for j in range(args.n_vmode)],
                       index=visits.index)
    print(f"v-modes finite: {int(np.isfinite(vmodes).all(axis=1).sum())}/{len(visits)}",
          flush=True)
    return pd.concat([visits, add], axis=1)


def _add_corner_zernikes(visits, args, client):
    try:
        zk_df = aos_state.fetch_corner_zernikes_consdb(
            client, visits["visit_id"].values, instrument=args.instrument,
            zk_noll=ZK_NOLL, corners=CORNERS)
        visits = visits.merge(zk_df, left_on="visit_id", right_index=True, how="left")
    except Exception as e:
        print(f"corner Zernikes skipped ({type(e).__name__}: {e})", flush=True)
    return visits


def _attach_m1m3_gradients_efd(visits, efd):
    """M1M3 bulk thermal gradients (x/y/z/radial) from the raw EFD, per night --
    the one telemetry item the ConsDB transform does not carry."""
    base = (visits[["day_obs", "seq_num", "obs_start"]]
            .rename(columns={"seq_num": "seq"}))
    parts = []
    for day, sub in base.groupby("day_obs"):
        try:
            g = telemetry._run_coro(telemetry.get_m1m3_gradients(efd, sub.copy()))
            parts.append(g[["day_obs", "seq"] + telemetry.GRAD_COLS])
        except Exception as e:
            print(f"(M1M3 gradients failed day {day} [{type(e).__name__}])", flush=True)
            s = sub[["day_obs", "seq"]].copy()
            for c in telemetry.GRAD_COLS:
                s[c] = np.nan
            parts.append(s)
    grad = (pd.concat(parts, ignore_index=True)
            .rename(columns={"seq": "seq_num"}))
    visits = visits.merge(grad, on=["day_obs", "seq_num"], how="left")
    if "z_gradient" in visits:
        print(f"z_gradient finite: {int(visits['z_gradient'].notna().sum())}/{len(visits)}",
              flush=True)
    return visits


# ----------------------------------------------------------------------------
# ConsDB transformed-EFD path (default)
# ----------------------------------------------------------------------------
def _collect_consdb(visits, args, client, efd, with_zernikes):
    visits = aos_consdb_efd.collect_consdb_telemetry(
        client, visits, config_dir=args.ofc_config_dir)
    visits = _add_vmodes(visits, args)
    if getattr(args, "gradients_from_efd", True):
        from lsst_efd_client import EfdClient
        if efd is None:
            efd = EfdClient(args.efd, output_mode="dataframe")
        visits = _attach_m1m3_gradients_efd(visits, efd)
    if with_zernikes:
        visits = _add_corner_zernikes(visits, args, client)
    return visits


# ----------------------------------------------------------------------------
# raw-EFD path (cross-check / fallback)
# ----------------------------------------------------------------------------
def _collect_efd(visits, args, client, efd, with_zernikes):
    from lsst_efd_client import EfdClient
    if efd is None:
        efd = EfdClient(args.efd, output_mode="dataframe")
    fit_table = Table.from_pandas(visits[["day_obs", "seq_num"]].astype(int))

    trim, dof_info = aos_trim.fetch_aggregated_dof_for_visits(
        fit_table, efd_client=efd, consdb_client=client)
    visits = pd.concat([visits, pd.DataFrame(
        trim, columns=[f"dof{i}" for i in range(aos_trim.N_DOF)],
        index=visits.index)], axis=1)
    print(f"DOF finite: {dof_info['n_dof']}/{len(visits)}", flush=True)

    lut, lut_info = aos_trim.fetch_hexapod_lut_for_visits(
        fit_table, efd_client=efd, consdb_client=client)
    visits = pd.concat([visits, pd.DataFrame(
        lut, columns=[f"lut_dof{i}" for i in range(10)],
        index=visits.index)], axis=1)
    print(f"hexapod LUT finite: {lut_info['n_lut']}/{len(visits)}", flush=True)

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

    visits = _add_vmodes(visits, args)
    if with_zernikes:
        visits = _add_corner_zernikes(visits, args, client)

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
