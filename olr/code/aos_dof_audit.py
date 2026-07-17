"""AOS Degree-of-Freedom audit — per-seq extraction of the LUT / Trim / Tweak /
Applied chain for the two hexapods and the M1M3/M2 bending modes, from the EFD
with ConsDB + Butler cross-checks.  SUMMIT-RSP ONLY (EFD + embargo Butler).

Chain (Aaron's terminology, see aos-dof-terminology):
    optical_state --PID--> Tweak (visitDoF) --accumulate--> Trim (aggregatedDoF)
    Trim --(ts_ofc map)--> per-component command --+ LUT(el,T)--> Applied

EFD sources (all lsst.sal.*, confirmed 2026-07-13):
    MTAOS.logevent_degreeOfFreedom          aggregatedDoF0..49 (Trim), visitDoF0..49 (Tweak)
    MTAOS.logevent_wavefrontError           nollZernikeValues*/Indices* (OFC input), extraId
    MTAOS.logevent_cameraHexapodCorrection  x,y,z,u,v,w + visitId  (Cam-hex command)
    MTAOS.logevent_m2HexapodCorrection      x,y,z,u,v,w + visitId  (M2-hex command)
    MTAOS.logevent_m1m3Correction           zForces0..155 + visitId
    MTAOS.logevent_m2Correction             zForces0..71  + visitId
    MTHexapod.logevent_compensatedPosition   x..w + salIndex (1=Cam, 2=M2)  (Applied)
    MTHexapod.logevent_uncompensatedPosition x..w + salIndex                (AOS command)
        -> LUT_hex = compensated - uncompensated
    MTM1M3.logevent_appliedActiveOpticForces zForces0..155 + fz,mx,my       (Applied AOS forces)
    MTMount.elevation                        actualPosition

Image link: visitId = day_obs*100000 + seq_num; EFD telemetry matched by time.

Emits one row per visit: output/<day_obs>/dof_audit.parquet.
"""
import argparse
import asyncio
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

HEX_AXES = ["x", "y", "z", "u", "v", "w"]      # hexapod: x,y,z µm ; u,v,w deg
N_DOF = 50
HEX_SAL = {1: "cam", 2: "m2"}                  # MTHexapod salIndex -> our name
DOF_GROUPS = {"M2 hex": range(0, 5), "Cam hex": range(5, 10),
              "M1M3 bend": range(10, 30), "M2 bend": range(30, 50)}


# ---------------------------------------------------------------- helpers ----
def night_window(day_obs):
    """UTC [t0, t1] covering the Chilean night of day_obs (local noon->noon)."""
    d = str(int(day_obs))
    t0 = Time(f"{d[:4]}-{d[4:6]}-{d[6:]}T12:00:00", scale="utc")
    return t0, t0 + TimeDelta(1.0, format="jd")


def _numbered(df, stem, n=None):
    """Pull stem0..stem{n-1} from a Series/row -> list; n auto if None."""
    if n is None:
        n = 1 + max([int(m.group(1)) for c in df.index
                     for m in [re.fullmatch(rf"{stem}(\d+)", c)] if m] or [-1])
    return [float(df.get(f"{stem}{i}", np.nan)) for i in range(n)]


def nearest(df, t):
    """Row of a time-indexed EFD df nearest to Timestamp t (None if empty)."""
    if df is None or len(df) == 0:
        return None
    i = df.index.get_indexer([t], method="nearest")[0]
    return df.iloc[i] if i >= 0 else None


def pick_visit(df, vid, col="visitId"):
    """Last row of df whose `col` == vid (None if absent)."""
    if df is None or len(df) == 0 or col not in df.columns:
        return None
    sub = df[df[col].astype("int64", errors="ignore") == vid]
    return sub.iloc[-1] if len(sub) else None


def n_cols(df, stem):
    return sum(bool(re.fullmatch(rf"{stem}\d+", c)) for c in df.columns)


# ------------------------------------------------------------- extractor ----
class AosDofAudit:
    def __init__(self, efd_name="summit_efd"):
        from lsst_efd_client import EfdClient          # direct ctor: makeEfdClient() is broken
        self.efd = EfdClient(efd_name, output_mode="dataframe")

    async def _ts(self, topic, t0, t1):
        try:
            df = await self.efd.select_time_series(f"lsst.sal.{topic}", ["*"], t0, t1)
            print(f"  [efd] {topic}: {len(df)} rows")
            return df
        except Exception as e:
            print(f"  [efd] {topic}: {type(e).__name__}: {e}")
            return pd.DataFrame()

    async def fetch_night(self, day_obs):
        t0, t1 = night_window(day_obs)
        print(f"[dof_audit] {day_obs}  EFD window {t0.isot}..{t1.isot}")
        dof   = await self._ts("MTAOS.logevent_degreeOfFreedom", t0, t1)
        wfe   = await self._ts("MTAOS.logevent_wavefrontError", t0, t1)
        camC  = await self._ts("MTAOS.logevent_cameraHexapodCorrection", t0, t1)
        m2C   = await self._ts("MTAOS.logevent_m2HexapodCorrection", t0, t1)
        m1m3C = await self._ts("MTAOS.logevent_m1m3Correction", t0, t1)
        m2mC  = await self._ts("MTAOS.logevent_m2Correction", t0, t1)
        comp  = await self._ts("MTHexapod.logevent_compensatedPosition", t0, t1)
        uncmp = await self._ts("MTHexapod.logevent_uncompensatedPosition", t0, t1)
        aof   = await self._ts("MTM1M3.logevent_appliedActiveOpticForces", t0, t1)
        elev  = await self._ts("MTMount.elevation", t0, t1)

        if camC.empty:
            print("[dof_audit] no cameraHexapodCorrection this night -> empty table")
            return pd.DataFrame()

        n_wfe = n_cols(wfe, "nollZernikeValues")
        n_m1m3 = max(n_cols(m1m3C, "zForces"), n_cols(aof, "zForces"))
        n_m2f = n_cols(m2mC, "zForces")

        # one row per visit; anchor on the Cam-hex correction (has visitId + time)
        seen, rows = set(), []
        for t, r in camC.iterrows():
            vid = int(r.get("visitId", -1))
            if vid <= 0 or vid in seen:            # dedupe (FAM defocus sends 2/visit)
                continue
            seen.add(vid)
            row = dict(visit_id=vid, day_obs=vid // 100000, seq_num=vid % 100000,
                       corr_time=t, n_cam_corr=int((camC.get("visitId") == vid).sum()))

            # --- Trim / Tweak (nearest degreeOfFreedom to the correction time) ---
            dr = nearest(dof, t)
            for i in range(N_DOF):
                row[f"trim{i}"]  = float(dr.get(f"aggregatedDoF{i}", np.nan)) if dr is not None else np.nan
                row[f"tweak{i}"] = float(dr.get(f"visitDoF{i}", np.nan)) if dr is not None else np.nan

            # --- hexapod: command (MTAOS) | applied (comp) | AOS-cmd (uncomp) | LUT ---
            for ax in HEX_AXES:
                row[f"cam_cmd_{ax}"] = float(r.get(ax, np.nan))
            m2r = pick_visit(m2C, vid)
            for ax in HEX_AXES:
                row[f"m2_cmd_{ax}"] = float(m2r.get(ax, np.nan)) if m2r is not None else np.nan
            for sidx, nm in HEX_SAL.items():
                cp = nearest(comp[comp.salIndex == sidx] if "salIndex" in comp else comp, t)
                up = nearest(uncmp[uncmp.salIndex == sidx] if "salIndex" in uncmp else uncmp, t)
                for ax in HEX_AXES:
                    c = float(cp.get(ax, np.nan)) if cp is not None else np.nan
                    u = float(up.get(ax, np.nan)) if up is not None else np.nan
                    row[f"{nm}_comp_{ax}"], row[f"{nm}_uncomp_{ax}"] = c, u
                    row[f"{nm}_lut_{ax}"] = c - u

            # --- elevation (for the LUT lookup / consistency) ---
            er = nearest(elev, t)
            row["elevation"] = float(er.get("actualPosition", np.nan)) if er is not None else np.nan

            # --- OFC input (wavefrontError, matched by extraId then time) ---
            wr = pick_visit(wfe, vid, col="extraId")
            wr = wr if wr is not None else nearest(wfe, t)
            row["wfe_noll"]   = _numbered(wr, "nollZernikeIndices", n_wfe) if wr is not None else None
            row["wfe_values"] = _numbered(wr, "nollZernikeValues", n_wfe) if wr is not None else None

            # --- mirror-mode forces (command + applied) ---
            m1c = pick_visit(m1m3C, vid)
            row["m1m3_cmd_zForces"] = _numbered(m1c, "zForces", n_m1m3) if m1c is not None else None
            ar = nearest(aof, t)
            row["m1m3_applied_zForces"] = _numbered(ar, "zForces", n_m1m3) if ar is not None else None
            for k in ("fz", "mx", "my"):
                row[f"m1m3_applied_{k}"] = float(ar.get(k, np.nan)) if ar is not None else np.nan
            m2c = pick_visit(m2mC, vid)
            row["m2_cmd_zForces"] = _numbered(m2c, "zForces", n_m2f) if m2c is not None else None

            rows.append(row)

        df = pd.DataFrame(rows).sort_values("seq_num").reset_index(drop=True)
        print(f"[dof_audit] {len(df)} visits; wfe terms={n_wfe}, "
              f"m1m3 forces={n_m1m3}, m2 forces={n_m2f}")
        return df


# ---------------------------------------------- Butler / ConsDB enrichment ---
def add_butler_exposure(df, butler=None, instrument="LSSTCam"):
    """Add exposure obs_type + precise begin MJD from the (embargo) Butler."""
    if butler is None or df.empty:
        return df
    days = sorted(df.day_obs.unique())
    recs = {}
    for d in days:
        for r in butler.registry.queryDimensionRecords(
                "exposure", instrument=instrument, where=f"exposure.day_obs={int(d)}"):
            recs[(int(d), int(r.seq_num))] = (r.observation_type,
                                              float(r.timespan.begin.mjd) if r.timespan else np.nan)
    df["obs_type"] = [recs.get((int(a), int(b)), (None, np.nan))[0]
                      for a, b in zip(df.day_obs, df.seq_num)]
    df["exp_mjd"] = [recs.get((int(a), int(b)), (None, np.nan))[1]
                     for a, b in zip(df.day_obs, df.seq_num)]
    return df


def add_consdb(df, consdb=None, instrument="lsstcam"):
    """Add ConsDB elevation + z4..z28 (per-visit) for the EFD<->ConsDB check."""
    if consdb is None or df.empty:
        return df
    d0, d1 = int(df.day_obs.min()), int(df.day_obs.max())
    zsel = ", ".join(f"q.z{j} AS cdb_z{j}" for j in range(4, 29))
    q = (f"SELECT v.visit_id AS visit_id, v.altitude AS cdb_elevation, {zsel} "
         f"FROM cdb_{instrument}.visit1 AS v "
         f"JOIN cdb_{instrument}.ccdvisit1_quicklook AS q ON q.ccdvisit1_id = v.visit_id "
         f"WHERE v.day_obs >= {d0} AND v.day_obs <= {d1}")
    try:
        cdb = consdb.query(q).to_pandas() if hasattr(consdb.query(q), "to_pandas") else consdb.query(q)
        return df.merge(cdb, on="visit_id", how="left")
    except Exception as e:
        print(f"[dof_audit] ConsDB skipped: {type(e).__name__}: {e}")
        return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-obs", type=int, required=True)
    ap.add_argument("--efd-name", default="summit_efd")
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--no-butler", action="store_true")
    ap.add_argument("--no-consdb", action="store_true")
    args = ap.parse_args()

    audit = AosDofAudit(args.efd_name)
    df = asyncio.get_event_loop().run_until_complete(audit.fetch_night(args.day_obs))

    if not args.no_butler:
        try:
            from lsst.summit.utils import butlerUtils
            df = add_butler_exposure(df, butlerUtils.makeDefaultButler("LSSTCam", embargo=True))
        except Exception as e:
            print(f"[dof_audit] Butler enrichment skipped: {type(e).__name__}: {e}")
    if not args.no_consdb:
        try:
            from lsst.summit.utils import ConsDbClient
            df = add_consdb(df, ConsDbClient("http://consdb-pq.consdb:8080/consdb"))
        except Exception as e:
            print(f"[dof_audit] ConsDB enrichment skipped: {type(e).__name__}: {e}")

    out = Path(args.out_root) / str(args.day_obs) / "dof_audit.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[dof_audit] wrote {out}  ({len(df)} visits, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
