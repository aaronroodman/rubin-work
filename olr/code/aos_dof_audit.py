"""AOS Degree-of-Freedom audit — per-seq extraction of the LUT / Trim / Tweak /
Applied chain for the two hexapods and the M1M3/M2 bending modes, from the EFD
with ConsDB + Butler cross-checks.  SUMMIT-RSP ONLY (EFD + embargo Butler).

Chain (Aaron's terminology, see aos-dof-terminology):
    optical_state --PID--> Tweak (visitDoF) --accumulate--> Trim (aggregatedDoF)
    Trim --(ts_ofc map)--> per-component command --+ LUT(el,T)--> Applied

TIME SCALES.  EFD `private_sndStamp` is TAI seconds since 1970 (SAL convention);
the pandas index from select_time_series is tz-aware UTC.  Butler exposure times
are TAI.  We match EVERYTHING on TAI `private_sndStamp` to avoid the ~37 s UTC/TAI
offset.  Per-visit AOS quantities (Trim/Tweak/corrections/wfe) are matched to the
correction event's sndStamp (co-emitted); slowly-varying telemetry (hexapod
position, elevation) is matched to the EXPOSURE time (state DURING the image, not
~40-70 s later when the correction is emitted).

EFD sources (all lsst.sal.*, confirmed 2026-07-13):
    MTAOS.logevent_degreeOfFreedom          aggregatedDoF0..49 (Trim), visitDoF0..49 (Tweak)
    MTAOS.logevent_wavefrontError           nollZernikeValues*/Indices* (OFC input), extraId
    MTAOS.logevent_{cameraHexapod,m2Hexapod}Correction  x,y,z,u,v,w + visitId
    MTAOS.logevent_{m1m3,m2}Correction      zForces* + visitId
    MTHexapod.logevent_compensatedPosition   x..w + salIndex (1=Cam, 2=M2)  (Applied)
    MTHexapod.logevent_uncompensatedPosition x..w + salIndex                (AOS command)
        -> LUT_hex = compensated - uncompensated
    MTM1M3.logevent_appliedActiveOpticForces zForces* + fz,mx,my            (Applied AOS forces)
    MTMount.elevation                        actualPosition

Image link: visitId = day_obs*100000 + seq_num.

Emits one row per visit: output/<day_obs>/dof_audit.parquet.
"""
import argparse
import asyncio
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

# ConsDB lives on an internal host; bypass any HTTP proxy for it -- must be set
# at import time, before ConsDbClient/requests initializes (as in nightly_table).
os.environ["no_proxy"] = (os.environ["no_proxy"] + ",.consdb") if "no_proxy" in os.environ else ".consdb"

HEX_AXES = ["x", "y", "z", "u", "v", "w"]      # hexapod: x,y,z µm ; u,v,w deg
N_DOF = 50
HEX_SAL = {1: "cam", 2: "m2"}
DOF_GROUPS = {"M2 hex": range(0, 5), "Cam hex": range(5, 10),
              "M1M3 bend": range(10, 30), "M2 bend": range(30, 50)}


# ---------------------------------------------------------------- helpers ----
def night_window(day_obs):
    d = str(int(day_obs))
    t0 = Time(f"{d[:4]}-{d[4:6]}-{d[6:]}T12:00:00", scale="utc")
    return t0, t0 + TimeDelta(1.0, format="jd")


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _utc_unix(idx):
    """tz-aware (UTC) DatetimeIndex -> UTC POSIX seconds.  We match on the pandas
    index (unambiguously UTC) rather than private_sndStamp (scale ambiguous), so
    no UTC/TAI assumption is needed -- exposure times use astropy .unix (also UTC).
    Timedelta division is resolution-safe (index may be ns or µs)."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return ((idx - _EPOCH) / pd.Timedelta("1s")).to_numpy(float)


def _numbered(row, stem, n):
    return [float(row.get(f"{stem}{i}", np.nan)) for i in range(n)]


def n_cols(df, stem):
    return sum(bool(re.fullmatch(rf"{stem}\d+", c)) for c in df.columns)


def pick_visit(df, vid, col="visitId"):
    if df is None or len(df) == 0 or col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce")
    sub = df[v == vid]
    return sub.iloc[-1] if len(sub) else None


class TS:
    """A time series matched on the (UTC) pandas index via searchsorted."""
    def __init__(self, df):
        if df is not None and len(df):
            df = df.sort_index()
            self.df, self.s = df, _utc_unix(df.index)
        else:
            self.df, self.s = df, np.array([])

    def nearest(self, t):
        if self.s.size == 0:
            return None
        i = int(np.searchsorted(self.s, t))
        i = min(max(i, 0), self.s.size - 1)
        if i > 0 and abs(self.s[i - 1] - t) < abs(self.s[i] - t):
            i -= 1
        return self.df.iloc[i]

    def before(self, t):
        if self.s.size == 0:
            return None
        i = int(np.searchsorted(self.s, t, side="right")) - 1
        return self.df.iloc[i] if i >= 0 else None


# ---------------------------------------------------------- Butler exposures ---
def get_exposures(butler, day_obs, instrument="LSSTCam"):
    """(seq) obs_type / program / UTC exposure window (POSIX s) from the Butler."""
    out = {}
    for r in butler.registry.queryDimensionRecords(
            "exposure", instrument=instrument, where=f"exposure.day_obs={int(day_obs)}"):
        ts = r.timespan
        t0 = float(ts.begin.unix) if ts and ts.begin is not None else np.nan   # astropy .unix = UTC POSIX
        t1 = float(ts.end.unix) if ts and ts.end is not None else np.nan
        out[int(r.seq_num)] = dict(
            obs_type=r.observation_type,
            program=getattr(r, "science_program", None),
            t0=t0, t1=t1,
            mjd_begin=(float(ts.begin.utc.mjd) if ts and ts.begin is not None else np.nan))
    return out


# ------------------------------------------------------------- extractor ----
class AosDofAudit:
    def __init__(self, efd_name="summit_efd"):
        from lsst_efd_client import EfdClient          # makeEfdClient() is broken on this stack
        self.efd = EfdClient(efd_name, output_mode="dataframe")

    async def _ts(self, topic, t0, t1):
        try:
            df = await self.efd.select_time_series(f"lsst.sal.{topic}", ["*"], t0, t1)
            print(f"  [efd] {topic}: {len(df)} rows")
            return df
        except Exception as e:
            print(f"  [efd] {topic}: {type(e).__name__}: {e}")
            return pd.DataFrame()

    async def fetch_night(self, day_obs, exposures=None):
        exposures = exposures or {}
        t0, t1 = night_window(day_obs)
        print(f"[dof_audit] {day_obs}  EFD window {t0.isot}..{t1.isot}")
        dof   = TS(await self._ts("MTAOS.logevent_degreeOfFreedom", t0, t1))
        wfe_r = await self._ts("MTAOS.logevent_wavefrontError", t0, t1)
        camC  = await self._ts("MTAOS.logevent_cameraHexapodCorrection", t0, t1)
        m2C   = await self._ts("MTAOS.logevent_m2HexapodCorrection", t0, t1)
        m1m3C = await self._ts("MTAOS.logevent_m1m3Correction", t0, t1)
        m2mC  = await self._ts("MTAOS.logevent_m2Correction", t0, t1)
        comp  = await self._ts("MTHexapod.logevent_compensatedPosition", t0, t1)
        uncmp = await self._ts("MTHexapod.logevent_uncompensatedPosition", t0, t1)
        aof   = TS(await self._ts("MTM1M3.logevent_appliedActiveOpticForces", t0, t1))
        elev  = TS(await self._ts("MTMount.elevation", t0, t1))
        if camC.empty:
            print("[dof_audit] no cameraHexapodCorrection -> empty table")
            return pd.DataFrame()

        wfe = TS(wfe_r)
        comp_ts = {i: TS(comp[comp.salIndex == i]) if "salIndex" in comp else TS(comp) for i in HEX_SAL}
        uncmp_ts = {i: TS(uncmp[uncmp.salIndex == i]) if "salIndex" in uncmp else TS(uncmp) for i in HEX_SAL}
        n_wfe = n_cols(wfe_r, "nollZernikeValues")
        n_m1 = max(n_cols(m1m3C, "zForces"), n_cols(aof.df if aof.df is not None else pd.DataFrame(), "zForces"))
        n_m2 = n_cols(m2mC, "zForces")

        seen, rows = set(), []
        for t, r in camC.sort_index().iterrows():
            vid = int(pd.to_numeric(r.get("visitId", -1), errors="coerce") or -1)
            if vid <= 0 or vid in seen:                    # dedupe (FAM defocus sends 2/visit)
                continue
            seen.add(vid)
            seq = vid % 100000
            exp = exposures.get(seq, {})
            corr_unix = float(pd.Timestamp(t).timestamp())            # UTC POSIX of the correction
            exp_t0 = exp.get("t0", np.nan)
            exp_mid = np.nanmean([exp.get("t0", np.nan), exp.get("t1", np.nan)])
            tel_t = exp_mid if np.isfinite(exp_mid) else corr_unix    # telemetry @ exposure

            row = dict(visit_id=vid, day_obs=vid // 100000, seq_num=seq,
                       obs_type=exp.get("obs_type"), program=exp.get("program"),
                       exp_mjd=exp.get("mjd_begin", np.nan), exp_unix=exp_t0, corr_unix=corr_unix,
                       latency_s=(corr_unix - exp_t0) if np.isfinite(exp_t0) else np.nan,
                       n_cam_corr=int((pd.to_numeric(camC.get("visitId"), errors="coerce") == vid).sum()))

            # Trim / Tweak: this visit's DOF (co-emitted with the correction)
            dr = dof.nearest(corr_unix)
            for i in range(N_DOF):
                row[f"trim{i}"]  = float(dr.get(f"aggregatedDoF{i}", np.nan)) if dr is not None else np.nan
                row[f"tweak{i}"] = float(dr.get(f"visitDoF{i}", np.nan)) if dr is not None else np.nan

            # hexapod: command (MTAOS) | applied (comp @ exposure) | AOS-cmd (uncomp) | LUT
            for ax in HEX_AXES:
                row[f"cam_cmd_{ax}"] = float(r.get(ax, np.nan))
            m2r = pick_visit(m2C, vid)
            for ax in HEX_AXES:
                row[f"m2_cmd_{ax}"] = float(m2r.get(ax, np.nan)) if m2r is not None else np.nan
            for sidx, nm in HEX_SAL.items():
                cp, up = comp_ts[sidx].nearest(tel_t), uncmp_ts[sidx].nearest(tel_t)
                for ax in HEX_AXES:
                    c = float(cp.get(ax, np.nan)) if cp is not None else np.nan
                    u = float(up.get(ax, np.nan)) if up is not None else np.nan
                    row[f"{nm}_comp_{ax}"], row[f"{nm}_uncomp_{ax}"], row[f"{nm}_lut_{ax}"] = c, u, c - u

            er = elev.nearest(tel_t)
            row["elevation"] = float(er.get("actualPosition", np.nan)) if er is not None else np.nan

            wr = pick_visit(wfe_r, vid, col="extraId")
            wr = wr if wr is not None else wfe.nearest(corr_unix)
            row["wfe_noll"]   = _numbered(wr, "nollZernikeIndices", n_wfe) if wr is not None else None
            row["wfe_values"] = _numbered(wr, "nollZernikeValues", n_wfe) if wr is not None else None

            m1c = pick_visit(m1m3C, vid)
            row["m1m3_cmd_zForces"] = _numbered(m1c, "zForces", n_m1) if m1c is not None else None
            ar = aof.nearest(tel_t)
            row["m1m3_applied_zForces"] = _numbered(ar, "zForces", n_m1) if ar is not None else None
            for k in ("fz", "mx", "my"):
                row[f"m1m3_applied_{k}"] = float(ar.get(k, np.nan)) if ar is not None else np.nan
            m2c = pick_visit(m2mC, vid)
            row["m2_cmd_zForces"] = _numbered(m2c, "zForces", n_m2) if m2c is not None else None
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("seq_num").reset_index(drop=True)
        lat = df["latency_s"].to_numpy()
        print(f"[dof_audit] {len(df)} visits; wfe={n_wfe}, m1m3={n_m1}, m2={n_m2} forces; "
              f"correction latency (corr - exposure, TAI) median {np.nanmedian(lat):.1f}s "
              f"[{np.nanmin(lat):.1f}..{np.nanmax(lat):.1f}]")
        return df


def add_consdb(df, url="http://consdb-pq.consdb:8080/consdb", instrument="lsstcam"):
    """ConsDB elevation + z4..z28 per visit for the EFD<->ConsDB check."""
    if df.empty:
        return df
    try:
        from lsst.summit.utils import ConsDbClient          # no_proxy set at module import
        cdb = ConsDbClient(url)
    except Exception as e:
        print(f"[dof_audit] ConsDB client unavailable: {type(e).__name__}: {e}")
        return df
    d0, d1 = int(df.day_obs.min()), int(df.day_obs.max())
    zsel = ", ".join(f"q.z{j} AS cdb_z{j}" for j in range(4, 29))
    q = (f"SELECT v.visit_id AS visit_id, v.altitude AS cdb_elevation, {zsel} "
         f"FROM cdb_{instrument}.visit1 AS v "
         f"JOIN cdb_{instrument}.ccdvisit1_quicklook AS q ON q.ccdvisit1_id = v.visit_id "
         f"WHERE v.day_obs >= {d0} AND v.day_obs <= {d1}")
    try:
        res = cdb.query(q)
        cdb_df = res.to_pandas() if hasattr(res, "to_pandas") else res
        return df.merge(cdb_df, on="visit_id", how="left")
    except Exception as e:
        print(f"[dof_audit] ConsDB query skipped: {type(e).__name__}: {e}")
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

    exposures = {}
    if not args.no_butler:
        try:
            from lsst.summit.utils import butlerUtils
            exposures = get_exposures(butlerUtils.makeDefaultButler("LSSTCam", embargo=True), args.day_obs)
            print(f"[dof_audit] Butler: {len(exposures)} exposures for {args.day_obs}")
        except Exception as e:
            print(f"[dof_audit] Butler exposures skipped: {type(e).__name__}: {e}")

    audit = AosDofAudit(args.efd_name)
    df = asyncio.get_event_loop().run_until_complete(audit.fetch_night(args.day_obs, exposures))
    if not args.no_consdb and not df.empty:
        df = add_consdb(df)

    out = Path(args.out_root) / str(args.day_obs) / "dof_audit.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[dof_audit] wrote {out}  ({len(df)} visits, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
