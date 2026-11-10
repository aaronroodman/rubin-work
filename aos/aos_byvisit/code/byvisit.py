"""Per-visit audit of one MTAOS closed-loop step.

For a chosen visit this module collects, from the Engineering Facilities Database (EFD)
and the Consolidated Database (ConsDB):

- the corner wavefront sent to the Optical Feedback Controller (OFC)
  (``MTAOS.logevent_wavefrontError``, the Zernike *deviation*, Camera Coordinate System
  (CCS): ``dof_state`` reproduces the summit only when given the rotator angle), and the
  ConsDB ``ccdvisit1_quicklook`` z4..z28 for the same detectors;
- the Wavefront Estimation Pipeline (WEP) log lines MTAOS wrote for that image (donut
  sizes, tables found);
- the PID input and output, parsed from the MTAOS log, and the gains, Tweak
  (``visitDoF``) and Trim (``aggregatedDoF``) from ``MTAOS.logevent_degreeOfFreedom``;
- when the correction was issued and which exposure it first applied to;

and reproduces the OFC state estimate and PID step with the summit versions of
``ts_ofc`` and ``ts_config_mttcs``.

The state estimate and v-modes come from ts_ofc's ``StateEstimator`` built from the
summit config, not from `rubinwork.aos_state.make_state_estimator`: the stack's
``ts_ofc`` (a development branch) and local config differ from the summit's by 4-28 %
relative in the recovered state, and the point here is to reproduce what the summit did.

Two corrections matter for a visit V: the one **in effect** during V (computed from an
earlier image, issued before V opened) and the one **computed from** V's own wavefront.

DOF units: µm for hexapod translations and bending modes, deg for hexapod tilts. The
control set is 22 DOF (both hexapods, M1M3 B1-B7, M2 B1-B5) with 12 v-modes.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

N_DOF = 50
DOF22 = list(range(0, 17)) + list(range(30, 35))
N_VMODES = 12
ZN21 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
CORNER_IDS = [191, 195, 199, 203]
CORNER_NAMES = {191: "R00_SW0", 195: "R04_SW0", 199: "R40_SW0", 203: "R44_SW0"}
HEX_AXES = ["dz", "dx", "dy", "rx", "ry"]
DOF_NAMES = ([f"M2 {a}" for a in HEX_AXES] + [f"Cam {a}" for a in HEX_AXES]
             + [f"M1M3 B{i}" for i in range(1, 21)] + [f"M2 B{i}" for i in range(1, 21)])
DOF_UNITS = (["um", "um", "um", "deg", "deg"] * 2) + ["um"] * 40

MTAOS_TOPICS = [
    "logevent_wavefrontError", "logevent_degreeOfFreedom", "logevent_rejectedDegreeOfFreedom",
    "logevent_cameraHexapodCorrection", "logevent_m2HexapodCorrection",
    "logevent_m1m3Correction", "logevent_m2Correction", "logevent_closedLoopState",
    "logevent_mirrorStresses", "logevent_logMessage", "command_issueCorrection",
    "command_offsetDOF", "command_resetCorrection", "command_rejectCorrection",
]

# Versions the summit MTAOS reported (softwareVersions, configurationApplied) for
# 20261108. Check `software_versions` for a later night.
TS_OFC_TAG = "v4.8.1"
TS_CONFIG_MTTCS_TAG = "v0.22.12"
MTAOS_CONFIG = "v13"
# Normalization that reproduces the logged PID error; the summit's default.yaml does not.
NORM_YAML = "range0.5_fwhm-0.15.yaml"


# ------------------------------------------------------------------ data access ---
def exposures(cdb, day_obs, seq_lo, seq_hi):
    """ConsDB exposure rows for a range of seq_num.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
        ConsDB client.
    day_obs : `int`
        Observing night, YYYYMMDD.
    seq_lo, seq_hi : `int`
        Inclusive seq_num range.

    Returns
    -------
    table : `pandas.DataFrame`
        Indexed by seq_num, with ``t0_utc`` and ``t1_utc`` (`astropy.time.Time`, UTC)
        from ``obs_start`` / ``obs_end`` (TAI in ConsDB).
    """
    q = ("SELECT seq_num, exposure_id, obs_start, obs_end, img_type, observation_reason, "
         "science_program, band, exp_time, altitude, azimuth FROM cdb_lsstcam.exposure "
         f"WHERE day_obs={int(day_obs)} AND seq_num BETWEEN {int(seq_lo)} AND {int(seq_hi)} "
         "ORDER BY seq_num")
    res = cdb.query(q)
    ex = (res.to_pandas() if hasattr(res, "to_pandas") else res).set_index("seq_num")
    ex["t0_utc"] = [Time(t, scale="tai").utc for t in ex.obs_start]
    ex["t1_utc"] = [Time(t, scale="tai").utc for t in ex.obs_end]
    return ex


async def fetch_window(efd, t0, t1):
    """All MTAOS topics in a UTC window, plus rotator, elevation and hexapod positions.

    Parameters
    ----------
    efd : `lsst_efd_client.EfdClient`
        EFD client.
    t0, t1 : `astropy.time.Time`
        Window.

    Returns
    -------
    ev : `dict` [`str`, `pandas.DataFrame`]
        Keyed by MTAOS topic name, and ``rotator``, ``elevation``, ``hex_comp``,
        ``hex_uncomp``.
    """
    ev = {}
    for t in MTAOS_TOPICS:
        try:
            ev[t] = await efd.select_time_series(f"lsst.sal.MTAOS.{t}", ["*"], t0, t1)
        except Exception:
            ev[t] = pd.DataFrame()
    ev["rotator"] = await efd.select_time_series(
        "lsst.sal.MTRotator.rotation", ["actualPosition"], t0, t1)
    ev["elevation"] = await efd.select_time_series(
        "lsst.sal.MTMount.elevation", ["actualPosition"], t0, t1)
    # Hexapod events are sparse; reach back so the state before t0 is known.
    t0h = t0 - TimeDelta(1800, format="sec")
    for key, topic in [("hex_comp", "compensatedPosition"),
                       ("hex_uncomp", "uncompensatedPosition")]:
        ev[key] = await efd.select_time_series(
            f"lsst.sal.MTHexapod.logevent_{topic}", ["*"], t0h, t1)
    return ev


async def software_versions(efd, t):
    """MTAOS software and configuration versions in force at time ``t``.

    Returns
    -------
    versions : `dict` [`str`, `str`]
    """
    t0 = t - TimeDelta(14, format="jd")
    sv = await efd.select_time_series("lsst.sal.MTAOS.logevent_softwareVersions",
                                      ["cscVersion", "subsystemVersions"], t0, t)
    ca = await efd.select_time_series("lsst.sal.MTAOS.logevent_configurationApplied",
                                      ["schemaVersion", "url", "version"], t0, t)
    out = {}
    if len(sv):
        out.update(csc=sv.cscVersion.iloc[-1], subsystems=sv.subsystemVersions.iloc[-1])
    if len(ca):
        out.update(config_schema=ca.schemaVersion.iloc[-1], config_url=ca.url.iloc[-1],
                   config_version=ca.version.iloc[-1])
    return out


def _ts(t):
    return pd.Timestamp(t.utc.datetime, tz="UTC")


def _mean_in(df, col, t0, t1):
    if df is None or len(df) == 0:
        return np.nan
    s = df[(df.index >= _ts(t0)) & (df.index <= _ts(t1))]
    return float(s[col].mean()) if len(s) else np.nan


def exposure_pointing(ev, ex):
    """Mean rotator angle and elevation over each exposure.

    Returns
    -------
    ex : `pandas.DataFrame`
        Copy of `ex` with ``rot_deg`` (MTRotator actualPosition, deg) and ``el_deg``
        (MTMount actualPosition, deg).
    """
    ex = ex.copy()
    ex["rot_deg"] = [_mean_in(ev["rotator"], "actualPosition", a, b)
                     for a, b in zip(ex.t0_utc, ex.t1_utc)]
    ex["el_deg"] = [_mean_in(ev["elevation"], "actualPosition", a, b)
                    for a, b in zip(ex.t0_utc, ex.t1_utc)]
    return ex


def _vec(row, stem, n):
    return np.array([float(row[f"{stem}{i}"]) for i in range(n)])


# ------------------------------------------------------------------ OFC steps -----
def ofc_steps(ev, ex):
    """One row per OFC step (``degreeOfFreedom`` event) in the window.

    Parameters
    ----------
    ev : `dict`
        From `fetch_window`.
    ex : `pandas.DataFrame`
        From `exposures`.

    Returns
    -------
    steps : `pandas.DataFrame`
        Indexed by source seq_num, with the event time, the time the correction was
        issued (next ``command_issueCorrection``), the first exposure that opened after
        it (``applied_to_seq``), the gains, ``tweak`` (visitDoF, 50) and ``trim``
        (aggregatedDoF, 50).
    """
    d = ev["logevent_degreeOfFreedom"].sort_index()
    issued = ev["command_issueCorrection"].sort_index().index
    rows = []
    for i, (t, r) in enumerate(d.iterrows()):
        t_next = d.index[i + 1] if i + 1 < len(d) else None
        cand = issued[(issued > t) & ((issued < t_next) if t_next is not None else True)]
        t_iss = cand[0] if len(cand) else pd.NaT
        applied = np.nan
        if t_iss is not pd.NaT:
            later = [s for s, t0 in zip(ex.index, ex.t0_utc) if _ts(t0) > t_iss]
            applied = later[0] if later else np.nan
        vid = int(r.visitId)
        rows.append(dict(source_seq=vid % 100000, visit_id=vid, t_dof=t, t_issued=t_iss,
                         applied_to_seq=applied, kp=_vec(r, "kpGain", N_DOF),
                         ki=_vec(r, "kiGain", N_DOF), kd=_vec(r, "kdGain", N_DOF),
                         tweak=_vec(r, "visitDoF", N_DOF), trim=_vec(r, "aggregatedDoF", N_DOF),
                         optical_state_efd=_vec(r, "opticalState", N_DOF)))
    return pd.DataFrame(rows).set_index("source_seq")


def wavefront(ev, visit_id):
    """Corner Zernike deviation sent to the OFC for one image.

    Returns
    -------
    wfe : `pandas.DataFrame`
        Index sensorId, columns Z4..Z26 (`ZN21`), µm of wavefront, CCS. NaN rows are
        corners with no estimate.
    """
    w = ev["logevent_wavefrontError"]
    w = w[w.visitId == visit_id].sort_values("sensorId")
    noll = [int(w.iloc[0][f"nollZernikeIndices{i}"]) for i in range(len(ZN21))] if len(w) else ZN21
    assert noll == ZN21, f"unexpected Noll indices {noll}"
    vals = np.array([_vec(r, "nollZernikeValues", len(ZN21)) for _, r in w.iterrows()])
    return pd.DataFrame(vals, index=w.sensorId.astype(int).to_numpy(),
                        columns=[f"Z{j}" for j in ZN21])


def consdb_zernikes(cdb, visit_id, detectors=CORNER_IDS):
    """ConsDB ``ccdvisit1_quicklook`` z4..z28 for the corner detectors of one visit.

    The frame, and whether these are OPD or deviation, is not documented; compare with
    `wavefront` before relying on either.

    Returns
    -------
    zk : `pandas.DataFrame`
        Index detector, columns Z4..Z28, µm of wavefront.
    """
    zsel = ", ".join(f"q.z{j} AS z{j}" for j in range(4, 29))
    q = (f"SELECT v.detector AS det, {zsel} FROM cdb_lsstcam.ccdvisit1_quicklook q "
         "JOIN cdb_lsstcam.ccdvisit1 v ON v.ccdvisit_id = q.ccdvisit_id "
         f"WHERE v.visit_id = {int(visit_id)} AND v.detector IN "
         f"({','.join(str(d) for d in detectors)})")
    res = cdb.query(q)
    t = res.to_pandas() if hasattr(res, "to_pandas") else res
    t = t.set_index("det").sort_index()
    t.columns = [c.upper() for c in t.columns]
    return t.reindex(detectors)


# ------------------------------------------------------------------ MTAOS log -----
_ARR = r"\[([^\]]*)\]"


def _arr(s):
    return np.array(s.replace("\n", " ").split(), float)


def pid_log(ev, t_dof):
    """Parse the PID log message emitted just before a ``degreeOfFreedom`` event.

    Returns
    -------
    pid : `dict` [`str`, `numpy.ndarray`]
        ``error``, ``integral_raw``, ``integral_clipped``, ``uk``, each 22 values in the
        `DOF22` order, in DOF units. ``error`` = setpoint - state, setpoint 0, so the
        optical state is ``-error``; the Tweak is ``-uk``.
    """
    lm = ev["logevent_logMessage"]
    lm = lm[lm.message.str.contains("Error before any gains")]
    m = lm.iloc[int(np.argmin(np.abs((lm.index - t_dof).total_seconds())))].message
    keys = [("error", r"PID is " + _ARR), ("integral_raw", r"before clipping is " + _ARR),
            ("integral_clipped", r"Integral component is " + _ARR),
            ("uk", r"after Kp/Ki/Kd gains is " + _ARR)]
    return {k: _arr(re.search(p, m, re.S).group(1)) for k, p in keys}


def wep_log(ev, seq, t_dof, day_obs):
    """MTAOS log lines from image ingestion to the OFC step for one image.

    Returns
    -------
    info : `dict`
        ``donuts`` (`pandas.DataFrame`: detector, intra and extra donut size in px,
        out-of-focus offset in µm), ``tables_found``, ``tables_total``,
        ``measurements_claimed``, ``refocus_dz_um`` and ``lines`` (all messages,
        `pandas.Series`).
    """
    lm = ev["logevent_logMessage"].sort_index()
    tag = f"MC_O_{day_obs}_{seq:06d}"
    ing = lm[lm.message.str.contains(f"Image {tag}") & lm.message.str.contains("ingested")]
    t_start = ing.index[0] if len(ing) else t_dof - pd.Timedelta("90s")
    lines = lm[(lm.index >= t_start) & (lm.index <= t_dof + pd.Timedelta("1s"))].message
    donuts, last_det = [], None
    out = dict(tables_found=np.nan, tables_total=np.nan, measurements_claimed=np.nan,
               refocus_dz_um=np.nan)
    for msg in lines:
        if m := re.search(r"Detector (\d+): intra=([-\d.]+) px, extra=([-\d.]+) px", msg):
            last_det = dict(detector=int(m[1]), intra_px=float(m[2]), extra_px=float(m[3]))
            donuts.append(last_det)
        elif (m := re.search(r"out-of-focus offsets from donuts: ([-\d.]+) um", msg)) and last_det:
            last_det["defocus_offset_um"] = float(m[1])
        elif m := re.search(r"Found (\d+) non-empty tables out of (\d+)", msg):
            out.update(tables_found=int(m[1]), tables_total=int(m[2]))
        elif m := re.search(r"(\d+)/(\d+) wavefront error measurements were found", msg):
            out["measurements_claimed"] = int(m[1])
        elif m := re.search(r"refocus camera would be: dz=([-\d.eE+]+) um", msg):
            out["refocus_dz_um"] = float(m[1])
    out["donuts"] = pd.DataFrame(donuts).set_index("detector") if donuts else pd.DataFrame()
    out["lines"] = lines.str.slice(0, 300)
    return out


def applied_hexapods(ev, t):
    """Hexapod compensated and uncompensated positions in force at time ``t``.

    Returns
    -------
    hex : `pandas.DataFrame`
        Rows ``cam``, ``m2``; columns x, y, z (µm), u, v (deg) for ``comp_`` and
        ``uncomp_``, and their difference ``lut_`` (the LUT part).
    """
    out = {}
    ts = _ts(t)
    for sal, name in [(1, "cam"), (2, "m2")]:
        row = {}
        for key in ["hex_comp", "hex_uncomp"]:
            df = ev[key]
            df = df[(df.salIndex == sal) & (df.index <= ts)] if len(df) else df
            r = df.iloc[-1] if len(df) else None
            for ax in "xyzuv":
                row[f"{key[4:]}_{ax}"] = float(r[ax]) if r is not None else np.nan
        for ax in "xyzuv":
            row[f"lut_{ax}"] = row[f"comp_{ax}"] - row[f"uncomp_{ax}"]
        out[name] = row
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ reproduction --
def export_summit_sources(dest, ts_ofc_repo, ts_config_repo,
                          ts_ofc_tag=TS_OFC_TAG, config_tag=TS_CONFIG_MTTCS_TAG):
    """Export the summit ``ts_ofc`` and ``ts_config_mttcs`` tags with ``git archive``.

    The local clones are only read. Skips a tag already exported.

    Returns
    -------
    ofc_python, config_dir : `pathlib.Path`
        ``ts_ofc`` python directory and the MTAOS OFC config directory.
    """
    dest = Path(dest)
    ofc_dir, cfg_dir = dest / f"ts_ofc_{ts_ofc_tag}", dest / f"ts_config_mttcs_{config_tag}"
    for repo, tag, out, path in [(ts_ofc_repo, ts_ofc_tag, ofc_dir, "python"),
                                 (ts_config_repo, config_tag, cfg_dir, "MTAOS")]:
        if not (out / path).exists():
            out.mkdir(parents=True, exist_ok=True)
            arc = subprocess.run(["git", "-C", str(repo), "archive", tag, path],
                                 check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", str(out)], input=arc, check=True)
    return ofc_dir / "python", cfg_dir / "MTAOS" / MTAOS_CONFIG / "ofc"


def use_summit_ts_ofc(ofc_python):
    """Put the exported ``ts_ofc`` first on ``sys.path``.

    Must run before anything imports ``lsst.ts.ofc``; the stack's copy is a
    development branch whose state estimate differs from the summit's.
    """
    if "lsst.ts.ofc" in sys.modules:
        got = sys.modules["lsst.ts.ofc"].__file__
        if not got.startswith(str(ofc_python)):
            raise RuntimeError(f"lsst.ts.ofc already imported from {got}; restart the kernel")
        return
    sys.path.insert(0, str(ofc_python))
    # lsst and lsst.ts are pkgutil namespace packages; once imported, their __path__
    # is fixed and ignores sys.path.
    for name, sub in [("lsst", "lsst"), ("lsst.ts", "lsst/ts")]:
        mod = sys.modules.get(name)
        if mod is not None and isinstance(mod.__path__, list):
            mod.__path__.insert(0, str(Path(ofc_python) / sub))
    import lsst.ts.ofc as m
    assert m.__file__.startswith(str(ofc_python)), m.__file__


def summit_state_estimator(config_dir):
    """OFC `StateEstimator` configured as the summit ran it: 22 DOF, 12 v-modes.

    Returns
    -------
    se : `lsst.ts.ofc.state_estimator.StateEstimator`
    """
    from lsst.ts.ofc import OFCData
    from lsst.ts.ofc.state_estimator import StateEstimator

    od = OFCData("lsst", config_dir=Path(config_dir))
    od.normalization_weights = np.array(
        od.load_yaml_file(Path(config_dir) / "normalization_weights" / NORM_YAML))
    od.controller.pop("truncation_threshold", None)
    od.controller["truncation_index"] = N_VMODES
    od.comp_dof_idx = {"m2HexPos": np.ones(5, bool), "camHexPos": np.ones(5, bool),
                       "M1M3Bend": np.arange(20) < 7, "M2Bend": np.arange(20) < 5}
    od.zn_selected = np.array(ZN21)
    se = StateEstimator(od)
    assert list(od.dof_idx) == DOF22 and se.truncate_index == N_VMODES
    return se


def reproduce_step(se, wfe, rot_deg, band, kp, ki=None, kd=None):
    """Reproduce one OFC state estimate and PID step from the corner wavefront.

    NaN corners are dropped first, as ``OFC.calculate_corrections`` does. The rotator
    angle goes to ``dof_state`` as is: adding the controller's ``rotation_offset`` of
    90 deg rotates the hexapod dx/dy pairs away from the logged values.

    Parameters
    ----------
    se : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `summit_state_estimator`.
    wfe : `pandas.DataFrame`
        From `wavefront`.
    rot_deg : `float`
        Mean physical rotator angle over the exposure, deg.
    band : `str`
        Filter band, such as ``"u"``.
    kp, ki, kd : `numpy.ndarray`
        Gains, 50 values (dimensionless). Only the proportional term is reproduced;
        ``ki`` and ``kd`` are checked to be zero.

    Returns
    -------
    rep : `dict`
        ``corners_used`` (sensor ids), ``state`` (22, DOF units), ``vmodes`` (12,
        normalized v-mode amplitudes, dimensionless), ``error`` (= -state),
        ``uk`` (= kp * error) and ``tweak50`` (= -uk on the 50-DOF vector).
    """
    from lsst.ts.ofc.utils import ControlBasis
    from lsst.ts.ofc.utils.ofc_data_helpers import get_sensor_names

    for g in (ki, kd):
        if g is not None and np.any(np.asarray(g)[DOF22] != 0):
            raise NotImplementedError("nonzero ki or kd: integral/derivative state not reproduced")
    ok = np.isfinite(wfe.to_numpy()).all(axis=1)
    full = np.zeros((ok.sum(), 23))   # dof_state indexes contiguous Noll from Z4
    full[:, [z - 4 for z in ZN21]] = wfe.to_numpy()[ok]
    names = get_sensor_names(ofc_data=se.ofc_data, sensor_ids=wfe.index.to_numpy()[ok])
    state = np.asarray(se.dof_state(band.upper(), full, names, rot_deg, subtract_intrinsics=False))
    vm = np.asarray(se.dof_state(band.upper(), full, names, rot_deg, subtract_intrinsics=False,
                                 basis=ControlBasis.VMode))
    error = -state
    uk = np.asarray(kp)[DOF22] * error
    tweak50 = np.zeros(N_DOF)
    tweak50[DOF22] = -uk
    return dict(corners_used=list(wfe.index[ok]), state=state, vmodes=vm, error=error,
                uk=uk, tweak50=tweak50)


# ------------------------------------------------------------------ tables --------
def step_table(step, prev, pid, rep):
    """22-DOF table for one OFC step.

    Parameters
    ----------
    step, prev : `pandas.Series`
        This and the previous row of `ofc_steps` (``prev`` may be None).
    pid : `dict`
        From `pid_log`.
    rep : `dict`
        From `reproduce_step`.

    Returns
    -------
    table : `pandas.DataFrame`
        Index DOF name; units in the ``unit`` column.
    """
    idx = DOF22
    t = pd.DataFrame(index=[DOF_NAMES[i] for i in idx])
    t["unit"] = [DOF_UNITS[i] for i in idx]
    t["state (log, -error)"] = -pid["error"]
    t["state (repro)"] = rep["state"]
    t["integral clipped (log)"] = pid["integral_clipped"]
    t["kp"] = step.kp[idx]
    t["tweak (EFD visitDoF)"] = step.tweak[idx]
    t["tweak (repro)"] = rep["tweak50"][idx]
    t["tweak if P used clipped"] = -step.kp[idx] * pid["integral_clipped"]
    t["trim before"] = prev.trim[idx] if prev is not None else np.nan
    t["trim after"] = step.trim[idx]
    t["trim change"] = t["trim after"] - t["trim before"]
    return t
