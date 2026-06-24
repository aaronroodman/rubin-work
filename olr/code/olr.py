"""Open Loop Reproduction (OLR) core.

Reconstruct the open-loop wavefront from a night's closed-loop WFS Zernikes by
*adding back* the AOS correction (trim) that was applied during the night:

    z_change          = (sens_mat @ trim).reshape(4, 21), zero-padded at Z20/Z21
    olr_opd[c]        = zk_opd[c]       + z_change[c]
    olr_deviation[c]  = zk_deviation[c] + z_change[c]

``zk_intrinsic`` is unchanged, so the identity
``olr_deviation == olr_opd - zk_intrinsic`` is preserved.

Ported from Craig Lage's ``Open_Loop_Reproduction_Based_PID_Simulator`` notebook
(``lsst-so/ts_aos_analysis``), cells 7/9/11.  Craig only carries the deviation;
we carry the OPD too, for the measured-intrinsic-wavefront (MIW) analysis.

The sensitivity matrix needs ``lsst.ts.ofc`` and a checkout of
``ts_config_mttcs`` (its ``MTAOS/ofc`` directory).
"""

import numpy as np

# 4 corner WFS sensors (extra-focal SW0), in the canonical order.
SENSOR_IDS = [191, 195, 199, 203]
CORNER_DETNAMES = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
CORNER_NAMES = ["R00", "R04", "R40", "R44"]

# Collapse the 50 DOF down to the 22 used by the controller (5 M2 hex + 5 cam
# hex + 7 M1M3 bending + 5 M2 bending).
DEFAULT_DOF_INDICES = list(range(0, 17)) + list(range(30, 35))

# Pupil Noll indices the sensitivity matrix spans (z4-z19, z22-z26 = 21 terms;
# Z20/Z21 are skipped and zero-padded back in apply_trim).
DEFAULT_ZN_SELECTED = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
]
DEFAULT_TRUNCATION = 12


def build_olr_sensitivity_matrix(
    ofc_config_dir,
    truncation=DEFAULT_TRUNCATION,
    zn_selected=None,
    rotation_angle=0.0,
):
    """Build the (84 x 22) DOF->Zernike sensitivity matrix (Craig cell 7).

    Parameters
    ----------
    ofc_config_dir : str
        Path to ``ts_config_mttcs/MTAOS/ofc``.
    truncation : int
        OFC truncation index (number of v-modes kept).
    zn_selected : sequence of int, optional
        Pupil Noll indices to span (default z4-z19, z22-z26).
    rotation_angle : float
        Rotator angle the matrix is evaluated at (default 0.0, matches Craig).

    Returns
    -------
    sens_mat : ndarray (n_sensors*n_zn, 22)
        Forward sensitivity matrix.  With the defaults: (4*21, 22) = (84, 22).
    """
    # Lazy import: lsst.ts.ofc is only available on the RSP stack.
    from lsst.ts.ofc import OFC, OFCData
    from lsst.ts.ofc.state_estimator import StateEstimator

    if zn_selected is None:
        zn_selected = DEFAULT_ZN_SELECTED

    ofc_data = OFCData(name="lsst", config_dir=ofc_config_dir)
    ofc = OFC(ofc_data=ofc_data)

    # 22 active DOF: M1M3 bending modes 8-20 and M2 bending modes 6-20 off.
    comp_dof_idx = dict(
        m2HexPos=np.ones(5, dtype=bool),
        camHexPos=np.ones(5, dtype=bool),
        M1M3Bend=np.ones(20, dtype=bool),
        M2Bend=np.ones(20, dtype=bool),
    )
    comp_dof_idx["M1M3Bend"][7:] = False
    comp_dof_idx["M2Bend"][5:] = False

    ofc.set_truncation_index(truncation)
    ofc_data.zn_selected = np.array(zn_selected)
    ofc.ofc_data.comp_dof_idx = comp_dof_idx
    ofc.controller.reset_history()
    ofc.state_estimator.refresh_from_ofc_data()

    field_angles_ccs = [ofc_data.sample_points[det] for det in CORNER_DETNAMES]

    state_estimator = StateEstimator(ofc_data)
    sens_mat = state_estimator.get_sensitivity_matrix(field_angles_ccs, rotation_angle)
    return sens_mat


def apply_trim(zernikes, trim, sens_mat, subtract=True):
    """Apply (or remove) the trim correction to a (4, 23) Zernike stack.

    Craig cell 11, verbatim semantics: the matrix product spans 21 Zernikes per
    corner; Z20/Z21 are inserted back as zeros to align with the dense 23-vector
    (Noll 4..26).

    subtract=False ADDS the correction back -> open-loop reproduction.
    """
    zernikes_change = sens_mat @ trim
    zernikes_change = zernikes_change.reshape(4, 21)
    # Zero out Z20, Z21 (positions 16, 17 in the 0-based Noll-4 array).
    zernikes_change = np.insert(zernikes_change, obj=16, values=0, axis=1)
    zernikes_change = np.insert(zernikes_change, obj=16, values=0, axis=1)
    if subtract:
        return zernikes - zernikes_change
    else:
        return zernikes + zernikes_change


def _stack_corners(row, prefix):
    """Build a (4, 23) array from per-corner columns ``{prefix}_{corner}``.

    Returns None if any corner is missing or contains NaNs.
    """
    out = np.zeros((4, 23))
    for i, corner in enumerate(CORNER_NAMES):
        val = row.get(f"{prefix}_{corner}")
        if val is None:
            return None
        arr = np.asarray(val, dtype=float)
        if arr.shape[0] != 23 or not np.isfinite(arr).all():
            return None
        out[i, :] = arr
    return out


def extract_olr(table, sens_mat, indices=None, seq_min=None, seq_max=None):
    """Compute the OLR for every usable seq in a nightly table.

    For each seq the trim correction (sens_mat @ dof_state[indices]) is added to
    BOTH the OPD and the deviation corner stacks; the intrinsic is carried
    through unchanged.

    Parameters
    ----------
    table : pandas.DataFrame
        Nightly AOS table (one row per seq) with columns ``seq``, ``dof_state``,
        ``rotation_angle``, ``band`` and per-corner ``zk_opd_*``,
        ``zk_deviation_*``, ``zk_intrinsic_*``.
    sens_mat : ndarray
        From ``build_olr_sensitivity_matrix``.
    indices : sequence of int, optional
        Which of the 50 DOF map to the 22-column sensitivity matrix
        (default ``DEFAULT_DOF_INDICES``).
    seq_min, seq_max : int, optional
        Restrict to this inclusive seq range.

    Returns
    -------
    records : list of dict
        One dict per usable seq with keys: seq, rotation_angle, band, day_obs,
        dof_state, trim, and per-corner 23-vectors
        ``meas_opd_*``, ``meas_deviation_*``, ``intrinsic_*``,
        ``olr_opd_*``, ``olr_deviation_*``.
    """
    if indices is None:
        indices = DEFAULT_DOF_INDICES
    indices = list(indices)

    seqs = sorted(int(s) for s in table["seq"].dropna().unique())
    records = []
    n_skip = 0
    for seq in seqs:
        if seq_min is not None and seq < seq_min:
            continue
        if seq_max is not None and seq > seq_max:
            continue
        rows = table[table["seq"] == seq]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]

        opd = _stack_corners(row, "zk_opd")
        dev = _stack_corners(row, "zk_deviation")
        intr = _stack_corners(row, "zk_intrinsic")
        dof = row.get("dof_state")
        if opd is None or dev is None or dof is None:
            n_skip += 1
            continue
        dof = np.asarray(dof, dtype=float)
        if dof.shape[0] < max(indices) + 1 or not np.isfinite(dof[indices]).all():
            n_skip += 1
            continue

        trim = dof[indices]
        olr_opd = apply_trim(opd, trim, sens_mat, subtract=False)
        olr_dev = apply_trim(dev, trim, sens_mat, subtract=False)

        rec = {
            "seq": seq,
            "day_obs": int(row["day_obs"]) if "day_obs" in row else None,
            "rotation_angle": float(row.get("rotation_angle", np.nan)),
            "band": row.get("band"),
            "dof_state": dof,
            "trim": trim,
        }
        for i, corner in enumerate(CORNER_NAMES):
            rec[f"meas_opd_{corner}"] = opd[i]
            rec[f"meas_deviation_{corner}"] = dev[i]
            rec[f"intrinsic_{corner}"] = intr[i] if intr is not None else None
            rec[f"olr_opd_{corner}"] = olr_opd[i]
            rec[f"olr_deviation_{corner}"] = olr_dev[i]
        records.append(rec)

    if n_skip:
        print(f"  extract_olr: skipped {n_skip} seqs (missing/NaN zernikes or dof)")
    return records
