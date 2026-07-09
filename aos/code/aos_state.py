"""Shared AOS per-visit state helpers.

Single source of truth for the normalization-sensitive pieces of the per-visit
AOS table, so they cannot drift between the two callers:

- ``olr/code/nightly_table.py`` (Summit, all visits in a night, + Butler AOS join)
- ``blocks/t539_closedloop_aos.ipynb`` (USDF, selected visits across many days)

Provides:
- ``resolve_ofc_config_dir`` — locate the OFC config (v13) via TS_CONFIG_MTTCS_DIR
- ``build_geom_svd`` — the geom-normalized sensitivity-matrix SVD for a DOF subset
- ``project_dofs_to_vmodes`` — project a physical DOF vector onto the v-modes
- ``fetch_corner_zernikes_consdb`` — per-corner retrieved-wavefront OPD Zernikes

Normalization note: the v-modes use the OFC config's stored
``normalization_weights`` (v13), which ARE the official geom_mean
``n_j = r_j^0.5 * f_j^-0.5`` (field-averaged FWHM) — the same normalization
``build_ofc_svd`` / the bounce analysis use via the ts_config_mttcs yaml. Do NOT
recompute ``sqrt(range/fwhm)`` (that uses corner-point FWHM, ~sqrt(2) off) and do
NOT use ``OFCData()`` without ``config_dir`` (old non-geom default -> v1 becomes
the M2-tilt mode). See ``olr/docs/vmode_normalization.md``.
"""
import os

import numpy as np

# Corner wavefront sensors (SW0 half-chips): detector id -> raft name
CORNERS = {191: "R00_SW0", 195: "R04_SW0", 199: "R40_SW0", 203: "R44_SW0"}
SENSOR_NAMES = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]

# Zernikes kept for the wavefront: Z4..Z26 excluding Z20, Z21 (Noll) -> 21 terms
ZK_NOLL = [z for z in range(4, 27) if z not in (20, 21)]

# DOF subsets (indices into the 50-DOF OFC state) and their v-mode truncation
DOF_SETS = {
    "hexapod_10": list(range(0, 10)),
    "standard_22": sorted(list(range(0, 17)) + list(range(30, 35))),
    "all_50": list(range(0, 50)),
}
N_MODES = {"hexapod_10": 10, "standard_22": 12, "all_50": 20}

__all__ = [
    "CORNERS", "SENSOR_NAMES", "ZK_NOLL", "DOF_SETS", "N_MODES",
    "resolve_ofc_config_dir", "make_state_estimator", "vmodes_from_dofs",
    "build_geom_svd", "project_dofs_to_vmodes", "recover_optical_state",
    "fetch_corner_zernikes_consdb",
]

# Component layout of the 50-DOF OFC state (for comp_dof_idx construction)
_DOF_COMPONENTS = {"m2HexPos": (0, 5), "camHexPos": (5, 5),
                   "M1M3Bend": (10, 20), "M2Bend": (30, 20)}


def _comp_dof_idx(dof_indices):
    """Build the OFCData.comp_dof_idx boolean dict from active DOF indices."""
    active = set(dof_indices)
    return {name: np.array([(start + i) in active for i in range(n)], dtype=bool)
            for name, (start, n) in _DOF_COMPONENTS.items()}


def make_state_estimator(config_dir=None, dof_set="standard_22", version="v13"):
    """OFC StateEstimator configured with geom (config) weights + a DOF subset.

    This is the canonical v-mode engine: ``StateEstimator.get_vmodes_from_dofs``
    uses the OFC config's stored ``normalization_weights`` (the official
    geom_mean) and its own sensitivity-matrix SVD. Using it everywhere
    guarantees identical v-modes across nightly_report / OLR / t539 — important
    because degenerate singular-value pairs make the individual v-mode vectors
    basis-dependent (only v1 and degenerate-pair magnitudes are unique), so all
    code must share the same Vh. Requires the LSST stack (lsst.ts.ofc).
    """
    from lsst.ts.ofc import OFCData
    from lsst.ts.ofc.state_estimator import StateEstimator

    if config_dir is None:
        config_dir = resolve_ofc_config_dir(version)
    ofc = OFCData("lsst", config_dir=config_dir)
    ofc.comp_dof_idx = _comp_dof_idx(DOF_SETS[dof_set])
    return StateEstimator(ofc)


def vmodes_from_dofs(dof_state, state_estimator, n_modes=12):
    """v-modes via ``StateEstimator.get_vmodes_from_dofs`` (canonical method).

    Parameters
    ----------
    dof_state : array (50,) or (n, 50)  -- physical DOF vector(s) / trim.
    state_estimator : from make_state_estimator.
    n_modes : int, default 12.

    Returns
    -------
    ndarray (n, n_modes). Rows with any non-finite active DOF -> NaN.
    """
    idx = state_estimator.ofc_data.dof_idx
    arr = np.atleast_2d(np.asarray(dof_state, dtype=float))
    out = np.full((len(arr), n_modes), np.nan)
    for k, d in enumerate(arr):
        if np.all(np.isfinite(d[idx])):
            out[k] = np.asarray(state_estimator.get_vmodes_from_dofs(d))[:n_modes]
    return out


def resolve_ofc_config_dir(version="v13"):
    """Locate the OFC config dir. Prefer TS_CONFIG_MTTCS_DIR (set on the stack);
    fall back to the USDF packages path."""
    ts = os.environ.get("TS_CONFIG_MTTCS_DIR")
    if ts:
        return os.path.join(ts, "MTAOS", version, "ofc")
    return f"/home/r/roodman/u/LSST/packages/ts_config_mttcs/MTAOS/{version}/ofc"


def build_geom_svd(config_dir=None, dof_set="standard_22", version="v13"):
    """Geom-normalized OFC sensitivity-matrix SVD for a DOF subset.

    Uses the OFC config's stored ``normalization_weights`` (the official
    geom_mean). Requires the LSST stack (``lsst.ts.ofc``).

    Returns a dict with U, s, V, dof_indices, norm_vector, n_modes, config_dir.
    ``V`` has shape (n_dof_sub, n_dof_sub); v-mode j amplitude for a physical DOF
    vector d is ``V[:, j] . (d[dof_indices] / norm_vector)``.
    """
    from lsst.ts.ofc import OFCData, SensitivityMatrix

    if config_dir is None:
        config_dir = resolve_ofc_config_dir(version)
    zn = np.array(ZK_NOLL)
    ofc = OFCData("lsst", config_dir=config_dir)
    ofc.zn_selected = zn
    field_angles = [ofc.sample_points[s] for s in SENSOR_NAMES]
    sens = SensitivityMatrix(ofc).evaluate(field_angles, 0.0)[:, zn - 4, :]
    A_full = sens.reshape((-1, sens.shape[2]))
    dof_indices = DOF_SETS[dof_set]
    A_sub = A_full[:, dof_indices]
    norm_vector = ofc.normalization_weights[dof_indices]
    U, s, Vh = np.linalg.svd(A_sub @ np.diag(norm_vector), full_matrices=False)
    return dict(U=U, s=s, V=Vh.T, dof_indices=list(dof_indices),
                norm_vector=np.asarray(norm_vector), n_modes=len(s),
                config_dir=config_dir)


def project_dofs_to_vmodes(dof_state, svd, n_modes=None):
    """Project physical DOF vector(s) onto v-modes.

    ``v_j = V[:, j] . (dof[dof_indices] / norm_vector)``. Numbered from 0; caller
    labels v1.. as needed. Rows with any non-finite DOF become NaN.

    Parameters
    ----------
    dof_state : array (50,) or (n, 50)
    svd : dict from build_geom_svd
    n_modes : int, optional (default = min(12, available))

    Returns
    -------
    ndarray (n, n_modes)
    """
    V = svd["V"]
    idx = svd["dof_indices"]
    w = svd["norm_vector"]
    if n_modes is None:
        n_modes = min(12, V.shape[1])
    arr = np.atleast_2d(np.asarray(dof_state, dtype=float))
    out = np.full((len(arr), n_modes), np.nan)
    for k, d in enumerate(arr):
        sub = d[idx]
        if np.all(np.isfinite(sub)):
            out[k] = V[:, :n_modes].T @ (sub / w)
    return out


def recover_optical_state(z_dev, svd, n_modes=12):
    """Recover the optical-state DoF from a measured corner-deviation wavefront.

    This is Aaron's ``optical_state``: the DoF obtained by running the measured
    Zernike *deviations* (OPD - intrinsic) through the SVD (least-squares onto
    the top-``n_modes`` controllable subspace). It is NOT the Trim
    (aggregatedDoF) nor the Tweak.

    Parameters
    ----------
    z_dev : ndarray (n_rows,)
        Measured per-corner deviation Zernikes, flattened in the SAME order as
        the SVD rows: corner-major, 4 corners x len(ZK_NOLL).
    svd : dict from build_geom_svd.
    n_modes : int, default 12.

    Returns
    -------
    dof_full : ndarray (50,)
        Physical DoF; the geom DOF subset is filled, other entries 0.
    vmode_amps : ndarray (n_modes,)
        v-mode amplitudes of the optical state, in THIS svd's basis. (For the
        canonical basis, pass dof_full through StateEstimator.get_vmodes_from_dofs.)
    zk_constrained : ndarray (n_rows,)
        Controllable projection of z_dev (the part reproducible by the DOF in
        the kept subspace) — reconstruct-from-optical_state. Basis-invariant.

    Math: z_dev = U s V^T (dof_sub/w)  ->  v = (U[:,:k].T z_dev)/s[:k],
    dof_sub = w * (V[:,:k] v),  zk_constrained = U[:,:k] (s[:k]*v).
    """
    U, s, V = svd["U"], svd["s"], svd["V"]
    idx, w = svd["dof_indices"], svd["norm_vector"]
    k = n_modes
    v = (U[:, :k].T @ np.asarray(z_dev, float)) / s[:k]
    dof_full = np.zeros(50)
    dof_full[idx] = w * (V[:, :k] @ v)
    zk_con = U[:, :k] @ (s[:k] * v)
    return dof_full, v, zk_con


def fetch_corner_zernikes_consdb(cdb_client, visit_ids, instrument="lsstcam",
                                 zk_noll=None, corners=None):
    """Per-corner retrieved-wavefront OPD Zernikes from ConsDB ccdvisit1_quicklook.

    The corner WFS (dets 191/195/199/203) produce donut Zernikes for every
    on-sky exposure; ConsDB stores them per-CCD in ccdvisit1_quicklook.z4..z28,
    already associated with the visit. These are the total OPD (microns), not the
    intrinsic-subtracted deviation.

    Returns a DataFrame indexed by visit_id with columns ``z{noll}_{corner}``.
    """
    import pandas as pd

    zk_noll = zk_noll if zk_noll is not None else ZK_NOLL
    corners = corners if corners is not None else CORNERS
    if len(visit_ids) == 0:
        return pd.DataFrame()
    ids = ",".join(str(int(v)) for v in visit_ids)
    dets = ",".join(str(d) for d in corners)
    zcols = [f"z{z}" for z in range(4, 29)]
    q = f"""
        SELECT cv.visit_id, cv.detector, {", ".join("cq." + z for z in zcols)}
        FROM cdb_{instrument}.ccdvisit1_quicklook cq,
             cdb_{instrument}.ccdvisit1 cv
        WHERE cv.ccdvisit_id = cq.ccdvisit_id
          AND cv.detector IN ({dets})
          AND cv.visit_id IN ({ids})
    """
    zk = cdb_client.query(q).to_pandas()
    for z in zcols:
        zk[z] = pd.to_numeric(zk[z], errors="coerce")
    zk["corner"] = zk["detector"].map(corners)
    rows = {}
    for vid, g in zk.groupby("visit_id"):
        row = {}
        for _, r in g.iterrows():
            cname = r["corner"]
            if cname is None:
                continue
            for z in zk_noll:
                row[f"z{z}_{cname}"] = r[f"z{z}"]
        rows[vid] = row
    return pd.DataFrame.from_dict(rows, orient="index")
