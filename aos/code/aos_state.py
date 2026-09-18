"""Shared AOS per-visit state helpers.

Single source of truth for the normalization-sensitive pieces of the per-visit
AOS table, so they cannot drift between the two callers:

- ``olr/code/nightly_table.py`` (Summit, all visits in a night, + Butler AOS join)
- ``blocks/t539_closedloop_aos.ipynb`` (USDF, selected visits across many days)

Provides:
- ``make_state_estimator`` — **the single sanctioned v-mode engine**, wrapping ts_ofc's
  ``StateEstimator`` with the required normalization asserted
- ``vmodes_from_dofs`` — project physical DOF onto the v-modes through that estimator
- ``resolve_ofc_config_dir`` — locate the OFC config (v13) via TS_CONFIG_MTTCS_DIR
- ``corner_recovery_basis`` — SVD of the corner-evaluated sensitivity, for *inverting*
  a measured wavefront; the recovery basis, not the v-mode basis
- ``recover_optical_state`` — measured corner wavefront -> DOF, v-modes, constrained
  wavefront
- ``fetch_corner_zernikes_consdb`` — per-corner retrieved-wavefront OPD Zernikes

Never build the sensitivity-matrix SVD outside OFC code. Every v-mode in this
repository comes from ``make_state_estimator``, which is what makes the obsolete
normalization unreachable rather than merely discouraged.

Two bases, deliberately: v-modes are always reported in the ``make_state_estimator``
basis (``StateEstimator.Vh``, the full 899-row Double Zernike slab — what MTAOS runs),
while the *inversion* of a measured 84-row corner wavefront happens in
``corner_recovery_basis``, because ``Vh`` does not span that problem and leaves a
2.3e-02 µm wavefront-residual floor. See ``recover_optical_state`` and
``aos/docs/status/corner_recovery_route_comparison.md``.

Every sensitivity matrix here is evaluated at camera rotator angle
``SMATRIX_ROTATION_ANGLE_DEG`` = 0.0 deg by AOS group decision, so wavefronts must be
derotated into the telescope frame (``ZK_FRAME`` = ``'ocs'``) before inversion.

Normalization: the v-modes use the OFC config's stored ``normalization_weights``
(v13), which ARE the official geom_mean ``n_j = r_j^0.5 * f_j^-0.5`` (field-averaged
FWHM) — the same normalization ``build_ofc_svd`` and the bounce analysis use via the
ts_config_mttcs yaml. Do NOT recompute ``sqrt(range/fwhm)`` (that uses corner-point
FWHM, ~sqrt(2) off), and do NOT use ``OFCData()`` without ``config_dir``: the bare
default resolves ``range-fwhm.yaml``, the obsolete normalization whose sensitivity
matrix retains a dependence on physical units. Measured, ``standard_22``: it makes v1
an essentially orthogonal mode, ``|cos(v1_obsolete, v1_required)| = 1.2e-05``
(dimensionless), with v1 per µm of CamHex dz collapsing from -8.9153e-04 to
-1.2933e-09 (dimensionless amplitude per µm) — v1 becomes the M2-tilt mode.

Note that ``range-fwhm.yaml`` is reachable *inside* the v13 config directory too:
``init.yaml`` selects the required normalization but ``dz_controller.yaml`` and
``oic_controller.yaml`` select the obsolete one, and ``pid_controller.yaml`` selects
``default.yaml``. So the config directory alone does not settle it, which is why
``make_state_estimator`` asserts the resolved filename.

See ``olr/docs/vmode_normalization.md`` and ``aos/docs/studies/smatrix_vmode.md``.
"""
import os

import numpy as np

# Corner wavefront sensors (SW0 half-chips): detector id -> raft name
CORNERS = {191: "R00_SW0", 195: "R04_SW0", 199: "R40_SW0", 203: "R44_SW0"}
SENSOR_NAMES = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]

# Zernikes kept for the wavefront: Z4..Z26 excluding Z20, Z21 (Noll) -> 21 terms
ZK_NOLL = [z for z in range(4, 27) if z not in (20, 21)]

# The only acceptable OFC normalization. The bare ``OFCData('lsst')`` default resolves
# OBSOLETE_NORM_YAML, whose sensitivity matrix retains a dependence on physical units.
# The two differ non-uniformly per DOF -- ratios of required over obsolete, dimensionless
# per-DOF weight: 10.45 (M2Hex dz), 948.8 (dx, dy), 0.00579 (rx, ry) -- so substituting one
# for the other rotates the v-mode basis rather than rescaling it. ``znmin``, ``znmax``,
# ``sample_points`` and ``sensitivity_matrix`` are identical between the two configs, so a
# bare OFCData that reads only those is harmless; anything touching normalization is not.
REQUIRED_NORM_YAML = "range0.5_fwhm-0.15.yaml"
OBSOLETE_NORM_YAML = "range-fwhm.yaml"

# DOF subsets (indices into the 50-DOF OFC state) and their v-mode truncation
DOF_SETS = {
    "hexapod_10": list(range(0, 10)),
    "standard_22": sorted(list(range(0, 17)) + list(range(30, 35))),
    "all_50": list(range(0, 50)),
}
N_MODES = {"hexapod_10": 10, "standard_22": 12, "all_50": 20}

# Canonical 22-DOF reduced set, as an explicit index list for
# ``ofc_svd.build_ofc_svd(..., n_dof=DOF22)``. It is NOT the first 22 contiguous
# indices: passing the scalar 22 silently selects DOF 0-21, i.e. 10 rigid + the first
# 12 M1M3 bending modes, instead of 10 rigid + 7 M1M3 + 5 M2. Always pass this list.
DOF22 = DOF_SETS["standard_22"]

__all__ = [
    "CORNERS", "SENSOR_NAMES", "ZK_NOLL", "DOF_SETS", "N_MODES", "DOF22",
    "REQUIRED_NORM_YAML", "OBSOLETE_NORM_YAML", "SMATRIX_ROTATION_ANGLE_DEG",
    "ZK_FRAME",
    "resolve_ofc_config_dir", "make_state_estimator", "vmodes_from_dofs",
    "corner_recovery_basis", "recover_optical_state",
    "fetch_corner_zernikes_consdb",
]

# Names retired when the hand-rolled SVD was replaced by the OFC `StateEstimator`,
# mapped to what supersedes each. A caller that still asks for one gets a message
# naming the replacement, rather than an AttributeError that looks like a typo --
# these names appear in notebooks and in three sibling topics, which are not
# checked by any static import analysis.
_RETIRED = {
    "build_geom_svd":
        "make_state_estimator(dof_set=..., version=...) -- the OFC StateEstimator, "
        "which owns the single sanctioned SVD of the sensitivity matrix",
    "project_dofs_to_vmodes":
        "vmodes_from_dofs(dof_state, state_estimator, n_modes=...)",
}


def __getattr__(name):
    """Raise a named error for a retired helper; normal AttributeError otherwise.

    Parameters
    ----------
    name : `str`
        Attribute requested from this module.

    Raises
    ------
    AttributeError
        Always. For a name in `_RETIRED` the message gives the replacement, so an
        `import`-time failure says what to call instead.

    Notes
    -----
    Module-level ``__getattr__`` (PEP 562) is consulted only after normal lookup
    fails, so it cannot shadow a live name. It also fires for
    ``from aos_state import build_geom_svd``, which is how every stale caller in
    this repository reached the removed function.
    """
    if name in _RETIRED:
        raise AttributeError(
            f"aos_state.{name} was removed. Never build the sensitivity-matrix SVD "
            f"outside OFC code -- degenerate singular-value pairs leave the "
            f"individual v-mode vectors basis-dependent, so all code must share one "
            f"Vh. Use {_RETIRED[name]}.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Component layout of the 50-DOF OFC state (for comp_dof_idx construction)
_DOF_COMPONENTS = {"m2HexPos": (0, 5), "camHexPos": (5, 5),
                   "M1M3Bend": (10, 20), "M2Bend": (30, 20)}


def _comp_dof_idx(dof_indices):
    """Build the OFCData.comp_dof_idx boolean dict from active DOF indices."""
    active = set(dof_indices)
    return {name: np.array([(start + i) in active for i in range(n)], dtype=bool)
            for name, (start, n) in _DOF_COMPONENTS.items()}


def make_state_estimator(config_dir=None, dof_set="standard_22", version="v13",
                         n_modes=None):
    """OFC `StateEstimator` — the single sanctioned v-mode engine.

    Every v-mode in this repository comes from here. `StateEstimator` is what MTAOS
    runs on the summit, so using it everywhere guarantees identical v-modes across
    `olr/`, `blocks/`, `aos/` and `common/`. This matters beyond tidiness: degenerate
    singular-value pairs leave the individual v-mode vectors basis-dependent — only
    v1 and the degenerate-pair magnitudes are unique — so all code must share one
    `Vh`. Never build the SVD outside OFC code.

    Parameters
    ----------
    config_dir : `str`, optional
        OFC config directory. Default resolves `version` via `TS_CONFIG_MTTCS_DIR`.
    dof_set : `str`, optional
        Key into `DOF_SETS`: ``'hexapod_10'``, ``'standard_22'`` or ``'all_50'``.
    version : `str`, optional
        OFC config version used when `config_dir` is None. Must be one whose
        controller selects `REQUIRED_NORM_YAML`.
    n_modes : `int`, optional
        Sets `truncate_index`, the number of v-modes returned. The **mode count is
        set by this, not by the DOF set** — `Vh` is ``(n_dof, n_dof)`` either way, and
        the controller yaml default of 12 silently caps a 34-mode scheme. Default
        leaves the configured value.

    Returns
    -------
    se : `lsst.ts.ofc.state_estimator.StateEstimator`
        With `Vh`, `S`, `normalization_matrix` already built.

    Raises
    ------
    RuntimeError
        If the resolved controller normalization is not `REQUIRED_NORM_YAML`.

    Notes
    -----
    The normalization is asserted rather than overridden, so a misconfigured config
    directory surfaces as an error instead of being silently corrected. Frames are
    handled the same way: `recover_optical_state` requires its wavefront in `ZK_FRAME`
    and refuses any other value rather than rotating it for the caller.

    Every sensitivity matrix derived from this estimator is evaluated at camera rotator
    angle `SMATRIX_ROTATION_ANGLE_DEG` = 0.0 deg, by AOS group decision — evaluating at
    the observed rotator angle would redefine the v-modes visit by visit. Wavefronts
    must therefore be derotated to the telescope frame before inversion. Anything in
    this repository that calls `get_sensitivity_matrix` directly with a nonzero
    rotation angle is departing from that convention and should say why.

    `zn_selected` is set to the 21 rapid-analysis Zernikes for the wavefront-to-DOF
    solve, but it does **not** affect the v-mode basis: `StateEstimator` builds `Vh`
    from the whole sensitivity slab flattened to 899 rows (31 focal x 29 pupil),
    applying no Zernike selection (`state_estimator.py:93-96`); `zn_idx` enters only
    `get_sensitivity_matrix`. See `aos/docs/studies/smatrix_vmode.md`.

    Requires the LSST stack (`lsst.ts.ofc`) and `$TS_CONFIG_MTTCS_DIR`.
    """
    from lsst.ts.ofc import OFCData
    from lsst.ts.ofc.state_estimator import StateEstimator

    if config_dir is None:
        config_dir = resolve_ofc_config_dir(version)
    ofc = OFCData("lsst", config_dir=config_dir)
    ofc.configure_controller()
    got = ofc.controller.get("normalization_weights_filename")
    if got != REQUIRED_NORM_YAML:
        raise RuntimeError(
            f"refusing to build v-modes with normalization {got!r}: this repository "
            f"requires {REQUIRED_NORM_YAML!r}. The bare OFCData('lsst') default is "
            f"{OBSOLETE_NORM_YAML!r}, the obsolete normalization whose sensitivity "
            f"matrix retains a dependence on physical units; its weights differ from "
            f"the required ones non-uniformly per DOF (ratios 10.45 for M2Hex dz, "
            f"948.8 for dx/dy, 0.00579 for rx/ry, dimensionless), so it rotates the "
            f"v-mode basis rather than rescaling it. Pass a config_dir whose "
            f"controller selects {REQUIRED_NORM_YAML!r} (v13 does)."
        )
    # Honoured by get_sensitivity_matrix (the wavefront -> DOF solve); inert for Vh.
    ofc.zn_selected = np.array(ZK_NOLL)
    ofc.comp_dof_idx = _comp_dof_idx(DOF_SETS[dof_set])
    se = StateEstimator(ofc)
    if n_modes is not None:
        se.truncate_index = int(n_modes)
    return se


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
    # The estimator's own truncate_index, not n_modes, decides how many v-modes
    # get_vmodes_from_dofs returns, and OFC defaults it to 12. Asking for more than the
    # estimator was built for used to surface as a bare numpy broadcast error naming
    # neither knob; check it here so the message says which call to fix.
    n_avail = int(getattr(state_estimator, "truncate_index", n_modes))
    if n_modes > n_avail:
        raise ValueError(
            f"asked for {n_modes} v-modes but the StateEstimator returns {n_avail} "
            f"(its truncate_index). Pass n_modes={n_modes} to make_state_estimator "
            f"as well, so the estimator and this call agree."
        )
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
    return f"/sdf/group/rubin/u/roodman/LSST/packages/ts_config_mttcs/MTAOS/{version}/ofc"


# The camera rotator angle at which every sensitivity matrix in this repository is
# evaluated. This is an AOS group decision, not an implementation shortcut: evaluating
# the S-matrix at the observed rotator angle is formally more correct, but it also
# redefines the v-modes visit by visit, which would make the control loop's basis a
# moving target and put a rotator-angle confound into every v-mode time series. The
# fidelity given up is small -- the rotation-dependent wobble in recovered DOF is
# 7.3e-03 to 1.5e-02 relative (dimensionless, over a truth scale of max|DOF| = 645.7 µm
# or arcsec) -- and the near-degenerate singular-value pairs (consecutive fractional
# gaps of 0.78% and 1.75%) mean a small rotation can swap or arbitrarily mix two modes
# rather than rotating the basis smoothly.
#
# Consequence for callers: the WAVEFRONT must be derotated into the telescope frame
# before inversion. Fixing the matrix at rotator zero does not make rotation go away,
# it moves the responsibility to the caller. See ``frame-conventions-ccs-ocs``.
SMATRIX_ROTATION_ANGLE_DEG = 0.0

# The Zernike frame every wavefront entering `recover_optical_state` must be in. OCS is
# the telescope-fixed frame the sensitivity matrix lives in; CCS is camera-fixed and
# rotates with the rotator. Passing CCS Zernikes is not a small error -- recovered DOF
# then drift with rotator angle and invert sign by 90 deg (measured: M2-hexapod dx from
# +81.5 µm at rotator 0 deg to -12.1 µm at 90 deg, against a +100.0 µm truth).
ZK_FRAME = "ocs"

_CORNER_BASIS_CACHE = {}


def corner_recovery_basis(state_estimator):
    """SVD of the corner-evaluated sensitivity, for inverting measured wavefronts.

    The recovery basis, as distinct from the v-mode basis. `StateEstimator.Vh` comes
    from the full Double Zernike (DZ) slab flattened to 899 rows (31 focal x 29 pupil)
    with **no** field-point evaluation and **no** pupil-Zernike selection, so it does
    not span the 84-row (4 corners x 21 Zernikes) problem a measured corner wavefront
    actually poses. Inverting in it leaves an irreducible floor of 2.3e-02 µm of
    wavefront residual RMS even on noiseless data drawn from its own subspace. This
    function decomposes the matrix that is actually being inverted, which closes to
    machine precision (2.1e-14 µm residual RMS).

    Always evaluated at camera rotator angle `SMATRIX_ROTATION_ANGLE_DEG` = 0.0 deg.
    There is deliberately **no** rotation-angle argument: see that constant for the
    reasoning, and note that the wavefront must therefore be derotated into the
    telescope frame before it is passed to `recover_optical_state`.

    Everything comes from `state_estimator`, so the required normalization asserted by
    `make_state_estimator` carries over: the sensitivity is
    `get_sensitivity_matrix(..., normalize=False)` and the weights are its
    `normalization_matrix` diagonal. Verified to reproduce the retired
    ``build_geom_svd`` singular values to ``max|Δs| = 0.000e+00``.

    Parameters
    ----------
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `make_state_estimator`.

    Returns
    -------
    basis : `dict`
        ``U`` `(n_rows, n_sv)`, ``s`` `(n_sv,)` (DZ sensitivity units, µm of wavefront
        per normalized DOF unit), ``V`` `(n_dof, n_sv)`, ``dof_indices`` `list`,
        ``norm_vector`` `(n_dof,)` in per-DOF weight units, ``n_modes`` `int` and
        ``rotation_angle`` `float` [deg], always 0.0.

    Notes
    -----
    `get_sensitivity_matrix` costs about 270 ms per call because it evaluates the DZ
    polynomial, so the result is cached per `state_estimator`. Uncached, a per-visit
    rebuild would cost roughly 5.8 hours over 76,577 visits.
    """
    key = id(state_estimator)
    if key in _CORNER_BASIS_CACHE:
        return _CORNER_BASIS_CACHE[key]

    idx = [int(d) for d in state_estimator.ofc_data.dof_idx]
    field_angles = [state_estimator.ofc_data.sample_points[s] for s in SENSOR_NAMES]
    # Zernike-selected (zn_idx), corner-evaluated, unnormalized: the matrix being inverted.
    sens = state_estimator.get_sensitivity_matrix(
        field_angles, SMATRIX_ROTATION_ANGLE_DEG, normalize=False, truncate=False)
    norm_vector = np.asarray(state_estimator.normalization_matrix).diagonal().copy()
    U, s, Vh = np.linalg.svd(sens @ np.diag(norm_vector), full_matrices=False)
    basis = dict(U=U, s=s, V=Vh.T, dof_indices=idx,
                 norm_vector=norm_vector, n_modes=len(s),
                 rotation_angle=SMATRIX_ROTATION_ANGLE_DEG)
    _CORNER_BASIS_CACHE[key] = basis
    return basis


def recover_optical_state(z_dev, state_estimator, n_modes=None, zk_frame=ZK_FRAME):
    """Recover the optical-state DOF from a measured corner-deviation wavefront.

    This is the ``optical_state``: the DOF obtained by inverting the measured Zernike
    *deviations* (measured OPD minus intrinsic) onto the top-``n_modes`` controllable
    subspace. It is **not** the Trim (aggregated DOF) nor the Tweak — see
    ``aos-dof-terminology``.

    Hybrid by design. The **inversion** uses `corner_recovery_basis`, the SVD of the
    corner-evaluated Zernike-selected matrix, because that is the matrix being
    inverted and it closes to machine precision. The reported **v-modes** are in the
    `make_state_estimator` basis, so they are directly comparable with the commanded
    LUT and Trim v-modes and with what the Main Telescope AOS reports on the summit.
    Those are two different bases: measured principal angles between the retained DOF
    subspaces reach 4.768 deg for ``standard_22``/12 and 89.951 deg for ``all_50``/34,
    so the distinction is not cosmetic.

    Parameters
    ----------
    z_dev : `numpy.ndarray`
        Measured per-corner deviation Zernikes in µm of wavefront, flattened
        corner-major as ``4 corners x len(ZK_NOLL)`` = 84 values, matching
        `SENSOR_NAMES` order. **Must already be derotated into the telescope frame**
        (OCS), because the sensitivity matrix is fixed at rotator zero — see
        `SMATRIX_ROTATION_ANGLE_DEG`. Passing camera-frame (CCS) Zernikes silently
        gives a rotator-dependent answer that inverts sign by 90 deg of rotation.
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `make_state_estimator`.
    n_modes : `int`, optional
        Modes retained in the inversion. Default uses the estimator's
        `truncate_index`, which `make_state_estimator` sets from ``n_modes``.
    zk_frame : `str`, optional
        Frame of `z_dev`; must be `ZK_FRAME` (``'ocs'``). Present so that a caller
        holding camera-frame Zernikes has to confront the frame rather than get a
        plausible-looking wrong answer — the failure is silent in the numbers.

    Returns
    -------
    dof_full : `numpy.ndarray`
        Physical DOF, length 50; the used-DOF subset is filled and the rest are zero.
        Units are µm for translations and bending modes, arcsec for tilts.
    vmode_amps : `numpy.ndarray`
        v-mode amplitudes (dimensionless) in the **`make_state_estimator` basis**,
        length `truncate_index`, from `get_vmodes_from_dofs`.
    zk_constrained : `numpy.ndarray`
        Controllable projection of `z_dev` in µm of wavefront — the part reproducible
        by DOF in the kept subspace. Basis-invariant, and what
        `olr/code/nightly_table.py` consumes.

    Notes
    -----
    Math, with ``w`` the per-DOF normalization weights and ``k = n_modes``:
    ``z_dev = U s V^T (dof_sub / w)`` gives ``v_rec = (U[:, :k].T z_dev) / s[:k]``,
    ``dof_sub = w * (V[:, :k] v_rec)`` and ``zk_constrained = U[:, :k] (s[:k] v_rec)``.
    ``v_rec`` is in the recovery basis and is deliberately **not** returned; the DOF
    are re-projected onto the v-mode basis instead.

    Rows of `z_dev` containing non-finite values make the result non-finite; screen
    them in the caller.
    """
    if str(zk_frame).lower() != ZK_FRAME:
        raise ValueError(
            f"recover_optical_state got zk_frame={zk_frame!r}; it requires "
            f"{ZK_FRAME!r}. The sensitivity matrix is fixed at camera rotator angle "
            f"{SMATRIX_ROTATION_ANGLE_DEG} deg (see SMATRIX_ROTATION_ANGLE_DEG), so the "
            f"wavefront must be derotated into the telescope frame first. Camera-frame "
            f"(CCS) input is not rejected by the numbers -- it returns a "
            f"rotator-dependent answer that inverts sign over 90 deg of rotation "
            f"(measured: M2-hexapod dx from +81.5 to -12.1 µm against a +100.0 µm "
            f"truth) -- which is why this is an explicit argument."
        )

    basis = corner_recovery_basis(state_estimator)
    U, s, V = basis["U"], basis["s"], basis["V"]
    idx, w = basis["dof_indices"], basis["norm_vector"]

    z = np.asarray(z_dev, dtype=float).ravel()
    if z.size != U.shape[0]:
        raise ValueError(
            f"z_dev has {z.size} values; the corner-evaluated matrix has "
            f"{U.shape[0]} rows ({len(SENSOR_NAMES)} corners x {len(ZK_NOLL)} Zernikes, "
            f"corner-major in SENSOR_NAMES order)."
        )

    k = int(n_modes) if n_modes is not None else int(state_estimator.truncate_index)
    k = min(k, len(s))

    v_rec = (U[:, :k].T @ z) / s[:k]
    dof_full = np.zeros(50)
    dof_full[idx] = w * (V[:, :k] @ v_rec)
    zk_con = U[:, :k] @ (s[:k] * v_rec)

    # Report v-modes in the sanctioned basis, not the recovery basis.
    vmode_amps = np.asarray(state_estimator.get_vmodes_from_dofs(dof_full), dtype=float)
    return dof_full, vmode_amps, zk_con


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
