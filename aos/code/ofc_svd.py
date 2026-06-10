"""OFC sensitivity-matrix SVD and degree-of-freedom recovery.

Shared machinery for projecting per-visit Double-Zernike vectors onto the
OFC sensitivity-matrix singular modes (u-modes / v-modes) and recovering
physical degrees of freedom (DOF).  Mirrors the construction used in
``build_measured_intrinsic`` (Path A): the sensitivity matrix is sliced to
a focal-``k`` x pupil-``j`` block, right-multiplied by the geom_mean
normalization, and SVD'd; ``n_keep`` u-modes are retained.

Importing this module is cheap — it only needs numpy.  The actual SVD build
(:func:`build_ofc_svd`) imports ``lsst.ts.ofc`` + ``yaml`` lazily, so callers
on non-RSP machines can import the constants and :func:`recover_dof_per_visit`
without the LSST stack present.

Conventions
-----------
* The OFC sensitivity matrix axis index equals the Noll index: axis 0 is the
  unused Noll-0 placeholder, axis 4 is Z4, etc.  Slicing the focal axis with
  ``[k_min:k_max+1]`` keeps focal Noll ``k_min..k_max`` inclusive; the pupil
  axis is advanced-indexed with the explicit ``iZs`` list (so it skips j=20,
  j=21 when absent).
* Row layout of the flattened ``S`` (and hence the column order of the raw
  DZ vector ``w``) is ``row = (k - k_min) * n_j + j_idx`` — ``k`` outer, ``j``
  inner — captured in :attr:`OFCSvd.kj_grid`.
* v-mode amplitude ``c_i = a_i / sigma_i`` (geom-mean-normalized DOF coords);
  physical DOF = ``N @ V_eff @ diag(1/sigma) @ a``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    'LABELS_50DOF', 'DOF_UNITS_50', 'DOF_GROUPS', 'DEFAULT_NORM_YAML',
    'find_ofc_normalization_yaml', 'load_normalization_weights',
    'recover_dof_per_visit', 'OFCSvd', 'build_ofc_svd',
    'dz_table_to_W', 'project_dz_table', 'vmode_fwhm_scale',
]

# 50-DOF state vector: M2 hexapod (5), Camera hexapod (5),
# M1M3 bending (20), M2 bending (20).
LABELS_50DOF = [
    'M2_dz', 'M2_dx', 'M2_dy', 'M2_rx', 'M2_ry',
    'Cam_dz', 'Cam_dx', 'Cam_dy', 'Cam_rx', 'Cam_ry',
    'B1_1', 'B1_2', 'B1_3', 'B1_4', 'B1_5',
    'B1_6', 'B1_7', 'B1_8', 'B1_9', 'B1_10',
    'B1_11', 'B1_12', 'B1_13', 'B1_14', 'B1_15',
    'B1_16', 'B1_17', 'B1_18', 'B1_19', 'B1_20',
    'B2_1', 'B2_2', 'B2_3', 'B2_4', 'B2_5',
    'B2_6', 'B2_7', 'B2_8', 'B2_9', 'B2_10',
    'B2_11', 'B2_12', 'B2_13', 'B2_14', 'B2_15',
    'B2_16', 'B2_17', 'B2_18', 'B2_19', 'B2_20',
]
DOF_UNITS_50 = (
    ['μm', 'μm', 'μm', 'arcsec', 'arcsec'] +   # M2 hex
    ['μm', 'μm', 'μm', 'arcsec', 'arcsec'] +   # Cam hex
    ['μm'] * 20 +                                        # M1M3 bending
    ['μm'] * 20                                          # M2 bending
)

# Generic 0-based index groups into the 50-DOF vector.  ``hex_piston`` /
# ``hex_decenter`` / ``hex_tiptilt`` interleave Camera-then-M2 for plotting.
DOF_GROUPS = {
    'm2_hex':        list(range(0, 5)),
    'cam_hex':       list(range(5, 10)),
    'm1m3_bending':  list(range(10, 30)),
    'm2_bending':    list(range(30, 50)),
    'hex_piston':    [5, 0],
    'hex_decenter':  [6, 7, 1, 2],
    'hex_tiptilt':   [8, 9, 3, 4],
}

DEFAULT_NORM_YAML = 'range0.5_fwhm-0.15.yaml'   # geom_mean normalization


def find_ofc_normalization_yaml(user_path=None, name=DEFAULT_NORM_YAML):
    """Locate the OFC normalization-weights yaml inside ts_config_mttcs.

    Path is ``$TS_CONFIG_MTTCS_DIR/MTAOS/ofc/normalization_weights/<name>``.
    An explicit ``user_path`` wins when it exists.
    """
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(
                f'user-supplied ofc normalization yaml {p} does not exist')
        return p
    root = os.environ.get('TS_CONFIG_MTTCS_DIR')
    if not root:
        raise RuntimeError(
            'TS_CONFIG_MTTCS_DIR is not set; setup ts_config_mttcs (eups) '
            'or pass an explicit normalization yaml path.')
    target = Path(root) / 'MTAOS' / 'ofc' / 'normalization_weights' / name
    if not target.exists():
        raise FileNotFoundError(
            f'OFC normalization yaml not found at {target} '
            '(try `git pull` in ts_config_mttcs).')
    return target


def load_normalization_weights(user_path=None, name=DEFAULT_NORM_YAML):
    """Return the OFC normalization weights (1-D array) from yaml."""
    import yaml
    with open(find_ofc_normalization_yaml(user_path, name)) as fp:
        return np.array(yaml.safe_load(fp), dtype=float)


def recover_dof_per_visit(A_modes, V, Sigma, N_diag, n_keep):
    """Recover physical DOF estimates per visit from u-mode amplitudes.

    With ``S = S_orig @ N`` (right-side normalized), ``SVD = U Σ Vᵀ``::

        A     = U_eff^T @ w_raw            (μm, mode amps)
        dof_n = V_eff @ diag(1/σ_eff) @ A  (normalized)
        dof   = N @ dof_n                  (physical units)

    Parameters
    ----------
    A_modes : ndarray (n_visits, n_keep)
        u-mode amplitudes per visit (= ``U_effᵀ @ w_raw``).
    V : ndarray (n_dof, n_dof)
    Sigma : ndarray (n_dof,)
    N_diag : ndarray (n_dof,)
        OFC normalization weights (diagonal of N).
    n_keep : int

    Returns
    -------
    dof : ndarray (n_visits, n_dof)
        Physical DOF in the mixed units of :data:`DOF_UNITS_50`.
    """
    V_eff = V[:, :n_keep]
    inv_sig = 1.0 / Sigma[:n_keep]
    A = np.where(np.isfinite(A_modes), A_modes, 0.0)
    dof_norm = (V_eff * inv_sig[None, :]) @ A.T          # (n_dof, n_visits)
    dof_phys = N_diag[:, None] * dof_norm
    return dof_phys.T                                    # (n_visits, n_dof)


@dataclass
class OFCSvd:
    """Result of :func:`build_ofc_svd`: the kept SVD plus the (k, j) layout.

    Use :meth:`project_amplitudes` (raw DZ vectors -> u-mode amps ``A``),
    :meth:`vmodes` (``A`` -> v-mode amplitudes ``c_i = a_i/σ_i``) and
    :meth:`dof` (``A`` -> physical DOF).
    """
    U_eff: np.ndarray
    V: np.ndarray
    Sigma: np.ndarray
    normalization_weights: np.ndarray
    kj_grid: list
    n_keep_eff: int
    k_min: int
    k_max: int
    iZs: list
    n_dof: int
    vmode_labels: list = field(default_factory=list)

    def project_amplitudes(self, W):
        """Raw DZ matrix ``W`` (n_visits, n_kj) in :attr:`kj_grid` order ->
        u-mode amplitudes ``A`` (n_visits, n_keep).  NaNs are treated as 0."""
        return np.where(np.isfinite(W), W, 0.0) @ self.U_eff

    def vmodes(self, A_modes):
        """u-mode amps -> v-mode amplitudes ``c_i = a_i/σ_i``."""
        return np.asarray(A_modes) / self.Sigma[:self.n_keep_eff][None, :]

    def dof(self, A_modes):
        """u-mode amps -> physical DOF (n_visits, n_dof)."""
        return recover_dof_per_visit(
            A_modes, self.V, self.Sigma, self.normalization_weights,
            self.n_keep_eff)


def build_ofc_svd(iZs, k_min, k_max, n_keep, ofc_normalization_yaml=None,
                  instrument='lsst', norm_yaml_name=DEFAULT_NORM_YAML):
    """Build the OFC sensitivity-matrix SVD for a focal-k x pupil-j slice.

    Imports ``lsst.ts.ofc`` lazily.  Returns an :class:`OFCSvd`.
    """
    from lsst.ts.ofc import OFCData

    normalization_weights = load_normalization_weights(
        ofc_normalization_yaml, norm_yaml_name)
    normalization_matrix = np.diag(normalization_weights)

    S_full = np.asarray(OFCData(instrument).sensitivity_matrix)
    iZs_arr = np.asarray(iZs, dtype=int)
    n_k = int(k_max) - int(k_min) + 1
    n_j = len(iZs_arr)
    S_slab = S_full[int(k_min):int(k_max) + 1, iZs_arr, :]   # (n_k, n_j, n_dof)
    n_dof = S_slab.shape[-1]
    S = S_slab.reshape(-1, n_dof) @ normalization_matrix
    kj_grid = [(int(k_min + ki), int(iZs_arr[ji]))
               for ki in range(n_k) for ji in range(n_j)]

    U, Sigma, Vh = np.linalg.svd(S, full_matrices=False)
    V = Vh.T
    n_keep_eff = min(int(n_keep), U.shape[1])
    return OFCSvd(
        U_eff=U[:, :n_keep_eff], V=V, Sigma=Sigma,
        normalization_weights=normalization_weights, kj_grid=kj_grid,
        n_keep_eff=n_keep_eff, k_min=int(k_min), k_max=int(k_max),
        iZs=[int(j) for j in iZs_arr], n_dof=n_dof,
        vmode_labels=[f'v{m + 1}' for m in range(n_keep_eff)])  # 1-based


def dz_table_to_W(fit_table, prefix, kj_grid):
    """Pack per-visit raw DZ vectors into ``W`` (n_visits, n_kj) in
    ``kj_grid`` column order.  Missing columns stay NaN.  Works with any
    table exposing ``colnames`` and column access (astropy QTable / Table)."""
    n = len(fit_table)
    W = np.full((n, len(kj_grid)), np.nan)
    for ci, (k, j) in enumerate(kj_grid):
        col = f'{prefix}_z{j}_c{k}'
        if col in fit_table.colnames:
            W[:, ci] = np.asarray(fit_table[col], dtype=float)
    return W


def project_dz_table(fit_table, prefix, svd):
    """Project a per-visit DZ fit table onto the SVD modes.

    Returns ``(vmodes, dof, A_modes, W)`` where ``vmodes`` is
    (n_visits, n_keep), ``dof`` is (n_visits, n_dof), ``A_modes`` is the
    u-mode amplitudes and ``W`` the raw DZ matrix.
    """
    W = dz_table_to_W(fit_table, prefix, svd.kj_grid)
    A = svd.project_amplitudes(W)
    return svd.vmodes(A), svd.dof(A), A, W


def vmode_fwhm_scale(svd):
    """Per-v-mode FWHM scale ``g_i`` (arcsec per unit ``c_i``) for the
    geom_mean normalization, or None if ``lsst.ts.ofc`` lacks the pieces.

    ``fwhm_i = c_i * g_i``, ``g_i = ||(r_j/n_j) ⊙ v_i||`` with range
    weights ``r_j`` and the loaded normalization ``n_j``; for geom_mean
    ``f_j n_j = sqrt(r_j f_j) = r_j/n_j``.
    """
    try:
        from lsst.ts.ofc import BendModeToForce, OFCData
        ofc = OFCData('lsst')
        m1m3 = ofc.m1m3_force_range / 20
        m2 = ofc.m2_force_range / 20
        range_weights = np.concatenate((
            ofc.rb_stroke,
            m1m3 / np.max(np.abs(BendModeToForce('M1M3', ofc).rot_mat), axis=0),
            m2 / np.max(np.abs(BendModeToForce('M2', ofc).rot_mat), axis=0),
        ))
        rf_sqrt = range_weights / np.asarray(svd.normalization_weights, float)
        return np.sqrt(((rf_sqrt[:, None] * svd.V[:, :svd.n_keep_eff]) ** 2)
                       .sum(axis=0))
    except Exception:
        return None
