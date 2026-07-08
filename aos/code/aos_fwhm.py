"""AOS-FWHM helpers: map a residual double-Zernike wavefront to a PSF FWHM
contribution across the focal plane, via ts_wep's convertZernikesToPsfWidth.

Shared by run_bounce.py (differential correctable-FWHM metric) and, in spirit,
run_wfs_dof_compare.py.  A "residual DZ" dW is a vector over an OFCSvd.kj_grid
(focal-k x pupil-j); evaluating it at a field point gives the pupil-Zernike
vector z_j = sum_k dW_{k,j} Z_k(pos), which converts to an arcsec FWHM
contribution (Z4+ quadrature).  The projection residual (I - U_eff U_eff^T)dW
is the part an OFC correction of that scheme cannot remove.

Needs lsst.ts.intrinsic.wavefront.ofc_svd (focal_zernike_at_points) and, for
the FWHM conversion, lsst.ts.wep.utils.convertZernikesToPsfWidth.  RSP-only.
"""
import numpy as np

FP_RADIUS = 1.75


def fp_grid(R=FP_RADIUS, step=0.35):
    """Area-uniform grid of field points (deg) within radius R (deg)."""
    g = np.arange(-R, R + 1e-9, step)
    xx, yy = np.meshgrid(g, g)
    m = (xx ** 2 + yy ** 2) <= R ** 2
    return np.column_stack([xx[m], yy[m]])


def focal_basis(svd, noll, pos_deg, fp_radius=FP_RADIUS):
    """B (npos*nj, n_kj): focal-plane Zernike basis at field positions pos_deg
    (deg), so  z_j(pos) = (B @ dW).reshape(npos, nj)  for dW over svd.kj_grid."""
    from lsst.ts.intrinsic.wavefront.ofc_svd import focal_zernike_at_points
    jpos = {int(j): i for i, j in enumerate(noll)}
    nj = len(noll)
    B = np.zeros((len(pos_deg) * nj, len(svd.kj_grid)))
    for ci, (tx, ty) in enumerate(pos_deg):
        rho = np.hypot(tx, ty) / fp_radius
        th = np.arctan2(ty, tx)
        for ki, (k, j) in enumerate(svd.kj_grid):
            if int(j) in jpos:
                B[ci * nj + jpos[int(j)], ki] = float(focal_zernike_at_points(k, rho, th))
    return B


def zj_to_fwhm(Z, noll, conv):
    """Z (..., nj) pupil-Zernike vectors (µm) at Noll ``noll`` -> FWHM (arcsec).
    Pads to a Noll-4-start contiguous vector (Z1-3 excluded), applies ts_wep
    convertZernikesToPsfWidth (per-Zernike arcsec) and quadrature-sums."""
    Z = np.atleast_2d(np.asarray(Z, float))
    jmax = max(noll)
    full = np.zeros((Z.shape[0], jmax - 3))          # column 0 == Noll 4
    src = [i for i, j in enumerate(noll) if j >= 4]
    full[:, [noll[i] - 4 for i in src]] = Z[:, src]
    dpsf = np.asarray(conv(full), float)             # (n, jmax-3) arcsec per Zernike
    return np.sqrt(np.nansum(dpsf ** 2, axis=1))


def residual_dW(svd, dW):
    """(I - U_eff U_effᵀ) dW: the part of the residual DZ that the scheme's OFC
    correction (svd.U_eff) cannot remove."""
    dW = np.asarray(dW, float)
    U = svd.U_eff
    return dW - U @ (U.T @ dW)


def fp_fwhm(svd, noll, dW, grid_pos, conv, reduce=np.nanmedian):
    """Reduced (median by default) AOS-FWHM (arcsec) across grid_pos for a
    residual DZ dW (n_kj,)."""
    B = focal_basis(svd, noll, grid_pos)
    z = (np.asarray(dW, float) @ B.T).reshape(len(grid_pos), len(noll))
    return float(reduce(zj_to_fwhm(z, noll, conv)))
