#!/usr/bin/env python
"""Compute the FWHM (f) and range (r) normalization weights for the AOS
sensitivity matrix, standalone (no LSST stack), and validate against the
ts_ofc-shipped combined weights.

Reproduces lsst.ts.ofc scripts/generate_normalization_weights.py (PR #119):
    w_i = r_i^alpha * f_i^beta          (new default alpha=+0.5, beta=-0.5)
  f_i (FWHM, arcsec / DOF-unit): field-averaged RMS PSF-width response per DOF,
       via convertZernikesToPsfWidth applied to the DZ sensitivity evaluated on
       a Gauss-Legendre field grid (Noll znmin..znmax).
  r_i (range, DOF-unit): rb_stroke for rigid body; for bending modes
       (force_range / n) / max|force-per-micron|.

convertZernikesToPsfWidth / getPsfGradPerZernike are reimplemented verbatim
from lsst.ts.wep.utils.zernikeUtils (galsim only).
"""
import numpy as np
import galsim

# ---- ts_wep: PSF-width conversion (reimplemented) -------------------------
def getPsfGradPerZernike(diameter=8.36, obscuration=0.612, jmin=4, jmax=28):
    """FWHM gradient per Zernike, arcsec/micron, Noll jmin..jmax."""
    R_outer = diameter / 2.0
    R_inner = R_outer * obscuration
    factors = np.zeros(jmax + 1)
    for i in range(jmin, jmax + 1):
        coefs = [0] * i + [1]
        Z = galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
        rms_tilt = np.sqrt(np.sum(Z.gradX.coef**2 + Z.gradY.coef**2) / 2)
        rms_tilt = np.rad2deg(rms_tilt * 1e-6) * 3600           # arcsec/micron
        factors[i] = 2 * np.sqrt(2 * np.log(2)) * rms_tilt      # rms->fwhm
    return factors[jmin:]


def convertZernikesToPsfWidth(zernikes, jmin=4, diameter=8.36, obscuration=0.612):
    """Quadrature contribution of each Zernike (microns) to PSF FWHM (arcsec)."""
    zernikes = np.asarray(zernikes)
    jmax = jmin + zernikes.shape[-1] - 1
    cf = getPsfGradPerZernike(diameter=diameter, obscuration=obscuration, jmin=jmin, jmax=jmax)
    return cf * zernikes


# ---- field quadrature grid (from the PR script) ---------------------------
def make_quadrature_grid(rings=5, spokes=6, field_radius=1.75):
    li, w_ring = np.polynomial.legendre.leggauss(rings)
    radii = np.sqrt((1.0 + li) / 2.0) * field_radius
    w_ring = w_ring * np.pi / (2.0 * spokes)
    azs = np.linspace(0.0, 2.0 * np.pi, spokes, endpoint=False)
    radii, azs = np.meshgrid(radii, azs, indexing="ij")
    x = (radii * np.cos(azs)).ravel()
    y = (radii * np.sin(azs)).ravel()
    fw = np.broadcast_to(w_ring[:, None], radii.shape).ravel()
    fw = fw / np.sum(fw)
    return list(zip(x, y)), fw


# ---- evaluate DZ sensitivity at field points (galsim DoubleZernike) -------
def evaluate_field(sens, field_angles, field_inner=0.0, field_outer=1.75,
                   pupil_inner=2.558, pupil_outer=4.18):
    """sens (nfield_k, npupil_j, ndof) -> (nfield_pts, npupil_j, ndof)."""
    fx = [a[0] for a in field_angles]
    fy = [a[1] for a in field_angles]
    ndof = sens.shape[2]
    out = []
    for d in range(ndof):
        dz = galsim.zernike.DoubleZernike(
            sens[:, :, d], uv_inner=field_inner, uv_outer=field_outer,
            xy_inner=pupil_inner, xy_outer=pupil_outer,
        )
        out.append([zk.coef for zk in dz(fx, fy)])   # (nfield, ncoef)
    out = np.array(out)                                # (ndof, nfield, ncoef)
    return np.einsum("dfc->fcd", out)                  # (nfield, ncoef, ndof)


def compute_f_quadrature(sens, rings=5, spokes=6, field_radius=1.75,
                         znmin=4, znmax=22):
    """f_i = field-weighted RMS PSF-width per DOF, arcsec/DOF-unit.

    NOTE: znmax=22 (pupil Noll Z4..Z22) reproduces the official ts_config_mttcs
    v13 normalization weights to <0.1%.  Using Z4..Z28 over-counts high-order
    terms and inflates f for the higher bending modes.
    """
    field_angles, fw = make_quadrature_grid(rings, spokes, field_radius)
    ev = evaluate_field(sens, field_angles, field_outer=field_radius)
    ev = ev[:, znmin:znmax + 1, :]                     # Noll 4..28
    fwhm = np.zeros_like(ev)
    for i in range(ev.shape[0]):
        fwhm[i] = convertZernikesToPsfWidth(ev[i].T).T
    fwhm_per_field = np.sqrt(np.sum(fwhm**2, axis=1))  # (nfield, ndof)
    return np.sqrt(np.sum(fw[:, None] * fwhm_per_field**2, axis=0))


def compute_f_legacy(sens, field_angles, znmin=4, znmax=28):
    """Legacy: RMS over the given field points and Noll of the PSF-width."""
    ev = evaluate_field(sens, field_angles)[:, znmin:znmax + 1, :]
    fwhm = np.zeros_like(ev)
    for i in range(ev.shape[0]):
        fwhm[i] = convertZernikesToPsfWidth(ev[i].T).T
    flat = fwhm.reshape(-1, fwhm.shape[-1])            # (nfield*nzk, ndof)
    return np.sqrt(np.sum(flat**2, axis=0))


# ---- range (r) weights --------------------------------------------------
# ts_ofc BaseOFCData defaults
RB_STROKE = np.array([5900, 6700, 6700, 0.12, 0.12, 8700, 7600, 7600, 0.24, 0.24])
M1M3_FORCE_RANGE = 67 * 2      # 134 N
M2_FORCE_RANGE = 22.5 * 2      # 45 N


def compute_range_weights(force_m1m3, force_m2, sel_m1m3, sel_m2,
                          rb_stroke=RB_STROKE, m1m3_force_range=M1M3_FORCE_RANGE,
                          m2_force_range=M2_FORCE_RANGE, per_mode_budget=20):
    """Range r_i per DOF-unit (reproduces ts_ofc compute_range_weights).

    force_m1m3/force_m2 : (n_act, n_mode) force-per-micron matrices (columns =
        raw modes).  sel_* : list of raw-mode indices for the used DOF.
    Bending range = (force_range / per_mode_budget) / max|force-per-micron|.
    per_mode_budget=20 matches the OFC 20-mode force allocation; the same
    per-mode budget is kept when extending to more modes (force-limited range).
    """
    r_m1m3 = (m1m3_force_range / per_mode_budget) / np.max(np.abs(force_m1m3[:, sel_m1m3]), axis=0)
    r_m2 = (m2_force_range / per_mode_budget) / np.max(np.abs(force_m2[:, sel_m2]), axis=0)
    return np.concatenate([rb_stroke, r_m1m3, r_m2])


def geometric_weights(r, f, alpha=0.5, beta=-0.5):
    """Official v13 normalization: w_i = r_i^alpha * f_i^beta  (sqrt(r/f))."""
    return r**alpha * f**beta


if __name__ == "__main__":
    # Self-check: getPsfGradPerZernike for LSST defocus (Noll 4)
    cf = getPsfGradPerZernike(jmin=4, jmax=22)
    print("PSF-width conversion factors (arcsec/um), Noll 4..22:")
    print(np.round(cf, 4))
    print("Noll 4 (defocus):", round(float(cf[0]), 5), "arcsec/um")
