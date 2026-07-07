#!/usr/bin/env python3
"""Shared GalSim + HSM PSF rendering / FWHM measurement.

Renders galsim.OpticalPSF(Zernikes) convolved with a Kolmogorov atmosphere and
measures adaptive (HSM) moments, giving a "truth" PSF FWHM that captures the
full diffraction + nonlinear-in-aberration behaviour the analytic
convertZernikesToPsfWidth formula approximates.

Used by both run_psf_fp_maps.py (focal-plane PSF maps / closed-loop sims) and
run_wfs_dof_compare.py (GalSim delivered-FWHM cross-check of the formula page).

The Kolmogorov atmosphere and the Aperture (pupil) are wavefront-independent and
built ONCE per (lambda) via build_psf_tools; only the per-wavefront
OpticalPSF.drawImage + FindAdaptiveMom cost is paid per call.
"""
import numpy as np

PIXSCALE = 0.2          # arcsec / pixel (LSSTCam)
DIAM, OBSC = 8.36, 0.612
ATM_FWHM = 0.6          # arcsec, Kolmogorov seeing
STAMP = 64              # pixels
LAM_NM = {'u': 368., 'g': 478., 'r': 622., 'i': 754., 'z': 869., 'y': 971.}
LN256 = np.log(256.0)


def _higher_moments(arr, ix0, iy0, Mxx, Mxy, Myy):
    """Standardized 3rd/4th moments in the adaptive-Gaussian-whitened frame."""
    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - ix0; dy = yy - iy0
    M = np.array([[Mxx, Mxy], [Mxy, Myy]]); Minv = np.linalg.inv(M)
    r2 = Minv[0, 0] * dx**2 + 2 * Minv[0, 1] * dx * dy + Minv[1, 1] * dy**2
    w = np.exp(-0.5 * r2) * arr
    ev, R = np.linalg.eigh(M)
    Mhalfinv = R @ np.diag(1.0 / np.sqrt(ev)) @ R.T          # M^{-1/2}
    xi = Mhalfinv[0, 0] * dx + Mhalfinv[0, 1] * dy
    eta = Mhalfinv[1, 0] * dx + Mhalfinv[1, 1] * dy
    s = w.sum()
    m = lambda p, q: float((w * xi**p * eta**q).sum() / s)   # noqa: E731
    M30, M21, M12, M03 = m(3, 0), m(2, 1), m(1, 2), m(0, 3)
    M40, M22, M04 = m(4, 0), m(2, 2), m(0, 4)
    return dict(coma1=M30 + M12, coma2=M21 + M03,
                trefoil1=M30 - 3 * M12, trefoil2=3 * M21 - M03,
                kurtosis=M40 + M04 + 2 * M22)


def build_psf_tools(lam_nm):
    """Kolmogorov atmosphere + cached Aperture (pupil reused across all renders)."""
    import galsim
    atm = galsim.Kolmogorov(fwhm=ATM_FWHM)
    aper = galsim.Aperture(diam=DIAM, obscuration=OBSC, lam=lam_nm)
    return atm, aper


def render_measure(zk_um, noll, lam_nm, atm, aper, with_optics=True):
    """Render OpticalPSF(zk)⊗atm, measure with HSM. Returns dict or None on failure.

    Returns fwhm (arcsec), T (trace of the 2nd-moment matrix, arcsec^2 —
    additive under convolution so the optics-only trace = T_total - T_atm),
    ellipticity (e1,e2,e) and the standardized higher moments."""
    import galsim
    if with_optics:
        ab = np.zeros(max(noll) + 1)
        lam_um = lam_nm / 1000.0
        for j, z in zip(noll, zk_um):
            if np.isfinite(z):
                ab[j] = z / lam_um                      # waves
        psf = galsim.Convolve(
            galsim.OpticalPSF(lam=lam_nm, diam=DIAM, aper=aper, aberrations=ab.tolist()), atm)
    else:
        psf = atm
    try:
        img = psf.drawImage(nx=STAMP, ny=STAMP, scale=PIXSCALE)
        res = galsim.hsm.FindAdaptiveMom(img, strict=True)
    except Exception:
        return None
    sig = res.moments_sigma                              # pixels (det radius)
    e1, e2 = res.observed_shape.e1, res.observed_shape.e2
    e = np.hypot(e1, e2)
    T = 2.0 * sig**2 / np.sqrt(max(1.0 - e**2, 1e-6))    # pixels^2 (trace)
    fwhm = np.sqrt(T / 2.0 * LN256) * PIXSCALE           # arcsec
    out = dict(fwhm=fwhm, T=T * PIXSCALE**2, e1=e1, e2=e2, e=e)
    Mxx = (T / 2) * (1 + e1); Myy = (T / 2) * (1 - e1); Mxy = (T / 2) * e2
    ix0 = res.moments_centroid.x - img.xmin; iy0 = res.moments_centroid.y - img.ymin
    out.update(_higher_moments(img.array, ix0, iy0, Mxx, Mxy, Myy))
    return out


def atm_trace(lam_nm, atm=None, aper=None):
    """Second-moment trace (arcsec^2) of the pure Kolmogorov atmosphere — the
    quantity subtracted from a rendered PSF's trace to isolate the optics."""
    if atm is None:
        atm, aper = build_psf_tools(lam_nm)
    r = render_measure(None, [4], lam_nm, atm, aper, with_optics=False)
    return r['T']


def optics_fwhm(zk_um, noll, lam_nm, atm, aper, T_atm):
    """GalSim+HSM optics-only FWHM (arcsec) for one wavefront: render
    OpticalPSF(zk)⊗Kolmogorov, HSM, then remove the atmosphere in the additive
    second-moment-trace domain (T_optics = T_total - T_atm →
    FWHM = sqrt(T_optics/2·ln256)).  NaN on HSM failure; 0 if the optics trace
    underflows (wavefront ≈ 0)."""
    r = render_measure(zk_um, noll, lam_nm, atm, aper, with_optics=True)
    if r is None:
        return np.nan
    T_opt = r['T'] - T_atm
    if T_opt <= 0:
        return 0.0
    return float(np.sqrt(T_opt / 2.0 * LN256))


# --------------------------------------------------------------------------
# Parallel batch: many independent wavefronts -> optics FWHM.
# GalSim objects are rebuilt per worker (cheap, once) rather than pickled.
# --------------------------------------------------------------------------
_W = {}          # per-worker cache: lam_nm -> (atm, aper, T_atm)


def _worker_init(lam_nm):
    atm, aper = build_psf_tools(lam_nm)
    _W['tools'] = (lam_nm, atm, aper, atm_trace(lam_nm, atm, aper))


def _worker_fwhm(args):
    zk_um, noll = args
    lam_nm, atm, aper, T_atm = _W['tools']
    return optics_fwhm(zk_um, noll, lam_nm, atm, aper, T_atm)


def optics_fwhm_batch(zk_rows, noll, lam_nm, workers=1):
    """optics FWHM (arcsec) for a list/array of wavefront vectors (each length
    len(noll)).  ``workers`` > 1 uses a process pool (each worker builds its own
    atm/aper once).  Returns a float array aligned to ``zk_rows``."""
    zk_rows = [np.asarray(z, float) for z in zk_rows]
    if workers and workers > 1 and len(zk_rows) > workers:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_worker_init, initargs=(lam_nm,)) as pool:
            return np.array(pool.map(_worker_fwhm, [(z, noll) for z in zk_rows]))
    atm, aper = build_psf_tools(lam_nm)
    T_atm = atm_trace(lam_nm, atm, aper)
    return np.array([optics_fwhm(z, noll, lam_nm, atm, aper, T_atm) for z in zk_rows])
