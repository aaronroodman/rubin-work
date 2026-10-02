"""Atmospheric seeing applied to defocused donut images, and Danish donut fits.

Two ways of adding a long-exposure atmospheric point-spread function (PSF) to a
donut are provided:

- separable: convolve the optics-only donut image on the detector with the
  atmospheric PSF scaled by the plate scale of that detector plane;
- ray kick: give every ray entering the pupil its own angular deflection drawn
  from the atmospheric PSF, then trace it through the optics.

and a wrapper around ``danish.SingleDonutModel`` that fits a (geometric-optics)
donut model with a Kolmogorov atmospheric kernel, returning the apparent
Zernike coefficients and seeing FWHM.

Units: lengths in metres, angles in radians unless a name ends in ``_arcsec``;
Zernike coefficients in metres of wavefront (Noll indexing, annular, eps 0.612)
unless stated otherwise.
"""

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.signal import fftconvolve

import fresnel_donut as fd     # imports finufft before batoid (OpenMP order)
import batoid
import galsim

ARCSEC = np.pi / 180 / 3600    # rad per arcsec


def vonkarman_r0(fwhm_arcsec, wavelength, L0):
    """Fried parameter r0 (m, at `wavelength`) giving a von Karman PSF FWHM.

    Parameters
    ----------
    fwhm_arcsec : `float`
        Target FWHM of the long-exposure von Karman PSF, arcsec.
    wavelength : `float`
        Wavelength, m.
    L0 : `float`
        Outer scale, m.
    """
    lam_nm = wavelength * 1e9

    def resid(r0):
        return galsim.VonKarman(lam=lam_nm, r0=r0, L0=L0).calculateFWHM(
            scale=0.005) - fwhm_arcsec

    return brentq(resid, 0.03, 1.0, xtol=1e-6)


def vonkarman(fwhm_arcsec, wavelength, L0):
    """`galsim.VonKarman` with the given FWHM (arcsec), wavelength (m), L0 (m)."""
    r0 = vonkarman_r0(fwhm_arcsec, wavelength, L0)
    return galsim.VonKarman(lam=wavelength * 1e9, r0=r0, L0=L0)


def plate_scale(optic, wavelength, dtheta=1e-5):
    """Detector displacement per unit field angle, m/rad, from the chief ray.

    Evaluated at the optic's own detector plane, so for a defocused detector
    it includes the non-telecentric term (displacement grows by
    ~(1 + dz / L_exit_pupil)).
    """
    c0 = fd.chief_ray_point(optic, wavelength, 0.0, 0.0)
    cx = fd.chief_ray_point(optic, wavelength, dtheta, 0.0)
    return (cx[0] - c0[0]) / dtheta


def psf_kernel(psf, pixel, plate, npix=601):
    """Atmospheric PSF drawn as a detector-plane kernel, unit sum.

    Parameters
    ----------
    psf : `galsim.GSObject`
        Atmospheric PSF in arcsec.
    pixel : `float`
        Kernel sample spacing on the detector, m.
    plate : `float`
        Plate scale of the detector plane, m/rad.
    npix : `int`
        Kernel size (odd), samples.
    """
    scale_arcsec = pixel / plate / ARCSEC
    k = psf.drawImage(nx=npix, ny=npix, scale=scale_arcsec,
                      method='no_pixel').array
    return k / k.sum()


def smear(image, kernel):
    """Convolve an image [y, x] with a centred kernel; unit total flux out."""
    out = fftconvolve(image, kernel, mode='same')
    out = np.clip(out, 0.0, None)
    return out / out.sum()


def rebin(image, factor):
    """Sum `factor` x `factor` blocks of an image; unit total flux out."""
    ny, nx = image.shape
    out = image[:ny // factor * factor, :nx // factor * factor].reshape(
        ny // factor, factor, nx // factor, factor).sum(axis=(1, 3))
    return out / out.sum()


def kicked_positions(optic, wavelength, nrays, psf, seed=0,
                     batch=4_000_000):
    """Detector positions of rays each deflected by an atmospheric angle.

    Rays are drawn uniformly over the entrance pupil; each gets a field angle
    drawn from `psf` (by GalSim photon shooting) and is traced through the
    optic.

    Returns
    -------
    x, y : `numpy.ndarray`
        Detector positions (local frame) of the unvignetted rays, m.
    """
    rng = np.random.default_rng(seed)
    gs_rng = galsim.BaseDeviate(seed + 1)
    half = optic.pupilSize / 2
    xs, ys = [], []
    remaining = nrays
    while remaining > 0:
        n = min(batch, remaining)
        u = rng.uniform(-half, half, n)
        v = rng.uniform(-half, half, n)
        photons = psf.shoot(n, gs_rng)
        rv = batoid.RayVector.fromFieldAngles(
            photons.x * ARCSEC, photons.y * ARCSEC, optic=optic,
            wavelength=wavelength, x=u, y=v)
        optic.trace(rv)
        ok = ~rv.vignetted & ~rv.failed
        xs.append(rv.x[ok])
        ys.append(rv.y[ok])
        remaining -= n
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Danish fits
# ---------------------------------------------------------------------------

def reference_zernikes(optic, wavelength, jmax=66, eps=0.612,
                       focal_length=10.31):
    """Transverse-aberration Zernikes of the (defocused) optic, m of wavefront.

    This is the geometric reference that a Danish fit adds its fitted offsets
    to (``batoid.analysis.zernikeTA``, chief-ray referenced): Danish maps pupil
    position u to focal position ``-focal_length * grad W(u)`` with these
    coefficients, so they must reproduce the traced ray positions.

    Notes
    -----
    For the defocused Rubin donut, truncating at jmax = 28 leaves 0.17 um rms
    (0.7 um max) ray-position error and a spurious -0.005 arcsec intra/extra
    difference in the fitted seeing FWHM; jmax = 66 gives 0.004 um rms.
    """
    return batoid.analysis.zernikeTA(
        optic, 0.0, 0.0, wavelength, nrad=20, naz=120, reference='chief',
        jmax=jmax, eps=eps, focal_length=focal_length) * wavelength


def ta_ray_residual(optic, wavelength, z_ref, nrays=400_000, seed=3,
                    R_outer=4.18, eps=0.612, focal_length=10.31):
    """RMS and max distance (m) between traced rays and Danish's mapping.

    Danish places the ray from pupil point u at ``-focal_length * grad W(u)``
    relative to the chief ray, with W the Zernike series `z_ref` (m). This
    returns how far the traced batoid rays land from those positions.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-R_outer, R_outer, nrays)
    v = rng.uniform(-R_outer, R_outer, nrays)
    rv = batoid.RayVector.fromStop(u, v, optic=optic, wavelength=wavelength,
                                   dirCos=fd.dircos(0.0, 0.0))
    optic.trace(rv)
    ok = ~rv.vignetted & ~rv.failed
    c = fd.chief_ray_point(optic, wavelength)
    Z = galsim.zernike.Zernike(z_ref, R_outer=R_outer, R_inner=eps * R_outer)
    dx = rv.x[ok] - c[0] + focal_length * Z.gradX(u[ok], v[ok])
    dy = rv.y[ok] - c[1] + focal_length * Z.gradY(u[ok], v[ok])
    d = np.hypot(dx, dy)
    return np.sqrt(np.mean(d**2)), d.max()


def danish_fit(image, z_ref, z_terms=tuple(range(4, 23)), flux=1e6,
               sky_var=100.0, fwhm0=0.8, R_outer=4.18, eps=0.612,
               focal_length=10.31, pixel_scale=10e-6):
    """Fit a single donut with Danish's geometric model plus Kolmogorov seeing.

    Parameters
    ----------
    image : `numpy.ndarray`
        Donut stamp [y, x] (odd size), any normalisation; rescaled to `flux`.
    z_ref : `numpy.ndarray`
        Reference Zernikes, m of wavefront (from `reference_zernikes`).
    z_terms : `tuple` of `int`
        Noll indices fitted as offsets to `z_ref`.
    flux : `float`
        Total counts the stamp is scaled to (noise-free), counts.
    sky_var : `float`
        Sky variance per pixel used in the chi weights, counts^2.
    fwhm0 : `float`
        Starting Kolmogorov FWHM, arcsec.

    Returns
    -------
    result : `dict`
        ``fwhm`` (arcsec, Kolmogorov), ``dx``, ``dy`` (arcsec), ``z_fit``
        (m, offsets for `z_terms`), ``z_apparent`` (m, full coefficient array
        z_ref + offsets), ``resid_rms`` (fraction of peak pixel), ``cost``,
        ``nfev``, ``status``.
    """
    import danish
    npix = image.shape[0]
    factory = danish.DonutTriangleFactory(
        R_outer=R_outer, R_inner=eps * R_outer, pupil_R_outer=R_outer,
        pupil_R_inner=eps * R_outer, focal_length=focal_length,
        pixel_scale=pixel_scale)
    model = danish.SingleDonutModel(factory, z_ref=z_ref, z_terms=z_terms,
                                    thx=0.0, thy=0.0, npix=npix)
    data = image / image.sum() * flux
    x0 = model.pack_params(flux=flux, dx=0.0, dy=0.0, fwhm=fwhm0,
                           z_fit=[0.0] * len(z_terms))
    lower = [0.0, -5.0, -5.0, 0.05] + [-np.inf] * len(z_terms)
    upper = [np.inf, 5.0, 5.0, 5.0] + [np.inf] * len(z_terms)
    r = least_squares(model.chi, x0=x0, jac=model.jac, args=(data, sky_var),
                      bounds=(lower, upper), x_scale='jac', ftol=1e-10,
                      xtol=1e-10, gtol=1e-10, max_nfev=200)
    p = model.unpack_params(r.x)
    z_app = np.array(z_ref, dtype=float)
    for term, dz in zip(z_terms, p['z_fit']):
        z_app[term] += dz
    mod = model.model(**p)
    return dict(fwhm=p['fwhm'], dx=p['dx'], dy=p['dy'],
                z_fit=np.array(p['z_fit']), z_apparent=z_app,
                resid_rms=np.sqrt(np.mean((mod - data)**2)) / data.max(),
                cost=r.cost, nfev=r.nfev, status=r.status, model=mod / flux)


# ---------------------------------------------------------------------------
# Edge widths
# ---------------------------------------------------------------------------

def edge_widths(r, profile, lo=0.2, hi=0.8):
    """10/90-style widths of the inner (rising) and outer (falling) edges.

    The plateau level is the median of `profile` between the radii where it
    first and last exceeds half its maximum, shrunk inward by 25% of that span
    on each side. Returns the radial distance between the `lo` and `hi`
    crossings of that level on each edge, m, as ``(inner, outer)``.
    """
    half = profile > 0.5 * profile.max()
    i0, i1 = np.argmax(half), len(half) - 1 - np.argmax(half[::-1])
    span = i1 - i0
    plateau = np.median(profile[i0 + span // 4:i1 - span // 4])
    mid = (i0 + i1) // 2

    def crossing(level, rising):
        if rising:
            seg_r, seg_p = r[:mid], profile[:mid]
            j = np.argmax(seg_p >= level)
        else:
            seg_r, seg_p = r[mid:][::-1], profile[mid:][::-1]
            j = np.argmax(seg_p >= level)
        return np.interp(level, [seg_p[j - 1], seg_p[j]], [seg_r[j - 1], seg_r[j]])

    inner = crossing(hi * plateau, True) - crossing(lo * plateau, True)
    outer = crossing(lo * plateau, False) - crossing(hi * plateau, False)
    return inner, outer
