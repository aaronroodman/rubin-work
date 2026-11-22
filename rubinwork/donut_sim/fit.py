"""Danish fit of one donut stamp, as ts_wep's ``DanishAlgorithm`` fits a single donut.

Mirrors ``lsst.ts.wep.estimation.danish.DanishAlgorithm._estimateSingleZk`` (ts_wep
develop, 2026-10): the stamp is binned by its mean over ``binning`` x ``binning`` blocks
and trimmed to odd size, the reference wavefront is the defocused TA (transverse
aberration) Zernikes, Noll 0..78, the fit starts at the reference with ``fwhm`` 1.0
arcsec and flux the image sum, ``fwhm`` is bounded to 0.1-5.0 arcsec, and the sky
variance per binned pixel is the unbinned sky variance over ``binning**2``. Danish's
``chi`` adds the model as Poisson variance.

Import as ``from rubinwork.donut_sim import fit``.
"""
import danish
import numpy as np
from scipy.optimize import least_squares

__all__ = ["bin_stamp", "factory", "centre_offset_px", "fit_donut", "metrics"]

FWHM_START_ARCSEC = 1.0
FWHM_BOUNDS_ARCSEC = (0.1, 5.0)


def bin_stamp(img, binning):
    """Mean over ``binning`` x ``binning`` blocks, then trimmed to odd size (ts_wep
    ``binArray`` and ``prepImage``)."""
    img = np.asarray(img, dtype=float)
    if binning > 1:
        n = img.shape[0] // binning * binning
        img = img[:n, :n].reshape(n // binning, binning, n // binning, binning).mean((1, 3))
    if img.shape[0] % 2 == 0:
        img = img[:-1, :-1]
    return img


def centre_offset_px(npix, binning):
    """Where the chief ray of an odd ``npix`` stamp, centred on its middle pixel, sits
    after `bin_stamp`, relative to Danish's model centre (binned pixels, x and y alike).

    Danish draws its Kolmogorov kernel on a ``(n - 1) // 2`` pixel galsim image, ``n``
    the binned stamp side: when that is even the kernel, and so the model, is centred
    +0.5 px from the middle pixel's centre; when odd, on it (danish 1.4.0).
    """
    c = (npix - 1) / 2
    n = npix // binning
    n_odd = n if n % 2 else n - 1
    danish_centre = (n_odd - 1) / 2 + (0.5 if ((n_odd - 1) // 2) % 2 == 0 else 0.0)
    return (c + 0.5) / binning - 0.5 - danish_centre


def factory(instr, binning, spider_angle):
    """Danish donut factory for a ts_wep `Instrument` at `binning` (pixel scale
    ``instr.pixelSize * binning``); `spider_angle` in deg, `None` leaves the vanes out."""
    return danish.DonutFactory(
        R_outer=instr.radius, R_inner=instr.radius * instr.obscuration,
        mask_params=instr.maskParams, focal_length=instr.focalLength,
        pixel_scale=instr.pixelSize * binning, spider_angle=spider_angle)


def fit_donut(fac, img, z_ref_m, thx_rad, thy_rad, noll, sky_var, lstsq_kwargs):
    """Fit flux, dx, dy, fwhm and the Zernike deviations `noll` from `z_ref_m`.

    Parameters
    ----------
    fac : `danish.DonutFactory`
    img : `numpy.ndarray`
        Binned, odd stamp (electrons per binned pixel, as the binned mean).
    z_ref_m : `numpy.ndarray`
        Reference Zernikes, Noll 0..jmax (m of wavefront).
    thx_rad, thy_rad : `float`
        Field angle, CCS (rad).
    noll : `list` [`int`]
        Fitted Noll indices.
    sky_var : `float`
        Sky variance per binned pixel (electrons²).
    lstsq_kwargs : `dict`
        Extra `scipy.optimize.least_squares` arguments.

    Returns
    -------
    result : `dict`
        ``flux`` (electrons, binned-mean units), ``dx``, ``dy`` (arcsec), ``fwhm``
        (arcsec), ``z_fit_m`` (deviations from the reference, m), ``chi2_dof``
        (chi² over pixels minus parameters), ``nfev``, ``status``, ``success``,
        ``cost``, ``optimality``, and ``model`` (image).
    """
    m = danish.SingleDonutModel(fac, z_ref=z_ref_m, z_terms=list(noll), thx=thx_rad,
                                thy=thy_rad, npix=img.shape[0])
    x0 = m.pack_params(flux=float(img.sum()), dx=0.0, dy=0.0, fwhm=FWHM_START_ARCSEC,
                       bkg=[], z_fit=[0.0] * len(noll))
    lo = m.pack_params(flux=0.0, dx=-np.inf, dy=-np.inf, fwhm=FWHM_BOUNDS_ARCSEC[0],
                       bkg=[], z_fit=[-np.inf] * len(noll))
    hi = m.pack_params(flux=np.inf, dx=np.inf, dy=np.inf, fwhm=FWHM_BOUNDS_ARCSEC[1],
                       bkg=[], z_fit=[np.inf] * len(noll))
    r = least_squares(m.chi, jac=m.jac, x0=x0, bounds=(lo, hi), args=(img, sky_var),
                      **lstsq_kwargs)
    p = m.unpack_params(r.x)
    chi = m.chi(r.x, img, sky_var)
    return dict(flux=p["flux"], dx=p["dx"], dy=p["dy"], fwhm=p["fwhm"],
                z_fit_m=np.asarray(p["z_fit"], float),
                chi2_dof=float(np.sum(chi ** 2) / (chi.size - r.x.size)),
                nfev=int(r.nfev), status=int(r.status), success=bool(r.success),
                cost=float(r.cost), optimality=float(r.optimality),
                model=m.model(**p))


def metrics(sim, mod):
    """``l1`` = half the summed absolute difference of the unit-sum images (fraction of
    the flux misplaced) and ``pow`` = summed squared difference over the summed squared
    sim (residual power fraction); both dimensionless."""
    a, b = sim / sim.sum(), mod / mod.sum()
    return 0.5 * float(np.abs(a - b).sum()), float(((a - b) ** 2).sum() / (a ** 2).sum())
