"""Track geometry and width measurement for cosmic-ray muon tracks.

Shared by ``find_cosmics.py`` and ``analyze_diffusion.py`` in this study. Pixel
coordinates are (x, y) in the trimmed detector frame, in pixels; images are in
electrons after ISR.

The width method follows Fisher-Levine & Nomerotski 2015, JINST 10 C08006: fit a
straight line to the track, cut it into sections along its length, and fit a
pixel-integrated Gaussian to each section's pixel values against perpendicular
distance from the line. The reported width is the total sigma, the fitted
Gaussian plus the pixel term ``pitch / sqrt(12)`` in quadrature.
"""

import numpy as np
from scipy.optimize import curve_fit

PIXEL_PITCH_UM = 10.0
THICKNESS_UM = 100.0
K_B_OVER_E = 8.617333e-5  # V/K


def nmad(x):
    """Normalized median absolute deviation of the finite values of ``x``."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def line_fit(x, y, w):
    """Intensity-weighted principal axis of a set of pixels.

    Parameters
    ----------
    x, y : `numpy.ndarray`
        Pixel centres, pixels.
    w : `numpy.ndarray`
        Weights, electrons; negative values are clipped to zero.

    Returns
    -------
    fit : `dict`
        ``xc``, ``yc`` centroid (pixels); ``ux``, ``uy`` unit vector along the
        track; ``lam1``, ``lam2`` weighted variances along and across it
        (pixels squared).
    """
    w = np.clip(w, 0, None)
    sw = w.sum()
    xc, yc = (w * x).sum() / sw, (w * y).sum() / sw
    dx, dy = x - xc, y - yc
    cxx = (w * dx * dx).sum() / sw
    cyy = (w * dy * dy).sum() / sw
    cxy = (w * dx * dy).sum() / sw
    evals, evecs = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    ux, uy = evecs[:, 1]
    return dict(xc=xc, yc=yc, ux=ux, uy=uy, lam1=evals[1], lam2=max(evals[0], 0.0))


def project(x, y, fit):
    """Distance along (``t``) and across (``d``) the track line, pixels."""
    dx, dy = x - fit["xc"], y - fit["yc"]
    t = dx * fit["ux"] + dy * fit["uy"]
    d = -dx * fit["uy"] + dy * fit["ux"]
    return t, d


def _pixel_offsets(ux, uy, n_sub=4):
    """Perpendicular offsets of an n_sub x n_sub grid of points within a pixel.

    Averaging a profile over these offsets integrates it over the pixel area, for
    a track with unit vector (ux, uy). Pixels.
    """
    g = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    sx, sy = np.meshgrid(g, g)
    return (-sx * uy + sy * ux).ravel()


def _gauss_pix(d, amp, d0, sig, offsets):
    z = (d[:, None] + offsets[None, :] - d0) / sig
    return amp * np.exp(-0.5 * z * z).mean(axis=1)


def fit_profile(d, v, noise_e, offsets, sig0=0.4, frac_err=0.1, n_iter=3):
    """Fit a zero-background, pixel-integrated Gaussian to a track's cross-section.

    The Gaussian is integrated over each pixel's area, so its width ``sigma_int``
    excludes the pixel term. ``sigma`` adds that term back in quadrature: the
    variance of a 1-pixel square projected on any direction is 1/12 pixel^2, so
    ``sigma = sqrt(sigma_int^2 + 1/12)`` is the total width, comparable to
    sigma0 = pitch / sqrt(12) at zero drift. Per-pixel errors are computed from
    the model, not the data, and refit ``n_iter`` times.

    Parameters
    ----------
    d : `numpy.ndarray`
        Perpendicular distance of each pixel centre from the line, pixels.
    v : `numpy.ndarray`
        Pixel values, electrons.
    noise_e : `float`
        Read noise per pixel, electrons.
    offsets : `numpy.ndarray`
        Sub-pixel perpendicular offsets from `_pixel_offsets`, pixels.
    sig0 : `float`, optional
        Starting value of ``sigma_int``, pixels.
    frac_err : `float`, optional
        Fractional model error per pixel, dimensionless; covers the charge
        varying along the track within one section.
    n_iter : `int`, optional
        Fits with errors from the previous fit's model.

    Returns
    -------
    result : `dict` or `None`
        ``sigma``, ``sigma_int`` and ``sigma_err`` (pixels), ``d0`` (pixels),
        ``amp`` (electrons, peak of the unintegrated Gaussian), ``chi2_dof`` with
        and ``chi2_dof_nomod`` without the fractional model error, and ``ndof``;
        `None` if the fit fails.
    """
    if d.size < 5 or np.nansum(v) <= 0:
        return None

    def model(dd, amp, d0, sig):
        return _gauss_pix(dd, amp, d0, sig, offsets)

    p = [max(v.max(), 1.0), 0.0, sig0]
    mu = np.clip(v, 0, None)
    try:
        for _ in range(n_iter):
            err = np.sqrt(noise_e ** 2 + mu + (frac_err * mu) ** 2)
            p, pcov = curve_fit(model, d, v, p0=p, sigma=err, absolute_sigma=True,
                                bounds=([0, -1.5, 0.05], [np.inf, 1.5, 3.0]),
                                maxfev=5000)
            mu = np.clip(model(d, *p), 0, None)
    except (RuntimeError, ValueError):
        return None
    perr = np.sqrt(np.diag(pcov))
    if not np.isfinite(perr[2]):
        return None
    ndof = d.size - 3
    r = v - mu
    chi2 = (r ** 2 / (noise_e ** 2 + mu + (frac_err * mu) ** 2)).sum() / ndof
    chi2_nomod = (r ** 2 / (noise_e ** 2 + mu)).sum() / ndof
    sig_tot = np.sqrt(p[2] ** 2 + 1 / 12)
    return dict(amp=p[0], d0=p[1], sigma=sig_tot, sigma_int=p[2],
                sigma_err=perr[2] * p[2] / sig_tot, chi2_dof=chi2,
                chi2_dof_nomod=chi2_nomod, ndof=ndof)


def track_sections(img, mask_pix, fit, t_min, t_max, noise_e, sec_len=3.0,
                   window=3.5, trim=0.5, max_sections=10, n_iter=3):
    """Width of a track in sections along its length.

    Parameters
    ----------
    img : `numpy.ndarray`
        Stamp image, electrons; NaN marks masked pixels.
    mask_pix : `numpy.ndarray`
        Boolean, pixels to exclude (other sources, defects).
    fit : `dict`
        Line from `line_fit`, in stamp pixel coordinates.
    t_min, t_max : `float`
        Track ends along the line, pixels.
    noise_e : `float`
        Read noise, electrons.
    sec_len : `float`, optional
        Target section length along the track, pixels.
    window : `float`, optional
        Half width across the track of the pixels used, pixels.
    trim : `float`, optional
        Length dropped at each end before sectioning, pixels.
    max_sections : `int`, optional
        Upper limit on the number of sections.
    n_iter : `int`, optional
        Passed to `fit_profile`.

    Returns
    -------
    sections : `list` [`dict`]
        Per section: ``frac`` (section centre as a fraction of the track
        length, dimensionless, 0 at ``t_min``), the `fit_profile` result,
        ``npix`` and ``charge_per_px`` (electrons per pixel of length).
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t, d = project(xx.ravel(), yy.ravel(), fit)
    v = img.ravel()
    good = np.isfinite(v) & ~mask_pix.ravel() & (np.abs(d) <= window)
    a, b = t_min + trim, t_max - trim
    nsec = int(np.clip(np.floor((b - a) / sec_len), 3, max_sections))
    edges = np.linspace(a, b, nsec + 1)
    offsets = _pixel_offsets(fit["ux"], fit["uy"])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = good & (t >= lo) & (t < hi)
        res = fit_profile(d[sel], v[sel], noise_e, offsets, n_iter=n_iter)
        if res is None:
            continue
        res["frac"] = (0.5 * (lo + hi) - t_min) / (t_max - t_min)
        res["npix"] = int(sel.sum())
        res["charge_per_px"] = float(v[sel].sum() / (hi - lo))
        out.append(res)
    return out


def find_ends(img, fit, t_lo, t_hi, window=3.0, level_frac=0.5, pad=4.0):
    """Track ends at half the median charge per unit length.

    A uniform line segment blurred by any symmetric spread falls to half its
    linear charge density at its true end, so the half-level points recover the
    end even where the charge per pixel is spread thin, as at the back surface.

    Parameters
    ----------
    img : `numpy.ndarray`
        Stamp image, electrons; NaN marks masked pixels.
    fit : `dict`
        Line from `line_fit`.
    t_lo, t_hi : `float`
        First-guess ends along the line, pixels.
    window : `float`, optional
        Half width across the track of the pixels summed, pixels.
    level_frac : `float`, optional
        Fraction of the median linear density defining an end, dimensionless.
    pad : `float`, optional
        Distance searched beyond the first-guess ends, pixels.

    Returns
    -------
    t_a, t_b : `float`
        Ends along the line, pixels.
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t, d = project(xx.ravel(), yy.ravel(), fit)
    v = img.ravel()
    ok = np.isfinite(v) & (np.abs(d) <= window)
    edges = np.arange(t_lo - pad, t_hi + pad + 1.0, 1.0)
    prof, _ = np.histogram(t[ok], bins=edges, weights=v[ok])
    cen = 0.5 * (edges[1:] + edges[:-1])
    inner = (cen > t_lo + 2) & (cen < t_hi - 2)
    if inner.sum() < 3:
        inner = (cen >= t_lo) & (cen <= t_hi)
    level = level_frac * np.median(prof[inner])
    above = np.where(prof >= level)[0]
    if above.size < 2 or level <= 0:
        return t_lo, t_hi
    i, j = above[0], above[-1]

    def cross(k0, k1):
        y0, y1 = prof[k0], prof[k1]
        if y1 == y0:
            return cen[k1]
        return cen[k0] + (level - y0) / (y1 - y0) * (cen[k1] - cen[k0])
    t_a = cross(i - 1, i) if i > 0 else cen[i]
    t_b = cross(j + 1, j) if j < cen.size - 1 else cen[j]
    return float(t_a), float(t_b)


def theory_k(volts, temp_k):
    """Diffusion coefficient k in sigma_diff = k * sqrt(D * dz), dimensionless.

    From Fisher-Levine eq. 3.3, sigma_diff = sqrt(2 (kT/e) D dz / V) for a uniform
    field V / D.

    Parameters
    ----------
    volts : `float`
        Back-bias voltage, V.
    temp_k : `float`
        CCD temperature, K.
    """
    return np.sqrt(2 * K_B_OVER_E * temp_k / volts)
