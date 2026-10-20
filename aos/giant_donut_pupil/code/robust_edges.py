"""Robust donut edges against azimuth (step 6 of pupil pass 2).

Per 2 deg azimuth bin the radial profile is built as in `pupil_edges.edges`
(mean of valid pixels per 0.5 pixel radial step), then smoothed over 3 steps
(1.5 pixels): a step holds only a few pixels. A bin's edge is kept when:

* flat top: at least ``MIN_TOP_PIX`` valid pixels between 0.70 and 0.90 r_out
  (well inside both edges), and a flat-top S/N (median level over its
  standard error, from the per-pixel noise) above ``MIN_TOP_SNR``;
* not a spider bin, a Batoid-vignetted bin or a neighbour-masked bin, nor
  adjacent to one;
* a single crossing: inside a window of ±``WINDOW_PIX`` about the expected
  edge (the model edge given for this image; where none is given, the
  harmonic fit of a first pass with a ±0.12 r_out window about 0.62 r_out and
  r_out),
  the profile crosses 50 % of the flat top exactly once, monotonically
  (no reversal larger than 3 sigma, background plus shot noise, between the
  25 % and 75 % levels). If
  it crosses more than once, the crossing nearest the expected edge is taken
  and the bin is flagged ``multi``.

Then each edge, per image, is fitted against azimuth by a circle plus
Fourier harmonics up to m = 4 with a Huber loss (statsmodels RLM, HuberT),
and bins more than 3 robust sigma (nMAD of the RLM residuals) off the fit are
rejected. The per-bin ``quality`` is one of ``ok``, ``multi`` (kept, nearest
crossing), ``flagged`` (spider, vignetted, masked or a neighbour of one),
``low_top`` (too few pixels or low S/N), ``no_crossing``, ``outlier`` (off the
Huber fit).

Run, from ``code/``: called from `pass2_report.py`.
"""
import numpy as np

NBIN = 180
DR_PIX = 0.5
MIN_TOP_PIX = 30
MIN_TOP_SNR = 20.0
WINDOW_PIX = 12.0
M_MAX = 4
CLIP_SIGMA = 3.0
TOP_WINDOW = (0.70, 0.90)
SMOOTH_STEPS = 3
WIDE_WINDOW = 0.12

__all__ = ['profiles', 'bin_edges', 'harmonic_fit', 'robust_edges']


def profiles(image, valid, cx, cy, r_out, noise_e):
    """Per-bin radial profile and flat-top statistics.

    Returns
    -------
    prof : `numpy.ndarray`
        Shape (NBIN, nr), mean electrons per pixel, NaN where no valid pixel.
    rad : `numpy.ndarray`
        Radial step centres, pixels.
    top, top_snr, top_n : `numpy.ndarray`
        Flat-top median level (electrons per pixel), its S/N (dimensionless)
        and the number of valid pixels in the flat-top window, per bin.
    bad : `numpy.ndarray`
        Invalid-pixel counts per bin and radial step.
    """
    n = image.shape[0]
    yy, xx = np.mgrid[:n, :n]
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    phi = np.mod(np.degrees(np.arctan2(dy, dx)), 360.0)
    rmax = 1.3 * r_out
    sel = r < rmax
    kb = np.minimum((phi[sel] / (360 / NBIN)).astype(int), NBIN - 1)
    kr = (r[sel] / DR_PIX).astype(int)
    nr = int(rmax / DR_PIX) + 1
    flat = kb * nr + kr
    good = valid[sel]
    tot = np.bincount(flat[good], image[sel][good], NBIN * nr).reshape(NBIN, nr)
    cnt = np.bincount(flat[good], minlength=NBIN * nr).reshape(NBIN, nr)
    bad = np.bincount(flat[~good], minlength=NBIN * nr).reshape(NBIN, nr)
    with np.errstate(invalid='ignore', divide='ignore'):
        prof = tot / cnt
    rad = (np.arange(nr) + 0.5) * DR_PIX
    tw = (rad > TOP_WINDOW[0] * r_out) & (rad < TOP_WINDOW[1] * r_out)
    top = np.nanmedian(np.where(cnt[:, tw] > 0, prof[:, tw], np.nan), axis=1)
    top_n = cnt[:, tw].sum(axis=1)
    top_snr = top / (noise_e / np.sqrt(np.maximum(top_n, 1)))
    return prof, rad, top, top_snr, top_n, bad, cnt


def _smooth(p):
    """Running mean over `SMOOTH_STEPS` radial steps, ignoring NaN."""
    ok = np.isfinite(p)
    k = np.ones(SMOOTH_STEPS)
    num = np.convolve(np.where(ok, p, 0.0), k, mode='same')
    den = np.convolve(ok.astype(float), k, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _crossings(p, rad, half, lo, hi):
    """Radii where `p` crosses `half` between `lo` and `hi` (linear interpolation)."""
    w = (rad >= lo) & (rad <= hi) & np.isfinite(p)
    i = np.nonzero(w)[0]
    if len(i) < 3:
        return np.array([]), np.array([])
    q = p[i] - half
    s = np.nonzero(np.sign(q[:-1]) * np.sign(q[1:]) < 0)[0]
    r0 = rad[i[s]] + (0 - q[s]) / (q[s + 1] - q[s]) * DR_PIX
    return r0, np.sign(q[s + 1] - q[s])


def bin_edges(prof, rad, top, top_snr, top_n, expected_in, expected_out, noise_e, cnt,
              window=WINDOW_PIX):
    """Inner and outer edge per bin, with a quality code.

    Returns
    -------
    edges : `dict`
        ``r_inner``, ``r_outer`` (pixels, NaN when not measured) and
        ``q_inner``, ``q_outer`` (`str` arrays).
    """
    out = {}
    for edge, exp, direction in (('inner', expected_in, +1), ('outer', expected_out, -1)):
        r = np.full(NBIN, np.nan)
        q = np.array(['no_crossing'] * NBIN, dtype=object)
        for b in range(NBIN):
            if top_n[b] < MIN_TOP_PIX or not np.isfinite(top[b]) or top_snr[b] < MIN_TOP_SNR:
                q[b] = 'low_top'
                continue
            p = _smooth(prof[b])
            half = 0.5 * top[b]
            r0, sg = _crossings(p, rad, half, exp[b] - window, exp[b] + window)
            r0 = r0[sg == direction]
            if len(r0) == 0:
                continue
            k = int(np.argmin(np.abs(r0 - exp[b])))
            r[b] = r0[k]
            q[b] = 'ok' if len(r0) == 1 else 'multi'
            # Monotonic between the 25 % and 75 % levels, within the noise.
            # Background plus source shot noise, per smoothed radial step.
            sig = np.sqrt(noise_e ** 2 + np.clip(p, 0, None)) / np.sqrt(
                np.maximum(SMOOTH_STEPS * cnt[b], 1))
            w = (rad > r[b] - WINDOW_PIX) & (rad < r[b] + WINDOW_PIX) & np.isfinite(p)
            seg = p[w] * direction
            band = (p[w] > 0.25 * top[b]) & (p[w] < 0.75 * top[b])
            if band.sum() > 2:
                d = np.diff(seg[band])
                if np.any(d < -3 * sig[w][band][1:]):
                    q[b] = 'multi'
        out[f'r_{edge}'] = r
        out[f'q_{edge}'] = q
    return out


def harmonic_fit(r, az_deg, keep):
    """Circle plus Fourier harmonics to m = 4, Huber RLM, on the kept bins.

    Returns
    -------
    model : `numpy.ndarray`
        Fitted edge radius at every bin, pixels (NaN if under 2 M_MAX + 4 bins).
    sigma : `float`
        nMAD of the RLM residuals, pixels.
    """
    import statsmodels.api as sm
    from rubinwork.common.utils import nmad
    phi = np.deg2rad(az_deg)
    cols = [np.ones_like(phi)]
    for m in range(1, M_MAX + 1):
        cols += [np.cos(m * phi), np.sin(m * phi)]
    a = np.column_stack(cols)
    use = keep & np.isfinite(r)
    if use.sum() < 2 * M_MAX + 4:
        return np.full(NBIN, np.nan), np.nan
    res = sm.RLM(r[use], a[use], M=sm.robust.norms.HuberT()).fit()
    return a @ res.params, float(nmad(res.resid))


def robust_edges(image, valid, cx, cy, r_out, noise_e, flagged_in, flagged_out,
                 expected_in=None, expected_out=None):
    """All of step 6 for one image.

    Parameters
    ----------
    flagged_in, flagged_out : `numpy.ndarray` [`bool`]
        Bins excluded from the start (spider, vignetted, masked).
    expected_in, expected_out : `numpy.ndarray`, optional
        Model edges (pixels) to centre the crossing window on; 0.62 r_out and
        r_out when absent or NaN.

    Returns
    -------
    res : `dict`
        ``r_inner``, ``r_outer`` (robust, NaN where rejected), the raw crossings
        ``r_inner_raw``/``r_outer_raw``, quality codes, the harmonic models
        ``fit_inner``/``fit_outer`` and their robust sigma (pixels).
    """
    prof, rad, top, top_snr, top_n, bad, cnt = profiles(image, valid, cx, cy, r_out, noise_e)
    ei = np.full(NBIN, 0.62 * r_out) if expected_in is None else \
        np.where(np.isfinite(expected_in), expected_in, 0.62 * r_out)
    eo = np.full(NBIN, r_out) if expected_out is None else \
        np.where(np.isfinite(expected_out), expected_out, r_out)
    az = (np.arange(NBIN) + 0.5) * 360 / NBIN
    # First pass: wide window about the expected edge; its harmonic fit fills
    # the bins where no expected edge is given.
    wide = bin_edges(prof, rad, top, top_snr, top_n, ei, eo, noise_e, cnt,
                     window=WIDE_WINDOW * r_out)
    for edge, flag, exp in (('inner', flagged_in, expected_in), ('outer', flagged_out, expected_out)):
        keep = np.isin(wide[f'q_{edge}'], ('ok', 'multi')) & ~flag
        first, _ = harmonic_fit(wide[f'r_{edge}'], az, keep)
        fill = first if exp is None else np.where(np.isfinite(exp), exp, first)
        if edge == 'inner':
            ei = np.where(np.isfinite(fill), fill, ei)
        else:
            eo = np.where(np.isfinite(fill), fill, eo)
    e = bin_edges(prof, rad, top, top_snr, top_n, ei, eo, noise_e, cnt)
    res = dict(top=top, top_snr=top_snr, top_n=top_n)
    for edge, flag in (('inner', flagged_in), ('outer', flagged_out)):
        flag = flag | np.roll(flag, 1) | np.roll(flag, -1)
        r = e[f'r_{edge}']
        q = e[f'q_{edge}'].copy()
        q[flag] = 'flagged'
        keep = np.isin(q, ('ok', 'multi'))
        model, sigma = harmonic_fit(r, az, keep)
        if np.isfinite(sigma) and sigma > 0:
            off = keep & (np.abs(r - model) > CLIP_SIGMA * max(sigma, 0.1))
            q[off] = 'outlier'
            keep &= ~off
            model, sigma = harmonic_fit(r, az, keep)
        res[f'r_{edge}_raw'] = r
        res[f'r_{edge}'] = np.where(keep, r, np.nan)
        res[f'q_{edge}'] = q
        res[f'fit_{edge}'] = model
        res[f'sigma_{edge}'] = sigma
    return res
