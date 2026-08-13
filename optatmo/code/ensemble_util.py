"""Shared helpers for the optatmo ensemble plots (ensemble_corners / _vmodes).

Robust linear fit (Huber slope + intercept, NOT forced through the origin), the
donut-blur lookup with a danish-visits fallback, and common scatter / summary
drawing.
"""
import os

import numpy as np
import pandas as pd

import campaign as camp

VISITS = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'


def robust_fit(x, y):
    """Huber robust linear fit y = slope*x + intercept (OLS fallback).

    Returns dict(slope, intercept, r, n) over the finite pairs.  Unlike an
    origin fit this reports a free intercept.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(slope=np.nan, intercept=np.nan, r=np.nan, n=len(x))
    try:
        import statsmodels.api as sm
        rlm = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
        b, s = float(rlm.params[0]), float(rlm.params[1])   # const, slope
    except Exception:
        s, b = (float(v) for v in np.polyfit(x, y, 1))
    return dict(slope=s, intercept=b, r=float(np.corrcoef(x, y)[0, 1]), n=len(x))


def danish_blur(day):
    """{our-seq: median_blur_arcsec} from the danish visits table (fallback)."""
    if not os.path.exists(VISITS):
        return {}
    v = pd.read_parquet(VISITS)
    v = v[v.day_obs == day]
    if 'median_blur_arcsec' not in v.columns:
        return {}
    return {int(s) + 1: float(b) for s, b in zip(v.seq_num, v.median_blur_arcsec)}


def donut_blur(seq, day, fallback):
    """Donut blur (arcsec) for a visit: in-focus-visit value from the cwfs
    parquet if present, else the danish visits-table median."""
    cwf = camp.cwfs_path(int(f'{day}{seq:05d}'))
    if os.path.exists(cwf):
        cw = pd.read_parquet(cwf)
        if 'donut_blur' in cw.columns and np.isfinite(cw['donut_blur']).any():
            return float(np.nanmedian(cw['donut_blur'].to_numpy()))
    return fallback.get(seq, np.nan)


def fit_line(ax, x, y, lim=None):
    """Draw the 1:1 (dashed) and robust fit (red) lines; return the fit dict."""
    f = robust_fit(x, y)
    if lim is None:
        v = np.concatenate([np.asarray(x, float), np.asarray(y, float), [1e-9]])
        v = v[np.isfinite(v)]
        lim = np.nanmax(np.abs(v)) * 1.1 if len(v) else 1.0
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7)               # 1:1
    if np.isfinite(f['slope']):
        xs = np.array([-lim, lim])
        ax.plot(xs, f['slope'] * xs + f['intercept'], 'r-', lw=0.9)  # robust fit
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return f


def scatter_fit(ax, x, y, title, c='C0'):
    """Single-colour scatter with 1:1 + robust fit line; annotate r/m/b."""
    ax.scatter(x, y, s=15, c=c, alpha=0.75)
    f = fit_line(ax, x, y)
    ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    ax.axhline(0, color='0.85', lw=0.4); ax.axvline(0, color='0.85', lw=0.4)
    if np.isfinite(f['slope']):
        ax.set_title(f'{title}  r={f["r"]:+.2f} m={f["slope"]:+.2f} '
                     f'b={f["intercept"]:+.3f}', fontsize=7.5)
    else:
        ax.set_title(f'{title} (n/a)', fontsize=7.5)
    return f


def scatter_fit_xy(ax, x, y, title, c='C4'):
    """Scatter + robust fit over the DATA range (not origin-centred) -- for
    quantities like FWHM that don't straddle zero."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=15, c=c, alpha=0.8)
    f = robust_fit(x, y)
    if m.sum() >= 3 and np.ptp(x[m]) > 0:
        lo = float(min(x[m].min(), y[m].min())); hi = float(max(x[m].max(), y[m].max()))
        pad = 0.05 * (hi - lo + 1e-9); L = np.array([lo - pad, hi + pad])
        ax.plot(L, L, 'k--', lw=0.7)
        ax.plot(L, f['slope'] * L + f['intercept'], 'r-', lw=0.9)
        ax.set_xlim(*L); ax.set_ylim(*L)
        ax.set_title(f'{title} r={f["r"]:+.2f} m={f["slope"]:+.2f} '
                     f'b={f["intercept"]:+.3f}', fontsize=7.5)
    else:
        ax.set_title(f'{title} (n/a)', fontsize=7.5)
    ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    return f


def summary_bars(pdf, plt, labels, fits, title):
    """Three stacked bar charts (Pearson r, slope, intercept) over `labels`."""
    r = [f['r'] for f in fits]; s = [f['slope'] for f in fits]
    b = [f['intercept'] for f in fits]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(3, 1, figsize=(max(8, 0.4 * len(labels)), 9))
    ax[0].bar(x, r, color='C0'); ax[0].axhline(1, color='0.6', lw=0.6, ls=':')
    ax[0].axhline(0, color='k', lw=0.6); ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_ylabel('Pearson r'); ax[0].set_title(title)
    ax[1].bar(x, s, color='C2'); ax[1].axhline(1, color='0.6', lw=0.6, ls=':')
    ax[1].axhline(0, color='k', lw=0.6); ax[1].set_ylabel('robust slope')
    ax[2].bar(x, b, color='C3'); ax[2].axhline(0, color='k', lw=0.6)
    ax[2].set_ylabel('robust intercept')
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
