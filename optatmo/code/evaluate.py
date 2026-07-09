"""
Accuracy evaluation: momfit analytic (unweighted, geometric) moments vs
HSM adaptive-weighted moments of a Fraunhofer(optics) (x) VonKarman(seeing)
PSF -- the estimator PIFF/OptAtmoPSF actually fits to Rubin stars.

For an ensemble of realistic aberration states swept over a range of seeing
FWHM, we compute both estimators and ask, per moment, whether the analytic
value predicts the HSM value via a *stable* linear relation (so a fixed
calibration would work in a fit), and how the residual compares to the
per-star HSM measurement error and to the moment's spread across the ensemble
(the "signal" the fit must resolve).

Outputs: a scatter figure (analytic vs HSM, coloured by FWHM), an .npz of all
measurements, and a printed summary that FINDINGS.md is built from.

Run:  /opt/local/bin/python3 evaluate.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moments_analytic import AnnularPupil, analytic_moments
from moments_hsm import measure_hsm_moments
import psf_model as pm

# ---------------- configuration ----------------
SEED = 202607
N_STATES = 160
FWHM_GRID = np.array([0.6, 0.8, 1.0, 1.2])   # arcsec
L0 = 25.0
SNR_TARGET = 300.0        # per-star flux S/N for error scale
NGRID = 256               # pupil sampling for analytic moments
OUTDIR = 'output'

# Per-term rms of the free aberration state (microns), Noll 1-indexed, now
# spanning Z4-Z22.  Amplitudes decrease with order, scaled to the MIW field
# variation (Z5-Z8 ~0.1 um).  A per-state amplitude multiplier (AMP_TIERS)
# additionally probes a large dynamic range of aberration strength.
NMAX = 22
ABER_RMS = np.zeros(NMAX + 1)
ABER_RMS[4] = 0.12                       # defocus
ABER_RMS[5:9] = 0.12                     # astig / coma (Z5-Z8)
ABER_RMS[9:12] = 0.06                    # trefoil / spherical (Z9-Z11)
ABER_RMS[12:16] = 0.04                   # 2nd astig/coma (Z12-Z15)
ABER_RMS[16:23] = 0.025                  # higher (Z16-Z22)
AMP_TIERS = np.array([0.5, 1.0, 2.0, 3.0])   # per-state amplitude multipliers

MOMENTS_2 = ['e0', 'e1', 'e2']
MOMENTS_3 = ['M21', 'M12', 'M30', 'M03']
MOMENTS_4 = ['M22', 'M31', 'M13', 'M40', 'M04']
ALL_MOMENTS = MOMENTS_2 + MOMENTS_3 + MOMENTS_4


def build_ensemble(rng):
    states = rng.normal(0.0, 1.0, size=(N_STATES, NMAX + 1)) * ABER_RMS[None, :]
    tiers = AMP_TIERS[np.arange(N_STATES) % len(AMP_TIERS)]
    states = states * tiers[:, None]
    return states, tiers


def run():
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pup = AnnularPupil(diam=pm.DIAM, obscuration=pm.OBSCURATION, ngrid=NGRID)
    aper = pm.make_aperture()
    states, tiers = build_ensemble(rng)
    wf_rms = np.sqrt((states ** 2).sum(axis=1))   # total wavefront rms (um)

    nrows = N_STATES * len(FWHM_GRID)
    rec = {m + '_hsm': np.full(nrows, np.nan) for m in ALL_MOMENTS}
    rec.update({m + '_err': np.full(nrows, np.nan) for m in ALL_MOMENTS})
    rec.update({m + '_an': np.full(nrows, np.nan) for m in ALL_MOMENTS})
    rec.update({m + '_ano': np.full(nrows, np.nan) for m in ALL_MOMENTS})  # optics-only
    fwhm_col = np.full(nrows, np.nan)
    state_col = np.full(nrows, -1)
    wfrms_col = np.full(nrows, np.nan)

    k = 0
    nfail = 0
    for si in range(N_STATES):
        coef = states[si]
        # optics-only analytic (seeing independent) -- compute once per state
        ano = analytic_moments(coef, pup)
        for fwhm in FWHM_GRID:
            img = pm.draw_psf(coef, fwhm, L0=L0, aper=aper, flux=1.0)
            # set per-pixel noise for target flux S/N
            p = img.array.astype(float)
            nv = (p ** 2).sum() / SNR_TARGET ** 2
            mom, err = measure_hsm_moments(img, noise_var=nv)
            if mom is None:
                nfail += 1
                k += 1
                continue
            sig_atm = fwhm / 2.3548
            an = analytic_moments(coef, pup, sigma_atm_arcsec=sig_atm)
            for m in ALL_MOMENTS:
                rec[m + '_hsm'][k] = mom[m]
                rec[m + '_err'][k] = np.sqrt(err[m])
                rec[m + '_an'][k] = an[m]
                rec[m + '_ano'][k] = ano[m]
            fwhm_col[k] = fwhm
            state_col[k] = si
            wfrms_col[k] = wf_rms[si]
            k += 1
        if (si + 1) % 25 == 0:
            print(f'  {si+1}/{N_STATES} states done')

    rec['fwhm'] = fwhm_col
    rec['state'] = state_col
    rec['wf_rms'] = wfrms_col
    np.savez(os.path.join(OUTDIR, 'optatmo_moment_accuracy.npz'), **rec)
    print(f'HSM failures: {nfail}/{nrows}')

    summary = analyze(rec)
    make_figure(rec, summary)
    return rec, summary


def _linfit(x, y):
    """Least-squares y = a*x + b; return a, b, R^2, residual rms."""
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (a * x + b)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss if ss > 0 else np.nan
    return a, b, r2, np.sqrt(np.mean(resid ** 2))


def analyze(rec):
    """Per-moment: linear analytic->HSM relation, slope stability vs seeing,
    residual vs measurement error and vs ensemble signal."""
    summary = {}
    for m in ALL_MOMENTS:
        # use the atmosphere-augmented analytic for 2nd order (size), optics-only
        # for the seeing-independent shape/higher moments -- report both.
        for tag, an_key in [('an', m + '_an'), ('ano', m + '_ano')]:
            x = rec[an_key]
            y = rec[m + '_hsm']
            a, b, r2, rms = _linfit(x, y)
            # slope per FWHM bin (stability)
            slopes = []
            for fwhm in FWHM_GRID:
                sel = rec['fwhm'] == fwhm
                if sel.sum() > 5:
                    slopes.append(_linfit(x[sel], y[sel])[0])
            slope_spread = np.std(slopes) / (abs(np.mean(slopes)) + 1e-30)
            # slope stability across amplitude (low vs high wavefront rms)
            amp = rec['wf_rms']
            med_amp = np.nanmedian(amp)
            aslopes = []
            for sel in [amp <= med_amp, amp > med_amp]:
                if sel.sum() > 5:
                    aslopes.append(_linfit(x[sel], y[sel])[0])
            amp_slope_spread = (np.std(aslopes) / (abs(np.mean(aslopes)) + 1e-30)
                                if len(aslopes) == 2 else np.nan)
            med_err = np.nanmedian(rec[m + '_err'])
            signal = np.nanstd(y)
            summary[(m, tag)] = dict(slope=a, intercept=b, r2=r2, resid_rms=rms,
                                     slope_spread=slope_spread,
                                     amp_slope_spread=amp_slope_spread,
                                     med_err=med_err, signal=signal,
                                     resid_over_err=rms / (med_err + 1e-30),
                                     resid_over_signal=rms / (signal + 1e-30))
    # print
    print('\n=== analytic -> HSM linear relation (optics-only analytic, '
          'parity-corrected; bigger Z4-Z22 ensemble) ===')
    print(f'{"moment":8s} {"slope":>8s} {"R^2":>7s} {"resid/sig":>10s} '
          f'{"resid/err":>10s} {"slopeVar_see":>12s} {"slopeVar_amp":>12s}')
    for m in ALL_MOMENTS:
        s = summary[(m, 'ano')]
        print(f'{m:8s} {s["slope"]:8.3f} {s["r2"]:7.3f} '
              f'{s["resid_over_signal"]:10.2f} {s["resid_over_err"]:10.2f} '
              f'{s["slope_spread"]:12.2f} {s["amp_slope_spread"]:12.2f}')
    print('\n(size e0 uses atmosphere-augmented analytic:)')
    s = summary[('e0', 'an')]
    print(f'e0(+atm) slope {s["slope"]:.3f} R^2 {s["r2"]:.3f} '
          f'resid/sig {s["resid_over_signal"]:.2f} slopeVar {s["slope_spread"]:.2f}')
    return summary


def make_figure(rec, summary):
    import os
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    panels = MOMENTS_2 + [''] + MOMENTS_3 + MOMENTS_4
    cmap = plt.cm.viridis
    norm = plt.Normalize(FWHM_GRID.min(), FWHM_GRID.max())
    for ax, m in zip(axes.flat, panels):
        if m == '':
            ax.axis('off')
            continue
        tag = 'an' if m == 'e0' else 'ano'
        x = rec[m + '_' + tag]
        y = rec[m + '_hsm']
        sc = ax.scatter(x, y, c=rec['fwhm'], cmap=cmap, norm=norm, s=8, alpha=0.6)
        s = summary[(m, tag)]
        lim = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(lim, s['slope'] * lim + s['intercept'], 'r-', lw=1)
        ax.set_title(f'{m}  slope={s["slope"]:.2f} R2={s["r2"]:.2f}\n'
                     f'resid/err={s["resid_over_err"]:.1f} '
                     f'resid/sig={s["resid_over_signal"]:.2f}', fontsize=9)
        ax.set_xlabel(f'analytic {m}' + (' (+atm)' if tag == 'an' else ''))
        ax.set_ylabel(f'HSM {m}')
        ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, location='right')
    cb.set_label('seeing FWHM (arcsec)')
    fig.suptitle('momfit analytic moments vs HSM weighted moments '
                 '(Fraunhofer (x) VonKarman)', fontsize=13)
    fig.savefig(os.path.join(OUTDIR, 'optatmo_moment_accuracy.png'),
                dpi=120, bbox_inches='tight')
    print(f'\nSaved {OUTDIR}/optatmo_moment_accuracy.png')


if __name__ == '__main__':
    run()
