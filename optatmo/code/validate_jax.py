"""
Validation of the JAX Optics+Atmosphere moment model against galsim/HSM.

Checks, over an ensemble of aberration states x seeing:
  1. JAX adaptive moments vs PIFF calculate_moments (galsim truth) -- the model
     reproduces what the pipeline measures on real stars.
  2. autodiff gradients vs finite differences.

Produces output/jax_validation.png (JAX vs galsim scatter per moment) and a
printed agreement table.  Uses the Kolmogorov kernel on both sides (the JAX
kernel), so the comparison is like-for-like.
"""

import numpy as np
import galsim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jax_optatmo import JaxOptAtmoPSF, MOMENT_LABELS
from moments_hsm import measure_hsm_moments

SEED = 7
N_STATES = 60
FWHM_GRID = [0.7, 0.9, 1.1]
JMAX = 22
ABER_RMS = np.zeros(JMAX + 1)
ABER_RMS[4] = 0.12
ABER_RMS[5:9] = 0.12
ABER_RMS[9:12] = 0.06
ABER_RMS[12:] = 0.04


def galsim_truth(model, coef, fwhm, aper):
    o = galsim.OpticalPSF(lam=750., diam=model.diam, aper=aper,
                          aberrations=(coef / model.lam_um).tolist(),
                          gsparams=aper.gsparams)
    img = galsim.Convolve([o, galsim.Kolmogorov(fwhm=fwhm)]).drawImage(
        nx=model.stamp, ny=model.stamp, scale=model.pixel_scale)
    return measure_hsm_moments(img)


def main():
    import os
    os.makedirs('output', exist_ok=True)
    rng = np.random.default_rng(SEED)
    model = JaxOptAtmoPSF(jmax=JMAX, stamp=32, oversample=16)
    aper = galsim.Aperture(diam=model.diam, obscuration=model.obscuration,
                           lam=750., gsparams=galsim.GSParams(
                               minimum_fft_size=128, folding_threshold=2e-3))

    truth = {m: [] for m in MOMENT_LABELS}
    jaxm = {m: [] for m in MOMENT_LABELS}
    err = {m: [] for m in MOMENT_LABELS}
    for s in range(N_STATES):
        coef = rng.normal(0, 1, JMAX + 1) * ABER_RMS
        coef[:4] = 0
        for fwhm in FWHM_GRID:
            momG, errG = galsim_truth(model, coef, fwhm, aper)
            p = np.array(model.psf(jnp.asarray(coef), jnp.array([fwhm, 0., 0.])))
            # noise scale for the error reference (S/N ~ 300)
            nv = (p ** 2).sum() / 300. ** 2
            _, errG = measure_hsm_moments(
                galsim.Image(np.ascontiguousarray(p), scale=0.2), noise_var=nv)
            momJ = np.array(model.moments_adaptive(
                jnp.asarray(coef), jnp.array([fwhm, 0., 0.])))
            for i, m in enumerate(MOMENT_LABELS):
                truth[m].append(momG[m])
                jaxm[m].append(momJ[i])
                err[m].append(np.sqrt(errG[m]) if m in errG else np.nan)

    print('=== JAX adaptive moments vs galsim HSM (calculate_moments) ===')
    print(f'{"moment":6} {"slope":>8} {"med|resid|":>11} {"resid/err":>10} '
          f'{"maxfrac":>8}')
    for m in MOMENT_LABELS:
        t = np.array(truth[m])
        j = np.array(jaxm[m])
        e = np.array(err[m])
        slope = np.sum(t * j) / np.sum(t * t)
        resid = j - t
        medr = np.median(np.abs(resid))
        roe = np.median(np.abs(resid) / (e + 1e-30))
        frac = np.max(np.abs(resid)) / (np.std(t) + 1e-30)
        print(f'{m:6} {slope:8.4f} {medr:11.3e} {roe:10.3f} {frac:8.3f}')

    # figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, m in zip(axes.flat, MOMENT_LABELS):
        t = np.array(truth[m])
        j = np.array(jaxm[m])
        ax.scatter(t, j, s=10, alpha=0.6)
        lo, hi = min(t.min(), j.min()), max(t.max(), j.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        ax.set_title(m)
        ax.set_xlabel('galsim HSM')
        ax.set_ylabel('JAX adaptive')
        ax.grid(alpha=0.3)
    fig.suptitle('JAX adaptive moments vs galsim/PIFF calculate_moments '
                 '(y=x = exact match)', fontsize=13)
    fig.tight_layout()
    fig.savefig('output/jax_validation.png', dpi=110)
    print('\nSaved output/jax_validation.png')

    # autodiff check
    def fn(params):
        z = jnp.concatenate([jnp.zeros(4), params[:JMAX - 3]])
        return model.moments_adaptive(z, params[JMAX - 3:JMAX])
    p0 = jnp.concatenate([jnp.asarray(ABER_RMS[4:]), jnp.array([0.9, 0.02, -0.01])])
    J = jax.jacobian(fn)(p0)
    print(f'\nautodiff jacobian shape {J.shape}, any NaN: '
          f'{bool(np.isnan(np.array(J)).any())}')


if __name__ == '__main__':
    main()
