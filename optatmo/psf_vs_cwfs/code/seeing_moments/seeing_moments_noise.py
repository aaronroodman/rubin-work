"""
Effect of shot noise on weighted second moments vs seeing FWHM.

PSF = Kolmogorov(seeing) (convolved with) optical aberrations (Z4-Z11),
scaled to the flux of a 17th-mag star (40% throughput), with a 250 e-/pixel
sky that is subtracted but whose Poisson shot noise remains.  For each FWHM
the clean image is drawn once, then many noise realizations are added and HSM
adaptive moments measured to get the mean and scatter of e1*T, e2*T, and T.
"""

import numpy as np
import matplotlib.pyplot as plt
import galsim

# ---- optical / detector configuration ----
WAVELENGTH = 750.0       # nm
DIAM = 8.36              # m
OBSCURATION = 0.61
PIXEL_SCALE = 0.2        # arcsec / pixel
STAMP = 128              # pixels
SEED = 314159

# ---- photometry: 17th mag AB star, Rubin-like ----
MAG = 17.0
EFF_AREA_M2 = 35.0       # effective collecting area
EXPTIME = 30.0           # s
DLAM_NM = 150.0          # effective bandwidth
THROUGHPUT = 0.40
SKY_E = 250.0            # sky electrons per pixel (subtracted, shot noise kept)

N_NOISE = 300            # noise realizations per FWHM
FWHM_VALUES = np.arange(0.6, 1.2001, 0.05)
lam_um = WAVELENGTH * 1e-3


def ab_photons(mag):
    h, c = 6.626e-34, 3.0e8
    lam0 = WAVELENGTH * 1e-9
    f_nu = 10 ** (-0.4 * (mag + 48.6)) * 1e-7 * 1e4   # J/m^2/s/Hz
    dnu = c * (DLAM_NM * 1e-9) / lam0 ** 2
    return f_nu / (h * c / lam0) * dnu * EFF_AREA_M2 * EXPTIME * THROUGHPUT


NPHOT = ab_photons(MAG)
print(f"17th-mag star total flux = {NPHOT:.3e} e-  (sky {SKY_E:.0f} e-/pix)\n")

# ---- fixed aberration realization (same as realization 0 of the prior study) ----
rng_ab = np.random.default_rng(SEED)
ab_microns = np.zeros(12)
ab_microns[4] = 0.3
ab_microns[5:12] = rng_ab.normal(0.0, 0.3, size=7)
optic = galsim.OpticalPSF(lam=WAVELENGTH, diam=DIAM, obscuration=OBSCURATION,
                          aberrations=(ab_microns / lam_um).tolist())

ud = galsim.BaseDeviate(SEED + 1)


def moments(img):
    hsm = img.FindAdaptiveMom()
    e1, e2 = hsm.observed_shape.e1, hsm.observed_shape.e2
    e = np.hypot(e1, e2)
    T = 2.0 * (hsm.moments_sigma * PIXEL_SCALE) ** 2 / np.sqrt(1.0 - e ** 2)
    return e1 * T, e2 * T, T


nf = len(FWHM_VALUES)
clean = np.zeros((nf, 3))                 # e1T, e2T, T  (no noise)
mean  = np.zeros((nf, 3))
std   = np.zeros((nf, 3))

for i, fwhm in enumerate(FWHM_VALUES):
    psf = galsim.Convolve([galsim.Kolmogorov(fwhm=fwhm), optic]).withFlux(NPHOT)
    base = psf.drawImage(nx=STAMP, ny=STAMP, scale=PIXEL_SCALE, method='auto')
    clean[i] = moments(base)

    samples = []
    noise = galsim.PoissonNoise(ud, sky_level=SKY_E)   # adds sky, Poisson, subtracts sky
    for _ in range(N_NOISE):
        noisy = base.copy()
        noisy.addNoise(noise)
        try:
            samples.append(moments(noisy))
        except galsim.GalSimError:
            pass
    samples = np.array(samples)
    mean[i] = samples.mean(axis=0)
    std[i]  = samples.std(axis=0)
    print(f"FWHM {fwhm:.2f}: "
          f"e1T {clean[i,0]:+.4f} +/- {std[i,0]:.4f}   "
          f"e2T {clean[i,1]:+.4f} +/- {std[i,1]:.4f}   "
          f"({len(samples)}/{N_NOISE} ok)")

# ---- plots: clean curve + noise scatter band ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
labels = [r'$e_1 (I_{xx}+I_{yy})$', r'$e_2 (I_{xx}+I_{yy})$', r'$I_{xx}+I_{yy}$']
for k, ax in enumerate(axes):
    ax.plot(FWHM_VALUES, clean[:, k], 'k-', lw=1.5, label='noiseless')
    ax.errorbar(FWHM_VALUES, mean[:, k], yerr=std[:, k], fmt='o', ms=4,
                color='C3', capsize=3, label=f'mean $\\pm$ scatter ({N_NOISE} noise)')
    ax.fill_between(FWHM_VALUES, mean[:, k] - std[:, k], mean[:, k] + std[:, k],
                    color='C3', alpha=0.15)
    ax.set_xlabel('Kolmogorov FWHM (arcsec)')
    ax.set_ylabel(labels[k] + r'  (arcsec$^2$)')
    ax.set_title(labels[k])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(f'Shot-noise scatter, 17th-mag star ({NPHOT:.1e} e-), '
             f'sky {SKY_E:.0f} e-/pix', y=1.02)
fig.tight_layout()
fig.savefig('seeing_moments_noise.png', dpi=130, bbox_inches='tight')
print("\nSaved seeing_moments_noise.png")
print(f"\nShot-noise scatter on e1T: {std[:,0].min():.4f}-{std[:,0].max():.4f} arcsec^2")
print(f"Shot-noise scatter on e2T: {std[:,1].min():.4f}-{std[:,1].max():.4f} arcsec^2")
print(f"(scatter grows with FWHM as the PSF spreads over more sky-noise pixels)")