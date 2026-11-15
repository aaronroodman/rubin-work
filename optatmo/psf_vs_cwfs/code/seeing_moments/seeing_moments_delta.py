"""
Seeing-induced drift of the spin-2 moments e1*T, e2*T (T = Ixx+Iyy) across
the FWHM range, over many random aberration realizations.

For each realization (Z4..Z11 random, rms 0.3 um) the FWHM is swept 0.6->1.2
arcsec and e1*T, e2*T measured.  delta is the signed change across the range
(value at max FWHM minus value at min FWHM).  Since the curves are monotonic
in FWHM this equals the min-to-max swing.  A scatter of delta(e1*T) vs
delta(e2*T) shows how large the seeing-dependence can be -- a systematic for
deriving aberrations from stellar moments.
"""

import numpy as np
import matplotlib.pyplot as plt
import galsim

WAVELENGTH = 750.0
DIAM = 8.36
OBSCURATION = 0.61
PIXEL_SCALE = 0.2
STAMP = 128
SEED = 314159
N_REAL = 1000
RMS_MICRONS = 0.3
FWHM_VALUES = np.arange(0.6, 1.2001, 0.05)
lam_um = WAVELENGTH * 1e-3

rng = np.random.default_rng(SEED)


def measure(optic, fwhm):
    psf = galsim.Convolve([galsim.Kolmogorov(fwhm=fwhm), optic])
    img = psf.drawImage(nx=STAMP, ny=STAMP, scale=PIXEL_SCALE, method='auto')
    hsm = img.FindAdaptiveMom()
    e1, e2 = hsm.observed_shape.e1, hsm.observed_shape.e2
    e = np.hypot(e1, e2)
    T = 2.0 * (hsm.moments_sigma * PIXEL_SCALE) ** 2 / np.sqrt(1.0 - e ** 2)
    return e1 * T, e2 * T


nf = len(FWHM_VALUES)
d_e1T = np.zeros(N_REAL)   # signed: value(max FWHM) - value(min FWHM)
d_e2T = np.zeros(N_REAL)
pp_e1T = np.zeros(N_REAL)  # peak-to-peak (max - min)
pp_e2T = np.zeros(N_REAL)
ab_rms = np.zeros(N_REAL)  # rms of the non-symmetric (spin-2 driving) terms

for r in range(N_REAL):
    ab_microns = np.zeros(12)
    ab_microns[4:12] = rng.normal(0.0, RMS_MICRONS, size=8)
    optic = galsim.OpticalPSF(lam=WAVELENGTH, diam=DIAM, obscuration=OBSCURATION,
                              aberrations=(ab_microns / lam_um).tolist())
    e1T = np.empty(nf)
    e2T = np.empty(nf)
    for i, fwhm in enumerate(FWHM_VALUES):
        e1T[i], e2T[i] = measure(optic, fwhm)
    d_e1T[r] = e1T[-1] - e1T[0]
    d_e2T[r] = e2T[-1] - e2T[0]
    pp_e1T[r] = e1T.max() - e1T.min()
    pp_e2T[r] = e2T.max() - e2T.min()
    # non-symmetric Zernikes (exclude Z4 defocus & Z11 spherical, both spin-0)
    ab_rms[r] = np.sqrt(np.mean(ab_microns[[5, 6, 7, 8, 9, 10]] ** 2))
    if (r + 1) % 100 == 0:
        print(f"  {r+1}/{N_REAL} done")

dmag = np.hypot(d_e1T, d_e2T)
print(f"\n|delta(e1T,e2T)| over FWHM range (arcsec^2):")
print(f"  median {np.median(dmag):.4f}, 90th pct {np.percentile(dmag,90):.4f}, "
      f"max {dmag.max():.4f}")
print(f"delta(e1T): std {d_e1T.std():.4f}, range [{d_e1T.min():+.4f}, {d_e1T.max():+.4f}]")
print(f"delta(e2T): std {d_e2T.std():.4f}, range [{d_e2T.min():+.4f}, {d_e2T.max():+.4f}]")

# ---- plots ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(d_e1T, d_e2T, c=dmag, s=10, cmap='viridis', alpha=0.7)
axes[0].axhline(0, color='gray', lw=0.8)
axes[0].axvline(0, color='gray', lw=0.8)
axes[0].set_xlabel(r'$\Delta(e_1 T)$  [$e_1T$(1.2'') $-$ $e_1T$(0.6'')]  (arcsec$^2$)')
axes[0].set_ylabel(r'$\Delta(e_2 T)$  (arcsec$^2$)')
axes[0].set_title(f'Seeing-induced drift of spin-2 moment\n{N_REAL} aberration realizations')
axes[0].set_aspect('equal', 'box')
axes[0].grid(alpha=0.3)
cb = fig.colorbar(sc, ax=axes[0], shrink=0.8)
cb.set_label(r'$|\Delta|$  (arcsec$^2$)')

axes[1].hist(dmag, bins=50, color='C0', alpha=0.8)
axes[1].axvline(np.median(dmag), color='k', ls='--',
                label=f'median {np.median(dmag):.4f}')
axes[1].axvline(np.percentile(dmag, 90), color='C3', ls='--',
                label=f'90th pct {np.percentile(dmag,90):.4f}')
axes[1].set_xlabel(r'$|\Delta(e_1T, e_2T)|$  (arcsec$^2$)')
axes[1].set_ylabel('realizations')
axes[1].set_title('Magnitude of seeing-induced drift')
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig('seeing_moments_delta.png', dpi=130)
print("\nSaved seeing_moments_delta.png")

np.savez('seeing_moments_delta.npz', d_e1T=d_e1T, d_e2T=d_e2T,
         pp_e1T=pp_e1T, pp_e2T=pp_e2T, ab_rms=ab_rms)