"""
Weighted second moments vs seeing FWHM, now varying Z4 across realizations.

Each realization draws Z4..Z11 randomly (rms 0.3 um, mean 0).  For each one
the Kolmogorov FWHM is varied and HSM adaptive moments measured.  Curves of
e1*T and e2*T (T = Ixx+Iyy) vs FWHM are colored by their Z4 value.
(Shot noise is negligible at 17th mag, so it is omitted here.)
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
N_REAL = 16
RMS_MICRONS = 0.3        # rms for ALL of Z4..Z11 now
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
    return e1 * T, e2 * T, T


nf = len(FWHM_VALUES)
e1T = np.zeros((N_REAL, nf))
e2T = np.zeros((N_REAL, nf))
Tr  = np.zeros((N_REAL, nf))
z4  = np.zeros(N_REAL)
ab_all = np.zeros((N_REAL, 12))

for r in range(N_REAL):
    ab_microns = np.zeros(12)
    ab_microns[4:12] = rng.normal(0.0, RMS_MICRONS, size=8)   # Z4..Z11 all random
    z4[r] = ab_microns[4]
    ab_all[r] = ab_microns
    optic = galsim.OpticalPSF(lam=WAVELENGTH, diam=DIAM, obscuration=OBSCURATION,
                              aberrations=(ab_microns / lam_um).tolist())
    for i, fwhm in enumerate(FWHM_VALUES):
        e1T[r, i], e2T[r, i], Tr[r, i] = measure(optic, fwhm)

print("Aberrations (microns), Z4..Z11 per realization:")
print("  real " + " ".join(f"Z{j:>2d}" for j in range(4, 12)))
for r in range(N_REAL):
    print(f"  {r:3d}  " + " ".join(f"{ab_all[r, j]:+5.2f}" for j in range(4, 12)))

# ---- plots, colored by Z4 ----
vmax = np.max(np.abs(z4))
norm = plt.Normalize(-vmax, vmax)
cmap = plt.cm.coolwarm

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for r in range(N_REAL):
    col = cmap(norm(z4[r]))
    axes[0].plot(FWHM_VALUES, e1T[r], '-', color=col, alpha=0.85)
    axes[1].plot(FWHM_VALUES, e2T[r], '-', color=col, alpha=0.85)
    axes[2].plot(FWHM_VALUES, Tr[r],  '-', color=col, alpha=0.85)

labels = [r'$e_1 (I_{xx}+I_{yy})$', r'$e_2 (I_{xx}+I_{yy})$', r'$I_{xx}+I_{yy}$']
for ax, lab in zip(axes, labels):
    ax.set_xlabel('Kolmogorov FWHM (arcsec)')
    ax.set_ylabel(lab + r'  (arcsec$^2$)')
    ax.set_title(lab + f'  ({N_REAL} realizations)')
    ax.grid(alpha=0.3)
axes[0].axhline(0, color='gray', lw=0.8)
axes[1].axhline(0, color='gray', lw=0.8)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.01)
cb.set_label(r'Z4 (microns)')
fig.suptitle('Random Z4..Z11 (rms 0.3 um); curves colored by Z4', y=1.02)
fig.savefig('seeing_moments_varyz4.png', dpi=130, bbox_inches='tight')
print("\nSaved seeing_moments_varyz4.png")

# ---- does T's seeing-zero-point depend on Z4? (defocus inflates size) ----
T_at_min = Tr[:, 0]
print(f"\nCorrelation of Z4 with T at FWHM 0.6 arcsec: "
      f"{np.corrcoef(np.abs(z4), T_at_min)[0,1]:+.2f} (|Z4| vs size)")