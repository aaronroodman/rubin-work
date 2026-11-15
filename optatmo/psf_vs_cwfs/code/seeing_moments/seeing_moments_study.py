"""
Study of weighted second moments vs seeing FWHM for a PSF modeled as
Kolmogorov(seeing) (convolved with) optical aberrations (Zernike Z4-Z11).

For each of several random aberration realizations (Z4 fixed at 0.3 um,
Z5-Z11 random with rms 0.3 um), the Kolmogorov FWHM is varied and HSM
adaptive moments are measured.  e1*(Ixx+Iyy) and e2*(Ixx+Iyy) are reported
vs FWHM, one curve per realization.
"""

import numpy as np
import matplotlib.pyplot as plt
import galsim

# ---- configuration ----
WAVELENGTH = 750.0       # nm
DIAM = 8.36              # m (Rubin primary)
OBSCURATION = 0.61       # Rubin pupil obscuration fraction
PIXEL_SCALE = 0.2        # arcsec / pixel
STAMP = 128              # pixels
SEED = 314159            # master seed
N_REAL = 12              # number of random aberration realizations
Z4_MICRONS = 0.3         # fixed defocus
RMS_MICRONS = 0.3        # rms of Z5..Z11

FWHM_VALUES = np.arange(0.6, 1.2001, 0.05)  # arcsec

lam_um = WAVELENGTH * 1e-3
rng = np.random.default_rng(SEED)


def measure(optic, fwhm):
    """Return (e1, e2, T) for Kolmogorov(fwhm) convolved with optic."""
    psf = galsim.Convolve([galsim.Kolmogorov(fwhm=fwhm), optic])
    img = psf.drawImage(nx=STAMP, ny=STAMP, scale=PIXEL_SCALE, method='auto')
    hsm = img.FindAdaptiveMom()
    e1, e2 = hsm.observed_shape.e1, hsm.observed_shape.e2
    e = np.hypot(e1, e2)
    # Ixx+Iyy in arcsec^2:  T = 2 sigma^2 / sqrt(1 - e^2)
    T = 2.0 * (hsm.moments_sigma * PIXEL_SCALE) ** 2 / np.sqrt(1.0 - e ** 2)
    return e1, e2, T


# ---- build realizations and measure ----
nf = len(FWHM_VALUES)
e1T = np.zeros((N_REAL, nf))
e2T = np.zeros((N_REAL, nf))
Tr  = np.zeros((N_REAL, nf))
ab_all = np.zeros((N_REAL, 12))

for r in range(N_REAL):
    ab_microns = np.zeros(12)            # Noll-indexed, use 4..11
    ab_microns[4] = Z4_MICRONS
    ab_microns[5:12] = rng.normal(0.0, RMS_MICRONS, size=7)
    ab_all[r] = ab_microns
    optic = galsim.OpticalPSF(lam=WAVELENGTH, diam=DIAM, obscuration=OBSCURATION,
                              aberrations=(ab_microns / lam_um).tolist())
    for i, fwhm in enumerate(FWHM_VALUES):
        e1T[r, i], e2T[r, i], Tr[r, i] = (lambda e1, e2, T: (e1 * T, e2 * T, T))(
            *measure(optic, fwhm))

print("Aberrations (microns), Z4..Z11 per realization:")
print("  real " + " ".join(f"Z{j:>2d}" for j in range(4, 12)))
for r in range(N_REAL):
    print(f"  {r:3d}  " + " ".join(f"{ab_all[r, j]:+5.2f}" for j in range(4, 12)))

# ---- plots ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
cmap = plt.cm.viridis(np.linspace(0, 0.95, N_REAL))

for r in range(N_REAL):
    axes[0].plot(FWHM_VALUES, e1T[r], '-', color=cmap[r], alpha=0.8)
    axes[1].plot(FWHM_VALUES, e2T[r], '-', color=cmap[r], alpha=0.8)
    axes[2].plot(FWHM_VALUES, Tr[r],  '-', color=cmap[r], alpha=0.8)

for ax, ttl, ylab in zip(
        axes,
        [r'$e_1 (I_{xx}+I_{yy})$', r'$e_2 (I_{xx}+I_{yy})$', r'$I_{xx}+I_{yy}$'],
        [r'$e_1 (I_{xx}+I_{yy})$  (arcsec$^2$)',
         r'$e_2 (I_{xx}+I_{yy})$  (arcsec$^2$)',
         r'$I_{xx}+I_{yy}$  (arcsec$^2$)']):
    ax.set_xlabel('Kolmogorov FWHM (arcsec)')
    ax.set_ylabel(ylab)
    ax.set_title(ttl + f'  ({N_REAL} realizations)')
    ax.grid(alpha=0.3)
axes[0].axhline(0, color='gray', lw=0.8)
axes[1].axhline(0, color='gray', lw=0.8)

fig.tight_layout()
fig.savefig('seeing_moments_study.png', dpi=130)
print("\nSaved seeing_moments_study.png")

# ---- summary: absolute drift of the spin-2 moment across the FWHM range ----
# Percentage drift is misleading because baselines cross zero; report the
# absolute change (arcsec^2) and compare to the spread across realizations.
d_e1T = e1T[:, -1] - e1T[:, 0]
d_e2T = e2T[:, -1] - e2T[:, 0]
spread_e1 = np.std(np.concatenate([e1T[:, 0], e1T[:, -1]]))
spread_e2 = np.std(np.concatenate([e2T[:, 0], e2T[:, -1]]))

print("\nAbsolute change of e_i*T from FWHM 0.6 -> 1.2 arcsec (arcsec^2):")
print(f"  e1*T:  mean |drift| {np.mean(np.abs(d_e1T)):.4f}, "
      f"max {np.max(np.abs(d_e1T)):.4f}   (spread across realizations {spread_e1:.4f})")
print(f"  e2*T:  mean |drift| {np.mean(np.abs(d_e2T)):.4f}, "
      f"max {np.max(np.abs(d_e2T)):.4f}   (spread across realizations {spread_e2:.4f})")
print(f"  T   :  grows {np.mean(Tr[:, 0]):.3f} -> {np.mean(Tr[:, -1]):.3f} "
      f"(x{np.mean(Tr[:, -1] / Tr[:, 0]):.2f}) with seeing")
