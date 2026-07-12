"""
Per-Zernike-mode decomposition: geometric vs diffractive moment content.

For each single Noll mode Z_j at fixed amplitude, compare the exact geometric
moments (analytic phase-gradient integral, parity-corrected) with the HSM
moments of the true optics-only Fraunhofer PSF.  The ratio geom/Fraunhofer per
moment order shows which moments a geometric (momfit-analytic) model can
reproduce and which are intrinsically diffractive -- independent of the
atmosphere or the weighting choice.

Output: output/optatmo_geom_vs_diffraction.png + printed table.
"""

import numpy as np
import galsim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moments_analytic import AnnularPupil, analytic_moments
from moments_hsm import measure_hsm_moments
import psf_model as pm

AMP = 0.20            # micron per mode
MODES = [j for j in range(4, 27) if j not in (20, 21)]   # AOS Z4..Z26 omit Z20,Z21
FINE_SCALE = 0.01
FINE_STAMP = 256


def fraunhofer_optics_moments(coef, aper):
    optical = galsim.OpticalPSF(lam=pm.LAM, diam=pm.DIAM, aper=aper,
                                aberrations=(coef / (pm.LAM * 1e-3)).tolist(),
                                gsparams=pm._GSP)
    img = optical.drawImage(nx=FINE_STAMP, ny=FINE_STAMP, scale=FINE_SCALE,
                            method='auto')
    m, _ = measure_hsm_moments(img)
    return m


def main():
    import os
    os.makedirs('output', exist_ok=True)
    pup = AnnularPupil(ngrid=512)
    aper = galsim.Aperture(diam=pm.DIAM, obscuration=pm.OBSCURATION,
                           lam=pm.LAM, gsparams=pm._GSP)

    order2 = ['e1', 'e2']
    order3 = ['M21', 'M12', 'M30', 'M03']
    order4 = ['M31', 'M13', 'M40', 'M04']
    # collect signed (Fraunhofer, geometric) pairs for every excited component
    pts = {2: [], 3: [], 4: []}
    print(f'{"mode":>5} | dominant 2nd (Fr, geom)   | dominant 3rd (Fr, geom)')
    for j in MODES:
        coef = np.zeros(23)
        coef[j] = AMP
        g = analytic_moments(coef, pup)
        f = fraunhofer_optics_moments(coef, aper)
        for order, keys in [(2, order2), (3, order3), (4, order4)]:
            for k in keys:
                if abs(f[k]) > 1e-6 or abs(g[k]) > 1e-6:
                    pts[order].append((f[k], g[k]))
        d2 = max(order2, key=lambda k: abs(f[k]))
        d3 = max(order3, key=lambda k: abs(f[k]))
        print(f'{j:>5} | {d2}: {f[d2]:+.2e},{g[d2]:+.2e} | '
              f'{d3}: {f[d3]:+.2e},{g[d3]:+.2e}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, order, c in zip(axes, [2, 3, 4], ['C0', 'C1', 'C3']):
        a = np.array(pts[order])
        ax.scatter(a[:, 0], a[:, 1], s=30, color=c, alpha=0.7)
        lim = np.abs(a).max() * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='geom = Fraunhofer')
        ax.axhline(0, color='gray', lw=0.6)
        ax.axvline(0, color='gray', lw=0.6)
        # slope of best fit through origin
        slope = np.sum(a[:, 0] * a[:, 1]) / np.sum(a[:, 0] ** 2)
        ax.plot([-lim, lim], [-slope * lim, slope * lim], color=c, lw=1,
                label=f'fit slope={slope:.2f}')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f'{order}-order moments')
        ax.set_xlabel('Fraunhofer HSM (truth)')
        ax.set_ylabel('geometric analytic')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', 'box')
    fig.suptitle('Geometric (momfit-analytic) vs Fraunhofer moments, optics-only, '
                 'per Zernike mode Z4-Z22 (0.2 µm)\n'
                 'points on the dashed line = geometric captures it; '
                 'scattered/off-axis = diffractive', fontsize=12)
    fig.tight_layout()
    fig.savefig('output/optatmo_geom_vs_diffraction.png', dpi=120)
    print('\nSaved output/optatmo_geom_vs_diffraction.png')


if __name__ == '__main__':
    main()
