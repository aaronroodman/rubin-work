"""
End-to-end validation of the fit on simulated stars: inject known DZ +
atmosphere + moment-offset (jitter) parameters, generate HSM moments with the
JAX model, add noise, then fit and check parameter recovery.

This validates the fit driver / optimiser / parameter plumbing (the physics is
already validated against galsim in validate_jax.py).
"""

import numpy as np
import jax.numpy as jnp

from config import load_config
import dz as dzmod
from config import ParamLayout
from model import Forward
import fit as fitmod

SEED = 20260709


def main():
    cfg = load_config('config.yaml')
    # smaller grid for a fast self-consistency test
    cfg['geometry']['stamp'] = 24
    cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'Kolmogorov'   # faster; VonKarman also works

    # --- machinery-correctness test: fit only well-constrained modes ---
    # (coma/trefoil have distinct 3rd-order signatures; defocus sets the size;
    #  astigmatism at focus and constant ellipticity are degenerate with the
    #  atmosphere shear / moment offsets -- deferred to the SVD test.)
    cfg['dz_terms'] = [
        {'pupil': 4, 'focal': [1]},           # defocus (size)
        {'pupil': 7, 'focal': [1, 2, 3]},     # coma + field variation
        {'pupil': 8, 'focal': [1, 2, 3]},
        {'pupil': 9, 'focal': [1]},           # trefoil
        {'pupil': 10, 'focal': [1]},
    ]
    cfg['atmosphere']['fit'] = ['fwhm']        # fix shear (degenerate w/ astig)
    cfg['moment_offsets']['moments'] = []      # no jitter term in this test
    cfg['fit']['moments'] = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03']

    rng = np.random.default_rng(SEED)
    n_stars = 50
    R = 1.6
    r = R * np.sqrt(rng.uniform(0, 1, n_stars))
    ph = rng.uniform(0, 2 * np.pi, n_stars)
    thx, thy = r * np.cos(ph), r * np.sin(ph)
    rot = rng.uniform(-np.pi, np.pi, n_stars)

    model = fitmod.build_model(cfg)
    jmax = cfg['geometry']['jmax']
    G, dz_names = dzmod.build_dz_design(cfg['dz_terms'], thx, thy,
                                        cfg['fit']['fov_radius_deg'], jmax)
    layout = ParamLayout(cfg, dz_names)

    # ---- truth parameters ----
    p_true = np.zeros(layout.n)
    p_true[layout.i_dz] = rng.normal(0, 0.10, layout.n_dz)   # DZ coeffs (um)
    # atmosphere truth
    atm_true = {'fwhm': 0.85, 'g1': 0.03, 'g2': -0.02}
    for i, a in enumerate(layout.atm_free):
        p_true[layout.n_dz + i] = atm_true[a]
    # moment-offset (jitter) truth
    off_true = {'e1': 0.0025, 'e2': -0.0015}
    for i, m in enumerate(layout.offset_moments):
        p_true[layout.i_off.start + i] = off_true.get(m, 0.0)

    z0 = np.zeros((n_stars, jmax + 1))    # zero MIW baseline for this test

    # generate noiseless truth moments, then set errors + add noise
    dummy = np.zeros((n_stars, 12))
    fwd0 = Forward(model, layout, z0, G, dummy, np.ones_like(dummy),
                   cfg['fit']['moments'], cfg['fit']['weights'])
    M_true = np.array(fwd0.moments(jnp.asarray(p_true)))

    # per-moment error scale (representative HSM errors at good S/N)
    escale = {'e0': 1.5e-3, 'e1': 2e-4, 'e2': 2e-4,
              'M21': 4e-5, 'M12': 4e-5, 'M30': 4e-5, 'M03': 4e-5,
              'M22': 1.5e-3, 'M31': 4e-4, 'M13': 4e-4, 'M40': 5e-5, 'M04': 5e-5}
    from jax_optatmo import MOMENT_LABELS
    err = np.array([[escale[m] for m in MOMENT_LABELS]] * n_stars)
    noise = rng.normal(0, err)
    data = M_true + noise

    catalog = {'thx_deg': thx, 'thy_deg': thy, 'rotator_rad': rot,
               'moments': data, 'errors': err}

    res, layout, fwd = fitmod.run_fit(cfg, catalog, miw=None)

    # ---- report recovery ----
    p_fit = res.x
    print(f'\n{"param":10} {"true":>10} {"fit":>10} {"err_scale":>10}')
    # DZ
    for i, nm in enumerate(layout.dz_names):
        print(f'{nm:10} {p_true[i]:10.4f} {p_fit[i]:10.4f}')
    for i, a in enumerate(layout.atm_free):
        j = layout.n_dz + i
        print(f'{"atm_"+a:10} {p_true[j]:10.4f} {p_fit[j]:10.4f}')
    for i, m in enumerate(layout.offset_moments):
        j = layout.i_off.start + i
        print(f'{"off_"+m:10} {p_true[j]:10.5f} {p_fit[j]:10.5f}')

    dz_rms = np.sqrt(np.mean((p_fit[layout.i_dz] - p_true[layout.i_dz]) ** 2))
    print(f'\nDZ recovery RMS: {dz_rms:.4f} um   '
          f'(injected rms 0.06, noise-limited floor ~ few e-3)')
    print(f'atm fwhm err: {p_fit[layout.n_dz]-p_true[layout.n_dz]:+.4f} arcsec')


if __name__ == '__main__':
    main()
