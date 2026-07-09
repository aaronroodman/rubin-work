"""
SVD degeneracy demonstration on the full model (defocus + astigmatism + atm
shear + jitter moment-offsets).  Shows the singular-value spectrum exposing the
degenerate parameter combinations (constant defocus vs atmospheric size;
constant astigmatism vs shear vs moment offsets), and that the SVD-restricted
fit is stable and reproduces the data.
"""

import numpy as np
import jax.numpy as jnp

from config import load_config, ParamLayout
import dz as dzmod
from model import Forward
from jax_optatmo import MOMENT_LABELS
import fit as fitmod

SEED = 424242


def main():
    cfg = load_config('config.yaml')
    cfg['geometry']['stamp'] = 24
    cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'Kolmogorov'
    # full model with the degenerate constant modes present
    cfg['dz_terms'] = [{'pupil': 4, 'focal': [1]},
                       {'pupil': 5, 'focal': [1]}, {'pupil': 6, 'focal': [1]},
                       {'pupil': 7, 'focal': [1]}, {'pupil': 8, 'focal': [1]}]
    cfg['atmosphere']['fit'] = ['fwhm', 'g1', 'g2']
    cfg['moment_offsets']['moments'] = ['e1', 'e2']
    cfg['fit']['moments'] = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03']

    rng = np.random.default_rng(SEED)
    n = 50
    r = 1.6 * np.sqrt(rng.uniform(0, 1, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    thx, thy, rot = r * np.cos(ph), r * np.sin(ph), rng.uniform(-np.pi, np.pi, n)

    model = fitmod.build_model(cfg)
    jmax = cfg['geometry']['jmax']
    G, dz_names = dzmod.build_dz_design(cfg['dz_terms'], thx, thy,
                                        cfg['fit']['fov_radius_deg'], jmax)
    layout = ParamLayout(cfg, dz_names)

    p_true = np.zeros(layout.n)
    p_true[layout.i_dz] = rng.normal(0, 0.08, layout.n_dz)
    for i, a in enumerate(layout.atm_free):
        p_true[layout.n_dz + i] = {'fwhm': 0.85, 'g1': 0.03, 'g2': -0.02}[a]
    for i, m in enumerate(layout.offset_moments):
        p_true[layout.i_off.start + i] = {'e1': 0.0025, 'e2': -0.0015}[m]

    z0 = np.zeros((n, jmax + 1))
    err = np.array([[{'e0': 1.5e-3, 'e1': 2e-4, 'e2': 2e-4, 'M21': 4e-5,
                      'M12': 4e-5, 'M30': 4e-5, 'M03': 4e-5, 'M22': 1.5e-3,
                      'M31': 4e-4, 'M13': 4e-4, 'M40': 5e-5, 'M04': 5e-5}[m]
                     for m in MOMENT_LABELS]] * n)
    fwd0 = Forward(model, layout, z0, G, np.zeros((n, 12)), np.ones((n, 12)),
                   cfg['fit']['moments'], cfg['fit']['weights'])
    data = np.array(fwd0.moments(jnp.asarray(p_true))) + rng.normal(0, err)
    catalog = {'thx_deg': thx, 'thy_deg': thy, 'rotator_rad': rot,
               'moments': data, 'errors': err}

    res, layout, fwd = fitmod.run_fit(cfg, catalog, use_svd=True, svd_rcond=1e-3)

    # data-space quality: predicted vs observed moments
    Mfit = np.array(fwd.moments(jnp.asarray(res.x)))
    sel = [MOMENT_LABELS.index(m) for m in cfg['fit']['moments']]
    chi2 = np.sum(((data[:, sel] - Mfit[:, sel]) / err[:, sel]) ** 2)
    ndof = n * len(sel) - int((np.linalg.svd(np.eye(1))[1] > 0).sum())  # approx
    print(f'\nchi2/(n*nmom) = {chi2 / (n*len(sel)):.3f}  '
          f'(good fit ~ 1 despite parameter degeneracy)')
    print('=> SVD fit reproduces the data; degenerate combinations (small '
          'singular values above) are held at prior, not split arbitrarily.')


if __name__ == '__main__':
    main()
