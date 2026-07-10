"""
OptAtmo moment-fit driver.

Assembles the model from a YAML config + a star catalog (field positions,
rotator, HSM moments + errors), fits DZ + atmosphere + moment-offset params by
minimising the moment chi^2 with jax gradients, and returns the fitted params.

A star catalog is a dict of arrays:
    thx_deg, thy_deg, rotator_rad : (n_stars,)
    moments : (n_stars, 12)   in MOMENT_LABELS order
    errors  : (n_stars, 12)
"""

import numpy as np
import jax
from scipy.optimize import minimize

from jax_optatmo import JaxOptAtmoPSF, MOMENT_LABELS
from config import ParamLayout
import dz as dzmod
from model import Forward


def build_model(cfg):
    g = cfg['geometry']
    return JaxOptAtmoPSF(jmax=g['jmax'], diam=g['diam'],
                         obscuration=g['obscuration'], lam_nm=g['lam_nm'],
                         pixel_scale=g['pixel_scale'], stamp=g['stamp'],
                         oversample=g['oversample'], annular=g.get('annular', False),
                         kernel=cfg['atmosphere']['kernel'],
                         L0=cfg['atmosphere']['L0'])


def assemble(cfg, catalog, miw=None):
    """Build (model, layout, Forward) from config + star catalog."""
    model = build_model(cfg)
    jmax = cfg['geometry']['jmax']
    thx, thy = catalog['thx_deg'], catalog['thy_deg']

    G, dz_names = dzmod.build_dz_design(
        cfg['dz_terms'], thx, thy, cfg['fit']['fov_radius_deg'], jmax)
    layout = ParamLayout(cfg, dz_names)

    # MIW baseline (optional; zeros if not provided). MIWCalib.zernikes needs
    # the per-cell detector; pass it when the catalog carries it.
    if miw is not None:
        z0 = np.nan_to_num(miw.zernikes(thx, thy, catalog['rotator_rad'], jmax,
                                        catalog['detector']))
    else:
        z0 = np.zeros((len(thx), jmax + 1))

    fwd = Forward(model, layout, z0, G, catalog['moments'], catalog['errors'],
                  cfg['fit']['moments'], cfg['fit']['weights'])
    return model, layout, fwd


def svd_analysis(fwd, layout, p0, verbose=True):
    """SVD of the observable Jacobian at p0 to expose degeneracies.

    Returns (U, S, Vt) of J = d(residuals)/dp.  Small singular values are
    near-degenerate parameter combinations (e.g. constant defocus vs
    atmospheric size; constant astigmatism vs shear vs moment offsets)."""
    # forward-mode: n_par (few) << n_residuals (many) -> jacfwd is far cheaper
    jac = jax.jacfwd(lambda p: fwd.residuals(p).reshape(-1))
    J = np.asarray(jac(jnp.asarray(p0)))
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    if verbose:
        print('SVD singular values (large=constrained, ~0=degenerate):')
        for i, s in enumerate(S):
            tag = '  <-- degenerate' if s < 1e-3 * S[0] else ''
            top = np.argsort(np.abs(Vt[i]))[::-1][:3]
            combo = ' + '.join(f'{Vt[i,j]:+.2f}*{layout.names[j]}' for j in top)
            print(f'  s[{i:2d}]={s:10.3e}  ~ {combo}{tag}')
    return U, S, Vt


def run_fit(cfg, catalog, miw=None, p0=None, use_svd=False, svd_rcond=1e-3,
            verbose=True):
    """Fit the moment model.  If use_svd, optimise only in the identifiable
    subspace (singular values > svd_rcond * max), holding degenerate
    combinations at their p0 value (the paper's SVD degeneracy control)."""
    model, layout, fwd = assemble(cfg, catalog, miw)
    if p0 is None:
        p0 = layout.initial()
    p0 = np.asarray(p0, float)

    if use_svd:
        _, S, Vt = svd_analysis(fwd, layout, p0, verbose=verbose)
        keep = S > svd_rcond * S[0]
        Vk = jnp.asarray(Vt[keep].T)          # (n_par, n_keep)
        p0j = jnp.asarray(p0)

        def cost_q(q):
            return fwd.cost(p0j + Vk @ q)
        vg = jax.jit(jax.value_and_grad(cost_q))

        def scipy_fun(q):
            v, g = vg(jnp.asarray(q))
            return float(v), np.asarray(g, float)

        opt = cfg.get('optimizer', {})
        q0 = np.zeros(int(keep.sum()))
        rq = minimize(scipy_fun, q0, jac=True, method='L-BFGS-B',
                      options={'maxiter': opt.get('maxiter', 300)})
        p_fit = np.asarray(p0j + Vk @ jnp.asarray(rq.x))
        rq.x = p_fit
        if verbose:
            print(f'fit(SVD, {int(keep.sum())}/{layout.n} modes): '
                  f'success={rq.success} nit={rq.nit} cost={rq.fun:.4e}')
        return rq, layout, fwd

    vg = jax.jit(jax.value_and_grad(fwd.cost))

    def scipy_fun(p):
        v, g = vg(jnp.asarray(p))
        return float(v), np.asarray(g, dtype=float)

    opt = cfg.get('optimizer', {})
    res = minimize(scipy_fun, p0, jac=True, method=opt.get('method', 'L-BFGS-B'),
                   bounds=layout.bounds(),
                   options={'maxiter': opt.get('maxiter', 300),
                            'gtol': opt.get('gtol', 1e-8)})
    if verbose:
        print(f'fit: success={res.success} nit={res.nit} cost={res.fun:.4e}')
    return res, layout, fwd


import jax.numpy as jnp  # noqa: E402  (used in run_fit closure)
