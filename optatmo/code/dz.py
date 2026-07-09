"""
Double-Zernike (DZ) field model for the optical wavefront free parameters.

The wavefront Zernike coefficient of pupil mode k varies across the field of
view; that variation is expanded in focal-plane Zernikes F_f(field):

    delta z_k(field) = sum_f  a_{k,f} F_f(thx/R_fov, thy/R_fov)

The free parameters are the {a_{k,f}}.  Because this is LINEAR in the
parameters, we precompute a design tensor G of shape (n_stars, jmax+1, n_par)
so that the per-star pupil-Zernike contribution is simply

    dz[i] = G[i] @ params            (jnp, differentiable, cheap)

This is the generalisation of the old PIFF optatmo `ofit_double_zernike_terms`
(pupil index, number of focal terms); here each pupil mode gets an explicit
list of focal Noll indices.
"""

import numpy as np
import galsim


def parse_dz_terms(dz_terms):
    """Normalise the config DZ term spec into (pupil_k, [focal_noll...]).

    Accepts either {'pupil': k, 'focal': [f,...]} or {'pupil': k, 'nfocal': n}
    (first n Noll focal indices 1..n), or the old [k, n] pair form.
    """
    out = []
    for t in dz_terms:
        if isinstance(t, (list, tuple)):
            k, n = int(t[0]), int(t[1])
            out.append((k, list(range(1, n + 1))))
        elif 'focal' in t:
            out.append((int(t['pupil']), [int(f) for f in t['focal']]))
        else:
            n = int(t['nfocal'])
            out.append((int(t['pupil']), list(range(1, n + 1))))
    return out


def build_dz_design(dz_terms, thx_deg, thy_deg, fov_radius_deg, jmax):
    """Build the DZ design tensor and parameter names.

    :param dz_terms:        list parsed by parse_dz_terms.
    :param thx_deg, thy_deg: star field positions (degrees), length n_stars.
    :param fov_radius_deg:  normalisation radius for the focal Zernikes.
    :param jmax:            max pupil Noll index in the wavefront vector.
    :returns: (G, param_names)
        G           : ndarray (n_stars, jmax+1, n_par)
        param_names : list of 'z{k}f{f}' strings, length n_par
    """
    terms = parse_dz_terms(dz_terms)
    thx = np.asarray(thx_deg, float) / fov_radius_deg
    thy = np.asarray(thy_deg, float) / fov_radius_deg
    n_stars = thx.size

    max_focal = max((max(fl) for _, fl in terms), default=1)
    fbasis = galsim.zernike.zernikeBasis(max_focal, thx, thy, R_outer=1.0)
    # fbasis[f, i] = F_f at star i  (index 0 unused)

    param_names = []
    cols = []          # (pupil_k, focal_f) per parameter
    for k, focal_list in terms:
        for f in focal_list:
            param_names.append(f'z{k}f{f}')
            cols.append((k, f))
    n_par = len(cols)

    G = np.zeros((n_stars, jmax + 1, n_par))
    for p, (k, f) in enumerate(cols):
        G[:, k, p] = fbasis[f]
    return G, param_names
