"""
v-Mode fit: parametrize the optical wavefront by AOS sensitivity-matrix
u-mode/v-mode amplitudes (from ofc_svd) instead of raw Double-Zernike terms.

Per-star wavefront = MIW(intrinsic) + sum_v A_v * [G_kj . U_eff]_v, where
U_eff (n_kj, n_vmode) maps v-mode amplitudes to double-Zernike coefficients
W (over kj_grid = (focal_k, pupil_j)) and G_kj evaluates the focal-Zernike
field pattern per star.  Fit the 12 amplitudes A + atmosphere; this bakes in
the DOF<->wavefront correlations and degeneracy structure (replaces the
parameter-space SVD).
"""

import numpy as np
import galsim


def build_vmode_design(npz_path, thx_deg, thy_deg, jmax, fp_radius=1.75):
    """Per-star wavefront design from v-mode amplitudes.

    :returns: (G_v, vmode_names, meta) with
        G_v : ndarray (n_stars, jmax+1, n_vmode)  s.t. wavefront = G_v @ A
    """
    d = np.load(npz_path)
    U = d['U_eff']                    # (n_kj, n_vmode)
    kj = d['kj_grid']                 # (n_kj, 2) = (focal_k, pupil_j)
    n_kj, n_v = U.shape
    max_focal = int(kj[:, 0].max())

    thx = np.asarray(thx_deg, float) / fp_radius
    thy = np.asarray(thy_deg, float) / fp_radius
    fbasis = galsim.zernike.zernikeBasis(max_focal, thx, thy, R_outer=1.0)
    # fbasis[k, i] = focal Zernike Noll k at star i

    n_stars = thx.size
    # G_kj[star, pupil_j, kj_col] = F_{focal_k}(star)
    G_kj = np.zeros((n_stars, jmax + 1, n_kj))
    for c in range(n_kj):
        k_focal, j_pupil = int(kj[c, 0]), int(kj[c, 1])
        if j_pupil <= jmax:
            G_kj[:, j_pupil, c] = fbasis[k_focal]
    # contract with U_eff -> per-star wavefront from v-mode amplitudes
    G_v = np.einsum('sjc,cv->sjv', G_kj, U)          # (n_stars, jmax+1, n_v)
    names = [f'v{i+1}' for i in range(n_v)]
    meta = dict(U_eff=U, kj_grid=kj, iZs=d['iZs'], fp_radius=fp_radius)
    return G_v, names, meta


def dz_coeffs_from_vmodes(A, npz_path):
    """v-mode amplitudes -> double-Zernike coeffs W (dict keyed by (k,j))."""
    d = np.load(npz_path)
    W = d['U_eff'] @ np.asarray(A)                    # (n_kj,)
    return {(int(k), int(j)): float(w)
            for (k, j), w in zip(d['kj_grid'], W)}


def wavefront_at(A, npz_path, thx_deg, thy_deg, jmax=22, fp_radius=1.75):
    """Evaluate the v-mode (deviation) wavefront Zernike vector at field positions.

    :returns: ndarray (n_points, jmax+1) of Zj (microns) at each (thx,thy).
    """
    G_v, _, _ = build_vmode_design(npz_path, thx_deg, thy_deg, jmax, fp_radius)
    return np.einsum('sjv,v->sj', G_v, np.asarray(A))


def model_moments_at(model, npz_path, A, atm, thx_deg, thy_deg, rot_rad,
                     miw=None, jmax=22, fp_radius=1.75, batch=256):
    """Model HSM moments at arbitrary field positions (for data-vs-model plots).

    wavefront = MIW(thx,thy,rot) + G_v(thx,thy) @ A ; moments via the JAX model.
    """
    import jax
    import jax.numpy as jnp
    G_v = build_vmode_design(npz_path, thx_deg, thy_deg, jmax, fp_radius)[0]
    dev = np.einsum('sjv,v->sj', G_v, np.asarray(A))          # (n, jmax+1)
    z = dev.copy()
    if miw is not None:
        z = z + np.nan_to_num(miw.zernikes(thx_deg, thy_deg, rot_rad, jmax))
    atm_j = jnp.asarray(atm)
    fn = jax.jit(lambda zz: model.moments_adaptive(zz, atm_j))
    out = []
    for i in range(0, len(z), batch):
        out.append(np.array(jax.vmap(fn)(jnp.asarray(z[i:i + batch]))))
    return np.concatenate(out, axis=0)
