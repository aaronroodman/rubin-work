"""
Batched forward model + cost for the OptAtmo moment fit.

Assembles, for a set of stars, the model HSM moments as a differentiable
function of the flat parameter vector p = [dz, atm_free, moment_offsets]:

    z_i   = z0_i (MIW baseline)  +  G_i @ dz          (per-star wavefront)
    m_i   = JaxOptAtmoPSF.moments_adaptive(z_i, atm)  +  offsets
    cost  = sum_i sum_{m in fit} w_m ((d_im - m_im) / e_im)^2  / n_stars

Everything is jax; use jax.value_and_grad(cost) for the optimiser.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jax_optatmo import MOMENT_LABELS


class Forward:
    def __init__(self, model, layout, z0, G, data, err,
                 fit_moments, weights, reg_lambda=0.0):
        self.model = model
        self.layout = layout
        self.z0 = jnp.asarray(z0)                 # (n_stars, jmax+1)
        self.G = jnp.asarray(G)                   # (n_stars, jmax+1, n_dz)
        self.n_dz = layout.n_dz

        self.d = jnp.asarray(data)                # (n_stars, 12)
        self.e = jnp.asarray(err)
        self.sel = jnp.asarray([MOMENT_LABELS.index(m) for m in fit_moments])
        self.w = jnp.asarray([weights.get(m, 1.0) for m in fit_moments])

        # static index maps for assembling atm[3] and offsets[12] from p
        self.atm_init = jnp.asarray([layout.atm_init[a]
                                     for a in ['fwhm', 'g1', 'g2']])
        self.atm_free_pos = [(['fwhm', 'g1', 'g2'].index(a), layout.n_dz + i)
                             for i, a in enumerate(layout.atm_free)]
        self.off_pos = [(MOMENT_LABELS.index(m), layout.i_off.start + i)
                        for i, m in enumerate(layout.offset_moments)]
        self.reg_lambda = float(reg_lambda)   # Tikhonov L2 penalty on v-mode amps

    def _atm(self, p):
        atm = self.atm_init
        for ai, pi in self.atm_free_pos:
            atm = atm.at[ai].set(p[pi])
        return atm

    def _offsets(self, p):
        off = jnp.zeros(len(MOMENT_LABELS))
        for mi, pi in self.off_pos:
            off = off.at[mi].set(p[pi])
        return off

    def moments(self, p):
        dz = p[:self.n_dz]
        Z = self.z0 + jnp.einsum('skp,p->sk', self.G, dz)
        atm = self._atm(p)
        M = jax.vmap(lambda z: self.model.moments_adaptive(z, atm))(Z)
        return M + self._offsets(p)[None, :]

    def residuals(self, p):
        M = self.moments(p)
        r = (self.d[:, self.sel] - M[:, self.sel]) / self.e[:, self.sel]
        return r * jnp.sqrt(self.w)[None, :]

    def cost(self, p):
        r = self.residuals(p)
        chi2 = jnp.sum(r ** 2) / r.shape[0]
        reg = self.reg_lambda * jnp.sum(p[:self.n_dz] ** 2)   # Tikhonov on v-modes
        return chi2 + reg
