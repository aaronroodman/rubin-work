"""
YAML configuration + parameter layout for the OptAtmo moment fit.

Loads a config dict and builds a ParamLayout that packs/unpacks the flat fit
vector p = [dz_params, atm_params, moment_offsets] and knows which entries are
free, their bounds, and normalisation scales.
"""

import numpy as np
import yaml

from jax_optatmo import MOMENT_LABELS

ATM_NAMES = ['fwhm', 'g1', 'g2']


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class ParamLayout:
    """Packs/unpacks the flat parameter vector and applies normalisation.

    Segments: DZ params (n_dz), atmosphere (only the free ones), moment
    offsets (one per selected moment).  `scales` normalises each parameter so
    a unit step has a comparable effect (improves optimiser conditioning).
    """

    def __init__(self, cfg, dz_names):
        self.cfg = cfg
        self.dz_names = list(dz_names)
        self.n_dz = len(dz_names)

        atm_cfg = cfg['atmosphere']
        self.atm_free = list(atm_cfg.get('fit', ATM_NAMES))
        self.atm_init = {k: float(atm_cfg['init'][k]) for k in ATM_NAMES}
        self.atm_bounds = {k: tuple(atm_cfg['bounds'][k]) for k in ATM_NAMES}

        self.offset_moments = list(cfg.get('moment_offsets', {})
                                   .get('moments', []))
        self.offset_init = float(cfg.get('moment_offsets', {}).get('init', 0.0))

        # names / order of the free parameters
        self.names = (list(self.dz_names)
                      + [f'atm_{a}' for a in self.atm_free]
                      + [f'off_{m}' for m in self.offset_moments])
        self.n = len(self.names)

        # index slices
        self.i_dz = slice(0, self.n_dz)
        self.i_atm = slice(self.n_dz, self.n_dz + len(self.atm_free))
        self.i_off = slice(self.n_dz + len(self.atm_free), self.n)

    def initial(self):
        p = np.zeros(self.n)
        for i, a in enumerate(self.atm_free):
            p[self.n_dz + i] = self.atm_init[a]
        p[self.i_off] = self.offset_init
        return p

    def bounds(self):
        b = [(-np.inf, np.inf)] * self.n_dz
        for a in self.atm_free:
            b.append(self.atm_bounds[a])
        b += [(-np.inf, np.inf)] * len(self.offset_moments)
        return b

    def atm_vector(self, p):
        """Full [fwhm, g1, g2] jnp-ready list from the free atm params + fixed."""
        vals = dict(self.atm_init)
        for i, a in enumerate(self.atm_free):
            vals[a] = p[self.n_dz + i]
        return [vals['fwhm'], vals['g1'], vals['g2']]

    def offset_vector(self, p):
        """Length-12 moment-offset vector (zeros except selected moments)."""
        off = [0.0] * len(MOMENT_LABELS)
        for i, m in enumerate(self.offset_moments):
            off[MOMENT_LABELS.index(m)] = p[self.i_off][i]
        return off
