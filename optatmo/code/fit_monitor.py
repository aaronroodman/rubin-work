"""Fit monitoring for the v-mode OptAtmo fit.

Records the objective (reduced chi2 + Tikhonov reg) and the full parameter
vector at EVERY objective evaluation, marks L-BFGS-B iteration boundaries (via
the scipy callback), and times the fit.  Produces a diagnostics plot:
  1. cost vs evaluation (objective and, if reg>0, the data-only reduced chi2),
     log scale, with iteration boundaries;
  2. v-mode amplitudes vs evaluation;
  3. atmosphere params (fwhm, g1, g2) vs evaluation.
and a stats dict (nfev, njev, nit, wall time, final cost) saved with the fit.
"""
import time

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FitMonitor:
    def __init__(self, label='', verbose=False, checkpoint=None):
        self.costs = []            # objective at every evaluation
        self.params = []           # parameter vector at every evaluation
        self.iter_evals = []       # evaluation count at each iteration boundary
        self.t_fit = None
        self._t0 = None
        self.label = label         # tag for the printed progress lines
        self.verbose = verbose     # print a cost line at every evaluation
        self.checkpoint = checkpoint   # path: rewrite the trace each iteration

    def _elapsed(self):
        return (time.perf_counter() - self._t0) if self._t0 is not None else 0.0

    def objective(self, vg):
        """Wrap a jitted value_and_grad(cost) into a scipy fun(p) -> (f, grad)
        that records (cost, params) on every call (and, if verbose, prints the
        cost at every function evaluation -- flushed, so a killed job's log
        still shows the chi2 trajectory)."""
        def fun(p):
            v, g = vg(jnp.asarray(p))
            self.costs.append(float(v))
            self.params.append(np.asarray(p, float).copy())
            if self.verbose:
                print(f'[{self.label}] eval {len(self.costs):4d}  '
                      f'cost {float(v):.6g}  t {self._elapsed():6.0f}s', flush=True)
            return float(v), np.asarray(g, float)
        return fun

    def callback(self, xk, *args):
        self.iter_evals.append(len(self.costs))
        if self.checkpoint:        # persist the trace so a time-limit kill still
            self.save(self.checkpoint)   # leaves a reviewable partial result

    def save(self, path):
        np.savez(path, costs=np.asarray(self.costs, float),
                 params=np.asarray(self.params, float),
                 iter_evals=np.asarray(self.iter_evals, int))

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        self.t_fit = time.perf_counter() - self._t0

    def stats(self, res):
        return dict(
            nfev=int(res.nfev), njev=int(getattr(res, 'njev', res.nfev)),
            nit=int(res.nit), n_eval=len(self.costs),
            time_s=float(self.t_fit if self.t_fit is not None else np.nan),
            final_cost=float(res.fun), success=bool(res.success),
            costs=np.asarray(self.costs, float),
            params=np.asarray(self.params, float),
            iter_evals=np.asarray(self.iter_evals, int))

    def summary_line(self, res):
        s = self.stats(res)
        return (f"nfev={s['nfev']} njev={s['njev']} nit={s['nit']} "
                f"n_eval={s['n_eval']} time={s['time_s']:.1f}s "
                f"final_cost={s['final_cost']:.5g} success={s['success']}")

    def plot(self, path, res, i_dz, vmode_names, atm_idx, atm_names,
             reg_lambda=0.0, title=''):
        costs = np.asarray(self.costs, float)
        P = np.asarray(self.params, float)              # (n_eval, n_param)
        ev = np.arange(1, len(costs) + 1)
        dz = P[:, i_dz]                                 # (n_eval, n_v)
        reg = reg_lambda * np.sum(dz ** 2, axis=1)
        chi2 = costs - reg                              # data-only reduced chi2

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for e in self.iter_evals:                       # iteration boundaries
            for a in ax:
                a.axvline(e, color='0.88', lw=0.5, zorder=0)
        ax[0].semilogy(ev, np.clip(costs, 1e-15, None), '-', lw=1.2,
                       color='C0', label='objective (χ²ᵣ + reg)')
        if reg_lambda > 0:
            ax[0].semilogy(ev, np.clip(chi2, 1e-15, None), '--', lw=1.0,
                           color='C1', label='reduced χ² (data)')
        ax[0].set_xlabel('function evaluation'); ax[0].set_ylabel('cost')
        ax[0].set_title('cost vs evaluation'); ax[0].legend(fontsize=7)

        for i in range(dz.shape[1]):
            ax[1].plot(ev, dz[:, i], lw=0.8,
                       label=vmode_names[i] if dz.shape[1] <= 14 else None)
        ax[1].set_xlabel('function evaluation'); ax[1].set_ylabel('amplitude')
        ax[1].set_title(f'v-mode amplitudes vs evaluation ({dz.shape[1]} modes)')
        if dz.shape[1] <= 14:
            ax[1].legend(fontsize=6, ncol=2)

        for idx, nm in zip(atm_idx, atm_names):
            ax[2].plot(ev, P[:, idx], lw=1.3, label=nm)
        ax[2].set_xlabel('function evaluation'); ax[2].set_ylabel('value')
        ax[2].set_title('atmosphere params vs evaluation'); ax[2].legend(fontsize=8)

        fig.suptitle(f'{title}  |  {self.summary_line(res)}')
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
