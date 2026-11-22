"""A–B–A optical-state change across a telescope bounce, with Monte Carlo errors.

Per (night, leg) of a bounce, each B visit is differenced against the A (reference)
visits on either side of it, with A interpolated linearly in time to the B time. The
leg's ΔDZ is the median of those A–B–A differences over its B visits, in µm of
wavefront. ΔDZ is then recovered into degrees of freedom (DOF) and v-modes under one of
two schemes:

``22_12``
    the 22-DOF set (M2 and camera hexapods, B1_1-B1_7, B2_1-B2_5) with 12 modes kept,
    by the linear truncated solve of `rubinwork.bounce.legs` (``invert_truncated``);
    v-modes are the 12 of the ``standard_22`` StateEstimator.
``rbr_50_50``
    all 50 DOF and 50 modes by Range-Bounded Recovery (RBR), which is nonlinear; v-modes
    are the 50 of the ``all_50`` StateEstimator.

Errors are the standard deviation over Monte Carlo (MC) draws pushed through the same
A–B–A median and recovery. The noise model, fitted from the same visits:

- white noise on every DZ term, at the per-visit RMS of the B−A pairs (leg median
  removed), correlated across terms by the Ledoit–Wolf shrinkage estimate;
- red noise on DZ(1,4) only, a power-law structure function
  ``D(tau) = 2 sigma_W^2 + sigma_R^2 (tau / T0)^alpha`` fitted to same-setting visit
  differences in lag bins.

Hexapod rotations are in deg; translations and bending-mode amplitudes in µm.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.covariance import ledoit_wolf

from . import legs as bl

__all__ = ["SCHEMES", "leg_tables", "leg_visits", "aba_matrix", "white_noise", "red_noise",
           "Recovery", "make_recovery", "run"]

SCHEMES = ("22_12", "rbr_50_50")
"""Recovery schemes `make_recovery` knows."""


def leg_tables(ft, bounce, prefix, day_obs):
    """Visits, B−A pairs and legs of one bounce over a list of nights.

    Parameters
    ----------
    ft : `astropy.table.QTable`
        The per-visit DZ fit table after `rubinwork.bounce.analysis.load_fit_table`.
    bounce : `dict`
        One entry of the resolved configuration's ``bounces`` list.
    prefix : `str`
        DZ fit column prefix.
    day_obs : `list` [`int`]
        Nights to use; nights with no bounce visits are skipped.

    Returns
    -------
    visits : `pandas.DataFrame`
        One row per (night, leg, visit): ``row`` (index into `ft`), ``seq_num``,
        ``t_min`` (minutes since the night's first bounce visit) and ``setting``
        (``'A'``, ``'B'`` or ``'other'``).
    pairs : `pandas.DataFrame`
        The time-ordered B−A pairs of `rubinwork.bounce.legs.run_bounce`: ``row_A``,
        ``row_B`` and ``lag_min`` (min).
    legs : `pandas.DataFrame`
        One row per (night, leg) with B pairs: ``b_value_deg``, the nominal B position
        in deg.
    """
    dobs = np.asarray(ft['day_obs']).astype(int)
    seq = np.asarray(ft['seq_num']).astype(int)
    mjd = np.asarray(ft['mjd'], float)
    prog = bl.bounce_program_mask(ft, bounce)
    visits, pair_rows, legs = [], [], []
    for d in day_obs:
        night = (dobs == d) & prog
        if not night.any():
            continue
        rb = bl.run_bounce(ft, bounce, prefix, [1], [4], day_obs=d)
        t0 = mjd[night].min()
        for lab, cb in rb['comparisons'].items():
            if not cb['pairs']:
                continue
            legs.append({'night': d, 'leg': lab, 'b_value_deg': bl.leg_b_value(lab)})
            for i in np.nonzero(night)[0]:
                s = 'A' if rb['ref_mask'][i] else 'B' if cb['comp_mask'][i] else 'other'
                visits.append({'night': d, 'leg': lab, 'row': i, 'seq_num': seq[i],
                               't_min': (mjd[i] - t0) * 1440.0, 'setting': s})
            for r, c in cb['pairs']:
                pair_rows.append({'night': d, 'leg': lab, 'row_A': r, 'row_B': c,
                                  'lag_min': (mjd[c] - mjd[r]) * 1440.0})
    return pd.DataFrame(visits), pd.DataFrame(pair_rows), pd.DataFrame(legs)


def leg_visits(visits, night, leg):
    """A and B visits of one (night, leg), one row per visit, in sequence order."""
    v = visits[(visits.night == night) & (visits.leg == leg) & visits.setting.isin(['A', 'B'])]
    return v.drop_duplicates('seq_num').sort_values('seq_num')


def aba_matrix(v):
    """Weights of each A–B–A difference on the visits of `v`.

    Parameters
    ----------
    v : `pandas.DataFrame`
        One leg's visits from `leg_visits`.

    Returns
    -------
    M : `numpy.ndarray`, shape (n_aba, len(v))
        Row b is B visit b minus the two neighbouring A visits interpolated in time to
        it. A B visit without an A on both sides gets no row.
    """
    s = v.setting.to_numpy()
    t = v.t_min.to_numpy()
    A = np.nonzero(s == 'A')[0]
    rows = []
    for i in np.nonzero(s == 'B')[0]:
        a0 = A[A < i]
        a1 = A[A > i]
        if len(a0) and len(a1):
            a0, a1 = a0[-1], a1[0]
            f = (t[i] - t[a0]) / (t[a1] - t[a0])
            w = np.zeros(len(s))
            w[i] = 1
            w[a0] -= 1 - f
            w[a1] -= f
            rows.append(w)
    return np.array(rows)


def white_noise(W, pairs):
    """Per-visit white noise of every DZ term, from the B−A pairs.

    Parameters
    ----------
    W : `numpy.ndarray`, shape (n_visit, n_term)
        Per-visit DZ, µm of wavefront.
    pairs : `pandas.DataFrame`
        From `leg_tables`.

    Returns
    -------
    noise : `dict`
        ``sigma_w`` (n_term,) per-visit sigma in µm: ``sqrt(sum d^2 / (2 (N − G)))`` over
        N pairs in G (night, leg) groups, each group's median removed. ``cov`` the
        (n_term, n_term) covariance in µm², ``shrink`` the Ledoit–Wolf shrinkage
        (dimensionless; 0 is the sample correlation, 1 the identity), ``n_pairs``,
        ``n_groups``.

    Notes
    -----
    The pairs are about 2.8 min apart, so ``sigma_w`` includes that much drift: an upper
    limit on the white floor.
    """
    groups = pairs.groupby(['night', 'leg']).indices
    dp = W[pairs.row_B.to_numpy()] - W[pairs.row_A.to_numpy()]
    dc = np.zeros_like(dp)
    for ix in groups.values():
        dc[ix] = dp[ix] - np.median(dp[ix], 0)
    n, g = len(dc), len(groups)
    sw = np.sqrt((dc ** 2).sum(0) / (2 * (n - g)))
    c, shrink = ledoit_wolf(dc / (np.sqrt(2) * sw), assume_centered=True)
    corr = c / np.sqrt(np.outer(np.diag(c), np.diag(c)))
    return {'sigma_w': sw, 'cov': corr * np.outer(sw, sw), 'shrink': float(shrink),
            'n_pairs': n, 'n_groups': g}


def red_noise(W, visits, term, sigma_w, lag_bins, t0_min):
    """Red-noise structure function of one DZ term from same-setting visit differences.

    Parameters
    ----------
    W : `numpy.ndarray`, shape (n_visit, n_term)
        Per-visit DZ, µm of wavefront.
    visits : `pandas.DataFrame`
        From `leg_tables`.
    term : `int`
        Column of `W` to fit (DZ(1,4) in practice).
    sigma_w : `float`
        That term's white sigma per visit, µm; held fixed.
    lag_bins : `list` [`float`]
        Lag bin edges, min. Bins with fewer than 5 differences are dropped.
    t0_min : `float`
        Reference lag T0, min.

    Returns
    -------
    red : `dict`
        ``sigma_r`` (µm over T0) and ``alpha`` (dimensionless) with ``*_err``, the fit
        ``chi2`` and ``dof``, and per bin ``lag_min`` (median lag), ``n`` and ``D``
        (µm²).

    Notes
    -----
    A visits are grouped by night, B visits by (night, leg), so two B elevations of one
    night are never differenced. Bin errors are Gaussian, ``D sqrt(2 / n)``; differences
    that share visits are correlated, so those errors are optimistic.
    """
    vv = visits[visits.setting.isin(['A', 'B'])].copy()
    vv['grp'] = np.where(vv.setting == 'A', 'A', vv.leg)
    vv = vv.drop_duplicates(['night', 'grp', 'seq_num'])
    dys, lags = [], []
    for _, g in vv.groupby(['night', 'grp']):
        g = g.sort_values('t_min')
        tt = g.t_min.to_numpy()
        rr = g.row.to_numpy()
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                dys.append(W[rr[j], term] - W[rr[i], term])
                lags.append(tt[j] - tt[i])
    dys = np.array(dys)
    lags = np.array(lags)
    bi = np.digitize(lags, lag_bins) - 1
    D, nb, tb = [], [], []
    for k in range(len(lag_bins) - 1):
        m = bi == k
        if m.sum() >= 5:
            D.append((dys[m] ** 2).mean())
            nb.append(int(m.sum()))
            tb.append(np.median(lags[m]))
    D, nb, tb = np.array(D), np.array(nb), np.array(tb)
    ey = D * np.sqrt(2 / nb)
    fr = least_squares(lambda p: (2 * sigma_w ** 2 + p[0] ** 2 * (tb / t0_min) ** p[1] - D) / ey,
                       [0.1, 1.5], bounds=([0, 0.1], [1, 2.5]))
    err = np.sqrt(np.diag(np.linalg.inv(fr.jac.T @ fr.jac)))
    return {'sigma_r': float(fr.x[0]), 'alpha': float(fr.x[1]), 'sigma_r_err': float(err[0]),
            'alpha_err': float(err[1]), 'chi2': float((fr.fun ** 2).sum()), 'dof': len(D) - 2,
            'lag_min': tb, 'n': nb, 'D': D}


class Recovery:
    """One recovery scheme: ΔDZ (µm of wavefront) to ΔDOF, plus the maps around it.

    Attributes
    ----------
    scheme : `str`
    dof_index : `list` [`int`]
        The DOF recovered, as indices into the 50-DOF list.
    labels, units : `list` [`str`]
        Per recovered DOF; units are ``'um'`` or ``'deg'``.
    forward : `numpy.ndarray`, shape (n_term, n_dof)
        µm of wavefront per unit DOF.
    to_vmode : `numpy.ndarray`, shape (n_dof, n_mode)
        Δv = ΔDOF @ to_vmode (v-modes dimensionless).
    linear : `numpy.ndarray` or `None`
        (n_dof, n_term) matrix when the scheme is linear, else `None`.
    """

    def __init__(self, scheme, svd, solve, linear, to_vmode, dof_index):
        from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF
        self.scheme = scheme
        self.svd = svd
        self._solve = solve
        self.linear = linear
        self.to_vmode = to_vmode
        self.dof_index = list(dof_index)
        self.labels = [LABELS_50DOF[i] for i in self.dof_index]
        self.units = ['deg' if i in (3, 4, 8, 9) else 'um' for i in self.dof_index]
        self.forward = svd.estimator.forward_operator()
        self.kj = list(svd.kj_grid)

    def __call__(self, dz):
        """ΔDOF of one ΔDZ (n_term,) or of a stack of them (n, n_term)."""
        dz = np.asarray(dz)
        if self.linear is not None:
            return dz @ self.linear.T
        if dz.ndim == 1:
            return np.asarray(self._solve(dz))
        return np.array([self._solve(x) for x in dz])


def make_recovery(scheme, cfg):
    """Build the `Recovery` of one scheme from a resolved bounce configuration.

    Parameters
    ----------
    scheme : `str`
        One of `SCHEMES`.
    cfg : `dict`
        From `rubinwork.bounce.legs.resolve_config`. Reads ``pupil_j_range``,
        ``focal_k_range``, ``ofc_normalization_yaml``, and per scheme
        ``n_keep_reduced`` (22/12) or ``rbr_n_keep``, ``n_dof`` and the ``rbr_*`` knobs
        (RBR 50/50).
    """
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd

    from rubinwork import aos_state

    iZs = list(cfg['pupil_j_range'])
    kr = cfg['focal_k_range']
    if scheme == '22_12':
        dof = list(aos_state.DOF22)
        nk = int(cfg['n_keep_reduced'])
        svd = build_ofc_svd(iZs, int(min(kr)), int(max(kr)), nk, n_dof=dof,
                            ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
        inv_t = bl.rbr_module().invert_truncated
        P = np.array([inv_t(e, svd) for e in np.eye(len(svd.kj_grid))]).T
        se = aos_state.make_state_estimator(dof_set='standard_22', n_modes=nk)
        T_V = aos_state.vmodes_from_dofs(np.eye(50), se, n_modes=nk)[dof]
        return Recovery(scheme, svd, None, P, T_V, dof)
    if scheme == 'rbr_50_50':
        svd = build_ofc_svd(iZs, int(min(kr)), int(max(kr)), int(cfg['rbr_n_keep']),
                            n_dof=cfg['n_dof'],
                            ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
        se = aos_state.make_state_estimator(dof_set='all_50', n_modes=50)
        T_V = aos_state.vmodes_from_dofs(np.eye(50), se, n_modes=50)

        def solve(w):
            return svd.estimator.dof_range_bounded(
                w, kappa=float(cfg['rbr_kappa']), power=int(cfg['rbr_power']),
                penalty_weight=float(cfg['rbr_penalty_weight']),
                full_dof=bool(cfg['rbr_full_dof']))
        return Recovery(scheme, svd, solve, None, T_V, range(50))
    raise ValueError(f'unknown scheme {scheme!r}; known: {SCHEMES}')


def run(W, visits, legs, rec, noise, red, term14, *, n_mc, seed, t0_min):
    """A–B–A Δ, recovery and MC errors for every leg.

    Parameters
    ----------
    W : `numpy.ndarray`, shape (n_visit, n_term)
        Per-visit DZ in `rec`'s term order, µm of wavefront.
    visits, legs : `pandas.DataFrame`
        From `leg_tables`.
    rec : `Recovery`
    noise : `dict`
        From `white_noise`.
    red : `dict`
        From `red_noise`, for term `term14`.
    term14 : `int`
        Column of the red-noise term.
    n_mc : `int`
        MC draws per leg.
    seed : `int`
        Seed of the one generator drawn from leg by leg, in `legs` order.
    t0_min : `float`
        Red-noise reference lag, min.

    Returns
    -------
    results : `list` [`dict`]
        Per leg, in `legs` order: ``n_A``, ``n_B``, ``n_aba``; ``dz``, ``fit``,
        ``resid`` (n_term,) in µm with ``*_err``; ``dof`` (n_dof,) in `rec.units` and
        ``vmode`` (n_mode,) dimensionless, with ``*_err``; ``dof_mc`` (n_mc, n_dof).

    Notes
    -----
    For a nonlinear scheme the recovery is solved at ΔDZ + δ for each noise draw δ, so
    the errors need no linearisation. ``fit`` is ``forward @ ΔDOF``; for 22/12 it is the
    projection of ΔDZ onto the 12 kept u-modes.
    """
    nt = W.shape[1]
    LW = np.linalg.cholesky(noise['cov'] + 1e-12 * np.trace(noise['cov']) / nt * np.eye(nt))
    rng = np.random.default_rng(seed)
    alpha, sr = red['alpha'], red['sigma_r']
    out = []
    for L in legs.itertuples():
        v = leg_visits(visits, L.night, L.leg)
        M = aba_matrix(v)
        dz = np.median(M @ W[v.row.to_numpy()], 0)
        u = (v.t_min.to_numpy() - v.t_min.min()) / t0_min
        K = 0.5 * (u[:, None] ** alpha + u[None, :] ** alpha
                   - np.abs(u[:, None] - u[None, :]) ** alpha)
        Lt = np.linalg.cholesky(K + 1e-9 * np.eye(len(u)))
        z = rng.normal(size=(n_mc, len(v), nt)) @ LW.T
        z[:, :, term14] += sr * rng.normal(size=(n_mc, len(v))) @ Lt.T
        dz_mc = dz + np.median(np.einsum('bi,mik->mbk', M, z), 1)
        dof = rec(dz)
        dof_mc = rec(dz_mc)
        fit = rec.forward @ dof
        fit_mc = dof_mc @ rec.forward.T
        s = v.setting.to_numpy()
        out.append({'n_A': int((s == 'A').sum()), 'n_B': int((s == 'B').sum()),
                    'n_aba': len(M),
                    'dz': dz, 'dz_err': dz_mc.std(0, ddof=1),
                    'fit': fit, 'fit_err': fit_mc.std(0, ddof=1),
                    'resid': dz - fit, 'resid_err': (dz_mc - fit_mc).std(0, ddof=1),
                    'dof': dof, 'dof_err': dof_mc.std(0, ddof=1), 'dof_mc': dof_mc,
                    'vmode': dof @ rec.to_vmode,
                    'vmode_err': (dof_mc @ rec.to_vmode).std(0, ddof=1)})
    return out
