"""Science visit pairs that change elevation at fixed azimuth, rotator and band, and the
change in optical state across each pair from the Danish 1.3 corner wavefront sensors (CWFS).
"""
import numpy as np
import pandas as pd

from rubinwork import aos_state

#: Corner rafts in `aos_state.SENSOR_NAMES` order.
RAFTS = ['R00', 'R04', 'R40', 'R44']


def find_pairs(sci, min_del, max_daz, max_drot, max_dt_min, consecutive_only):
    """Pair science visits within each night that change elevation at fixed pointing.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        One row per science visit with ``day_obs``, ``seq_num``, ``visit_id``, ``band``,
        ``science_program``, ``obs_start_mjd`` [d], ``altitude_deg`` [deg],
        ``azimuth_deg_consdb`` [deg] and ``rotator_angle_deg`` [deg].
    min_del : `float`
        Minimum |elevation change| [deg].
    max_daz, max_drot : `float`
        Maximum |azimuth change| and |camera rotator angle change| [deg].
    max_dt_min : `float`
        Maximum time between the two visit starts [min].
    consecutive_only : `bool`
        Keep only pairs whose visits are consecutive among the night's science visits.

    Returns
    -------
    pairs : `pandas.DataFrame`
        One row per pair, earlier visit ``_1`` and later ``_2``, with ``d_el_deg``,
        ``d_az_deg``, ``d_rot_deg`` (later minus earlier, [deg]) and ``dt_min`` [min].
    """
    rows = []
    for day_obs, g in sci.groupby('day_obs'):
        g = g.sort_values('obs_start_mjd').reset_index(drop=True)
        t = g['obs_start_mjd'].to_numpy() * 1440.0
        el = g['altitude_deg'].to_numpy()
        az = g['azimuth_deg_consdb'].to_numpy()
        rot = g['rotator_angle_deg'].to_numpy()
        band = g['band'].to_numpy()
        i, j = np.triu_indices(len(g), 1)
        if consecutive_only:
            keep = j == i + 1
            i, j = i[keep], j[keep]
        d_az = (az[j] - az[i] + 180.0) % 360.0 - 180.0
        ok = ((np.abs(el[j] - el[i]) >= min_del) & (np.abs(d_az) < max_daz)
              & (np.abs(rot[j] - rot[i]) < max_drot) & (band[i] == band[j])
              & (t[j] - t[i] <= max_dt_min))
        for a, b, daz in zip(i[ok], j[ok], d_az[ok]):
            rows.append({'day_obs': day_obs,
                         'seq_num_1': g.at[a, 'seq_num'], 'seq_num_2': g.at[b, 'seq_num'],
                         'visit_id_1': g.at[a, 'visit_id'], 'visit_id_2': g.at[b, 'visit_id'],
                         'band': band[a], 'science_program': g.at[a, 'science_program'],
                         'el_1_deg': el[a], 'el_2_deg': el[b], 'd_el_deg': el[b] - el[a],
                         'az_1_deg': az[a], 'd_az_deg': daz,
                         'rot_1_deg': rot[a], 'd_rot_deg': rot[b] - rot[a],
                         'dt_min': t[b] - t[a]})
    return pd.DataFrame(rows)


def attach_deltas(pairs, per_visit, cols, prefix=''):
    """Add per-visit values at both ends of each pair and their change (2 minus 1).

    Parameters
    ----------
    pairs : `pandas.DataFrame`
        Output of `find_pairs`.
    per_visit : `pandas.DataFrame`
        Indexed by ``visit_id``, carrying `cols`.
    cols : `list` [`str`]
        Columns to difference.
    prefix : `str`, optional
        Output columns are ``<prefix><col>_1``, ``<prefix><col>_2`` and ``d_<prefix><col>``.

    Returns
    -------
    pairs : `pandas.DataFrame`
    """
    v1 = per_visit.reindex(pairs['visit_id_1'])[cols].to_numpy(float)
    v2 = per_visit.reindex(pairs['visit_id_2'])[cols].to_numpy(float)
    new = {}
    for k, c in enumerate(cols):
        new[f'{prefix}{c}_1'] = v1[:, k]
        new[f'{prefix}{c}_2'] = v2[:, k]
        new[f'd_{prefix}{c}'] = v2[:, k] - v1[:, k]
    return pd.concat([pairs, pd.DataFrame(new, index=pairs.index)], axis=1)


def corner_deviation(ccd, visit_ids, zk_noll=aos_state.ZK_NOLL):
    """Per-corner OCS deviation wavefront (total minus intrinsic) from `cwfs_science`.

    Parameters
    ----------
    ccd : `pandas.DataFrame`
        The `cwfs_science` ``ccd_visits`` table.
    visit_ids : `array_like` [`int`]
        Visits wanted.
    zk_noll : `list` [`int`], optional
        Noll indices, in the sensitivity matrix's row order.

    Returns
    -------
    z_dev : `pandas.DataFrame`
        Indexed by ``visit_id``, ``4 * len(zk_noll)`` columns ``z{j}_{raft}``, corner-major
        in `RAFTS` order, µm of wavefront in OCS. Only visits with every value finite.

    Notes
    -----
    Each raft's value is the mean of its two half-chips' quality-cut medians (SW0
    extra-focal, SW1 intra-focal), so a raft needs both halves. The per-raft MTAOS mean in
    ``raft_visits`` would be closer to what the summit used, but it is NaN on 0.449 of
    raft rows (dimensionless fraction of all raft rows) in build 20261026.
    """
    sub = ccd[ccd['visit'].isin(np.asarray(visit_ids))]
    cols = {}
    for j in zk_noll:
        cols[f'z{j}'] = (sub[f'zk_OCS_median_z{j}'] - sub[f'zk_intrinsic_OCS_median_z{j}'])
    dev = pd.DataFrame(cols).assign(visit=sub['visit'], raft=sub['raft'])
    n_half = dev.groupby(['visit', 'raft']).size()
    raft = dev.groupby(['visit', 'raft']).mean()
    raft = raft[n_half == 2]
    wide = raft.unstack('raft')
    want = [(f'z{j}', r) for r in RAFTS for j in zk_noll]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(want))
    wide.columns = [f'{z}_{r}' for z, r in want]
    wide.index.name = 'visit_id'
    return wide[np.isfinite(wide.to_numpy()).all(axis=1)]


def recover_states(z_dev, trim, state_estimator, n_modes, dof_set='standard_22'):
    """Deviation DOF, Trim and optical state per visit, in DOF and v-mode space.

    Parameters
    ----------
    z_dev : `pandas.DataFrame`
        From `corner_deviation`, µm of wavefront in OCS.
    trim : `pandas.DataFrame`
        Indexed by ``visit_id``, columns ``dof0``..``dof49``: the commanded Trim [µm, deg].
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `aos_state.make_state_estimator`.
    n_modes : `int`
        v-modes retained (12 for the 22/12 scheme).
    dof_set : `str`, optional
        Key into `aos_state.DOF_SETS`; the estimator's DOF set.

    Returns
    -------
    out : `pandas.DataFrame`
        Indexed by ``visit_id``. For each DOF ``k`` in the estimator's DOF set:
        ``dev_dof{k}`` (recovered from the deviation alone), ``trim_dof{k}`` and
        ``state_dof{k}`` = Trim minus deviation [µm, deg]; and ``dev_v{m}``,
        ``trim_v{m}``, ``state_v{m}`` for the v-modes [dimensionless]. ``resid_rms_um`` is
        the subspace residual [µm of wavefront].

    Notes
    -----
    The optical state is ``Trim - Deviation`` (`open_loop.optical_state_dofs`); the hexapod
    look-up table (LUT) is not in it. A visit without a finite Trim is dropped.
    """
    from rubinwork import open_loop
    idx = aos_state.DOF_SETS[dof_set]
    vids = [v for v in z_dev.index if v in trim.index]
    tr = trim.reindex(vids)[[f'dof{k}' for k in range(50)]].to_numpy(float)
    good = np.isfinite(tr).all(axis=1)
    vids = list(np.asarray(vids)[good])
    tr = tr[good]
    dev = np.full((len(vids), 50), np.nan)
    resid = np.full(len(vids), np.nan)
    for i, v in enumerate(vids):
        z = z_dev.loc[v].to_numpy(float)
        d, _, zc = aos_state.recover_optical_state(z, state_estimator, n_modes=n_modes)
        dev[i] = d
        resid[i] = np.sqrt(np.mean((z - zc) ** 2))
    state = open_loop.optical_state_dofs(dev, tr)
    vm = {name: aos_state.vmodes_from_dofs(x, state_estimator, n_modes=n_modes)
          for name, x in (('dev', dev), ('trim', tr), ('state', state))}
    cols = {}
    for name, x in (('dev', dev), ('trim', tr), ('state', state)):
        for k in idx:
            cols[f'{name}_dof{k}'] = x[:, k]
        for m in range(n_modes):
            cols[f'{name}_v{m + 1}'] = vm[name][:, m]
    cols['resid_rms_um'] = resid
    return pd.DataFrame(cols, index=pd.Index(vids, name='visit_id'))


def huber_lines_fixed_scale(x, y, freq, scale, start, t=1.345, max_iter=50, tol=1e-6):
    """Huber robust lines ``y = a + b x`` for many columns at once, with the scale held fixed.

    The bootstrap engine: one call fits every column of ``y`` by iteratively reweighted
    least squares with Huber weights at a fixed residual scale. With ``scale`` set to the
    converged scale of a statsmodels ``RLM(..., M=HuberT())`` fit and ``freq`` all ones, it
    returns that fit's parameters. Holding the scale fixed across bootstrap resamples drops
    a second-order term from the slope error and avoids a median per iteration.

    Parameters
    ----------
    x : `numpy.ndarray`
        Shape ``(n,)``, the regressor (here the elevation change, deg).
    y : `numpy.ndarray`
        Shape ``(n, m)``, one column per quantity, in that quantity's units.
    freq : `numpy.ndarray`
        Shape ``(n,)``, non-negative frequency weights (resample counts) [dimensionless].
    scale : `numpy.ndarray`
        Shape ``(m,)``, fixed residual scale per column, in the units of ``y``.
    start : `numpy.ndarray`
        Shape ``(2, m)``, starting ``(a, b)`` per column.
    t : `float`, optional
        Huber threshold in units of ``scale`` [dimensionless]; statsmodels' default.
    max_iter : `int`, optional
    tol : `float`, optional
        Stop when no parameter moves by more than ``tol`` times its column's scale.

    Returns
    -------
    params : `numpy.ndarray`
        Shape ``(2, m)``: intercept ``a`` (units of y) and slope ``b`` (units of y per unit
        x) per column.
    """
    a, b = np.array(start[0], float), np.array(start[1], float)
    xc = x[:, None]
    for _ in range(max_iter):
        u = np.abs(y - a - b * xc) / scale
        w = freq[:, None] * np.minimum(1.0, t / np.maximum(u, 1e-300))
        sw, sx, sy = w.sum(0), (w * xc).sum(0), (w * y).sum(0)
        sxx, sxy = (w * xc ** 2).sum(0), (w * xc * y).sum(0)
        b_new = (sw * sxy - sx * sy) / (sw * sxx - sx ** 2)
        a_new = (sy - b_new * sx) / sw
        done = np.all(np.abs(np.r_[a_new - a, b_new - b]) <= tol * np.r_[scale, scale])
        a, b = a_new, b_new
        if done:
            break
    return np.vstack([a, b])


def night_bootstrap(x, y, nights, masks, scale, start, n_boot=1000, seed=20261120):
    """Slope errors from resampling whole nights, for several sub-samples at once.

    Pairs from one night share visits and that night's alignment, so they are not
    independent; drawing nights with replacement keeps that correlation inside each draw.

    Parameters
    ----------
    x, y : `numpy.ndarray`
        As in `huber_lines_fixed_scale`; ``y`` shape ``(n, m)``.
    nights : `numpy.ndarray`
        Shape ``(n,)``, the ``day_obs`` of each pair.
    masks : `dict` [`str`, `numpy.ndarray`]
        Sub-sample name to boolean mask of shape ``(n,)``.
    scale, start : `dict` [`str`, `numpy.ndarray`]
        Per sub-sample, the fixed scale ``(m,)`` and starting parameters ``(2, m)``, from the
        statsmodels fit of that sub-sample.
    n_boot : `int`, optional
        Number of resamples.
    seed : `int`, optional

    Returns
    -------
    slopes : `dict` [`str`, `numpy.ndarray`]
        Per sub-sample, the resampled slopes, shape ``(n_boot, m)``, units of y per unit x.
    """
    rng = np.random.default_rng(seed)
    uniq, night_idx = np.unique(nights, return_inverse=True)
    out = {k: np.empty((n_boot, y.shape[1])) for k in masks}
    sub = {k: (x[m], y[m], night_idx[m]) for k, m in masks.items()}
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        for k, (xs, ys, ni) in sub.items():
            out[k][i] = huber_lines_fixed_scale(xs, ys, counts[ni].astype(float),
                                                scale[k], start[k])[1]
    return out


def huber_cluster_slope_err(x, y, clusters, params, scale, t=1.345):
    """Cluster-robust (sandwich) slope errors of Huber lines, for many columns at once.

    Pairs from one night share visits and that night's alignment, so the errors treat
    each night as one independent unit: ``V = A^-1 B A^-1 nc / (nc - 1)``, with ``A`` the
    summed Huber derivative and ``B`` the sum over nights of the outer product of each
    night's summed score. On the 20261120 sample it matched a 1000-resample night bootstrap
    to within 5% for every quantity, at a cost of under a second.

    Parameters
    ----------
    x : `numpy.ndarray`
        Shape ``(n,)``, the regressor (here the elevation change, deg).
    y : `numpy.ndarray`
        Shape ``(n, m)``, one column per quantity, in that quantity's units.
    clusters : `numpy.ndarray`
        Shape ``(n,)``, the cluster label of each row (here ``day_obs``).
    params : `numpy.ndarray`
        Shape ``(2, m)``, fitted intercept and slope per column.
    scale : `numpy.ndarray`
        Shape ``(m,)``, the fit's residual scale per column, in the units of ``y``.
    t : `float`, optional
        Huber threshold in units of ``scale`` [dimensionless].

    Returns
    -------
    slope_err : `numpy.ndarray`
        Shape ``(m,)``, units of y per unit x.
    """
    xc = x[:, None]
    u = (y - params[0] - params[1] * xc) / scale
    psi = np.clip(u, -t, t)
    dpsi = (np.abs(u) < t).astype(float)
    a00, a01, a11 = dpsi.sum(0) / scale, (dpsi * xc).sum(0) / scale, (dpsi * xc ** 2).sum(0) / scale
    _, ci = np.unique(clusters, return_inverse=True)
    nc = ci.max() + 1
    g0 = np.zeros((nc, y.shape[1]))
    g1 = np.zeros((nc, y.shape[1]))
    np.add.at(g0, ci, psi)
    np.add.at(g1, ci, psi * xc)
    b00, b01, b11 = (g0 * g0).sum(0), (g0 * g1).sum(0), (g1 * g1).sum(0)
    det = a00 * a11 - a01 ** 2
    var = (a01 ** 2 * b00 - 2 * a01 * a00 * b01 + a00 ** 2 * b11) / det ** 2 * nc / (nc - 1)
    return np.sqrt(var)
