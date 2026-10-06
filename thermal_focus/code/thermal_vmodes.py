"""Thermal response of all 34 v-modes, not just the defocus mode.

The thermal-focus deliverable models one quantity: v-mode 1, which is almost pure uniform
defocus. This module asks the same question of every mode the 50-degree-of-freedom (DOF)
recovery retains -- whether any other v-mode carries a component predictable from telescope
thermal telemetry.

The response for mode ``k`` is the **optical state**, ``Trim - Deviation``, which is the
negative of the stored open-loop column::

    y_k [dimensionless v-mode amplitude] = -v{k}_olr = v{k}_trim - v{k}

Verified against `thermal_focus_lib.MEASURED_SIGN` to 6.7e-16 (dimensionless) over modes 1 to
34, so this is the v-mode-1 convention generalized and not a new one. Unlike the v-mode-1
deliverable, the response is **left dimensionless** rather than divided into µm of equivalent
hexapod dz: that conversion is specific to defocus, and no single physical axis stands in for
the higher modes.

Three things make this more than running the existing fit 34 times.

**The null matters more than the fit.** Most modes are expected to carry no thermal signal, so
the question is per-mode skill against an intercept-only model, not goodness of fit. Skill is
the fractional reduction in out-of-fold residual nMAD, with both terms night-grouped.

**34 simultaneous tests need a correction.** `mode_table` reports a Benjamini-Hochberg
false-discovery-rate threshold alongside the raw per-mode statistic, so a mode is called
thermal only if it survives it.

**The high modes are measurement noise.** Four corner wavefront sensors (CWFS) constrain 84
Zernike values, and the recovery's own scatter grows with mode index; above roughly v12 the
per-visit measurement error dominates the astrophysical signal. `noise_floor_table` reports
where that sits so a reader does not take a high-mode slope at face value.

Nights are held out whole throughout, via `thermal_focus_fit.evaluate`. Only 2.7% of the truss
temperature's variance is within-night, so consecutive visits are near-duplicates in feature
space and a visit-level split leaks; see `thermal_focus_fit` for the measured size of that leak.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

import thermal_focus_fit as F                                    # noqa: E402
import thermal_focus_lib as L                                    # noqa: E402
from common.utils import nmad                                    # noqa: E402

#: Variant carrying the primary result: range-bounded recovery (RBR), which is physically
#: realizable on 99.5% of visits. The unconstrained ``v50_34__batoid__consdb_v1`` asks a median
#: 33x the available actuator stroke on every visit, so its high-mode amplitudes are partly
#: fitting an unreachable state. See ``olr/docs/scheme_comparison.md``.
PRIMARY_VARIANT = 'v50_34_rbr__batoid__consdb_v1'

#: The two variants compared to separate an intrinsic-route effect from a thermal one. Both are
#: unconstrained 50/34, so the solver is held fixed and only the intrinsic differs -- there is no
#: RBR arm on the measured intrinsic wavefront (MIW) route.
INTRINSIC_PAIR = ('v50_34__batoid__consdb_v1', 'v50_34__miw__consdb_v1')

#: V-modes retained by the 50-DOF recovery.
N_MODES = 34

#: Mode index above which four corner sensors constrain the state poorly, so a fitted slope is
#: reported but not interpreted. Set from `noise_floor_table`, not assumed.
WELL_CONSTRAINED_MAX = 12

#: False-discovery-rate level for the 34 simultaneous per-mode tests (dimensionless).
FDR_Q = 0.05


def response_columns(n_modes=N_MODES):
    """Stored open-loop column names, in mode order.

    Parameters
    ----------
    n_modes : `int`, optional
        Number of v-modes the variant retains.

    Returns
    -------
    cols : `list` [`str`]
        ``['v1_olr', ..., 'v{n_modes}_olr']``.
    """
    return [f'v{k}_olr' for k in range(1, n_modes + 1)]


def attach_mode_response(df, mode, out_col='y'):
    """Set `out_col` to one v-mode's optical state, the negated open-loop value.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``v{mode}_olr``.
    mode : `int`
        V-mode index, 1-based.
    out_col : `str`, optional
        Column written, defaulting to the ``y`` that `thermal_focus_fit.evaluate` reads.

    Returns
    -------
    out : `pandas.DataFrame`
        Copy with `out_col` set [dimensionless v-mode amplitude].

    Notes
    -----
    The sign is the one `thermal_focus_lib.attach_response` applies to v-mode 1, written here
    directly against the stored open-loop column rather than recomputed from ``v_trim`` and
    ``v_meas``. The two agree to 6.7e-16 (dimensionless); the stored column is preferred because
    it is non-null exactly where the open-loop state was recoverable.
    """
    col = f'v{mode}_olr'
    if col not in df.columns:
        raise KeyError(f'{col} absent; optical_state must be read with wide=True on a variant '
                       f'retaining at least {mode} v-modes')
    out = df.copy()
    out[out_col] = -out[col].to_numpy(float)
    return out


def null_nmad(df, ycol='y', n_splits=F.N_SPLITS):
    """Out-of-fold residual nMAD of an intercept-only model, nights held out whole.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying `ycol` and ``day_obs``.
    ycol : `str`, optional
    n_splits : `int`, optional
        Folds, matched to the fitted model's.

    Returns
    -------
    nmad_null : `float`
        Residual nMAD [same units as `ycol`] of predicting each held-out night by the median of
        the training nights.

    Notes
    -----
    The intercept is the **median** of the training folds, not the mean, so the null is the
    robust counterpart of the Huber fit it is compared against. A mean-based null would be pulled
    by the same one-sided tail the Huber fit is chosen to resist, flattering the fit.
    """
    from sklearn.model_selection import GroupKFold

    y = df[ycol].to_numpy(float)
    groups = df['day_obs'].to_numpy()
    pred = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(y, y, groups=groups):
        pred[test] = np.nanmedian(y[train])
    return nmad(y - pred)


def fit_mode(df, features, mode, model='huber', n_splits=F.N_SPLITS, verbose=False):
    """Night-grouped fit and null for one v-mode.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``v{mode}_olr``, ``day_obs`` and the feature columns.
    features : `list` [`str`]
        Thermal feature columns.
    mode : `int`
        V-mode index, 1-based.
    model : `str`, optional
        Key for `thermal_focus_fit.make_model`.
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    row : `dict`
        ``mode``, ``n_visits``, ``n_nights``, ``nmad_null`` and ``nmad_fit`` [dimensionless
        v-mode amplitude], ``skill`` (fractional nMAD reduction, dimensionless), ``r2``, and
        ``coef_<feature>`` per feature [dimensionless v-mode amplitude per feature unit].
    """
    d = attach_mode_response(df, mode)
    d = d[np.isfinite(d['y'])]
    if len(d) < 2 * n_splits or d['day_obs'].nunique() < n_splits:
        return dict(mode=mode, n_visits=len(d), n_nights=int(d['day_obs'].nunique()),
                    nmad_null=np.nan, nmad_fit=np.nan, skill=np.nan, r2=np.nan)

    res = F.evaluate(d, features, model=model, n_splits=n_splits, verbose=False)
    n0 = null_nmad(d, n_splits=n_splits)
    row = dict(mode=mode, n_visits=len(d), n_nights=int(d['day_obs'].nunique()),
               nmad_null=n0, nmad_fit=res['nmad'],
               skill=np.nan if not np.isfinite(n0) or n0 == 0 else 1.0 - res['nmad'] / n0,
               r2=res['r2'])
    if res['coefs'] is not None and len(res['coefs']):
        med = np.nanmedian(res['coefs'], axis=0)
        for name, c in zip(features, med):
            row[f'coef_{name}'] = float(c)
    if verbose:
        print(f'v{mode}: nMAD {row["nmad_fit"]:.4g} against null {row["nmad_null"]:.4g} '
              f'(dimensionless v-mode amplitude), skill {row["skill"]:+.3f} (dimensionless), '
              f'n={row["n_visits"]} visits over {row["n_nights"]} nights')
    return row


def bh_threshold(stats, q=FDR_Q):
    """Benjamini-Hochberg cut on a set of per-mode skill statistics.

    Parameters
    ----------
    stats : `array_like` [`float`]
        Per-mode skill (fractional nMAD reduction, dimensionless). Larger is stronger.
    q : `float`, optional
        False-discovery-rate level (dimensionless).

    Returns
    -------
    keep : `numpy.ndarray` [`bool`]
        Modes called thermal at this level.
    cut : `float`
        Skill value at the threshold, NaN if nothing survives [dimensionless].

    Notes
    -----
    Skill has no analytic null distribution here -- the response is spatially correlated between
    modes and the folds are not independent -- so the empirical null is taken from the modes
    themselves: the median skill over all 34 and its nMAD set a z-like scale, and the
    Benjamini-Hochberg step-up is applied to the resulting one-sided tail probabilities. This is
    a screening rule for which modes deserve a closer look, **not** a calibrated significance
    claim. A mode near the threshold should be confirmed by `thermal_focus_fit.nested_comparison`
    on that mode alone.
    """
    from scipy.stats import norm

    s = np.asarray(stats, float)
    good = np.isfinite(s)
    keep = np.zeros(len(s), bool)
    if good.sum() < 3:
        return keep, np.nan
    scale = nmad(s[good])
    if not np.isfinite(scale) or scale == 0:
        return keep, np.nan
    z = (s - np.nanmedian(s[good])) / scale
    p = np.where(good, norm.sf(z), np.nan)

    order = np.argsort(np.where(good, p, np.inf))
    m = int(good.sum())
    n_sig = 0
    for i, idx in enumerate(order[:m], start=1):
        if p[idx] <= q * i / m:
            n_sig = i
    if not n_sig:
        return keep, np.nan
    keep[order[:n_sig]] = True
    return keep, float(np.nanmin(s[keep]))


def mode_table(df, features=None, variant=PRIMARY_VARIANT, model='huber',
               n_modes=N_MODES, n_splits=F.N_SPLITS, q=FDR_Q, verbose=True):
    """Thermal skill of every v-mode, with the multiple-comparison cut applied.

    Parameters
    ----------
    df : `pandas.DataFrame`
        One row per visit, carrying the ``v*_olr`` columns, ``day_obs`` and the features.
    features : `list` [`str`], optional
        Defaults to `thermal_focus_lib.DELIVERABLE_GROUPS` resolved -- the five thermal channels.
    variant : `str`, optional
        Recorded in the result for provenance; not used to filter.
    model : `str`, optional
    n_modes : `int`, optional
    n_splits : `int`, optional
    q : `float`, optional
        False-discovery-rate level (dimensionless).
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        One row per mode, sorted by descending skill, with ``thermal`` [`bool`] from
        `bh_threshold` and ``well_constrained`` [`bool`] for ``mode <= WELL_CONSTRAINED_MAX``.
    """
    if features is None:
        features = L.resolve_features(L.DELIVERABLE_GROUPS)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f'features absent from the frame: {", ".join(missing)}')

    rows = [fit_mode(df, features, k, model=model, n_splits=n_splits, verbose=False)
            for k in range(1, n_modes + 1)]
    tab = pd.DataFrame(rows)
    keep, cut = bh_threshold(tab['skill'].to_numpy(float), q=q)
    tab['thermal'] = keep
    tab['well_constrained'] = tab['mode'] <= WELL_CONSTRAINED_MAX
    tab.attrs['variant'] = variant
    tab.attrs['features'] = list(features)
    tab.attrs['skill_cut'] = cut
    tab = tab.sort_values('skill', ascending=False, na_position='last').reset_index(drop=True)

    if verbose:
        n_th = int(tab['thermal'].sum())
        print(f'variant {variant}, {model} on {len(features)} thermal features, '
              f'{n_splits}-fold night-grouped')
        cut_s = 'nothing survives' if not np.isfinite(cut) else f'skill >= {cut:+.3f}'
        print(f'{n_th} of {n_modes} modes called thermal at FDR q={q} (dimensionless): {cut_s}')
        show = tab.head(8)
        print('  mode  skill  nMAD fit  nMAD null  n_visits  well-constrained')
        for _, r in show.iterrows():
            print(f'  v{int(r["mode"]):<4d} {r["skill"]:+.3f}  {r["nmad_fit"]:.4g}    '
                  f'{r["nmad_null"]:.4g}     {int(r["n_visits"]):<8d} '
                  f'{"yes" if r["well_constrained"] else "NO"}')
        print('  (skill = fractional reduction in out-of-fold residual nMAD against an '
              'intercept-only null, dimensionless; nMAD in dimensionless v-mode amplitude)')
    return tab


def noise_floor_table(df, n_modes=N_MODES, verbose=True):
    """Per-mode measurement scatter against night-to-night scatter.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying the ``v*_olr`` columns, ``day_obs`` and ``seq_num``.
    n_modes : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        Per mode: ``within_night`` and ``between_night`` nMAD [dimensionless v-mode amplitude],
        and ``signal_to_noise`` (between over within, dimensionless).

    Notes
    -----
    Within-night scatter is the nMAD of successive-visit differences divided by sqrt(2), which
    estimates the per-visit measurement error without assuming the state is constant across a
    night: a slow thermal drift contributes to the between-night term instead. A mode whose
    ``signal_to_noise`` is near 1 carries no more night-to-night structure than its own
    measurement noise, so a thermal slope fitted to it is not interpretable however it scores.
    """
    out = []
    for k in range(1, n_modes + 1):
        col = f'v{k}_olr'
        if col not in df.columns:
            continue
        d = df[['day_obs', 'seq_num', col]].dropna().sort_values(['day_obs', 'seq_num'])
        within = []
        for _, g in d.groupby('day_obs'):
            v = g[col].to_numpy(float)
            if len(v) > 2:
                within.append(np.diff(v) / np.sqrt(2.0))
        w = nmad(np.concatenate(within)) if within else np.nan
        per_night = d.groupby('day_obs')[col].median().to_numpy(float)
        b = nmad(per_night) if len(per_night) > 2 else np.nan
        out.append(dict(mode=k, within_night=w, between_night=b,
                        signal_to_noise=np.nan if not np.isfinite(w) or w == 0 else b / w))
    tab = pd.DataFrame(out)
    if verbose:
        ok = tab[tab['signal_to_noise'] > 2.0]['mode']
        hi = int(ok.max()) if len(ok) else 0
        print('per-mode scatter [dimensionless v-mode amplitude]:')
        for _, r in tab.iterrows():
            if int(r['mode']) <= 14 or int(r['mode']) == n_modes:
                print(f'  v{int(r["mode"]):<3d} within-night {r["within_night"]:.4g}  '
                      f'between-night {r["between_night"]:.4g}  ratio '
                      f'{r["signal_to_noise"]:.2f} (dimensionless)')
        print(f'  highest mode with between/within > 2 (dimensionless): v{hi}')
    return tab


def intrinsic_comparison(tab_batoid, tab_miw, verbose=True):
    """Per-mode thermal skill on the two intrinsic routes, side by side.

    Parameters
    ----------
    tab_batoid, tab_miw : `pandas.DataFrame`
        `mode_table` results for `INTRINSIC_PAIR`, both unconstrained 50/34.
    verbose : `bool`, optional

    Returns
    -------
    cmp : `pandas.DataFrame`
        ``mode``, ``skill_batoid``, ``skill_miw`` and their difference (all dimensionless), plus
        the ``thermal`` flag from each route.

    Notes
    -----
    The intrinsic enters the recovery as ``Deviation = OPD - intrinsic``, and the MIW differs
    from the batoid prediction by a static offset per rotator angle. A static offset moves a
    fitted intercept, not a thermal slope, so **the expectation is that the two routes agree on
    which modes are thermal**. A mode where they disagree is either dominated by the
    rotator-angle-dependent part of the MIW-batoid difference -- which correlates with elevation
    through the observing pattern, and so with temperature -- or is marginal in both.
    """
    cmp = tab_batoid[['mode', 'skill', 'thermal']].merge(
        tab_miw[['mode', 'skill', 'thermal']], on='mode', suffixes=('_batoid', '_miw'))
    cmp['skill_diff'] = cmp['skill_miw'] - cmp['skill_batoid']
    cmp = cmp.sort_values('mode').reset_index(drop=True)
    if verbose:
        dis = cmp[cmp['thermal_batoid'] != cmp['thermal_miw']]
        print(f'intrinsic routes agree on {len(cmp) - len(dis)} of {len(cmp)} modes')
        if len(dis):
            print('  disagreeing modes (thermal on one route only):')
            for _, r in dis.iterrows():
                print(f'    v{int(r["mode"]):<3d} skill batoid {r["skill_batoid"]:+.3f} '
                      f'miw {r["skill_miw"]:+.3f} (dimensionless)')
        print(f'  median |skill difference| over all modes: '
              f'{np.nanmedian(np.abs(cmp["skill_diff"])):.4f} (dimensionless)')
    return cmp
