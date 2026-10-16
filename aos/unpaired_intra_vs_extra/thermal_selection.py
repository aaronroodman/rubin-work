"""Greedy forward selection of thermal predictors for a per-visit target.

The earlier blur study ranked candidate thermal quantities by their own correlation with
the target and added them in that order. The thermal quantities are themselves strongly
correlated, so that order double-counts: the second variable added may carry almost no
information the first did not.

This module instead re-ranks the remaining candidates at every step by what each one
*adds* given those already chosen, measured as the reduction in the robust (Huber)
residual scatter of the fit. Signs come from the first round, before any
orthogonalisation, so they describe each variable's own relation to the target.

Developed in the `unpaired_intra_vs_extra` study; intended to move to `rubinwork` once
it has been used on more than one target.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


def nmad(values):
    """Normalized median absolute deviation, the robust scatter.

    Parameters
    ----------
    values : `array_like`
        Values whose scatter is wanted, in any unit.

    Returns
    -------
    scatter : `float`
        1.4826 times the median absolute deviation about the median, in the unit of
        `values`.
    """
    values = np.asarray(values, dtype=float)
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def huber_fit(target, design):
    """Huber robust linear fit.

    Parameters
    ----------
    target : `numpy.ndarray`
        Target values, shape ``(n,)``, in the target's unit.
    design : `numpy.ndarray`
        Predictor columns, shape ``(n, k)``, each in its own unit. A constant is added.

    Returns
    -------
    result : `statsmodels.robust.robust_linear_model.RLMResults`
        The fit, whose ``fittedvalues`` are in the target's unit.
    """
    exog = sm.add_constant(np.asarray(design, dtype=float).reshape(len(target), -1))
    return sm.RLM(np.asarray(target, dtype=float), exog,
                  M=sm.robust.norms.HuberT()).fit()


def variance_inflation(design, column):
    """Variance inflation factor of one predictor against the others.

    Parameters
    ----------
    design : `numpy.ndarray`
        Predictor columns, shape ``(n, k)``.
    column : `int`
        Index of the predictor to test.

    Returns
    -------
    vif : `float`
        Variance inflation factor, dimensionless. Large values mean the predictor is
        nearly a linear combination of the others. Infinite when it is exactly so.
    """
    design = np.asarray(design, dtype=float)
    if design.shape[1] < 2:
        return 1.0
    others = np.delete(design, column, axis=1)
    exog = sm.add_constant(others)
    fit = sm.OLS(design[:, column], exog).fit()
    if fit.rsquared >= 1.0:
        return np.inf
    return 1.0 / (1.0 - fit.rsquared)


def select_predictors(target, predictors, n_max=5, min_gain_frac=0.01):
    """Choose thermal predictors by greedy forward selection on incremental gain.

    Step (a) ranks every candidate by its own Spearman correlation with the target and
    records its sign. Step (b) takes the leading candidate, then re-ranks the remaining
    ones by how much each reduces the Huber residual scatter *given* those already
    chosen, and takes the largest reduction. Step (c) repeats to `n_max`.

    Parameters
    ----------
    target : `pandas.Series`
        Per-visit target, in its own unit (arcsec for blur, micrometres of wavefront for
        a Zernike coefficient).
    predictors : `pandas.DataFrame`
        Candidate telemetry on the same index as `target`, each column in its own unit.
    n_max : `int`, optional
        Number of predictors to select.
    min_gain_frac : `float`, optional
        Stop early if the best remaining candidate reduces the residual scatter by less
        than this fraction (dimensionless) of the current scatter.

    Returns
    -------
    steps : `pandas.DataFrame`
        One row per selected predictor, in selection order: ``variable``,
        ``sign_round1`` (+1 or -1, from step (a)), ``rho_own`` (its own Spearman rho
        with the target, dimensionless), ``rho_partial`` (Spearman rho of the candidate
        against the current residual, dimensionless), ``nmad_after`` (residual scatter
        after including it, in the target's unit), ``gain_frac`` (fractional reduction,
        dimensionless) and ``vif`` (dimensionless).
    ranking : `pandas.DataFrame`
        The step (a) ranking of every candidate: ``rho_own`` and ``sign_round1``.
    fit : `statsmodels.robust.robust_linear_model.RLMResults`
        The final fit on the selected predictors.
    """
    frame = pd.concat([target.rename('_target'), predictors], axis=1).dropna()
    values = frame['_target'].to_numpy()
    candidates = [c for c in predictors.columns if c in frame.columns]

    # Step (a): own correlation and sign, before any orthogonalisation.
    own = {}
    for name in candidates:
        rho = spearmanr(frame[name].to_numpy(), values).statistic
        own[name] = 0.0 if not np.isfinite(rho) else rho
    ranking = pd.DataFrame(
        {'rho_own': pd.Series(own),
         'sign_round1': pd.Series({k: (1 if v >= 0 else -1) for k, v in own.items()})}
    )
    ranking = ranking.reindex(ranking['rho_own'].abs().sort_values(ascending=False).index)

    chosen = []
    scatter = nmad(values - np.median(values))
    steps = []
    fit = None

    for _ in range(min(n_max, len(candidates))):
        remaining = [c for c in ranking.index if c not in chosen]
        if not remaining:
            break

        residual = values - fit.fittedvalues if fit is not None else values - np.median(values)

        best = None
        for name in remaining:
            design = frame[chosen + [name]].to_numpy()
            try:
                trial = huber_fit(values, design)
            except Exception:
                continue
            trial_scatter = nmad(values - trial.fittedvalues)
            if best is None or trial_scatter < best[1]:
                best = (name, trial_scatter, trial)
        if best is None:
            break

        name, trial_scatter, trial = best
        gain_frac = (scatter - trial_scatter) / scatter if scatter > 0 else 0.0
        if chosen and gain_frac < min_gain_frac:
            break

        rho_partial = spearmanr(frame[name].to_numpy(), residual).statistic
        chosen.append(name)
        design = frame[chosen].to_numpy()
        steps.append({
            'variable': name,
            'sign_round1': int(ranking.loc[name, 'sign_round1']),
            'rho_own': float(ranking.loc[name, 'rho_own']),
            'rho_partial': float(rho_partial) if np.isfinite(rho_partial) else np.nan,
            'nmad_after': float(trial_scatter),
            'gain_frac': float(gain_frac),
            'vif': float(variance_inflation(design, len(chosen) - 1)),
        })
        scatter = trial_scatter
        fit = trial

    return pd.DataFrame(steps), ranking, fit
