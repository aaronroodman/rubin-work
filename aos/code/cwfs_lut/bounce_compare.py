"""Compare the survey pointing trends against the measured AOS bounce test.

The bounce test throws the telescope between two pointings and measures the change in recovered
optical state from Full Array Mode (FAM) wavefronts across the whole focal plane. This study
measures the same dependence from four corner sensors over the science survey. Both are
estimates of how the optical state varies with pointing, from different data and different
retrievals, so comparing them tests whether a look-up table fitted from ordinary science visits
reproduces the dedicated campaign.

Three things have to be got right, and each one changes the answer.

Match the arm, not just the solver
----------------------------------
The bounce Δ is a paired difference of the recovered **deviation** (`bounce_lib.paired_delta`
over `invert_truncated`), and never subtracts the Trim. The stored ``dof*_olr`` is the
**open-loop** state, ``Deviation - Trim``. Comparing one against the other mixes two quantities
that differ by the Trim. It costs real agreement: the lateral M2-plus-camera dx sum matches the
bounce to 3% on the deviation arm against 17% on the open-loop arm. So `SURVEY_ARMS` carries
both and ``deviation`` is the default.

Units: nothing converts
-----------------------
Both sides store the four hexapod tilts in deg. `ofc_svd.DOF_UNITS_50` labels them arcsec and
the bounce ``unit`` column copies that label without scaling any value, so the label is wrong
and the numbers are not. `load_bounce_stats` therefore **drops that column** and reattaches
units from `cwfs_lut_lib.dof_label`, which is what stops the mislabel propagating into a
comparison.

Three spaces, because per-DOF alone is degenerate
-------------------------------------------------
The two retrievals put the same lateral wavefront into different hexapods: on the rotator leg
the individual hexapod ratios run 0.03 to 5.3 (dimensionless, survey over bounce) while the
M2-plus-camera sums agree to a few percent. The bounce note says so directly -- the rigid-body
split is not uniquely pinned by either retrieval, while the rigid-body wavefront is preserved.
So the comparison is built per DOF (the literal claim), on the lateral sums (invariant to the
split), and as a cosine similarity per subspace (the aggregate).

The subspace result is the headline: the six hexapod translations agree at cosine +0.827
(dimensionless) and the 40 bending modes at +0.057, even though 25 of those 40 are individually
significant above 3 sigma on the bounce side. That **measures** the field-sampling limit the
study previously only asserted. It also makes v-mode space the dirtiest test rather than the
cleanest -- the v-mode basis is bending-dominated, so it inherits the bending disagreement.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                   # -> cwfs_lut_lib
_ROOT = _HERE.parents[2]                                         # repo root
sys.path.insert(0, str(_ROOT))

import cwfs_lut_lib as C                                         # noqa: E402

#: Bounce run whose products are read. The lead result of
#: ``notes/aos-bounce-test-summary/note.md``: the measured-intrinsic-referenced fit over the
#: April-to-July nights, which covers the most legs.
DEFAULT_BOUNCE_RUN = 'danish_1_2_A_50_34_i_5rot_july'

#: Bounce program to the pointing angle it throws.
BOUNCE_ANGLE = {'T724_rotator': 'rotator_angle_deg',
                'T720_elevation': 'elevation_deg'}

#: Bounce recovery arm to its (value, error) column pair. ``svd`` is the unconstrained truncated
#: inversion, ``rbr`` the range-bounded one.
#:
#: ``svd`` is the headline. The bounce note is explicit that for a look-up-table fit, which wants
#: the amplitudes themselves, the default recovery remains the estimator, and that an RBR
#: rigid-body amplitude is a constrained estimate that should not be quoted as a measurement of
#: hexapod motion. ``rbr`` is carried so the matched-solver pair can be reported.
BOUNCE_ARMS = {'svd': ('delta', 'delta_err'),
               'rbr': ('delta_rbr', 'delta_rbr_err')}

#: Survey arm to its (DOF, v-mode) stored-column templates. ``deviation`` is the arm the bounce Δ
#: is built from; ``open_loop`` is ``Deviation - Trim`` and differs from it by the Trim.
SURVEY_ARMS = {'deviation': ('dof{j}', 'v{k}'),
               'open_loop': ('dof{j}_olr', 'v{k}_olr')}

#: The six hexapod translations, where the two retrievals actually agree (cosine similarity
#: +0.827, dimensionless, on the rotator leg against +0.057 on the 40 bending modes).
HEXAPOD_TRANSLATION_DOF = (0, 1, 2, 5, 6, 7)

#: Lateral axes as (name, M2 DOF, camera DOF). Summing the two hexapods is invariant to how a
#: retrieval splits a lateral wavefront between them, which per-DOF comparison is not.
LATERAL_SUM_AXES = (('dz', 0, 5), ('dx', 1, 6), ('dy', 2, 7))

#: Subspaces the cosine similarity is reported over, as (name, DOF indices).
DOF_SUBSPACES = (('hexapod_translation', HEXAPOD_TRANSLATION_DOF),
                 ('hexapod_tilt', C.HEX_TILT_DOF),
                 ('bending', tuple(range(10, 50))),
                 ('all_dof', tuple(range(50))))

#: |Δ|/error (dimensionless) above which a bounce entry carries information. Below it the bounce
#: side is noise and would only dilute an aggregate statistic toward zero.
MIN_SIGNIFICANCE = 3.0


def load_bounce_stats(run=DEFAULT_BOUNCE_RUN, root=None):
    """Read the bounce test's per-DOF and per-v-mode delta table.

    Parameters
    ----------
    run : `str`, optional
        Bounce run directory under ``aos/output/bounce``.
    root : `pathlib.Path`, optional
        Repository root; defaults to the one this file sits under.

    Returns
    -------
    stats : `pandas.DataFrame`
        One row per (bounce, leg, night, kind, index), carrying ``delta`` and ``delta_rbr`` in
        each entry's own unit, their errors, ``significance`` (dimensionless), the throw
        geometry and ``n_pairs``.

    Raises
    ------
    FileNotFoundError
        Naming the expected path. The bounce run is another study's product and may not exist
        in a given checkout.

    Notes
    -----
    The file's ``unit`` column is a copy of `ofc_svd.DOF_UNITS_50` and labels the four hexapod
    tilt DOF arcsec, while no value in the table is scaled to match -- the tilts are deg, the
    same as the stored survey state. The column is **dropped** here and units are reattached
    from `cwfs_lut_lib.dof_label`, so the mislabel cannot reach a comparison even though it
    remains in the file.
    """
    root = pathlib.Path(root) if root else _ROOT
    path = root / 'aos' / 'output' / 'bounce' / run / 'bounce_dof_stats.parquet'
    if not path.exists():
        raise FileNotFoundError(
            f'bounce_dof_stats.parquet not found at {path}; the bounce study produces it, '
            f'see aos/code/bounce/run_bounce.py')
    stats = pd.read_parquet(path)
    return stats.drop(columns=['unit'], errors='ignore')


def _entry_label(kind, index):
    """Name and unit of one bounce entry, from this study's own labels."""
    if kind.startswith('vmode'):
        return f'v{int(index) + 1}', 'dimensionless'
    return C.dof_label(int(index))


def bounce_legs(stats, bounce='T724_rotator', kind='dof', night='all'):
    """One bounce program's legs, with the pointing throw of each.

    Parameters
    ----------
    stats : `pandas.DataFrame`
        `load_bounce_stats` result.
    bounce : `str`, optional
        Key of `BOUNCE_ANGLE`.
    kind : `str`, optional
        Recovery space, ``'dof'`` or ``'vmode'`` and the reduced variants.
    night : `str`, optional
        ``'all'`` for the pooled result, or one ``day_obs`` as a string.

    Returns
    -------
    legs : `pandas.DataFrame`
        One row per leg: ``comparison``, ``n_visits``, ``n_pairs``, ``throw_deg`` [deg of the
        thrown angle, comparison minus reference] and ``cross_throw_deg`` [deg, the other
        angle's change over the same leg].

    Notes
    -----
    ``cross_throw_deg`` is the leg's cleanliness. The rotator leg pins elevation to 0.003 deg
    across a 60.017 deg rotator throw, so its Δ is rotator-only; the elevation legs hold the
    rotator inside 1 deg.
    """
    thrown = BOUNCE_ANGLE[bounce]
    other = ('rotator_angle_deg' if thrown == 'elevation_deg' else 'elevation_deg')
    col = {'elevation_deg': ('elevation_deg', 'ref_elevation_deg'),
           'rotator_angle_deg': ('rot_angle_deg', 'ref_rot_angle_deg')}
    t_now, t_ref = col[thrown]
    o_now, o_ref = col[other]

    sel = stats[(stats['bounce'] == bounce) & (stats['kind'] == kind)
                & (stats['night'].astype(str) == str(night))]
    rows = []
    for comparison, g in sel.groupby('comparison', sort=False):
        r = g.iloc[0]
        rows.append(dict(comparison=comparison,
                         n_visits=int(r['n_visits']), n_pairs=int(r['n_pairs']),
                         throw_deg=float(r[t_now]) - float(r[t_ref]),
                         cross_throw_deg=float(r[o_now]) - float(r[o_ref])))
    return pd.DataFrame(rows).sort_values('throw_deg').reset_index(drop=True)


def bounce_slope(stats, bounce='T724_rotator', kind='dof', arm='svd', night='all'):
    """Bounce Δ divided by its pointing throw, as a slope per deg.

    Parameters
    ----------
    stats : `pandas.DataFrame`
        `load_bounce_stats` result.
    bounce, kind, night : `str`, optional
        As `bounce_legs`.
    arm : `str`, optional
        Key of `BOUNCE_ARMS`.

    Returns
    -------
    tab : `pandas.DataFrame`
        One row per (leg, entry): ``comparison``, ``throw_deg``, ``n_pairs``, ``index``,
        ``label``, ``unit``, ``slope`` [entry unit per deg of the thrown angle],
        ``slope_err``, ``delta``, ``delta_err`` and ``significance`` (dimensionless).

    Raises
    ------
    ValueError
        If `arm` is unknown, or if its value column is entirely absent for this `kind`. RBR is
        solved in DOF space only, so ``delta_rbr`` is all-NaN for every ``vmode`` and reduced
        kind; returning a table of NaN there would read as a measured null.

    Notes
    -----
    Dividing one leg's Δ by its throw is a slope only if the dependence is linear across that
    throw. For the rotator leg -- a single +60.017 deg throw -- that is the only slope the data
    define. For the five elevation legs the assumption is testable, and `compare_elevation_legs`
    is where it gets tested rather than assumed.

    The throw is a difference of telemetry angles whose own error is negligible against the Δ
    error, so ``slope_err`` is ``delta_err`` over the absolute throw.
    """
    if arm not in BOUNCE_ARMS:
        raise ValueError(f'unknown bounce arm {arm!r}; expected one of {sorted(BOUNCE_ARMS)}')
    vcol, ecol = BOUNCE_ARMS[arm]
    legs = bounce_legs(stats, bounce=bounce, kind=kind, night=night).set_index('comparison')
    sel = stats[(stats['bounce'] == bounce) & (stats['kind'] == kind)
                & (stats['night'].astype(str) == str(night))]
    if vcol not in sel.columns or not sel[vcol].notna().any():
        raise ValueError(
            f'bounce arm {arm!r} has no finite {vcol} for kind={kind!r}; range-bounded '
            f'recovery is solved in DOF space only, so use arm="svd" outside kind="dof"')

    rows = []
    for _, r in sel.iterrows():
        comparison = r['comparison']
        if comparison not in legs.index:
            continue
        throw = float(legs.loc[comparison, 'throw_deg'])
        if not np.isfinite(throw) or throw == 0.0:
            continue
        label, unit = _entry_label(kind, r['index'])
        delta, derr = float(r[vcol]), float(r[ecol])
        rows.append(dict(comparison=comparison, throw_deg=throw,
                         n_pairs=int(legs.loc[comparison, 'n_pairs']),
                         index=int(r['index']), label=label, unit=unit,
                         slope=delta / throw, slope_err=abs(derr / throw),
                         delta=delta, delta_err=derr,
                         significance=float(r['significance'])))
    return pd.DataFrame(rows)


def survey_slopes(df, angle_col, arm='deviation', kind='dof', indices=None, verbose=False):
    """Huber slope of each stored DOF or v-mode against one pointing angle.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit frame carrying `angle_col` and the arm's stored columns.
    angle_col : `str`
        ``'elevation_deg'`` or ``'rotator_angle_deg'``.
    arm : `str`, optional
        Key of `SURVEY_ARMS`.
    kind : `str`, optional
        ``'dof'`` for the 50 DOF, ``'vmode'`` for the 34 v-modes.
    indices : `iterable` [`int`], optional
        0-based entries to fit; all of them by default.
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        One row per entry: ``index``, ``label``, ``unit``, then the `cwfs_lut_lib.huber_trend`
        keys -- ``slope`` [entry unit per deg], ``slope_err``, ``n``, ``span`` [deg],
        ``resid_nmad``, ``pearson_r`` and ``spearman_rho``.

    Notes
    -----
    Differs from `cwfs_lut_lib.trend_table` in the two ways the bounce comparison needs: it
    reaches the **deviation** arm as well as the open-loop one, and it covers all 50 DOF and all
    34 v-modes rather than the ten rigid-body axes.

    ``index`` is 0-based throughout, to match the bounce table's ``index``. The stored v-mode
    columns are 1-based, so entry 0 is ``v1``; getting that wrong rotates the whole v-mode
    comparison by one mode without raising.
    """
    if arm not in SURVEY_ARMS:
        raise ValueError(f'unknown survey arm {arm!r}; expected one of {sorted(SURVEY_ARMS)}')
    if angle_col not in df.columns:
        raise KeyError(f'{angle_col} absent; the pointing columns arrived in 090b185')
    dof_t, vmode_t = SURVEY_ARMS[arm]
    n_entries = 34 if kind.startswith('vmode') else 50
    if indices is None:
        indices = range(n_entries)

    x = df[angle_col].to_numpy(float)
    rows = []
    for i in indices:
        col = (vmode_t.format(k=int(i) + 1) if kind.startswith('vmode')
               else dof_t.format(j=int(i)))
        if col not in df.columns:
            continue
        label, unit = _entry_label(kind, i)
        rows.append(dict(index=int(i), label=label, unit=unit,
                         **C.huber_trend(x, df[col].to_numpy(float))))
    tab = pd.DataFrame(rows)
    if verbose:
        print(f'survey {arm} {kind} against {angle_col}: {len(tab)} entries, '
              f'n={len(df)} visits')
    return tab


def compare_per_dof(bounce_tab, survey_tab):
    """Entry-by-entry slope comparison, bounce against survey.

    Parameters
    ----------
    bounce_tab : `pandas.DataFrame`
        `bounce_slope` result for one leg.
    survey_tab : `pandas.DataFrame`
        `survey_slopes` result for the matching angle, arm and kind.

    Returns
    -------
    cmp : `pandas.DataFrame`
        ``index``, ``label``, ``unit``, ``slope_bounce``, ``slope_bounce_err``,
        ``slope_survey``, ``slope_survey_err``, ``difference`` [entry unit per deg, survey minus
        bounce], ``difference_sigma`` (dimensionless, the difference over the quadrature sum of
        the two errors), ``ratio`` (dimensionless, survey over bounce) and ``significance``.

    Notes
    -----
    ``ratio`` is reported but is the weaker statistic: on the rotator leg the individual hexapod
    ratios span 0.03 to 5.3 (dimensionless) while the lateral sums agree to a few percent,
    because the two retrievals split one lateral wavefront differently between the two
    hexapods. Read `compare_lateral_sums` before reading a per-DOF ratio.
    """
    b = bounce_tab.set_index('index')
    s = survey_tab.set_index('index')
    rows = []
    for i in sorted(set(b.index) & set(s.index)):
        sb, eb = float(b.loc[i, 'slope']), float(b.loc[i, 'slope_err'])
        ss, es = float(s.loc[i, 'slope']), float(s.loc[i, 'slope_err'])
        denom = float(np.hypot(eb, es))
        rows.append(dict(index=int(i), label=b.loc[i, 'label'], unit=b.loc[i, 'unit'],
                         slope_bounce=sb, slope_bounce_err=eb,
                         slope_survey=ss, slope_survey_err=es,
                         difference=ss - sb,
                         difference_sigma=(ss - sb) / denom if denom > 0 else np.nan,
                         ratio=ss / sb if sb != 0 else np.nan,
                         significance=float(b.loc[i, 'significance'])))
    return pd.DataFrame(rows)


def compare_lateral_sums(stats, df, angle_col, bounce='T724_rotator', bounce_arm='svd',
                         survey_arm='deviation', night='all'):
    """Per-axis sum over the two hexapods, the degeneracy-robust comparison.

    Parameters
    ----------
    stats : `pandas.DataFrame`
        `load_bounce_stats` result.
    df : `pandas.DataFrame`
        Per-visit frame.
    angle_col : `str`
        Pointing angle to fit the survey side against.
    bounce, night : `str`, optional
        As `bounce_legs`.
    bounce_arm, survey_arm : `str`, optional
        Keys of `BOUNCE_ARMS` and `SURVEY_ARMS`.

    Returns
    -------
    cmp : `pandas.DataFrame`
        One row per entry of `LATERAL_SUM_AXES`: ``axis``, ``comparison``, ``slope_bounce``,
        ``slope_bounce_err``, ``slope_survey``, ``slope_survey_err``, ``difference``,
        ``difference_sigma``, ``ratio`` -- all slopes in µm per deg of the thrown angle.

    Notes
    -----
    Summed on the **per-visit state** survey-side, so the Huber fit sees the sum and the
    reported error is that fit's own; adding two fitted slopes instead would need their
    covariance, which the per-entry tables do not carry.

    Bounce-side errors are added in quadrature, which assumes the two hexapods' recovery errors
    are independent. The degeneracy makes them correlated, so that bar is an **upper estimate**
    rather than a confidence interval.
    """
    dof_t, _ = SURVEY_ARMS[survey_arm]
    bt = bounce_slope(stats, bounce=bounce, kind='dof', arm=bounce_arm, night=night)
    x = df[angle_col].to_numpy(float)

    rows = []
    for axis, j_m2, j_cam in LATERAL_SUM_AXES:
        c_m2, c_cam = dof_t.format(j=j_m2), dof_t.format(j=j_cam)
        if c_m2 not in df.columns or c_cam not in df.columns:
            continue
        res = C.huber_trend(x, df[c_m2].to_numpy(float) + df[c_cam].to_numpy(float))
        for comparison, g in bt.groupby('comparison', sort=False):
            by = g.set_index('index')
            if j_m2 not in by.index or j_cam not in by.index:
                continue
            sb = float(by.loc[j_m2, 'slope']) + float(by.loc[j_cam, 'slope'])
            eb = float(np.hypot(by.loc[j_m2, 'slope_err'], by.loc[j_cam, 'slope_err']))
            ss, es = float(res['slope']), float(res['slope_err'])
            denom = float(np.hypot(eb, es))
            rows.append(dict(axis=axis, comparison=comparison,
                             slope_bounce=sb, slope_bounce_err=eb,
                             slope_survey=ss, slope_survey_err=es,
                             difference=ss - sb,
                             difference_sigma=(ss - sb) / denom if denom > 0 else np.nan,
                             ratio=ss / sb if sb != 0 else np.nan))
    return pd.DataFrame(rows)


def cosine_similarity(a, b):
    """Normalized inner product of two slope vectors (dimensionless).

    Parameters
    ----------
    a, b : `array_like` [`float`]
        Same-length slope vectors. Non-finite pairs are dropped.

    Returns
    -------
    cos : `float`
        ``+1`` when the two point the same way, ``-1`` when opposed, ``0`` when orthogonal;
        NaN if either vector has no length.

    Notes
    -----
    Answers "do the two retrievals point the same way", which is the question the rigid-body
    degeneracy makes interesting -- it is invariant to overall scale, so a retrieval that gets
    every axis right but half as large still scores +1. The amplitude is reported separately as
    the least-squares ``scale``.
    """
    av, bv = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[m], bv[m]
    na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return np.nan
    return float(av @ bv / (na * nb))


def compare_subspaces(bounce_tab, survey_tab, subspaces=DOF_SUBSPACES,
                      min_significance=MIN_SIGNIFICANCE):
    """Aggregate agreement per subspace, as a cosine similarity.

    Parameters
    ----------
    bounce_tab : `pandas.DataFrame`
        `bounce_slope` result for one leg.
    survey_tab : `pandas.DataFrame`
        `survey_slopes` result for the matching angle, arm and kind.
    subspaces : `sequence` [`tuple`], optional
        ``(name, indices)`` pairs; `DOF_SUBSPACES` by default. Pass a single
        ``('vmode', range(34))`` entry for the v-mode space.
    min_significance : `float`, optional
        Bounce |Δ|/error floor for an entry to be included.

    Returns
    -------
    agr : `pandas.DataFrame`
        One row per subspace: ``subspace``, ``n_terms``, ``n_significant``,
        ``cosine_similarity`` (dimensionless), ``scale`` (dimensionless, the least-squares
        survey-over-bounce amplitude ratio), ``norm_bounce`` and ``norm_survey`` (slope-vector
        2-norms, in the subspace's unit per deg).

    Notes
    -----
    Only entries clearing `min_significance` on the bounce side enter: an insignificant bounce Δ
    is noise and would pull the similarity toward zero regardless of what the survey says.

    ``all_dof`` sums mixed units and is therefore blunt -- the hexapod translations in µm
    dominate the norm, so it is in practice the translation result. The per-subspace rows are
    the ones to read.
    """
    b = bounce_tab.set_index('index')
    s = survey_tab.set_index('index')
    shared = sorted(set(b.index) & set(s.index))
    rows = []
    for name, idx in subspaces:
        keep = [i for i in shared if i in set(int(j) for j in idx)]
        n_terms = len(keep)
        keep = [i for i in keep if abs(float(b.loc[i, 'significance'])) >= min_significance]
        bv = np.array([float(b.loc[i, 'slope']) for i in keep])
        sv = np.array([float(s.loc[i, 'slope']) for i in keep])
        m = np.isfinite(bv) & np.isfinite(sv)
        bv, sv = bv[m], sv[m]
        denom = float(bv @ bv)
        rows.append(dict(subspace=name, n_terms=n_terms, n_significant=int(len(bv)),
                         cosine_similarity=cosine_similarity(bv, sv),
                         scale=float(bv @ sv / denom) if denom > 0 else np.nan,
                         norm_bounce=float(np.linalg.norm(bv)),
                         norm_survey=float(np.linalg.norm(sv))))
    return pd.DataFrame(rows)


def _wls_through_origin(x, y, err):
    """Weighted least-squares slope with no intercept, plus chi2/dof.

    A leg of zero throw produces zero Δ by construction, so the line passes through the
    origin and the five legs constrain one parameter.
    """
    x, y, err = np.asarray(x, float), np.asarray(y, float), np.asarray(err, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    x, y, err = x[m], y[m], err[m]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, 0
    w = 1.0 / err ** 2
    denom = float(w @ (x * x))
    if denom <= 0:
        return np.nan, np.nan, np.nan, 0
    slope = float(w @ (x * y) / denom)
    slope_err = float(np.sqrt(1.0 / denom))
    dof = len(x) - 1
    chi2 = float(w @ (y - slope * x) ** 2)
    return slope, slope_err, (chi2 / dof if dof > 0 else np.nan), dof


def compare_elevation_legs(stats, df, bounce_arm='svd', survey_arm='deviation',
                           dof_indices=HEXAPOD_TRANSLATION_DOF, night='all'):
    """The five elevation legs as points, with a first-order linear fit through them.

    Parameters
    ----------
    stats : `pandas.DataFrame`
        `load_bounce_stats` result.
    df : `pandas.DataFrame`
        Per-visit frame.
    bounce_arm, survey_arm : `str`, optional
        Keys of `BOUNCE_ARMS` and `SURVEY_ARMS`.
    dof_indices : `iterable` [`int`], optional
        DOF to report.
    night : `str`, optional

    Returns
    -------
    legs : `pandas.DataFrame`
        One row per (leg, DOF): ``index``, ``label``, ``unit``, ``comparison``, ``throw_deg``,
        ``n_pairs``, ``delta`` [DOF unit], ``delta_err``, ``slope`` [DOF unit per deg],
        ``slope_err`` and ``significance``.
    fits : `pandas.DataFrame`
        One row per DOF: ``index``, ``label``, ``unit``, ``slope_linear`` [DOF unit per deg] and
        ``slope_linear_err`` from a weighted fit of Δ against throw through the origin,
        ``chi2_linear`` and ``chi2_cos`` as chi2/dof with ``dof`` given, and ``slope_survey``
        over the same elevation range.

    Notes
    -----
    Weighted least squares on the per-leg errors, not Huber: five points with known unequal
    errors is a weighting problem, and outlier rejection has nothing to act on.

    **A chi2/dof above 1 here is not grounds to reject the linear form.** The bounce test
    carries atmospheric turbulence and other stochastic terms that the per-leg error bars do
    not capture, so a perfect chi2/dof is not expected. The linear term is a first-order
    approximation to a dependence that is physically closer to cos(elevation), and it is much
    better than no correction -- ``slope_linear`` is the deliverable and the two chi2/dof
    columns are honest scatter indicators beside it.

    ``chi2_cos`` fits Δ against ``cos(elevation) - cos(ref_elevation)`` instead, for comparison;
    neither form reaches chi2/dof near 1 on the strongest axes.
    """
    bounce = 'T720_elevation'
    bt = bounce_slope(stats, bounce=bounce, kind='dof', arm=bounce_arm, night=night)
    sel = stats[(stats['bounce'] == bounce) & (stats['kind'] == 'dof')
                & (stats['night'].astype(str) == str(night))]
    elev = sel.drop_duplicates('comparison').set_index('comparison')
    surv = survey_slopes(df, 'elevation_deg', arm=survey_arm, kind='dof',
                         indices=dof_indices).set_index('index')

    legs = bt[bt['index'].isin(list(dof_indices))].copy().reset_index(drop=True)
    rows = []
    for i in dof_indices:
        g = legs[legs['index'] == i]
        if not len(g):
            continue
        throw = g['throw_deg'].to_numpy(float)
        delta = g['delta'].to_numpy(float)
        derr = g['delta_err'].to_numpy(float)
        slope, slope_err, chi2_lin, dof_lin = _wls_through_origin(throw, delta, derr)
        # Gravity loading on a lateral axis goes as cos(elevation), so the same five deltas are
        # also fitted against the cosine difference -- the comparison that says whether the
        # linear form is costing anything over the physically motivated one.
        ref = np.array([float(elev.loc[c, 'ref_elevation_deg']) for c in g['comparison']])
        now = np.array([float(elev.loc[c, 'elevation_deg']) for c in g['comparison']])
        dcos = np.cos(np.radians(now)) - np.cos(np.radians(ref))
        _, _, chi2_cos, dof_cos = _wls_through_origin(dcos, delta, derr)
        label, unit = C.dof_label(int(i))
        rows.append(dict(index=int(i), label=label, unit=unit,
                         slope_linear=slope, slope_linear_err=slope_err,
                         chi2_linear=chi2_lin, dof_linear=dof_lin,
                         chi2_cos=chi2_cos, dof_cos=dof_cos,
                         slope_survey=float(surv.loc[i, 'slope']) if i in surv.index else np.nan,
                         slope_survey_err=(float(surv.loc[i, 'slope_err'])
                                           if i in surv.index else np.nan)))
    return legs, pd.DataFrame(rows)


def compare_all(stats, frames, angle_col, bounce='T724_rotator', night='all',
                min_significance=MIN_SIGNIFICANCE, verbose=True):
    """Every (bounce arm, survey arm, variant) pairing, in all three spaces.

    Parameters
    ----------
    stats : `pandas.DataFrame`
        `load_bounce_stats` result.
    frames : `dict` [`str`, `pandas.DataFrame`]
        Per-visit frames keyed by variant id, as `run_cwfs_lut.main` builds them.
    angle_col : `str`
        Pointing angle; must be the one `bounce` throws.
    bounce, night : `str`, optional
    min_significance : `float`, optional
    verbose : `bool`, optional

    Returns
    -------
    out : `dict` [`str`, `pandas.DataFrame`]
        Keyed ``'per_dof'``, ``'lateral_sum'``, ``'subspace'`` and ``'vmode'``. Each carries
        ``bounce_arm``, ``survey_arm`` and ``variant`` columns, so every pairing sits in one
        long frame rather than a nest of dicts.

    Notes
    -----
    All pairings are produced because the solver choice is not settled by this study, but the
    **headline is bounce ``svd`` against survey ``deviation`` on the unconstrained variant**,
    following the bounce note: for a look-up-table fit, which wants the amplitudes themselves,
    the default recovery remains the estimator. RBR-to-RBR is the matched-solver cross-check.

    The v-mode space is unconstrained-only on both sides, because RBR is solved in DOF space
    and the bounce table's ``delta_rbr`` is all-NaN for ``kind='vmode'``.
    """
    if BOUNCE_ANGLE[bounce] != angle_col:
        raise ValueError(f'{bounce} throws {BOUNCE_ANGLE[bounce]}, not {angle_col}')
    out = {k: [] for k in ('per_dof', 'lateral_sum', 'subspace', 'vmode')}

    for variant, df in frames.items():
        for b_arm in BOUNCE_ARMS:
            try:
                bt = bounce_slope(stats, bounce=bounce, kind='dof', arm=b_arm, night=night)
            except ValueError:
                continue
            for s_arm in SURVEY_ARMS:
                st = survey_slopes(df, angle_col, arm=s_arm, kind='dof')
                tag = dict(bounce_arm=b_arm, survey_arm=s_arm, variant=variant)
                out['per_dof'].append(compare_per_dof(bt, st).assign(**tag))
                out['subspace'].append(
                    compare_subspaces(bt, st, min_significance=min_significance).assign(**tag))
                out['lateral_sum'].append(
                    compare_lateral_sums(stats, df, angle_col, bounce=bounce,
                                         bounce_arm=b_arm, survey_arm=s_arm,
                                         night=night).assign(**tag))
        # V-mode space carries no RBR arm on the bounce side, so it is fitted once per survey
        # arm against the unconstrained bounce recovery.
        bv = bounce_slope(stats, bounce=bounce, kind='vmode', arm='svd', night=night)
        for s_arm in SURVEY_ARMS:
            sv = survey_slopes(df, angle_col, arm=s_arm, kind='vmode')
            tag = dict(bounce_arm='svd', survey_arm=s_arm, variant=variant)
            cmp_v = compare_per_dof(bv, sv).assign(**tag)
            agr = compare_subspaces(bv, sv, subspaces=(('vmode', tuple(range(34))),),
                                    min_significance=min_significance).assign(**tag)
            out['vmode'].append(cmp_v)
            out['subspace'].append(agr)

    res = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
           for k, v in out.items()}
    if verbose:
        print(f'bounce comparison on {bounce}, {angle_col}: '
              f'{len(res["per_dof"])} per-DOF rows, {len(res["vmode"])} v-mode rows, '
              f'{len(res["subspace"])} subspace rows over {len(frames)} variants')
    return res
