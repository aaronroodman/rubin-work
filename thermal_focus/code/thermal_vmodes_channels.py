"""Per-channel thermal screen: every v-mode against every thermal telemetry line.

The all-v-mode study's first pass fitted five channels jointly per mode and reported one skill
number. That hid two things a per-channel view shows. Skill carries no sign, so v-modes 3 and 18
scored positively while *anti*-correlating with truss temperature; and a joint fit on five
channels cannot say which channel a mode responds to, which is the question a physical
explanation needs.

So this module fits **one channel at a time** -- a Huber robust line per (v-mode, channel) pair,
reporting Spearman rho alongside -- builds the full grid, and for any mode whose strongest
channel clears `RHO_STRONG` refits that mode on its leading `N_LEAD` channels together. The
before-and-after residual nMAD is what says whether the combination explains the mode.

The physical expectation this is testing, from ``smatrix/docs/plots.md``: the M1M3 z-gradient
drives Z4 defocus plus Z11 spherical, and the radial gradient drives Z4 + Z11 + Z22. So a v-mode
carrying spherical should respond to ``m1m3_z_gradient_c_per_m``, and one carrying Z22 to
``m1m3_radial_gradient_c_per_m``. A grid that shows those pairings is evidence for a thermal
origin; one that shows a mode responding only to a camera air temperature is not.

Nights are held out whole throughout, as everywhere in this study: only 2.7% of truss-temperature
variance is within-night, so a visit-level split leaks.
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
import thermal_vmodes as TV                                      # noqa: E402
from common.utils import nmad                                    # noqa: E402

#: Spearman rho magnitude (dimensionless) at which a (v-mode, channel) pair is followed up with
#: a combined fit. Aaron's threshold.
RHO_STRONG = 0.4

#: Channels combined for a mode that clears `RHO_STRONG`, taken in order of descending
#: |Spearman rho|.
N_LEAD = 4

#: Maximum |Spearman rho| between two channels before the weaker is treated as a duplicate of
#: the stronger and skipped when assembling a mode's leading set.
#:
#: Without this the "leading 4" for v-mode 1 are camera body Y-, Y+, X+ and L2 Y+, which
#: correlate with each other at rho 0.92 to 0.95: four measurements of one quantity. Fitting
#: them together *lost* skill (+0.740 on the single best channel against +0.694 on four), since
#: the Huber fit splits one real coefficient across four collinear columns and pays variance for
#: it. Skipping near-duplicates makes the leading set four distinct thermal drivers, which is
#: what the combination is meant to test.
CHANNEL_DUP_RHO = 0.9

#: Base thermal channels, with label and unit. Order sets the heatmap's column order: structure
#: temperature first, then M1M3 shape terms, then camera body and lenses, so physically related
#: columns sit together and a block of correlated columns is visible as a block.
BASE_CHANNELS = (
    ('truss_temp_mean_c', 'TMA truss mean', 'deg C'),
    ('m1m3_z_gradient_c_per_m', 'M1M3 z gradient', 'deg C/m'),
    ('m1m3_radial_gradient_c_per_m', 'M1M3 radial gradient', 'deg C/m'),
    ('m1m3_x_gradient_c_per_m', 'M1M3 x gradient', 'deg C/m'),
    ('m1m3_y_gradient_c_per_m', 'M1M3 y gradient', 'deg C/m'),
    ('m1m3_r2_coeff_c', 'M1M3 quadratic radial', 'deg C'),
    ('m1_r2_coeff_c', 'M1 quadratic radial', 'deg C'),
    ('m3_r2_coeff_c', 'M3 quadratic radial', 'deg C'),
    ('cam_AverageTemp', 'camera average', 'deg C'),
    ('cam_AmbAirtemp', 'ambient air', 'deg C'),
    ('cam_CamBodyXPlusTemp', 'camera body X+', 'deg C'),
    ('cam_CamBodyYPlusTemp', 'camera body Y+', 'deg C'),
    ('cam_CamBodyYMinusTemp', 'camera body Y-', 'deg C'),
    ('cam_CamHousXPlusTemp', 'camera housing X+', 'deg C'),
    ('cam_CamHousXMinusTemp', 'camera housing X-', 'deg C'),
    ('cam_CamHousYPlusTemp', 'camera housing Y+', 'deg C'),
    ('cam_CamHousYMinusTemp', 'camera housing Y-', 'deg C'),
    ('cam_L1XMinusTemp', 'L1 lens X-', 'deg C'),
    ('cam_L1YMinusTemp', 'L1 lens Y-', 'deg C'),
    ('cam_L2XPlusTemp', 'L2 lens X+', 'deg C'),
    ('cam_L2XMinusTemp', 'L2 lens X-', 'deg C'),
    ('cam_L2YPlusTemp', 'L2 lens Y+', 'deg C'),
)

#: Derived difference channels, as ``(name, minuend, subtrahend, label, unit)``. A difference is
#: carried as its own column because an absolute temperature and an excess over ambient are
#: different physical drivers: a uniform warming of camera and air together changes neither
#: spacing nor figure, while an excess does.
DIFF_CHANNELS = (
    ('cam_minus_ambient_c', 'cam_AverageTemp', 'cam_AmbAirtemp',
     'camera - ambient air', 'deg C'),
    ('truss_minus_ambient_c', 'truss_temp_mean_c', 'cam_AmbAirtemp',
     'truss - ambient air', 'deg C'),
    ('cam_minus_truss_c', 'cam_AverageTemp', 'truss_temp_mean_c',
     'camera - truss', 'deg C'),
    ('l1_minus_l2_x_c', 'cam_L1XMinusTemp', 'cam_L2XMinusTemp',
     'L1 - L2 at X-', 'deg C'),
    ('cam_hous_x_asym_c', 'cam_CamHousXPlusTemp', 'cam_CamHousXMinusTemp',
     'camera housing X+ - X-', 'deg C'),
    ('cam_hous_y_asym_c', 'cam_CamHousYPlusTemp', 'cam_CamHousYMinusTemp',
     'camera housing Y+ - Y-', 'deg C'),
)


#: Noll indices the corner recovery spans, in `corner_recovery_basis` row order.
NOLL_MIN, NOLL_MAX = 4, 24

#: Zernikes the M1M3 thermal prediction names, with the channel each should drive. From
#: ``smatrix/docs/plots.md``: the z-gradient gives Z4 defocus plus Z11 spherical, the radial
#: gradient gives Z4 + Z11 + Z22 at 0.0185 µm of Z22 per deg C/m.
PREDICTED_PAIRS = (
    (4, 'm1m3_z_gradient_c_per_m', 'defocus Z4 from the z-gradient'),
    (11, 'm1m3_z_gradient_c_per_m', 'spherical Z11 from the z-gradient'),
    (4, 'm1m3_radial_gradient_c_per_m', 'defocus Z4 from the radial gradient'),
    (11, 'm1m3_radial_gradient_c_per_m', 'spherical Z11 from the radial gradient'),
    (22, 'm1m3_radial_gradient_c_per_m', 'second spherical Z22 from the radial gradient'),
)


def vmode_zernike_content(n_modes=TV.N_MODES, dof_set='all_50'):
    """Field-averaged Zernike content of each v-mode, from the corner recovery basis.

    Parameters
    ----------
    n_modes : `int`, optional
        V-modes to report.
    dof_set : `str`, optional
        `aos_state.DOF_SETS` key; ``'all_50'`` is the 50/34 basis this study uses.

    Returns
    -------
    content : `pandas.DataFrame`
        Index ``noll`` (4 to 24), columns ``v1``..``v{n_modes}``, values the root-mean-square
        amplitude of that Noll term over the four corners [dimensionless, unit v-mode amplitude].

    Notes
    -----
    `aos_state.corner_recovery_basis` returns ``U`` as (84, 50): four corners by 21 Noll terms,
    corner-major. Squaring and averaging over corners gives each mode's field-averaged content,
    which is what to compare against a prediction phrased in Zernikes.

    Needed because the natural guess about which v-mode is which aberration is wrong. V-mode 12
    is **not** spherical -- it is dominated by Z15 -- and spherical Z11 lives mostly in v21 and
    v16, with Z22 in v14 and v31. Reading the thermal grid against the prediction requires this
    mapping rather than mode index.
    """
    sys.path.insert(0, str(_ROOT / 'aos' / 'code'))
    import aos_state

    se = aos_state.make_state_estimator(dof_set=dof_set)
    basis = aos_state.corner_recovery_basis(se)
    n_noll = NOLL_MAX - NOLL_MIN + 1
    u = basis['U'][:, :n_modes].reshape(4, n_noll, n_modes)
    rms = np.sqrt((u ** 2).mean(axis=0))
    return pd.DataFrame(rms, index=pd.Index(range(NOLL_MIN, NOLL_MAX + 1), name='noll'),
                        columns=[f'v{k}' for k in range(1, n_modes + 1)])


def modes_carrying(content, noll, top=3):
    """The v-modes with the most of one Noll term.

    Parameters
    ----------
    content : `pandas.DataFrame`
        `vmode_zernike_content` result.
    noll : `int`
        Noll index.
    top : `int`, optional

    Returns
    -------
    modes : `list` [`tuple`]
        ``(mode, amplitude)`` pairs, strongest first.
    """
    if noll not in content.index:
        return []
    row = content.loc[noll]
    order = row.sort_values(ascending=False).head(top)
    return [(int(k[1:]), float(v)) for k, v in order.items()]


def prediction_table(grid, content, pairs=PREDICTED_PAIRS, top=3, verbose=True):
    """Does the mode carrying a predicted Zernike respond to the predicted channel?

    Parameters
    ----------
    grid : `pandas.DataFrame`
        `channel_grid` result.
    content : `pandas.DataFrame`
        `vmode_zernike_content` result.
    pairs : `tuple`, optional
        As `PREDICTED_PAIRS`.
    top : `int`, optional
        V-modes per Zernike to test.
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        One row per (predicted Zernike, v-mode) pair: the mode's amplitude in that Noll term,
        and its Spearman rho and skill against the predicted channel.

    Notes
    -----
    This is the direct test of the S-matrix expectation. A prediction is supported when the modes
    that actually carry the Zernike show a correlation with the named channel; it fails when they
    do not, whatever other modes happen to correlate.
    """
    rows = []
    for noll, channel, label in pairs:
        for mode, amp in modes_carrying(content, noll, top=top):
            g = grid[(grid['mode'] == mode) & (grid['channel'] == channel)]
            r = g.iloc[0] if len(g) else None
            rows.append(dict(
                noll=noll, channel=channel, prediction=label, mode=mode,
                zernike_amplitude=amp,
                spearman_rho=np.nan if r is None else float(r['spearman_rho']),
                pearson_r=np.nan if r is None else float(r['pearson_r']),
                skill=np.nan if r is None else float(r['skill'])))
    tab = pd.DataFrame(rows)
    if verbose and len(tab):
        print('S-matrix prediction, tested on the modes that carry each Zernike:')
        for label, sub in tab.groupby('prediction', sort=False):
            got = ', '.join(f'v{int(r["mode"])} rho {r["spearman_rho"]:+.3f}'
                            for _, r in sub.iterrows())
            print(f'  {label:<46s} {got}')
    return tab


def attach_differences(df, diffs=DIFF_CHANNELS, verbose=True):
    """Add the derived difference columns, where both inputs are present.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit frame carrying the base channels.
    diffs : `tuple`, optional
        As `DIFF_CHANNELS`.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Copy with one column per computable difference [deg C].

    Notes
    -----
    A difference is skipped rather than filled when either input is absent from the frame, so a
    narrower load degrades the grid instead of raising.
    """
    out = df.copy()
    made = []
    for name, a, b, _label, _unit in diffs:
        if a in out.columns and b in out.columns:
            out[name] = out[a].to_numpy(float) - out[b].to_numpy(float)
            made.append(name)
    if verbose:
        print(f'derived {len(made)} of {len(diffs)} difference channels [deg C]: '
              f'{", ".join(made) if made else "none"}')
    return out


def channel_table(channels=BASE_CHANNELS, diffs=DIFF_CHANNELS):
    """Channel names, labels and units, base then derived.

    Parameters
    ----------
    channels : `tuple`, optional
    diffs : `tuple`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        ``channel``, ``label``, ``unit`` and ``kind`` (``base`` or ``difference``).
    """
    rows = [dict(channel=c, label=lab, unit=u, kind='base') for c, lab, u in channels]
    rows += [dict(channel=n, label=lab, unit=u, kind='difference')
             for n, _a, _b, lab, u in diffs]
    return pd.DataFrame(rows)


def single_channel_fit(df, mode, channel, n_splits=F.N_SPLITS):
    """Huber line and rank correlation for one v-mode against one thermal channel.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``v{mode}_olr``, `channel` and ``day_obs``.
    mode : `int`
        V-mode index, 1-based.
    channel : `str`
        Thermal column.
    n_splits : `int`, optional
        Night-grouped folds for the out-of-fold residual.

    Returns
    -------
    row : `dict`
        ``mode``, ``channel``, ``n`` visits, ``slope`` [dimensionless v-mode amplitude per
        channel unit] and its formal error, ``pearson_r`` and ``spearman_rho`` (dimensionless),
        ``nmad_null`` and ``nmad_fit`` [dimensionless v-mode amplitude], and ``skill``
        (dimensionless).

    Notes
    -----
    The correlations are **per-visit**, which is what makes a sign available; the nMAD pair is
    **night-grouped and out-of-fold**, which is what makes the improvement honest. Reporting both
    is deliberate: a mode can show a strong per-visit correlation driven by a handful of nights
    and gain nothing out of fold, and the two columns together say so.
    """
    import statsmodels.api as sm
    from scipy.stats import pearsonr, spearmanr

    nan = dict(mode=mode, channel=channel, n=0, slope=np.nan, slope_err=np.nan,
               pearson_r=np.nan, spearman_rho=np.nan, nmad_null=np.nan, nmad_fit=np.nan,
               skill=np.nan)
    if channel not in df.columns:
        return nan
    d = TV.attach_mode_response(df, mode)
    m = np.isfinite(d['y']) & np.isfinite(d[channel])
    d = d[m]
    if len(d) < 2 * n_splits or d['day_obs'].nunique() < n_splits:
        return dict(nan, n=len(d))

    x = d[channel].to_numpy(float)
    y = d['y'].to_numpy(float)
    if np.ptp(x) == 0:
        return dict(nan, n=len(d))

    fit = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    res = F.evaluate(d, [channel], model='huber', n_splits=n_splits, verbose=False)
    n0 = TV.null_nmad(d, n_splits=n_splits)
    return dict(mode=mode, channel=channel, n=len(d),
                slope=float(fit.params[1]), slope_err=float(fit.bse[1]),
                pearson_r=float(pearsonr(x, y)[0]),
                spearman_rho=float(spearmanr(x, y)[0]),
                nmad_null=n0, nmad_fit=res['nmad'],
                skill=np.nan if not np.isfinite(n0) or n0 == 0 else 1.0 - res['nmad'] / n0)


def channel_grid(df, channels=None, n_modes=TV.N_MODES, n_splits=F.N_SPLITS, verbose=True):
    """Every v-mode against every thermal channel, one channel at a time.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit frame with the ``v*_olr`` columns, the channels and ``day_obs``.
    channels : `list` [`str`], optional
        Defaults to every entry of `BASE_CHANNELS` and `DIFF_CHANNELS` present in `df`.
    n_modes : `int`, optional
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    grid : `pandas.DataFrame`
        One row per (mode, channel) pair, as `single_channel_fit`.
    """
    if channels is None:
        known = list(channel_table()['channel'])
        channels = [c for c in known if c in df.columns]
    if not channels:
        raise KeyError('no known thermal channel present in the frame; load_science must carry '
                       'them through TELEMETRY_COLS')

    rows = []
    for k in range(1, n_modes + 1):
        for c in channels:
            rows.append(single_channel_fit(df, k, c, n_splits=n_splits))
        if verbose and k % 5 == 0:
            print(f'  ... through v{k} of {n_modes}')
    grid = pd.DataFrame(rows)
    grid.attrs['channels'] = list(channels)
    grid.attrs['n_modes'] = int(n_modes)

    if verbose:
        best = leading_channels(grid, rho_strong=RHO_STRONG)
        print(f'{len(channels)} channels x {n_modes} modes = {len(grid)} single-channel fits')
        print(f'{len(best)} modes have a channel at |Spearman rho| >= {RHO_STRONG} '
              f'(dimensionless)')
        for _, r in best.iterrows():
            print(f'  v{int(r["mode"]):<3d} {r["channel"]:<32s} rho {r["spearman_rho"]:+.3f}  '
                  f'Pearson r {r["pearson_r"]:+.3f}  n {int(r["n"])} visits')
    return grid


def pivot_rho(grid, value='spearman_rho', channels=None):
    """Grid reshaped for a heatmap: modes down, channels across.

    Parameters
    ----------
    grid : `pandas.DataFrame`
        `channel_grid` result.
    value : `str`, optional
        Column to pivot.
    channels : `list` [`str`], optional
        Column order; defaults to the grid's own recorded order.

    Returns
    -------
    wide : `pandas.DataFrame`
        Index ``mode``, columns the channels, values `value`.
    """
    wide = grid.pivot(index='mode', columns='channel', values=value)
    order = channels or grid.attrs.get('channels') or list(wide.columns)
    return wide[[c for c in order if c in wide.columns]]


def leading_channels(grid, rho_strong=RHO_STRONG):
    """The strongest channel per mode, for modes that clear the threshold.

    Parameters
    ----------
    grid : `pandas.DataFrame`
        `channel_grid` result.
    rho_strong : `float`, optional
        Minimum |Spearman rho| (dimensionless).

    Returns
    -------
    best : `pandas.DataFrame`
        One row per qualifying mode, the single strongest pair, sorted by descending
        |Spearman rho|.
    """
    g = grid[np.isfinite(grid['spearman_rho'])].copy()
    if not len(g):
        return g
    g['abs_rho'] = g['spearman_rho'].abs()
    best = g.sort_values('abs_rho', ascending=False).groupby('mode', as_index=False).first()
    best = best[best['abs_rho'] >= rho_strong]
    return best.sort_values('abs_rho', ascending=False).reset_index(drop=True)


def select_lead(df, ranked, n_lead=N_LEAD, dup_rho=CHANNEL_DUP_RHO):
    """Take the strongest channels, skipping near-duplicates of ones already chosen.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit frame holding the candidate channels.
    ranked : `sequence` [`str`]
        Channel names in descending order of |Spearman rho| against the mode.
    n_lead : `int`, optional
        How many to return.
    dup_rho : `float`, optional
        |Spearman rho| between two channels above which the weaker is a duplicate.

    Returns
    -------
    lead : `list` [`str`]
        Up to `n_lead` channels, mutually below `dup_rho`.
    dropped : `list` [`tuple`]
        ``(skipped, kept, rho)`` for each rejection, so the choice is auditable.

    Notes
    -----
    Greedy and order-dependent by design: the strongest channel is always kept, and a later
    channel is admitted only if it brings something the kept ones do not. The 24 camera
    temperatures are nearly one signal, so without this a mode's "leading 4" can be four
    thermometers on the same structure.
    """
    from scipy.stats import spearmanr

    lead, dropped = [], []
    for c in ranked:
        if c not in df.columns:
            continue
        dup = False
        for k in lead:
            m = np.isfinite(df[c]) & np.isfinite(df[k])
            if m.sum() < 10:
                continue
            rho = spearmanr(df.loc[m, c], df.loc[m, k])[0]
            if np.isfinite(rho) and abs(rho) >= dup_rho:
                dropped.append((c, k, float(rho)))
                dup = True
                break
        if not dup:
            lead.append(c)
        if len(lead) >= n_lead:
            break
    return lead, dropped


def combined_fit(df, mode, grid, n_lead=N_LEAD, n_splits=F.N_SPLITS, dup_rho=CHANNEL_DUP_RHO,
                 verbose=True):
    """Refit one mode on its leading channels together, and report before against after.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit frame.
    mode : `int`
        V-mode index, 1-based.
    grid : `pandas.DataFrame`
        `channel_grid` result, used to rank this mode's channels.
    n_lead : `int`, optional
        How many channels to combine.
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    res : `dict`
        ``mode``, ``channels`` used, ``n`` visits, ``nmad_null``, ``nmad_single`` and
        ``nmad_combined`` [dimensionless v-mode amplitude], the two skills (dimensionless), and
        ``residual`` / ``response`` / ``prediction`` arrays for plotting.

    Notes
    -----
    The leading channels are chosen by |Spearman rho| and are often strongly correlated with each
    other -- the camera temperatures especially. That makes the individual combined coefficients
    unreliable while leaving the *fit* sound, so no per-channel coefficient is reported here. The
    question this answers is how much of the mode the thermal telemetry explains, not which
    channel deserves the credit.
    """
    g = grid[(grid['mode'] == mode) & np.isfinite(grid['spearman_rho'])].copy()
    if not len(g):
        return dict(mode=mode, channels=[], n=0, nmad_null=np.nan, nmad_single=np.nan,
                    nmad_combined=np.nan, skill_single=np.nan, skill_combined=np.nan)
    g['abs_rho'] = g['spearman_rho'].abs()
    g = g.sort_values('abs_rho', ascending=False)
    lead, dropped = select_lead(df, list(g['channel']), n_lead=n_lead, dup_rho=dup_rho)
    single = g.iloc[0]

    d = TV.attach_mode_response(df, mode)
    keep = np.isfinite(d['y'])
    for c in lead:
        keep &= np.isfinite(d[c])
    d = d[keep]
    if len(d) < 2 * n_splits or d['day_obs'].nunique() < n_splits:
        return dict(mode=mode, channels=lead, n=len(d), nmad_null=np.nan, nmad_single=np.nan,
                    nmad_combined=np.nan, skill_single=np.nan, skill_combined=np.nan)

    res = F.evaluate(d, lead, model='huber', n_splits=n_splits, verbose=False)
    n0 = TV.null_nmad(d, n_splits=n_splits)
    y = d['y'].to_numpy(float)

    # Out-of-fold prediction, rebuilt here so the scatter and histogram show the same held-out
    # quantity the nMAD reports rather than an in-sample fit.
    from sklearn.model_selection import GroupKFold
    import statsmodels.api as sm
    pred = np.full(len(d), np.nan)
    X = d[lead].to_numpy(float)
    groups = d['day_obs'].to_numpy()
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        f = sm.RLM(y[tr], sm.add_constant(X[tr]), M=sm.robust.norms.HuberT()).fit()
        pred[te] = f.predict(sm.add_constant(X[te]))

    out = dict(mode=mode, channels=lead, n=len(d), nmad_null=n0,
               nmad_single=float(single['nmad_fit']), nmad_combined=res['nmad'],
               skill_single=float(single['skill']),
               skill_combined=(np.nan if not np.isfinite(n0) or n0 == 0
                               else 1.0 - res['nmad'] / n0),
               lead_channel=str(single['channel']),
               lead_rho=float(single['spearman_rho']),
               n_duplicates_skipped=len(dropped),
               response=y, prediction=pred, residual=y - pred,
               day_obs=d['day_obs'].to_numpy())
    if verbose:
        print(f'v{mode}: {len(lead)} channels {lead}')
        if dropped:
            shown = '; '.join(f'{c} ~ {k} at rho {r:+.2f}' for c, k, r in dropped[:3])
            print(f'  skipped {len(dropped)} near-duplicate channels (|rho| >= {dup_rho}): '
                  f'{shown}')
        print(f'  null {n0:.4g} -> single {out["nmad_single"]:.4g} -> combined '
              f'{out["nmad_combined"]:.4g} (dimensionless v-mode amplitude), '
              f'skill {out["skill_single"]:+.3f} -> {out["skill_combined"]:+.3f} '
              f'(dimensionless), n {len(d)} visits')
    return out


def combined_table(df, grid, rho_strong=RHO_STRONG, n_lead=N_LEAD, n_splits=F.N_SPLITS,
                   dup_rho=CHANNEL_DUP_RHO, verbose=True):
    """Combined fits for every mode whose strongest channel clears the threshold.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Per-visit frame.
    grid : `pandas.DataFrame`
        `channel_grid` result.
    rho_strong : `float`, optional
    n_lead : `int`, optional
    n_splits : `int`, optional
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        One row per followed-up mode, without the array columns.
    results : `list` [`dict`]
        The full `combined_fit` results, arrays included, for plotting.
    """
    best = leading_channels(grid, rho_strong=rho_strong)
    results = []
    for mode in best['mode'].astype(int):
        if verbose:
            print()
        results.append(combined_fit(df, int(mode), grid, n_lead=n_lead, n_splits=n_splits,
                                    dup_rho=dup_rho, verbose=verbose))
    drop = ('response', 'prediction', 'residual', 'day_obs')
    tab = pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in results])
    if len(tab):
        tab['channels'] = tab['channels'].apply(lambda cs: ','.join(cs))
        tab = tab.sort_values('skill_combined', ascending=False).reset_index(drop=True)
    return tab, results


def expectation_check(grid, verbose=True):
    """Whether the modes responding to M1M3 shape terms are the ones theory predicts.

    Parameters
    ----------
    grid : `pandas.DataFrame`
        `channel_grid` result.
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        Per M1M3 shape channel, the three modes with the strongest |Spearman rho|.

    Notes
    -----
    ``smatrix/docs/plots.md`` predicts the M1M3 z-gradient drives Z4 defocus plus Z11 spherical
    and the radial gradient drives Z4 + Z11 + Z22, at 0.0185 µm of Z22 per deg C/m. If the modes
    topping these channels are the defocus and spherical-like ones, the grid supports a thermal
    origin for them; if they are scattered across high modes with no shared structure, it does
    not.
    """
    shape = ['m1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m',
             'm1m3_r2_coeff_c', 'm1_r2_coeff_c', 'm3_r2_coeff_c']
    rows = []
    for c in shape:
        g = grid[(grid['channel'] == c) & np.isfinite(grid['spearman_rho'])].copy()
        if not len(g):
            continue
        g['abs_rho'] = g['spearman_rho'].abs()
        for rank, (_, r) in enumerate(g.sort_values('abs_rho', ascending=False).head(3).iterrows(),
                                      start=1):
            rows.append(dict(channel=c, rank=rank, mode=int(r['mode']),
                             spearman_rho=r['spearman_rho'], pearson_r=r['pearson_r'],
                             skill=r['skill']))
    tab = pd.DataFrame(rows)
    if verbose and len(tab):
        print('strongest modes per M1M3 shape channel (Spearman rho, dimensionless):')
        for c in shape:
            sub = tab[tab['channel'] == c]
            if not len(sub):
                continue
            got = ', '.join(f'v{int(r["mode"])} {r["spearman_rho"]:+.3f}'
                            for _, r in sub.iterrows())
            print(f'  {c:<32s} {got}')
    return tab


def nmad_summary(results):
    """Before-and-after residual scatter for the followed-up modes.

    Parameters
    ----------
    results : `list` [`dict`]
        `combined_fit` results.

    Returns
    -------
    tab : `pandas.DataFrame`
        ``mode``, the three nMAD values [dimensionless v-mode amplitude] and the fractional
        reduction the combination buys over the single best channel (dimensionless).
    """
    rows = []
    for r in results:
        if not np.isfinite(r.get('nmad_combined', np.nan)):
            continue
        gain = (np.nan if not np.isfinite(r['nmad_single']) or r['nmad_single'] == 0
                else 1.0 - r['nmad_combined'] / r['nmad_single'])
        rows.append(dict(mode=r['mode'], nmad_null=r['nmad_null'],
                         nmad_single=r['nmad_single'], nmad_combined=r['nmad_combined'],
                         gain_over_single=gain,
                         residual_nmad=nmad(r['residual'][np.isfinite(r['residual'])])))
    return pd.DataFrame(rows)
