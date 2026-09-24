"""The thermal-focus analysis: every fit and cross-check, one PDF, no network.

Reads the cached tables written by ``run_thermal_focus.py`` and produces one document. The
headline is that the telescope's uniform-defocus error is predictable from five thermal
channels — the Telescope Mount Assembly (TMA) truss temperature and the four M1M3 bulk
thermal gradients — with one band-independent Huber robust linear model.

Computed sections, in the order they are printed:

1. The response and the sample — the selection funnel and per-band coverage.
2. The thermal model — the deliverable fit, with whole nights held out.
3. Machine-learning legibility — night-grouped against visit-level scoring, model comparison,
   per-fold coefficients and the Full Array Mode (FAM) cross-check of the truss slope.
4. Camera-body temperature — an alternative thermometer, indistinguishable as a regressor.
5. What adds nothing — the channels whose gain does not exceed the across-band scatter.
5b. The quadratic radial M1M3 terms — three temperature fields going as radius squared, over the
   whole mirror, the M1 annulus and the M3 inner disc, tested for focus information the four
   bulk gradients cannot express: raw correlation, partial correlation with the deliverable
   features stripped from both sides, a night-grouped nested model comparison, and the
   substitution case a decision to switch from the gradients would rest on.
6. Residual shape — the one-sided positive tail that makes the fits robust rather than least
   squares.
7. Within-night behaviour — per-night elevation slopes and the rising-against-falling
   hysteresis test.
8. Band changes — the step a per-band correction injects, and what the shared model does to it.
9. FAM blocks — within-block focus drift, and whether the thermal correction helps.
10. The conversion table — the v-mode-1 to hexapod dz factor across projection schemes,
    including the 10-degree-of-freedom/1-mode case the online system would use.
11. The standalone calculator — ``trim_calculator.py``, which inlines the fitted coefficients,
    checked against the pipeline fitted here so the two cannot drift apart unnoticed.
12. The truss temperature alone — the one-thermometer correction the five-channel model must
    beat, scored the same night-grouped way.
13. One night-level train/test split — a single explicit holdout, so the training page can show
    the same fit on nights it saw beside nights it never saw.
14. The focus error as degrees of freedom — each visit's measured v-mode 1 back-projected into
    the camera and M2 hexapod dz it is built from, over all visits and at the start of each night.

The PDF is organised into three parts rather than following that numbering:

* **Part 1, before the correction** — what the focus error is, what the truss temperature alone
  achieves, and where the signal lives.
* **Part 2, training** — the vocabulary (night-grouped, out-of-fold, per-fold) in plain language,
  the train/test holdout, the fitted model, and what each extra channel buys.
* **Part 3, all the data** — elevation, FAM blocks, the conversion table, the calculator check,
  and the degrees of freedom the measured focus error corresponds to.

Invocation::

    python code/run_thermal_focus_analysis.py
    python code/run_thermal_focus_analysis.py --day-obs-range 20251103 20260713
    python code/run_thermal_focus_analysis.py --no-pdf

Notes
-----
No stage here touches the network or the value-added DuckDB: everything comes from the parquet
files the build stage cached, which is what makes iterating on a fit cheap. If a column is
missing, the build stage is what needs re-running.
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages              # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

import thermal_focus_fit as F                                     # noqa: E402
import thermal_focus_lib as L                                     # noqa: E402
import trim_calculator as T                                       # noqa: E402
from common.utils import nmad                                     # noqa: E402

#: Bands in plotting order, bluest first, with a colour each.
BAND_COLOUR = {'u': '#3b4cc0', 'g': '#4fa845', 'r': '#d62728',
               'i': '#8c564b', 'z': '#7f4fa8', 'y': '#bcab2a'}

#: Feature groups tried in the ablation, beyond the deliverable set. Each is scored
#: night-grouped against the deliverable so a gain has to beat the across-band scatter.
ABLATION_GROUPS = (('truss',), ('zgrad',), ('truss', 'zgrad'), ('truss', 'grads'),
                   ('truss', 'r2grads'), ('truss', 'grads', 'r2all'),
                   ('truss', 'grads', 'r2split'), ('truss', 'grads', 'r2grads'),
                   ('truss', 'grads', 'camtemp'), ('truss', 'grads', 'wind'),
                   ('truss', 'grads', 'elev'), ('truss', 'grads', 'hexhist'),
                   ('truss', 'grads', 'camtemp', 'wind', 'elev', 'hexhist'))


def load(out_dir, fam_dir_name, day_obs_range=None, verbose=True):
    """Read the cached tables and apply the feature cut.

    Parameters
    ----------
    out_dir : `pathlib.Path`
        Directory holding ``thermal_focus.parquet``.
    fam_dir_name : `str`
        Subdirectory holding ``thermal_focus_fam.parquet``.
    day_obs_range : `tuple` [`int`] or `None`, optional
        Inclusive night range as ``YYYYMMDD``, to restrict below the cached span.
    verbose : `bool`, optional
        Print the funnel.

    Returns
    -------
    sci : `pandas.DataFrame`
        Science visits with all five deliverable features present.
    fam : `pandas.DataFrame` or `None`
        FAM triplets, or None when the file is absent.
    features : `list` [`str`]
        The deliverable feature column names.

    Raises
    ------
    SystemExit
        If the science table is absent, naming the command that writes it.

    Notes
    -----
    The `thermal_focus_lib.TRUSS_TEMP_MAX_C` cut is applied here as well as in the build
    stage, so a cache written before the cut existed gives the same sample as one written
    after it. It is a pure row filter on a cached column, so re-applying it costs nothing
    and removes the only way the two stages could disagree on the sample.
    """
    sci_path = out_dir / 'thermal_focus.parquet'
    if not sci_path.exists():
        raise SystemExit(f'{sci_path} is absent; build it with\n  python '
                         f'code/run_thermal_focus.py')
    sci = pd.read_parquet(sci_path)
    features = L.resolve_features(L.DELIVERABLE_GROUPS)

    n_cached = len(sci)
    if day_obs_range:
        sci = sci[(sci.day_obs >= day_obs_range[0]) & (sci.day_obs <= day_obs_range[1])]
    n_span = len(sci)
    hot = sci['truss_temp_mean_c'] > L.TRUSS_TEMP_MAX_C
    n_hot_visits, n_hot_nights = int(hot.sum()), int(sci.loc[hot, 'day_obs'].nunique())
    sci = sci[~hot]
    sci = sci[sci[features].notna().all(axis=1)].reset_index(drop=True)

    fam_path = out_dir / fam_dir_name / 'thermal_focus_fam.parquet'
    fam = pd.read_parquet(fam_path) if fam_path.exists() else None
    if fam is not None and day_obs_range:
        fam = fam[(fam.day_obs >= day_obs_range[0]) & (fam.day_obs <= day_obs_range[1])]
    if fam is not None and 'truss_temp_mean_c' in fam.columns:
        fam = fam[~(fam['truss_temp_mean_c'] > L.TRUSS_TEMP_MAX_C)]

    if verbose:
        print(f'cached science visits                 : {n_cached}')
        if day_obs_range:
            print(f'  within day_obs {day_obs_range[0]} to {day_obs_range[1]}   : {n_span}')
        print(f'  truss temperature above {L.TRUSS_TEMP_MAX_C:.0f} deg C   : '
              f'-{n_hot_visits} visits on {n_hot_nights} nights')
        print(f'  with all {len(features)} deliverable features : {len(sci)}')
        print(f'  -> {len(sci)} visits, {sci["day_obs"].nunique()} nights, day_obs '
              f'{int(sci["day_obs"].min())} to {int(sci["day_obs"].max())}')
        print(f'uncorrected response [um of equivalent hexapod dz]: '
              f'median {sci["y"].median():+.1f}, nMAD {nmad(sci["y"].to_numpy()):.1f}')
        per_band = sci['band'].value_counts()
        print('per band: ' + ', '.join(f'{b} {int(per_band.get(b, 0))}' for b in BAND_COLOUR
                                      if b in per_band.index))
        if fam is not None:
            print(f'FAM triplets: {len(fam)} over {fam["day_obs"].nunique()} nights')
    return sci, fam, features


# ------------------------------------------------------------------------------- the sections

def section_sample(sci, features, verbose=True):
    """Coverage of the sample, per band and per night.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits.
    features : `list` [`str`]
        The deliverable feature columns.
    verbose : `bool`, optional
        Print the feature means.

    Returns
    -------
    out : `dict`
        ``per_band`` (`pandas.DataFrame`), ``per_night`` (`pandas.DataFrame`),
        ``feature_means`` (`dict`) and ``interp_frac`` (per cent of visits whose truss
        temperature was filled by within-night interpolation).
    """
    per_band = (sci.groupby('band')
                .agg(n=('y', 'size'), median=('y', 'median'),
                     nmad=('y', lambda v: nmad(v.to_numpy(float))))
                .reset_index())
    per_night = (sci.groupby('day_obs')
                 .agg(n=('y', 'size'), median=('y', 'median'),
                      truss=('truss_temp_mean_c', 'median'))
                 .reset_index())
    means = {c: float(sci[c].mean()) for c in features}
    interp = 0.0
    if 'truss_temp_mean_c_interpolated' in sci.columns:
        interp = 100.0 * float(sci['truss_temp_mean_c_interpolated'].fillna(False).mean())
    if verbose:
        print('feature means over the sample')
        for c, v in means.items():
            print(f'  {c:32s} {v:+10.5f} {F.FEATURE_UNITS.get(c, "?")}')
        print(f'truss temperature filled by within-night interpolation: {interp:.1f}% of visits')
        print(f'per-night response median spans {per_night["median"].min():+.1f} to '
              f'{per_night["median"].max():+.1f} um of equivalent hexapod dz over '
              f'{len(per_night)} nights')
    return dict(per_band=per_band, per_night=per_night, feature_means=means,
                interp_frac=interp)


def section_ablation(sci, verbose=True):
    """Night-grouped score for each feature set, against the across-band scatter.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits.
    verbose : `bool`, optional
        Print the table.

    Returns
    -------
    out : `pandas.DataFrame`
        Per feature set: ``groups``, ``n_features``, ``resid_nmad`` [µm of equivalent hexapod
        dz], ``r2`` (dimensionless) and ``gain`` against the deliverable set (dimensionless,
        deliverable nMAD over this set's nMAD).

    Notes
    -----
    A gain is only meaningful against the scatter it has to beat. Adding the ten camera and
    wind channels raises mean R² by about +0.04 (dimensionless), which is smaller than the
    across-band standard deviation of the baseline R² itself — so the extra channels are not
    buying a real improvement, and the five-channel linear model stands.
    """
    rows = []
    base_nmad = None
    for groups in ABLATION_GROUPS:
        feats = L.resolve_features(groups)
        have = [c for c in feats if c in sci.columns]
        if len(have) != len(feats):
            continue
        d = sci[sci[have].notna().all(axis=1)]
        if len(d) < 1000:
            continue
        r = F.evaluate(d.reset_index(drop=True), have, verbose=False)
        if tuple(groups) == tuple(L.DELIVERABLE_GROUPS):
            base_nmad = r['nmad']
        rows.append(dict(groups='+'.join(groups), n_features=len(have), n=len(d),
                         resid_nmad=r['nmad'], r2=r['r2']))
    out = pd.DataFrame(rows)
    if base_nmad:
        out['gain'] = base_nmad / out.resid_nmad
    if verbose:
        print('night-grouped feature ablation, Huber '
              '[residual nMAD in um of equivalent hexapod dz]')
        for _, r in out.iterrows():
            g = f'   gain {r.gain:.3f}x' if 'gain' in out.columns else ''
            print(f'  {r.groups:44s} {int(r.n_features):2d} features  n {int(r.n):6d}  '
                  f'nMAD {r.resid_nmad:6.1f}  R2 {r.r2:+.3f}{g}')
        if 'gain' in out.columns:
            print('  gain is dimensionless, deliverable-set nMAD over this set nMAD; '
                  'above 1 is better')
    return out


def section_r2grads(sci, features, model='huber', verbose=True):
    """The quadratic-in-radius M1M3 thermal terms, above and beyond the four bulk gradients.

    A temperature field going as radius squared bends the mirror into a shape much closer to
    pure defocus than a linear radial ramp does, so it is the term most likely to move focus.
    Three such terms are available -- over the whole mirror, over the M1 annulus alone and over
    the M3 inner disc alone. The question is not whether each correlates with focus, since the
    whole thermal field does, but whether any of them carries focus information the four bulk
    gradients already in the model cannot express.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, carrying the columns of `thermal_focus_lib.R2_COLS`.
    features : `list` [`str`]
        The deliverable model's features, used as the baseline and as the controls partialled
        out of each candidate.
    model : `str`, optional
        Key for `thermal_focus_fit.make_model`.
    verbose : `bool`, optional
        Print the tables.

    Returns
    -------
    out : `dict`
        ``available`` (`list` of the quadratic columns present), ``coverage`` (per cent of
        visits each resolves), ``lines`` (per column, `F.huber_line` against the response, slope
        in µm of equivalent hexapod dz per unit normalized radius-squared amplitude),
        ``partial`` (per column, `F.partial_correlation` against the deliverable features),
        ``nested`` (per candidate set, `F.nested_comparison`), ``redundancy`` (per column, the
        `F.huber_line` against ``m1m3_radial_gradient_c_per_m``) and ``swap`` (the nested result
        for the quadratic terms **substituted for** the four gradients rather than added to
        them). `None` for any entry the sample cannot support.

    Notes
    -----
    Three things are reported because they answer three different questions and routinely
    disagree. The raw correlation says the term tracks focus; the partial correlation says
    whether it still does once the incumbent gradients have had their share; and the
    night-grouped nested comparison says whether that surviving information generalises to
    nights the fit never saw. Only the third is a performance claim.

    The substitution row is what a decision to switch would rest on: it fits the truss
    temperature plus the three quadratic terms in place of the truss plus the four bulk
    gradients, on the same rows, so a smaller feature set is not credited for the easier sample.
    """
    cols = [c for c, _ in L.R2_COLS if c in sci.columns]
    out = {'available': cols, 'coverage': {}, 'lines': {}, 'partial': {}, 'nested': {},
           'redundancy': {}, 'swap': None}
    if not cols:
        if verbose:
            print('  no quadratic radial columns in the table; run '
                  'value_added/code/build_m1m3_thermal_r2.py, then rebuild the cached table')
        return out
    for c in cols:
        out['coverage'][c] = 100.0 * float(sci[c].notna().mean())
        out['lines'][c] = F.huber_line(sci[c], sci['y'])
        out['partial'][c] = F.partial_correlation(sci, 'y', c, features, model=model)
        if 'm1m3_radial_gradient_c_per_m' in sci.columns:
            out['redundancy'][c] = F.huber_line(sci['m1m3_radial_gradient_c_per_m'], sci[c])

    candidates = [('r2all', ['m1m3_r2_coeff_c']),
                  ('r2split', ['m1_r2_coeff_c', 'm3_r2_coeff_c']),
                  ('r2grads', cols)]
    for name, add in candidates:
        add = [c for c in add if c in cols]
        if not add:
            continue
        out['nested'][name] = F.nested_comparison(sci, features, add, model=model,
                                                  n_splits=F.N_SPLITS, verbose=False)

    truss = L.resolve_features(('truss',))
    swap_features = truss + cols
    ext = list(features) + [c for c in cols if c not in features]
    d = sci[['y', 'day_obs'] + ext].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) > 1000:
        d = d.reset_index(drop=True)
        out['swap'] = {
            'n': len(d), 'n_nights': int(d['day_obs'].nunique()),
            'gradients': F.evaluate(d, features, model=model, verbose=False),
            'quadratic': F.evaluate(d, swap_features, model=model, verbose=False)}

    if verbose:
        print('coverage and the raw relation to the response')
        for c, label in L.R2_COLS:
            if c not in cols:
                continue
            r = out['lines'][c]
            print(f'  {label:36s} resolves {out["coverage"][c]:5.2f}% of visits; slope '
                  f'{r["slope"]:+8.1f} +/- {r["slope_err"]:6.1f} um of equivalent hexapod dz '
                  f'per unit normalized r^2 amplitude, Pearson r {r["pearson_r"]:+.4f}, '
                  f'Spearman rho {r["spearman_rho"]:+.4f}, n {r["n"]}')
        if out['redundancy']:
            print('\nhow much each duplicates the existing M1M3 radial gradient')
            for c, label in L.R2_COLS:
                r = out['redundancy'].get(c)
                if not r:
                    continue
                print(f'  {label:36s} against the radial gradient: Pearson r '
                      f'{r["pearson_r"]:+.4f}, Spearman rho {r["spearman_rho"]:+.4f}, '
                      f'n {r["n"]} (both dimensionless)')
            print('  a high value here is why the partial correlation below is the test that '
                  'matters, not the raw one')
        print('\npartial correlation with the response, both sides stripped of the '
              f'{len(features)} deliverable features')
        for c, label in L.R2_COLS:
            p = out['partial'].get(c)
            if not p:
                continue
            print(f'  {label:36s} raw Pearson r {p["raw_pearson_r"]:+.4f} -> partial '
                  f'{p["partial_pearson_r"]:+.4f}, partial Spearman rho '
                  f'{p["partial_spearman_rho"]:+.4f}, residual slope {p["slope"]:+8.1f} +/- '
                  f'{p["slope_err"]:6.1f} um of equivalent hexapod dz per unit normalized '
                  f'r^2 amplitude, n {p["n"]} (correlations dimensionless)')
        print('\nnight-grouped nested comparison: does the surviving information generalise')
        for name, res in out['nested'].items():
            print(f'  baseline + {name:8s} on {res["n"]:6d} visits over {res["n_nights"]:3d} '
                  f'nights: nMAD {res["nmad_base"]:6.1f} -> {res["nmad_extended"]:6.1f} um of '
                  f'equivalent hexapod dz, gain {res["gain"]:.4f} (dimensionless, baseline '
                  f'over extended), delta R2 {res["delta_r2"]:+.4f} (dimensionless)')
        s = out['swap']
        if s:
            g, q = s['gradients'], s['quadratic']
            print(f'\nsubstitution rather than addition, on the same {s["n"]} visits over '
                  f'{s["n_nights"]} nights')
            print(f'  truss + four bulk gradients   ({len(features):2d} features): nMAD '
                  f'{g["nmad"]:6.1f} um of equivalent hexapod dz, R2 {g["r2"]:+.4f}')
            print(f'  truss + three quadratic terms ({len(swap_features):2d} features): nMAD '
                  f'{q["nmad"]:6.1f} um of equivalent hexapod dz, R2 {q["r2"]:+.4f}')
            better = g['nmad'] / q['nmad'] if q['nmad'] else float('nan')
            print(f'  ratio {better:.4f} (dimensionless, gradient nMAD over quadratic nMAD); '
                  'above 1 favours switching to the quadratic terms')
    return out


def section_camtemp(sci, verbose=True):
    """The camera-body temperature as an alternative to the truss thermometer.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``coverage`` per cent for each thermometer, ``truss_line`` and ``cam_line``
        (`F.huber_line` results, slope in µm of equivalent hexapod dz per °C), and
        ``correlation`` between the two thermometers.

    Notes
    -----
    The two thermometers are close to interchangeable as regressors, which is the useful
    result: the focus model does not depend on a channel that might drop out. The truss is
    kept as the deliverable because its coverage is higher and it is the quantity the
    look-up table is itself a function of.
    """
    out = {'coverage': {}}
    for col in ('truss_temp_mean_c', 'cam_AverageTemp'):
        if col in sci.columns:
            out['coverage'][col] = 100.0 * float(sci[col].notna().mean())
    out['truss_line'] = F.huber_line(sci['truss_temp_mean_c'], sci['y'])
    if 'cam_AverageTemp' in sci.columns:
        out['cam_line'] = F.huber_line(sci['cam_AverageTemp'], sci['y'])
        ok = sci[['truss_temp_mean_c', 'cam_AverageTemp']].notna().all(axis=1)
        if ok.sum() > 100:
            out['correlation'] = F.huber_line(sci.loc[ok, 'truss_temp_mean_c'],
                                              sci.loc[ok, 'cam_AverageTemp'])
    if verbose:
        for col, cov in out['coverage'].items():
            print(f'  {col:22s} resolves {cov:.2f}% of visits')
        for key, name in (('truss_line', 'TMA truss temperature'),
                          ('cam_line', 'camera-body AverageTemp')):
            r = out.get(key)
            if r:
                print(f'  {name:26s} slope {r["slope"]:+8.2f} +/- {r["slope_err"]:.2f} um of '
                      f'equivalent hexapod dz per deg C, Pearson r {r["pearson_r"]:+.4f}, '
                      f'Spearman rho {r["spearman_rho"]:+.4f}, n {r["n"]}')
        c = out.get('correlation')
        if c:
            print(f'  the two thermometers against each other: slope {c["slope"]:+.4f} deg C '
                  f'camera-body per deg C truss, Pearson r {c["pearson_r"]:+.4f}, '
                  f'Spearman rho {c["spearman_rho"]:+.4f}, n {c["n"]}')
    return out


def section_elevation(sci, resid, verbose=True):
    """Within-night elevation behaviour of the corrected residual.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, needing ``altitude_deg`` and ``obs_start_mjd``.
    resid : `array_like`
        Out-of-fold residual of the deliverable model [µm of equivalent hexapod dz].
    verbose : `bool`, optional
        Print the summary.

    Returns
    -------
    out : `pandas.DataFrame`
        Per-night elevation slopes from `F.per_night_elevation`, or empty when elevation or
        the observation time is absent from the cached table.

    Notes
    -----
    The residual is used rather than the raw response, so an elevation dependence found here
    is one the thermal model does not already account for. The slew direction comes from a
    centred 21-visit rolling median of elevation with a deadband, not from the sign of the
    per-visit difference, which alternates while tracking.
    """
    need = ('altitude_deg', 'obs_start_mjd')
    if any(c not in sci.columns for c in need):
        if verbose:
            print(f'  elevation section skipped: {", ".join(c for c in need if c not in sci.columns)} '
                  f'absent from the cached table')
        return pd.DataFrame()
    d = sci.copy()
    d['resid'] = resid
    d['direction'] = pd.concat([F.label_direction(g) for _, g in d.groupby('day_obs')]
                               ).reindex(d.index)
    if verbose:
        vc = d.direction.value_counts()
        print(f'  slew-direction labelling (centred {F.DIRECTION_WINDOW}-visit rolling median '
              f'of elevation, deadband {F.DIRECTION_DEADBAND} deg per visit): '
              + ', '.join(f'{k} {int(v)}' for k, v in vc.items()))
    return F.per_night_elevation(d, ycol='resid', verbose=verbose)


def section_fam(fam, sci, features, verbose=True):
    """Within-block focus drift, and whether the thermal correction reduces it.

    Parameters
    ----------
    fam : `pandas.DataFrame`
        FAM triplets from the build stage.
    sci : `pandas.DataFrame`
        Science visits, used to fit the model that is then applied to the FAM rows.
    features : `list` [`str`]
        The deliverable feature columns.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``sets`` (`pandas.DataFrame` of per-block scatter), ``median_p2p_uncorrected`` and
        ``median_p2p_corrected`` [µm of equivalent hexapod dz], ``ratio`` (dimensionless,
        corrected over uncorrected), ``n_improved``, ``n_sets``, and
        ``n_sets_trim_frozen`` — sets whose ``v1_trim`` within-set peak-to-peak is exactly zero.

    Notes
    -----
    The expected negative result, and the reason it is kept: **the thermal correction makes
    within-block scatter worse**. The cause is that the quantity the model actually predicts
    does not move inside a block.

    The response is ``(v1_trim + MEASURED_SIGN * v1) / v1_per_um_dz`` with
    ``MEASURED_SIGN = -1.0`` (dimensionless), so it carries a commanded term and a measured term
    of opposite sign. Between nights the commanded term dominates — ``v1_trim`` carries a
    between-night variance fraction of 91.2% against 25.0% for the measured ``v1`` (both
    dimensionless, between-night over total) — so the fitted model is essentially a model of the
    commanded Trim. Inside a FAM block the Trim is **exactly constant**: the within-set
    peak-to-peak is identically zero in 44 of 45 clean sets, because the AOS does not re-command
    Trim while a ladder runs. The response there reduces to ``-v1 / v1_per_um_dz``, the measured
    term alone and of the opposite sign, which is why the within-set slope against truss
    temperature is −81.10 ± 17.58 against +124.38 µm of equivalent hexapod dz per °C between
    nights.

    This is not telemetry noise: the truss temperature is resolved inside a block (12 distinct
    values per set, monotonic in 26 of 45) and a within-set permutation test puts the observed
    slope about 4 null-sigma out. Nor is it a sign error — adding the prediction rather than
    subtracting it does reduce within-set scatter, but that fits the measured term with a model
    of the commanded term and would not survive a block in which Trim moved.
    """
    if fam is None or not len(fam):
        return {}
    have = [c for c in features if c in fam.columns]
    if len(have) != len(features):
        if verbose:
            print(f'  FAM section limited: {", ".join(c for c in features if c not in have)} '
                  f'absent from the FAM table')
        return {}
    d = fam[fam[have].notna().all(axis=1)].copy()
    if not len(d):
        return {}

    full = F.fit_full(sci, features, verbose=False)
    d['pred'] = full['model'].predict(d[have].to_numpy(float))
    d['y_corrected'] = d['y'] - d['pred']

    # A FAM block is derived from held pointing, not stored, so it is assigned here rather than
    # read. Grouping by night instead would substitute a whole night for a 12-triplet block and
    # silently change what every within-set number below means.
    d = F.assign_blocks(d)
    d, info = F.select_sets(d, verbose=verbose)
    if not len(d):
        if verbose:
            print('  no clean 12-triplet set survives the cuts')
        return {}
    set_col = 'set_id'
    scatter_cols = {'y': 'um of equivalent hexapod dz',
                    'y_corrected': 'um of equivalent hexapod dz',
                    'pred': 'um of equivalent hexapod dz'}
    # The commanded Trim is measured, not assumed, because it is the whole explanation of the
    # negative result below: if it does not move within a set, the response there is the measured
    # term alone, which enters with the opposite sign from the one the model was fitted on.
    if 'v1_trim' in d.columns:
        scatter_cols['v1_trim'] = 'dimensionless v-mode-1 amplitude'
    sets = F.within_set_scatter(d, set_col, scatter_cols, verbose=False)
    ok = sets[['y_p2p', 'y_corrected_p2p']].notna().all(axis=1)
    s = sets[ok]
    out = dict(sets=sets, set_col=set_col, info=info, n_sets=int(len(s)),
               median_p2p_uncorrected=float(s.y_p2p.median()) if len(s) else float('nan'),
               median_p2p_corrected=float(s.y_corrected_p2p.median()) if len(s) else float('nan'),
               median_p2p_prediction=float(s.pred_p2p.median()) if len(s) else float('nan'),
               n_improved=int((s.y_corrected_p2p < s.y_p2p).sum()) if len(s) else 0)
    out['ratio'] = (out['median_p2p_corrected'] / out['median_p2p_uncorrected']
                    if out['median_p2p_uncorrected'] else float('nan'))
    out['n_sets_trim_frozen'] = (int((s.v1_trim_p2p == 0.0).sum())
                                 if len(s) and 'v1_trim_p2p' in s.columns else None)
    if verbose:
        print(f'  within-set scatter over {out["n_sets"]} sets keyed on {set_col} '
              f'[um of equivalent hexapod dz]')
        print(f'    uncorrected response, median peak-to-peak : '
              f'{out["median_p2p_uncorrected"]:.1f}')
        print(f'    thermally corrected, median peak-to-peak  : '
              f'{out["median_p2p_corrected"]:.1f}')
        print(f'    the prediction\'s own within-set swing     : '
              f'{out["median_p2p_prediction"]:.1f}')
        print(f'    ratio {out["ratio"]:.2f} (dimensionless, corrected over uncorrected); '
              f'{out["n_improved"]} of {out["n_sets"]} sets improve')
        if out['n_sets_trim_frozen'] is not None:
            print(f'    sets whose commanded Trim is exactly frozen       : '
                  f'{out["n_sets_trim_frozen"]} of {out["n_sets"]}')
        if out['ratio'] > 1:
            print('    -> the between-night correction makes within-block scatter WORSE. The '
                  'response is\n       (v1_trim - v1) / v1_per_um_dz and the fit is dominated by '
                  'the commanded v1_trim term\n       (91.2% between-night variance, '
                  'dimensionless, against 25.0% for the measured v1).\n       With Trim frozen '
                  'inside the block the response is the measured term alone, of the\n       '
                  'opposite sign, so the model cannot track it. This is not telemetry noise: '
                  'the truss\n       is resolved within a set and the reversed slope is about 4 '
                  'permutation-null sigma out.')
    # DZ(k=1, j=4) is the uniform-defocus Double Zernike coefficient: the same physical quantity
    # as the response, measured by the FAM fit rather than recovered from the optical state, so
    # its within-set scatter is an independent estimate of the same drift.
    # The exactly-12 requirement is strict, and it is fair to ask whether the surviving sets are
    # a selected subsample. Relaxing the floor is the check: the within-set scatter has to be
    # insensitive to it, or the number is about which blocks survived rather than about focus.
    floors = []
    blocks = F.assign_blocks(fam[fam[have].notna().all(axis=1)].copy())
    blocks = blocks[~blocks.day_obs.isin(L.LUT_EPOCH_OFFSET_NIGHTS)]
    sizes = blocks[blocks.block >= 0].groupby('block').size()
    for floor in (12, 8, 6):
        o = blocks[blocks.block.isin(sizes[sizes >= floor].index)]
        if not len(o):
            continue
        p = F.within_set_scatter(o, 'block', {'y': ''}, verbose=False)['y_p2p'].dropna()
        if len(p):
            floors.append(dict(floor=floor, n_sets=int(len(p)),
                               n_nights=int(o.day_obs.nunique()), median_p2p=float(p.median())))
    out['size_floors'] = floors
    if verbose and floors:
        print('  sensitivity to the set-size floor [um of equivalent hexapod dz]')
        for r in floors:
            print(f'    at least {r["floor"]:2d} triplets per set: {r["n_sets"]:3d} sets over '
                  f'{r["n_nights"]:2d} nights, median within-set peak-to-peak '
                  f'{r["median_p2p"]:.1f}')

    if 'dz_k1_j4' in d.columns:
        dz_sets = F.within_set_scatter(d, set_col, {'dz_k1_j4': 'um of wavefront'},
                                       verbose=verbose)
        p = dz_sets['dz_k1_j4_p2p'].dropna()
        if len(p):
            out['dz_k1_j4_p2p_um_wf'] = float(p.median())
            out['dz_k1_j4_p2p_um_dz'] = abs(out['dz_k1_j4_p2p_um_wf'] * L.DZ_UM_PER_UM_WF)
            if verbose:
                print(f'    DZ(k=1,j=4) within-set peak-to-peak '
                      f'{out["dz_k1_j4_p2p_um_wf"]:.4f} um of wavefront = '
                      f'{out["dz_k1_j4_p2p_um_dz"]:.1f} um of equivalent hexapod dz, against '
                      f'the response\'s {out["median_p2p_uncorrected"]:.1f} um')
    return out


def section_calculator(sci, features, full, verbose=True):
    """Check the standalone calculator against the fitted pipeline.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science table, as loaded.
    features : `list` [`str`]
        Deliverable feature columns, in fitted order.
    full : `dict`
        Result of `thermal_focus_fit.fit_full`, carrying the fitted pipeline under ``model``.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``max_abs_diff_um`` and ``median_diff_um`` over the sample, ``worked_max_abs_diff_um``
        over `trim_calculator.TEST_CASES`, and ``n`` — all µm of equivalent hexapod dz.

    Notes
    -----
    `trim_calculator` is a standalone copy with every coefficient inlined to two decimals, so
    that it can be run on a summit machine with nothing but numpy. Inlining means it can drift
    from the fit silently, which is exactly what this section exists to catch: if a coefficient
    here is re-fitted and the calculator is not updated, ``max_abs_diff_um`` grows from rounding
    noise to something that matters.
    """
    cols = ['truss_temp_mean_c', 'm1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
            'm1m3_radial_gradient_c_per_m', 'm1m3_x_gradient_c_per_m']
    if any(c not in sci.columns for c in cols) or list(features) != cols:
        if verbose:
            print('  the fitted feature set is not the calculator\'s five channels, so the '
                  'calculator is not comparable here; skipped')
        return {}
    calc = T.predict_focus_error_um(*[sci[c].to_numpy(float) for c in cols],
                                   warn_extrapolation=False)
    diff = calc - np.asarray(full['pred'], float)
    worked = max(abs(T.predict_focus_error_um(**inp, warn_extrapolation=False) - exp)
                 for _, inp, exp in T.TEST_CASES)
    out = {'n': int(len(sci)),
           'max_abs_diff_um': float(np.nanmax(np.abs(diff))),
           'median_diff_um': float(np.nanmedian(diff)),
           'worked_max_abs_diff_um': float(worked)}
    if verbose:
        print(f'  calculator against the fitted pipeline over {out["n"]} visits: '
              f'max |difference| {out["max_abs_diff_um"]:.4f}, median '
              f'{out["median_diff_um"]:+.4f} um of equivalent hexapod dz')
        print(f'  its {len(T.TEST_CASES)} worked cases agree with their stated values to '
              f'{out["worked_max_abs_diff_um"]:.4f} um of equivalent hexapod dz')
        print(f'  inlined coefficients: intercept {T.INTERCEPT_UM:+.2f} um, truss '
              f'{T.TRUSS_UM_PER_C:+.2f} um per deg C')
    return out


def section_trussonly(sci, model='huber', verbose=True):
    """Fit the truss temperature alone, night-grouped, as the correction to beat.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science table, as loaded.
    model : `str`, optional
        Key for `thermal_focus_fit.make_model`.
    verbose : `bool`, optional
        Print the comparison against the uncorrected response.

    Returns
    -------
    out : `dict`
        ``resid`` and ``pred`` [µm of equivalent hexapod dz], ``nmad``, ``r2``, ``slope`` and
        ``slope_err`` [µm of equivalent hexapod dz per °C], ``pearson_r`` and ``spearman_rho``
        (all dimensionless), and ``uncorrected_nmad``.

    Notes
    -----
    This is the one-feature correction an observer could apply from a single thermometer, and it
    is what the four M1M3 gradients have to improve on to earn their place. Reported next to the
    truss scatter plot rather than only as a table row, so the residual it leaves is visible as a
    distribution and not just as one number.
    """
    cv = F.evaluate(sci, ['truss_temp_mean_c'], model=model, verbose=False)
    line = F.huber_line(sci['truss_temp_mean_c'], sci['y'])
    out = {'resid': cv['resid'], 'pred': cv['pred'], 'nmad': float(cv['nmad']),
           'r2': float(cv['r2']), 'slope': float(line['slope']),
           'slope_err': float(line.get('slope_err', np.nan)),
           'pearson_r': float(line['pearson_r']), 'spearman_rho': float(line['spearman_rho']),
           'uncorrected_nmad': float(nmad(sci['y'].to_numpy(float)))}
    if verbose:
        print(f'  truss temperature alone, night-grouped: residual nMAD {out["nmad"]:.1f} um of '
              f'equivalent hexapod dz, R2 {out["r2"]:+.3f} (dimensionless)')
        print(f'  against the uncorrected {out["uncorrected_nmad"]:.1f} um, an improvement of '
              f'{out["uncorrected_nmad"] / out["nmad"]:.2f}x (dimensionless, uncorrected nMAD '
              f'over residual nMAD)')
        print(f'  the fitted line is {out["slope"]:+.2f} um of equivalent hexapod dz per deg C, '
              f'Pearson r {out["pearson_r"]:+.4f}, Spearman rho {out["spearman_rho"]:+.4f}, '
              f'n {len(sci)}')
    return out


def section_holdout(sci, features, model='huber', frac_test=0.2, verbose=True):
    """One night-level train/test split, so train and test can be shown side by side.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science table, as loaded.
    features : `list` [`str`]
        Feature column names.
    model : `str`, optional
        Key for `thermal_focus_fit.make_model`.
    frac_test : `float`, optional
        Fraction of **nights** held out [dimensionless].
    verbose : `bool`, optional
        Print both scores.

    Returns
    -------
    out : `dict`
        ``train`` and ``test``, each a dict of ``y``, ``pred`` and ``resid`` [µm of equivalent
        hexapod dz] plus ``nmad``, ``r2``, ``r2_robust``, ``n_visits`` and ``n_nights``; and
        ``optimism`` (dimensionless, test nMAD over train nMAD).

    Notes
    -----
    The 5-fold `sklearn.model_selection.GroupKFold` used everywhere else predicts every visit
    exactly once with its own night held out, so there is no single train set and no single test
    set to plot. This section makes one explicit split instead, purely so the training page can
    show the fit on the nights it saw beside the same fit on nights it never saw. The nights are
    split by a hash of ``day_obs`` rather than by date, so the test nights are spread across the
    season instead of being the last few weeks — a date split would confound held-out with late.

    The headline score in this document remains the night-grouped cross-validated one, which uses
    every night: this split is a demonstration, and its test score is noisier because it rests on
    a fifth of the nights.

    Both R² values are formed against the **whole sample's** response variance rather than each
    subset's own, so that they are comparable. Scoring each subset against its own variance makes
    the test R² look better than the train R² whenever the held-out nights happen to span a wider
    range of response — a property of which nights were drawn, not of the fit. The nMAD values,
    being absolute, need no such care and are the ones to compare.

    Even shared-denominator R² is misleading here, and by a large factor. The train residual has
    nMAD 57.6 µm but standard deviation 317.6 µm of equivalent hexapod dz: a handful of visits
    with enormous focus errors, which the Huber loss correctly refuses to chase, dominate the
    variance. Those visits happen to fall on nights that landed in the train set, so the
    variance-based R² is far worse on train than on test for a reason that has nothing to do with
    generalisation. ``r2_robust`` therefore replaces both variances with squared nMAD, which is
    the quantity the robust fit actually minimises the scale of, and it is what the training page
    shows. ``r2`` is retained only so the discrepancy between the two is visible.
    """
    nights = np.sort(sci['day_obs'].unique())
    rng = np.random.default_rng(0)
    test_nights = set(rng.choice(nights, size=max(1, int(round(frac_test * len(nights)))),
                                 replace=False).tolist())
    is_test = sci['day_obs'].isin(test_nights).to_numpy()

    X = sci[features].to_numpy(float)
    y = sci['y'].to_numpy(float)
    m = F.make_model(model)
    m.fit(X[~is_test], y[~is_test])

    out = {}
    var_all = float(np.nanvar(y))
    nmad_all = float(nmad(y))
    for name, sel in (('train', ~is_test), ('test', is_test)):
        p = m.predict(X[sel])
        r = y[sel] - p
        out[name] = {'y': y[sel], 'pred': p, 'resid': r, 'nmad': float(nmad(r)),
                     'r2': float(1.0 - np.nanvar(r) / var_all),
                     'r2_robust': float(1.0 - (nmad(r) / nmad_all) ** 2),
                     'n_visits': int(sel.sum()),
                     'n_nights': int(sci.loc[sel, 'day_obs'].nunique())}
    out['optimism'] = out['test']['nmad'] / out['train']['nmad']
    if verbose:
        for name in ('train', 'test'):
            d = out[name]
            print(f'  {name:5s}: {d["n_visits"]:6d} visits over {d["n_nights"]:3d} nights, '
                  f'residual nMAD {d["nmad"]:.1f} um of equivalent hexapod dz, '
                  f'robust R2 {d["r2_robust"]:+.3f}, variance R2 {d["r2"]:+.3f} (dimensionless)')
        print(f'  test over train nMAD {out["optimism"]:.2f}x (dimensionless): the price of '
              f'predicting a night the fit never saw')
        print(f'  the two R2 columns disagree because a few extreme-focus visits dominate the '
              f'variance; the nMAD pair is the comparison to read')
    return out


def section_dof(sci, pred, v1_per_um_dz, dof_set='all_50', n_modes=34, verbose=True):
    """Express the thermal correction as the Trim degrees of freedom that would apply it.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science table, carrying the identity columns ``day_obs``, ``seq_num`` and
        ``obs_start_mjd``.
    pred : `array_like`
        Predicted focus error per visit [µm of equivalent hexapod dz], out of fold. This is the
        correction the model asks for, and it is what the DOF are computed from.
    v1_per_um_dz : `float`
        Conversion from µm of total hexapod dz travel to dimensionless v-mode-1 amplitude.
    dof_set : `str`, optional
        ts_ofc degree-of-freedom (DOF) set name for the projection.
    n_modes : `int`, optional
        Number of v-modes retained.
    verbose : `bool`, optional
        Print the per-DOF content and the resulting distributions.

    Returns
    -------
    out : `dict`
        ``unit`` — the DOF vector for v1 = 1.0, one entry per named DOF [µm or arcsec];
        ``dof`` — a `pandas.DataFrame` of the per-visit Trim DOF that would apply the correction,
        with the identity columns;
        ``start`` — the same restricted to the first visit of each night;
        ``names`` — the DOF columns carried, in descending order of magnitude;
        ``start_slope`` — per hexapod DOF, the `thermal_focus_fit.huber_line` fit of the
        start-of-night value against Modified Julian Date [µm per d];
        ``start_vs_night`` — per hexapod DOF, ``start_nmad``, ``night_nmad`` and their ``ratio``
        (dimensionless, start-of-night nMAD over per-night-median nMAD).

    Notes
    -----
    **The quantity is the correction, not the residual.** These are the Trim DOF an observer would
    command to apply the thermal correction: the predicted focus error is converted back to a
    v-mode-1 amplitude, ``v1_applied = pred * v1_per_um_dz``, and that amplitude is back-projected
    into DOF. The commanded term enters the response with a positive sign, so the amplitude needed
    in the Trim to cancel a predicted error is that error in v-mode-1 units, with no sign flip.
    Plotting the corrected residual instead would answer a different question — how well the
    correction worked — which the training pages already cover.

    The back-projection is `lsst.ts.ofc`'s own inverse, ``StateEstimator.get_dofs_from_vmodes``,
    which is ``normalization_matrix @ (v_modes @ Vh)``. The normalization matrix is **not**
    optional: projecting with ``Vh[0]`` alone gives a DOF vector whose forward projection is
    v1 = +0.0141 with another mode at 0.227, rather than the v1 = +1.0000000000 with a largest
    other mode of 2.6e-16 that the normalized inverse round-trips to.

    Setting every other v-mode to zero is not an approximation to be apologised for. ``Vh`` is
    orthonormal, so the zero-other-modes vector is the exact minimum-norm DOF vector consistent
    with the measured v1 — the unique answer with no component in any other mode. What it does
    **not** claim is that the telescope's other modes were actually zero; it is the defocus part
    of the state, expressed in DOF.

    Only two DOF carry the defocus: camera hexapod dz and M2 hexapod dz, which move together in
    a fixed ratio because v-mode 1 is one direction in DOF space. The mirror bending modes M1M3
    B3 and M2 B5 appear at +0.0094 and +0.0076 µm per unit v1, so over the range of corrections
    the model asks for they stay sub-nanometre and negligible; M2 B4 appears only at +0.0002 µm
    per unit v1, below even those. They are reported so the document can say how small they are
    rather than because an observer would set them.

    The start-of-night spread is compared against the **night-to-night** spread of the per-night
    medians over the same nights, not against the all-visit nMAD. The latter mixes within-night and
    between-night scatter over every visit, so comparing a per-night statistic to it would not be
    like for like.

    All four DOF return the same ratio, the same significance and the same correlation
    coefficients, because each is a fixed multiple of the one v-mode-1 amplitude. That is a
    property of the projection, not four independent measurements.
    """
    import aos_state
    se = aos_state.make_state_estimator(dof_set=dof_set, n_modes=n_modes)
    v = np.zeros(se.truncate_index)
    v[0] = 1.0
    unit_vec = np.asarray(se.get_dofs_from_vmodes(v), float)

    # The four DOF v-mode 1 actually contains, largest first. The hexapod dz pair carries it;
    # the two bending modes are kept so the document can say how small they are rather than
    # leaving a reader to wonder whether they were dropped.
    wanted = ((5, 'cam_hex_dz_um', 'camera hexapod dz', 'um'),
              (0, 'm2_hex_dz_um', 'M2 hexapod dz', 'um'),
              (12, 'm1m3_b3_um', 'M1M3 bending mode B3', 'um'),
              (34, 'm2_b5_um', 'M2 bending mode B5', 'um'))

    # The correction the model asks for, as a v-mode-1 amplitude. The commanded Trim term enters
    # the response with a positive sign, so this is the amplitude to put into the Trim.
    v1_applied = np.asarray(pred, float) * v1_per_um_dz
    keep = [c for c in ('visit_id', 'day_obs', 'seq_num', 'obs_start_mjd', 'band',
                        'truss_temp_mean_c') if c in sci.columns]
    dof = sci[keep].copy()
    dof['v1_applied'] = v1_applied
    names = []
    for idx, col, _, _ in wanted:
        dof[col] = unit_vec[idx] * v1_applied
        names.append(col)

    order = ['day_obs'] + [c for c in ('seq_num', 'visit_id') if c in dof.columns]
    start = dof.sort_values(order).groupby('day_obs', as_index=False).first()

    out = {'unit': {col: float(unit_vec[idx]) for idx, col, _, _ in wanted},
           'labels': {col: label for _, col, label, _ in wanted},
           'units': {col: u for _, col, _, u in wanted},
           'dof': dof, 'start': start, 'names': names,
           'start_slope': {}, 'start_vs_night': {}}

    # The start-of-night spread is worth comparing against the night-to-night spread over the same
    # nights, not against the all-visit nMAD, which mixes within-night and between-night scatter.
    for col in names:
        a = start[col].to_numpy(float)
        per_night = dof.groupby('day_obs')[col].median().to_numpy(float)
        out['start_vs_night'][col] = {'start_nmad': float(nmad(a)),
                                      'night_nmad': float(nmad(per_night)),
                                      'ratio': float(nmad(a) / nmad(per_night))}
        if 'obs_start_mjd' in start.columns:
            mjd = start['obs_start_mjd'].to_numpy(float)
            ok = np.isfinite(mjd) & np.isfinite(a)
            if ok.sum() > 10:
                out['start_slope'][col] = F.huber_line(mjd[ok], a[ok])

    if verbose:
        print(f'  back-projection {dof_set}/{n_modes}, '
              f'StateEstimator.get_dofs_from_vmodes = normalization_matrix @ (v @ Vh)')
        print(f'  DOF content of v-mode 1 at v1 = 1.0 (dimensionless):')
        for _, col, label, u in wanted:
            print(f'    {label:24s} {out["unit"][col]:+12.4f} {u} per unit v1')
        p99 = float(np.nanpercentile(np.abs(v1_applied), 99))
        print(f'  applied |v1| 99th percentile {p99:.5f} (dimensionless), so the two bending '
              f'modes reach at most {abs(out["unit"]["m1m3_b3_um"]) * p99 * 1e3:.4f} and '
              f'{abs(out["unit"]["m2_b5_um"]) * p99 * 1e3:.4f} nm -- negligible')
        # The bending modes are thousandths of a um, so they are reported in nm; printed in um
        # every digit that distinguishes them rounds away.
        scale = {c: (1e3, 'nm') if c in names[2:] else (1.0, 'um') for c in names}
        print(f'  per-visit Trim DOF to apply the correction, over {len(dof)} visits')
        for col in names:
            sc, u = scale[col]
            a = dof[col].to_numpy(float) * sc
            print(f'    {out["labels"][col]:24s} median {np.nanmedian(a):+10.4f}  '
                  f'nMAD {nmad(a):9.4f}  p1 {np.nanpercentile(a, 1):+10.4f}  '
                  f'p99 {np.nanpercentile(a, 99):+10.4f} {u}')
        print(f'  start-of-night visits: {len(start)} nights, MJD '
              f'{start["obs_start_mjd"].min():.3f} to {start["obs_start_mjd"].max():.3f}'
              if 'obs_start_mjd' in start.columns else
              f'  start-of-night visits: {len(start)} nights')
        for col in names:
            sc, u = scale[col]
            a = start[col].to_numpy(float) * sc
            print(f'    start of night, {out["labels"][col]:22s} median '
                  f'{np.nanmedian(a):+10.4f}  nMAD {nmad(a):9.4f} {u}')
        for col in names:
            sc, u = scale[col]
            v = out['start_vs_night'][col]
            print(f'    {out["labels"][col]:24s} start-of-night nMAD '
                  f'{v["start_nmad"] * sc:7.4f} {u} against per-night-median nMAD '
                  f'{v["night_nmad"] * sc:7.4f} {u}, ratio {v["ratio"]:.2f}x (dimensionless)')
        for col in names:
            sc, u = scale[col]
            line = out['start_slope'].get(col)
            if line is not None:
                print(f'    {out["labels"][col]:24s} against date {line["slope"] * sc:+.5f} +/- '
                      f'{line["slope_err"] * sc:.5f} {u} per d '
                      f'({abs(line["slope"]) / line["slope_err"]:.1f} standard errors), '
                      f'Pearson r {line["pearson_r"]:+.4f}, '
                      f'Spearman rho {line["spearman_rho"]:+.4f}, n {line["n"]}')
    return out


# ------------------------------------------------------------------------------------ figures

def _text_page(pdf, title, lines):
    """Write one text page into the PDF.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    title : `str`
        Page heading.
    lines : `list` [`str`]
        Body, one entry per line; pre-formatted monospace.
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.95, title, fontsize=13, weight='bold', va='top')
    fig.text(0.06, 0.91, '\n'.join(lines), fontsize=7.4, family='monospace', va='top')
    pdf.savefig(fig)
    plt.close(fig)


def figure_sample(pdf, sci, samp):
    """Sample coverage: where the focus error sits night by night, and where the signal lives.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    sci : `pandas.DataFrame`
        Science table, as loaded.
    samp : `dict`
        Result of `section_sample`, whose ``per_night`` frame holds the per-night medians.

    Notes
    -----
    The right-hand panel is the whole case for night-grouped scoring in one plot: collapsing each
    night to its median leaves a clean temperature relation, which is to say the signal is a
    between-night one. The per-visit scatter around it is what the four gradients then address.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    pn = samp['per_night']
    ax = axes[0]
    ax.plot(np.arange(len(pn)), pn['median'], '.', ms=5, color='#1f77b4')
    ax.set_xlabel('night index, in day_obs order')
    ax.set_ylabel('night median focus error\n[um of equivalent hexapod dz]')
    ax.set_title(f'Per-night median over {len(pn)} nights')
    ax.axhline(0, color='0.6', lw=0.8)

    ax = axes[1]
    ax.plot(pn['truss'], pn['median'], 'o', ms=4, color='#2ca02c')
    ax.set_xlabel('night median TMA truss temperature [deg C]')
    ax.set_ylabel('night median focus error\n[um of equivalent hexapod dz]')
    ax.set_title('Between nights, where the signal lives')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def figure_r2(pdf, sci, r2res, features):
    """The three quadratic radial terms against the focus error, raw and partialled.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    sci : `pandas.DataFrame`
        Science table.
    r2res : `dict`
        Result of `section_r2grads`.
    features : `list` [`str`]
        The deliverable features, named in the axis labels as what was partialled out.

    Notes
    -----
    Top row raw, bottom row partialled. The comparison between the two rows is the point: a raw
    relation that survives partialling carries focus information the incumbent gradients do not,
    and one that collapses was the gradients' signal seen twice. Night medians are plotted rather
    than every visit because the relation being tested is a between-night one; the fitted line is
    the Huber fit to all visits, not to the medians.
    """
    cols = r2res['available']
    lab = dict(L.R2_COLS)
    fig, axes = plt.subplots(2, len(cols), figsize=(4.1 * len(cols), 8.0), squeeze=False)
    for j, c in enumerate(cols):
        ax = axes[0][j]
        d = sci[[c, 'y', 'day_obs']].dropna()
        pn = d.groupby('day_obs').median()
        ax.plot(pn[c], pn['y'], 'o', ms=3.5, color='#1f77b4')
        r = r2res['lines'][c]
        xs = np.linspace(pn[c].min(), pn[c].max(), 20)
        ax.plot(xs, r['intercept'] + r['slope'] * xs, '-', color='#d62728', lw=1.2)
        ax.set_xlabel(f'{lab[c]}\n[deg C per unit normalized r^2 amplitude]', fontsize=8)
        ax.set_ylabel('night median focus error\n[um of equivalent hexapod dz]', fontsize=8)
        ax.set_title(f'raw: Pearson r {r["pearson_r"]:+.3f}, '
                     f'Spearman rho {r["spearman_rho"]:+.3f}', fontsize=9)

        ax = axes[1][j]
        ctrl = [f for f in features if f in sci.columns]
        d = sci[[c, 'y', 'day_obs'] + ctrl].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) > 100:
            C = d[ctrl].to_numpy(float)
            res = {}
            for col in (c, 'y'):
                m = F.make_model('huber')
                v = d[col].to_numpy(float)
                m.fit(C, v)
                res[col] = v - m.predict(C)
            e = pd.DataFrame({'x': res[c], 'y': res['y'], 'day_obs': d['day_obs'].to_numpy()})
            pe = e.groupby('day_obs').median()
            ax.plot(pe['x'], pe['y'], 'o', ms=3.5, color='#7f4fa8')
            p = r2res['partial'][c]
            xs = np.linspace(pe['x'].min(), pe['x'].max(), 20)
            ax.plot(xs, p['slope'] * xs, '-', color='#d62728', lw=1.2)
            ax.axhline(0, color='0.7', lw=0.7)
            ax.axvline(0, color='0.7', lw=0.7)
            ax.set_title(f'partialled: Pearson r {p["partial_pearson_r"]:+.3f}, '
                         f'Spearman rho {p["partial_spearman_rho"]:+.3f}', fontsize=9)
        ax.set_xlabel(f'{lab[c]} residual\nafter the {len(ctrl)} deliverable features',
                      fontsize=8)
        ax.set_ylabel('focus error residual\n[um of equivalent hexapod dz]', fontsize=8)
    fig.suptitle('Quadratic radial M1M3 thermal terms, raw (top) and above and beyond the '
                 'bulk gradients (bottom)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def figure_before(pdf, sci, trussonly):
    """Before any correction: what the truss temperature alone can and cannot do.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    sci : `pandas.DataFrame`
        Science table, as loaded.
    trussonly : `dict`
        Result of `section_trussonly`.

    Notes
    -----
    Pairs the truss scatter with the residual the truss-only correction leaves, so the reader sees
    the one-thermometer correction as a distribution rather than as one nMAD in a table. That
    residual, not the uncorrected response, is what the four M1M3 gradients have to improve on.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.plot(sci['truss_temp_mean_c'], sci['y'], ',', color='0.5', alpha=0.5)
    xx = np.linspace(sci['truss_temp_mean_c'].min(), sci['truss_temp_mean_c'].max(), 10)
    line = F.huber_line(sci['truss_temp_mean_c'], sci['y'])
    ax.plot(xx, line['intercept'] + line['slope'] * xx, '-', color='#d62728', lw=1.8,
            label=f'Huber {line["slope"]:+.1f} um per deg C')
    ax.set_xlabel('TMA truss temperature [deg C]')
    ax.set_ylabel('uncorrected response\n[um of equivalent hexapod dz]')
    ax.set_ylim(-800, 1200)
    ax.set_title(f'Truss temperature alone: Pearson r {line["pearson_r"]:+.3f}, '
                 f'Spearman rho {line["spearman_rho"]:+.3f}')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bins = np.linspace(-800, 1200, 120)
    ax.hist(sci['y'], bins=bins, histtype='step', color='0.4',
            label=f'uncorrected, nMAD {trussonly["uncorrected_nmad"]:.1f} um')
    ax.hist(trussonly['resid'], bins=bins, histtype='step', color='#1f77b4',
            label=f'truss only, nMAD {trussonly["nmad"]:.1f} um')
    ax.set_xlabel('[um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('What a truss-only correction leaves')
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    for b, c in BAND_COLOUR.items():
        v = sci.loc[sci.band == b, 'y'].to_numpy(float)
        if len(v) > 50:
            ax.hist(v, bins=np.linspace(-800, 1200, 80), histtype='step', color=c,
                    label=f'{b} (n {len(v)})')
    ax.set_xlabel('uncorrected response [um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('Uncorrected response per band')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(sci['truss_temp_mean_c'], trussonly['resid'], ',', color='#1f77b4', alpha=0.5)
    ax.axhline(0, color='#d62728', lw=1.2)
    ax.set_xlabel('TMA truss temperature [deg C]')
    ax.set_ylabel('truss-only residual\n[um of equivalent hexapod dz]')
    ax.set_ylim(-800, 1200)
    ax.set_title('The truss relation is removed; the scatter is not')
    fig.suptitle('Before the correction: the truss temperature carries most of the focus error',
                 fontsize=11, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf.savefig(fig)
    plt.close(fig)


def figure_training(pdf, hold):
    """The training itself: the same fit on nights it saw and on nights it never saw.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    hold : `dict`
        Result of `section_holdout`.

    Notes
    -----
    Two scatter plots and two residual histograms, train above test, on the axes the training is
    judged on: predicted focus error against measured focus error. The point of the page is that
    the two look the same — the fit does not degrade on nights it never saw, which is what makes
    the correction usable tonight on a night that is not in the fit.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    lo, hi = -800, 1200
    bins = np.linspace(-500, 500, 100)
    for row, name, col in ((0, 'train', '#1f77b4'), (1, 'test', '#d62728')):
        d = hold[name]
        lab = ('nights used to fit the model' if name == 'train'
               else 'nights held out, never seen by the fit')
        ax = axes[row, 0]
        # Measured on x, predicted on y, matching the final-prediction page.
        ax.plot(d['y'], d['pred'], ',', color=col, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], '-', color='0.3', lw=1.2, label='perfect prediction')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('measured focus error [um of equivalent hexapod dz]')
        ax.set_ylabel('predicted focus error\n[um of equivalent hexapod dz]')
        ax.set_title(f'{name.upper()}: {lab}\n{d["n_visits"]} visits over {d["n_nights"]} '
                     f'nights, robust R2 {d["r2_robust"]:+.3f} (dimensionless)', fontsize=9)
        ax.legend(fontsize=7.5, loc='upper left')

        ax = axes[row, 1]
        ax.hist(d['resid'], bins=bins, histtype='step', color=col,
                label=f'measured minus predicted\nnMAD {d["nmad"]:.1f} um')
        ax.axvline(0, color='0.5', lw=0.8)
        ax.set_xlabel('measured minus predicted [um of equivalent hexapod dz]')
        ax.set_ylabel('visits')
        ax.set_title(f'{name.upper()} residual', fontsize=9)
        ax.legend(fontsize=7.5)
    fig.suptitle(f'Training: one split of whole nights, test over train nMAD '
                 f'{hold["optimism"]:.2f}x (dimensionless)', fontsize=11, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def figure_model(pdf, sci, cv, full):
    """The fit: residual per band, prediction against response, and the coefficient spread."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    # Measured on x, predicted on y: the measurement is the independent variable and the
    # prediction the dependent one.
    ax.plot(sci['y'], cv['pred'], ',', color='0.5', alpha=0.5)
    lo, hi = -800, 1200
    ax.plot([lo, hi], [lo, hi], '-', color='#d62728', lw=1.2, label='perfect prediction')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('measured focus error [um of equivalent hexapod dz]')
    ax.set_ylabel('predicted focus error\n[um of equivalent hexapod dz]\n'
                  '(each visit predicted with its own night held out)')
    ax.set_title(f'Final prediction against measurement, all {len(sci)} visits\n'
                 f'R2 {cv["r2"]:+.3f} (dimensionless), residual nMAD {cv["nmad"]:.1f} um',
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left')

    ax = axes[0, 1]
    ax.hist(sci['y'], bins=np.linspace(-800, 1200, 120), histtype='step', color='0.4',
            label=f'uncorrected, nMAD {nmad(sci["y"].to_numpy()):.1f} um')
    ax.hist(cv['resid'], bins=np.linspace(-800, 1200, 120), histtype='step', color='#d62728',
            label=f'corrected, nMAD {cv["nmad"]:.1f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_xlabel('focus error [um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('Focus error before and after the correction')
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    if cv['coefs'] is not None:
        # Each fold's coefficient as a fraction of the five-fold mean, so all five features share
        # one axis. Plotting the raw values instead needs a symlog axis spanning four decades, on
        # which the fold-to-fold spread -- the whole point of the panel -- is narrower than the
        # marker and the reader sees five bare dots.
        c = cv['coefs']
        mean = c.mean(axis=0)
        pos = np.arange(c.shape[1])
        for k in range(c.shape[0]):
            ax.plot(c[k] / mean, pos, 'o', ms=4, mfc='none', color='#1f77b4',
                    label='one fold' if k == 0 else None)
        ax.errorbar(np.ones_like(mean), pos, xerr=c.std(axis=0) / np.abs(mean),
                    fmt='|', ms=10, lw=1.4, color='#d62728',
                    label='mean +/- spread over folds')
        ax.set_yticks(pos)
        ax.set_yticklabels([f'{f.replace("_c_per_m", "").replace("_", " ")}\n{m:+.1f}'
                            for f, m in zip(full['features'], mean)], fontsize=7)
        ax.axvline(1.0, color='0.6', lw=0.8)
        ax.set_xlabel('fold coefficient / five-fold mean coefficient [dimensionless]\n'
                      '(the mean itself, in um of equivalent hexapod dz per feature unit, '
                      'is under each label)')
        ax.set_title(f'Coefficient stability: each of the {c.shape[0]} fits leaves out a\n'
                     f'different fifth of the nights and is refitted on the rest', fontsize=9.5)
        ax.legend(fontsize=7)

    # Physical units, not residual-over-nMAD: the reader wants to know how many um of focus
    # error survive the correction in each band, and dividing by each band's own scale hides
    # exactly that. The axis covers +/-300 um, which holds the bulk for every band; the per-band
    # nMAD in the legend is computed on every finite visit, not on the plotted range.
    ax = axes[1, 1]
    r = np.asarray(cv['resid'], float)
    lo_b, hi_b = -300.0, 300.0
    bins = np.linspace(lo_b, hi_b, 120)
    for b, col in BAND_COLOUR.items():
        v = r[(sci.band == b).to_numpy()]
        v = v[np.isfinite(v)]
        if len(v) > 50:
            ax.hist(np.clip(v, lo_b, hi_b), bins=bins, histtype='step', color=col,
                    density=True, label=f'{b}  nMAD {nmad(v):.0f} um')
    ax.set_yscale('log')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_xlim(lo_b, hi_b)
    ax.set_xlabel('focus error after correction [um of equivalent hexapod dz]\n'
                  '(axis clipped to +/-300 um; beyond piles into the end bins)')
    ax.set_ylabel('density [per um]')
    ax.set_title('Focus error after correction, by band')
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def figure_elevation(pdf, nights):
    """Per-night elevation slopes and the rising-minus-falling difference."""
    if not len(nights):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    a = nights.slope_all.dropna().to_numpy(float)
    ax.hist(a, bins=40, histtype='step', color='#1f77b4')
    ax.axvline(np.median(a), color='#d62728', lw=1.2,
               label=f'median {np.median(a):+.3f}, nMAD {nmad(a):.3f}')
    ax.set_xlabel('per-night residual slope\n[um of equivalent hexapod dz per deg]')
    ax.set_ylabel('nights')
    ax.set_title(f'Elevation slope, {len(a)} nights')
    ax.legend(fontsize=7.5)

    ax = axes[1]
    both = nights.dropna(subset=['difference'])
    if len(both):
        ax.errorbar(both.slope_down, both.slope_up,
                    xerr=both.err_down, yerr=both.err_up, fmt='o', ms=3, lw=0.6,
                    color='#2ca02c')
        lim = np.nanpercentile(np.abs(np.r_[both.slope_up, both.slope_down]), 98)
        ax.plot([-lim, lim], [-lim, lim], '-', color='0.5', lw=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel('falling-leg slope [um per deg]')
    ax.set_ylabel('rising-leg slope [um per deg]')
    ax.set_title(f'Hysteresis test, {len(both)} nights')

    ax = axes[2]
    o = nights.offset_all.dropna().to_numpy(float)
    ax.hist(o, bins=40, histtype='step', color='#7f4fa8')
    ax.axvline(np.median(o), color='#d62728', lw=1.2,
               label=f'median {np.median(o):+.1f}, nMAD {nmad(o):.1f}')
    ax.set_xlabel(f'per-night residual offset at {F.REF_ELEV_DEG:.0f} deg\n'
                  f'[um of equivalent hexapod dz]')
    ax.set_ylabel('nights')
    ax.set_title('What the thermal model leaves per night')
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def figure_fam(pdf, famres):
    """Within-block scatter before and after the correction."""
    if not famres or 'sets' not in famres:
        return
    s = famres['sets'].dropna(subset=['y_p2p', 'y_corrected_p2p'])
    if not len(s):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    lim = float(np.nanpercentile(np.r_[s.y_p2p, s.y_corrected_p2p], 98))
    ax.plot(s.y_p2p, s.y_corrected_p2p, 'o', ms=4, color='#d62728')
    ax.plot([0, lim], [0, lim], '-', color='0.5', lw=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('uncorrected within-set peak-to-peak\n[um of equivalent hexapod dz]')
    ax.set_ylabel('thermally corrected\n[um of equivalent hexapod dz]')
    ax.set_title(f'{famres["n_improved"]} of {famres["n_sets"]} sets improve; '
                 f'ratio {famres["ratio"]:.2f}')

    ax = axes[1]
    ax.hist(s.y_p2p, bins=30, histtype='step', color='0.4',
            label=f'uncorrected, median {s.y_p2p.median():.1f} um')
    ax.hist(s.y_corrected_p2p, bins=30, histtype='step', color='#d62728',
            label=f'corrected, median {s.y_corrected_p2p.median():.1f} um')
    ax.hist(s.pred_p2p, bins=30, histtype='step', color='#1f77b4',
            label=f'prediction swing, median {s.pred_p2p.median():.1f} um')
    ax.set_xlabel('within-set peak-to-peak [um of equivalent hexapod dz]')
    ax.set_ylabel('sets')
    ax.set_title('The prediction swings more than the drift')
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def figure_dof(pdf, dofres):
    """The Trim degrees of freedom that would apply the thermal correction, over all visits.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    dofres : `dict`
        Result of `section_dof`.

    Notes
    -----
    This is the size of the motion an observer would command, not what is left over after
    commanding it. The top row is the two hexapod dz values, which carry the whole defocus. The
    bottom row shows the two bending modes on their own axis in nm, because at 0.0094 and 0.0076 µm
    per unit v-mode 1 they never leave the sub-nanometre range and would be invisible on a µm axis
    — the point of plotting them is to show that they are negligible, not to read a value off them.

    All four axes are binned over the 1st to 99th percentile rather than the full range, with
    everything beyond piled into the end bins, so no visit is dropped from the count. The annotated
    median and nMAD are computed on every finite value, not on the clipped range.
    """
    d = dofres['dof']
    names = dofres['names']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, col, scale, unit in ((axes[0, 0], names[0], 1.0, 'um'),
                                 (axes[0, 1], names[1], 1.0, 'um'),
                                 (axes[1, 0], names[2], 1e3, 'nm'),
                                 (axes[1, 1], names[3], 1e3, 'nm')):
        a = d[col].to_numpy(float) * scale
        a = a[np.isfinite(a)]
        lo, hi = np.percentile(a, [1.0, 99.0])
        bins = np.linspace(lo, hi, 100)
        ax.hist(np.clip(a, lo, hi), bins=bins, histtype='step', color='#1f77b4')
        ax.axvline(float(np.median(a)), color='#d62728', lw=1.2,
                   label=f'median {np.median(a):+.4g} {unit}\nnMAD {nmad(a):.4g} {unit}\n'
                         f'axis clipped to 1st-99th percentile')
        ax.set_xlabel(f'{dofres["labels"][col]} to command as Trim [{unit}]')
        ax.set_ylabel('visits')
        ax.set_title(f'{dofres["labels"][col]}: '
                     f'{dofres["unit"][col]:+.4g} um per unit v-mode 1', fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle(f'The Trim degrees of freedom that would apply the thermal correction, '
                 f'{len(d)} visits (all other v-modes set to zero)', fontsize=11, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def figure_dof_start(pdf, dofres):
    """Start-of-night degrees of freedom against Modified Julian Date, and their distributions.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    dofres : `dict`
        Result of `section_dof`, whose ``start`` frame holds the first visit of each night.

    Notes
    -----
    The first visit of a night is the one the open-loop correction has to set focus for, before any
    wavefront measurement has been folded in, so the size of the Trim motion it asks for is the
    quantity that says how much work the correction is doing when it matters most.

    All four degrees of freedom are shown. The two mirror bending modes are plotted in nm on their
    own axes, since the correction never asks for more than a fraction of a nm of either — they are
    here to show that an observer applying this correction can leave them alone, which is a
    statement worth making explicitly rather than by omission.
    """
    s = dofres['start']
    if 'obs_start_mjd' not in s.columns or not len(s):
        return
    names = dofres['names']
    # The hexapod pair in um, the two bending modes in nm: on a shared um axis the mirror modes
    # would be a flat line on zero and the page would say nothing about them.
    scales = {names[0]: (1.0, 'um'), names[1]: (1.0, 'um'),
              names[2]: (1e3, 'nm'), names[3]: (1e3, 'nm')}
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.4 * len(names)))
    for row, col in enumerate(names):
        scale, unit = scales[col]
        a = s[col].to_numpy(float) * scale
        mjd = s['obs_start_mjd'].to_numpy(float)
        ax = axes[row, 0]
        ax.plot(mjd, a, 'o', ms=4, color='#1f77b4')
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axhline(float(np.nanmedian(a)), color='#d62728', lw=1.2,
                   label=f'median {np.nanmedian(a):+.4g} {unit}')
        ok = np.isfinite(mjd) & np.isfinite(a)
        if ok.sum() > 10:
            line = F.huber_line(mjd[ok], a[ok])
            xs = np.array([mjd[ok].min(), mjd[ok].max()])
            sig = abs(line['slope']) / line['slope_err'] if line['slope_err'] > 0 else np.nan
            ax.plot(xs, line['intercept'] + line['slope'] * xs, '-', color='#2ca02c', lw=1.2,
                    label=f'Huber {line["slope"]:+.4g} +/- {line["slope_err"]:.4g} {unit} per d\n'
                          f'({sig:.1f} standard errors, '
                          f'Pearson r {line["pearson_r"]:+.3f}, '
                          f'Spearman rho {line["spearman_rho"]:+.3f})')
        ax.set_xlabel('start-of-night Modified Julian Date [d]')
        ax.set_ylabel(f'{dofres["labels"][col]}\nto command as Trim [{unit}]')
        ax.set_title(f'{dofres["labels"][col]} at the start of each night, '
                     f'{len(s)} nights', fontsize=9.5)
        ax.legend(fontsize=7.5)

        ax = axes[row, 1]
        ax.hist(a[np.isfinite(a)], bins=40, histtype='step', color='#1f77b4')
        ax.axvline(float(np.nanmedian(a)), color='#d62728', lw=1.2,
                   label=f'median {np.nanmedian(a):+.4g} {unit}\nnMAD {nmad(a):.4g} {unit}')
        ax.set_xlabel(f'{dofres["labels"][col]} to command as Trim [{unit}]')
        ax.set_ylabel('nights')
        ax.set_title(f'{dofres["labels"][col]}, start of night', fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle('Start of night: the Trim the thermal correction would command',
                 fontsize=11, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default=None,
                    help='directory holding the cached tables; default '
                         'thermal_focus/output')
    ap.add_argument('--fam-dir-name', default='fam_danish_1_2',
                    help='subdirectory holding the FAM table')
    ap.add_argument('--day-obs-range', nargs=2, type=int, default=None,
                    metavar=('LO', 'HI'),
                    help='restrict to this inclusive night range as YYYYMMDD')
    ap.add_argument('--model', default='huber', choices=F.MODEL_NAMES,
                    help='model for the deliverable fit')
    ap.add_argument('--no-pdf', action='store_true', help='print the numbers, write no PDF')
    ap.add_argument('--pdf-name', default='thermal_focus.pdf', help='PDF filename')
    args = ap.parse_args()

    out_dir = (pathlib.Path(args.output_dir) if args.output_dir
               else _ROOT / 'thermal_focus' / 'output')
    day_obs_range = tuple(args.day_obs_range) if args.day_obs_range else None

    print('=== 1. the response and the sample ===')
    sci, fam, features = load(out_dir, args.fam_dir_name, day_obs_range)
    samp = section_sample(sci, features)

    print('\n=== 2. the thermal model ===')
    full = F.fit_full(sci, features, model=args.model)
    print()
    cv = F.evaluate(sci, features, model=args.model)

    print('\n=== 3. machine-learning legibility ===')
    split = F.split_comparison(sci, features, model=args.model)
    print()
    models = F.model_comparison(sci, features)
    print()
    coef_tab = F.coefficient_table(cv['coefs'], features,
                                   L.v1_per_um_dz_value(verbose=False))
    print()
    cmd = F.commanded_truss_slope(sci, L.v1_per_um_dz_value(verbose=False))
    print()
    bands = F.per_band_fit(sci, features, model=args.model)

    print('\n=== 4. camera-body temperature ===')
    cam = section_camtemp(sci)

    print('\n=== 5. what adds nothing ===')
    ablation = section_ablation(sci)

    print('\n=== 5b. the quadratic radial M1M3 terms, beyond the bulk gradients ===')
    r2res = section_r2grads(sci, features, model=args.model)

    print('\n=== 6. residual shape ===')
    # Two residuals, two questions. About a truss-only per-band fit the tail is the published
    # one-sided positive excess, which says the truss relation alone leaves a population of
    # visits above it. About the full five-feature fit that asymmetry is largely absorbed, which
    # says the gradients account for much of it. Both argue for robust fitting.
    print('about a truss-only per-band Huber fit:')
    truss_resid = np.full(len(sci), np.nan)
    for band, g in sci.groupby('band'):
        if len(g) < 200:
            continue
        r = F.huber_line(g['truss_temp_mean_c'], g['y'])
        idx = sci.index.get_indexer(g.index)
        truss_resid[idx] = (g['y'].to_numpy(float)
                            - (r['intercept'] + r['slope'] * g['truss_temp_mean_c']
                               .to_numpy(float)))
    tails_truss = F.residual_tail(sci, truss_resid)
    print('\nabout the full five-feature band-independent fit:')
    tails = F.residual_tail(sci, cv['resid'])

    print('\n=== 7. within-night behaviour against elevation ===')
    nights = section_elevation(sci, cv['resid'])

    print('\n=== 8. band changes ===')
    # The claim being tested is that fitting each band separately injects a step at every filter
    # change, because the coefficients swap while nothing physical happens. That needs the
    # per-band-corrected residual in the table, not just the shared one.
    d = sci.copy()
    d['resid'] = cv['resid']
    d['resid_per_band'] = np.nan
    for band, g in sci.groupby('band'):
        if len(g) < 200:
            continue
        r = F.evaluate(g.reset_index(drop=True), features, model=args.model,
                       n_splits=min(F.N_SPLITS, g['day_obs'].nunique()), verbose=False)
        d.loc[g.index, 'resid_per_band'] = r['resid']
    print('median |step| between consecutive visits within a night '
          '[um of equivalent hexapod dz]')
    steps = {'uncorrected': F.band_change_step(d, 'y', label='uncorrected'),
             'per-band models': F.band_change_step(d, 'resid_per_band',
                                                   label='per-band models'),
             'shared thermal model': F.band_change_step(d, 'resid',
                                                        label='shared thermal model')}

    print('\n=== 9. FAM blocks ===')
    famres = section_fam(fam, sci, features)

    print('\n=== 10. the v1 to hexapod dz conversion, per projection scheme ===')
    conv = L.v1_per_um_dz_table()

    print('\n=== 11. the standalone calculator against this fit ===')
    calc = section_calculator(sci, features, full)

    print('\n=== 12. the truss temperature alone, as the correction to beat ===')
    trussonly = section_trussonly(sci, model=args.model)

    print('\n=== 13. one night-level train/test split, for the training page ===')
    hold = section_holdout(sci, features, model=args.model)

    print('\n=== 14. the focus error as degrees of freedom ===')
    dofres = section_dof(sci, cv['pred'], L.v1_per_um_dz_value(verbose=False))

    if args.no_pdf:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / args.pdf_name
    with PdfPages(pdf_path) as pdf:
        # ---------------------------------------------------------- group 1: before correction
        _text_page(pdf, 'Part 1 of 3 - Before the correction: what is being predicted', [
            f'Sample      {len(sci)} science visits over {sci["day_obs"].nunique()} nights, '
            f'day_obs {int(sci.day_obs.min())} to {int(sci.day_obs.max())}',
            '',
            'THE QUANTITY BEING PREDICTED, called the focus error throughout:',
            '',
            f'  focus error = (v1_trim + {L.MEASURED_SIGN:+.1f} * v1) / '
            f'{L.v1_per_um_dz_value(verbose=False):.5e}',
            '              [um of equivalent hexapod dz, 0.5 um on each hexapod]',
            '',
            '  v1       v-mode 1 of the state the corner wavefront sensors MEASURED',
            '  v1_trim  v-mode 1 of the correction the Active Optics System had COMMANDED',
            '',
            '  v-mode 1 is the first singular vector of the AOS sensitivity matrix, which is',
            '  essentially uniform defocus. The difference between commanded and measured is',
            '  the focus error the closed loop had accumulated but not yet removed, and it is',
            '  what an open-loop correction from a temperature table would have to supply.',
            '',
            f'  Uncorrected: median {sci["y"].median():+.1f}, '
            f'nMAD {nmad(sci["y"].to_numpy()):.1f} um of equivalent hexapod dz.',
            '',
            'THE ONE-THERMOMETER CORRECTION, which everything later has to beat:',
            '',
            f'  Fitting the TMA truss temperature alone leaves a residual nMAD of '
            f'{trussonly["nmad"]:.1f} um',
            f'  of equivalent hexapod dz, against the uncorrected '
            f'{trussonly["uncorrected_nmad"]:.1f} um -- an improvement of',
            f'  {trussonly["uncorrected_nmad"] / trussonly["nmad"]:.2f}x (dimensionless, '
            f'uncorrected nMAD over residual nMAD).',
            f'  The fitted line is {trussonly["slope"]:+.2f} um of equivalent hexapod dz per '
            f'deg C,',
            f'  Pearson r {trussonly["pearson_r"]:+.4f}, Spearman rho '
            f'{trussonly["spearman_rho"]:+.4f}, n {len(sci)}.',
            '',
            '  The four M1M3 bulk thermal gradients are added because that residual is still',
            f'  large. With all five channels it falls to {cv["nmad"]:.1f} um; the next part '
            f'shows how.',
            '',
            'Feature means over the sample:',
            *[f'  {c:32s} {sci[c].mean():+12.5f} [{F.FEATURE_UNITS.get(c, "?")}]'
              for c in features],
        ])
        figure_before(pdf, sci, trussonly)
        figure_sample(pdf, sci, samp)

        # ------------------------------------------------- group 2: process and results of training
        _text_page(pdf, 'Part 2 of 3 - Training: the words used, and what they mean', [
            'Three terms appear on every plot in this part. Each names one precaution against',
            'the same trap: a model that recalls which night a visit came from instead of',
            'predicting focus from temperature.',
            '',
            'WHY THE TRAP EXISTS',
            '',
            '  Within one night the thermal telemetry barely moves: only 2.7% of the truss',
            '  temperature variance is within-night (dimensionless, within-night over total),',
            '  while 90.7% of the focus-error variance is between nights. Consecutive visits',
            '  are therefore near-duplicates in temperature but carry that night\'s own focus',
            '  offset. A model given some visits from a night and asked about others from the',
            '  same night can look up the offset rather than derive it, and would then fail on',
            '  a new night -- which is the only case that matters in operation.',
            '',
            'NIGHT-GROUPED',
            '',
            '  Whole nights are kept together. Every visit from one night is either all in the',
            '  training set or all in the test set, never split between them. The grouping is',
            '  on day_obs, through sklearn GroupKFold. This is not one variant among several:',
            '  it is the only scoring in this document that means anything.',
            '',
            'OUT-OF-FOLD PREDICTION',
            '',
            '  The nights are divided into 5 folds. The model is fitted 5 times, each time on',
            '  4 folds and used to predict the 5th. Every visit therefore ends up with exactly',
            '  one prediction, made by a fit that never saw that visit\'s night. Stacking those',
            '  5 sets of predictions gives one prediction per visit over the whole sample --',
            '  that stack is what "out-of-fold prediction" means, and it is what the summary',
            '  plots in Part 3 show. It is an honest prediction for all 68,000 visits at once,',
            '  which no single train/test split can give.',
            '',
            'PER-FOLD COEFFICIENTS',
            '',
            '  Those 5 fits each produce their own 5 coefficients. Comparing them says whether',
            '  the relation is a property of the telescope or of a particular set of nights:',
            '  a coefficient that keeps its sign and magnitude across all 5 fits is real, and',
            '  one that swings or changes sign is fitting whichever nights it was given.',
            '',
            'THE TRAIN/TEST PAGE THAT FOLLOWS',
            '',
            f'  The next page does something simpler, to show the mechanism directly: one',
            f'  single split, {hold["train"]["n_nights"]} nights to fit on and '
            f'{hold["test"]["n_nights"]} nights held out entirely. The nights are drawn at',
            '  random rather than by date, so the held-out nights are spread across the season',
            '  instead of being the last few weeks -- a date split would confound "held out"',
            '  with "late in the season".',
            '',
            f'  Fitted on the {hold["train"]["n_nights"]} training nights: residual nMAD '
            f'{hold["train"]["nmad"]:.1f} um of equivalent hexapod dz',
            f'  On the {hold["test"]["n_nights"]} held-out nights:        residual nMAD '
            f'{hold["test"]["nmad"]:.1f} um of equivalent hexapod dz',
            f'  Test over train {hold["optimism"]:.2f}x (dimensionless) -- the price of a night '
            f'the fit never saw.',
            '',
            '  The nMAD pair above is the comparison to read. The page quotes a ROBUST R2, formed',
            '  from squared nMAD rather than variance, because a few visits with very large focus',
            f'  errors dominate the variance: the training residual has nMAD '
            f'{hold["train"]["nmad"]:.1f} um but standard',
            f'  deviation {float(np.nanstd(hold["train"]["resid"])):.1f} um of equivalent hexapod '
            f'dz. Those visits fall on training nights, so an',
            f'  ordinary variance R2 reads {hold["train"]["r2"]:+.3f} on train against '
            f'{hold["test"]["r2"]:+.3f} on test -- worse on the nights',
            '  the model was fitted to, for a reason that is about which nights hold the outliers',
            f'  and not about the fit. The robust values are {hold["train"]["r2_robust"]:+.3f} '
            f'train and {hold["test"]["r2_robust"]:+.3f} test, which agree with',
            '  the nMAD ratio. Both are computed against the whole sample, not each subset\'s own',
            '  spread, so that train and test are on one scale.',
        ])
        figure_training(pdf, hold)

        _text_page(pdf, 'Part 2 of 3 - Training: the fitted model', [
            'The fitted equation, focus error in um of equivalent hexapod dz:',
            '',
            f'  = {full["intercept"]:+.2f}',
            *[f'    {c:+10.2f} * {f:32s} [per {F.FEATURE_UNITS.get(f, "?")}]'
              for f, c in zip(full['features'], full['coef'])],
            '',
            f'  reproduces the fitted pipeline to '
            f'{full["equation_max_abs_diff"]:.2e} um of equivalent hexapod dz',
            '',
            f'Night-grouped residual nMAD {cv["nmad"]:.1f} um of equivalent hexapod dz, '
            f'R2 {cv["r2"]:+.3f} (dimensionless)',
            f'Visit-level (LEAKY) residual nMAD {split["leaky_nmad"]:.1f} um; '
            f'optimism {split["optimism"]:.2f}x (dimensionless)',
            '',
            'Model comparison, night-grouped [um of equivalent hexapod dz]:',
            *[f'  {r.model:12s} nMAD {r.resid_nmad:7.1f}   R2 {r.r2:+.3f}   '
              f'improvement {r.improvement:.2f}x (dimensionless)'
              for _, r in models.iterrows()],
            '',
            'Per-fold coefficients [um of equivalent hexapod dz per feature unit]:',
            *[f'  {r.feature:32s} {r["mean"]:+10.2f} +/- {r["std"]:8.2f}'
              f'{"" if r.sign_stable else "   SIGN FLIPS across folds"}'
              for _, r in coef_tab.iterrows()],
            '',
            f'Commanded truss slope against the FAM Double Zernike value '
            f'{F.FAM_TRUSS_SLOPE:+.5f}',
            '[dimensionless v-mode-1 amplitude of the Trim per deg C]:',
            *[f'  {r.band:>4s}  n {int(r.n):6d}  {r.slope:+.5f} +/- {r.slope_err:.5f}  '
              f'difference {r.difference:+.5f} ({r.n_sigma:.1f} standard errors)'
              for _, r in cmd.iterrows()],
            '  This, not the fitted response coefficient, is the like-for-like test: the FAM',
            '  value is a slope of the COMMANDED Trim, while the response is Trim minus the',
            '  measured state and so a different quantity.',
        ])
        figure_model(pdf, sci, cv, full)

        _text_page(pdf, 'Part 2 of 3 - Training: what the model does and does not buy', [
            'Feature ablation, night-grouped [residual nMAD in um of equivalent hexapod dz]:',
            *[f'  {r.groups:44s} {int(r.n_features):2d} feat  nMAD {r.resid_nmad:6.1f}  '
              f'R2 {r.r2:+.3f}' + (f'  gain {r.gain:.3f}x' if 'gain' in ablation.columns else '')
              for _, r in ablation.iterrows()],
            '  gain is dimensionless, deliverable-set nMAD over this set nMAD',
            '',
            'Per band, night-grouped [um of equivalent hexapod dz]:',
            *[f'  {r.band}  n {int(r.n):6d}  uncorrected {r.uncorrected_nmad:6.1f}  '
              f'shared {r.shared_nmad:6.1f}  own {r.own_nmad:6.1f}  '
              f'own truss {r.own_truss:+8.2f} um per deg C' for _, r in bands.iterrows()],
            '',
            'Band-change step, median |step| between consecutive visits within a night',
            '[um of equivalent hexapod dz]:',
            *[f'  {k:22s} band change {v["median_change"]:7.1f} (n {v["n_change"]})  '
              f'same band {v["median_same"]:7.1f} (n {v["n_same"]})  '
              f'ratio {v["ratio"]:.2f} (dimensionless)' for k, v in steps.items()],
            '',
            f'Residual tails beyond +/-3 nMAD, per band '
            f'(a Gaussian gives 0.135% on each side):',
            '  about a truss-only per-band Huber fit --',
            *[f'    {r.band}  above {r.frac_above:5.2f}%  below {r.frac_below:5.2f}%  '
              f'ratio {r.ratio:5.1f} (dimensionless, above over below)'
              for _, r in tails_truss.iterrows()],
            '  about the full five-feature band-independent fit --',
            *[f'    {r.band}  above {r.frac_above:5.2f}%  below {r.frac_below:5.2f}%  '
              f'ratio {r.ratio:5.1f} (dimensionless, above over below)'
              for _, r in tails.iterrows()],
            '  Both residuals are far heavier-tailed than a Gaussian in every band, which is',
            '  the reason the fits are robust rather than ordinary least squares. The direction',
            '  is not uniform here: i, r, y and z are tail-heavy positive about the truss-only',
            '  fit while g and u are tail-heavy negative. An earlier one-sided-positive result',
            '  was measured on v1_total, which includes the hexapod look-up-table term, in',
            '  dimensionless v-mode units over four bands -- a different statistic from this',
            '  one, so the two are not expected to agree band by band.',
            '',
            'Camera-body temperature as an alternative thermometer:',
            *[f'  {c:22s} resolves {v:.2f}% of visits' for c, v in cam['coverage'].items()],
            *[f'  {n:26s} slope {cam[k]["slope"]:+8.2f} +/- {cam[k]["slope_err"]:.2f} '
              f'um of equivalent hexapod dz per deg C, Pearson r {cam[k]["pearson_r"]:+.4f}'
              for k, n in (('truss_line', 'TMA truss temperature'),
                           ('cam_line', 'camera-body AverageTemp')) if cam.get(k)],
        ])

        r2_lines = ['Quadratic radial section skipped: the cached table carries none of the '
                    'm1m3_r2_coeff_c, m1_r2_coeff_c or m3_r2_coeff_c columns. Run',
                    'value_added/code/build_m1m3_thermal_r2.py, then rebuild the cached table.']
        if r2res['available']:
            lab = dict(L.R2_COLS)
            r2_lines = [
                'A temperature field going as radius squared bends the mirror much closer to',
                'pure defocus than a linear radial ramp does, so it is the term most likely to',
                'move focus. Three are fitted, over three thermocouple populations: the whole',
                'mirror, the M1 annulus alone and the M3 inner disc alone. The unit is deg C per',
                'unit normalized radius-squared amplitude -- the quadratic shape is orthogonal',
                'to the constant, linear-radius and depth terms and scaled to unit',
                'root-mean-square over each population\'s own sensors, so it carries only the',
                'curvature those terms cannot express.',
                '',
                'Raw relation to the focus error:',
                *[f'  {lab[c]:36s} {r2res["coverage"][c]:5.2f}% of visits  slope '
                  f'{r2res["lines"][c]["slope"]:+8.1f} +/- '
                  f'{r2res["lines"][c]["slope_err"]:6.1f} um per unit amplitude  '
                  f'Pearson r {r2res["lines"][c]["pearson_r"]:+.4f}  '
                  f'Spearman rho {r2res["lines"][c]["spearman_rho"]:+.4f}'
                  for c in r2res['available']],
                '',
                'How much each duplicates the existing M1M3 radial gradient (dimensionless):',
                *[f'  {lab[c]:36s} Pearson r {r2res["redundancy"][c]["pearson_r"]:+.4f}  '
                  f'Spearman rho {r2res["redundancy"][c]["spearman_rho"]:+.4f}  '
                  f'n {r2res["redundancy"][c]["n"]}'
                  for c in r2res['available'] if r2res['redundancy'].get(c)],
                '',
                'Partial correlation with the focus error, both sides stripped of the truss',
                'temperature and the four bulk gradients -- the "above and beyond" test:',
                *[f'  {lab[c]:36s} raw r {r2res["partial"][c]["raw_pearson_r"]:+.4f} -> '
                  f'partial r {r2res["partial"][c]["partial_pearson_r"]:+.4f}  '
                  f'partial rho {r2res["partial"][c]["partial_spearman_rho"]:+.4f}  '
                  f'slope {r2res["partial"][c]["slope"]:+8.1f} +/- '
                  f'{r2res["partial"][c]["slope_err"]:6.1f} um per unit amplitude'
                  for c in r2res['available'] if r2res['partial'].get(c)],
                '',
                'Night-grouped nested comparison -- does the surviving information generalise',
                'to nights the fit never saw [residual nMAD in um of equivalent hexapod dz]:',
                *[f'  baseline + {n:8s} n {r["n"]:6d} over {r["n_nights"]:3d} nights  '
                  f'nMAD {r["nmad_base"]:6.1f} -> {r["nmad_extended"]:6.1f}  '
                  f'gain {r["gain"]:.4f}  delta R2 {r["delta_r2"]:+.4f}'
                  for n, r in r2res['nested'].items()],
                '  gain and delta R2 are dimensionless; gain is baseline nMAD over extended',
            ]
            if r2res['swap']:
                s = r2res['swap']
                ratio = (s['gradients']['nmad'] / s['quadratic']['nmad']
                         if s['quadratic']['nmad'] else float('nan'))
                r2_lines += [
                    '',
                    f'Substitution rather than addition, on the same {s["n"]} visits over '
                    f'{s["n_nights"]} nights:',
                    f'  truss + four bulk gradients   nMAD {s["gradients"]["nmad"]:6.1f} um of '
                    f'equivalent hexapod dz  R2 {s["gradients"]["r2"]:+.4f}',
                    f'  truss + three quadratic terms nMAD {s["quadratic"]["nmad"]:6.1f} um of '
                    f'equivalent hexapod dz  R2 {s["quadratic"]["r2"]:+.4f}',
                    f'  ratio {ratio:.4f} (dimensionless, gradient nMAD over quadratic nMAD); '
                    'above 1 favours switching',
                ]
        _text_page(pdf, 'Part 2 of 3 - Training: the quadratic radial M1M3 terms', r2_lines)
        if r2res['available']:
            figure_r2(pdf, sci, r2res, features)

        elev_lines = ['Elevation section skipped: the cached table lacks altitude_deg or '
                      'obs_start_mjd']
        if len(nights):
            a = nights.slope_all.dropna().to_numpy(float)
            o = nights.offset_all.dropna().to_numpy(float)
            both = nights.dropna(subset=['difference'])
            elev_lines = [
                f'Per-night slope of the residual against elevation, {len(a)} nights',
                f'  median {np.median(a):+.3f}, nMAD {nmad(a):.3f} um of equivalent hexapod '
                f'dz per deg',
                f'  median formal error {nights.err_all.median():.3f} um per deg',
                '',
                f'Per-night residual offset at {F.REF_ELEV_DEG:.0f} deg elevation, '
                f'{len(o)} nights',
                f'  median {np.median(o):+.1f}, nMAD {nmad(o):.1f} um of equivalent hexapod dz',
                f'  within-night residual nMAD {nights.resid_nmad_all.median():.1f} um; '
                f'night-to-night over within-night '
                f'{nmad(o) / nights.resid_nmad_all.median():.2f} (dimensionless)',
                '',
                f'Rising minus falling slope, {len(both)} nights with both legs',
                f'  median {both.difference.median():+.3f}, '
                f'nMAD {nmad(both.difference.to_numpy(float)):.3f} um per deg',
                f'  nights beyond 3 combined standard errors: '
                f'{int((both.difference_sigma.abs() > 3).sum())} of {len(both)}',
                f'  rising steeper than falling on {int((both.difference > 0).sum())} '
                f'of {len(both)} nights',
            ]
        fam_lines = ['FAM section skipped: no FAM table, or it lacks the thermal features']
        if famres:
            fam_lines = [
                f'Within-set scatter over {famres["n_sets"]} clean 12-triplet FAM sets '
                f'[um of equivalent hexapod dz]',
                f'  uncorrected response, median peak-to-peak : '
                f'{famres["median_p2p_uncorrected"]:.1f}',
                f'  thermally corrected, median peak-to-peak  : '
                f'{famres["median_p2p_corrected"]:.1f}',
                f'  the prediction\'s own within-set swing     : '
                f'{famres["median_p2p_prediction"]:.1f}',
                f'  ratio {famres["ratio"]:.2f} (dimensionless, corrected over uncorrected); '
                f'{famres["n_improved"]} of {famres["n_sets"]} sets improve',
                '',
                'A between-night model is not a within-block model, because the term the model',
                'actually predicts does not move inside a block. The response is',
                '(v1_trim - v1) / v1_per_um_dz, and the fit is dominated by the commanded',
                'v1_trim term: 91.2% between-night variance fraction (dimensionless, between',
                'over total) against 25.0% for the measured v1. Inside a FAM block the AOS does',
                'not re-command Trim, so v1_trim is exactly frozen and the response reduces to',
                '-v1 / v1_per_um_dz -- the measured term alone, of the opposite sign. That is',
                'why the within-set slope against truss temperature is -81.10 +/- 17.58 against',
                '+124.38 um of equivalent hexapod dz per deg C between nights.',
                '',
                'This is not telemetry noise (the truss is resolved within a set, 12 distinct',
                'values, and the reversed slope is about 4 permutation-null sigma out) and not a',
                'sign error: adding the prediction rather than subtracting it does reduce the',
                'scatter, to 32.7 um from 34.9 um, but that fits the measured term with a model',
                'of the commanded term and would not survive a block in which Trim moved.',
            ]
        # -------------------------------------------- group 3: summary over all the data
        _text_page(pdf, 'Part 3 of 3 - All the data: elevation, FAM blocks and the conversion', [
            *elev_lines, '', *fam_lines, '',
            'v-mode-1 to hexapod dz conversion, per projection scheme',
            '[dimensionless v-mode-1 amplitude per um; um of hexapod dz per unit v1]',
            *[f'  {r["dof_set"]:>12s}/{r["n_modes"]:<3d} mean {r["mean_mag"]:.5e} per um   '
              f'{r["um_per_v1_shared"]:8.1f} um shared / {r["um_per_v1_camera"]:8.1f} um '
              f'camera-alone   axes agree {r["axes_agree_pct"]:.2f}%' for r in conv],
            '',
            'The shared and camera-alone columns are two definitions of "equivalent dz", not',
            'two estimates of one number: shared splits the motion between the camera and M2',
            'hexapods, camera-alone holds M2 still. The study reports the shared convention.',
            '',
            *([f'Standalone calculator (trim_calculator.py, numpy only) against this fit, over',
               f'{calc["n"]} visits: max |difference| {calc["max_abs_diff_um"]:.4f}, median '
               f'{calc["median_diff_um"]:+.4f} um of equivalent hexapod dz.',
               f'Its {len(T.TEST_CASES)} worked test cases agree with their stated values to '
               f'{calc["worked_max_abs_diff_um"]:.4f} um.',
               'The difference is rounding: the calculator inlines each coefficient to two',
               'decimals so it can be copied to a summit machine and read by eye.']
              if calc else
              ['The standalone calculator was not compared: the fitted feature set here is not',
               'its five thermal channels.']),
        ])
        figure_elevation(pdf, nights)
        figure_fam(pdf, famres)

        # The hexapod pair is tens of um; the two bending modes are thousandths of a um. Printing
        # both in um makes the bending rows read as +0.0013 +/- 0.0000, so they are shown in nm.
        _nm_cols = set(dofres['names'][2:])
        _sc = lambda c: 1e3 if c in _nm_cols else 1.0
        _un = lambda c: 'nm' if c in _nm_cols else 'um'
        _text_page(pdf, 'Part 3 of 3 - All the data: the correction as degrees of freedom', [
            'The correction above is one number per visit, a focus error in um of equivalent',
            'hexapod dz. An observer acts on degrees of freedom (DOF), so this part converts the',
            'PREDICTED correction into the Trim DOF that would apply it: how far each DOF would',
            'have to move to put the thermal correction on the telescope.',
            '',
            '  v1_applied = predicted focus error * v1_per_um_dz, then back-projected to DOF.',
            '  The commanded Trim term enters the response with a positive sign, so the amplitude',
            '  needed in the Trim to cancel a predicted error is that error in v-mode-1 units,',
            '  with no sign flip. These are motions to command, not residuals left over after',
            '  commanding them -- how well the correction works is what Part 2 measures.',
            '',
            'HOW THE CONVERSION IS DONE',
            '',
            '  dof = normalization_matrix @ (v_modes @ Vh), which is ts_ofc\'s own inverse,',
            '  StateEstimator.get_dofs_from_vmodes, at dof_set all_50 with 34 modes retained.',
            '  The normalization matrix is not optional: using Vh[0] alone gives a DOF vector',
            '  whose forward projection is v1 = +0.0141 with another mode at 0.227, instead of',
            '  the v1 = +1.0000000000 with a largest other mode of 2.6e-16 that the normalized',
            '  inverse round-trips to.',
            '',
            '  All other v-modes are set to zero. Vh is orthonormal, so that is not an',
            '  approximation but the exact minimum-norm DOF vector consistent with the applied',
            '  v1 -- the unique answer with no component in any other mode. It is the smallest',
            '  motion that delivers the required defocus, which is what an observer wants.',
            '',
            'WHAT v-MODE 1 CONTAINS, at v1 = 1.0 (dimensionless)',
            '',
            *[f'  {dofres["labels"][c]:24s} {dofres["unit"][c]:+14.4f} um per unit v1'
              for c in dofres['names']],
            '',
            '  Two DOF carry the defocus, the camera and M2 hexapod dz, and they move together',
            '  in a fixed ratio because v-mode 1 is one direction in DOF space. The two mirror',
            '  bending modes are real but tiny: at the 99th-percentile |v1| the correction asks',
            f'  for, they reach {abs(dofres["unit"][dofres["names"][2]]) * float(np.nanpercentile(np.abs(dofres["dof"]["v1_applied"]), 99)) * 1e3:.3f} nm and '
            f'{abs(dofres["unit"][dofres["names"][3]]) * float(np.nanpercentile(np.abs(dofres["dof"]["v1_applied"]), 99)) * 1e3:.3f} nm, '
            f'so an observer can leave them alone. M2 bending mode B4',
            '  does not appear at all: it enters at +0.0002 um per unit v1, below even those.',
            '',
            'PER-VISIT TRIM TO COMMAND',
            '',
            *[f'  {dofres["labels"][c]:24s} median '
              f'{np.nanmedian(dofres["dof"][c].to_numpy(float)) * _sc(c):+10.4f}  nMAD '
              f'{nmad(dofres["dof"][c].to_numpy(float)) * _sc(c):9.4f}  '
              f'p1 {np.nanpercentile(dofres["dof"][c].to_numpy(float), 1) * _sc(c):+10.4f}  '
              f'p99 {np.nanpercentile(dofres["dof"][c].to_numpy(float), 99) * _sc(c):+10.4f} '
              f'{_un(c)}'
              for c in dofres['names']],
            '',
            f'START OF NIGHT, the first visit of each of {len(dofres["start"])} nights',
            '',
            *[f'  {dofres["labels"][c]:24s} median '
              f'{np.nanmedian(dofres["start"][c].to_numpy(float)) * _sc(c):+10.4f}  nMAD '
              f'{nmad(dofres["start"][c].to_numpy(float)) * _sc(c):9.4f} {_un(c)}'
              for c in dofres['names']],
            '',
            '  The first visit of a night is the one an open-loop correction has to set focus',
            '  for, before any wavefront measurement has been folded in, so this is the size of',
            '  the Trim motion the correction asks for when it matters most.',
            '',
            *[f'  {dofres["labels"][c]:24s} start-of-night nMAD '
              f'{dofres["start_vs_night"][c]["start_nmad"] * _sc(c):7.4f} {_un(c)} against the '
              f'per-night-median '
              f'{dofres["start_vs_night"][c]["night_nmad"] * _sc(c):7.4f} {_un(c)}, '
              f'{dofres["start_vs_night"][c]["ratio"]:.2f}x'
              for c in dofres['names']],
            '',
            '  The Trim asked for at the start of a night is further from its own night\'s centre',
            '  than that centre is from the season\'s. The comparison is against the',
            '  night-to-night spread of the',
            '  per-night medians over the same nights, not against the all-visit nMAD, which mixes',
            '  within-night and between-night scatter over every visit.',
            '',
            *[f'  {dofres["labels"][c]:22s} {dofres["start_slope"][c]["slope"] * _sc(c):+10.5f} '
              f'+/- {dofres["start_slope"][c]["slope_err"] * _sc(c):8.5f} {_un(c):2s} per d  '
              f'({abs(dofres["start_slope"][c]["slope"]) / dofres["start_slope"][c]["slope_err"]:.1f}'
              f' std err)  r {dofres["start_slope"][c]["pearson_r"]:+.4f}  '
              f'rho {dofres["start_slope"][c]["spearman_rho"]:+.4f}'
              for c in dofres['names'] if c in dofres['start_slope']],
            '  (slope against date; r is Pearson, rho is Spearman, over '
            f'{len(dofres["start"])} nights)',
            '',
            '  Every DOF gives the same significance, Pearson r and Spearman rho, because all four',
            '  are a fixed multiple of the one v-mode-1 amplitude. At 2.4 standard errors over',
            f'  {len(dofres["start"])} nights this is a weak positive trend, not a detection: the',
            '  Trim the correction asks for at the start of a night is mostly scatter about a fixed',
            '  offset. It is worth re-testing as the season lengthens rather than quoting as a',
            '  measured drift.',
        ])
        figure_dof(pdf, dofres)
        figure_dof_start(pdf, dofres)
    print(f'\nwrote {pdf_path}')


if __name__ == '__main__':
    main()
