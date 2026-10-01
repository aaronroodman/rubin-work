"""The thermal-focus analysis: every fit and cross-check, one PDF, no network.

Reads the cached tables written by ``run_thermal_focus.py`` and produces one document. The
headline is that the telescope's uniform-defocus error is predictable from five thermal
channels — the Telescope Mount Assembly (TMA) truss temperature and the four M1M3 bulk
thermal gradients — with one band-independent Huber robust linear model.

Computed sections, in the order they are printed:

1. The open-loop focus and the sample — the selection funnel, per-band coverage and the nightly
   medians, including the outlier nights.
2. Settling the model — each candidate telemetry term added to a mean truss temperature baseline
   on its own and ranked, then added cumulatively strongest-first, so the feature set the
   deliverable carries rests on one comparable sequence rather than on scattered tables.
3. The deliverable thermal model — the fit, in sample over every night, the model comparison and
   the Full Array Mode (FAM) cross-check of the truss slope.
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
12. The truss temperature alone — the one-thermometer correction the five-channel model must beat.
13. The focus correction as degrees of freedom — each visit's measured v-mode 1 back-projected into
    the camera and M2 hexapod dz it is built from, over all visits and at the start of each night.
14. Against the Trim the initial alignment block settled on — an independent measurement of the
    focus the telescope needed, not derived from the quantity the model was fitted to.

The PDF runs in one linear order, which is not the order above: the study description and the
summary plots first, then the telemetry-term comparisons that settle the model, then the resulting
trims. The page sequence is the study, the open-loop focus by band and against truss temperature,
the truss temperature over the whole database, the nightly medians, how the model is settled, the
individual and cumulative term grids, the term summary, the fitted model, the quadratic radial
terms, elevation and hysteresis, FAM blocks, and the correction as degrees of freedom.

Every residual nMAD in the document is **in sample**, fitted on every night. A linear fit carrying
one slope per telemetry quantity cannot memorise a night, so a fold-based presentation measured
nothing the one retained optimism check does not state in a single line; see `section_terms`.

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

#: Robust deviations beyond which a night median is called an outlier [dimensionless, deviation
#: over the nMAD of the night medians]. Used for the per-night open-loop focus and for the
#: BLOCK-T539 Trim, so the two pages name outliers on one definition.
OUTLIER_NIGHT_Z = 4.0

#: Candidate telemetry terms evaluated one at a time against the mean truss temperature baseline,
#: then added cumulatively strongest-first. Each is ``(column, short label)``.
#:
#: The whole-mirror ``m1m3_r2_coeff_c`` is deliberately absent: the quadratic radial terms enter
#: here as the M1 annulus and the M3 inner disc separately, which is the split that could carry
#: information the four bulk gradients cannot. The whole-mirror term is still tested, against the
#: bulk gradients and partialled, in `section_r2grads`.
CANDIDATE_TERMS = (
    ('m1m3_z_gradient_c_per_m', 'M1M3 z gradient'),
    ('m1m3_y_gradient_c_per_m', 'M1M3 y gradient'),
    ('m1m3_radial_gradient_c_per_m', 'M1M3 radial gradient'),
    ('m1m3_x_gradient_c_per_m', 'M1M3 x gradient'),
    ('m1_r2_coeff_c', 'M1 quadratic radial'),
    ('m3_r2_coeff_c', 'M3 quadratic radial'),
    ('cam_AverageTemp', 'camera-body temperature'),
)

#: The baseline every candidate term is added to: the mean TMA truss temperature alone.
BASELINE_FEATURES = ('truss_temp_mean_c',)


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
    t539 : `pandas.DataFrame` or `None`
        One row per night of the initial alignment block, or None when the file is absent.
    truss_all : `pandas.DataFrame` or `None`
        Mean TMA truss temperature for every exposure in the database, or None when the file is
        absent. Deliberately **not** restricted by ``day_obs_range``: its purpose is to show the
        whole range of conditions the fitted span sits inside.
    features : `list` [`str`]
        The deliverable feature column names.
    funnel : `dict`
        The selection counts, so the opening page can state what was excluded and by how much:
        ``n_cached``, ``n_span``, ``n_hot_visits``, ``n_hot_nights``, ``n_no_features`` and
        ``n_lut_nights`` (the LUT-epoch nights cut in the build stage, from
        `thermal_focus_lib.LUT_EPOCH_OFFSET_NIGHTS`).

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
    n_before_feat = len(sci)
    sci = sci[sci[features].notna().all(axis=1)].reset_index(drop=True)
    funnel = dict(n_cached=n_cached, n_span=n_span, n_hot_visits=n_hot_visits,
                  n_hot_nights=n_hot_nights, n_no_features=n_before_feat - len(sci),
                  n_lut_nights=len(L.LUT_EPOCH_OFFSET_NIGHTS))

    fam_path = out_dir / fam_dir_name / 'thermal_focus_fam.parquet'
    fam = pd.read_parquet(fam_path) if fam_path.exists() else None
    if fam is not None and day_obs_range:
        fam = fam[(fam.day_obs >= day_obs_range[0]) & (fam.day_obs <= day_obs_range[1])]
    if fam is not None and 'truss_temp_mean_c' in fam.columns:
        fam = fam[~(fam['truss_temp_mean_c'] > L.TRUSS_TEMP_MAX_C)]

    # The build stage already applied the LUT-epoch and truss-temperature cuts to this table, and
    # the features it carries are suffixed `_first`, so the row filter above does not apply here.
    t539_path = out_dir / 'thermal_focus_t539.parquet'
    t539 = pd.read_parquet(t539_path) if t539_path.exists() else None
    if t539 is not None and day_obs_range:
        t539 = t539[(t539.day_obs >= day_obs_range[0])
                    & (t539.day_obs <= day_obs_range[1])].reset_index(drop=True)

    # No day_obs restriction and no cut of any kind: this table is the context the fitted sample
    # is shown against, so narrowing it to the fitted span would defeat its purpose.
    truss_path = out_dir / 'thermal_focus_truss_all.parquet'
    truss_all = pd.read_parquet(truss_path) if truss_path.exists() else None

    if verbose:
        print(f'cached science visits                 : {n_cached}')
        if day_obs_range:
            print(f'  within day_obs {day_obs_range[0]} to {day_obs_range[1]}   : {n_span}')
        print(f'  truss temperature above {L.TRUSS_TEMP_MAX_C:.0f} deg C   : '
              f'-{n_hot_visits} visits on {n_hot_nights} nights')
        print(f'  with all {len(features)} deliverable features : {len(sci)}')
        print(f'  -> {len(sci)} visits, {sci["day_obs"].nunique()} nights, day_obs '
              f'{int(sci["day_obs"].min())} to {int(sci["day_obs"].max())}')
        print(f'open-loop focus [um of equivalent hexapod dz]: '
              f'median {sci["y"].median():+.1f}, nMAD {nmad(sci["y"].to_numpy()):.1f}')
        per_band = sci['band'].value_counts()
        print('per band: ' + ', '.join(f'{b} {int(per_band.get(b, 0))}' for b in BAND_COLOUR
                                      if b in per_band.index))
        if fam is not None:
            print(f'FAM triplets: {len(fam)} over {fam["day_obs"].nunique()} nights')
        if t539 is not None:
            print(f'initial alignment runs: {len(t539)} nights, day_obs '
                  f'{int(t539["day_obs"].min())} to {int(t539["day_obs"].max())}')
        if truss_all is not None:
            nt = truss_all['truss_temp_mean_c'].notna()
            print(f'database-wide truss cache: {len(truss_all)} exposures over '
                  f'{truss_all["day_obs"].nunique()} nights, day_obs '
                  f'{int(truss_all["day_obs"].min())} to {int(truss_all["day_obs"].max())}, '
                  f'{int(nt.sum())} with a truss sample ({100 * nt.mean():.1f}%)')
    return sci, fam, t539, truss_all, features, funnel


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
        ``feature_means`` (`dict`), ``interp_frac`` (per cent of visits whose truss
        temperature was filled by within-night interpolation), ``night_line``
        (`thermal_focus_fit.huber_line` of night-median open-loop focus against night-median truss
        temperature) and ``outlier_nights`` (`pandas.DataFrame`: the nights whose median
        open-loop focus lies more than `OUTLIER_NIGHT_Z` robust deviations from the median of the
        night medians, with ``day_obs``, ``n``, ``median`` [µm of equivalent hexapod dz],
        ``truss`` [°C] and ``z``).

    Notes
    -----
    The outlier nights are flagged rather than removed. A night sitting far from the truss relation
    is a fact about the telescope on that night, and the point of naming it is so a reader can
    recognise it elsewhere in the document; the fits are robust, so these nights are already
    down-weighted rather than driving the slope.
    """
    per_band = (sci.groupby('band')
                .agg(n=('y', 'size'), median=('y', 'median'),
                     nmad=('y', lambda v: nmad(v.to_numpy(float))))
                .reset_index())
    per_night = (sci.groupby('day_obs')
                 .agg(n=('y', 'size'), median=('y', 'median'),
                      truss=('truss_temp_mean_c', 'median'))
                 .reset_index())

    # Robust z of each night median against the spread of the night medians themselves -- not
    # against the all-visit nMAD, which mixes within-night and between-night scatter.
    m = per_night['median'].to_numpy(float)
    centre, spread = float(np.median(m)), float(nmad(m))
    per_night['z'] = (m - centre) / spread if spread else np.nan
    outliers = (per_night[per_night['z'].abs() > OUTLIER_NIGHT_Z]
                .sort_values('z', key=abs, ascending=False).reset_index(drop=True))
    night_line = F.huber_line(per_night['truss'], per_night['median'])

    means = {c: float(sci[c].mean()) for c in features}
    interp = 0.0
    if 'truss_temp_mean_c_interpolated' in sci.columns:
        interp = 100.0 * float(sci['truss_temp_mean_c_interpolated'].fillna(False).mean())
    if verbose:
        print('feature means over the sample')
        for c, v in means.items():
            print(f'  {c:32s} {v:+10.5f} {F.FEATURE_UNITS.get(c, "?")}')
        print(f'truss temperature filled by within-night interpolation: {interp:.1f}% of visits')
        print(f'per-night open-loop focus median spans {per_night["median"].min():+.1f} to '
              f'{per_night["median"].max():+.1f} um of equivalent hexapod dz over '
              f'{len(per_night)} nights')
        print(f'night medians against night-median truss temperature: slope '
              f'{night_line["slope"]:+.2f} +/- {night_line["slope_err"]:.2f} um of equivalent '
              f'hexapod dz per deg C,')
        print(f'  Pearson r {night_line["pearson_r"]:+.4f}, Spearman rho '
              f'{night_line["spearman_rho"]:+.4f}, n {night_line["n"]} nights')
        print(f'outlier nights, beyond {OUTLIER_NIGHT_Z:.0f} robust deviations of the night '
              f'medians (centre {centre:+.1f}, nMAD {spread:.1f} um of equivalent hexapod dz): '
              f'{len(outliers)} of {len(per_night)}')
        for _, r in outliers.iterrows():
            print(f'  day_obs {int(r.day_obs)}  n {int(r.n):5d}  median open-loop focus '
                  f'{r["median"]:+8.1f} um of equivalent hexapod dz  truss {r.truss:+6.2f} deg C  '
                  f'z {r.z:+.1f} (dimensionless)')
    return dict(per_band=per_band, per_night=per_night, feature_means=means,
                interp_frac=interp, night_line=night_line, outlier_nights=outliers)


def section_terms(sci, model='huber', verbose=True):
    """Evaluate each candidate telemetry term against a mean truss temperature baseline.

    Two passes over one fixed set of rows. First each candidate of `CANDIDATE_TERMS` is added to
    the truss-only baseline on its own, and the candidates are ranked by the residual nMAD each
    leaves. Then they are added cumulatively in that ranked order, so a reader can see where the
    sequence stops paying.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, carrying ``y``, ``day_obs`` and the candidate columns.
    model : `str`, optional
        Key for `thermal_focus_fit.make_model`.
    verbose : `bool`, optional
        Print both tables and the ranking.

    Returns
    -------
    out : `dict`
        ``n_all`` and ``n_common`` (visits in the sample and in the common-row subset),
        ``available`` and ``missing`` (candidate columns present and absent),
        ``baseline`` (the truss-only model entry), ``individual`` and ``cumulative``
        (`list` [`dict`], each with ``label``, ``features``, ``n_features``, ``nmad`` [µm of
        equivalent hexapod dz], ``gain`` (dimensionless, baseline nMAD over this model's nMAD),
        ``coef_added`` [µm of equivalent hexapod dz per unit of the added feature] and
        ``coef_added_unit``, ``pearson_r`` and ``spearman_rho`` (dimensionless, predicted against
        measured open-loop focus), ``y``, ``pred`` and ``resid``), ``order`` (the candidate columns
        ranked strongest first) and ``night_grouped_check``.

    Notes
    -----
    **Every model here is fitted in sample**, on every night, and scored on the rows it was fitted
    to. The deliverable is a linear fit carrying one slope per telemetry quantity, which has no
    capacity to memorise a night, so the in-sample and held-out residuals are nearly the same
    number. ``night_grouped_check`` quantifies exactly that, with one `thermal_focus_fit.evaluate`
    call on the final cumulative model: it reports the in-sample nMAD, the 5-fold ``day_obs``-
    grouped nMAD and their ratio. It is the only cross-validation in this section and exists so the
    document can state the optimism rather than assert it.

    **All models are fitted and scored on exactly the same rows** -- the subset where the baseline
    and every available candidate are finite. A term whose telemetry resolves on fewer visits would
    otherwise be credited with the easier sample that implies, and the ranking would partly measure
    coverage rather than focus information.

    Ranking is on residual nMAD, not on a coefficient's formal significance. A term can carry a
    slope many standard errors from zero and still leave the scatter where it was, and the scatter
    is what an open-loop correction is judged on.

    **Nothing here selects the model.** The ranking is advisory: `thermal_focus_lib.DELIVERABLE_GROUPS`
    is what the rest of the document fits, and changing it is a deliberate hand edit weighing the
    residual nMAD against the operational cost of carrying another telemetry channel.
    """
    base = [c for c in BASELINE_FEATURES if c in sci.columns]
    available = [(c, lab) for c, lab in CANDIDATE_TERMS if c in sci.columns]
    missing = [c for c, _ in CANDIDATE_TERMS if c not in sci.columns]

    cols = base + [c for c, _ in available]
    d = sci[sci[cols].notna().all(axis=1)].reset_index(drop=True)
    n_all, n_common = len(sci), len(d)

    def _fit(features, label, added=None):
        r = F.fit_full(d, list(features), model=model, verbose=False)
        ln = F.huber_line(d['y'], r['pred'])
        return dict(label=label, features=list(features), n_features=len(features),
                    nmad=r['nmad'], gain=np.nan,
                    coef_added=(float(r['coef'][list(features).index(added)])
                                if added is not None else np.nan),
                    coef_added_unit=(F.FEATURE_UNITS.get(added, '?')
                                     if added is not None else ''),
                    pearson_r=ln['pearson_r'], spearman_rho=ln['spearman_rho'],
                    y=d['y'].to_numpy(float), pred=r['pred'], resid=r['resid'])

    baseline = _fit(base, 'truss temperature alone')
    baseline['gain'] = 1.0

    individual = []
    for c, lab in available:
        e = _fit(base + [c], f'truss + {lab}', added=c)
        e['gain'] = baseline['nmad'] / e['nmad'] if e['nmad'] else np.nan
        e['added'] = c
        individual.append(e)
    individual.sort(key=lambda e: e['nmad'])
    order = [e['added'] for e in individual]

    cumulative, feats = [baseline], list(base)
    labels = dict(available)
    for c in order:
        feats = feats + [c]
        e = _fit(feats, f'+ {labels[c]}', added=c)
        e['gain'] = baseline['nmad'] / e['nmad'] if e['nmad'] else np.nan
        e['added'] = c
        cumulative.append(e)

    check = {}
    if len(cumulative) > 1:
        final = cumulative[-1]['features']
        cvf = F.evaluate(d, final, model=model, verbose=False)
        ins = F.fit_full(d, final, model=model, verbose=False)
        check = dict(features=final, n=len(d), n_nights=int(d['day_obs'].nunique()),
                     nmad_in_sample=ins['nmad'], nmad_night_grouped=cvf['nmad'],
                     optimism=(cvf['nmad'] / ins['nmad'] if ins['nmad'] else np.nan))

    if verbose:
        print(f'candidate terms evaluated on the {n_common} visits of {n_all} where the baseline '
              f'and all {len(available)} candidates are finite')
        if missing:
            print(f'  candidate columns absent from the cached table: {", ".join(missing)}')
        print('  the whole-mirror m1m3_r2_coeff_c is deliberately not a candidate here; the '
              'quadratic')
        print('  radial terms enter as the M1 annulus and M3 inner disc separately')
        print(f'\nindividually, each added to the truss-only baseline '
              f'(nMAD {baseline["nmad"]:.1f} um of equivalent hexapod dz), ranked best first:')
        for e in individual:
            print(f'  {e["label"]:34s} nMAD {e["nmad"]:6.1f} um  gain {e["gain"]:.4f}  '
                  f'coef {e["coef_added"]:+10.2f} um per {e["coef_added_unit"]:22s} '
                  f'r {e["pearson_r"]:+.4f}  rho {e["spearman_rho"]:+.4f}')
        print(f'  gain is dimensionless, baseline nMAD over this model nMAD; r is Pearson and rho '
              f'Spearman, predicted against measured open-loop focus over n {n_common}')
        print('\ncumulatively, added strongest first:')
        for e in cumulative:
            print(f'  {e["label"]:34s} {e["n_features"]:2d} features  nMAD {e["nmad"]:6.1f} um  '
                  f'gain {e["gain"]:.4f}  r {e["pearson_r"]:+.4f}  rho {e["spearman_rho"]:+.4f}')
        if check:
            print(f'\nnight-grouped check on the full {len(check["features"])}-feature model, the '
                  f'only cross-validation in this document:')
            print(f'  in-sample residual nMAD      {check["nmad_in_sample"]:.1f} um of equivalent '
                  f'hexapod dz')
            print(f'  5-fold day_obs-grouped nMAD  {check["nmad_night_grouped"]:.1f} um of '
                  f'equivalent hexapod dz')
            print(f'  optimism {check["optimism"]:.3f} (dimensionless, night-grouped over '
                  f'in-sample) over {check["n_nights"]} nights')
    return dict(n_all=n_all, n_common=n_common, available=[c for c, _ in available],
                missing=missing, baseline=baseline, individual=individual,
                cumulative=cumulative, order=order, night_grouped_check=check)


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
        print(f'    open-loop focus, median peak-to-peak      : '
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
        Print the comparison against the open-loop focus.

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
        print('  DOF content of v-mode 1 at v1 = 1.0 (dimensionless):')
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


def section_t539(t539, full, features, dof_set='all_50', n_modes=34, verbose=True):
    """Compare the predicted degree-of-freedom trim against what the initial alignment settled on.

    Every other measure of this model is a residual against the recovered optical state — the same
    quantity the model was fitted to. This one is independent: the initial alignment block converges
    the commanded Trim at the start of each night without reference to the thermal telemetry, so the
    Trim it arrives at is a separate measurement of the focus the telescope actually needed.

    Parameters
    ----------
    t539 : `pandas.DataFrame`
        The initial-alignment table from ``run_thermal_focus.load_t539``: thermal features suffixed
        ``_first``, Trim degrees of freedom (DOF) suffixed ``_last``.
    full : `dict`
        Result of `thermal_focus_fit.fit_full`, carrying the fitted pipeline under ``model``.
    features : `list` [`str`]
        Feature columns of the deliverable model, in fit order.
    dof_set : `str`, optional
        `ts_ofc` DOF set for the back-projection.
    n_modes : `int`, optional
        v-modes retained.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``table`` — per-night predicted and actual trim per DOF; ``rows`` — the plotted quantities
        as ``(key, label, unit, scale)``; ``fits`` — per quantity, the
        `thermal_focus_fit.huber_line` result of actual against predicted; ``diff`` — per
        quantity, ``median`` and ``nmad`` of actual minus predicted; ``unit`` — the DOF content
        of one unit of v-mode-1 amplitude; ``outliers`` — nights whose settled camera-hexapod dz
        Trim lies more than `OUTLIER_NIGHT_Z` nMAD from the median of the nights, with ``day_obs``,
        the two hexapod dz Trim values [µm], the predicted focus error [µm of equivalent hexapod
        dz] and the robust ``dof5_last_z`` [dimensionless, deviation over the nMAD of the nights];
        ``dof5_last_center_um`` and ``dof5_last_nmad_um`` — that median and nMAD [µm].

    Notes
    -----
    **The two epochs are not simultaneous.** The prediction uses the telemetry at the run's first
    visit, the Trim comes from its last, and the run spans 10 exposures on a typical night and up
    to 45. Perfect agreement is therefore not expected; the question is whether the two track each
    other.

    **The hexapod split varies between nights.** The alignment is free to put focus on either
    hexapod, and does: each of the two is left at exactly zero on a handful of nights. So the
    per-hexapod comparison is degraded by a split that carries no optical meaning,
    while the v-mode-1 projection of the pair — the last row — is insensitive to it. That row is
    the one that answers the physical question; the two hexapod rows are kept so the split stays
    visible rather than hidden inside the combination.

    Pearson and Spearman are both reported, and here they genuinely disagree: the relation is far
    more monotonic than it is linear, because a few nights with large commanded Trim dominate a
    least-squares view of it. That is also why every fit is Huber rather than ordinary least
    squares.
    """
    import aos_state
    se = aos_state.make_state_estimator(dof_set=dof_set, n_modes=n_modes)
    v = np.zeros(se.truncate_index)
    v[0] = 1.0
    unit_vec = np.asarray(se.get_dofs_from_vmodes(v), float)

    wanted = ((5, 'dof5', 'camera hexapod dz', 'um', 1.0),
              (0, 'dof0', 'M2 hexapod dz', 'um', 1.0),
              (12, 'dof12', 'M1M3 bending mode B3', 'nm', 1e3),
              (34, 'dof34', 'M2 bending mode B5', 'nm', 1e3))

    df = t539.copy()
    X = np.column_stack([df[f'{c}_first'].to_numpy(float) for c in features])
    pred_um = np.asarray(full['model'].predict(X), float)

    # The commanded term enters the response positively, so the Trim amplitude that cancels a
    # predicted error is that error in v-mode-1 units, with no sign flip. Same rule as
    # `section_dof`.
    v1_pred = pred_um * L.v1_per_um_dz_value(dof_set=dof_set, n_modes=n_modes, verbose=False)
    df['focus_error_um'] = pred_um
    df['v1_pred'] = v1_pred

    rows, fits, diff = [], {}, {}
    for idx, key, label, unit, scale in wanted:
        df[f'{key}_pred'] = unit_vec[idx] * v1_pred
        rows.append((key, label, unit, scale))

    # The physical quantity: the v-mode-1 amplitude the commanded Trim pair actually carries,
    # as a least-squares projection onto the v-mode-1 direction in the two hexapod dz. This is
    # insensitive to how a given night split focus between the two hexapods.
    u5, u0 = float(unit_vec[5]), float(unit_vec[0])
    df['v1_actual'] = ((df['dof5_last'].to_numpy(float) * u5
                        + df['dof0_last'].to_numpy(float) * u0) / (u5 ** 2 + u0 ** 2))
    rows.append(('v1', 'v-mode-1 amplitude of the pair', 'dimensionless', 1.0))

    for key, label, unit, scale in rows:
        if key == 'v1':
            p, a = df['v1_pred'].to_numpy(float), df['v1_actual'].to_numpy(float)
        else:
            p, a = df[f'{key}_pred'].to_numpy(float), df[f'{key}_last'].to_numpy(float)
        fits[key] = F.huber_line(p * scale, a * scale)
        d = (a - p) * scale
        d = d[np.isfinite(d)]
        diff[key] = {'median': float(np.median(d)), 'nmad': float(nmad(d)),
                     'n': int(len(d))}

    # Outlier nights on the ACTUAL axis -- the Trim the alignment block settled on, which is the y
    # axis of `figure_t539`'s left panels. Flagged on the camera hexapod dz, the DOF an observer
    # reads, with the robust z measured against the median and nMAD of the nights themselves.
    a5 = df['dof5_last'].to_numpy(float)
    c5, s5 = float(np.nanmedian(a5)), float(nmad(a5))
    df['dof5_last_z'] = (a5 - c5) / s5 if s5 > 0 else np.nan
    t539_outliers = (df.loc[df['dof5_last_z'].abs() > OUTLIER_NIGHT_Z,
                            ['day_obs', 'dof5_last', 'dof0_last', 'focus_error_um',
                             'dof5_last_z']]
                     .reindex(df['dof5_last_z'].abs().sort_values(ascending=False).index)
                     .dropna(subset=['dof5_last_z']))

    out = {'table': df, 'rows': rows, 'fits': fits, 'diff': diff,
           'unit': {key: float(unit_vec[idx]) for idx, key, _, _, _ in wanted},
           'outliers': t539_outliers,
           'dof5_last_center_um': c5, 'dof5_last_nmad_um': s5,
           'zero_nights': {'dof5': int((df['dof5_last'] == 0).sum()),
                           'dof0': int((df['dof0_last'] == 0).sum())}}

    if verbose:
        print(f'  {len(df)} nights, day_obs {int(df["day_obs"].min())} to '
              f'{int(df["day_obs"].max())}')
        print(f'  run length [exposures]: median {df["n_run"].median():.0f}, '
              f'min {int(df["n_run"].min())}, max {int(df["n_run"].max())}')
        print(f'  nights leaving the camera hexapod dz Trim at exactly zero: '
              f'{out["zero_nights"]["dof5"]}; the M2 hexapod: {out["zero_nights"]["dof0"]}')
        print('  DOF content of v-mode 1 [um per unit amplitude]: '
              + ', '.join(f'{label} {out["unit"][key]:+.4f}'
                          for _, key, label, _, _ in wanted))
        for key, label, unit, scale in rows:
            ln, d = fits[key], diff[key]
            print(f'    {label:32s} n {ln["n"]:3d}  Pearson r {ln["pearson_r"]:+.4f}  '
                  f'Spearman rho {ln["spearman_rho"]:+.4f}')
            print(f'    {"":32s} slope {ln["slope"]:+.4f} +/- {ln["slope_err"]:.4f} '
                  f'({unit} actual per {unit} predicted)')
            print(f'    {"":32s} actual minus predicted: median {d["median"]:+.4f} {unit}, '
                  f'nMAD {d["nmad"]:.4f} {unit}')
        print(f'  camera hexapod dz Trim settled on: median {c5:+.2f} um, nMAD {s5:.2f} um over '
              f'{len(df)} nights')
        print(f'  nights beyond {OUTLIER_NIGHT_Z:.0f} nMAD on that axis: {len(t539_outliers)}')
        for _, r in t539_outliers.iterrows():
            print(f'    day_obs {int(r.day_obs)}  camera hexapod dz Trim {r.dof5_last:+9.2f} um, '
                  f'M2 hexapod dz Trim {r.dof0_last:+9.2f} um, predicted focus error '
                  f'{r.focus_error_um:+8.1f} um of equivalent hexapod dz, '
                  f'z {r.dof5_last_z:+.1f} (dimensionless)')
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
    """Nightly medians: open-loop focus, truss temperature, and the relation between them.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    sci : `pandas.DataFrame`
        Science table, as loaded.
    samp : `dict`
        Result of `section_sample`, whose ``per_night`` frame holds the per-night medians and
        whose ``outlier_nights`` names the nights labelled here.

    Notes
    -----
    Three panels, in the order a reader needs them: the open-loop focus night by night, the truss
    temperature night by night over the same nights, and the two against each other. The third is
    the whole case for treating this as a between-night problem in one plot: collapsing each night
    to its median leaves a clean temperature relation, which is to say the signal is a between-night
    one. The per-visit scatter around it is what the four M1M3 gradients then address.

    The outlier nights are labelled with their ``day_obs`` on the first and third panels, so a
    reader can see both how far they sit from the relation and where they fall in the season.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    pn = samp['per_night']
    out = samp['outlier_nights']
    idx = {int(d): i for i, d in enumerate(pn['day_obs'])}

    ax = axes[0]
    ax.plot(np.arange(len(pn)), pn['median'], '.', ms=5, color='#1f77b4')
    ax.axhline(0, color='0.6', lw=0.8)
    for _, r in out.iterrows():
        i = idx[int(r.day_obs)]
        ax.plot([i], [r['median']], 'o', ms=7, mfc='none', color='#d62728')
        ax.annotate(f'{int(r.day_obs)}', (i, r['median']), fontsize=6, color='#d62728',
                    xytext=(4, 0), textcoords='offset points', va='center')
    ax.set_xlabel('night index, in day_obs order')
    ax.set_ylabel('night median open-loop focus\n[um of equivalent hexapod dz]')
    ax.set_title(f'Open-loop focus per night, {len(pn)} nights\n'
                 f'{len(out)} nights beyond {OUTLIER_NIGHT_Z:.0f} robust deviations, labelled',
                 fontsize=9.5)

    ax = axes[1]
    ax.plot(np.arange(len(pn)), pn['truss'], '.', ms=5, color='#2ca02c')
    t = pn['truss'].to_numpy(float)
    ax.set_xlabel('night index, in day_obs order')
    ax.set_ylabel('night median TMA truss temperature [deg C]')
    ax.set_title(f'Truss temperature per night, over the same nights\n'
                 f'median {np.median(t):+.2f}, nMAD {nmad(t):.2f}, range {t.min():+.2f} to '
                 f'{t.max():+.2f} deg C', fontsize=9.5)

    ax = axes[2]
    ln = samp['night_line']
    ax.plot(pn['truss'], pn['median'], 'o', ms=4, color='#2ca02c')
    xs = np.linspace(float(pn['truss'].min()), float(pn['truss'].max()), 20)
    ax.plot(xs, ln['intercept'] + ln['slope'] * xs, '-', color='#d62728', lw=1.4,
            label=f'Huber {ln["slope"]:+.1f} +/- {ln["slope_err"]:.1f} um per deg C')
    for _, r in out.iterrows():
        ax.plot([r.truss], [r['median']], 'o', ms=7, mfc='none', color='#d62728')
        ax.annotate(f'{int(r.day_obs)}', (r.truss, r['median']), fontsize=6, color='#d62728',
                    xytext=(4, 0), textcoords='offset points', va='center')
    ax.set_xlabel('night median TMA truss temperature [deg C]')
    ax.set_ylabel('night median open-loop focus\n[um of equivalent hexapod dz]')
    ax.set_title(f'Between nights, where the signal lives\n'
                 f'Pearson r {ln["pearson_r"]:+.4f}, Spearman rho {ln["spearman_rho"]:+.4f}, '
                 f'n {ln["n"]} nights', fontsize=9.5)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def figure_truss_all(pdf, truss_all, sci):
    """Mean TMA truss temperature over every exposure in the database, with the fitted span marked.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    truss_all : `pandas.DataFrame`
        The ``thermal_focus_truss_all.parquet`` cache: one row per exposure in the value-added
        database, with ``truss_temp_mean_c`` [°C], ``day_obs``, ``img_type`` and
        ``obs_start_mjd`` [d].
    sci : `pandas.DataFrame`
        The fitted science sample, overplotted so a reader can see what it covers.

    Notes
    -----
    This page exists to put the fitted sample in context: it shows the whole range of thermal
    conditions the telescope has seen, and which part of it the focus model was fitted on.

    Two populations are on this page and the panels name both. The cache carries **every**
    exposure -- biases, darks, flats and acquisition frames included -- because its purpose is to
    show what the fitted sample excluded; the fitted sample is ``science`` exposures only, within
    one look-up-table epoch and below the temperature cut.

    **No correlation coefficient is quoted here**, deliberately. The temperature against night
    index is a seasonal cycle, periodic rather than monotonic, so a Pearson or Spearman value would
    satisfy the letter of the rule that every fit reports both while describing nothing: a sinusoid
    over a whole number of periods has a correlation near zero against time, which would read as
    "no relation" for the strongest structure on the page. Range, median and nMAD in °C are
    reported instead, which is what the panel is actually showing.
    """
    t = truss_all.dropna(subset=['truss_temp_mean_c'])
    fitted = set(int(d) for d in sci['day_obs'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    pn = (t.groupby('day_obs')['truss_temp_mean_c'].median().reset_index()
          .sort_values('day_obs').reset_index(drop=True))
    n_nights_all = int(truss_all['day_obs'].nunique())
    n_no_truss = n_nights_all - len(pn)
    inside = pn['day_obs'].isin(fitted).to_numpy()
    ax.plot(np.arange(len(pn))[~inside], pn.loc[~inside, 'truss_temp_mean_c'], '.', ms=5,
            color='0.6', label=f'not in the fitted sample ({int((~inside).sum())} nights)')
    ax.plot(np.arange(len(pn))[inside], pn.loc[inside, 'truss_temp_mean_c'], '.', ms=5,
            color='#1f77b4', label=f'fitted nights ({int(inside.sum())})')
    ax.axhline(L.TRUSS_TEMP_MAX_C, color='#d62728', lw=1.2, ls='--',
               label=f'cut at {L.TRUSS_TEMP_MAX_C:.0f} deg C')
    ax.set_xlabel('night index, in day_obs order')
    ax.set_ylabel('night median mean TMA truss temperature [deg C]')
    ax.set_title(f'All {len(pn)} nights with a truss sample, day_obs '
                 f'{int(pn.day_obs.min())} to {int(pn.day_obs.max())}\n'
                 f'{n_no_truss} further nights of {n_nights_all} carry no truss telemetry at all',
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    a = t['truss_temp_mean_c'].to_numpy(float)
    s = sci['truss_temp_mean_c'].to_numpy(float)
    s = s[np.isfinite(s)]
    bins = np.linspace(min(a.min(), s.min()), a.max(), 90)
    ax.hist(a, bins=bins, histtype='step', color='0.4',
            label=f'every exposure: median {np.median(a):+.2f}, nMAD {nmad(a):.2f} deg C, '
                  f'n {len(a)}')
    ax.hist(s, bins=bins, histtype='step', color='#1f77b4',
            label=f'fitted science visits: median {np.median(s):+.2f}, nMAD {nmad(s):.2f} deg C, '
                  f'n {len(s)}')
    ax.axvline(L.TRUSS_TEMP_MAX_C, color='#d62728', lw=1.2, ls='--')
    ax.set_xlabel('mean TMA truss temperature [deg C]')
    ax.set_ylabel('exposures')
    ax.set_title(f'Whole database {a.min():+.2f} to {a.max():+.2f} deg C,\nfitted sample '
                 f'{s.min():+.2f} to {s.max():+.2f} deg C', fontsize=9)
    ax.legend(fontsize=6.5)

    ax = axes[2]
    # day_obs is YYYYMMDD, so the month and day alone give the position in the year without
    # needing a date parse.
    doy = ((t['day_obs'].to_numpy() // 100) % 100 - 1) * 30.4 + (t['day_obs'].to_numpy() % 100)
    ax.plot(doy, a, ',', color='0.5', alpha=0.4)
    ax.axhline(L.TRUSS_TEMP_MAX_C, color='#d62728', lw=1.2, ls='--',
               label=f'cut at {L.TRUSS_TEMP_MAX_C:.0f} deg C')
    ax.set_xlabel('approximate day of year [d]')
    ax.set_ylabel('mean TMA truss temperature [deg C]')
    ax.set_title(f'The seasonal cycle, all {len(a)} exposures with a truss sample\n'
                 f'(southern hemisphere: warmest near day 1 and 365)', fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle('Mean TMA truss temperature over the whole value-added database, against the '
                 'span the focus model was fitted on', fontsize=10.5, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def figure_terms_grid(pdf, terms, key, title):
    """One page per evaluation stage: prediction scatter above, residual histogram below.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    terms : `dict`
        Result of `section_terms`.
    key : `str`
        ``'individual'`` or ``'cumulative'`` -- which list of models to draw.
    title : `str`
        Page heading.

    Notes
    -----
    Each model gets two panels, as item 7 asks: predicted against measured open-loop focus, and a
    one-dimensional residual histogram annotated with its nMAD. They are laid out as a grid rather
    than two panels per page because the individual pass alone is seven models, and fourteen pages
    of two panels could not be compared by eye.

    **Every scatter panel shares one axis range and every histogram one bin array.** A difference
    between panels is then a difference between the fits, not between the framings -- which is the
    only way a grid earns its place over a table of numbers.
    """
    entries = terms[key]
    if not entries:
        return
    ncol = 4
    nhalf = int(np.ceil(len(entries) / ncol))
    fig, axes = plt.subplots(2 * nhalf, ncol, figsize=(14, 3.6 * 2 * nhalf), squeeze=False)

    lo, hi = -800.0, 1200.0
    rlo, rhi = -500.0, 500.0
    rbins = np.linspace(rlo, rhi, 80)
    for k, e in enumerate(entries):
        block, col = k // ncol, k % ncol
        ax = axes[2 * block][col]
        ax.plot(e['y'], e['pred'], ',', color='0.5', alpha=0.5)
        ax.plot([lo, hi], [lo, hi], '-', color='#d62728', lw=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f'{e["label"]}\nnMAD {e["nmad"]:.1f} um, r {e["pearson_r"]:+.3f}, '
                     f'rho {e["spearman_rho"]:+.3f}', fontsize=7.5)
        if col == 0:
            ax.set_ylabel('predicted open-loop focus\n[um of equiv hexapod dz]', fontsize=7.5)
        ax.tick_params(labelsize=6.5)

        ax = axes[2 * block + 1][col]
        r = np.asarray(e['resid'], float)
        r = r[np.isfinite(r)]
        lab = f'nMAD {e["nmad"]:.1f} um'
        if np.isfinite(e['coef_added']):
            lab += f'\ncoef {e["coef_added"]:+.1f} um\nper {e["coef_added_unit"]}'
        ax.hist(np.clip(r, rlo, rhi), bins=rbins, histtype='step', color='#1f77b4', label=lab)
        ax.axvline(0, color='0.5', lw=0.8)
        ax.set_xlim(rlo, rhi)
        ax.set_xlabel('residual [um of equiv hexapod dz]', fontsize=7.5)
        if col == 0:
            ax.set_ylabel('visits', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6, loc='upper left')

    for k in range(len(entries), nhalf * ncol):
        block, col = k // ncol, k % ncol
        axes[2 * block][col].axis('off')
        axes[2 * block + 1][col].axis('off')

    fig.suptitle(f'{title}\nmeasured open-loop focus on the x axis of every scatter panel; '
                 f'all panels share one axis range and one bin array; '
                 f'n {terms["n_common"]} visits', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.055 / nhalf))
    pdf.savefig(fig)
    plt.close(fig)


def figure_terms_individual(pdf, terms):
    """Each candidate term added to the truss-only baseline on its own."""
    figure_terms_grid(pdf, terms, 'individual',
                      'Each candidate telemetry term added to the mean truss temperature '
                      'baseline on its own, ranked best first')


def figure_terms_cumulative(pdf, terms):
    """The candidate terms added cumulatively, strongest first."""
    figure_terms_grid(pdf, terms, 'cumulative',
                      'The same terms added cumulatively to the truss baseline, strongest first')


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
        ax.set_xlabel(f'{lab[c]} residual after the {len(ctrl)} deliverable features\n'
                      '[deg C per unit normalized r^2 amplitude]', fontsize=8)
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
    residual, not the open-loop focus itself, is what the four M1M3 gradients have to improve
    on.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.plot(sci['truss_temp_mean_c'], sci['y'], ',', color='0.5', alpha=0.5)
    xx = np.linspace(sci['truss_temp_mean_c'].min(), sci['truss_temp_mean_c'].max(), 10)
    line = F.huber_line(sci['truss_temp_mean_c'], sci['y'])
    ax.plot(xx, line['intercept'] + line['slope'] * xx, '-', color='#d62728', lw=1.8,
            label=f'Huber {line["slope"]:+.1f} +/- {line["slope_err"]:.1f} um per deg C, '
                  f'n {line["n"]}')
    ax.set_xlabel('TMA truss temperature [deg C]')
    ax.set_ylabel('open-loop focus\n[um of equivalent hexapod dz]')
    ax.set_ylim(-800, 1200)
    ax.set_title(f'Truss temperature alone: Pearson r {line["pearson_r"]:+.3f}, '
                 f'Spearman rho {line["spearman_rho"]:+.3f}')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bins = np.linspace(-800, 1200, 120)
    ax.hist(sci['y'], bins=bins, histtype='step', color='0.4',
            label=f'open-loop focus, nMAD {trussonly["uncorrected_nmad"]:.1f} um')
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
    ax.set_xlabel('open-loop focus [um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('Open-loop focus per band')
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


def figure_model(pdf, sci, full):
    """The deliverable fit: prediction against measurement, before and after, and by band.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    sci : `pandas.DataFrame`
        Science table, as loaded.
    full : `dict`
        Result of `thermal_focus_fit.fit_full` -- the in-sample fit over every night.

    Notes
    -----
    Three panels: the prediction against the measurement, the focus error before and after the
    correction as distributions, and what survives the correction in each band. The last is the
    remaining focus error an observer in a given filter would see.

    Every quantity on this page is in sample, fitted on every night. A coefficient-stability panel
    showing each cross-validation fold's coefficients used to sit here; it was removed with the
    rest of the fold presentation, because a linear fit carrying one slope per telemetry quantity
    has no capacity to memorise a night and so nothing for the folds to detect.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    # Measured on x, predicted on y: the measurement is the independent variable and the
    # prediction the dependent one.
    ax.plot(sci['y'], full['pred'], ',', color='0.5', alpha=0.5)
    lo, hi = -800, 1200
    ax.plot([lo, hi], [lo, hi], '-', color='#d62728', lw=1.2, label='perfect prediction')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    pm = F.huber_line(sci['y'], full['pred'])
    ax.set_xlabel('measured open-loop focus [um of equivalent hexapod dz]')
    ax.set_ylabel('predicted open-loop focus\n[um of equivalent hexapod dz]')
    ax.set_title(f'Prediction against measurement, all {len(sci)} visits\n'
                 f'residual nMAD {full["nmad"]:.1f} um, Pearson r {pm["pearson_r"]:+.4f}, '
                 f'Spearman rho {pm["spearman_rho"]:+.4f}', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper left')

    ax = axes[1]
    ax.hist(sci['y'], bins=np.linspace(-800, 1200, 120), histtype='step', color='0.4',
            label=f'open-loop focus, nMAD {nmad(sci["y"].to_numpy()):.1f} um')
    ax.hist(full['resid'], bins=np.linspace(-800, 1200, 120), histtype='step', color='#d62728',
            label=f'focus error after correction, nMAD {full["nmad"]:.1f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_xlabel('[um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('Focus error before and after the correction', fontsize=9.5)
    ax.legend(fontsize=7)

    # Physical units, not residual-over-nMAD: the reader wants to know how many um of focus
    # error survive the correction in each band, and dividing by each band's own scale hides
    # exactly that. The axis covers +/-300 um, which holds the bulk for every band; the per-band
    # nMAD in the legend is computed on every finite visit, not on the plotted range.
    ax = axes[2]
    r = np.asarray(full['resid'], float)
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
    ax.set_title('Remaining focus error, by band', fontsize=9.5)
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

    All four axes are binned over the 0.25th to 99.75th percentile rather than the full range, with
    everything beyond piled into the end bins, so no visit is dropped from the count. The annotated
    median and nMAD are computed on every finite value, not on the clipped range. The clip is
    deliberately wide: at the 1st to 99th percentile the end bins absorbed enough of the
    distribution to hide how far the tails of the commanded motion actually reach, which is the
    part of the distribution an observer most needs to see.
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
        lo, hi = np.percentile(a, [0.25, 99.75])
        bins = np.linspace(lo, hi, 100)
        ax.hist(np.clip(a, lo, hi), bins=bins, histtype='step', color='#1f77b4')
        ax.axvline(float(np.median(a)), color='#d62728', lw=1.2,
                   label=f'median {np.median(a):+.4g} {unit}\nnMAD {nmad(a):.4g} {unit}\n'
                         f'axis clipped to 0.25th-99.75th percentile')
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


def figure_t539(pdf, t539res):
    """Predicted against actual Trim from the initial alignment block, per degree of freedom.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open PDF to append to.
    t539res : `dict`
        Result of `section_t539`.

    Notes
    -----
    One row per quantity: on the left the Trim the alignment block settled on against the Trim the
    thermal telemetry at the run's start predicts, with the Huber line and a dashed line of exact
    agreement; on the right the histogram of actual minus predicted. Every row is drawn the same
    way, so the correlation each carries is read off the same axes rather than from framing.

    The two hexapods are in µm, the two mirror bending modes in nm, and the combined row is the
    dimensionless v-mode-1 amplitude — a shared axis would put the bending modes on zero.
    """
    if not t539res:
        return
    df, rows = t539res['table'], t539res['rows']
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.4 * len(rows)))
    for row, (key, label, unit, scale) in enumerate(rows):
        if key == 'v1':
            p = df['v1_pred'].to_numpy(float) * scale
            a = df['v1_actual'].to_numpy(float) * scale
        else:
            p = df[f'{key}_pred'].to_numpy(float) * scale
            a = df[f'{key}_last'].to_numpy(float) * scale
        line, d = t539res['fits'][key], t539res['diff'][key]

        ax = axes[row, 0]
        ax.plot(p, a, 'o', ms=4, color='#1f77b4')
        # Each axis is scaled to its own variable rather than to a shared range. Forcing a square
        # range would compress the two bending-mode rows into a vertical stripe at x = 0, because
        # the predicted amplitude there is a few nm against an actual Trim of hundreds -- which is
        # the point those rows make, but it would also hide the predicted spread and the fitted
        # line entirely. The line of exact agreement is still drawn, and runs off the panel where
        # the two ranges genuinely differ by that much.
        lo = float(np.nanmin(np.concatenate([p, a])))
        hi = float(np.nanmax(np.concatenate([p, a])))
        ends = np.array([lo, hi])
        ax.plot(ends, ends, '--', color='0.5', lw=1.0, label='exact agreement')
        xs = np.array([np.nanmin(p), np.nanmax(p)])
        ax.plot(xs, line['intercept'] + line['slope'] * xs, '-', color='#2ca02c', lw=1.2,
                label=f'Huber slope {line["slope"]:+.4g} +/- {line["slope_err"]:.4g}\n'
                      f'({unit} actual per {unit} predicted)\n'
                      f'Pearson r {line["pearson_r"]:+.4f}, '
                      f'Spearman rho {line["spearman_rho"]:+.4f}')
        ax.axhline(0, color='0.85', lw=0.8, zorder=0)
        ax.axvline(0, color='0.85', lw=0.8, zorder=0)
        # Set the limits last: the line of exact agreement spans both variables' combined range and
        # would otherwise autoscale the panel back to a square.
        xpad = 0.06 * (xs[1] - xs[0]) if xs[1] > xs[0] else 1.0
        ax.set_xlim(xs[0] - xpad, xs[1] + xpad)
        ylo, yhi = float(np.nanmin(a)), float(np.nanmax(a))
        ypad = 0.06 * (yhi - ylo) if yhi > ylo else 1.0
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.set_xlabel(f'predicted {label} from the thermal telemetry\nat the run\'s first '
                      f'visit [{unit}]')
        ax.set_ylabel(f'actual {label} Trim\nat the run\'s last visit [{unit}]')
        ax.set_title(f'{label}: actual against predicted, {line["n"]} nights', fontsize=9.5)
        ax.legend(fontsize=7.5)

        ax = axes[row, 1]
        diff = a - p
        ax.hist(diff[np.isfinite(diff)], bins=30, histtype='step', color='#1f77b4')
        ax.axvline(0, color='0.6', lw=0.8)
        ax.axvline(d['median'], color='#d62728', lw=1.2,
                   label=f'median {d["median"]:+.4g} {unit}\nnMAD {d["nmad"]:.4g} {unit}')
        ax.set_xlabel(f'actual minus predicted {label} [{unit}]')
        ax.set_ylabel('nights')
        ax.set_title(f'{label}: actual minus predicted', fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle('The initial alignment block: predicted Trim against what it settled on',
                 fontsize=11, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.985))
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

    print('=== 1. the open-loop focus and the sample ===')
    sci, fam, t539, truss_all, features, funnel = load(out_dir, args.fam_dir_name, day_obs_range)
    samp = section_sample(sci, features)

    print('\n=== 2. settling the model: the candidate telemetry terms in sequence ===')
    terms = section_terms(sci, model=args.model)

    print('\n=== 3. the deliverable thermal model, fitted in sample on every night ===')
    full = F.fit_full(sci, features, model=args.model)
    print()
    models = F.model_comparison(sci, features)
    print()
    cmd = F.commanded_truss_slope(sci, L.v1_per_um_dz_value(verbose=False))
    print()
    bands = F.per_band_fit(sci, features, model=args.model)

    print('\n=== 4. camera-body temperature ===')
    cam = section_camtemp(sci)

    print('\n=== 5. what adds nothing ===')
    section_ablation(sci)   # printed, not shown: the wind, elevation and hexapod-history nulls

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
    tails = F.residual_tail(sci, full['resid'])

    print('\n=== 7. within-night behaviour against elevation ===')
    nights = section_elevation(sci, full['resid'])

    print('\n=== 8. band changes ===')
    # The claim being tested is that fitting each band separately injects a step at every filter
    # change, because the coefficients swap while nothing physical happens. That needs the
    # per-band-corrected residual in the table, not just the shared one.
    d = sci.copy()
    d['resid'] = full['resid']
    d['resid_per_band'] = np.nan
    for band, g in sci.groupby('band'):
        if len(g) < 200:
            continue
        r = F.fit_full(g.reset_index(drop=True), features, model=args.model, verbose=False)
        d.loc[g.index, 'resid_per_band'] = r['resid']
    print('median |step| between consecutive visits within a night '
          '[um of equivalent hexapod dz]')
    steps = {'open-loop focus': F.band_change_step(d, 'y', label='open-loop focus'),
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

    print('\n=== 13. the focus correction as degrees of freedom ===')
    dofres = section_dof(sci, full['pred'], L.v1_per_um_dz_value(verbose=False))

    print('\n=== 14. against the Trim the initial alignment block settled on ===')
    if t539 is None or not len(t539):
        print('  thermal_focus_t539.parquet is absent; build it with\n'
              '    python code/run_thermal_focus.py')
        t539res = {}
    else:
        t539res = section_t539(t539, full, features)

    if args.no_pdf:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / args.pdf_name
    with PdfPages(pdf_path) as pdf:
        # --------------------------------- page 1: what the study is, and what it is fitted on
        _conv = L.v1_per_um_dz_value(verbose=False)
        _conv_row = next((r for r in conv
                          if r['dof_set'] == 'all_50' and r['n_modes'] == 34), conv[0])
        _text_page(pdf, 'The thermal-focus study: what is predicted, and from what', [
            'WHAT IS PREDICTED',
            '',
            '  The telescope\'s START-OF-NIGHT OPEN-LOOP FOCUS: the defocus the telescope needs',
            '  before any wavefront measurement has been folded in. Predicting it from thermal',
            '  telemetry alone lets focus be set open loop, at the start of a night, from',
            '  temperatures the observatory already records.',
            '',
            '  The quantity is the degree-of-freedom content of V-MODE 1 -- the first singular',
            '  vector of the Active Optics System (AOS) sensitivity matrix, which is essentially',
            '  uniform defocus. The study works in v-mode space throughout: the per-corner',
            '  Zernike deviations are passed through the Optical Feedback Control (OFC)',
            '  sensitivity-matrix singular value decomposition, so the degrees of freedom and',
            '  v-mode amplitudes are recovered properly. Four field points cannot separate a',
            '  field-constant defocus from a field tilt, so a four-corner mean Z4 is NOT what is',
            '  fitted here.',
            '',
            f'  open-loop focus = (v1_trim + {L.MEASURED_SIGN:+.1f} * v1) / {_conv:.5e}',
            '                    [um of equivalent hexapod dz]',
            '',
            '  v1       v-mode 1 of the state the corner wavefront sensors MEASURED',
            '  v1_trim  v-mode 1 of the correction the AOS had already COMMANDED',
            '',
            f'  Over the sample: median {sci["y"].median():+.1f}, '
            f'nMAD {nmad(sci["y"].to_numpy()):.1f} um of equivalent hexapod dz.',
            '',
            '  The PREDICTORS are the Telescope Mount Assembly (TMA) truss temperatures and the',
            '  M1M3 thermal gradients -- telemetry only. No wavefront measurement enters the',
            '  prediction, which is precisely what makes it open loop.',
            '',
            'THE CONVERSION TO MICRONS, AND ITS LIMIT',
            '',
            f'  One unit of v-mode-1 amplitude is {_conv_row["um_per_v1_shared"]:.1f} um of total',
            '  defocus travel, split 0.5 um on the camera hexapod and 0.5 um on M2. That is the',
            '  convention this document reports, and it is what "equivalent hexapod dz" means.',
            '',
            '  THE CONVERSION IS NOT EXACT AT THE PERCENT LEVEL, and is quoted to give a physical',
            '  sense of the size of a focus change rather than as a calibration:',
            '',
            '    - three independent routes to it agree only to 1.2% (dimensionless, spread over',
            '      mean): the camera-only inverse, the minimum-norm singular-value solution and',
            f'      the full 50-degree-of-freedom pseudo-inverse ({L.DZ_UM_PER_UM_WF:+.4f} um of',
            '      equivalent hexapod dz per um of wavefront).',
            f'    - the shared and camera-alone conventions differ by '
            f'{100 * abs(_conv_row["um_per_v1_shared"] - _conv_row["um_per_v1_camera"]) / _conv_row["um_per_v1_shared"]:.1f}% '
            f'(dimensionless):',
            f'      {_conv_row["um_per_v1_shared"]:.1f} um sharing the motion between the two '
            f'hexapods against',
            f'      {_conv_row["um_per_v1_camera"]:.1f} um moving the camera hexapod alone.',
            '',
            'THE SAMPLE',
            '',
            f'  {len(sci)} science visits over {sci["day_obs"].nunique()} nights, day_obs '
            f'{int(sci.day_obs.min())} to {int(sci.day_obs.max())}.',
            '',
            '  The Zernikes are those derived in the Consolidated Database (ConsDB), because they',
            '  are the CONSISTENT AND COMPREHENSIVE data set: the four corner wavefront sensors',
            '  are read out on every science visit, so the sample is the whole survey rather than',
            '  the few hundred dedicated Full Array Mode (FAM) visits. The day_obs span is the',
            '  range over which ONE CONSISTENT hexapod look-up table (LUT) was in force -- the',
            '  commanded baseline has to mean the same thing on every night in the fit.',
            '',
            '  WHAT WAS EXCLUDED:',
            '',
            f'    {funnel["n_lut_nights"]} nights of an offset LUT epoch, on which the commanded '
            f'baseline does not mean',
            '      the same thing as on neighbouring nights (cut in the build stage).',
            f'    THE HOTTER DATA: every visit above {L.TRUSS_TEMP_MAX_C:.0f} deg C of mean truss '
            f'temperature -- 217 science visits on',
            '      day_obs 20251118 and 20251119, between +22.88 and +25.07 deg C. This is an',
            '      isolated warm population, detached from the rest of the sample by an empty',
            '      interval of 5.1792 deg C, so a cut inside that gap removes it with no boundary',
            '      sensitivity. The cut is applied in the build stage and re-applied here, where it',
            f'      now removes {funnel["n_hot_visits"]} further visits on '
            f'{funnel["n_hot_nights"]} nights.',
            f'    {funnel["n_no_features"]} visits missing at least one of the '
            f'{len(features)} deliverable features.',
            '    Non-science image types and out-of-band visits, cut in the build stage: the fit is',
            '      on ordinary survey exposures in the six filters.',
            '',
            'Feature means over the sample:',
            *[f'  {c:32s} {sci[c].mean():+12.5f} [{F.FEATURE_UNITS.get(c, "?")}]'
              for c in features],
        ])
        figure_before(pdf, sci, trussonly)

        if truss_all is not None and len(truss_all):
            figure_truss_all(pdf, truss_all, sci)
        else:
            print('  thermal_focus_truss_all.parquet is absent; the database-wide truss page is '
                  'skipped. Build it with\n    python code/run_thermal_focus.py --only-truss-all')
        figure_sample(pdf, sci, samp)

        # ------------------------------- page 5: how the model is settled, and why no folds
        _chk = terms['night_grouped_check']
        _text_page(pdf, 'How the model is settled, and why there are no folds', [
            'The feature set this document delivers is settled by adding each candidate telemetry',
            'term to a mean truss temperature baseline and reading the residual scatter it leaves.',
            'The two pages that follow show that sequence. No train/test split and no',
            'cross-validation fold enters it, for the reasons below.',
            '',
            'WHY A SPLIT BY VISIT WOULD BE WRONG',
            '',
            '  Within one night the thermal telemetry barely moves: only 2.7% of the mean truss',
            '  temperature variance is within-night (dimensionless, within-night over total),',
            '  while 90.7% of the open-loop focus variance is between nights. Consecutive visits',
            '  are therefore near-duplicates in temperature while carrying that night\'s own focus',
            '  offset. A model given some visits from a night and asked about others from the SAME',
            '  night could recall the offset rather than derive focus from temperature.',
            '',
            '  So any split of this sample must be BY NIGHT, never by visit. A visit-level split',
            '  would report a number far better than the telescope would deliver on a new night,',
            '  which is the only case that matters in operation.',
            '',
            'WHY NO SPLIT IS NEEDED HERE',
            '',
            '  The deliverable is a LOW-DIMENSIONAL LINEAR HUBER FIT: one slope per telemetry',
            f'  quantity plus an intercept, {len(features)} slopes over '
            f'{len(sci)} visits. A model of that form has no',
            '  capacity to memorise a night -- there is nowhere for a night\'s identity to be',
            '  stored. In-sample and held-out residuals are therefore nearly the same number, and',
            '  a fold-based approach buys nothing that is not already visible.',
            '',
            '  EVERY RESIDUAL nMAD IN THIS DOCUMENT IS IN SAMPLE, fitted on every night. That is',
            '  also the model an observer would actually be handed: fitted on all the data there',
            '  is.',
            '',
            'THE ONE CHECK THAT THIS IS TRUE',
            '',
            *([f'  On the full {len(_chk["features"])}-feature model, over {_chk["n"]} visits and '
               f'{_chk["n_nights"]} nights:',
               '',
               f'    in-sample residual nMAD                 '
               f'{_chk["nmad_in_sample"]:.1f} um of equivalent hexapod dz',
               f'    5-fold day_obs-grouped residual nMAD    '
               f'{_chk["nmad_night_grouped"]:.1f} um of equivalent hexapod dz',
               f'    optimism {_chk["optimism"]:.3f} (dimensionless, night-grouped nMAD over '
               f'in-sample nMAD)',
               '',
               '  That ratio is the entire cost of fitting and scoring on the same nights, and it',
               '  is the only cross-validated number in this document.',
               '',
               '  It is a statement about MODEL CAPACITY, not about the sample. The same',
               '  measurement applied to gradient-boosted trees on these same visits gives',
               f'  {F.LEAK_FACTOR:.1f} (dimensionless) -- a model with enough capacity to memorise '
               f'a night does',
               '  memorise it, and would need the folds this linear fit does not.']
              if _chk else
              ['  The night-grouped check could not be computed: no candidate terms were '
               'available.']),
        ])
        figure_terms_individual(pdf, terms)
        figure_terms_cumulative(pdf, terms)

        # ------------------------------------ page 8: the term summary, naming no winner
        _ind, _cum = terms['individual'], terms['cumulative']
        _text_page(pdf, 'Which telemetry terms earn their place', [
            f'All models below are fitted and scored on the SAME {terms["n_common"]} visits of '
            f'{terms["n_all"]} -- the subset where the',
            'baseline and every candidate are finite. A term whose telemetry resolves on fewer',
            'visits would otherwise be credited with the easier sample that implies, and the',
            'ranking would partly measure coverage rather than focus information.',
            '',
            f'Baseline, mean TMA truss temperature alone: nMAD '
            f'{terms["baseline"]["nmad"]:.1f} um of equivalent hexapod dz',
            '',
            'EACH CANDIDATE ADDED ON ITS OWN, ranked by the scatter it leaves:',
            '',
            '  term                                 nMAD    gain    coefficient of the added term',
            *[f'  {e["label"]:34s} {e["nmad"]:6.1f}  {e["gain"]:.4f}  '
              f'{e["coef_added"]:+9.2f} um per {e["coef_added_unit"]}'
              for e in _ind],
            '',
            '  nMAD is in um of equivalent hexapod dz; gain is dimensionless, baseline nMAD over',
            '  this model nMAD; the coefficient is um of equivalent hexapod dz per unit of the',
            f'  added term, over n {terms["n_common"]} visits.',
            '',
            '  Ranking is on residual nMAD, not on a coefficient\'s formal significance: a term',
            '  can carry a slope many standard errors from zero and still leave the scatter where',
            '  it was, and the scatter is what an open-loop correction is judged on.',
            '',
            'THE SAME TERMS ADDED CUMULATIVELY, strongest first:',
            '',
            '  model                              features   nMAD    gain   Pearson r  Spearman',
            *[f'  {e["label"]:34s} {e["n_features"]:2d}     {e["nmad"]:6.1f}  '
              f'{e["gain"]:.4f}  {e["pearson_r"]:+.4f}   {e["spearman_rho"]:+.4f}'
              for e in _cum],
            '',
            '  r and rho are Pearson and Spearman of the predicted against the measured open-loop',
            f'  focus, both dimensionless, over n {terms["n_common"]} visits.',
            '',
            'CAMERA-BODY TEMPERATURE AS AN ALTERNATIVE THERMOMETER:',
            '',
            *[f'  {c:22s} resolves {v:.2f}% of visits' for c, v in cam['coverage'].items()],
            *[line
              for k, n in (('truss_line', 'TMA truss temperature'),
                           ('cam_line', 'camera-body AverageTemp')) if cam.get(k)
              for line in (f'  {n:26s} slope {cam[k]["slope"]:+8.2f} +/- '
                           f'{cam[k]["slope_err"]:.2f} um of equivalent hexapod dz per deg C',
                           f'  {"":26s} Pearson r {cam[k]["pearson_r"]:+.4f}, Spearman rho '
                           f'{cam[k]["spearman_rho"]:+.4f}, n {cam[k]["n"]}')],
            '',
            'WHAT THIS PAGE DOES NOT DO',
            '',
            '  It does not pick a winner, and no threshold is applied to the gains above. The',
            '  choice is a judgement that weighs the residual nMAD a term buys against the',
            '  operational cost of carrying another telemetry channel: the quadratic radial terms',
            '  require value_added/code/build_m1m3_thermal_r2.py to have been run over the night,',
            '  and the camera-body temperature can drop out of the telemetry stream entirely.',
            '',
            f'  The model fitted on every page that follows is the deliverable set, '
            f'{"+".join(L.DELIVERABLE_GROUPS)} --',
            f'  {", ".join(features[:3])},',
            f'  {", ".join(features[3:])}.',
            '  Changing it is a deliberate hand edit of thermal_focus_lib.DELIVERABLE_GROUPS,',
            '  informed by the two pages above.',
        ])

        _text_page(pdf, 'The deliverable thermal model', [
            'The fitted equation, focus error in um of equivalent hexapod dz:',
            '',
            f'  = {full["intercept"]:+.2f}',
            *[f'    {c:+10.2f} * {f:32s} [per {F.FEATURE_UNITS.get(f, "?")}]'
              for f, c in zip(full['features'], full['coef'])],
            '',
            f'  reproduces the fitted pipeline to '
            f'{full["equation_max_abs_diff"]:.2e} um of equivalent hexapod dz',
            '',
            f'Residual nMAD {full["nmad"]:.1f} um of equivalent hexapod dz, in sample over '
            f'{len(sci)} visits',
            f'and {sci["day_obs"].nunique()} nights, against the open-loop focus nMAD of '
            f'{nmad(sci["y"].to_numpy()):.1f} um --',
            f'an improvement of {nmad(sci["y"].to_numpy()) / full["nmad"]:.2f}x (dimensionless, '
            f'open-loop focus nMAD over residual nMAD).',
            *([f'The night-grouped value is {_chk["nmad_night_grouped"]:.1f} um, optimism '
               f'{_chk["optimism"]:.3f} (dimensionless); see "How the',
               'model is settled" above.'] if _chk else []),
            '',
            'Model comparison, night-grouped [um of equivalent hexapod dz]:',
            *[f'  {r.model:12s} nMAD {r.resid_nmad:7.1f}   R2 {r.r2:+.3f}   '
              f'improvement {r.improvement:.2f}x (dimensionless)'
              for _, r in models.iterrows()],
            '  These are night-grouped, unlike every other nMAD in this document, and',
            '  deliberately so: comparing a gradient-boosted tree with a linear fit is exactly',
            '  the case where the grouping changes the answer, because the tree has the capacity',
            '  to memorise a night and the linear fit does not.',
            '',
            'Remaining focus error per band [um of equivalent hexapod dz]:',
            *[f'  {r.band}  n {int(r.n):6d}  open-loop focus {r.uncorrected_nmad:6.1f}  '
              f'shared model {r.shared_nmad:6.1f}  own model {r.own_nmad:6.1f}  '
              f'own truss {r.own_truss:+8.2f} um per deg C' for _, r in bands.iterrows()],
            '  "shared model" is this one band-independent fit; "own model" is a separate fit per',
            '  band. The two agree closely enough that the band-independent model is what is',
            '  delivered, and a per-band model would inject a step at every filter change:',
            '',
            'Band-change step, median |step| between consecutive visits within a night',
            '[um of equivalent hexapod dz]:',
            *[f'  {k:22s} band change {v["median_change"]:7.1f} (n {v["n_change"]})  '
              f'same band {v["median_same"]:7.1f} (n {v["n_same"]})  '
              f'ratio {v["ratio"]:.2f} (dimensionless)' for k, v in steps.items()],
            '',
            'Residual tails beyond +/-3 nMAD, per band '
            '(a Gaussian gives 0.135% on each side):',
            '  about a truss-only per-band Huber fit --',
            *[f'    {r.band}  above {r.frac_above:5.2f}%  below {r.frac_below:5.2f}%  '
              f'ratio {r.ratio:5.1f} (dimensionless, above over below)'
              for _, r in tails_truss.iterrows()],
            f'  about the full {len(features)}-feature band-independent fit --',
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
            f'Commanded truss slope against the FAM Double Zernike value '
            f'{F.FAM_TRUSS_SLOPE:+.5f}',
            '[dimensionless v-mode-1 amplitude of the Trim per deg C]:',
            *[f'  {r.band:>4s}  n {int(r.n):6d}  {r.slope:+.5f} +/- {r.slope_err:.5f}  '
              f'difference {r.difference:+.5f} ({r.n_sigma:.1f} standard errors)'
              for _, r in cmd.iterrows()],
            '  This, not the fitted open-loop focus coefficient, is the like-for-like test: the',
            '  FAM value is a slope of the COMMANDED Trim, while the open-loop focus is Trim',
            '  minus the measured state and so a different quantity.',
        ])
        figure_model(pdf, sci, full)

        r2_lines =['Quadratic radial section skipped: the cached table carries none of the '
                    'm1m3_r2_coeff_c, m1_r2_coeff_c or m3_r2_coeff_c columns. Run',
                    'value_added/code/build_m1m3_thermal_r2.py, then rebuild the cached table.']
        if r2res['available']:
            # Short population names, not the full R2_COLS labels: the full ones pad to 36
            # characters and would push these rows past the right edge of the page.
            lab = {'m1m3_r2_coeff_c': 'whole mirror', 'm1_r2_coeff_c': 'M1 annulus',
                   'm3_r2_coeff_c': 'M3 inner disc'}
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
                'Raw relation to the focus error [slope in um of equivalent hexapod dz per unit',
                'normalized radius-squared amplitude; correlations dimensionless]:',
                *[f'  {lab[c]:14s} {r2res["coverage"][c]:6.2f}% of visits  slope '
                  f'{r2res["lines"][c]["slope"]:+8.1f} +/- '
                  f'{r2res["lines"][c]["slope_err"]:6.1f}  '
                  f'Pearson r {r2res["lines"][c]["pearson_r"]:+.4f}  '
                  f'Spearman rho {r2res["lines"][c]["spearman_rho"]:+.4f}'
                  for c in r2res['available']],
                '',
                'How much each duplicates the existing M1M3 radial gradient (dimensionless):',
                *[f'  {lab[c]:14s} Pearson r {r2res["redundancy"][c]["pearson_r"]:+.4f}  '
                  f'Spearman rho {r2res["redundancy"][c]["spearman_rho"]:+.4f}  '
                  f'n {r2res["redundancy"][c]["n"]}'
                  for c in r2res['available'] if r2res['redundancy'].get(c)],
                '',
                'Partial correlation with the focus error, both sides stripped of the truss',
                'temperature and the four bulk gradients -- the "above and beyond" test',
                '[slope in um of equivalent hexapod dz per unit normalized amplitude]:',
                *[f'  {lab[c]:14s} raw r {r2res["partial"][c]["raw_pearson_r"]:+.4f} -> '
                  f'partial r {r2res["partial"][c]["partial_pearson_r"]:+.4f}  '
                  f'partial rho {r2res["partial"][c]["partial_spearman_rho"]:+.4f}  '
                  f'slope {r2res["partial"][c]["slope"]:+8.1f} +/- '
                  f'{r2res["partial"][c]["slope_err"]:6.1f}'
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
        _text_page(pdf, 'The quadratic radial M1M3 terms', r2_lines)
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
                f'  open-loop focus, median peak-to-peak      : '
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
        _text_page(pdf, 'Elevation, hysteresis and the dz conversion', [
            *elev_lines,
            '',
            '  The rising-versus-falling comparison is kept as a NULL RESULT. A real elevation',
            '  hysteresis -- the truss settling differently going up than coming down -- would',
            '  show as a systematic rising-minus-falling slope difference. The sign test gives',
            '  p = 0.084 (dimensionless), rising steeper on the majority of nights but not at a',
            '  level worth correcting for. The plots are kept so the absence is on the record.',
            '',
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
            *(['Standalone calculator (trim_calculator.py, numpy only) against this fit, over',
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

        _text_page(pdf, 'FAM blocks: a between-night model is not a within-block model',
                   fam_lines)
        figure_fam(pdf, famres)

        # The hexapod pair is tens of um; the two bending modes are thousandths of a um. Printing
        # both in um makes the bending rows read as +0.0013 +/- 0.0000, so they are shown in nm.
        _nm_cols = set(dofres['names'][2:])
        _sc = lambda c: 1e3 if c in _nm_cols else 1.0
        _un = lambda c: 'nm' if c in _nm_cols else 'um'
        _text_page(pdf, 'The correction as degrees of freedom', [
            'The correction above is one number per visit, a focus error in um of equivalent',
            'hexapod dz. An observer acts on degrees of freedom (DOF), so this part converts the',
            'PREDICTED correction into the Trim DOF that would apply it: how far each DOF would',
            'have to move to put the thermal correction on the telescope.',
            '',
            '  v1_applied = predicted focus error * v1_per_um_dz, then back-projected to DOF.',
            '  The commanded Trim term enters the response with a positive sign, so the amplitude',
            '  needed in the Trim to cancel a predicted error is that error in v-mode-1 units,',
            '  with no sign flip. These are motions to command, not residuals left over after',
            '  commanding them -- how well the correction works is what the deliverable model page',
            '  measures.',
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

        # ------------------------------------- group 4: against what the observatory actually did
        if t539res:
            _t = t539res['table']
            _u = t539res['unit']
            # `section_t539` already reports each difference in the row's own unit, so no further
            # scaling is applied here -- the nm rows would otherwise read a factor of 1000 high.
            _fmt = {'dof5': ('camera hexapod dz', 'um'),
                    'dof0': ('M2 hexapod dz', 'um'),
                    'dof12': ('M1M3 bending B3', 'nm'),
                    'dof34': ('M2 bending B5', 'nm'),
                    'v1': ('v-mode-1 of the pair', 'dimensionless')}
            _text_page(pdf, 'Against the Trim the initial alignment block settled on', [
                f'Sample      {len(_t)} nights of the initial alignment block BLOCK-T539, '
                f'day_obs {int(_t.day_obs.min())} to {int(_t.day_obs.max())}',
                '',
                'WHY THIS COMPARISON IS DIFFERENT FROM EVERY OTHER NUMBER IN THIS DOCUMENT:',
                '',
                '  Every residual so far is against the optical state recovered from the corner',
                '  wavefront sensors, which is the quantity the model was fitted to. The initial',
                '  alignment block converges the commanded Trim at the start of each night without',
                '  reference to the thermal telemetry, so the Trim it arrives at is an independent',
                '  measurement of the focus the telescope actually needed.',
                '',
                'HOW THE TWO EPOCHS ARE PICKED:',
                '',
                '  Per night, the exposures of img_type science or acq are sorted by seq_num and',
                '  the first 10 are taken, those with a BLOCK-T539 program are selected, and the',
                '  run is extended through the contiguous seq_num from the lowest of them. The',
                '  prediction uses the thermal telemetry at the run\'s FIRST visit; the Trim comes',
                '  from its LAST. So the two are separated by the whole run, not simultaneous.',
                '',
                f'  Run length [exposures]: median {_t["n_run"].median():.0f}, '
                f'min {int(_t["n_run"].min())}, max {int(_t["n_run"].max())}. The block is not a',
                '  fixed 10 exposures, so the separation between the two epochs varies by night.',
                '',
                'ACTUAL TRIM AT THE RUN\'S END AGAINST THE PREDICTED TRIM:',
                '',
                *[f'  {_fmt[k][0]:21s} [{_fmt[k][1]:13s}] '
                  f'r {t539res["fits"][k]["pearson_r"]:+.4f}  '
                  f'rho {t539res["fits"][k]["spearman_rho"]:+.4f}  slope '
                  f'{t539res["fits"][k]["slope"]:+7.3f}  diff median '
                  f'{t539res["diff"][k]["median"]:+8.3f}  nMAD '
                  f'{t539res["diff"][k]["nmad"]:7.3f}'
                  for k, _, _, _ in t539res['rows']],
                '  (the bracketed unit applies to the two diff columns; r is Pearson and rho is',
                f'   Spearman over {len(_t)} nights; the slope is dimensionless, actual per',
                '   predicted in that same unit; diff is actual minus predicted)',
                '',
                *[f'  {_fmt[k][0]:21s} Huber slope {t539res["fits"][k]["slope"]:+7.3f} +/- '
                  f'{t539res["fits"][k]["slope_err"]:6.3f} (dimensionless), '
                  f'{abs(t539res["fits"][k]["slope"]) / t539res["fits"][k]["slope_err"]:4.1f} '
                  f'std err'
                  for k, _, _, _ in t539res['rows']],
                '',
                '  Pearson and Spearman disagree, and the Spearman value is the one to read: the',
                '  relation is far more monotonic than it is linear, because a few nights with',
                '  large commanded Trim dominate a least-squares view of it. Every fit above is',
                '  Huber for the same reason.',
                '',
                'WHY THE PER-HEXAPOD ROWS ARE THE WEAKER ONES:',
                '',
                '  The alignment is free to put focus on either hexapod, and does: it leaves the',
                f'  camera hexapod dz Trim at exactly zero on '
                f'{t539res["zero_nights"]["dof5"]} of {len(_t)} nights and the M2 hexapod on '
                f'{t539res["zero_nights"]["dof0"]}.',
                '  That split carries no optical meaning, and it degrades the two hexapod rows',
                '  while leaving the combined v-mode-1 projection of the pair unaffected. The',
                '  combined row is what answers the physical question; the two hexapod rows are',
                '  kept so the split stays visible rather than hidden inside the combination.',
                '',
                f'  combined projection = (dof5 * {_u["dof5"]:.4f} + dof0 * {_u["dof0"]:.4f}) / '
                f'({_u["dof5"]:.4f}^2 + {_u["dof0"]:.4f}^2)',
                '',
                'THE MIRROR FIGURE DEGREES OF FREEDOM, AS MEASURED QUANTITIES:',
                '',
                f'  v-mode 1 contains {_u["dof12"] * 1e3:+.3f} nm of M1M3 bending B3 and '
                f'{_u["dof34"] * 1e3:+.3f} nm of M2 bending B5 per unit',
                f'  amplitude, against {_u["dof5"]:+.1f} and {_u["dof0"]:+.1f} um for the two '
                f'hexapod dz. The predicted bending Trim',
                f'  therefore spans {np.nanmax(np.abs(_t["dof12_pred"])) * 1e3:.3f} nm and '
                f'{np.nanmax(np.abs(_t["dof34_pred"])) * 1e3:.3f} nm over these nights, while the '
                f'actual Trim',
                f'  has a standard deviation of {_t["dof12_last"].std() * 1e3:.1f} nm and '
                f'{_t["dof34_last"].std() * 1e3:.1f} nm -- larger by factors of '
                f'{_t["dof12_last"].std() / np.nanmax(np.abs(_t["dof12_pred"])):.1f} and '
                f'{_t["dof34_last"].std() / np.nanmax(np.abs(_t["dof34_pred"])):.1f}',
                '  (dimensionless, actual standard deviation over predicted span).',
                '',
                'THE OUTLIER NIGHTS ON THE ACTUAL-TRIM AXIS:',
                '',
                f'  Over these {len(_t)} nights the camera hexapod dz Trim the block settled on has',
                f'  median {t539res["dof5_last_center_um"]:+.2f} um and nMAD '
                f'{t539res["dof5_last_nmad_um"]:.2f} um. The nights beyond '
                f'{OUTLIER_NIGHT_Z:.0f} nMAD of that are the',
                '  points far up or down the vertical axis of the plots on the next page:',
                '',
                *([line for _, r in t539res['outliers'].iterrows() for line in (
                      f'    day_obs {int(r.day_obs)}  camera hexapod dz Trim '
                      f'{r.dof5_last:+9.2f} um, M2 hexapod dz Trim {r.dof0_last:+9.2f} um,',
                      f'      predicted focus error {r.focus_error_um:+8.1f} um of equivalent '
                      f'hexapod dz, z {r.dof5_last_z:+.1f} (dimensionless)')]
                  if len(t539res['outliers']) else
                  [f'    none: no night lies beyond {OUTLIER_NIGHT_Z:.0f} nMAD on this axis.']),
                '',
                '  z is dimensionless, the night\'s deviation over the nMAD of the nights. A night',
                '  that is an outlier here is one on which the alignment block asked for an unusual',
                '  amount of focus, which is not by itself a failure of the thermal prediction --',
                '  the predicted focus error on the same line says whether the model saw it coming.',
            ])
            figure_t539(pdf, t539res)
    print(f'\nwrote {pdf_path}')


if __name__ == '__main__':
    main()
