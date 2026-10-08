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
3. The deliverable thermal model — the fit, in sample over every night.
4. Camera-body temperature — an alternative thermometer, indistinguishable as a regressor.
5. What adds nothing — the channels whose gain does not exceed the across-band scatter.
5b. The quadratic radial M1M3 terms — three temperature fields going as radius squared, over the
   whole mirror, the M1 annulus and the M3 inner disc, tested for focus information the four
   bulk gradients cannot express: raw correlation, partial correlation with the deliverable
   features stripped from both sides, a night-grouped nested model comparison, and the
   substitution case a decision to switch from the gradients would rest on.
6. Residual shape — the one-sided positive tail that makes the fits robust rather than least
   squares.
7. Closed-loop focus performance — the measured v-mode-1 deviation on its own, with no commanded
   term and no thermal model, which is the performance an open-loop prediction is judged against.
8. The filter look-up table (LUT) — the signed focus step across every band transition. An exact
   filter LUT would move focus by zero on a filter change, and the antisymmetry of a transition
   pair separates a real filter offset from a focus drift that straddles the change.
8b. Outlier nights — the visits beyond three robust sigma of the fitted model, counted per night.
9. FAM blocks — the thermal prediction against the measured focus at the in-focus acquisition
   visit of each Full Array Mode (FAM) block, on the Danish 1.2 wavefront retrieval. The same
   test the science-visit pages make, on an independent sample and an independent retrieval.
10. The conversion table — the v-mode-1 to hexapod dz factor across projection schemes,
    including the 10-degree-of-freedom/1-mode case the online system would use.
11. The standalone calculator — ``trim_calculator.py``, which reads the fitted coefficients from
    ``trim_coefficients.yaml``, checked against the pipeline fitted here so the two cannot drift
    apart unnoticed.
12. The truss temperature alone — the one-thermometer correction the five-channel model must beat.
13. The focus correction as degrees of freedom — each visit's measured v-mode 1 back-projected into
    the camera and M2 hexapod dz it is built from, over all visits and at the start of each night.
14. Against the Trim the initial alignment block settled on — an independent measurement of the
    focus the telescope needed, not derived from the quantity the model was fitted to.

The PDF runs in one linear order, which is not the order above: the study description and the
summary plots first, then the telemetry-term comparisons that settle the model, then the resulting
trims, then the independent checks. The page sequence is the study, the open-loop focus by band and
against truss temperature, the truss temperature over the whole database, the nightly medians, the
individual and cumulative term grids, the term summary, the fitted model and its plots, the
quadratic radial terms, the FAM in-focus comparison, the correction as degrees of freedom over all
visits and at the start of each night, closed-loop performance, the filter LUT, the outlier-night
counts, and the initial alignment block — its outlier nights as a table, the prediction at the
first visit of the night, and the settled Trim per degree of freedom.

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
import test_trim_calculator as TT                                 # noqa: E402
import common.utils as U                                          # noqa: E402
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

#: Initial-alignment block duration taken to be the normal, converged length of the shortened
#: post-`T539_SHORT_EPOCH_DAY_OBS` block [min]. Nights outside this band are tabulated, since the
#: block is nominally a fixed sequence and a long one means it struggled to converge.
T539_NOMINAL_DURATION_MIN = (6.0, 9.0)

#: First `day_obs` of the shortened initial-alignment block. The block's median duration drops
#: from 23.1 min to 8.0 min across this date, so the two epochs are tabulated separately rather
#: than pooled into one bimodal distribution.
T539_SHORT_EPOCH_DAY_OBS = 20260201

#: Sun altitude defining the night's reference epoch [deg]. 0 deg is geometric sunset, the
#: "0 degree twilight" an observer times the start of the night against.
TWILIGHT_REF_ALT_DEG = 0.0

#: Hours after `TWILIGHT_REF_ALT_DEG` twilight separating the early part of the night from the
#: rest [h]. The residual's degradation is a transient confined to the first few hours rather
#: than a drift across the night, so a single split reports it better than a linear slope does:
#: the nMAD is flat either side of 3 h and steps across it.
EARLY_NIGHT_SPLIT_H = 3.0

#: Robust deviations beyond which a single VISIT is called an outlier of the fitted model
#: [dimensionless, residual over the nMAD of the residual]. Looser than `OUTLIER_NIGHT_Z`
#: because it counts visits rather than naming nights: a Gaussian would put 0.27% of visits
#: beyond it, so the measured fraction is readable as a statement about the residual's tails.
OUTLIER_VISIT_Z = 3.0

#: Fewest visits a night must carry before its outlier fraction is ranked [exposures]. A night of
#: a dozen visits can reach a high fraction on one outlier, which says nothing about the model.
MIN_OUTLIER_NIGHT_N = 50

#: Fewest filter changes an ordered band pair must carry, in BOTH directions, before its
#: antisymmetry is reported [transitions]. The rarer pairs carry medians dominated by a handful of
#: steps, and an accidental sign agreement in those would read as a focus drift.
MIN_TRANSITION_N = 20

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
        ``per_night_band`` (`pandas.DataFrame`: visits per night per band, indexed by date over
        every calendar day of the span so unobserved nights are gaps, columns in `BAND_COLOUR`
        order), ``feature_means`` (`dict`), ``interp_frac`` (per cent of visits whose truss
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

    # Visits per night per band, as a date-indexed count table. Reindexed over every calendar date
    # from the first night to the last, so a night the survey did not observe shows as a gap
    # rather than being closed up against its neighbour.
    pnb = (sci.pivot_table(index='day_obs', columns='band', values='y', aggfunc='size')
           .reindex(columns=[b for b in BAND_COLOUR if b in sci['band'].unique()])
           .fillna(0.0))
    pnb.index = pd.to_datetime(pnb.index.astype(int).astype(str), format='%Y%m%d')
    per_night_band = pnb.reindex(pd.date_range(pnb.index.min(), pnb.index.max(), freq='D'))

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
    return dict(per_band=per_band, per_night=per_night, per_night_band=per_night_band,
                feature_means=means, interp_frac=interp, night_line=night_line,
                outlier_nights=outliers)


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


def section_closed_loop(sci, verbose=True):
    """Closed-loop focus performance: the measured v-mode-1 deviation on its own.

    Every other page predicts the OPEN-loop focus, which combines the commanded Trim with the
    measured state. This section asks a different and simpler question: how well is the closed
    loop actually holding focus? That is the measured v-mode-1 deviation alone, converted to
    microns, with no commanded term and no thermal model.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, needing ``v1`` [dimensionless v-mode-1 amplitude], ``day_obs`` and
        ``obs_start_mjd`` [d].
    verbose : `bool`, optional
        Print the summary.

    Returns
    -------
    out : `dict`
        ``dz`` -- per-visit measured focus [µm of equivalent hexapod dz]; ``median`` and ``nmad``
        of it [µm]; ``p1`` and ``p99``, the 1st and 99th percentiles [µm], which bound the
        plotted range; ``p_lo``, ``p_hi``, ``min`` and ``max`` [µm], which bound the full-range
        page; ``n``; ``nightly`` -- per-night ``median``, ``nmad`` [µm], ``n`` and
        ``obs_start_mjd`` [d]; ``drift`` -- the `thermal_focus_fit.huber_line` of nightly median
        against date, or ``None`` when there are too few nights.

    Notes
    -----
    The sign is that of the stored v-modes divided by the positive conversion factor, so a
    negative value means the measured state carried defocus of the sign that a positive hexapod
    dz removes. The median is the standing offset the loop does not remove; the nMAD is the
    visit-to-visit scatter about it, and is the number that says how well focus is held.
    """
    conv = L.v1_per_um_dz_value(verbose=False)
    d = sci[np.isfinite(sci['v1'])].copy()
    d['meas_dz_um'] = d['v1'].to_numpy(float) / conv
    a = d['meas_dz_um'].to_numpy(float)
    out = {'dz': a, 'median': float(np.median(a)), 'nmad': float(nmad(a)), 'n': int(len(a)),
           'p1': float(np.percentile(a, 1)), 'p99': float(np.percentile(a, 99)),
           # The far tails, for the full-range page: the distribution reaches several hundred
           # microns either side, which the 1st-to-99th view deliberately crops away.
           'p_lo': float(np.percentile(a, 0.1)), 'p_hi': float(np.percentile(a, 99.9)),
           'min': float(a.min()), 'max': float(a.max())}

    g = d.groupby('day_obs')
    nightly = pd.DataFrame({'median': g['meas_dz_um'].median(),
                            'nmad': g['meas_dz_um'].apply(lambda v: nmad(v.to_numpy(float))),
                            'n': g['meas_dz_um'].size()}).reset_index()
    if 'obs_start_mjd' in d.columns:
        nightly = nightly.merge(g['obs_start_mjd'].min().rename('obs_start_mjd').reset_index(),
                                on='day_obs', how='left')
    out['nightly'] = nightly

    out['drift'] = None
    if 'obs_start_mjd' in nightly.columns and len(nightly) > 10:
        ok = nightly['obs_start_mjd'].notna() & nightly['median'].notna()
        if int(ok.sum()) > 10:
            out['drift'] = F.huber_line(nightly.loc[ok, 'obs_start_mjd'].to_numpy(float),
                                        nightly.loc[ok, 'median'].to_numpy(float))

    if verbose:
        print(f'  measured v-mode-1 focus over {out["n"]} visits and {len(nightly)} nights '
              f'[um of equivalent hexapod dz]')
        print(f'    median {out["median"]:+.2f}, nMAD {out["nmad"]:.2f}, '
              f'1st percentile {out["p1"]:+.1f}, 99th {out["p99"]:+.1f}')
        print(f'    per-night median: median over nights {nightly["median"].median():+.2f} um, '
              f'nMAD over nights {nmad(nightly["median"].to_numpy(float)):.2f} um')
        print(f'    per-night nMAD  : median over nights {nightly["nmad"].median():.2f} um')
        if out['drift']:
            dr = out['drift']
            print(f'    nightly median against date: slope {dr["slope"]:+.5f} +/- '
                  f'{dr["slope_err"]:.5f} um of equivalent hexapod dz per d '
                  f'({abs(dr["slope"]) / dr["slope_err"]:.1f} standard errors), '
                  f'Pearson r {dr["pearson_r"]:+.4f}, Spearman rho {dr["spearman_rho"]:+.4f}, '
                  f'n {dr["n"]} nights')
    return out


def section_intranight(sci, resid, n_bins=10, split_h=EARLY_NIGHT_SPLIT_H, verbose=True):
    """Does the prediction's quality change over the course of a night?

    The model carries no time-of-night term: it sees the thermal telemetry and nothing else. So
    if its residual has a systematic shape against time since sunset, some part of the focus
    drift is being driven by something the thermal channels do not capture — most plausibly a
    thermal lag, where a structure's temperature at a given moment does not yet reflect the heat
    already moving through it.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, needing ``day_obs`` and ``obs_start_mjd`` [d].
    resid : `array_like`
        Actual minus predicted open-loop focus, positionally aligned with `sci`
        [µm of equivalent hexapod dz].
    n_bins : `int`, optional
        Equal-count bins in time since twilight used for the binned median and nMAD.
    split_h : `float`, optional
        Hours after twilight separating the early night from the rest [h].
    verbose : `bool`, optional
        Print the per-bin table, the trend fits and the early-against-late split.

    Returns
    -------
    out : `dict` or `None`
        ``hours`` and ``resid`` (per visit, hours since the night's `TWILIGHT_REF_ALT_DEG`
        sunset and the residual [µm of equivalent hexapod dz]); ``bins``
        (`pandas.DataFrame` with ``centre`` [h], ``lo`` and ``hi`` [h], ``median`` and ``nmad``
        [µm], ``n``); ``median_trend`` and ``nmad_trend`` (`thermal_focus_fit.huber_line` of the
        binned median and binned nMAD against bin centre, so the slope is µm per h); ``split``
        (``split_h`` [h], and ``early``/``late`` each with ``median``, ``nmad`` [µm] and ``n``,
        plus ``nmad_ratio``, dimensionless, early over late); ``p1`` and ``p99`` of ``hours``;
        ``n``. `None` when the timing columns are absent.

    Notes
    -----
    Time is measured from the night's own geometric sunset rather than from its first exposure,
    so the zero means the same physical thing in June as in December. Over this season sunset
    moves by about 1.9 h, which is large enough that a first-exposure reference would smear any
    real time-of-night effect.

    The bins hold equal visit counts rather than equal widths, so the nMAD of each is estimated
    from the same number of visits and the late-night bins — which are thinly populated, since
    not every night runs to dawn — are widened rather than left noisy.

    **The effect is a transient, not a drift,** which is why the split is reported alongside the
    linear trends. The residual nMAD is roughly flat from 3 h after sunset to dawn and steps up
    sharply before it, and the median carries a positive bias over the same early window. A
    straight line through the whole night therefore understates it: the line is pulled down by a
    long flat tail and comes out at only about 2 standard errors on the median, while the step
    across `split_h` is unambiguous. Physically this is what a thermal lag looks like — early in
    the night the structure is still shedding the day's heat, so a temperature read at that
    moment does not yet describe the focus the glass is heading for.
    """
    if 'obs_start_mjd' not in sci.columns or 'day_obs' not in sci.columns:
        return None
    r = np.asarray(resid, float)
    mjd = sci['obs_start_mjd'].to_numpy(float)
    tw = U.evening_twilight_mjd(sci['day_obs'].to_numpy(int), alt_deg=TWILIGHT_REF_ALT_DEG)
    hours = (mjd - tw) * 24.0
    ok = np.isfinite(hours) & np.isfinite(r)
    if ok.sum() < 10 * n_bins:
        return None
    hours, r = hours[ok], r[ok]

    # Equal-count edges via quantiles; `np.unique` guards the degenerate case of a quantile
    # repeating, which would otherwise make an empty bin.
    edges = np.unique(np.quantile(hours, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.digitize(hours, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 10:
            continue
        rows.append({'centre': float(np.median(hours[m])), 'lo': float(edges[b]),
                     'hi': float(edges[b + 1]), 'median': float(np.median(r[m])),
                     'nmad': float(nmad(r[m])), 'n': int(m.sum())})
    bins = pd.DataFrame(rows)
    out = {'hours': hours, 'resid': r, 'bins': bins, 'n': int(len(r)),
           'p1': float(np.percentile(hours, 1)), 'p99': float(np.percentile(hours, 99)),
           'median_trend': None, 'nmad_trend': None}
    if len(bins) > 3:
        out['median_trend'] = F.huber_line(bins['centre'].to_numpy(float),
                                           bins['median'].to_numpy(float))
        out['nmad_trend'] = F.huber_line(bins['centre'].to_numpy(float),
                                         bins['nmad'].to_numpy(float))

    early, late = hours < split_h, hours >= split_h
    sp = {'split_h': float(split_h)}
    for key, m in (('early', early), ('late', late)):
        sp[key] = ({'median': float(np.median(r[m])), 'nmad': float(nmad(r[m])),
                    'n': int(m.sum())} if m.sum() > 10 else None)
    sp['nmad_ratio'] = (sp['early']['nmad'] / sp['late']['nmad']
                        if sp['early'] and sp['late'] and sp['late']['nmad'] > 0 else np.nan)
    out['split'] = sp

    if verbose:
        print(f'  residual against time since {TWILIGHT_REF_ALT_DEG:.0f} deg twilight, '
              f'{out["n"]} visits in {len(bins)} equal-count bins')
        print('    hours since twilight   median [um]   nMAD [um]   n visits')
        for _, b in bins.iterrows():
            print(f'    {b["lo"]:5.2f} to {b["hi"]:5.2f} h       '
                  f'{b["median"]:+7.2f}      {b["nmad"]:7.2f}     {int(b["n"]):6d}')
        for key, what, unit in (('median_trend', 'binned median', 'um'),
                                ('nmad_trend', 'binned nMAD', 'um')):
            t = out[key]
            if t:
                print(f'    {what} against hours since twilight: slope {t["slope"]:+.3f} '
                      f'+/- {t["slope_err"]:.3f} {unit} of equivalent hexapod dz per h '
                      f'({abs(t["slope"]) / t["slope_err"]:.1f} standard errors), '
                      f'Pearson r {t["pearson_r"]:+.3f}, Spearman rho '
                      f'{t["spearman_rho"]:+.3f}, n {t["n"]} bins')
        if sp['early'] and sp['late']:
            print(f'    the effect is a step, not a drift -- split at {split_h:.0f} h after '
                  f'twilight [um of equivalent hexapod dz]:')
            print(f'      earlier: median {sp["early"]["median"]:+.2f}, nMAD '
                  f'{sp["early"]["nmad"]:.2f}, n {sp["early"]["n"]} visits')
            print(f'      later  : median {sp["late"]["median"]:+.2f}, nMAD '
                  f'{sp["late"]["nmad"]:.2f}, n {sp["late"]["n"]} visits')
            print(f'      nMAD ratio {sp["nmad_ratio"]:.2f} (dimensionless, early over late)')
    return out


def section_filter_lut(sci, verbose=True):
    """Focus steps across a filter change: how well the filter look-up table is working.

    Each filter sits at a different optical thickness, so the Active Optics System carries a
    per-filter focus offset in its look-up table (LUT). If that LUT were exact, changing filter
    would not move focus. The signed step in open-loop focus across a filter change is therefore
    a direct measurement of the residual error in the filter LUT.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, needing ``day_obs``, ``seq_num``, ``band`` and ``y`` [µm of equivalent
        hexapod dz].
    verbose : `bool`, optional
        Print the per-transition table.

    Returns
    -------
    out : `dict`
        ``pairs`` -- one row per ordered band pair with ``from``, ``to``, ``median`` [µm of
        equivalent hexapod dz], ``nmad`` [µm] and ``n``; ``change`` and ``same`` -- the signed
        steps across a filter change and between consecutive same-band visits [µm];
        ``change_median``, ``change_nmad``, ``same_median``, ``same_nmad`` [µm] with ``n_change``
        and ``n_same``; ``per_band`` -- per-band ``median`` and ``nmad`` of the open-loop focus
        [µm] with ``n``; ``antisym`` -- one row per unordered pair with the two signed medians
        and their ``sum`` [µm], which separates a genuine filter offset from a drift.

    Notes
    -----
    **Antisymmetry is what distinguishes the two causes.** A real filter-LUT offset reverses sign
    when the transition is taken the other way, so the two signed medians sum to about zero. A
    pair whose two legs carry the SAME sign is not a filter offset at all but a focus drift that
    happens to straddle the filter change -- the telescope was moving in focus for another
    reason, and the filter change merely marks where it was sampled.

    Steps are taken within a night only, between consecutive ``seq_num``, so the daytime gap is
    never crossed.
    """
    d = sci[np.isfinite(sci['y'])].sort_values(['day_obs', 'seq_num'])
    rows, same = [], []
    for _, g in d.groupby('day_obs'):
        g = g.sort_values('seq_num')
        step = g['y'].diff()
        changed = g['band'].ne(g['band'].shift(1))
        ok = step.notna().to_numpy()
        ch = changed.to_numpy()
        for i in np.where(ok & ch)[0]:
            rows.append((g['band'].iloc[i - 1], g['band'].iloc[i], float(step.iloc[i])))
        same.append(step[(~changed) & step.notna()])

    s = pd.DataFrame(rows, columns=['from', 'to', 'step'])
    same = pd.concat(same).to_numpy(float) if same else np.array([])
    chg = s['step'].to_numpy(float) if len(s) else np.array([])

    pairs = (s.groupby(['from', 'to'])['step']
             .agg(median='median', n='size', nmad=lambda v: nmad(v.to_numpy(float)))
             .reset_index() if len(s) else pd.DataFrame(columns=['from', 'to', 'median', 'n',
                                                                 'nmad']))

    anti, seen = [], set()
    for _, r in pairs.iterrows():
        key = frozenset((r['from'], r['to']))
        if key in seen or int(r['n']) < MIN_TRANSITION_N:
            continue
        back = pairs[(pairs['from'] == r['to']) & (pairs['to'] == r['from'])]
        if not len(back) or int(back['n'].iloc[0]) < MIN_TRANSITION_N:
            continue
        seen.add(key)
        anti.append({'a': r['from'], 'b': r['to'],
                     'median_ab': float(r['median']), 'n_ab': int(r['n']),
                     'median_ba': float(back['median'].iloc[0]), 'n_ba': int(back['n'].iloc[0]),
                     'sum': float(r['median']) + float(back['median'].iloc[0])})
    anti = pd.DataFrame(anti)

    pb = d.groupby('band')['y']
    per_band = pd.DataFrame({'median': pb.median(),
                             'nmad': pb.apply(lambda v: nmad(v.to_numpy(float))),
                             'n': pb.size()}).reset_index()

    out = {'pairs': pairs, 'change': chg, 'same': same, 'antisym': anti, 'per_band': per_band,
           'change_median': float(np.median(chg)) if len(chg) else float('nan'),
           'change_nmad': float(nmad(chg)) if len(chg) else float('nan'),
           'n_change': int(len(chg)),
           'same_median': float(np.median(same)) if len(same) else float('nan'),
           'same_nmad': float(nmad(same)) if len(same) else float('nan'),
           'n_same': int(len(same))}

    if verbose:
        print('  signed step in open-loop focus [um of equivalent hexapod dz]')
        print(f'    across a filter change : median {out["change_median"]:+.2f}, '
              f'nMAD {out["change_nmad"]:.1f}, n {out["n_change"]}')
        print(f'    same band, consecutive : median {out["same_median"]:+.2f}, '
              f'nMAD {out["same_nmad"]:.1f}, n {out["n_same"]}')
        print(f'    ratio of the two nMAD {out["change_nmad"] / out["same_nmad"]:.2f} '
              f'(dimensionless, band-change nMAD over same-band nMAD)')
        print('  per-band open-loop focus [um of equivalent hexapod dz]')
        for _, r in per_band.iterrows():
            print(f'    {r["band"]}  n {int(r["n"]):6d}  median {r["median"]:+8.1f}  '
                  f'nMAD {r["nmad"]:7.1f}')
        if len(anti):
            print(f'  antisymmetry of each transition pair, n >= {MIN_TRANSITION_N} both ways '
                  f'[um of equivalent hexapod dz]')
            print('    a sum near zero is a real filter-LUT offset; both legs of one sign is a '
                  'focus drift')
            for _, r in anti.iterrows():
                print(f'    {r["a"]}->{r["b"]} {r["median_ab"]:+6.1f} (n {int(r["n_ab"]):4d})   '
                      f'{r["b"]}->{r["a"]} {r["median_ba"]:+6.1f} (n {int(r["n_ba"]):4d})   '
                      f'sum {r["sum"]:+6.1f}')
    return out


def section_outlier_nights(sci, resid, verbose=True):
    """Count the visits beyond 3 nMAD of the fitted model, night by night.

    Parameters
    ----------
    sci : `pandas.DataFrame`
        Science visits, needing ``day_obs`` and ``obs_start_mjd`` [d].
    resid : `array_like`
        In-sample residual of the deliverable model [µm of equivalent hexapod dz].
    verbose : `bool`, optional
        Print the worst nights.

    Returns
    -------
    out : `dict`
        ``nightly`` -- per night, ``n``, ``n_outlier``, ``frac_outlier`` [dimensionless,
        outliers over visits that night] and ``obs_start_mjd`` [d]; ``threshold_um`` -- the
        3 nMAD cut [µm of equivalent hexapod dz]; ``frac_all`` -- the fraction over the whole
        sample [dimensionless]; ``worst`` -- the ten nights with the largest outlier fraction,
        at or above `MIN_OUTLIER_NIGHT_N` visits.

    Notes
    -----
    The threshold is a single global 3 nMAD of the residual, not a per-night one, so a night on
    which the model did badly shows up as a high count rather than being renormalized away. A
    Gaussian would put 0.27% of visits beyond it, so the sample-wide fraction is itself a
    statement about the residual's tails.
    """
    r = np.asarray(resid, float)
    thr = OUTLIER_VISIT_Z * float(nmad(r[np.isfinite(r)]))
    d = sci.copy()
    d['_out'] = np.abs(r) > thr
    g = d.groupby('day_obs')
    nightly = pd.DataFrame({'n': g['_out'].size(), 'n_outlier': g['_out'].sum()}).reset_index()
    nightly['frac_outlier'] = nightly['n_outlier'] / nightly['n']
    if 'obs_start_mjd' in d.columns:
        nightly = nightly.merge(g['obs_start_mjd'].min().rename('obs_start_mjd').reset_index(),
                                on='day_obs', how='left')
    worst = (nightly[nightly['n'] >= MIN_OUTLIER_NIGHT_N]
             .sort_values('frac_outlier', ascending=False).head(10))
    out = {'nightly': nightly, 'threshold_um': thr, 'worst': worst,
           'frac_all': float(np.mean(np.abs(r[np.isfinite(r)]) > thr))}
    if verbose:
        print(f'  threshold {OUTLIER_VISIT_Z:.0f} nMAD = {thr:.1f} um of equivalent hexapod dz; '
              f'{100 * out["frac_all"]:.2f}% of all visits lie beyond it')
        print('    (a Gaussian would give 0.27%; the excess is the residual tail)')
        print(f'  worst nights by outlier fraction, at least {MIN_OUTLIER_NIGHT_N} visits:')
        for _, w in worst.iterrows():
            print(f'    day_obs {int(w["day_obs"])}  {int(w["n_outlier"]):4d} of '
                  f'{int(w["n"]):4d} visits  {100 * w["frac_outlier"]:5.1f}%')
    return out


def section_fam(fam, full, features, verbose=True):
    """The thermal prediction against the measured open-loop focus at the FAM in-focus visit.

    This is the same test the science-visit pages make, on an independent sample and a different
    wavefront retrieval. Every number elsewhere in this document rests on the Consolidated
    Database (ConsDB) corner-sensor Zernikes, because those are the only retrieval available on
    every science visit. The Full Array Mode (FAM) blocks are processed instead through the Danish
    1.2 pipeline, so if the prediction holds here it holds across two independent routes to the
    optical state.

    The state compared is that of the block's **in-focus acquisition visit** -- the recovered
    v-mode amplitudes are joined on ``acq_visit_id``, not on the defocused triplet exposures -- so
    the open-loop focus is formed exactly as it is for a science visit.

    Parameters
    ----------
    fam : `pandas.DataFrame`
        FAM table from the build stage, carrying ``y`` [µm of equivalent hexapod dz], the
        deliverable feature columns and ``day_obs``.
    full : `dict`
        Result of `thermal_focus_fit.fit_full` on the science visits. Its fitted model is applied
        unchanged to the FAM rows, so the FAM sample never sets a coefficient.
    features : `list` [`str`]
        The deliverable feature columns.
    verbose : `bool`, optional
        Print the comparison.

    Returns
    -------
    out : `dict`
        ``table`` -- the FAM rows that carry both a response and every feature, with ``pred`` and
        ``resid`` added [µm of equivalent hexapod dz]; ``y``, ``pred``, ``resid`` as arrays [µm];
        ``median`` and ``nmad`` of the residual [µm]; ``nmad_y`` -- the nMAD of the open-loop
        focus itself [µm], which the residual has to beat; ``gain`` (dimensionless, ``nmad_y``
        over ``nmad``); ``line`` -- the `thermal_focus_fit.huber_line` of predicted against
        measured; ``n`` visits and ``n_nights``.

    Notes
    -----
    The coefficients come from the science-visit fit, not from a fit to the FAM rows, so this is a
    genuine out-of-sample application and not a second fit that would be guaranteed to look good.
    A residual nMAD comparable to the science-visit one says the thermal relation is a property of
    the telescope rather than of the ConsDB retrieval.
    """
    if fam is None or not len(fam):
        return {}
    have = [c for c in features if c in fam.columns]
    if len(have) != len(features) or 'y' not in fam.columns:
        if verbose:
            miss = [c for c in features if c not in have] + (['y'] if 'y' not in fam else [])
            print(f'  FAM section skipped: {", ".join(miss)} absent from the FAM table')
        return {}

    d = fam[np.isfinite(fam['y'])].copy()
    d = d[np.all(np.isfinite(d[features].to_numpy(float)), axis=1)]
    if len(d) < 50:
        if verbose:
            print(f'  FAM section skipped: only {len(d)} rows carry a response and every feature')
        return {}

    pred = full['model'].predict(d[features].to_numpy(float))
    y = d['y'].to_numpy(float)
    resid = y - pred
    d['pred'] = pred
    d['resid'] = resid

    out = {'table': d, 'y': y, 'pred': pred, 'resid': resid,
           'median': float(np.median(resid)), 'nmad': float(nmad(resid)),
           'nmad_y': float(nmad(y)), 'n': int(len(d)),
           'n_nights': int(d['day_obs'].nunique()),
           'line': F.huber_line(y, pred)}
    out['gain'] = out['nmad_y'] / out['nmad'] if out['nmad'] else float('nan')

    if verbose:
        print(f'  {out["n"]} FAM in-focus acquisition visits over {out["n_nights"]} nights, '
              f'day_obs {int(d.day_obs.min())} to {int(d.day_obs.max())}, Danish 1.2 processing')
        print('  the science-visit coefficients applied unchanged to these rows '
              '[um of equivalent hexapod dz]')
        print(f'    open-loop focus      median {np.median(y):+8.1f}  nMAD {out["nmad_y"]:7.1f}')
        print(f'    after the correction median {out["median"]:+8.1f}  nMAD {out["nmad"]:7.1f}')
        print(f'    improvement {out["gain"]:.2f}x (dimensionless, open-loop focus nMAD over '
              f'residual nMAD)')
        ln = out['line']
        print(f'    predicted against measured: Huber slope {ln["slope"]:+.4f} +/- '
              f'{ln["slope_err"]:.4f} (dimensionless, predicted per measured),')
        print(f'      Pearson r {ln["pearson_r"]:+.4f}, Spearman rho '
              f'{ln["spearman_rho"]:+.4f}, n {ln["n"]}')
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
        over the worked cases in ``trim_test_cases.yaml``, and ``n`` — all µm of equivalent
        hexapod dz — plus ``n_worked_cases``.

    Notes
    -----
    `trim_calculator` is a standalone copy that reads its coefficients, rounded to two decimals,
    from ``trim_coefficients.yaml``. That copy can drift from the fit silently, which is exactly
    what this section exists to catch: if a coefficient here is re-fitted and the calculator's
    file is not updated, ``max_abs_diff_um`` grows from rounding noise to something that matters.
    """
    cols = ['truss_temp_mean_c', 'm1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
            'm1m3_radial_gradient_c_per_m', 'm1m3_x_gradient_c_per_m']
    if any(c not in sci.columns for c in cols) or list(features) != cols:
        if verbose:
            print('  the fitted feature set is not the calculator\'s five channels, so the '
                  'calculator is not comparable here; skipped')
        return {}
    calculator = T.TrimCalculator()
    _, pred_um, _ = calculator.predict_trim(*[sci[c].to_numpy(float) for c in cols],
                                            warn_extrapolation=False)
    diff = pred_um - np.asarray(full['pred'], float)
    _, test_cases = TT.load_test_cases()
    worked = max(abs(calculator.predict_trim(**inp, warn_extrapolation=False)[1] - exp)
                 for _, inp, exp in test_cases)
    out = {'n': int(len(sci)),
           'max_abs_diff_um': float(np.nanmax(np.abs(diff))),
           'median_diff_um': float(np.nanmedian(diff)),
           'worked_max_abs_diff_um': float(worked),
           'n_worked_cases': len(test_cases)}
    if verbose:
        print(f'  calculator against the fitted pipeline over {out["n"]} visits: '
              f'max |difference| {out["max_abs_diff_um"]:.4f}, median '
              f'{out["median_diff_um"]:+.4f} um of equivalent hexapod dz')
        print(f'  its {len(test_cases)} worked cases agree with their stated values to '
              f'{out["worked_max_abs_diff_um"]:.4f} um of equivalent hexapod dz')
        print(f'  coefficients from {calculator.path.name}: intercept '
              f'{calculator.intercept_um:+.2f} um, truss {calculator.truss_um_per_c:+.2f} um '
              f'per deg C')
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


def _t539_duration(df):
    """Summarise the wall-clock span of the initial alignment run, night by night.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The t539 table, needing ``run_duration_min`` [min] and ``day_obs``.

    Returns
    -------
    out : `dict` or `None`
        ``minutes`` (`numpy.ndarray`, per night [min]), ``day_obs`` (`numpy.ndarray`, aligned with
        it), ``median``, ``p16``, ``p84``, ``min`` and ``max`` [min], and ``n`` nights. `None` when
        the column is absent, which is the case for a table written before it existed.

    Notes
    -----
    This is the delay the run-first against run-last comparison carries: the prediction is made at
    the start of the span and the Trim is read at its end. The spread matters more than the median,
    because the run length is not fixed.
    """
    if 'run_duration_min' not in df.columns:
        return None
    d = df[['day_obs', 'run_duration_min']].dropna()
    if not len(d):
        return None
    m = d['run_duration_min'].to_numpy(float)
    return {'minutes': m, 'day_obs': d['day_obs'].to_numpy(int),
            'median': float(np.median(m)), 'p16': float(np.percentile(m, 16)),
            'p84': float(np.percentile(m, 84)), 'min': float(m.min()), 'max': float(m.max()),
            'n': int(len(m))}


def _t539_long_runs(df, nominal=T539_NOMINAL_DURATION_MIN, since=T539_SHORT_EPOCH_DAY_OBS):
    """Tabulate the shortened-epoch nights whose alignment block was not the nominal length.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The t539 table, needing ``day_obs``, ``run_duration_min`` [min], ``n_run`` [exposures]
        and ``obs_start_mjd_first`` [d].
    nominal : `tuple` [`float`], optional
        Low and high duration bounds taken to be a normal, converged block [min]. A night inside
        this band is not listed.
    since : `int`, optional
        First `day_obs` considered, the date the block was shortened.

    Returns
    -------
    out : `dict` or `None`
        ``table`` (`pandas.DataFrame`, one row per listed night, sorted by duration, with
        ``day_obs``, ``run_duration_min`` [min], ``n_run`` [exposures],
        ``min_after_twilight`` [min after the night's 0 deg sunset] and ``min_per_visit`` [min]),
        ``n_epoch`` nights in the epoch, ``n_nominal`` of them inside the band, ``n_short`` below
        it, ``n_long`` above it, ``nominal`` and ``since`` as given. `None` when the columns are
        absent.

    Notes
    -----
    The block is nominally a fixed sequence, so after it was shortened its duration should be
    close to constant. It is not: a tail runs to several times the nominal length. Two
    explanations separate on this table. If the long nights also carry more exposures, the block
    iterated further to converge; if they carry the nominal exposure count spread over a longer
    time, the delay is between exposures and so is readout, slew or an operator pause rather than
    the alignment itself. ``min_per_visit`` is the discriminator and is why the exposure count is
    tabulated beside the duration.

    Time after 0 deg twilight is included because the other candidate explanation is thermal: a
    block run early, while the dome is still dumping the day's heat, has further to converge. It
    is computed here from `common.utils.evening_twilight_mjd` rather than read from the
    value-added database, which does not carry it.
    """
    need = ('day_obs', 'run_duration_min', 'n_run', 'obs_start_mjd_first')
    if any(c not in df.columns for c in need):
        return None
    d = df[list(need)].dropna(subset=['run_duration_min'])
    d = d[d['day_obs'].astype(int) >= int(since)]
    if not len(d):
        return None

    lo, hi = float(nominal[0]), float(nominal[1])
    dur = d['run_duration_min'].to_numpy(float)
    out = d[(dur < lo) | (dur > hi)].copy()
    tw = U.evening_twilight_mjd(out['day_obs'].to_numpy(int), alt_deg=TWILIGHT_REF_ALT_DEG)
    out['min_after_twilight'] = (out['obs_start_mjd_first'].to_numpy(float) - tw) * 1440.0
    out['min_per_visit'] = out['run_duration_min'] / out['n_run'].clip(lower=1)
    out = (out[['day_obs', 'run_duration_min', 'n_run', 'min_after_twilight', 'min_per_visit']]
           .sort_values('run_duration_min', ascending=False).reset_index(drop=True))
    return {'table': out, 'n_epoch': int(len(d)),
            'n_nominal': int(((dur >= lo) & (dur <= hi)).sum()),
            'n_short': int((dur < lo).sum()), 'n_long': int((dur > hi).sum()),
            'nominal': (lo, hi), 'since': int(since)}


def _t539_sci1(df, full, features):
    """The prediction against the open-loop focus of the first science visit after the run.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The t539 table, needing ``y_sci1`` [µm of equivalent hexapod dz], ``gap_to_sci1_min``
        [min] and every feature in `features` suffixed ``_sci1``.
    full : `dict`
        Result of `thermal_focus_fit.fit_full`, carrying the fitted pipeline under ``model``.
    features : `list` [`str`]
        Feature columns of the deliverable model, in fit order.

    Returns
    -------
    out : `dict` or `None`
        ``actual`` and ``pred`` [µm of equivalent hexapod dz], ``day_obs``, ``gap_min`` [min],
        ``fit`` (`thermal_focus_fit.huber_line` of actual against predicted), ``diff_median`` and
        ``diff_nmad`` [µm], ``n`` nights, and ``outliers`` (`pandas.DataFrame` of the nights whose
        difference exceeds `OUTLIER_NIGHT_Z` robust deviations, with ``day_obs``, ``actual``,
        ``pred``, ``diff`` [µm], ``gap_min`` and ``z``). `None` when the columns are absent or
        nothing is finite.

    Notes
    -----
    This removes the one systematic the run-first against run-last comparison cannot avoid: those
    two epochs are separated by the whole alignment block, so the telescope's thermal state moves
    between them.

    **Both sides are evaluated at the same visit.** The prediction is made from the thermal
    telemetry of the first science visit after the block, not from the telemetry at the block's
    start, so there is no time gap left to correct for and no nights need be cut. An earlier
    version predicted at the block's first visit and kept only nights whose block-to-science gap
    was under `T539_SCI1_MAX_GAP_MIN`, which cost 130 of 149 nights to buy an approximation of
    what this does exactly.

    ``gap_min`` is carried through anyway, as a diagnostic: it no longer selects nights, but a
    long gap still means the block's converged Trim is stale by the time the science visit is
    taken, which is a property of the night rather than of the prediction.
    """
    need = ['y_sci1', 'gap_to_sci1_min'] + [f'{c}_sci1' for c in features]
    if any(c not in df.columns for c in need):
        return None
    actual_all = df['y_sci1'].to_numpy(float)
    X = np.column_stack([df[f'{c}_sci1'].to_numpy(float) for c in features])
    # The model cannot score a row with a missing feature, so those rows are predicted as NaN
    # rather than being dropped before the call, which would misalign the result with `df`.
    pred_all = np.full(len(df), np.nan)
    rows_ok = np.isfinite(X).all(axis=1)
    if rows_ok.any():
        pred_all[rows_ok] = np.asarray(full['model'].predict(X[rows_ok]), float)

    keep = np.isfinite(actual_all) & np.isfinite(pred_all)
    if not keep.any():
        return None

    actual, pred = actual_all[keep], pred_all[keep]
    day_obs = df['day_obs'].to_numpy(int)[keep]
    gap = df['gap_to_sci1_min'].to_numpy(float)[keep]
    diff = actual - pred
    centre, spread = float(np.median(diff)), float(nmad(diff))
    z = (diff - centre) / spread if spread > 0 else np.full(len(diff), np.nan)
    o = pd.DataFrame({'day_obs': day_obs, 'actual': actual, 'pred': pred, 'diff': diff,
                      'gap_min': gap, 'z': z})
    o = (o[np.abs(o['z']) > OUTLIER_NIGHT_Z]
         .sort_values('z', key=abs, ascending=False).reset_index(drop=True))
    return {'actual': actual, 'pred': pred, 'day_obs': day_obs, 'gap_min': gap,
            'fit': F.huber_line(actual, pred), 'diff_median': centre, 'diff_nmad': spread,
            'n': int(keep.sum()), 'outliers': o}


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
        ``dof5_last_center_um`` and ``dof5_last_nmad_um`` — that median and nMAD [µm];
        ``duration`` — `_t539_duration`, the wall-clock span of the run per night [min];
        ``sci1`` — `_t539_sci1`, the prediction evaluated at the first science visit after the
        run against that visit's own open-loop focus, which removes the delay the two epochs
        below carry; ``long_runs`` — `_t539_long_runs`, the shortened-block nights whose duration
        falls outside `T539_NOMINAL_DURATION_MIN`.

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
    # The outlier table is the deliverable page for these nights, so it carries both epochs:
    # the seq_num and truss temperature at the run's first and last exposure, the predicted and
    # actual v-mode-1 amplitude, and the Trim the block settled on for each hexapod.
    _owant = ['day_obs', 'seq_num_first', 'seq_num_last',
              'truss_temp_mean_c_first', 'truss_temp_mean_c_last',
              'v1_pred', 'v1_actual', 'dof5_last', 'dof0_last', 'focus_error_um', 'dof5_last_z']
    t539_outliers = (df.loc[df['dof5_last_z'].abs() > OUTLIER_NIGHT_Z,
                            [c for c in _owant if c in df.columns]]
                     .reindex(df['dof5_last_z'].abs().sort_values(ascending=False).index)
                     .dropna(subset=['dof5_last_z']))

    out = {'table': df, 'rows': rows, 'fits': fits, 'diff': diff,
           'unit': {key: float(unit_vec[idx]) for idx, key, _, _, _ in wanted},
           'outliers': t539_outliers,
           'dof5_last_center_um': c5, 'dof5_last_nmad_um': s5,
           'duration': _t539_duration(df),
           'sci1': _t539_sci1(df, full, features),
           'long_runs': _t539_long_runs(df),
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
        # The v1_dz comparison in focus units, which is the one the report plots and the one to
        # quote: the per-hexapod rows above split focus in a way the alignment chose freely.
        _conv = L.v1_per_um_dz_value(dof_set=dof_set, n_modes=n_modes, verbose=False)
        for _k, _lab in (('v1_actual', 'actual v1_dz, from the settled Trim'),
                         ('v1_pred', 'predicted v1_dz, from the thermal telemetry')):
            _v = df[_k].to_numpy(float) / _conv
            _v = _v[np.isfinite(_v)]
            print(f'    {_lab:46s} median {np.median(_v):+8.1f}  nMAD {nmad(_v):7.1f} '
                  f'um of equivalent hexapod dz  n {len(_v)}')
        print(f'  camera hexapod dz Trim settled on: median {c5:+.2f} um, nMAD {s5:.2f} um over '
              f'{len(df)} nights')
        print(f'  nights beyond {OUTLIER_NIGHT_Z:.0f} nMAD on that axis: {len(t539_outliers)}')
        for _, r in t539_outliers.iterrows():
            print(f'    day_obs {int(r.day_obs)}  camera hexapod dz Trim {r.dof5_last:+9.2f} um, '
                  f'M2 hexapod dz Trim {r.dof0_last:+9.2f} um, predicted focus error '
                  f'{r.focus_error_um:+8.1f} um of equivalent hexapod dz, '
                  f'z {r.dof5_last_z:+.1f} (dimensionless)')
        du = out['duration']
        if du:
            print(f'  run duration, first exposure start to last exposure end [min]: median '
                  f'{du["median"]:.1f}, 16th to 84th percentile {du["p16"]:.1f} to '
                  f'{du["p84"]:.1f}, range {du["min"]:.1f} to {du["max"]:.1f}, n {du["n"]} nights')
        s1 = out['sci1']
        if s1:
            ln = s1['fit']
            print(f'  prediction and measurement both at the first science visit after the run, '
                  f'no time gap: {s1["n"]} nights')
            print(f'    open-loop focus at that visit against the prediction: Huber slope '
                  f'{ln["slope"]:+.3f} +/- {ln["slope_err"]:.3f} (dimensionless, predicted per '
                  f'actual),')
            print(f'    Pearson r {ln["pearson_r"]:+.4f}, Spearman rho '
                  f'{ln["spearman_rho"]:+.4f}, n {ln["n"]} nights')
            print(f'    actual minus predicted: median {s1["diff_median"]:+.1f}, nMAD '
                  f'{s1["diff_nmad"]:.1f} um of equivalent hexapod dz')
            print(f'    nights beyond {OUTLIER_NIGHT_Z:.0f} nMAD of that difference: '
                  f'{len(s1["outliers"])}')
            for _, r in s1['outliers'].iterrows():
                print(f'      day_obs {int(r.day_obs)}  actual {r.actual:+8.1f}, predicted '
                      f'{r.pred:+8.1f}, difference {r["diff"]:+8.1f} um of equivalent hexapod '
                      f'dz, gap {r.gap_min:.1f} min, z {r.z:+.1f} (dimensionless)')
        lr = out['long_runs']
        if lr:
            lo, hi = lr['nominal']
            print(f'  block duration outside {lo:.0f} to {hi:.0f} min, day_obs >= '
                  f'{lr["since"]}: {len(lr["table"])} of {lr["n_epoch"]} nights '
                  f'({lr["n_short"]} under, {lr["n_long"]} over; {lr["n_nominal"]} nominal)')
            print('    day_obs   duration [min]  visits  min after 0 deg twilight  '
                  'min per visit')
            for _, r in lr['table'].iterrows():
                print(f'    {int(r.day_obs)}      {r.run_duration_min:8.1f}    '
                      f'{int(r.n_run):4d}          {r.min_after_twilight:8.1f}'
                      f'          {r.min_per_visit:6.2f}')
    return out


# ------------------------------------------------------------------------------------ figures

def _fmt_i(v):
    """Format a possibly-missing integer-valued quantity for a monospace table cell.

    Parameters
    ----------
    v : `float` or `int` or `None`
        The value, in whatever unit the table's header declares.

    Returns
    -------
    s : `str`
        The value with no decimals, or ``'--'`` when it is absent or not finite.
    """
    try:
        return f'{int(v)}' if v is not None and np.isfinite(float(v)) else '--'
    except (TypeError, ValueError):
        return '--'


def _fmt_f(v, nd=2):
    """Format a possibly-missing float for a monospace table cell.

    Parameters
    ----------
    v : `float` or `None`
        The value, in whatever unit the table's header declares.
    nd : `int`, optional
        Decimal places.

    Returns
    -------
    s : `str`
        The signed value, or ``'--'`` when it is absent or not finite.
    """
    try:
        return f'{float(v):+.{nd}f}' if v is not None and np.isfinite(float(v)) else '--'
    except (TypeError, ValueError):
        return '--'


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


def _clip_axes_to_percentile(ax, x, y, lo_pct=2.0, hi_pct=98.0, pad=0.08):
    """Set both axis ranges to a percentile of the plotted points, with a little padding.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes whose limits are set.
    x, y : `array_like`
        The plotted points, in whatever units the panel carries.
    lo_pct, hi_pct : `float`, optional
        Percentiles bounding each axis [dimensionless, percent]. The 2nd and 98th are used rather
        than the 1st and 99th because these panels plot night medians, roughly 150 points, so the
        1st percentile clips only a point or two per tail and a single extreme night still sets
        the range.
    pad : `float`, optional
        Fraction of the resulting span added at each end [dimensionless].

    Notes
    -----
    A handful of nights carry a night median far enough out that autoscaling compresses every
    other point and the fitted line into a few pixels. Clipping the axes rather than the data
    leaves the fit itself computed on every point; only the view is narrowed. The panel title
    still carries the correlations over the full sample, so the number a reader quotes is not the
    clipped one.
    """
    for setlim, v in ((ax.set_xlim, x), (ax.set_ylim, y)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            continue
        a, b = np.percentile(v, [lo_pct, hi_pct])
        if b > a:
            setlim(a - pad * (b - a), b + pad * (b - a))


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
        _clip_axes_to_percentile(ax, pn[c].to_numpy(float), pn['y'].to_numpy(float))
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
            _clip_axes_to_percentile(ax, pe['x'].to_numpy(float), pe['y'].to_numpy(float))
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


def _actual_vs_predicted_pair(axes, y, pred, what, nbins=60, clip_pct=(0.5, 99.5)):
    """Draw the standard two-panel actual-against-predicted comparison.

    Parameters
    ----------
    axes : `list` [`matplotlib.axes.Axes`]
        Two axes: the scatter and the difference histogram.
    y, pred : `array_like`
        The measured and predicted quantity, same length [µm of equivalent hexapod dz].
    what : `str`
        Short description of the sample, used in the panel titles.
    nbins : `int`, optional
        Bins in the difference histogram.
    clip_pct : `tuple` [`float`], optional
        Percentiles bounding the plotted ranges [dimensionless, percent]. The median, nMAD and
        the fitted line are computed on every finite point, not on the clipped view.

    Notes
    -----
    Used for the FAM in-focus visits and for the first visit after initial alignment, so the two
    pages are read the same way and their numbers are directly comparable. The quantity is
    v1_dz -- v-mode 1 converted to µm of equivalent hexapod dz, 0.5 µm on the camera hexapod and
    0.5 µm on M2.
    """
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    diff = y - pred

    ax = axes[0]
    ax.plot(y, pred, 'o', ms=3.5, color='#1f77b4', alpha=0.65)
    both = np.r_[y, pred]
    lo, hi = np.percentile(both, clip_pct)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '-', color='#d62728', lw=1.2,
            label='perfect prediction')
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ln = F.huber_line(y, pred)
    ax.set_xlabel('actual v1_dz [um of equivalent hexapod dz]')
    ax.set_ylabel('predicted v1_dz\n[um of equivalent hexapod dz]')
    ax.set_title(f'{what}, n {len(y)}\nHuber slope {ln["slope"]:+.3f} +/- '
                 f'{ln["slope_err"]:.3f} (dimensionless), Pearson r {ln["pearson_r"]:+.3f}, '
                 f'Spearman rho {ln["spearman_rho"]:+.3f}', fontsize=8.5)
    ax.legend(fontsize=7.5, loc='upper left')

    ax = axes[1]
    dlo, dhi = np.percentile(diff, clip_pct)
    # A percentile clip cannot exclude a lone outlier from a sample of a few tens -- the 98th
    # percentile of 33 points still sits above it -- so the range is also held to a robust span
    # about the median. Whichever is tighter wins; the overflow piles into the end bin.
    _c, _s = float(np.median(diff)), float(nmad(diff))
    if _s > 0:
        dlo, dhi = max(dlo, _c - 6.0 * _s), min(dhi, _c + 6.0 * _s)
    dpad = 0.05 * (dhi - dlo)
    bins = np.linspace(dlo - dpad, dhi + dpad, nbins)
    ax.hist(np.clip(diff, bins[0], bins[-1]), bins=bins, histtype='step', color='#1f77b4')
    ax.axvline(float(np.median(diff)), color='#d62728', lw=1.3,
               label=f'median {np.median(diff):+.1f} um\nnMAD {nmad(diff):.1f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    # Held to the binned range, so a point outside it piles into the end bin instead of
    # stretching the axis and squashing the core -- which it does on a sample of a few tens.
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel('actual minus predicted v1_dz\n[um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title(f'The difference, {what.lower()}', fontsize=8.5)
    ax.legend(fontsize=8)
    return {'median': float(np.median(diff)), 'nmad': float(nmad(diff)), 'n': int(len(y))}


def figure_fam(pdf, famres):
    """The thermal prediction against the measured focus at the FAM in-focus visit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    famres : `dict`
        Result of `section_fam`.

    Notes
    -----
    The same two panels the science-visit pages carry, on the Full Array Mode (FAM) blocks and
    their Danish 1.2 wavefront retrieval. The coefficients are those fitted on the science visits
    and are applied here unchanged, so this is an out-of-sample test on an independent retrieval.
    """
    if not famres or 'y' not in famres:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _actual_vs_predicted_pair(axes, famres['y'], famres['pred'],
                              'FAM in-focus acquisition visits')
    fig.suptitle(f'FAM blocks, Danish 1.2 processing: {famres["n"]} in-focus visits over '
                 f'{famres["n_nights"]} nights\nopen-loop focus nMAD {famres["nmad_y"]:.1f} um, '
                 f'residual nMAD {famres["nmad"]:.1f} um of equivalent hexapod dz '
                 f'({famres["gain"]:.2f}x, dimensionless)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)


def figure_closed_loop(pdf, closed):
    """Closed-loop focus performance: the measured v-mode-1 deviation alone, and over time.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    closed : `dict`
        Result of `section_closed_loop`.

    Notes
    -----
    Three panels. The first is the per-visit distribution over the 1st to 99th percentile, with
    the median and the nMAD -- the median is the standing offset the loop does not remove, and the
    nMAD is how tightly focus is actually held. The second and third carry the per-night median
    and per-night nMAD against Modified Julian Date, which is what says whether that performance
    is steady over the season or drifting.

    This is the only page in the document that does **not** involve the commanded Trim or any
    thermal telemetry. It is the closed loop's own performance, against which an open-loop
    prediction has to be judged.
    """
    if not closed:
        return
    a = closed['dz']
    nt = closed['nightly']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    # The 2% outside the range is dropped rather than clipped into the end bins: a pile-up spike
    # at each edge is an artefact of the clip, not a feature of the distribution, and this page is
    # about the shape of the bulk. The median and the nMAD in the legend are over every visit.
    bins = np.linspace(closed['p1'], closed['p99'], 80)
    ax.hist(a[(a >= bins[0]) & (a <= bins[-1])], bins=bins, histtype='step', color='#1f77b4')
    ax.axvline(closed['median'], color='#d62728', lw=1.3,
               label=f'median {closed["median"]:+.2f} um\nnMAD {closed["nmad"]:.2f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_xlabel('measured focus deviation\n[um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title(f'Closed-loop focus residual, {closed["n"]} visits\n'
                 f'1st to 99th percentile, {closed["p1"]:+.0f} to {closed["p99"]:+.0f} um',
                 fontsize=9)
    ax.legend(fontsize=8)

    if 'obs_start_mjd' not in nt.columns:
        for ax in axes[1:]:
            ax.axis('off')
    else:
        ax = axes[1]
        ax.plot(nt['obs_start_mjd'], nt['median'], 'o', ms=4, color='#1f77b4')
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axhline(float(nt['median'].median()), color='#d62728', lw=1.2,
                   label=f'median over nights {nt["median"].median():+.2f} um\n'
                         f'nMAD over nights '
                         f'{nmad(nt["median"].to_numpy(float)):.2f} um')
        ax.set_xlabel('start-of-night Modified Julian Date [d]')
        ax.set_ylabel('night median focus deviation\n[um of equivalent hexapod dz]')
        ax.set_title(f'The standing offset, night by night, {len(nt)} nights', fontsize=9)
        ax.legend(fontsize=7.5)

        ax = axes[2]
        ax.plot(nt['obs_start_mjd'], nt['nmad'], 'o', ms=4, color='#7f4fa8')
        ax.axhline(float(nt['nmad'].median()), color='#d62728', lw=1.2,
                   label=f'median over nights {nt["nmad"].median():.2f} um')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('start-of-night Modified Julian Date [d]')
        ax.set_ylabel('night nMAD of focus deviation\n[um of equivalent hexapod dz]')
        ax.set_title('How tightly focus was held, night by night', fontsize=9)
        ax.legend(fontsize=7.5)

    fig.suptitle('Closed-loop performance: the measured v-mode-1 deviation alone, with no '
                 'commanded term and no thermal model', fontsize=10.5, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def figure_filter_lut(pdf, filt):
    """How well the filter look-up table works: focus steps across a filter change.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    filt : `dict`
        Result of `section_filter_lut`.

    Notes
    -----
    Three panels. The first is the signed median step for every ordered from-band to-band pair as
    a heat map, which is where the diagnosis is made: a genuine filter look-up table (LUT) error
    reverses sign when the transition is taken the other way, so the matrix is **antisymmetric**
    about its diagonal. A pair whose two legs carry the same sign is not a filter offset but a
    focus drift that happens to straddle the filter change.

    The second compares the signed step across a filter change with the step between consecutive
    same-band visits, which is the floor set by everything else that moves focus between two
    exposures. The third gives the per-band open-loop focus, which is the standing offset each
    filter sits at rather than the step on entering it.
    """
    if not filt:
        return
    pairs, anti, per_band = filt['pairs'], filt['antisym'], filt['per_band']
    bands = list(per_band['band'])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    M = np.full((len(bands), len(bands)), np.nan)
    for _, r in pairs.iterrows():
        if r['from'] in bands and r['to'] in bands and int(r['n']) >= MIN_TRANSITION_N:
            M[bands.index(r['from']), bands.index(r['to'])] = r['median']
    # The colour scale is set by the 90th percentile of |median step| rather than its maximum, so
    # one strong pair does not flatten every other cell to white. Cells beyond it saturate, and
    # the number printed in each cell is the value regardless.
    _a = np.abs(M[np.isfinite(M)])
    vmax = float(np.percentile(_a, 90)) if len(_a) else 1.0
    im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(bands)), bands)
    ax.set_yticks(range(len(bands)), bands)
    for i in range(len(bands)):
        for j in range(len(bands)):
            if np.isfinite(M[i, j]):
                # White on a saturated cell, dark on a pale one, so no value is lost in the
                # colour the scale assigned it.
                shade = 'white' if abs(M[i, j]) > 0.6 * vmax else '0.1'
                ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center', fontsize=7.5,
                        color=shade)
    ax.set_xlabel('to band')
    ax.set_ylabel('from band')
    ax.set_title(f'Signed median step on changing filter\n[um of equivalent hexapod dz]; blank '
                 f'cells carry fewer than {MIN_TRANSITION_N} transitions', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046,
                 label='median step [um of equiv hexapod dz]')

    ax = axes[1]
    lo, hi = np.percentile(np.r_[filt['change'], filt['same']], [0.5, 99.5])
    bins = np.linspace(lo, hi, 70)
    ax.hist(np.clip(filt['same'], lo, hi), bins=bins, histtype='step', color='0.4', density=True,
            label=f'same band, n {filt["n_same"]}\nmedian {filt["same_median"]:+.2f} um, '
                  f'nMAD {filt["same_nmad"]:.1f} um')
    ax.hist(np.clip(filt['change'], lo, hi), bins=bins, histtype='step', color='#d62728',
            density=True,
            label=f'filter change, n {filt["n_change"]}\nmedian '
                  f'{filt["change_median"]:+.2f} um, nMAD {filt["change_nmad"]:.1f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('signed step between consecutive visits\n[um of equivalent hexapod dz]')
    ax.set_ylabel('visit pairs, normalized')
    ax.set_title(f'A filter change costs {filt["change_nmad"] / filt["same_nmad"]:.2f}x the '
                 f'same-band scatter\n(dimensionless, band-change nMAD over same-band nMAD)',
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    x = np.arange(len(per_band))
    ax.errorbar(x, per_band['median'], yerr=per_band['nmad'], fmt='o', ms=6, lw=1.2,
                capsize=4, color='#1f77b4')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xticks(x, [f'{b}\nn {int(n)}' for b, n in zip(per_band['band'], per_band['n'])])
    ax.set_ylabel('open-loop focus [um of equivalent hexapod dz]')
    ax.set_xlabel('band')
    ax.set_title('Where each filter sits in focus\nmedian, error bars are the nMAD within the '
                 'band', fontsize=9)

    _sums = ('; pair sums [um]: '
             + ', '.join(f'{r["a"]}{r["b"]} {r["sum"]:+.1f}' for _, r in anti.iterrows())
             if len(anti) else '')
    fig.suptitle('The filter look-up table: an exact LUT would move focus by zero on a filter '
                 'change\nan antisymmetric cell pair is a real filter offset, two cells of one '
                 f'sign are a focus drift{_sums}', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)


def figure_outlier_nights(pdf, outl):
    """Visits beyond 3 nMAD of the fitted model, counted per night against date.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    outl : `dict`
        Result of `section_outlier_nights`.

    Notes
    -----
    Two panels, the count and the fraction, both against Modified Julian Date. The count says
    where the model failed; the fraction says whether it failed on the whole night or on a few
    visits of it. A night at a fraction near 1 is one the model got wrong throughout, which is a
    different failure from a night that merely carried many visits.

    The threshold is a single global 3 nMAD of the residual rather than a per-night one, so a bad
    night registers as a high count instead of being renormalized away.
    """
    if not outl:
        return
    nt = outl['nightly']
    if 'obs_start_mjd' not in nt.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(nt['obs_start_mjd'], nt['n_outlier'], 'o', ms=4, color='#d62728')
    ax.set_xlabel('start-of-night Modified Julian Date [d]')
    ax.set_ylabel('visits beyond 3 nMAD [exposures]')
    ax.set_title(f'Outlier count per night, {len(nt)} nights\nthreshold '
                 f'{outl["threshold_um"]:.1f} um of equivalent hexapod dz', fontsize=9)

    ax = axes[1]
    ax.plot(nt['obs_start_mjd'], 100 * nt['frac_outlier'], 'o', ms=4, color='#1f77b4')
    ax.axhline(100 * outl['frac_all'], color='#d62728', lw=1.2,
               label=f'whole sample {100 * outl["frac_all"]:.2f}%')
    ax.axhline(0.27, color='#2ca02c', lw=1.2, ls='--',
               label='Gaussian expectation 0.27%')
    ax.set_xlabel('start-of-night Modified Julian Date [d]')
    ax.set_ylabel('fraction of the night beyond 3 nMAD [percent]')
    ax.set_title('What fraction of each night the model got wrong', fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle(f'Where the thermal model fails: visits beyond '
                 f'{OUTLIER_VISIT_Z:.0f} nMAD = {outl["threshold_um"]:.1f} um of equivalent '
                 f'hexapod dz, night by night', fontsize=10.5, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def figure_t539_first_visit(pdf, t539res):
    """Predicted against actual v1_dz at the first visit after the initial alignment block.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    t539res : `dict`
        Result of `section_t539`, whose ``table`` carries one row per night.

    Notes
    -----
    The quantity is the v-mode-1 amplitude of the commanded hexapod pair that the initial
    alignment block ``BLOCK-T539`` settled on, converted to µm of equivalent hexapod dz, against
    the same quantity predicted from the thermal telemetry at the run's first visit. That is the
    operationally decisive number: it is how much focus an observer would have been wrong by had
    the thermal prediction been applied open loop instead of running the alignment block.

    The alignment block converges without reference to the thermal telemetry, so the two sides are
    independent measurements of the focus the telescope needed.
    """
    if not t539res or 'table' not in t539res:
        return
    t = t539res['table']
    need = ('v1_actual', 'v1_pred')
    if any(c not in t.columns for c in need):
        return
    conv = L.v1_per_um_dz_value(verbose=False)
    y = t['v1_actual'].to_numpy(float) / conv
    pred = t['v1_pred'].to_numpy(float) / conv
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _actual_vs_predicted_pair(axes, y, pred, 'First visit after initial alignment')
    fig.suptitle(f'Prediction quality from the first visit of the night: '
                 f'{len(t)} BLOCK-T539 nights, day_obs {int(t.day_obs.min())} to '
                 f'{int(t.day_obs.max())}\nactual is the v-mode-1 Trim the alignment block '
                 f'settled on; predicted is from the thermal telemetry at the run\'s first visit',
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)


def figure_visits_per_night(pdf, sci, samp):
    """Visits per night per band over the season, and the per-band totals.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    sci : `pandas.DataFrame`
        Science table, as loaded.
    samp : `dict`
        Result of `section_sample`, whose ``per_night_band`` frame holds the counts.

    Notes
    -----
    What the sample is made of, which the opening page states only as a total. One bar per
    ``day_obs``, bands stacked in wavelength order and coloured by `BAND_COLOUR`. Nights the
    survey did not observe are gaps rather than being closed up, so the bar chart reads as a
    calendar.

    The band mix is strongly uneven — i and z carry most of the sample — which is why the model is
    fitted band-independent and checked per band rather than fitted per band: the thin bands would
    otherwise be fitted on far too little.
    """
    pnb = samp.get('per_night_band')
    if pnb is None or not len(pnb):
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), width_ratios=(3.0, 1.0))

    ax = axes[0]
    bottom = np.zeros(len(pnb), float)
    for b in pnb.columns:
        v = pnb[b].to_numpy(float)
        ax.bar(pnb.index, v, bottom=bottom, width=1.0, color=BAND_COLOUR.get(b, '0.5'),
               label=f'{b}  {int(np.nansum(v))}', linewidth=0)
        bottom = bottom + np.nan_to_num(v)
    ax.set_xlabel('day_obs')
    ax.set_ylabel('science visits in the sample [visits per night]')
    # Outside the axes: the busiest nights reach the top of the frame, so an inset legend would
    # sit on top of the data.
    ax.legend(fontsize=7.5, title='band, total visits', title_fontsize=7.5, ncol=6,
              loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_horizontalalignment('right')

    ax = axes[1]
    tot = pnb.sum(axis=0)
    ypos = np.arange(len(tot))[::-1]
    ax.barh(ypos, tot.to_numpy(float),
            color=[BAND_COLOUR.get(b, '0.5') for b in tot.index])
    for yy, b in zip(ypos, tot.index):
        ax.text(tot[b], yy, f'  {int(tot[b])}', va='center', fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(list(tot.index))
    ax.set_xlim(0, 1.18 * float(tot.max()))
    ax.set_xlabel('science visits in the sample [visits]')
    ax.set_ylabel('band')
    ax.set_title(f'Per-band totals, {int(tot.sum())} visits', fontsize=9.5)

    _busy = bottom[bottom > 0]
    fig.suptitle(f'The sample night by night: {len(sci)} science visits over '
                 f'{sci["day_obs"].nunique()} nights, day_obs {int(sci.day_obs.min())} to '
                 f'{int(sci.day_obs.max())}\nnight median {int(np.median(_busy))} visits, '
                 f'busiest {int(_busy.max())}, one bar per day_obs',
                 fontsize=10, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)


def figure_measured_v1(pdf, closed):
    """The focus actually achieved in closed loop: the measured v-mode 1, over its full range.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    closed : `dict`
        Result of `section_closed_loop`.

    Notes
    -----
    The measured term alone — no commanded Trim, no thermal model — so this is the focus error the
    closed loop actually left on the telescope, visit by visit.

    Two views of one distribution. The left panel is the **full** range on a logarithmic count
    axis, which is the only way the tails are visible at all: they reach several hundred microns
    either side of a core a few tens of microns wide. The right is the same distribution linear
    over its 0.1st to 99.9th percentile, which is the shape of the core. The closed-loop page
    elsewhere in this document crops to the 1st to 99th percentile, so the extent shown here is
    deliberately wider than anything else in the report.
    """
    if not closed:
        return
    a = np.asarray(closed['dz'], float)
    a = a[np.isfinite(a)]
    if not len(a):
        return
    med, nm = closed['median'], closed['nmad']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    bins = np.linspace(a.min(), a.max(), 160)
    ax.hist(a, bins=bins, histtype='step', color='#1f77b4')
    ax.set_yscale('log')
    ax.axvline(med, color='#d62728', lw=1.3,
               label=f'median {med:+.2f} um\nnMAD {nm:.2f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_xlabel('measured v1_dz, achieved in closed loop\n[um of equivalent hexapod dz]')
    ax.set_ylabel('visits per bin')
    ax.set_title(f'Full range, {len(a)} visits\n{a.min():+.0f} to {a.max():+.0f} um of '
                 f'equivalent hexapod dz', fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    lo, hi = closed['p_lo'], closed['p_hi']
    core = a[(a >= lo) & (a <= hi)]
    ax.hist(core, bins=np.linspace(lo, hi, 100), histtype='step', color='#1f77b4')
    ax.axvline(med, color='#d62728', lw=1.3, label=f'median {med:+.2f} um')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_xlabel('measured v1_dz, achieved in closed loop\n[um of equivalent hexapod dz]')
    ax.set_ylabel('visits per bin')
    ax.set_title(f'The core, 0.1st to 99.9th percentile\n{lo:+.0f} to {hi:+.0f} um, '
                 f'{len(core)} of {len(a)} visits', fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle('Closed-loop focus achieved: the measured v-mode 1 alone, with no commanded '
                 'Trim and no thermal model', fontsize=10.5, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    pdf.savefig(fig)
    plt.close(fig)


def figure_t539_duration(pdf, t539res):
    """How long the initial alignment run takes, night by night.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    t539res : `dict`
        Result of `section_t539`, whose ``duration`` holds the per-night span.

    Notes
    -----
    This is the delay inside the run-first against run-last comparison: the prediction is made
    from the telemetry at the first exposure of the block and the Trim is read at the last, so
    this histogram is how far apart those two epochs are in time.

    Measured as first exposure start to last exposure end. The run length is not fixed — from a
    couple of exposures to 45 — so the spread is wide and the second panel carries it against
    date, where a change in how the block is run over the season would show.
    """
    du = (t539res or {}).get('duration')
    if not du:
        return
    m, nights = du['minutes'], du['day_obs']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    hi = float(np.percentile(m, 98))
    bins = np.linspace(0.0, max(hi, du['median'] * 2.0), 40)
    ax.hist(np.clip(m, bins[0], bins[-1]), bins=bins, histtype='step', color='#1f77b4')
    ax.axvline(du['median'], color='#d62728', lw=1.3,
               label=f'median {du["median"]:.1f} min\n16th to 84th {du["p16"]:.1f} to '
                     f'{du["p84"]:.1f} min')
    ax.set_xlabel('initial alignment run duration\n[min, first exposure start to last '
                  'exposure end]')
    ax.set_ylabel('nights')
    ax.set_title(f'How long the alignment block runs, {du["n"]} nights\nrange '
                 f'{du["min"]:.1f} to {du["max"]:.1f} min; the last bin holds the overflow',
                 fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    dates = pd.to_datetime(nights.astype(str), format='%Y%m%d')
    ax.plot(dates, m, 'o', ms=4, color='#1f77b4')
    ax.axhline(du['median'], color='#d62728', lw=1.2,
               label=f'median {du["median"]:.1f} min')
    ax.set_yscale('log')
    ax.set_xlabel('day_obs')
    ax.set_ylabel('run duration [min]')
    ax.set_title('Run duration night by night, logarithmic', fontsize=9)
    ax.legend(fontsize=8)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_horizontalalignment('right')

    fig.suptitle('The delay inside the initial-alignment comparison: the span the block covers '
                 'between its first and last exposure', fontsize=10, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)


def figure_t539_sci1(pdf, t539res):
    """The prediction against the first science visit after the alignment run, near-simultaneous.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    t539res : `dict`
        Result of `section_t539`, whose ``sci1`` holds the comparison.

    Notes
    -----
    A better-matched comparison than the page before: both the prediction and the measurement are
    taken at the **same visit**, the first science exposure after the block ends. The thermal
    drift across the alignment run — the leading systematic in the run-first against run-last
    comparison — is therefore absent by construction rather than suppressed by a cut, and every
    night with a following science visit is kept.

    This is still the start-of-night question, and the one the page before answers badly. The
    telescope is at its least thermally settled then, so this is where an open-loop focus
    prediction has to work if it is to replace the alignment block.

    Outlier nights are labelled with their ``day_obs`` on the scatter, the same convention as
    `figure_sample`.
    """
    s1 = (t539res or {}).get('sci1')
    if not s1:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # A 2nd-to-98th clip rather than the usual 0.5-to-99.5: the sample is a few tens of nights,
    # so the standard clip cannot exclude a single far outlier and one night would otherwise set
    # the scale of both panels and squash the rest into a corner.
    _actual_vs_predicted_pair(axes, s1['actual'], s1['pred'],
                              'First science visit after the run',
                              nbins=36, clip_pct=(2.0, 98.0))

    # Labelled at the point, or pinned just inside the frame with an arrow when the clip above
    # has put the night off-axis, so a labelled outlier is never invisible.
    ax = axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for _, r in s1['outliers'].iterrows():
        inside = x0 <= r.actual <= x1 and y0 <= r.pred <= y1
        if inside:
            ax.plot([r.actual], [r.pred], 'o', ms=8, mfc='none', color='#d62728')
            ax.annotate(f'{int(r.day_obs)}', (r.actual, r.pred), fontsize=6, color='#d62728',
                        xytext=(5, 0), textcoords='offset points', va='center')
        else:
            ax.annotate(f'{int(r.day_obs)}\nactual {r.actual:+.0f} um, off scale',
                        (min(max(r.actual, x0), x1), min(max(r.pred, y0), y1)),
                        fontsize=6, color='#d62728', ha='right', va='bottom',
                        xytext=(-6, 6), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))

    fig.suptitle(f'Simultaneous test: prediction and measurement both at the first science visit '
                 f'after the block\n{s1["n"]} nights, no time gap between the two sides; '
                 f'{len(s1["outliers"])} night{"" if len(s1["outliers"]) == 1 else "s"} beyond '
                 f'{OUTLIER_NIGHT_Z:.0f} nMAD labelled', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_intranight(pdf, intra):
    """The residual against time since sunset: does the prediction decay over a night?

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    intra : `dict`
        Result of `section_intranight`.

    Notes
    -----
    Three panels. The scatter carries every visit with the binned median over it, so a shape is
    visible against the spread it has to beat. The two lower panels separate the two ways the
    prediction could be time-dependent: a drift in the binned **median** is a bias the model
    could absorb with a time term, while a change in the binned **nMAD** is a change in how
    predictable focus is at all, which a time term cannot fix.

    The model has no time-of-night input, so any structure here is a statement about the
    telemetry, not about the fit.

    The early-against-late split from `section_intranight` is drawn over the two lower panels and
    stated in the page title, because the effect is a step in the first few hours rather than a
    drift and the fitted line alone would understate it.
    """
    if not intra:
        return
    bins = intra['bins']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))

    ax = axes[0][0]
    h, r = intra['hours'], intra['resid']
    ax.plot(h, r, '.', ms=1.0, color='#1f77b4', alpha=0.18, rasterized=True)
    ax.plot(bins['centre'], bins['median'], 'o-', ms=5, color='#d62728', lw=1.4,
            label='binned median')
    ax.axhline(0, color='0.4', lw=0.8)
    rlo, rhi = np.percentile(r, (0.5, 99.5))
    ax.set_ylim(rlo, rhi)
    ax.set_xlim(intra['p1'], intra['p99'])
    ax.set_xlabel(f'time since {TWILIGHT_REF_ALT_DEG:.0f} deg twilight [h]')
    ax.set_ylabel('actual minus predicted v1_dz\n[um of equivalent hexapod dz]')
    ax.set_title(f'Every visit, n {intra["n"]} (0.5th to 99.5th percentile shown)', fontsize=8.5)
    ax.legend(fontsize=7.5)

    ax = axes[0][1]
    ax.bar(bins['centre'], bins['n'], width=0.85 * (bins['hi'] - bins['lo']),
           color='#7f7f7f', edgecolor='none')
    ax.set_xlabel(f'time since {TWILIGHT_REF_ALT_DEG:.0f} deg twilight [h]')
    ax.set_ylabel('visits')
    ax.set_title('Visits per bin (equal-count bins, so this shows the bin widths)',
                 fontsize=8.5)

    for ax, key, trend, label, colour in (
            (axes[1][0], 'median', 'median_trend', 'median residual', '#d62728'),
            (axes[1][1], 'nmad', 'nmad_trend', 'residual nMAD', '#2ca02c')):
        ax.errorbar(bins['centre'], bins[key],
                    xerr=[bins['centre'] - bins['lo'], bins['hi'] - bins['centre']],
                    fmt='o', ms=5, color=colour, lw=1.0, capsize=0)
        t = intra[trend]
        if t:
            xs = np.array([bins['centre'].min(), bins['centre'].max()])
            ax.plot(xs, t['intercept'] + t['slope'] * xs, '-', color='0.3', lw=1.2,
                    label=f'Huber slope {t["slope"]:+.2f} +/- {t["slope_err"]:.2f} um per h\n'
                          f'({abs(t["slope"]) / t["slope_err"]:.1f} standard errors), '
                          f'Spearman rho {t["spearman_rho"]:+.2f}')
            ax.legend(fontsize=7.5)
        if key == 'median':
            ax.axhline(0, color='0.6', lw=0.8)
        sp = intra.get('split') or {}
        if sp.get('early') and sp.get('late'):
            ax.axvline(sp['split_h'], color='#9467bd', lw=1.0, ls='--')
            for part, ha in (('early', 'right'), ('late', 'left')):
                ax.axhline(sp[part][key], color='#9467bd', lw=0.9, ls=':')
                ax.text(sp['split_h'] + (-0.3 if ha == 'right' else 0.3), sp[part][key],
                        f'{sp[part][key]:+.1f}' if key == 'median' else f'{sp[part][key]:.1f}',
                        color='#9467bd', fontsize=7, ha=ha, va='bottom')
        ax.set_xlabel(f'time since {TWILIGHT_REF_ALT_DEG:.0f} deg twilight [h]')
        ax.set_ylabel(f'{label} [um of equivalent hexapod dz]')
        ax.set_title(f'Binned {label}; horizontal bars are the bin widths, dashed line the '
                     f'early/late split', fontsize=8.5)

    sp = intra.get('split') or {}
    extra = ''
    if sp.get('early') and sp.get('late'):
        extra = (f'\nthe effect is a step, not a drift: residual nMAD '
                 f'{sp["early"]["nmad"]:.1f} um before {sp["split_h"]:.0f} h after sunset '
                 f'against {sp["late"]["nmad"]:.1f} um after it '
                 f'({sp["nmad_ratio"]:.2f}x, dimensionless), with a '
                 f'{sp["early"]["median"]:+.1f} um median bias early')
    fig.suptitle('Intra-night behaviour of the prediction: residual against time since sunset\n'
                 'the model has no time-of-night term, so structure here is a property of the '
                 'telemetry' + extra, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    pdf.savefig(fig)
    plt.close(fig)


def figure_t539_long_runs(pdf, t539res):
    """The shortened-epoch nights whose alignment block was not the nominal length.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    t539res : `dict`
        Result of `section_t539`, whose ``long_runs`` holds the table.

    Notes
    -----
    A table rather than a plot, because the question is which specific nights misbehaved and what
    they have in common. The duration, the exposure count and the time after sunset are side by
    side so the two candidate explanations separate: more exposures means the block iterated
    further to converge, while the nominal count spread over longer means the delay was between
    exposures and so is not the alignment.
    """
    lr = (t539res or {}).get('long_runs')
    if not lr:
        return
    lo, hi = lr['nominal']
    t = lr['table']
    lines = [f'Nights on or after day_obs {lr["since"]}, when the block was shortened, whose '
             f'duration is outside {lo:.0f} to {hi:.0f} min.',
             f'{len(t)} of {lr["n_epoch"]} nights in the epoch: {lr["n_short"]} under '
             f'{lo:.0f} min, {lr["n_long"]} over {hi:.0f} min, {lr["n_nominal"]} nominal.',
             '',
             'day_obs    duration   visits   after 0 deg twilight   per visit',
             '              [min]              [min]                  [min]',
             '-' * 68]
    for _, r in t.iterrows():
        lines.append(f'{int(r.day_obs)}   {r.run_duration_min:8.1f}   {int(r.n_run):5d}   '
                     f'{r.min_after_twilight:17.1f}   {r.min_per_visit:9.2f}')
    nom = t[t['run_duration_min'] > hi]
    if len(nom):
        lines += ['', f'Over-length nights only, n {len(nom)}: median {nom["n_run"].median():.1f} '
                      f'exposures, median {nom["min_per_visit"].median():.2f} min per exposure, '
                      f'median {nom["min_after_twilight"].median():+.1f} min after twilight.']
    _text_page(pdf, f'Initial alignment block duration outside {lo:.0f} to {hi:.0f} min, '
                    f'day_obs >= {lr["since"]}', lines)


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

    The left column carries the points and a zero line only. A median line and a fitted trend
    against date sat there and were removed: the distribution in the right-hand column already
    states the median and the nMAD, and a trend against date over one season is a seasonal
    temperature cycle read through the correction rather than a drift of the telescope.
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
        ax.set_xlabel('start-of-night Modified Julian Date [d]')
        ax.set_ylabel(f'{dofres["labels"][col]}\nto command as Trim [{unit}]')
        ax.set_title(f'{dofres["labels"][col]} at the start of each night, '
                     f'{len(s)} nights', fontsize=9.5)

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
    F.per_band_fit(sci, features, model=args.model)   # printed, not shown

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
    F.residual_tail(sci, truss_resid)
    print('\nabout the full five-feature band-independent fit:')
    F.residual_tail(sci, full['resid'])

    print('\n=== 7. closed-loop focus performance, the measured v1 alone ===')
    closed = section_closed_loop(sci)

    print('\n=== 7b. intra-night variation of the prediction, against time since twilight ===')
    intra = section_intranight(sci, full['resid'])

    print('\n=== 8. the filter look-up table: focus steps across a filter change ===')
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
    F.band_change_step(d, 'y', label='open-loop focus')
    F.band_change_step(d, 'resid_per_band', label='per-band models')
    F.band_change_step(d, 'resid', label='shared thermal model')
    filt = section_filter_lut(sci)

    print('\n=== 8b. outliers beyond 3 nMAD, per night ===')
    outl = section_outlier_nights(sci, full['resid'])

    print('\n=== 9. FAM blocks: the prediction at the in-focus acquisition visit ===')
    famres = section_fam(fam, full, features)

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
        # --------------------------------------------------------------- page 1: the study
        # Aaron's own prose, verbatim in register and content: Goal, Method, Data Sample and
        # nothing else. The counts are computed rather than typed so the page cannot go stale,
        # but the sentences are his.
        _conv_row = next((r for r in conv
                          if r['dof_set'] == 'all_50' and r['n_modes'] == 34), conv[0])
        _text_page(pdf, 'Thermal Focus Study', [
            'GOAL',
            '',
            '  Determine DoF Trim values to set the focus v-mode (v1) based on thermal telemetry.',
            '',
            'METHOD',
            '',
            '  The first v-mode (v1) is the optical focus, so we reconstruct the open loop v1 from',
            '  the Trim minus Deviation of the relevant DoF, which are Camera and M2 dz and M1M3',
            '  bending B3 and M2 bending B5. Deviation is taken from the ConsDB Zernike wavefront,',
            '  since that is the only wavefront retrieval which is available for all science',
            '  visits. Since v1 is unit-less we convert to a focus equivalent v1_dz in microns,',
            '  assuming equal contributions from the Camera and M2 hexapod dz, given by',
            f'  {_conv_row["um_per_v1_shared"]:.1f} microns of Hexapod dz per unit v1.',
            '',
            'DATA SAMPLE',
            '',
            f'  {len(sci)} science visits over {sci["day_obs"].nunique()} nights, day_obs '
            f'{int(sci.day_obs.min())} to {int(sci.day_obs.max())}, except for',
            f'  {funnel["n_lut_nights"]} nights with different LUT, 217 visits on 20251118 and '
            f'20251119 with mean truss',
            f'  temperature between 23-25C, and {funnel["n_no_features"]} visits missing one of '
            f'the {len(features)} telemetry values.',
        ])
        figure_before(pdf, sci, trussonly)

        if truss_all is not None and len(truss_all):
            figure_truss_all(pdf, truss_all, sci)
        else:
            print('  thermal_focus_truss_all.parquet is absent; the database-wide truss page is '
                  'skipped. Build it with\n    python code/run_thermal_focus.py --only-truss-all')
        figure_sample(pdf, sci, samp)
        figure_visits_per_night(pdf, sci, samp)

        figure_terms_individual(pdf, terms)
        figure_terms_cumulative(pdf, terms)

        # ------------------------------------ page 8: the term summary, naming no winner
        _ind, _cum = terms['individual'], terms['cumulative']
        _text_page(pdf, 'Evaluate predictive power of thermal telemetry', [
            f'All models below are fitted and scored on the same {terms["n_common"]} visits of '
            f'{terms["n_all"]}, where the',
            'baseline and every candidate are finite.',
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
            f'  Pearson r and Spearman rho are of the predicted against the measured open-loop '
            f'focus, n {terms["n_common"]}.',
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
            f'The model carried by every page that follows is {"+".join(L.DELIVERABLE_GROUPS)}: '
            f'{", ".join(features[:3])},',
            f'{", ".join(features[3:])}.',
        ])

        _text_page(pdf, 'The deliverable thermal model', [
            'The fitted equation, focus error in um of equivalent hexapod dz:',
            '',
            f'  = {full["intercept"]:+.2f}',
            *[f'    {c:+10.2f} * {f:32s} [per {F.FEATURE_UNITS.get(f, "?")}]'
              for f, c in zip(full['features'], full['coef'])],
            '',
            *(['The standalone calculator trim_calculator.py stores these coefficients to two',
               'decimals in trim_coefficients.yaml, so it can be copied to a summit machine and',
               'read by eye. Against this',
               f'fit over {calc["n"]} visits: max |difference| '
               f'{calc["max_abs_diff_um"]:.4f} um of equivalent hexapod dz,',
               f'and its {calc["n_worked_cases"]} worked test cases agree with their stated '
               f'values to {calc["worked_max_abs_diff_um"]:.4f} um.']
              if calc else []),
        ])
        figure_model(pdf, sci, full)

        if r2res['available']:
            figure_r2(pdf, sci, r2res, features)

        figure_fam(pdf, famres)

        _text_page(pdf, 'DoF Trim from thermal focus prediction', [
            'Use the standard AOS singular-value-decomposition analysis with the predicted value',
            'of v1 and taking all other v-modes to be zero, we find values of the relevant DoF',
            'using 50/34 with the method StateEstimator.get_dofs_from_vmodes. The formula',
            'for each DoF is:',
            '',
            *[f'  {dofres["labels"][c]:24s} = {dofres["unit"][c]:+14.4f} um per unit v1'
              for c in dofres['names']],
        ])
        figure_dof(pdf, dofres)
        figure_dof_start(pdf, dofres)

        figure_closed_loop(pdf, closed)
        figure_measured_v1(pdf, closed)
        figure_intranight(pdf, intra)
        figure_filter_lut(pdf, filt)
        figure_outlier_nights(pdf, outl)

        # ------------------------------------- group 4: against what the observatory actually did
        if t539res:
            _t = t539res['table']
            _conv = L.v1_per_um_dz_value(verbose=False)
            # Both v1 columns are dimensionless amplitudes; divide by the conversion to put them
            # into um of equivalent hexapod dz, which is the unit the table's header declares.
            _text_page(pdf, 'The nights the alignment block settled far from the rest', [
                f'The {len(t539res["outliers"])} nights beyond '
                f'{OUTLIER_NIGHT_Z:.0f} nMAD of the median camera hexapod dz Trim the initial',
                f'alignment block BLOCK-T539 settled on, over {len(_t)} nights with that block. '
                f'Over those nights the',
                f'camera hexapod dz Trim has median '
                f'{t539res["dof5_last_center_um"]:+.2f} um and nMAD '
                f'{t539res["dof5_last_nmad_um"]:.2f} um; z is dimensionless,',
                'the night\'s deviation over that nMAD.',
                '',
                '                 seq_num        truss temp [deg C]       v1_dz [um equiv hex dz]'
                '      Trim after [um]',
                '  day_obs    first    after    first    after      predicted     actual'
                '      Camera dz    M2 dz     z',
                *([f'  {int(r.day_obs)}  {_fmt_i(r.get("seq_num_first")):>7s}  '
                   f'{_fmt_i(r.get("seq_num_last")):>7s}  '
                   f'{_fmt_f(r.get("truss_temp_mean_c_first"), 2):>7s}  '
                   f'{_fmt_f(r.get("truss_temp_mean_c_last"), 2):>7s}  '
                   f'{_fmt_f(r.get("v1_pred") / _conv if np.isfinite(r.get("v1_pred", np.nan)) else np.nan, 1):>13s}  '
                   f'{_fmt_f(r.get("v1_actual") / _conv if np.isfinite(r.get("v1_actual", np.nan)) else np.nan, 1):>9s}  '
                   f'{_fmt_f(r.get("dof5_last"), 1):>13s}  '
                   f'{_fmt_f(r.get("dof0_last"), 1):>9s}  '
                   f'{_fmt_f(r.get("dof5_last_z"), 1):>5s}'
                   for _, r in t539res['outliers'].sort_values('day_obs').iterrows()]
                  if len(t539res['outliers']) else
                  [f'  none: no night lies beyond {OUTLIER_NIGHT_Z:.0f} nMAD on this axis.']),
                '',
                '  seq_num first    the first exposure of the BLOCK-T539 run, which the thermal '
                'prediction is made from',
                '  seq_num after    the last exposure of that run, after the alignment has '
                'converged',
                '  truss temp       mean TMA truss temperature at those two exposures [deg C]; '
                'the pair says how far',
                '                   the thermal state moved while the block ran',
                '  v1_dz predicted  the thermal prediction, v-mode 1 converted to um of '
                'equivalent hexapod dz',
                '  v1_dz actual     the v-mode-1 amplitude of the Trim the block settled on, in '
                'the same unit',
                '  Trim after       the commanded Camera and M2 hexapod dz Trim at the run\'s '
                'end [um]',
                '',
                'A night that is an outlier here is one on which the alignment block asked for an',
                'unusual amount of focus, which is not by itself a failure of the thermal '
                'prediction --',
                'the predicted v1_dz on the same line says whether the model saw it coming. The '
                'alignment',
                'is free to put focus on either hexapod and does, so the Camera and M2 columns '
                'split in a way',
                'that carries no optical meaning; the v1_dz columns are the physical comparison.',
            ])
            figure_t539_duration(pdf, t539res)
            figure_t539_long_runs(pdf, t539res)
            figure_t539_sci1(pdf, t539res)
            figure_t539_first_visit(pdf, t539res)
            figure_t539(pdf, t539res)
    print(f'\nwrote {pdf_path}')


if __name__ == '__main__':
    main()
