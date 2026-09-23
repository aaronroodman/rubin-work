"""The thermal-focus analysis: every fit and cross-check, one PDF, no network.

Reads the cached tables written by ``run_thermal_focus.py`` and produces one document. The
headline is that the telescope's uniform-defocus error is predictable from five thermal
channels — the Telescope Mount Assembly (TMA) truss temperature and the four M1M3 bulk
thermal gradients — with one band-independent Huber robust linear model.

Sections, in the order they appear in the PDF:

1. The response and the sample — the selection funnel and per-band coverage.
2. The thermal model — the deliverable fit, with whole nights held out.
3. Machine-learning legibility — night-grouped against visit-level scoring, model comparison,
   per-fold coefficients and the Full Array Mode (FAM) cross-check of the truss slope.
4. Camera-body temperature — an alternative thermometer, indistinguishable as a regressor.
5. What adds nothing — the channels whose gain does not exceed the across-band scatter.
6. Residual shape — the one-sided positive tail that makes the fits robust rather than least
   squares.
7. Within-night behaviour — per-night elevation slopes and the rising-against-falling
   hysteresis test.
8. Band changes — the step a per-band correction injects, and what the shared model does to it.
9. FAM blocks — within-block focus drift, and whether the thermal correction helps.
10. The conversion table — the v-mode-1 to hexapod dz factor across projection schemes,
    including the 10-degree-of-freedom/1-mode case the online system would use.

Invocation::

    python code/thermal_focus/run_thermal_focus_analysis.py
    python code/thermal_focus/run_thermal_focus_analysis.py --day-obs-range 20251103 20260713
    python code/thermal_focus/run_thermal_focus_analysis.py --no-pdf

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
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

import thermal_focus_fit as F                                     # noqa: E402
import thermal_focus_lib as L                                     # noqa: E402
from common.utils import nmad                                     # noqa: E402

#: Bands in plotting order, bluest first, with a colour each.
BAND_COLOUR = {'u': '#3b4cc0', 'g': '#4fa845', 'r': '#d62728',
               'i': '#8c564b', 'z': '#7f4fa8', 'y': '#bcab2a'}

#: Feature groups tried in the ablation, beyond the deliverable set. Each is scored
#: night-grouped against the deliverable so a gain has to beat the across-band scatter.
ABLATION_GROUPS = (('truss',), ('zgrad',), ('truss', 'zgrad'), ('truss', 'grads'),
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
    """
    sci_path = out_dir / 'thermal_focus.parquet'
    if not sci_path.exists():
        raise SystemExit(f'{sci_path} is absent; build it with\n  python '
                         f'code/thermal_focus/run_thermal_focus.py')
    sci = pd.read_parquet(sci_path)
    features = L.resolve_features(L.DELIVERABLE_GROUPS)

    n_cached = len(sci)
    if day_obs_range:
        sci = sci[(sci.day_obs >= day_obs_range[0]) & (sci.day_obs <= day_obs_range[1])]
    n_span = len(sci)
    sci = sci[sci[features].notna().all(axis=1)].reset_index(drop=True)

    fam_path = out_dir / fam_dir_name / 'thermal_focus_fam.parquet'
    fam = pd.read_parquet(fam_path) if fam_path.exists() else None
    if fam is not None and day_obs_range:
        fam = fam[(fam.day_obs >= day_obs_range[0]) & (fam.day_obs <= day_obs_range[1])]

    if verbose:
        print(f'cached science visits                 : {n_cached}')
        if day_obs_range:
            print(f'  within day_obs {day_obs_range[0]} to {day_obs_range[1]}   : {n_span}')
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
        corrected over uncorrected), ``n_improved`` and ``n_sets``.

    Notes
    -----
    The expected negative result, and the reason it is kept: **the thermal correction makes
    within-block scatter worse**. The coefficients are fitted between nights, where the truss
    temperature moves by degrees; within a block it moves by hundredths of a degree, which is
    telemetry noise, and a coefficient of order +125 µm of equivalent hexapod dz per °C turns
    that noise into a prediction swing larger than the drift being corrected. A between-night
    model is not a within-block model, and this section is where that shows.
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
    sets = F.within_set_scatter(d, set_col, {'y': 'um of equivalent hexapod dz',
                                             'y_corrected': 'um of equivalent hexapod dz',
                                             'pred': 'um of equivalent hexapod dz'},
                                verbose=False)
    ok = sets[['y_p2p', 'y_corrected_p2p']].notna().all(axis=1)
    s = sets[ok]
    out = dict(sets=sets, set_col=set_col, info=info, n_sets=int(len(s)),
               median_p2p_uncorrected=float(s.y_p2p.median()) if len(s) else float('nan'),
               median_p2p_corrected=float(s.y_corrected_p2p.median()) if len(s) else float('nan'),
               median_p2p_prediction=float(s.pred_p2p.median()) if len(s) else float('nan'),
               n_improved=int((s.y_corrected_p2p < s.y_p2p).sum()) if len(s) else 0)
    out['ratio'] = (out['median_p2p_corrected'] / out['median_p2p_uncorrected']
                    if out['median_p2p_uncorrected'] else float('nan'))
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
        if out['ratio'] > 1:
            print('    -> the between-night correction makes within-block scatter WORSE, '
                  'because within a\n       block the thermal telemetry moves by telemetry '
                  'noise, not by a real temperature change')
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
    """Sample coverage: the response per band and the per-night medians against night."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    for b, c in BAND_COLOUR.items():
        v = sci.loc[sci.band == b, 'y'].to_numpy(float)
        if len(v) > 50:
            ax.hist(v, bins=np.linspace(-800, 1200, 80), histtype='step', color=c,
                    label=f'{b} (n {len(v)})')
    ax.set_xlabel('response [um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('Uncorrected response per band')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    pn = samp['per_night']
    ax.plot(np.arange(len(pn)), pn['median'], '.', ms=4, color='#1f77b4')
    ax.set_xlabel('night index, in day_obs order')
    ax.set_ylabel('night median response\n[um of equivalent hexapod dz]')
    ax.set_title(f'Per-night median, {len(pn)} nights')
    ax.axhline(0, color='0.6', lw=0.8)

    ax = axes[1, 0]
    ax.plot(sci['truss_temp_mean_c'], sci['y'], ',', color='0.5', alpha=0.5)
    r = F.huber_line(sci['truss_temp_mean_c'], sci['y'])
    xx = np.linspace(sci['truss_temp_mean_c'].min(), sci['truss_temp_mean_c'].max(), 10)
    ax.plot(xx, r['intercept'] + r['slope'] * xx, '-', color='#d62728', lw=1.6,
            label=f'Huber {r["slope"]:+.1f} um per deg C')
    ax.set_xlabel('TMA truss temperature [deg C]')
    ax.set_ylabel('response [um of equivalent hexapod dz]')
    ax.set_title(f'Truss temperature, Pearson r {r["pearson_r"]:+.3f}, '
                 f'Spearman rho {r["spearman_rho"]:+.3f}')
    ax.set_ylim(-800, 1200)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    pn = samp['per_night']
    ax.plot(pn['truss'], pn['median'], 'o', ms=3.5, color='#2ca02c')
    ax.set_xlabel('night median TMA truss temperature [deg C]')
    ax.set_ylabel('night median response\n[um of equivalent hexapod dz]')
    ax.set_title('Between-night, where the signal lives')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def figure_model(pdf, sci, cv, full):
    """The fit: residual per band, prediction against response, and the coefficient spread."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.plot(cv['pred'], sci['y'], ',', color='0.5', alpha=0.5)
    lo, hi = -800, 1200
    ax.plot([lo, hi], [lo, hi], '-', color='#d62728', lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('out-of-fold prediction [um of equivalent hexapod dz]')
    ax.set_ylabel('response [um of equivalent hexapod dz]')
    ax.set_title(f'Night-grouped, R2 {cv["r2"]:+.3f} (dimensionless)')

    ax = axes[0, 1]
    ax.hist(sci['y'], bins=np.linspace(-800, 1200, 120), histtype='step', color='0.4',
            label=f'uncorrected, nMAD {nmad(sci["y"].to_numpy()):.1f} um')
    ax.hist(cv['resid'], bins=np.linspace(-800, 1200, 120), histtype='step', color='#d62728',
            label=f'residual, nMAD {cv["nmad"]:.1f} um')
    ax.set_xlabel('[um of equivalent hexapod dz]')
    ax.set_ylabel('visits')
    ax.set_title('Before and after the thermal correction')
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    if cv['coefs'] is not None:
        c = cv['coefs']
        pos = np.arange(c.shape[1])
        ax.errorbar(c.mean(axis=0), pos, xerr=c.std(axis=0), fmt='o', ms=4, color='#1f77b4')
        ax.set_yticks(pos)
        ax.set_yticklabels([f.replace('_c_per_m', '').replace('_', ' ')
                            for f in full['features']], fontsize=7.5)
        ax.axvline(0, color='0.6', lw=0.8)
        ax.set_xscale('symlog', linthresh=100)
        ax.set_xlabel('coefficient [um of equivalent hexapod dz per feature unit]')
        ax.set_title(f'Per-fold coefficients, {c.shape[0]} folds')

    ax = axes[1, 1]
    r = np.asarray(cv['resid'], float)
    s = nmad(r)
    for b, col in BAND_COLOUR.items():
        v = r[(sci.band == b).to_numpy()]
        v = v[np.isfinite(v)]
        if len(v) > 50:
            ax.hist(v / s, bins=np.linspace(-8, 8, 120), histtype='step', color=col,
                    density=True, label=b)
    ax.set_yscale('log')
    ax.axvline(3, color='0.6', lw=0.8, ls='--')
    ax.axvline(-3, color='0.6', lw=0.8, ls='--')
    ax.set_xlabel('residual / residual nMAD [dimensionless]')
    ax.set_ylabel('density')
    ax.set_title('The one-sided positive tail, per band')
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default=None,
                    help='directory holding the cached tables; default '
                         'thermal_focus/output/thermal_focus')
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
               else _ROOT / 'thermal_focus' / 'output' / 'thermal_focus')
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

    if args.no_pdf:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / args.pdf_name
    with PdfPages(pdf_path) as pdf:
        _text_page(pdf, 'Thermal focus: the fitted model', [
            f'Sample      {len(sci)} science visits over {sci["day_obs"].nunique()} nights, '
            f'day_obs {int(sci.day_obs.min())} to {int(sci.day_obs.max())}',
            f'Response    (v1_trim + {L.MEASURED_SIGN:+.1f} * v1) / '
            f'{L.v1_per_um_dz_value(verbose=False):.5e}',
            '            [um of equivalent hexapod dz, 0.5 um on each hexapod]',
            f'Uncorrected median {sci["y"].median():+.1f}, '
            f'nMAD {nmad(sci["y"].to_numpy()):.1f} um of equivalent hexapod dz',
            '',
            'The fitted equation, response in um of equivalent hexapod dz:',
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
        figure_sample(pdf, sci, samp)
        figure_model(pdf, sci, cv, full)

        _text_page(pdf, 'Thermal focus: what the model does and does not buy', [
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
                'A between-night model is not a within-block model. Within a block the',
                'thermal telemetry moves by hundredths of a deg C, which is telemetry noise,',
                'and a coefficient of order +125 um of equivalent hexapod dz per deg C turns',
                'that noise into a prediction swing larger than the drift being corrected.',
            ]
        _text_page(pdf, 'Thermal focus: elevation, FAM blocks and the conversion', [
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
        ])
        figure_elevation(pdf, nights)
        figure_fam(pdf, famres)
    print(f'\nwrote {pdf_path}')


if __name__ == '__main__':
    main()
