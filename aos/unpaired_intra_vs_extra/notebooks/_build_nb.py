"""Assemble unpaired_intra_vs_extra.ipynb from cell sources.

Kept beside the notebook so the notebook can be regenerated rather than hand-edited as
JSON. Not part of the analysis.
"""

import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)})


def code(text):
    CELLS.append({'cell_type': 'code', 'execution_count': None, 'metadata': {},
                  'outputs': [], 'source': text.strip().splitlines(True)})


md('''
# Unpaired intra- vs extra-focal wavefront in the Danish 1.3 blitz FAM (v1)

**Author:** Aaron Roodman
**Date Created:** 2026-10-15
**Last Modified:** 2026-10-15
**Status:** In Progress
**Keywords:** AOS, FAM, Danish, intra/extra, Zernike, blur, thermal

## Description

Extends the per-visit blur study (`aos/output/fam_processing/blur_intra_extra/`) to the
Zernike coefficients. For each Full Array Mode (FAM) intra/extra pair, the wavefront
measured from the intra-focal donut is compared against the extra-focal donut, and the
blur study's thermal argument is repeated on the Zernikes.

The Danish 1.3 blitz product is **unpaired**: one row per donut per exposure, each its own
Danish fit, joined across the two sides of focus afterward on `(det_name, donut_id)`. So
intra - extra is a difference of two independent measurements of the same star.

Everything here is **per FAM visit**: the median over that visit's donuts, per side, as
the blur study did.

1. Per-side medians and their difference against visit ordinal, for all 21 Noll indices.
2. Which Zernikes have a difference substantial against the size of their own two-side mean.
3. The intra - extra blur against the intra - extra Z11, over FAM visits.
4. Greedy thermal-predictor selection on incremental gain, for the blur and for Z11.
5. The Huber-corrected residual against the two-side mean and against the extra-focal value.

## Change Log

| Date | Author | Description |
|------|--------|-------------|
| 2026-10-15 | Aaron Roodman | Initial version |
''')

md('''
## Setup

The per-visit medians are precomputed by `extract_visit_medians.py`, which is the only
code that touches the 13 GB donut table. The thermal telemetry is the cache the blur
study already built.
''')

code('''
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, spearmanr

from rubinwork.common.utils import repo_root

STUDY_DIR = repo_root() / 'aos' / 'unpaired_intra_vs_extra'
sys.path.insert(0, str(STUDY_DIR))

from thermal_selection import huber_fit, nmad, select_predictors

RUN = STUDY_DIR / 'output' / '20261015_visit_medians'
PDF_PATH = RUN / 'unpaired_intra_vs_extra.pdf'

PRIMARY = 'danish_1_3_v1000'
SECONDARY = 'danish_1_3_test'

# The 21 Noll indices the Danish 1.2 schema carries, from the table metadata.
NOLL = list(range(4, 20)) + list(range(22, 27))

plt.rcParams.update({'figure.dpi': 110, 'font.size': 8, 'figure.autolayout': False})
''')

code('''
def load_medians(variant):
    """Per-visit per-side medians for one `fam_tables` variant, quality visits only.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``'danish_1_3_v1000'``.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per visit passing ``visit_quality_pass``, ordered by night and sequence
        number, with a 0-based ``ordinal`` column (dimensionless) over the kept visits.
    """
    table = pd.read_parquet(RUN / f'visit_medians_{variant}.parquet')
    table = table[table['visit_quality_pass'].fillna(False)].copy()
    table = table.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)
    table['ordinal'] = np.arange(len(table))
    return table


visits = load_medians(PRIMARY)
visits_secondary = load_medians(SECONDARY)

print(f'{PRIMARY}: {len(visits)} quality visits, {visits.day_obs.nunique()} nights, '
      f'day_obs {visits.day_obs.min()}-{visits.day_obs.max()}')
print(f'{SECONDARY}: {len(visits_secondary)} quality visits')
print(f'donuts per visit: median {visits.n_donuts.median():.0f}, '
      f'min {visits.n_donuts.min()}, max {visits.n_donuts.max()} (dimensionless)')
''')

md('''
### Reproduction checks against the earlier blur study

The blur study reported, over 873 quality triplets: median intra - extra blur
+0.3440 arcsec on `danish_1_3_test`, and intra blurrier in 0.9954 of them
(dimensionless).
''')

code('''
for name, table in [(PRIMARY, visits), (SECONDARY, visits_secondary)]:
    delta = table['blur_intra'] - table['blur_extra']
    # The donut-median route recomputed here must agree with the visits.parquet column.
    from_visits = table['median_blur_intra_arcsec'] - table['median_blur_extra_arcsec']
    print(f'{name}: median(intra - extra) blur {np.median(delta):+.4f} arcsec '
          f'(visits.parquet route {np.median(from_visits):+.4f} arcsec), '
          f'intra > extra in {np.mean(delta > 0):.4f} (dimensionless) of {len(table)}')
''')

md('''
## 1. Per-side medians and their difference, all 21 Noll indices

Panel (a) is the median per side against visit ordinal, panel (b) the per-visit
difference. Night boundaries are drawn so within-night drift is separable by eye. Three
Noll indices per page keeps the ordinal axis readable.
''')

code('''
def night_boundaries(table):
    """Ordinals where the night changes, for vertical rules.

    Parameters
    ----------
    table : `pandas.DataFrame`
        Visit table carrying ``ordinal`` and ``day_obs``.

    Returns
    -------
    edges : `numpy.ndarray`
        Ordinal values (dimensionless) at which ``day_obs`` first changes.
    """
    change = table['day_obs'].to_numpy()[1:] != table['day_obs'].to_numpy()[:-1]
    return table['ordinal'].to_numpy()[1:][change]


def draw_nights(axis, edges):
    for edge in edges:
        axis.axvline(edge, color='0.85', lw=0.5, zorder=0)


EDGES = night_boundaries(visits)


def panel_pair(axis_sides, axis_diff, table, label, unit, intra, extra):
    """Draw the per-side medians and their difference for one quantity."""
    edges = night_boundaries(table)
    draw_nights(axis_sides, edges)
    axis_sides.plot(table['ordinal'], intra, '.', ms=2.0, color='tab:red',
                    label='intra-focal')
    axis_sides.plot(table['ordinal'], extra, '.', ms=2.0, color='tab:blue',
                    label='extra-focal')
    axis_sides.set_ylabel(f'{label} [{unit}]')
    axis_sides.legend(loc='upper right', fontsize=6, markerscale=3, framealpha=0.9)

    difference = intra - extra
    draw_nights(axis_diff, edges)
    axis_diff.plot(table['ordinal'], difference, '.', ms=2.0, color='tab:purple')
    axis_diff.axhline(0.0, color='k', lw=0.5, ls=':')
    axis_diff.axhline(np.median(difference), color='tab:green', lw=0.8,
                      label=f'median {np.median(difference):+.4f} {unit}')
    axis_diff.set_ylabel(f'intra - extra [{unit}]')
    axis_diff.legend(loc='upper right', fontsize=6, framealpha=0.9)
''')

code('''
pdf = PdfPages(PDF_PATH)


def save(figure):
    pdf.savefig(figure, bbox_inches='tight')
    plt.close(figure)


# The blur, as the blur study plotted it, for continuity.
figure, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True)
panel_pair(axes[0], axes[1], visits, 'median donut blur', 'arcsec',
           visits['blur_intra'], visits['blur_extra'])
axes[0].set_title(f'Blur per side of focus and its difference - {PRIMARY}, '
                  f'{len(visits)} quality visits, {visits.day_obs.nunique()} nights')
axes[1].set_xlabel('FAM visit ordinal (dimensionless)')
save(figure)

# Then the 21 Zernikes, three per page.
PER_PAGE = 3
for start in range(0, len(NOLL), PER_PAGE):
    block = NOLL[start:start + PER_PAGE]
    figure, axes = plt.subplots(2, len(block), figsize=(3.4 * len(block), 4.6),
                                sharex=True, squeeze=False)
    for column, noll in enumerate(block):
        panel_pair(axes[0][column], axes[1][column], visits, f'median Z{noll}',
                   'um wf', visits[f'z{noll}_intra'], visits[f'z{noll}_extra'])
        axes[0][column].set_title(f'Z{noll}', fontsize=9)
        axes[1][column].set_xlabel('FAM visit ordinal')
    figure.suptitle(f'Per-side Zernike medians and intra - extra, {PRIMARY} '
                    f'(Noll {block[0]}-{block[-1]})', fontsize=9)
    save(figure)

print(f'pages so far: {pdf.get_pagecount()}')
''')

md('''
## 2. Which Zernikes have a substantial difference

For each Noll index: the median difference over visits, its robust scatter (nMAD), the
median absolute two-side mean, and the ratio

    |median(intra - extra)| / median(|(intra + extra)/2|)   (dimensionless)

which is the "substantial compared with the size of the mean" test. The nMAD is shown
beside it so a large ratio that is pure scatter is distinguishable from a real offset.
''')

code('''
def difference_summary(table, noll_indices=NOLL):
    """Per-Noll difference statistics over visits.

    Parameters
    ----------
    table : `pandas.DataFrame`
        Visit table with ``z{j}_intra`` / ``z{j}_extra`` columns [micrometres of
        wavefront].
    noll_indices : `list` [`int`], optional
        Noll indices to summarize.

    Returns
    -------
    summary : `pandas.DataFrame`
        Columns ``median_diff_um``, ``nmad_diff_um``, ``median_abs_mean_um`` [all
        micrometres of wavefront], ``ratio`` and ``nmad_over_median`` (both
        dimensionless), sorted by ``ratio`` descending.
    """
    rows = []
    for noll in noll_indices:
        intra = table[f'z{noll}_intra'].to_numpy()
        extra = table[f'z{noll}_extra'].to_numpy()
        difference = intra - extra
        two_side_mean = 0.5 * (intra + extra)
        median_diff = float(np.median(difference))
        median_abs_mean = float(np.median(np.abs(two_side_mean)))
        scatter = float(nmad(difference))
        rows.append({
            'noll': noll,
            'median_diff_um': median_diff,
            'nmad_diff_um': scatter,
            'median_abs_mean_um': median_abs_mean,
            'ratio': abs(median_diff) / median_abs_mean if median_abs_mean else np.nan,
            'nmad_over_median': scatter / abs(median_diff) if median_diff else np.nan,
        })
    return pd.DataFrame(rows).sort_values('ratio', ascending=False).reset_index(drop=True)


summary = difference_summary(visits)
summary_secondary = difference_summary(visits_secondary)

print(f'{PRIMARY}, {len(visits)} quality visits. Zernike coefficients in um of wavefront;')
print('ratio = |median diff| / median|two-side mean| (dimensionless).')
print(summary.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
''')

code('''
# The ranking figure: the ratio, with the median difference and its nMAD beside it.
order = summary['noll'].to_numpy()
position = np.arange(len(order))

figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2),
                            gridspec_kw={'width_ratios': [1.0, 1.3]})

axes[0].barh(position, summary['ratio'], color='tab:purple', alpha=0.8)
axes[0].axvline(1.0, color='k', lw=0.8, ls='--', label='difference equals the mean')
axes[0].set_yticks(position)
axes[0].set_yticklabels([f'Z{n}' for n in order])
axes[0].invert_yaxis()
axes[0].set_xlabel('|median(intra - extra)| / median|(intra + extra)/2|  (dimensionless)')
axes[0].set_title('Difference against the size of its own mean')
axes[0].legend(fontsize=6)

axes[1].errorbar(summary['median_diff_um'], position, xerr=summary['nmad_diff_um'],
                 fmt='o', ms=3, lw=0.8, capsize=2, color='tab:red')
axes[1].axvline(0.0, color='k', lw=0.8, ls=':')
axes[1].set_yticks(position)
axes[1].set_yticklabels([f'Z{n}' for n in order])
axes[1].invert_yaxis()
axes[1].set_xlabel('median(intra - extra) [um of wavefront], bars are nMAD over visits')
axes[1].set_title('Signed difference and its robust scatter')

figure.suptitle(f'Per-Noll intra - extra, {PRIMARY}, {len(visits)} quality visits',
                fontsize=9)
save(figure)
''')

code('''
# The pupil-model contrast: which differences v1000 moves against the blitz default.
contrast = summary[['noll', 'median_diff_um']].merge(
    summary_secondary[['noll', 'median_diff_um']], on='noll',
    suffixes=(f'_{PRIMARY}', f'_{SECONDARY}'))
contrast['delta_um'] = (contrast[f'median_diff_um_{PRIMARY}']
                        - contrast[f'median_diff_um_{SECONDARY}'])
contrast = contrast.reindex(contrast['delta_um'].abs().sort_values(ascending=False).index)

print('Pupil model contrast. median(intra - extra) per variant, um of wavefront;')
print(f'delta_um = {PRIMARY} minus {SECONDARY}.')
print(contrast.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

figure, axis = plt.subplots(figsize=(7.5, 4.0))
width = 0.4
position = np.arange(len(contrast))
axis.barh(position - width / 2, contrast[f'median_diff_um_{PRIMARY}'], width,
          label=PRIMARY, color='tab:blue', alpha=0.85)
axis.barh(position + width / 2, contrast[f'median_diff_um_{SECONDARY}'], width,
          label=SECONDARY, color='tab:orange', alpha=0.85)
axis.axvline(0.0, color='k', lw=0.8, ls=':')
axis.set_yticks(position)
axis.set_yticklabels([f'Z{int(n)}' for n in contrast['noll']])
axis.invert_yaxis()
axis.set_xlabel('median(intra - extra) [um of wavefront]')
axis.set_title('Pupil model: v1000 against the blitz default, same 873 visits')
axis.legend(fontsize=7)
save(figure)
''')

md('''
## 3. The blur difference against the Z11 difference

One point per FAM visit, coloured by night. Spearman rho and Pearson r with n, and a
Huber slope with its standard error. Reported for the full sample and within-night
de-meaned, since the blur study found roughly half the full-sample signal is night-level
covariance with the seeing.
''')

code('''
def within_night(table, values):
    """Subtract each night's median, leaving the within-night variation.

    Parameters
    ----------
    table : `pandas.DataFrame`
        Visit table carrying ``day_obs``.
    values : `pandas.Series`
        Values on the same index, in any unit.

    Returns
    -------
    demeaned : `pandas.Series`
        `values` minus the median of its own night, in the unit of `values`.
    """
    return values - values.groupby(table['day_obs']).transform('median')


def correlate(x_values, y_values, x_label, y_label, x_unit, y_unit):
    """Spearman, Pearson and a Huber slope for one pair of per-visit quantities."""
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    x_kept = np.asarray(x_values)[keep]
    y_kept = np.asarray(y_values)[keep]
    rho = spearmanr(x_kept, y_kept).statistic
    r = pearsonr(x_kept, y_kept).statistic
    fit = huber_fit(y_kept, x_kept.reshape(-1, 1))
    return {
        'x': x_label, 'y': y_label, 'n': int(keep.sum()),
        'spearman_rho': float(rho), 'pearson_r': float(r),
        'huber_slope': float(fit.params[1]), 'huber_slope_err': float(fit.bse[1]),
        'slope_unit': f'{y_unit} per {x_unit}',
    }


def scatter_by_night(axis, table, x_values, y_values, x_label, y_label):
    """Scatter one per-visit pair, coloured by night, with the Huber line drawn."""
    nights = np.sort(table['day_obs'].unique())
    colours = plt.cm.viridis(np.linspace(0, 1, len(nights)))
    for colour, night in zip(colours, nights):
        mask = (table['day_obs'] == night).to_numpy()
        axis.plot(np.asarray(x_values)[mask], np.asarray(y_values)[mask], '.', ms=3,
                  color=colour, alpha=0.8)
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    x_kept = np.asarray(x_values)[keep]
    fit = huber_fit(np.asarray(y_values)[keep], x_kept.reshape(-1, 1))
    grid = np.linspace(np.min(x_kept), np.max(x_kept), 50)
    axis.plot(grid, fit.params[0] + fit.params[1] * grid, '-', color='tab:red', lw=1.2)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
''')

code('''
# Z11 is the expected term; the summary above decides whether others qualify.
FOLLOW_UP = [11, 14, 26, 6]

visits['dblur'] = visits['blur_intra'] - visits['blur_extra']
for noll in NOLL:
    visits[f'dz{noll}'] = visits[f'z{noll}_intra'] - visits[f'z{noll}_extra']

records = []
for noll in FOLLOW_UP:
    records.append(correlate(visits['dblur'], visits[f'dz{noll}'],
                             'dblur', f'dZ{noll}', 'arcsec', 'um wf'))
    records.append(correlate(within_night(visits, visits['dblur']),
                             within_night(visits, visits[f'dz{noll}']),
                             'dblur (within night)', f'dZ{noll} (within night)',
                             'arcsec', 'um wf'))
correlations = pd.DataFrame(records)
print('intra - extra blur against intra - extra Zernike, per FAM visit.')
print(correlations.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
''')

code('''
figure, axes = plt.subplots(2, len(FOLLOW_UP), figsize=(3.1 * len(FOLLOW_UP), 5.6))
for column, noll in enumerate(FOLLOW_UP):
    row = correlations[(correlations.y == f'dZ{noll}')].iloc[0]
    scatter_by_night(axes[0][column], visits, visits['dblur'], visits[f'dz{noll}'],
                     'intra - extra blur [arcsec]', f'intra - extra Z{noll} [um wf]')
    axes[0][column].set_title(
        f'Z{noll}: rho {row.spearman_rho:+.3f}, r {row.pearson_r:+.3f}\\n'
        f'slope {row.huber_slope:+.4f} +/- {row.huber_slope_err:.4f} um wf/arcsec, '
        f'n = {row.n}', fontsize=7)

    row_wn = correlations[(correlations.y == f'dZ{noll} (within night)')].iloc[0]
    scatter_by_night(axes[1][column], visits, within_night(visits, visits['dblur']),
                     within_night(visits, visits[f'dz{noll}']),
                     'intra - extra blur, within night [arcsec]',
                     f'intra - extra Z{noll}, within night [um wf]')
    axes[1][column].set_title(
        f'within night: rho {row_wn.spearman_rho:+.3f}\\n'
        f'slope {row_wn.huber_slope:+.4f} +/- {row_wn.huber_slope_err:.4f} um wf/arcsec',
        fontsize=7)

figure.suptitle(f'Blur difference against Zernike difference, {PRIMARY}, '
                f'{len(visits)} quality visits, colour is night', fontsize=9)
save(figure)
''')

md('''
## 4. Thermal predictor selection

The earlier blur study ranked the candidates by their own correlation with the target and
added them in that order. The thermal quantities are themselves strongly correlated, so
that order double-counts.

`select_predictors` instead:

- **(a)** ranks every candidate by its own Spearman rho and **records its sign**, before
  any orthogonalisation;
- **(b)** takes the leading variable, then re-ranks the remaining candidates by what each
  *adds* given those already chosen - the reduction in the Huber residual nMAD - and takes
  the largest;
- **(c)** continues to `n_max`, which is settable.

Each step reports the incremental reduction in the target's unit, the partial correlation
against the current residual, and a variance inflation factor (dimensionless) so a step
that nearly repeats an earlier variable is visible rather than silent.
''')

code('''
N_PREDICTORS = 5

thermal = pd.read_parquet(repo_root() / 'aos' / 'output' / 'fam_processing'
                          / 'blur_intra_extra' / 'thermal_join_per_side_cam.parquet')

PER_SIDE = ['m1m3_air_temp', 'm2_air_temp', 'cam_air_temp', 'outside_temp',
            'truss_temp_mean_c', 'm1m3_x_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
            'm1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m',
            'm1m3_r2_coeff_c', 'm1m3_rms_c', 'm1m3_mean_temp_c', 'm1m3_std_temp_c',
            'm1m3_range_temp_c', 'm1_mean_temp_c', 'm3_mean_temp_c', 'cam_AverageTemp']

joined = visits.merge(thermal, on=['day_obs', 'seq_num'], how='left')

# The two exposures see the same thermal state to a few parts in a thousand (blur study),
# so one side's value stands for the pair; the extra-focal exposure is the `visit`.
candidates = pd.DataFrame(index=joined.index)
for name in PER_SIDE:
    candidates[name] = joined[f'{name}_extra']
candidates['cam_body_minus_m1m3_air'] = (joined['cam_AverageTemp_extra']
                                         - joined['m1m3_air_temp_extra'])
candidates['m1m3_glass_minus_air'] = (joined['m1m3_mean_temp_c_extra']
                                      - joined['m1m3_air_temp_extra'])
candidates['truss_minus_outside'] = (joined['truss_temp_mean_c_extra']
                                     - joined['outside_temp_extra'])
candidates['cam_air_minus_outside'] = (joined['cam_air_temp_extra']
                                       - joined['outside_temp_extra'])
candidates['m2_air_minus_m1m3_air'] = (joined['m2_air_temp_extra']
                                       - joined['m1m3_air_temp_extra'])

print(f'{candidates.shape[1]} candidate thermal quantities over {len(joined)} visits')
''')

code('''
TARGETS = [('dblur', joined['blur_intra'] - joined['blur_extra'], 'arcsec')]
for noll in FOLLOW_UP:
    TARGETS.append((f'dZ{noll}',
                    joined[f'z{noll}_intra'] - joined[f'z{noll}_extra'], 'um wf'))

selections = {}
for label, target, unit in TARGETS:
    steps, ranking, fit = select_predictors(target.rename(label), candidates,
                                            n_max=N_PREDICTORS)
    selections[label] = {'steps': steps, 'ranking': ranking, 'fit': fit,
                         'target': target, 'unit': unit}
    start = nmad((target - np.median(target.dropna())).dropna())
    print(f'=== {label}: starting nMAD {start:.5f} {unit}, '
          f'{len(steps)} predictors selected')
    print(steps.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
''')

code('''
# One page per target: the step (a) ranking, and the incremental gain per step.
for label in selections:
    entry = selections[label]
    steps, ranking, unit = entry['steps'], entry['ranking'], entry['unit']

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4),
                                gridspec_kw={'width_ratios': [1.3, 1.0]})

    top = ranking.head(12)
    position = np.arange(len(top))
    axes[0].barh(position, top['rho_own'],
                 color=['tab:blue' if v >= 0 else 'tab:red' for v in top['rho_own']],
                 alpha=0.85)
    axes[0].axvline(0.0, color='k', lw=0.8)
    axes[0].set_yticks(position)
    axes[0].set_yticklabels(top.index, fontsize=6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f'Spearman rho against {label} (dimensionless)')
    axes[0].set_title('Step (a): own correlation, top 12, sign by colour', fontsize=8)

    position = np.arange(len(steps))
    axes[1].bar(position, steps['gain_frac'] * 100.0, color='tab:green', alpha=0.85)
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(
        [f'{v}\\n(VIF {w:.1f})' for v, w in zip(steps['variable'], steps['vif'])],
        rotation=35, ha='right', fontsize=5)
    axes[1].set_ylabel('reduction in residual nMAD [per cent of current]')
    axes[1].set_title(f'Steps (b)-(c): incremental gain, nMAD '
                      f'{steps.nmad_after.iloc[0]:.4f} to '
                      f'{steps.nmad_after.iloc[-1]:.4f} {unit}', fontsize=8)

    figure.suptitle(f'Thermal predictor selection for {label} [{unit}], '
                    f'{PRIMARY}', fontsize=9)
    save(figure)
''')

md('''
## 5. After the Huber thermal correction

The step that carried the blur study's conclusion: subtract the fitted thermal model and
look at the residual against the two-side mean and against the extra-focal value. If the
correlation flattens, the thermal variables were driving the difference rather than the
overall seeing or the wavefront level.
''')

code('''
records = []
for label in selections:
    entry = selections[label]
    target, fit, unit = entry['target'], entry['fit'], entry['unit']

    used = pd.concat([target.rename('_t'), candidates], axis=1).dropna().index
    residual = pd.Series(np.nan, index=joined.index)
    residual.loc[used] = target.loc[used].to_numpy() - fit.fittedvalues
    entry['residual'] = residual

    noll = None if label == 'dblur' else int(label[2:])
    if noll is None:
        reference_extra = joined['blur_extra']
        reference_mean = 0.5 * (joined['blur_intra'] + joined['blur_extra'])
        reference_unit = 'arcsec'
    else:
        reference_extra = joined[f'z{noll}_extra']
        reference_mean = 0.5 * (joined[f'z{noll}_intra'] + joined[f'z{noll}_extra'])
        reference_unit = 'um wf'

    entry['reference_extra'] = reference_extra
    entry['reference_mean'] = reference_mean
    entry['reference_unit'] = reference_unit

    for reference_label, reference in [('extra-focal', reference_extra),
                                       ('two-side mean', reference_mean)]:
        before = correlate(reference, target, reference_label, f'{label} before',
                           reference_unit, unit)
        after = correlate(reference, residual, reference_label, f'{label} after',
                          reference_unit, unit)
        records.append({'target': label, 'against': reference_label,
                        'slope_before': before['huber_slope'],
                        'err_before': before['huber_slope_err'],
                        'slope_after': after['huber_slope'],
                        'err_after': after['huber_slope_err'],
                        'unit': f'{unit} per {reference_unit}', 'n': after['n']})

    entry['nmad_before'] = nmad((target - np.median(target.dropna())).dropna())
    entry['nmad_after'] = nmad(residual.dropna())

flattening = pd.DataFrame(records)
print('Huber slope of the difference against the reference, before and after the '
      'thermal correction.')
print(flattening.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print()
for label in selections:
    entry = selections[label]
    print(f'{label}: nMAD {entry["nmad_before"]:.5f} -> {entry["nmad_after"]:.5f} '
          f'{entry["unit"]} '
          f'({100 * (1 - entry["nmad_after"] / entry["nmad_before"]):.1f} per cent '
          f'reduction, dimensionless)')
''')

code('''
for label in selections:
    entry = selections[label]
    unit, reference_unit = entry['unit'], entry['reference_unit']

    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for row, (reference_label, reference) in enumerate(
            [('extra-focal', entry['reference_extra']),
             ('two-side mean', entry['reference_mean'])]):
        for column, (stage, values) in enumerate(
                [('before', entry['target']), ('after', entry['residual'])]):
            axis = axes[row][column]
            scatter_by_night(axis, joined, reference, values,
                             f'{reference_label} [{reference_unit}]',
                             f'{label} {stage} [{unit}]')
            match = flattening[(flattening.target == label)
                               & (flattening.against == reference_label)].iloc[0]
            slope = match.slope_before if stage == 'before' else match.slope_after
            error = match.err_before if stage == 'before' else match.err_after
            axis.set_title(f'{stage}: slope {slope:+.4f} +/- {error:.4f} '
                           f'{unit}/{reference_unit}', fontsize=7)

    figure.suptitle(
        f'{label} against its reference, before and after the Huber thermal correction '
        f'({len(entry["steps"])} predictors)\\n'
        f'nMAD {entry["nmad_before"]:.5f} -> {entry["nmad_after"]:.5f} {unit}',
        fontsize=9)
    save(figure)
''')

md('''
## Close the PDF

One notebook, one PDF, one `study.md` - the study rule.
''')

code('''
page_count = pdf.get_pagecount()
pdf.close()
print(f'wrote {PDF_PATH}')
print(f'pages: {page_count} (dimensionless)')
''')

code('''
# Machine-readable companions to the figures, beside the PDF.
summary.to_parquet(RUN / 'per_noll_difference_summary.parquet', index=False)
correlations.to_parquet(RUN / 'dblur_vs_dzk_correlations.parquet', index=False)
flattening.to_parquet(RUN / 'thermal_correction_flattening.parquet', index=False)
pd.concat([entry['steps'].assign(target=label) for label, entry in selections.items()]
          ).to_parquet(RUN / 'thermal_selection_steps.parquet', index=False)
print(f'wrote 4 parquet summaries to {RUN}')
''')

notebook = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {'display_name': 'LSST', 'language': 'python', 'name': 'lsst'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

path = pathlib.Path(__file__).parent / 'unpaired_intra_vs_extra.ipynb'
path.write_text(json.dumps(notebook, indent=1))
print(f'wrote {path}, {len(CELLS)} cells')
