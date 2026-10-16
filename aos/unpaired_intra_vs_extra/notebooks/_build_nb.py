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

One FAM visit is dropped as a clear outlier: `day_obs` 20260327 `seq_num` 228, whose
two-side mean sits at Z4 = +4.4153 um of wavefront - a robust z of +11.8 (dimensionless)
against the sample, with Z6 at -11.7 and Z5 at -6.1. The next largest |mean Z4| in the
sample is +1.089 um of wavefront, so it is isolated by a factor of 4 rather than being the
tail of a distribution.

Robust scatter throughout is `scipy.stats.median_abs_deviation(..., scale='normal')`,
wrapped as `nmad` in `thermal_selection.py`.
''')

code('''
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
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

# One FAM visit is a clear outlier and is dropped: its two-side mean sits at
# Z4 = +4.4153 um of wavefront, a robust z of +11.8 (dimensionless) against the sample,
# with Z6 at -11.7 and Z5 at -6.1 as well. The next largest |mean Z4| in the sample is
# +1.089 um of wavefront, so the gap is a factor of 4 and the visit is isolated rather
# than the tail of a distribution.
EXCLUDED_VISITS = [(20260327, 228)]

plt.rcParams.update({'figure.dpi': 110, 'font.size': 8, 'figure.autolayout': False})
''')

code('''
def load_medians(variant, excluded=EXCLUDED_VISITS):
    """Per-visit per-side medians for one `fam_tables` variant, quality visits only.

    Parameters
    ----------
    variant : `str`
        Variant name, e.g. ``'danish_1_3_v1000'``.
    excluded : `list` [`tuple`], optional
        ``(day_obs, seq_num)`` pairs to drop as outliers.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per visit passing ``visit_quality_pass`` and not excluded, ordered by
        night and sequence number, with a 0-based ``ordinal`` column (dimensionless)
        over the kept visits.
    """
    table = pd.read_parquet(RUN / f'visit_medians_{variant}.parquet')
    table = table[table['visit_quality_pass'].fillna(False)].copy()

    keys = list(zip(table['day_obs'], table['seq_num']))
    drop = np.array([key in set(excluded) for key in keys])
    if drop.any():
        for day_obs, seq_num in sorted(set(excluded) & set(keys)):
            row = table[(table.day_obs == day_obs) & (table.seq_num == seq_num)].iloc[0]
            mean_z4 = 0.5 * (row['z4_intra'] + row['z4_extra'])
            print(f'{variant}: dropping day_obs {day_obs} seq_num {seq_num}, '
                  f'two-side mean Z4 {mean_z4:+.4f} um of wavefront')
    table = table[~drop]

    table = table.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)
    table['ordinal'] = np.arange(len(table))
    return table


visits = load_medians(PRIMARY)
visits_secondary = load_medians(SECONDARY)

print(f'{PRIMARY}: {len(visits)} visits kept, {visits.day_obs.nunique()} nights, '
      f'day_obs {visits.day_obs.min()}-{visits.day_obs.max()}')
print(f'{SECONDARY}: {len(visits_secondary)} visits kept')
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
    median = np.median(difference)
    scatter = nmad(difference)
    draw_nights(axis_diff, edges)
    axis_diff.plot(table['ordinal'], difference, '.', ms=2.0, color='tab:purple')
    axis_diff.axhline(0.0, color='k', lw=0.5, ls=':')
    axis_diff.axhline(median, color='tab:green', lw=0.8,
                      label=f'median {median:+.4f} {unit}')
    # Robust RMS of the difference over visit pairs, as a +/- 1 nMAD band.
    axis_diff.axhspan(median - scatter, median + scatter, color='tab:green', alpha=0.13,
                      label=f'nMAD {scatter:.4f} {unit}')
    axis_diff.set_ylabel(f'intra - extra [{unit}]')
    axis_diff.legend(loc='upper right', fontsize=6, framealpha=0.9)
    return median, scatter
''')

code('''
pdf = PdfPages(PDF_PATH)


def save(figure):
    pdf.savefig(figure, bbox_inches='tight')
    plt.close(figure)


# The blur, as the blur study plotted it, for continuity.
figure, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True,
                            layout='constrained')
panel_pair(axes[0], axes[1], visits, 'median donut blur', 'arcsec',
           visits['blur_intra'], visits['blur_extra'])
axes[0].set_title(f'Blur per side of focus and its difference - {PRIMARY}, '
                  f'{len(visits)} visits, {visits.day_obs.nunique()} nights')
axes[1].set_xlabel('FAM visit ordinal (dimensionless)')
save(figure)

# Then the 21 Zernikes, three per page. `layout='constrained'` and the explicit
# per-axis y-labels keep each column's label off its left neighbour's panel; the
# default tight layout let them overlap.
PER_PAGE = 3
ordinal_scatter = {}
for start in range(0, len(NOLL), PER_PAGE):
    block = NOLL[start:start + PER_PAGE]
    figure, axes = plt.subplots(2, len(block), figsize=(4.1 * len(block), 4.9),
                                sharex=True, squeeze=False, layout='constrained')
    for column, noll in enumerate(block):
        median, scatter = panel_pair(
            axes[0][column], axes[1][column], visits, f'median Z{noll}', 'um wf',
            visits[f'z{noll}_intra'], visits[f'z{noll}_extra'])
        ordinal_scatter[noll] = (median, scatter)
        axes[0][column].set_title(
            f'Z{noll}: intra - extra median {median:+.4f}, nMAD {scatter:.4f} um wf',
            fontsize=7)
        axes[1][column].set_xlabel('FAM visit ordinal')
    figure.suptitle(f'Per-side Zernike medians and intra - extra, {PRIMARY} '
                    f'(Noll {block[0]}-{block[-1]}), {len(visits)} visits', fontsize=9)
    save(figure)

print(f'pages so far: {pdf.get_pagecount()}')
print('intra - extra median and robust RMS (nMAD) over visit pairs, um of wavefront:')
for noll, (median, scatter) in ordinal_scatter.items():
    print(f'  Z{noll:<3d} {median:+.4f}  nMAD {scatter:.4f}')
''')

md('''
## 2. Which Zernikes have a substantial difference

For each Noll index: the median difference over visits, its robust scatter (nMAD, the
robust RMS of the difference over visit pairs), the median absolute two-side mean, and the
ratio

    |median(intra - extra)| / median(|(intra + extra)/2|)   (dimensionless)

which is the "substantial compared with the size of the mean" test. The nMAD is shown
beside it so a large ratio that is pure scatter is distinguishable from a real offset.

The follow-up set carried into sections 3-5 is then chosen **from this table**, not fixed
in advance: a Noll index qualifies on either a ratio at or above 1 (the difference is as
large as its own mean) or a robust scatter in the top 8. The second test is what brings in
Z4, whose median difference is modest but whose excursions over visits are among the
largest in the sample.
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

figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), layout='constrained',
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

figure, axis = plt.subplots(figsize=(7.8, 4.4), layout='constrained')
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
# The follow-up set is chosen from the data, not fixed in advance: a Noll index
# qualifies on EITHER a difference substantial against its own two-side mean (ratio >= 1)
# OR a large robust scatter of that difference over visit pairs (nMAD in the top 8).
# Z4 qualifies on the second test - its median difference is modest but its excursions
# over visits are among the largest in the sample, which the ordinal plots show plainly.
RATIO_THRESHOLD = 1.0
N_BY_SCATTER = 8

by_ratio = set(summary.loc[summary['ratio'] >= RATIO_THRESHOLD, 'noll'])
by_scatter = set(summary.nlargest(N_BY_SCATTER, 'nmad_diff_um')['noll'])
FOLLOW_UP = sorted(by_ratio | by_scatter)

print(f'follow-up set, {len(FOLLOW_UP)} Noll indices: {FOLLOW_UP}')
print(f'  qualified by ratio >= {RATIO_THRESHOLD}: {sorted(by_ratio)}')
print(f'  qualified by nMAD in the top {N_BY_SCATTER}: {sorted(by_scatter)}')
print(f'  on both counts: {sorted(by_ratio & by_scatter)}')

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
PER_ROW = 4
for start in range(0, len(FOLLOW_UP), PER_ROW):
    block = FOLLOW_UP[start:start + PER_ROW]
    figure, axes = plt.subplots(2, len(block), figsize=(3.5 * len(block), 6.0),
                                squeeze=False, layout='constrained')
    for column, noll in enumerate(block):
        row = correlations[(correlations.y == f'dZ{noll}')].iloc[0]
        scatter_by_night(axes[0][column], visits, visits['dblur'], visits[f'dz{noll}'],
                         'intra - extra blur [arcsec]',
                         f'intra - extra Z{noll} [um wf]')
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
            f'slope {row_wn.huber_slope:+.4f} +/- {row_wn.huber_slope_err:.4f} '
            f'um wf/arcsec', fontsize=7)

    figure.suptitle(f'Blur difference against Zernike difference, {PRIMARY}, '
                    f'{len(visits)} visits, colour is night '
                    f'(Z{block[0]}-Z{block[-1]})', fontsize=9)
    save(figure)
''')

md("""
## 3b. Z4 and the intra-before-extra time ordering

A FAM triplet is taken in a fixed order - intra, extra, in-focus - so the intra-focal
exposure always precedes the extra-focal one, here by a median **46.4 s**. On many nights
the two-side mean Z4 drifts steadily, which is truss thermal expansion. A fixed time
ordering against a drifting Z4 would produce a dZ4 that is partly an artifact of *when*
each side was measured rather than a difference between the two sides.

The test: fit the intra-focal Z4 as a function of time within each night, interpolate it
onto each pair's **extra-focal** epoch, and recompute dZ4 from the interpolated value. If
the time ordering drives dZ4, the corrected difference shrinks.

Only Z4 gets this treatment (Aaron, 2026-10-16); it is the term whose nightly drift is
large compared with its own difference.
""")

code("""
times = pd.read_parquet(RUN / 'exposure_times.parquet')
midpoint = times.set_index('exposure_id')['exp_midpt_mjd']

visits['t_intra'] = visits['intra_visit'].map(midpoint)
visits['t_extra'] = visits['extra_visit'].map(midpoint)
gap_seconds = (visits['t_extra'] - visits['t_intra']) * 86400.0

print(f'exposure times: {times.shape[0]} exposures over {times.day_obs.nunique()} nights')
print(f'missing: {visits.t_intra.isna().sum()} intra, {visits.t_extra.isna().sum()} extra')
print(f'intra to extra gap [s]: median {gap_seconds.median():.1f}, '
      f'10th pct {gap_seconds.quantile(0.1):.1f}, '
      f'90th pct {gap_seconds.quantile(0.9):.1f}')
""")

code("""
def interpolate_intra_to_extra(table, noll):
    \"\"\"Interpolate one side's per-visit Zernike median onto the other side's epoch.

    Within each night the intra-focal series ``z{noll}_intra`` against its own exposure
    time is interpolated onto each pair's extra-focal exposure time, so the recomputed
    difference compares the two sides at the *same* epoch.

    Parameters
    ----------
    table : `pandas.DataFrame`
        Visit table carrying ``day_obs``, ``t_intra``, ``t_extra`` [days] and the
        per-side Zernike medians [micrometres of wavefront].
    noll : `int`
        Noll index to interpolate.

    Returns
    -------
    interpolated : `pandas.Series`
        The intra-focal coefficient evaluated at the extra-focal epoch, in micrometres
        of wavefront, on `table`'s index.
    \"\"\"
    result = pd.Series(np.nan, index=table.index)
    for _, night in table.groupby('day_obs'):
        night = night.sort_values('t_intra')
        finite = night[['t_intra', f'z{noll}_intra']].notna().all(axis=1)
        if finite.sum() < 2:
            continue
        result.loc[night.index] = np.interp(
            night['t_extra'], night.loc[finite, 't_intra'],
            night.loc[finite, f'z{noll}_intra'])
    return result


TIME_ORDER_NOLL = 4

visits[f'z{TIME_ORDER_NOLL}_intra_interp'] = interpolate_intra_to_extra(
    visits, TIME_ORDER_NOLL)
visits[f'dz{TIME_ORDER_NOLL}_interp'] = (visits[f'z{TIME_ORDER_NOLL}_intra_interp']
                                         - visits[f'z{TIME_ORDER_NOLL}_extra'])

raw = visits[f'z{TIME_ORDER_NOLL}_intra'] - visits[f'z{TIME_ORDER_NOLL}_extra']
corrected = visits[f'dz{TIME_ORDER_NOLL}_interp']
shift = visits[f'z{TIME_ORDER_NOLL}_intra_interp'] - visits[f'z{TIME_ORDER_NOLL}_intra']

# Per-night drift rate of the two-side mean, for scale.
rates = []
for day_obs, night in visits.groupby('day_obs'):
    night = night.sort_values('t_extra')
    if len(night) < 5:
        continue
    hours = (night['t_extra'] - night['t_extra'].min()) * 24.0
    mean_z4 = 0.5 * (night[f'z{TIME_ORDER_NOLL}_intra']
                     + night[f'z{TIME_ORDER_NOLL}_extra'])
    rates.append({'day_obs': day_obs, 'n_visits': len(night),
                  'drift_um_per_hour': float(np.polyfit(hours, mean_z4, 1)[0])})
drift = pd.DataFrame(rates)

print(f'Z{TIME_ORDER_NOLL} two-side mean drift per night [um of wavefront per hour]:')
print(drift.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print()
print(f'dZ{TIME_ORDER_NOLL} raw          : median {np.median(raw):+.4f} um wf, '
      f'nMAD {nmad(raw):.4f} um wf')
print(f'dZ{TIME_ORDER_NOLL} interpolated : median {np.median(corrected.dropna()):+.4f} '
      f'um wf, nMAD {nmad(corrected):.4f} um wf')
print(f'change in nMAD: {100 * (nmad(corrected) / nmad(raw) - 1):+.2f} per cent '
      f'(dimensionless)')
print(f'shift applied: median {np.median(shift.dropna()):+.5f} um wf, '
      f'nMAD {nmad(shift):.5f} um wf')
print(f'drift x gap, order of magnitude: '
      f'{np.median(np.abs(drift.drift_um_per_hour)) * gap_seconds.median() / 3600:.5f} '
      f'um wf')
""")

code("""
figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.6), sharex=True, layout='constrained')

draw_nights(axes[0], EDGES)
axes[0].plot(visits['ordinal'], 0.5 * (visits[f'z{TIME_ORDER_NOLL}_intra']
                                       + visits[f'z{TIME_ORDER_NOLL}_extra']),
             '.', ms=2.0, color='tab:brown')
axes[0].set_ylabel(f'two-side mean Z{TIME_ORDER_NOLL} [um wf]')
axes[0].set_title(f'The drifting quantity: two-side mean Z{TIME_ORDER_NOLL}, '
                  f'{np.median(np.abs(drift.drift_um_per_hour)):.3f} um wf/hour '
                  f'median |drift| per night', fontsize=8)

for axis, values, label in [(axes[1], raw, 'raw'),
                            (axes[2], corrected, 'interpolated to the extra epoch')]:
    draw_nights(axis, EDGES)
    axis.plot(visits['ordinal'], values, '.', ms=2.0, color='tab:purple')
    axis.axhline(0.0, color='k', lw=0.5, ls=':')
    median = np.median(values.dropna()) if hasattr(values, 'dropna') else np.median(values)
    scatter = nmad(values)
    axis.axhline(median, color='tab:green', lw=0.8)
    axis.axhspan(median - scatter, median + scatter, color='tab:green', alpha=0.13)
    axis.set_ylabel(f'dZ{TIME_ORDER_NOLL} {label} [um wf]')
    axis.set_title(f'{label}: median {median:+.4f}, nMAD {scatter:.4f} um wf',
                   fontsize=8)

axes[2].set_xlabel('FAM visit ordinal (dimensionless)')
figure.suptitle(f'Does the intra-before-extra ordering explain dZ{TIME_ORDER_NOLL}? '
                f'Gap is {gap_seconds.median():.1f} s', fontsize=9)
save(figure)
""")

md("""
## 3c. Per-donut differences

So far every difference has been taken between the two sides' per-visit medians. The
difference can instead be formed **per donut** and then summarized over the visit, which
gives a second quantity the visit medians cannot: the **spread of the difference across
the focal plane within a single visit**.

No positional matching is needed. Each row of `donuts.parquet` is already a matched pair:
`pair_sides` joins the two sides of focus on `(det_name, donut_id)`, where `donut_id` is
the Gaia source identifier. That key is exact - a paired star carries identical
`coord_ra` / `coord_dec` on both sides - and it keeps the stars fitted successfully on
both sides, about **96.8 per cent** of the per-side fitted rows. Pixel-position matching
is deliberately *not* used: the same star lands up to **34.7 pixels** apart on the two
sides because the two exposures point slightly differently, so any tolerance tight enough
to be safe would reject real pairs (`aos/docs/studies/fam_processing.md`).

`extract_visit_medians.py` computes `dz{j}_donut_median` and `dz{j}_donut_nmad` per visit
from the per-donut differences, so these plots need no extra pass over the 13 GB table.
""")

code("""
# Consistency check: the median of the per-donut differences must match the difference of
# the per-side medians, up to the difference between a median of differences and a
# difference of medians.
check = []
for noll in NOLL:
    per_donut = visits[f'dz{noll}_donut_median']
    of_medians = visits[f'z{noll}_intra'] - visits[f'z{noll}_extra']
    check.append({'noll': noll,
                  'median_per_donut_um': float(np.median(per_donut)),
                  'median_of_medians_um': float(np.median(of_medians)),
                  'max_abs_diff_um': float(np.max(np.abs(per_donut - of_medians))),
                  'median_within_visit_nmad_um':
                      float(np.median(visits[f'dz{noll}_donut_nmad']))})
per_donut_summary = pd.DataFrame(check)

print('Per-donut differences against the difference of per-side medians, '
      'um of wavefront.')
print('within-visit nMAD is the spread of the per-donut difference across the focal '
      'plane.')
print(per_donut_summary.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print()
print(f'blur: per-donut median {np.median(visits.dblur_donut_median):+.4f} arcsec, '
      f'within-visit nMAD {np.median(visits.dblur_donut_nmad):.4f} arcsec')
""")

code("""
# Median and nMAD of the per-donut difference against FAM visit ordinal, two panels per
# Noll index, for the follow-up set.
for start in range(0, len(FOLLOW_UP), PER_PAGE):
    block = FOLLOW_UP[start:start + PER_PAGE]
    figure, axes = plt.subplots(2, len(block), figsize=(4.1 * len(block), 4.9),
                                sharex=True, squeeze=False, layout='constrained')
    for column, noll in enumerate(block):
        median_series = visits[f'dz{noll}_donut_median']
        scatter_series = visits[f'dz{noll}_donut_nmad']

        axis = axes[0][column]
        draw_nights(axis, EDGES)
        axis.plot(visits['ordinal'], median_series, '.', ms=2.0, color='tab:purple')
        axis.axhline(0.0, color='k', lw=0.5, ls=':')
        axis.axhline(np.median(median_series), color='tab:green', lw=0.8,
                     label=f'median {np.median(median_series):+.4f} um wf')
        axis.set_ylabel(f'median per-donut dZ{noll} [um wf]')
        axis.legend(loc='upper right', fontsize=6, framealpha=0.9)
        axis.set_title(f'Z{noll}', fontsize=9)

        axis = axes[1][column]
        draw_nights(axis, EDGES)
        axis.plot(visits['ordinal'], scatter_series, '.', ms=2.0, color='tab:orange')
        axis.axhline(np.median(scatter_series), color='tab:green', lw=0.8,
                     label=f'median {np.median(scatter_series):.4f} um wf')
        axis.set_ylabel(f'within-visit nMAD of dZ{noll} [um wf]')
        axis.legend(loc='upper right', fontsize=6, framealpha=0.9)
        axis.set_xlabel('FAM visit ordinal')

    figure.suptitle(f'Per-donut intra - extra: median (top) and within-visit robust '
                    f'scatter (bottom), {PRIMARY} (Z{block[0]}-Z{block[-1]})',
                    fontsize=9)
    save(figure)
""")

code("""
# The two scatters side by side: within a visit (across the focal plane) against
# between visits. They answer different questions, so the ratio is worth seeing.
comparison = []
for noll in NOLL:
    between = nmad(visits[f'z{noll}_intra'] - visits[f'z{noll}_extra'])
    within = float(np.median(visits[f'dz{noll}_donut_nmad']))
    comparison.append({'noll': noll, 'within_visit_um': within,
                       'between_visit_um': between,
                       'ratio': within / between if between else np.nan})
scatter_comparison = pd.DataFrame(comparison).sort_values('within_visit_um',
                                                          ascending=False)

print('Robust scatter of the intra - extra difference, um of wavefront.')
print('within_visit: across the focal plane in one visit (per-donut).')
print('between_visit: visit to visit (per-visit medians).')
print(scatter_comparison.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

figure, axis = plt.subplots(figsize=(6.4, 5.6), layout='constrained')
axis.plot(scatter_comparison['between_visit_um'], scatter_comparison['within_visit_um'],
          'o', ms=4, color='tab:blue')
for _, row in scatter_comparison.iterrows():
    axis.annotate(f'Z{int(row.noll)}',
                  (row['between_visit_um'], row['within_visit_um']),
                  fontsize=6, xytext=(3, 3), textcoords='offset points')
limits = [0, max(scatter_comparison['between_visit_um'].max(),
                 scatter_comparison['within_visit_um'].max()) * 1.1]
axis.plot(limits, limits, '--', color='k', lw=0.8, label='1:1')
axis.set_xlabel('between-visit nMAD of the difference [um wf]')
axis.set_ylabel('within-visit nMAD of the difference [um wf]')
axis.set_title('Where the scatter lives: across the focal plane vs visit to visit',
               fontsize=9)
axis.legend(fontsize=7)
save(figure)
""")

md("""
### 3d. The within-visit nMAD excursions are a single common effect

The within-visit nMAD of dZ5 and dZ6 sit at a typical **0.26 and 0.25 um of wavefront** -
about what normal donut-to-donut fit covariance would give - but both excur to
**0.61 um of wavefront** on some visits, and **they excur on the same visits**
(Aaron, 2026-10-16). Other terms do the same.

Quantified below: this is not a per-term quirk but **one common per-visit scale factor**
that inflates every term's within-visit scatter together. It tracks the intra-focal blur
and the number of surviving donuts, both independently, which is what measurement noise
rather than wavefront structure would do.
""")

code("""
nmad_by_noll = pd.DataFrame(
    {noll: visits[f'dz{noll}_donut_nmad'].to_numpy() for noll in NOLL},
    index=visits.index)

print('within-visit nMAD of the per-donut difference [um of wavefront]:')
print(f'{"noll":>5} {"median":>8} {"95th pct":>9} {"max":>8} {"max/median":>11}')
for noll in NOLL:
    column = nmad_by_noll[noll]
    print(f'{noll:>5} {column.median():>8.4f} {column.quantile(0.95):>9.4f} '
          f'{column.max():>8.4f} {column.max() / column.median():>11.2f}')

# Pairwise correlation of the excursions, Aaron's observation first.
pairs = [(5, 6), (6, 11), (5, 11), (6, 7), (5, 8), (5, 7), (5, 4), (4, 7)]
print()
print('Spearman rho between the within-visit nMAD series, '
      f'n = {len(nmad_by_noll)} visits:')
for first, second in pairs:
    rho = spearmanr(nmad_by_noll[first], nmad_by_noll[second]).statistic
    print(f'  Z{first} vs Z{second}: {rho:+.4f}')

all_pairs = [spearmanr(nmad_by_noll[a], nmad_by_noll[b]).statistic
             for i, a in enumerate(NOLL) for b in NOLL[i + 1:]]
print(f'median over all {len(all_pairs)} term pairs: {np.median(all_pairs):+.4f}')
""")

code("""
# One common scale factor? Normalize each term by its own median, take logs so a
# multiplicative factor becomes additive, and look at the leading principal component.
normalized = nmad_by_noll / nmad_by_noll.median()
log_normalized = np.log(normalized.to_numpy())
centred = log_normalized - log_normalized.mean(axis=0)

_, singular_values, components = np.linalg.svd(centred, full_matrices=False)
variance_fraction = singular_values ** 2 / np.sum(singular_values ** 2)
scores = centred @ components[0] * np.sign(np.median(components[0]))
visits['nmad_common_factor'] = scores

print('variance fraction of the log within-visit nMAD principal components '
      '(dimensionless):')
for index in range(4):
    print(f'  PC{index + 1}: {variance_fraction[index]:.3f}')
same_sign = bool(np.all(components[0] > 0) or np.all(components[0] < 0))
print(f'PC1 loadings all the same sign: {same_sign}')
print()
print('what the common factor tracks (Spearman rho, '
      f'n = {len(visits)} visits):')
for column, label, unit in [('blur_intra', 'intra-focal blur', 'arcsec'),
                            ('blur_extra', 'extra-focal blur', 'arcsec'),
                            ('n_donuts', 'surviving donut pairs', 'dimensionless')]:
    rho = spearmanr(visits['nmad_common_factor'], visits[column]).statistic
    print(f'  {label} [{unit}]: {rho:+.4f}')


def partial_spearman(target, predictor, controls):
    \"\"\"Spearman rho of two variables after regressing both on the controls.\"\"\"
    design = sm.add_constant(np.asarray(controls, dtype=float))
    target_residual = (np.asarray(target, dtype=float)
                       - sm.OLS(np.asarray(target, dtype=float), design)
                       .fit().fittedvalues)
    predictor_residual = (np.asarray(predictor, dtype=float)
                          - sm.OLS(np.asarray(predictor, dtype=float), design)
                          .fit().fittedvalues)
    return spearmanr(target_residual, predictor_residual).statistic


print()
print('the two are nearly independent of each other '
      f'(blur_intra vs n_donuts rho '
      f'{spearmanr(visits.blur_intra, visits.n_donuts).statistic:+.4f}), and both '
      'survive controlling for the other:')
print('  common vs intra-focal blur, given n_donuts:   '
      f'{partial_spearman(visits.nmad_common_factor, visits.blur_intra, visits[["n_donuts"]]):+.4f}')
print('  common vs surviving pairs, given intra blur:  '
      f'{partial_spearman(visits.nmad_common_factor, visits.n_donuts, visits[["blur_intra"]]):+.4f}')
""")

code("""
figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), layout='constrained')

# Aaron's observation, drawn: Z5 and Z6 excurse together.
axis = axes[0][0]
draw_nights(axis, EDGES)
axis.plot(visits['ordinal'], nmad_by_noll[5], '.', ms=2.0, color='tab:blue',
          label='Z5')
axis.plot(visits['ordinal'], nmad_by_noll[6], '.', ms=2.0, color='tab:red',
          label='Z6')
axis.set_xlabel('FAM visit ordinal')
axis.set_ylabel('within-visit nMAD [um wf]')
axis.set_title(f'Z5 and Z6 excurse on the same visits: rho '
               f'{spearmanr(nmad_by_noll[5], nmad_by_noll[6]).statistic:+.3f}',
               fontsize=8)
axis.legend(fontsize=6, markerscale=3)

axis = axes[0][1]
axis.plot(nmad_by_noll[5], nmad_by_noll[6], '.', ms=2.5, color='tab:purple')
axis.set_xlabel('within-visit nMAD of dZ5 [um wf]')
axis.set_ylabel('within-visit nMAD of dZ6 [um wf]')
axis.set_title(f'medians {nmad_by_noll[5].median():.3f} and '
               f'{nmad_by_noll[6].median():.3f}, maxima '
               f'{nmad_by_noll[5].max():.3f} and {nmad_by_noll[6].max():.3f} um wf',
               fontsize=8)

# The common factor and its two drivers.
axis = axes[1][0]
axis.bar(np.arange(len(NOLL)), components[0] * np.sign(np.median(components[0])),
         color='tab:green', alpha=0.85)
axis.set_xticks(np.arange(len(NOLL)))
axis.set_xticklabels([f'{n}' for n in NOLL], fontsize=5, rotation=90)
axis.set_xlabel('Noll index')
axis.set_ylabel('PC1 loading (dimensionless)')
axis.set_title(f'One common factor: PC1 of the log nMAD carries '
               f'{variance_fraction[0]:.1%} of the variance,\\nall loadings the same '
               f'sign', fontsize=8)

axis = axes[1][1]
scatter_by_night(axis, visits, visits['blur_intra'], visits['nmad_common_factor'],
                 'intra-focal blur [arcsec]',
                 'common nMAD factor (dimensionless, log scale)')
axis.set_title(f'the factor tracks the intra-focal blur: rho '
               f'{spearmanr(visits.nmad_common_factor, visits.blur_intra).statistic:+.3f}'
               f'\\nand anti-tracks surviving pairs: rho '
               f'{spearmanr(visits.nmad_common_factor, visits.n_donuts).statistic:+.3f}',
               fontsize=8)

figure.suptitle('Within-visit nMAD excursions: one common per-visit scale factor',
                fontsize=9)
save(figure)
""")

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
against the current residual, and a **variance inflation factor (VIF)** so a step that
nearly repeats an earlier variable is visible rather than silent.

VIF is `1 / (1 - R^2)` (dimensionless), where `R^2` comes from regressing the newly added
predictor on those already chosen. It is the factor by which collinearity inflates the
variance of that predictor's fitted coefficient. VIF = 1 means orthogonal to the others;
VIF = 10 is the conventional warning line; VIF = 80 means the variable is 98.75 per cent
predictable from the others, so its coefficient is barely identified even though adding it
still reduced the residual scatter. It is reported as a diagnostic, not used as a gate.
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

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout='constrained',
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

md("""
### Each difference against its leading thermal variable, and against the full model

Two views per target. First the difference against the single leading thermal variable,
which is the one-variable picture the selection starts from. Then the difference against
the **fitted combination** of all selected variables - the model prediction on the x axis,
so a perfect model would put the points on the 1:1 line.
""")

code("""
def model_prediction(label):
    \"\"\"Fitted thermal model values for one target, on the visit index.

    Parameters
    ----------
    label : `str`
        Target key in `selections`.

    Returns
    -------
    prediction : `pandas.Series`
        The Huber fit's prediction, in the target's unit, NaN where telemetry is
        incomplete.
    \"\"\"
    entry = selections[label]
    used = pd.concat([entry['target'].rename('_t'), candidates], axis=1).dropna().index
    prediction = pd.Series(np.nan, index=joined.index)
    prediction.loc[used] = entry['fit'].fittedvalues
    return prediction


PER_ROW_THERMAL = 3
target_labels = list(selections)

for start in range(0, len(target_labels), PER_ROW_THERMAL):
    block = target_labels[start:start + PER_ROW_THERMAL]
    figure, axes = plt.subplots(2, len(block), figsize=(3.6 * len(block), 6.2),
                                squeeze=False, layout='constrained')
    for column, label in enumerate(block):
        entry = selections[label]
        target, unit, steps = entry['target'], entry['unit'], entry['steps']
        leading = steps['variable'].iloc[0]

        # Top row: against the leading thermal variable alone.
        axis = axes[0][column]
        scatter_by_night(axis, joined, candidates[leading], target,
                         f'{leading}', f'{label} [{unit}]')
        own = steps['rho_own'].iloc[0]
        axis.set_title(f'{label} vs {leading}\\nrho {own:+.3f}, '
                       f'step 1 cuts nMAD by {100 * steps.gain_frac.iloc[0]:.1f} pct',
                       fontsize=7)

        # Bottom row: against the full fitted combination, with the 1:1 line.
        axis = axes[1][column]
        prediction = model_prediction(label)
        scatter_by_night(axis, joined, prediction, target,
                         f'model of {len(steps)} thermal variables [{unit}]',
                         f'{label} [{unit}]')
        finite = np.isfinite(prediction) & np.isfinite(target)
        limits = [float(np.nanmin(prediction[finite])),
                  float(np.nanmax(prediction[finite]))]
        axis.plot(limits, limits, '--', color='k', lw=0.8, label='1:1')
        axis.legend(fontsize=6)
        rho = spearmanr(prediction[finite], target[finite]).statistic
        residual_scatter = nmad(target - prediction)
        axis.set_title(f'{label} vs the full thermal model\\n'
                       f'rho {rho:+.3f}, residual nMAD {residual_scatter:.4f} {unit}',
                       fontsize=7)

    figure.suptitle('Each difference against its leading thermal variable (top) and '
                    'against the full fitted model (bottom)', fontsize=9)
    save(figure)
""")

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

    # `layout='constrained'` with a taller figure: the default layout let the lower
    # row's x-axis labels run into the upper row's titles.
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 7.4), layout='constrained')
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
