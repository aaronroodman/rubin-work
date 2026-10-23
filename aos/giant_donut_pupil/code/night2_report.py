"""Night 2 report: ``night2_<day_obs>.pdf`` and the comparison tables, from run tables only.

Per night run (`night2_pipeline.py`) and per group of exposures (defocus and, on 20260127, the
rotator group), the pass 6 quantities against night 1 (20251023, run ``20261023_pupil6``,
``isr_phys`` and the step 4b and step 5 tables), each as median, nMAD and error on the median
(1.2533 nMAD / sqrt(n)) over sides:

- spider translation dy, dx and rotation (OCS image, pixels and deg; `pass5_analysis`
  translation fit, fit-a center);
- m = 1 amplitude of the shadow azimuth residual (deg);
- inner- and outer-edge circle dy (OCS, data − model, pixels);
- residual rotation δ with physical_rotator_angle (deg), raw and divided by the night 1
  sim recovery (0.76 extra);
- inner/outer ratio excess (dimensionless);
- the v4.02s silhouette inner-edge miss at ψ 160–200 deg (extra), 1.6–1.725 deg (pixels),
  where the outer band has sides;
- the free vane width (mm, pupil) on the central band.

Writes ``comparison.parquet``/``.csv`` (one row per night, group and quantity) and the PDF.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/night2_report.py --night 20260127
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_analysis as pa  # noqa: E402
import pass5_common as pc  # noqa: E402

from rubinwork.common.utils import nmad  # noqa: E402

OUTPUT = pathlib.Path(__file__).resolve().parents[1] / 'output'
NIGHT1 = OUTPUT / '20261023_pupil6'
RECOVERY = {'extra': 0.76, 'intra': 0.86}
PSI_RANGES = {'intra': (330.0, 30.0), 'extra': (160.0, 200.0)}
PSI_BAND_DEG = (1.6, 1.725)
QUANTITIES = (
    ('dy_pix', 'spider translation dy, OCS image (pixels)'),
    ('dx_pix', 'spider translation dx, OCS image (pixels)'),
    ('rot_deg', 'spider pattern rotation, translation fit (deg)'),
    ('m1_amp_deg', 'spider m = 1 amplitude (deg)'),
    ('inner_dy_pix', 'inner-edge circle dy, data − model, OCS (pixels)'),
    ('outer_dy_pix', 'outer-edge circle dy, data − model, OCS (pixels)'),
    ('delta_deg', 'residual rotation δ, physical_rotator_angle (deg)'),
    ('delta_corr_deg', 'δ / sim recovery (deg)'),
    ('ratio_excess', 'inner/outer ratio excess (dimensionless)'),
    ('sil_miss_pix', 'v4.02s silhouette inner-edge miss, ψ range, 1.6-1.725 deg (pixels)'),
    ('vane_width_mm', 'free vane width, central band (mm, pupil)'),
    ('chi2_dof_c', 'fit c chi2/dof (dof 3.8e5-3.0e6)'),
)


def err_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.2533 * nmad(x) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(median=np.nan, nmad=np.nan, err=np.nan, n=0)
    return dict(median=float(np.median(x)), nmad=nmad(x) if len(x) > 1 else np.nan,
                err=err_median(x), n=len(x))


def _in_range(psi, lo, hi):
    return (psi >= lo) | (psi < hi) if lo > hi else (psi >= lo) & (psi < hi)


def per_side(cfg_dir, sides, cfg_s=None):
    """One row per side: every quantity of `QUANTITIES` that is per side."""
    f = pd.read_parquet(cfg_dir / 'fits.parquet')
    sp = pd.read_parquet(cfg_dir / 'spiders.parquet')
    tf = pa.translation_fit(sp)
    tf = tf[tf.centre == 'fit_a'].set_index('key')
    eo = pd.read_parquet(cfg_dir / 'centre_edge_offsets.parquet')
    eo = eo[eo.centre == 'fit_a']
    out = f[['key', 'side', 'fit_ok', 'chi2_dof', 'ratio_excess']].rename(
        columns={'chi2_dof': 'chi2_dof_c'}).set_index('key')
    for q in ('dy_pix', 'dx_pix', 'rot_deg'):
        out[q] = tf[q].reindex(out.index)
    for edge in ('inner', 'outer'):
        out[f'{edge}_dy_pix'] = eo[eo.edge == edge].set_index('key').dy_ocs_pix.reindex(out.index)
    m1 = []
    for key, g in pa.found(sp).groupby('key'):
        if len(g) >= pa.MIN_SHADOWS:
            m1.append((key, pc.m1_fit(g.az_half_ocs_deg_dmm, g.az_half_ocs_deg_model)['amp']))
    out['m1_amp_deg'] = pd.Series(dict(m1)).reindex(out.index)
    if (cfg_dir / 'rotation_fit.parquet').exists():
        rf = pd.read_parquet(cfg_dir / 'rotation_fit.parquet').set_index('key')
        out['delta_deg'] = rf.delta_best_deg.reindex(out.index)
        out['delta_corr_deg'] = out.delta_deg / out.side.map(RECOVERY)
    out = out.join(sides.set_index('key')[[c for c in ('visit', 'band', 'sensor_class', 'field_r_deg',
                                                       'field_az_ocs_deg', 'rtp_deg', 'step_mm')
                                            if c in sides]], how='left')
    if cfg_s is not None:
        out['sil_miss_pix'] = silhouette_miss(cfg_s, sides).reindex(out.index)
    return out[out.fit_ok].reset_index()


def silhouette_miss(cfg_dir, sides):
    """Per side, median data − model inner edge in the ψ range (pixels), 1.6-1.725 deg science."""
    b = pd.read_parquet(cfg_dir / 'bins.parquet')
    s = sides.set_index('key')
    b = b[b.key.isin(s.index)].copy()
    for c in ('field_az_ocs_deg', 'field_r_deg', 'sensor_class'):
        b[c] = b.key.map(s[c])
    b = b[(b.sensor_class == 'science') & b.field_r_deg.between(*PSI_BAND_DEG) & ~b.spider
          & ~b.vignetted]
    b['psi_deg'] = np.mod(b.az_ocs_deg - (b.field_az_ocs_deg + 180.0), 360.0)
    b['miss'] = b.r_inner_data_pix - b.r_inner_c_pix
    out = {}
    for key, g in b.groupby('key'):
        lo, hi = PSI_RANGES[g.side.iloc[0]]
        x = g[_in_range(g.psi_deg, lo, hi)].miss.dropna()
        if len(x):
            out[key] = float(x.median())
    return pd.Series(out, dtype=float)


def night1():
    """Night 1 per side, from the pass 6 run (isr_phys, isr_phys_v402s, vane widths)."""
    s = pd.read_parquet(NIGHT1 / 'isr_phys' / 'sides.parquet')
    t = per_side(NIGHT1 / 'isr_phys', s, NIGHT1 / 'isr_phys_v402s')
    t['group'] = 'night 1: 20251023 r, 4 mm, rtp 0.0-0.15 deg'
    w = pd.read_parquet(NIGHT1 / 'vane_width_data.parquet').set_index('key')
    t['vane_width_mm'] = t.key.map(w.width_best_mm)
    return t


def groups(run, night, t):
    """Group labels of the night's sides: defocus and, for T680, the rotator group."""
    e = pd.read_parquet(run / 'exposures.parquet').set_index('visit_id')
    band = e.band.iloc[0]
    lab = []
    for r in t.itertuples():
        x = e.loc[r.visit]
        g = f'{night} {band}, {abs(x.step_mm):.0f} mm'
        if night == 20260127 and abs(x.step_mm) == 4.0:
            g += f', rtp {"+57" if x.rtp_deg > 30 else "-47" if x.rtp_deg < -30 else "-2..-9"} deg'
        if night == 20251027:
            g += f', M2 dx {x.d_dof1:+.0f} µm'
        if night == 20250920:
            g += f', M2 dx {x.d_dof1:+.0f} µm'
        lab.append(g)
    return lab


def load_night(run, night):
    s = pd.read_parquet(run / 'sides.parquet')
    cfg_s = run / 'v402s' if (run / 'v402s' / 'bins.parquet').exists() else None
    t = per_side(run / 'v402g', s, cfg_s)
    if (run / 'vane_width_data.parquet').exists():
        w = pd.read_parquet(run / 'vane_width_data.parquet')
        if len(w):
            t['vane_width_mm'] = t.key.map(w.set_index('key').width_best_mm)
    t['group'] = groups(run, night, t)
    return t


def comparison(tables):
    rows = []
    for t in tables:
        for (g, side), h in t.groupby(['group', 'side']):
            for q, label in QUANTITIES:
                if q not in h:
                    continue
                st = stat(h[q])
                rows.append(dict(group=g, side=side, quantity=q, label=label, **st))
    return pd.DataFrame(rows)


def text_page(pdf, title, text, fontsize=7):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0, 1, title + '\n\n' + text, family='monospace', fontsize=fontsize, va='top')
    pdf.savefig(fig)
    plt.close(fig)


def comparison_text(c):
    lines = []
    for q, label in QUANTITIES:
        g = c[(c.quantity == q) & (c.side == 'extra')]
        if not len(g):
            continue
        lines.append(f'{label}')
        for r in g.itertuples():
            lines.append(f'  {r.group:46s} median {r.median:+8.3f}  nMAD {r.nmad:7.3f}  '
                         f'err {r.err:6.3f}  n {r.n}')
    return '\n'.join(lines)


def page_quantity_bars(pdf, c, title):
    qs = [q for q, _ in QUANTITIES if q in set(c.quantity)]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, q in zip(axes.flat, qs):
        g = c[(c.quantity == q) & (c.side == 'extra')]
        x = np.arange(len(g))
        ax.errorbar(x, g['median'], yerr=g['err'], fmt='o', color='tab:blue', capsize=3)
        n1 = g[g.group.str.startswith('night 1')]
        if len(n1):
            ax.axhspan(n1['median'].iloc[0] - n1['err'].iloc[0], n1['median'].iloc[0]
                       + n1['err'].iloc[0], color='0.85')
        ax.axhline(0, color='0.5', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('night 1: ', 'N1 ') for s in g.group], rotation=35,
                           ha='right', fontsize=6)
        ax.set_title(dict(QUANTITIES)[q], fontsize=8)
    for ax in axes.flat[len(qs):]:
        ax.axis('off')
    fig.suptitle(title + ' (extra-focal; median and error on the median over sides; grey: '
                 'night 1 ± its error)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_field(pdf, tables, q, label):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for t in tables:
        for gname, g in t[t.side == 'extra'].groupby('group'):
            axes[0].plot(g.field_r_deg, g[q], 'o', ms=3, label=gname)
            axes[1].plot(g.field_az_ocs_deg, g[q], 'o', ms=3)
    axes[0].set_xlabel('field radius (deg)')
    axes[1].set_xlabel('field azimuth, OCS (deg)')
    for ax in axes:
        ax.set_ylabel(label)
        ax.axhline(0, color='0.5', lw=0.5)
    axes[0].legend(fontsize=6)
    fig.suptitle(f'{label} per side, extra-focal', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def page_rotator(pdf, t, n1):
    """dy and dx against rtp: a pupil-fixed (telescope) cause stays in OCS; a camera-fixed one
    turns with the rotator."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    both = pd.concat([n1.assign(night='20251023'), t.assign(night='night 2')])
    both = both[both.side == 'extra']
    for name, g in both.groupby('night'):
        for ax, q in zip(axes, ('dy_pix', 'dx_pix', 'inner_dy_pix')):
            ax.plot(g.rtp_deg, g[q], 'o', ms=2.5, alpha=0.6, label=name)
    for ax, q in zip(axes, ('dy_pix', 'dx_pix', 'inner_dy_pix')):
        ax.set_xlabel('physical_rotator_angle (deg)')
        ax.set_ylabel(dict(QUANTITIES)[q])
        ax.axhline(0, color='0.5', lw=0.5)
    axes[0].legend(fontsize=7)
    fig.suptitle('OCS-image offsets against rotator angle, extra-focal (a camera-fixed cause '
                 'would rotate its OCS direction with rtp)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=85)
    plt.close(fig)


def rotator_fit(t):
    """Huber fit of the OCS (dx, dy) per side to a fixed OCS vector plus a CCS-fixed vector
    rotated by rtp: [dx, dy]_OCS = a + R(rtp) b (pixels)."""
    import statsmodels.api as sm
    g = t[(t.side == 'extra')].dropna(subset=['dx_pix', 'dy_pix', 'rtp_deg'])
    if g.rtp_deg.max() - g.rtp_deg.min() < 20:
        return None
    r = np.radians(g.rtp_deg.to_numpy())
    c, s = np.cos(r), np.sin(r)
    one, zero = np.ones_like(c), np.zeros_like(c)
    X = np.vstack([np.column_stack([one, zero, c, -s]), np.column_stack([zero, one, s, c])])
    y = np.concatenate([g.dx_pix.to_numpy(), g.dy_pix.to_numpy()])
    f = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    names = ('ocs_dx_pix', 'ocs_dy_pix', 'ccs_dx_pix', 'ccs_dy_pix')
    out = {n: float(v) for n, v in zip(names, f.params)}
    out.update({f'{n}_err': float(v) for n, v in zip(names, f.bse)})
    out['n_sides'] = len(g)
    return out


def m2dx_fit(t):
    """dy, dx and inner-edge dy against the M2 dx step (pixels per mm), Huber."""
    import statsmodels.api as sm
    rows = []
    for q in ('dx_pix', 'dy_pix', 'inner_dy_pix', 'outer_dy_pix', 'rot_deg'):
        g = t[t.side == 'extra'].dropna(subset=[q, 'm2_dx_mm'])
        if g.m2_dx_mm.nunique() < 3:
            continue
        X = sm.add_constant(g.m2_dx_mm.to_numpy())
        f = sm.RLM(g[q].to_numpy(), X, M=sm.robust.norms.HuberT()).fit()
        rows.append(dict(quantity=q, slope_per_mm=float(f.params[1]), slope_err=float(f.bse[1]),
                         intercept=float(f.params[0]), intercept_err=float(f.bse[0]), n_sides=len(g),
                         n_steps=int(g.m2_dx_mm.nunique())))
    return pd.DataFrame(rows)


def thermal_text(runs):
    lines = []
    for night, run in runs:
        p = run / 'thermal.parquet'
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        for ref, g in t.groupby('is_reference'):
            lines.append(
                f'{night} {"reference" if ref else "exposures"} (n {len(g)}): camera ambient '
                f'{g.cam_AmbAirtemp.median():.2f} °C, camera body − ambient '
                f'{g.cam_body_minus_amb_c.median():+.2f} K, L1 − ambient {g.cam_l1_minus_amb_c.median():+.2f} K, '
                f'dome (camera Y−) {g.cam_DomeYMinusTemp.median():.2f} °C, M1M3 glass '
                f'{g.m1m3_mean_temp_c.median():.2f} °C, M1M3 − ambient {g.m1m3_minus_amb_c.median():+.2f} K, '
                f'M1M3 z gradient {g.m1m3_z_gradient_c_per_m.median():+.3f} K/m, wind '
                f'{g.wind_speed_ms.median():.1f} m/s')
    return '\n'.join(lines)


def reference_text(runs):
    lines = []
    for night, run in runs:
        p = run / 'reference.parquet'
        if p.exists():
            for r in pd.read_parquet(p).itertuples():
                lines.append(f'{night} reference {r.visit_id}: PSF FWHM {r.psf_fwhm_arcsec:.2f} arcsec '
                             f'(ConsDB psf_sigma_median), optical-state residual RMS '
                             f'{r.resid_rms_um:.3f} µm of wavefront, corner FWHM contribution '
                             f'{r.fwhm_cwfs_arcsec:.2f} arcsec, rtp {r.physical_rotator_angle:+.2f} deg')
    return '\n'.join(lines)


def report(run, night):
    run = pathlib.Path(run)
    n1 = night1()
    t = load_night(run, night)
    e = pd.read_parquet(run / 'exposures.parquet').set_index('visit_id')
    t['m2_dx_mm'] = t.visit.map(e.d_dof1) / 1e3
    c = comparison([n1, t])
    c.to_parquet(run / 'comparison.parquet')
    c.to_csv(run / 'comparison.csv', index=False, float_format='%.5g')
    t.to_parquet(run / 'per_side.parquet')
    t.to_csv(run / 'per_side.csv', index=False, float_format='%.5g')
    extra = {}
    rf = rotator_fit(pd.concat([n1, t[t.group.str.contains('4 mm')]]))
    if rf is not None:
        extra['rotator_fit'] = rf
    if night == 20251027:
        m2 = m2dx_fit(t)
        m2.to_csv(run / 'm2dx_response.csv', index=False, float_format='%.5g')
        extra['m2dx'] = m2.to_dict('records')
    (run / 'fits_extra.json').write_text(json.dumps(extra, indent=1))
    tally = pd.read_csv(run / 'tally.csv') if (run / 'tally.csv').exists() else None
    with PdfPages(run / f'night2_{night}.pdf') as pdf:
        head = (f'Run {run.name}. Exposures: {", ".join(str(v)[-5:] for v in e.index)} '
                f'(seq), band {e.band.iloc[0]}, defocus {sorted(set(e.step_mm))} mm.\n'
                'Setup as pupil pass 6: v4.02, intrinsicTA, fit c, rtp = physical_rotator_angle, '
                'full ISR (flat, CTI), guided tracer, robust edges; v4.02g with the δ grid, '
                'v4.02s for the silhouette check. Final selection rule, no hand labels.\n'
                'Per group: median, nMAD and error on the median (1.2533 nMAD / sqrt n) over sides; '
                'night 1 from run 20261023_pupil6.\n\n')
        if tally is not None:
            head += 'Selected donuts per exposure (donut counts):\n' + tally.to_string(index=False) + '\n\n'
        head += 'In-focus references:\n' + reference_text([(night, run)]) + '\n'
        head += 'Thermal context:\n' + thermal_text([(night, run)]) + '\n'
        if 'rotator_fit' in extra:
            r = extra['rotator_fit']
            head += ('\nSpider translation split into a telescope-fixed (OCS) and a camera-fixed '
                     '(CCS, turned by rtp) vector, Huber, extra-focal 4 mm sides of both nights '
                     f'(n {r["n_sides"]}):\n  OCS ({r["ocs_dx_pix"]:+.2f} ± {r["ocs_dx_pix_err"]:.2f}, '
                     f'{r["ocs_dy_pix"]:+.2f} ± {r["ocs_dy_pix_err"]:.2f}) pixels; CCS '
                     f'({r["ccs_dx_pix"]:+.2f} ± {r["ccs_dx_pix_err"]:.2f}, {r["ccs_dy_pix"]:+.2f} ± '
                     f'{r["ccs_dy_pix_err"]:.2f}) pixels\n')
        if 'm2dx' in extra:
            head += '\nResponse to the M2 dx step (Huber, per mm of M2 dx):\n'
            for r in extra['m2dx']:
                head += (f'  {r["quantity"]}: {r["slope_per_mm"]:+.3f} ± {r["slope_err"]:.3f} per mm '
                         f'(n {r["n_sides"]} sides, {r["n_steps"]} steps)\n')
        text_page(pdf, f'Night 2: {night}', head, 6.5)
        text_page(pdf, 'Comparison with night 1, extra-focal', comparison_text(c), 6)
        page_quantity_bars(pdf, c, f'{night} against 20251023')
        for q in ('dy_pix', 'inner_dy_pix', 'ratio_excess', 'vane_width_mm', 'delta_deg'):
            if q in t and t[q].notna().any():
                page_field(pdf, [n1, t], q, dict(QUANTITIES)[q])
        page_rotator(pdf, t, n1)
    print(c[c.side == 'extra'].pivot(index='quantity', columns='group', values='median').round(3)
          .to_string())
    print(json.dumps(extra, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--night', type=int, required=True)
    args = ap.parse_args()
    report(OUTPUT / f'20261023_night2_{args.night}', args.night)


if __name__ == '__main__':
    main()
