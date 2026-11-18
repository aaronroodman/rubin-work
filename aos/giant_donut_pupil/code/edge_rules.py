"""Edge rules R0-R3 with a per-edge spider cut, on the data and the fit c model of each side.

The hot rim beside the donut edges (the M1 inner-edge roll-off, `summary.md`) moves the
50 % crossing of the current finder. Four rules, applied the same way to data and model:

- **R0:** 50 % of the bin's flat top (median at 0.70-0.90 r_out), the `robust_edges`
  crossing; as now but with the per-edge spider cut.
- **R1:** 50 % of the local plateau, the median of the smoothed profile `PLATEAU_PIX`
  inside the edge (past the rim).
- **R2:** the radius of steepest slope of the profile smoothed over 2 pixels.
- **R3:** a blurred step b + P Φ(±(r − r0)/σ) fitted to the dark flank, from
  `FIT_DARK_PIX` outside the edge to where the profile first reaches `FIT_TOP_FRAC` of P;
  P is the R1 plateau, r0, σ and b are free.

Each rule's per-bin edges are cleaned as `robust_edges` does: a circle plus harmonics to
m = 4 (Huber), bins over 3 nMAD off are dropped; the nMAD is the rule's noise.

Spider cut, per edge: a bin is cut for an edge when a vane-shadow pixel of the fit c model
(vanes darken a lit pixel by more than 20 %) lies within `PAD_PIX` of that edge, with no
padding in azimuth. Vignetted and neighbour-masked bins are cut as before.

Rim contrast: the maximum of the smoothed profile within `RIM_PIX` on the bright side of
the R0 edge, over the R1 plateau (dimensionless).

Element labels are `psi_labels.labels_for_side` with no spider blanking, then blanked per
edge by the cut above.

Usage, from the repo root (one fit set per call; writes ``edge_rules_<run>_<fit_dir>``)::

    python aos/giant_donut_pupil/code/edge_rules.py --run 20261118_night2_20261107 \\
        --fit-dir v402s --workers 60

It writes ``bins.parquet`` and ``edge_rules.pdf`` (per-rule psi pages, the rim and width
pages, the tables) with ``rules_*.csv``.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import psi_labels as pl  # noqa: E402
import pupil_edges as pe  # noqa: E402
import robust_edges as rob  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
RULES = ('R0', 'R1', 'R2', 'R3')
PAD_PIX = 6.0
SHADOW_FRAC = 0.8
PLATEAU_PIX = (12.0, 25.0)
WINDOW_PIX = 12.0
SLOPE_PIX = 8.0
SLOPE_SMOOTH_STEPS = 4
FIT_DARK_PIX = 12.0
FIT_TOP_FRAC = 0.6
SIGMA_BOUNDS_PIX = (0.3, 8.0)
RIM_PIX = 10.0
FWHM_ARCSEC = 1.0
AZ = (np.arange(rob.NBIN) + 0.5) * 360 / rob.NBIN


def without_spider(params):
    return {k: v for k, v in params.items() if not k.startswith('Spider')}


def shadow_of(with_vanes, without_vanes):
    """Lit pixels the vanes darken by more than 1 - `SHADOW_FRAC`."""
    lit = without_vanes > 0.5 * np.median(without_vanes[without_vanes > 0])
    return lit & (with_vanes < SHADOW_FRAC * without_vanes)


def spider_per_edge(shadow, cx, cy, edge_r):
    """Bins where a shadow pixel lies within `PAD_PIX` of `edge_r` (pixels, per bin)."""
    n = shadow.shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - cx, yy - cy)
    phi = np.mod(np.degrees(np.arctan2(yy - cy, xx - cx)), 360.0)
    b = np.minimum((phi / (360 / rob.NBIN)).astype(int), rob.NBIN - 1)
    near = shadow & np.isfinite(edge_r[b]) & (np.abs(r - edge_r[b]) <= PAD_PIX)
    hit = np.zeros(rob.NBIN, bool)
    hit[np.unique(b[near])] = True
    return hit


def _mean_steps(p, k):
    ok = np.isfinite(p)
    w = np.ones(k)
    num = np.convolve(np.where(ok, p, 0.0), w, mode='same')
    den = np.convolve(ok.astype(float), w, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _plateau(ps, rad, e, direction):
    lo, hi = PLATEAU_PIX
    w = (rad > e + lo) & (rad < e + hi) if direction > 0 else (rad > e - hi) & (rad < e - lo)
    v = ps[w]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= 5 else np.nan


def _r1(ps, rad, e, P, direction):
    r0, sg = rob._crossings(ps, rad, 0.5 * P, e - WINDOW_PIX, e + WINDOW_PIX)
    r0 = r0[sg == direction]
    return float(r0[np.argmin(np.abs(r0 - e))]) if len(r0) else np.nan


def _r2(p, rad, e, direction):
    d = np.gradient(_mean_steps(p, SLOPE_SMOOTH_STEPS), rob.DR_PIX) * direction
    w = np.nonzero((rad > e - SLOPE_PIX) & (rad < e + SLOPE_PIX) & np.isfinite(d))[0]
    if len(w) < 5:
        return np.nan
    i = w[np.argmax(d[w])]
    if i in (w[0], w[-1]):
        return np.nan
    den = d[i - 1] - 2 * d[i] + d[i + 1]
    off = 0.5 * (d[i - 1] - d[i + 1]) / den if den < 0 else 0.0
    return float(rad[i] + np.clip(off, -1, 1) * rob.DR_PIX)


def _r3(p, ps, rad, cnt, noise, e, P, direction, x0):
    """Blurred-step fit to the dark flank; returns (r0, sigma) in pixels."""
    from scipy.optimize import least_squares
    from scipy.special import ndtr
    start = e - direction * FIT_DARK_PIX
    if direction > 0:
        up = np.nonzero((rad > start) & (rad < e + WINDOW_PIX) & (ps >= FIT_TOP_FRAC * P))[0]
        if not len(up):
            return np.nan, np.nan
        w = (rad >= start) & (rad <= rad[up[0]])
    else:
        up = np.nonzero((rad < start) & (rad > e - WINDOW_PIX) & (ps >= FIT_TOP_FRAC * P))[0]
        if not len(up):
            return np.nan, np.nan
        w = (rad <= start) & (rad >= rad[up[-1]])
    w &= np.isfinite(p) & (cnt > 0)
    if w.sum() < 8:
        return np.nan, np.nan
    r, y = rad[w], p[w]
    s = np.sqrt(noise ** 2 + np.clip(y, 0, None)) / np.sqrt(cnt[w])

    def res(x):
        return (x[2] + P * ndtr(direction * (r - x[0]) / x[1]) - y) / s

    x0 = [x0 if np.isfinite(x0) else e, 2.0, 0.0]
    lo = [e - WINDOW_PIX, SIGMA_BOUNDS_PIX[0], -np.inf]
    hi = [e + WINDOW_PIX, SIGMA_BOUNDS_PIX[1], np.inf]
    x0[0] = float(np.clip(x0[0], lo[0] + 1e-3, hi[0] - 1e-3))
    try:
        f = least_squares(res, x0, bounds=(lo, hi), loss='soft_l1', f_scale=3.0)
    except ValueError:
        return np.nan, np.nan
    return (float(f.x[0]), float(f.x[1])) if f.success else (np.nan, np.nan)


def _clean(r, keep):
    """Huber harmonic fit and 3 nMAD clip, as `robust_edges`; returns (edges, nMAD)."""
    keep = keep & np.isfinite(r)
    model, sigma = rob.harmonic_fit(r, AZ, keep)
    if np.isfinite(sigma) and sigma > 0:
        keep &= ~(np.abs(r - model) > rob.CLIP_SIGMA * max(sigma, 0.1))
        model, sigma = rob.harmonic_fit(r, AZ, keep)
    return np.where(keep, r, np.nan), sigma


def rules_for_image(image, valid, cx, cy, r_out, noise, flag_in, flag_out):
    """Edges per bin and rule for one image (pixels); also the R3 width and rim contrast."""
    res0 = rob.robust_edges(image, valid, cx, cy, r_out, noise, flag_in, flag_out, pad_bins=0)
    prof, rad, top, top_snr, top_n, _, cnt = rob.profiles(image, valid, cx, cy, r_out, noise)
    ok_top = (top_n >= rob.MIN_TOP_PIX) & np.isfinite(top) & (top_snr >= rob.MIN_TOP_SNR)
    out = {}
    for edge, flag, direction in (('inner', flag_in, +1), ('outer', flag_out, -1)):
        e_exp = res0[f'fit_{edge}']
        raw = {k: np.full(rob.NBIN, np.nan) for k in ('R1', 'R2', 'R3', 'sig3', 'P', 'rim')}
        for b in np.nonzero(ok_top & ~flag & np.isfinite(e_exp))[0]:
            p = prof[b]
            ps = rob._smooth(p)
            e = e_exp[b]
            P = _plateau(ps, rad, e, direction)
            if not np.isfinite(P) or P <= 0:
                continue
            raw['P'][b] = P
            r1 = _r1(ps, rad, e, P, direction)
            raw['R1'][b] = r1
            raw['R2'][b] = _r2(p, rad, e, direction)
            raw['R3'][b], raw['sig3'][b] = _r3(p, ps, rad, cnt[b], noise, e, P, direction, r1)
            e0 = res0[f'r_{edge}_raw'][b]
            if np.isfinite(e0):
                w = (rad > e0) & (rad < e0 + RIM_PIX) if direction > 0 else \
                    (rad < e0) & (rad > e0 - RIM_PIX)
                if np.isfinite(ps[w]).any():
                    raw['rim'][b] = np.nanmax(ps[w]) / P
        out[f'r_{edge}_R0'] = res0[f'r_{edge}']
        out[f'nmad_{edge}_R0'] = res0[f'sigma_{edge}']
        for k in ('R1', 'R2', 'R3'):
            out[f'r_{edge}_{k}'], out[f'nmad_{edge}_{k}'] = _clean(raw[k], ok_top & ~flag)
        out[f'sigma_{edge}_R3'] = np.where(np.isfinite(out[f'r_{edge}_R3']), raw['sig3'], np.nan)
        out[f'plateau_{edge}'] = raw['P']
        out[f'rim_{edge}'] = raw['rim']
    return out


def side_task(task):
    """All rules, the per-edge cut and the labels for one side; a per-bin frame."""
    row, g, run = task
    import psi_gallery as pg
    t0 = time.time()
    part = pathlib.Path(_PART_DIR) / f'{row.key.replace(":", "_")}.parquet'
    if part.exists():
        return pd.read_parquet(part)
    img, msk = pg.stamp(pl.STUDY_OUT / run, row.key)
    valid = ~msk
    params = pl.mask_for(row)
    model = pl.fit_model(row, params, FWHM_ARCSEC)
    bare = pl.fit_model(row, without_spider(params), FWHM_ARCSEC)
    cx, cy, noise, r_out = float(row.cx_a), float(row.cy_a), float(row.noise_e), float(row.r_out_pix)
    _, _, m_in, m_out, _ = pe.edges(img, valid, cx, cy, r_out)
    vig = g.vignetted.to_numpy(bool)
    base_in, base_out = vig | m_in, vig | m_out
    m0 = rob.robust_edges(model, valid, cx, cy, r_out, noise, base_in, base_out, pad_bins=0)
    shadow = shadow_of(model, bare)
    sp_in = spider_per_edge(shadow, cx, cy, m0['fit_inner'])
    sp_out = spider_per_edge(shadow, cx, cy, m0['fit_outer'])
    f_in, f_out = base_in | sp_in, base_out | sp_out
    d = rules_for_image(img, valid, cx, cy, r_out, noise, f_in, f_out)
    m = rules_for_image(model, valid, cx, cy, r_out, noise, f_in, f_out)
    lab = pl.labels_for_side(row, row, params, (cx, cy),
                             (np.zeros(rob.NBIN, bool), vig, m_in, m_out), noise, FWHM_ARCSEC)
    out = pd.DataFrame(dict(key=row.key, side=row.side, bin=np.arange(rob.NBIN),
                            az_ccs_deg=g.az_ccs_deg.to_numpy(), az_ocs_deg=g.az_ocs_deg.to_numpy(),
                            spider_old=g.spider.to_numpy(bool), spider_inner=sp_in,
                            spider_outer=sp_out, cut_inner=f_in, cut_outer=f_out))
    for src, res in (('data', d), ('model', m)):
        for k, v in res.items():
            out[f'{k}_{src}'] = v
    for edge, f in (('inner', f_in), ('outer', f_out)):
        for c in (f'element_{edge}', f'second_{edge}'):
            out[c] = np.where(f, None, lab[c].to_numpy())
        for c in (f'margin_{edge}_pix', f'second_margin_{edge}_pix'):
            out[c] = np.where(f, np.nan, lab[c].to_numpy())
    out['seconds'] = time.time() - t0
    part.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(part)
    print(f'{row.key} {out.seconds.iloc[0]:.0f} s', flush=True)
    return out


_PART_DIR = None


def _init_worker(mask_run, s_mask_dir, part_dir):
    global _PART_DIR
    pl._init(mask_run, s_mask_dir)
    _PART_DIR = part_dir


def run_rules(run, fit_dir, workers, out=None, limit=None):
    import multiprocessing as mp
    from rubinwork.common import study_run
    if out is None:
        out = study_run(STUDY, f'edge_rules_{run}_{fit_dir}').path
    out = pathlib.Path(out)
    c, b, mask_run = pl._load(run, fit_dir)
    c = pl.select(c)
    if limit:
        c = c.head(limit)
    part_dir = str(out / 'parts')
    tasks = [(r, b[(b.key == r.key) & (b.side == r.side)].sort_values('bin'), run)
             for r in c.itertuples(index=False)]
    tasks = [(pd.Series(r._asdict()), g, run) for r, g, run in tasks]
    initargs = (*map(str, mask_run), part_dir)
    if workers > 1:
        with mp.get_context('spawn').Pool(min(workers, len(tasks)), initializer=_init_worker,
                                          initargs=initargs, maxtasksperchild=4) as pool:
            parts = pool.map(side_task, tasks, chunksize=1)
    else:
        _init_worker(*initargs)
        parts = [side_task(t) for t in tasks]
    bins = pd.concat(parts, ignore_index=True)
    geo = c[['key', 'side', 'sensor_class', 'field_r_deg', 'field_az_ocs_deg', 'rtp_deg']]
    bins = bins.merge(geo, on=['key', 'side'])
    bins.to_parquet(out / 'bins.parquet')
    (out / 'source.txt').write_text(f'run {run}\nfit_dir {fit_dir}\nfwhm {FWHM_ARCSEC}\n')
    print('done', out, len(c), 'sides', flush=True)
    return out


def rule_frame(bins, rule):
    """Bins in the columns `psi_edges_report.page_psi` reads, for one rule."""
    import psi_edges_report as per
    b = bins.copy()
    b['psi_deg'] = np.mod(b.az_ocs_deg - (b.field_az_ocs_deg + 180.0), 360.0)
    b['pbin'] = np.minimum((b.psi_deg / per.BIN_DEG).astype(int), per.NBIN - 1)
    for e in ('inner', 'outer'):
        b[f'r_{e}_data_pix'] = b[f'r_{e}_{rule}_data']
        b[f'r_{e}_c_pix'] = b[f'r_{e}_{rule}_model']
        b[f'{e}_dmm'] = b[f'r_{e}_data_pix'] - b[f'r_{e}_c_pix']
    b['ratio_dmm'] = b.r_inner_data_pix / b.r_outer_data_pix - b.r_inner_c_pix / b.r_outer_c_pix
    b['spider'] = False
    return b


def _page_rim(pdf, g, title):
    """Rim contrast and R3 step width against psi, data and model, median over sides."""
    import matplotlib.pyplot as plt
    import psi_edges_report as per
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    centers = (np.arange(per.NBIN) + 0.5) * per.BIN_DEG
    for k, edge in enumerate(('inner', 'outer')):
        for j, (q, ylab) in enumerate(((f'rim_{edge}', f'{edge} rim contrast (dimensionless;\n'
                                        'peak within 10 pixels / plateau)'),
                                       (f'sigma_{edge}_R3', f'{edge} R3 step width σ (pixels)'))):
            ax = axes[2 * k + j]
            for src, col, kw in (('data', 'k', dict(ls='', marker='.', ms=3)),
                                 ('model', 'tab:red', dict(ls='-', lw=1.0))):
                m = g.groupby('pbin')[f'{q}_{src}'].median().reindex(range(per.NBIN))
                ax.plot(centers, m.to_numpy(), color=col, label=f'{src}' if src == 'data'
                        else 'fit c model', **kw)
            ax.set_ylabel(ylab, fontsize=8)
            ax.grid(alpha=0.3)
            if k == 0 and j == 0:
                ax.legend(fontsize=8)
    axes[-1].set_xlabel('psi (deg; 0 toward the field center)')
    n = g[['key', 'side']].drop_duplicates().shape[0]
    fig.suptitle(f'{title}: {n} sides, median per 2 deg psi bin', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def report(out_dir):
    """Per-rule psi pages, the rim and width pages, and the comparison tables."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import psi_edges_report as per
    out_dir = pathlib.Path(out_dir)
    bins = pd.read_parquet(out_dir / 'bins.parquet')
    src = dict(l.split(' ', 1) for l in (out_dir / 'source.txt').read_text().splitlines())
    tables, noise = [], []
    with PdfPages(out_dir / 'edge_rules.pdf') as pdf:
        for rule in RULES:
            b = rule_frame(bins, rule)
            for group, gtitle in per.GROUPS:
                g = per.group_of(b, group)
                if g[['key', 'side']].drop_duplicates().shape[0] < 3:
                    continue
                cons = per.page_psi(pdf, g, f'{src["run"]} {src["fit_dir"]} {rule} {gtitle}')
                t = per.element_table(cons, gtitle)
                t['rule'] = rule
                tables.append(t)
                for e in ('inner', 'outer'):
                    s = g.groupby(['key', 'side'])[f'nmad_{e}_{rule}_data'].first()
                    noise.append(dict(rule=rule, group=gtitle, edge=e,
                                      nmad_median_pix=float(s.median()), n_sides=len(s),
                                      bins_per_side=float(g[f'r_{e}_data_pix'].notna().sum()
                                                          / len(s))))
        for group, gtitle in per.GROUPS:
            g = per.group_of(rule_frame(bins, 'R0'), group)
            if g[['key', 'side']].drop_duplicates().shape[0] >= 3:
                _page_rim(pdf, g, f'{src["run"]} {src["fit_dir"]} {gtitle}')
        if not tables:
            print('no group with 3 or more sides; no report', flush=True)
            return
        t = pd.concat(tables, ignore_index=True)
        t = t[t.n_bins >= 5]
        wide = t.pivot_table(index=['group', 'edge', 'element', 'psi_from_deg', 'psi_to_deg'],
                             columns='rule', values='dmm_median_pix', aggfunc='first')
        wide = wide.reset_index()
        nz = pd.DataFrame(noise)
        for tab, title in ((wide, 'median data − model per element range (pixels), by rule'),
                           (nz, 'per-side nMAD of the data edges about their harmonic fit '
                                '(pixels), and measured bins per side')):
            fig, ax = plt.subplots(figsize=(11, 0.6 + 0.22 * (len(tab) + 2)))
            ax.axis('off')
            show = tab.copy()
            for c in show.columns:
                if show[c].dtype.kind == 'f':
                    show[c] = show[c].map(lambda v: f'{v:+.2f}' if np.isfinite(v) else '')
            tb = ax.table(cellText=show.to_numpy(), colLabels=list(show.columns), loc='center')
            tb.auto_set_font_size(False)
            tb.set_fontsize(7)
            ax.set_title(f'{src["run"]} {src["fit_dir"]}: {title}', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    t.to_csv(out_dir / 'rules_elements.csv', index=False, float_format='%.4g')
    wide.to_csv(out_dir / 'rules_elements_wide.csv', index=False, float_format='%.4g')
    nz.to_csv(out_dir / 'rules_noise.csv', index=False, float_format='%.4g')
    print('wrote', out_dir / 'edge_rules.pdf', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--fit-dir', required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', default=None, help='output directory; a new study run by default')
    p.add_argument('--limit', type=int, default=None, help='first N sides only (a test)')
    p.add_argument('--report-only', default=None, help='an existing edge_rules directory')
    a = p.parse_args()
    if a.report_only:
        report(a.report_only)
        return
    report(run_rules(a.run, a.fit_dir, a.workers, a.out, a.limit))


if __name__ == '__main__':
    main()
