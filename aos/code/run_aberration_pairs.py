#!/usr/bin/env python3
"""Per-donut primary->secondary aberration-pair analysis (script port of
study_aberrationpairs.ipynb).

For each primary aberration and its next higher-radial-order partner at the
same (|m|, parity), this asks whether the secondary co-varies with the primary
and whether that co-variation depends on the primary amplitude.  It works on
the **per-donut single-Zernike** values (``zk_<coord>`` in donuts.parquet), not
on the per-visit Double-Zernike fits — a distinct data product from the DZ-to-DZ
correlation analysis in study_doublezernike / dz_plotting.

For each pair, donuts are split into quartiles by the primary value and an OLS
line + Pearson r is fit per quartile; the per-quartile slope/intercept/r/n are
written to a summary parquet and rendered as a 2x2 density page per pair.

Operates on one Phase-1 combined param_set:
    output/<ps>/donuts.parquet   per-donut zk_<coord> (streamed by row group)
    output/<ps>/visits.parquet   nollIndices (pupil-j ordering)
    output/<ps>/fits.parquet     bad-fit flag (per visit, optional)

Writes:
    <output-dir>/aberration_pairs.pdf
    <output-dir>/aberration_pairs_summary.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mi_config as mc

try:
    from common.zernike_names import NOLL_NAMES
except Exception:                                         # pragma: no cover
    NOLL_NAMES = {}

# Primary -> Secondary pupil-Noll pairs (low radial order -> next higher at the
# same |m|/parity): defocus->spherical, astig->2nd-astig, coma->2nd-coma,
# trefoil->2nd-trefoil, spherical->2nd-spherical, tetrafoil->2nd-tetrafoil.
ABERRATION_PAIRS = [
    (4, 11), (5, 13), (6, 12), (7, 17), (8, 16),
    (9, 19), (10, 18), (11, 22), (14, 26), (15, 25),
]


def bad_visit_set(fits_path, fit_prefix, programs=None):
    """(day_obs, seq_num) flagged bad by the DZ fit (or program-filtered)."""
    bad = set()
    if not Path(fits_path).exists():
        print(f'  (no fits parquet at {fits_path}; skipping bad-fit cut)')
        return bad
    ft = QTable.read(str(fits_path))
    bf = None
    for cand in (f'{fit_prefix}_bad_fit', 'bad_fit'):
        if cand in ft.colnames:
            bf = cand
            break
    mask = np.asarray(ft[bf]).astype(bool) if bf else np.zeros(len(ft), bool)
    if programs and 'science_program' in ft.colnames:
        sp = np.asarray(ft['science_program']).astype(str)
        mask = mask | ~np.array([s in set(programs) for s in sp])
    for d, s in zip(np.asarray(ft['day_obs']).astype(int)[mask],
                    np.asarray(ft['seq_num']).astype(int)[mask]):
        bad.add((int(d), int(s)))
    return bad


def iZs_from_visits(visits_path):
    vt = QTable.read(str(visits_path))
    if 'nollIndices' not in vt.colnames:
        raise ValueError(f'{visits_path} has no nollIndices column')
    return [int(j) for j in np.asarray(vt['nollIndices'][0]).tolist()]


def load_donut_zk(donuts_path, coord_sys, bad_set):
    """Stack zk arrays for all matched donuts, dropping bad/filtered visits.
    Returns (zk (N, n_j), iZs-less — caller supplies iZs)."""
    pf = pq.ParquetFile(str(donuts_path))
    zk_col = f'zk_{coord_sys}'
    have_match = 'matched_intra_extra' in pf.schema_arrow.names
    cols = [zk_col, 'day_obs', 'seq_num'] + (
        ['matched_intra_extra'] if have_match else [])
    chunks, n_skip = [], 0
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=cols).to_pandas()
        if len(df) == 0:
            continue
        # Drop bad/filtered visits by the actual per-row (day_obs, seq_num),
        # not by the row group's first row — a row group is not guaranteed to
        # hold exactly one visit.
        vkey = list(zip(df['day_obs'].astype(int), df['seq_num'].astype(int)))
        bad_row = np.array([k in bad_set for k in vkey])
        if bad_row.any():
            n_skip += len(set(k for k, b in zip(vkey, bad_row) if b))
            df = df[~bad_row]
        if have_match:
            df = df[df['matched_intra_extra'].fillna(False)]
        if len(df):
            chunks.append(np.stack(df[zk_col].values))
    zk = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0))
    print(f'  {len(zk):,} matched donuts loaded; {n_skip} visits skipped')
    return zk


def quartile_masks(values, n_quartiles=4):
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    edges = np.unique(np.nanquantile(v[finite], np.linspace(0, 1, n_quartiles + 1)))
    masks = []
    for q in range(len(edges) - 1):
        lo, hi = edges[q], edges[q + 1]
        if q == len(edges) - 2:
            masks.append(finite & (v >= lo) & (v <= hi))
        else:
            masks.append(finite & (v >= lo) & (v < hi))
    return masks, edges


def quartile_fit_rows(x, y, n_quartiles=4):
    """Per-quartile (slope, intercept, r, n, x_lo, x_hi) of y vs x."""
    masks, edges = quartile_masks(x, n_quartiles=n_quartiles)
    rows = []
    for q, m in enumerate(masks):
        xv, yv = x[m], y[m]
        f = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[f], yv[f]
        row = dict(quartile=q + 1, x_lo=float(edges[q]), x_hi=float(edges[q + 1]),
                   n=int(len(xv)), slope=np.nan, intercept=np.nan, r=np.nan)
        if len(xv) >= 5:
            coef = np.polyfit(xv, yv, 1)
            row.update(slope=float(coef[0]), intercept=float(coef[1]),
                       r=float(np.corrcoef(xv, yv)[0, 1]))
        rows.append(row)
    return rows, edges


def plot_quartile_density_page(j_pri, j_sec, x_all, y_all, *, n_quartiles=4,
                               n_bins=80, plo=1.0, phi=99.0, label='', ncols=2):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    masks, edges = quartile_masks(x_all, n_quartiles=n_quartiles)
    nrows = (len(masks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), layout='constrained')
    axes = np.atleast_1d(axes).ravel()
    for q, m in enumerate(masks):
        ax = axes[q]
        x, y = x_all[m], y_all[m]
        f = np.isfinite(x) & np.isfinite(y)
        x, y = x[f], y[f]
        if len(x) < 5:
            ax.set_visible(False)
            continue
        x_lo, x_hi = np.nanpercentile(x, [plo, phi])
        y_lo, y_hi = np.nanpercentile(y, [plo, phi])
        xp = 0.02 * max(abs(x_hi - x_lo), 1e-6)
        yp = 0.02 * max(abs(y_hi - y_lo), 1e-6)
        x_lo -= xp; x_hi += xp; y_lo -= yp; y_hi += yp
        counts, xe, ye = np.histogram2d(
            x, y, bins=[np.linspace(x_lo, x_hi, n_bins + 1),
                        np.linspace(y_lo, y_hi, n_bins + 1)])
        cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('white', alpha=0)
        pcm = ax.pcolormesh(xe, ye, np.ma.masked_where(counts == 0, counts).T,
                            cmap=cmap, norm=LogNorm(vmin=1, vmax=max(counts.max(), 2)),
                            shading='auto')
        plt.colorbar(pcm, ax=ax, shrink=0.85, label='donuts / bin')
        ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
        in_box = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        if int(in_box.sum()) >= 5:
            mm, bb = np.polyfit(x[in_box], y[in_box], 1)
            xf = np.array([x_lo, x_hi])
            ax.plot(xf, mm * xf + bb, 'r-', lw=1.2, alpha=0.95)
            r = float(np.corrcoef(x[in_box], y[in_box])[0, 1])
            txt = (f'slope={mm:+.4f}\nintercept={bb:+.4f} μm\n'
                   f'r={r:+.4f}\nn={int(in_box.sum())}')
        else:
            txt = f'n={int(f.sum())}'
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha='left', va='top',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      alpha=0.85, lw=0))
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(f'Z{j_pri} ({NOLL_NAMES.get(j_pri, "?")}) [μm]', fontsize=9)
        ax.set_ylabel(f'Z{j_sec} ({NOLL_NAMES.get(j_sec, "?")}) [μm]', fontsize=9)
        ax.set_title(f'Q{q + 1}: Z{j_pri} ∈ [{edges[q]:+.3f}, {edges[q + 1]:+.3f}] μm',
                     fontsize=10)
        ax.tick_params(labelsize=8)
    for q in range(len(masks), len(axes)):
        axes[q].set_visible(False)
    fig.suptitle(f'Z{j_sec} ({NOLL_NAMES.get(j_sec, "?")})  vs  '
                 f'Z{j_pri} ({NOLL_NAMES.get(j_pri, "?")})    {label}', fontsize=13)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('donuts', help='output/<ps>/donuts.parquet')
    ap.add_argument('--visits', required=True, help='output/<ps>/visits.parquet')
    ap.add_argument('--fits', default=None, help='output/<ps>/fits.parquet (bad-fit cut)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--coord-sys', default='OCS')
    ap.add_argument('--param-set', default=None,
                    help='for analysis_config.yaml aberration_pairs overrides')
    ap.add_argument('--analysis-config', default=None,
                    help='analysis_config.yaml path (default: ../analysis_config.yaml)')
    ap.add_argument('--fit-prefix', default=None, help='override config fit_prefix')
    ap.add_argument('--label', default='')
    ap.add_argument('--n-quartiles', type=int, default=None,
                    help='override config n_quartiles')
    args = ap.parse_args()

    # knobs from analysis_config.yaml (CLI overrides win); code defaults last
    sect = mc.analysis_section(
        'aberration_pairs', args.param_set,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))
    n_quartiles = (args.n_quartiles if args.n_quartiles is not None
                   else int(sect.get('n_quartiles', 4)))
    fit_prefix = (args.fit_prefix if args.fit_prefix is not None
                  else sect.get('fit_prefix', 'z1toz6'))
    pairs_all = ([tuple(p) for p in sect['pairs']] if sect.get('pairs')
                 else ABERRATION_PAIRS)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    iZs = iZs_from_visits(args.visits)
    bad = bad_visit_set(args.fits, fit_prefix) if args.fits else set()
    print(f'[aberration_pairs] {args.donuts}  coord={args.coord_sys}')
    zk = load_donut_zk(args.donuts, args.coord_sys, bad)
    if len(zk) == 0:
        raise RuntimeError('No matched donuts loaded.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows, n_pages = [], 0
    pairs = [(p, s) for (p, s) in pairs_all if p in iZs and s in iZs]
    skipped = [(p, s) for (p, s) in pairs_all if (p, s) not in pairs]
    if skipped:
        print(f'  skipping pairs absent from iZs: {skipped}')
    with PdfPages(str(out / 'aberration_pairs.pdf')) as pdf:
        for j_pri, j_sec in pairs:
            x = zk[:, iZs.index(j_pri)]
            y = zk[:, iZs.index(j_sec)]
            qrows, _ = quartile_fit_rows(x, y, n_quartiles=n_quartiles)
            for rr in qrows:
                rows.append(dict(j_primary=j_pri, j_secondary=j_sec, **rr))
            fig = plot_quartile_density_page(j_pri, j_sec, x, y,
                                             n_quartiles=n_quartiles,
                                             label=args.label or args.coord_sys)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            n_pages += 1
    df = pd.DataFrame(rows)
    df.to_parquet(out / 'aberration_pairs_summary.parquet')
    print(f'  wrote aberration_pairs.pdf ({n_pages} pages) + '
          f'aberration_pairs_summary.parquet ({len(df)} rows)')


if __name__ == '__main__':
    main()
