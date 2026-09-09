#!/usr/bin/env python3
"""Check the k<=3 and k<=6 Double Zernike fits against the batoid intrinsic wavefront.

For every Full Array Mode (FAM) visit in a `param_set`, the pipeline's `fit` rule fits
the per-donut wavefront deviation — measured minus the batoid design intrinsic — to
focal-plane Noll Zernikes Z1..Z3 (`z1toz3`) and Z1..Z6 (`z1toz6`), one fit per pupil
Zernike. This script recomputes the residual of those stored fits, records robust
residual metrics per (visit, prefix, pupil Zernike), and plots both the fitted Double
Zernike (DZ) coefficients and the residual Zernike maps over the focal plane.

The residual reproduced here is the fit's own definition,

    resid = (zk_<coord> - zk_intrinsic_<coord>) - A(thx, thy) @ c

with ``A`` the focal-plane Noll basis on a unit disk of radius 1.75 deg and ``c`` the
stored coefficients, so the robust scatter of `resid` is directly comparable to the
``<prefix>_z<j>_scale`` column the Huber fit already writes.

Usage
-----
    python code/dzfit/run_dz_fit_check.py --param-set <ps>
    python code/dzfit/run_dz_fit_check.py --param-set <ps> --skip-metrics   # replot only
    python code/dzfit/run_dz_fit_check.py --param-set <ps> --max-visits 50  # quick look

Writes ``dz_fit_check.parquet`` (one row per visit, prefix and pupil Zernike) and
``dz_fit_check.pdf`` into ``--output-dir``.

Notes
-----
Streams ``donuts.parquet`` by row group, so peak memory is set by the largest row group
rather than by the full 9-million-donut table.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
from common.utils import nmad                                          # noqa: E402

from lsst.ts.intrinsic.wavefront.dz_fitting import (                   # noqa: E402
    derive_noll_indices, focal_plane_zernike_basis)

try:
    from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES
except Exception:                                                      # pragma: no cover
    NOLL_NAMES = {}

# The focal-plane radius the fit normalizes to, in degrees. Must match
# dz_fitting.focal_plane_zernike_basis's default or the reconstruction is wrong.
FP_RADIUS_DEG = 1.75

# Focal-Zernike term count per prefix.
PREFIX_MAX_K = {'z1toz3': 3, 'z1toz6': 6}

# Focal-plane half-width for the residual maps, in degrees. Wider than FP_RADIUS_DEG so
# that donuts just outside the fit-normalization radius still appear.
MAP_RADIUS_DEG = 1.8


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True,
                    help='param_set name under output/')
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'],
                    help='field-angle frame of the fit (default OCS)')
    ap.add_argument('--output-dir', default=None,
                    help='default: output/<param_set>/dzfit')
    ap.add_argument('--output-root', default=None,
                    help='default: <topic>/output')
    ap.add_argument('--skip-metrics', action='store_true',
                    help='reuse an existing dz_fit_check.parquet and only replot')
    ap.add_argument('--max-visits', type=int, default=None,
                    help='stop after this many visits (quick look)')
    ap.add_argument('--n-map-bins', type=int, default=24,
                    help='bins per axis in the residual maps (default 24)')
    ap.add_argument('--drop-bad-fit', dest='drop_bad_fit', action='store_true',
                    default=True, help='exclude visits flagged bad_fit (default)')
    ap.add_argument('--keep-bad-fit', dest='drop_bad_fit', action='store_false',
                    help='keep visits flagged bad_fit')
    return ap.parse_args()


def load_fit_table(fits_path, prefixes, drop_bad_fit):
    """Per-visit fit coefficients, indexed by ``(day_obs, seq_num)``.

    Returns
    -------
    coeffs : `dict`
        ``(day_obs, seq_num) -> {prefix: {pupil_j: ndarray of k coefficients}}``, the
        coefficients in µm of wavefront.
    meta : `pandas.DataFrame`
        The per-visit non-coefficient columns, for the time-history plots.
    """
    tab = pq.read_table(str(fits_path))
    names = set(tab.schema.names)
    df = tab.to_pandas()

    if drop_bad_fit:
        n_before = len(df)
        bad = np.zeros(len(df), bool)
        for cand in ('bad_fit', 'z1toz3_bad_fit', 'z1toz6_bad_fit'):
            if cand in names:
                bad |= df[cand].fillna(False).astype(bool).to_numpy()
        df = df[~bad].reset_index(drop=True)
        print(f'  bad-fit cut: kept {len(df)}/{n_before} visits')

    return df, names


def coeff_lookup(df, names, prefixes, iZs):
    """Build ``(day_obs, seq_num) -> {prefix: (n_j, k) coefficient array}``."""
    out = {}
    keys = list(zip(df['day_obs'].astype(int), df['seq_num'].astype(int)))
    per_prefix = {}
    for prefix, max_k in prefixes.items():
        arr = np.full((len(df), len(iZs), max_k), np.nan)
        for j_idx, iZ in enumerate(iZs):
            for ci in range(max_k):
                col = f'{prefix}_z{iZ}_c{ci + 1}'
                if col in names:
                    arr[:, j_idx, ci] = df[col].to_numpy(dtype=float)
        per_prefix[prefix] = arr
    for row, key in enumerate(keys):
        out[key] = {p: per_prefix[p][row] for p in prefixes}
    return out


def stream_residual_metrics(donuts_path, coord_sys, iZs, iZidx, coeffs_by_visit,
                            prefixes, n_map_bins, max_visits=None):
    """Recompute fit residuals donut by donut, accumulating metrics and maps.

    Returns
    -------
    rows : `list` of `dict`
        One row per (visit, prefix, pupil Zernike): robust residual scatter in µm of
        wavefront, and the corresponding deviation scatter for context.
    maps : `dict`
        ``(prefix, pupil_j) -> (sum, count)`` binned residual accumulators over the
        focal plane, in µm of wavefront.
    """
    pf = pq.ParquetFile(str(donuts_path))
    have = set(pf.schema_arrow.names)
    zk_col, in_col = f'zk_{coord_sys}', f'zk_intrinsic_{coord_sys}'
    dev_col = f'zk_deviation_{coord_sys}'
    thx_col, thy_col = f'thx_{coord_sys}', f'thy_{coord_sys}'
    use_dev = dev_col in have
    cols = [thx_col, thy_col, 'day_obs', 'seq_num']
    cols += [dev_col] if use_dev else [zk_col, in_col]
    if 'matched_intra_extra' in have:
        cols.append('matched_intra_extra')
    print(f'  deviation column: {dev_col if use_dev else f"{zk_col} - {in_col}"}')

    edges = np.linspace(-MAP_RADIUS_DEG, MAP_RADIUS_DEG, n_map_bins + 1)
    maps = {(p, iZ): [np.zeros((n_map_bins, n_map_bins)),
                      np.zeros((n_map_bins, n_map_bins))]
            for p in prefixes for iZ in iZs}

    rows = []
    # A visit's donuts can straddle row groups, so buffer per visit and flush a visit
    # only once a later visit appears.
    buf = defaultdict(list)
    n_visits_done = 0
    stop = False

    def flush(key):
        nonlocal n_visits_done
        parts = buf.pop(key, None)
        if not parts:
            return
        thx = np.concatenate([p[0] for p in parts])
        thy = np.concatenate([p[1] for p in parts])
        dev = np.concatenate([p[2] for p in parts], axis=0)
        cf = coeffs_by_visit.get(key)
        if cf is None:
            return
        ok = np.isfinite(thx) & np.isfinite(thy)
        if ok.sum() < 10:
            return
        thx, thy, dev = thx[ok], thy[ok], dev[ok]
        ix = np.clip(np.digitize(thx, edges) - 1, 0, n_map_bins - 1)
        iy = np.clip(np.digitize(thy, edges) - 1, 0, n_map_bins - 1)
        inside = (np.abs(thx) <= MAP_RADIUS_DEG) & (np.abs(thy) <= MAP_RADIUS_DEG)

        for prefix, max_k in prefixes.items():
            A, _ = focal_plane_zernike_basis(thx, thy, max_k, FP_RADIUS_DEG)
            for j_idx, iZ in enumerate(iZs):
                c = cf[prefix][j_idx]
                if not np.isfinite(c).all():
                    continue
                d = dev[:, iZidx[iZ]]
                resid = d - A @ c
                good = np.isfinite(resid)
                if good.sum() < 10:
                    continue
                rows.append(dict(
                    day_obs=key[0], seq_num=key[1], prefix=prefix, pupil_j=iZ,
                    n_donuts=int(good.sum()),
                    resid_nmad_um=nmad(resid[good]),
                    resid_rms_um=float(np.sqrt(np.mean(resid[good] ** 2))),
                    resid_median_um=float(np.median(resid[good])),
                    dev_nmad_um=nmad(d[np.isfinite(d)]),
                ))
                sel = good & inside
                np.add.at(maps[(prefix, iZ)][0], (ix[sel], iy[sel]), resid[sel])
                np.add.at(maps[(prefix, iZ)][1], (ix[sel], iy[sel]), 1.0)
        n_visits_done += 1
        if n_visits_done % 200 == 0:
            print(f'    ...{n_visits_done} visits')

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        if len(df) == 0:
            continue
        if 'matched_intra_extra' in df.columns:
            df = df[df['matched_intra_extra'].fillna(False).astype(bool)]
            if len(df) == 0:
                continue
        if use_dev:
            dev = np.stack(df[dev_col].to_numpy())
        else:
            dev = (np.stack(df[zk_col].to_numpy())
                   - np.stack(df[in_col].to_numpy()))
        thx = np.rad2deg(df[thx_col].to_numpy(dtype=float))
        thy = np.rad2deg(df[thy_col].to_numpy(dtype=float))
        dobs = df['day_obs'].to_numpy(dtype=np.int64)
        snum = df['seq_num'].to_numpy(dtype=np.int64)

        seen_here = set()
        for key in set(zip(dobs.tolist(), snum.tolist())):
            m = (dobs == key[0]) & (snum == key[1])
            buf[key].append((thx[m], thy[m], dev[m]))
            seen_here.add(key)
        # Flush every buffered visit not present in this row group.
        for key in [k for k in list(buf) if k not in seen_here]:
            if key in coeffs_by_visit:
                flush(key)
                if max_visits and n_visits_done >= max_visits:
                    stop = True
                    break
            else:
                buf.pop(key, None)
        if stop:
            break

    if not stop:
        for key in list(buf):
            flush(key)

    print(f'  residual metrics for {n_visits_done} visits, {len(rows)} rows')
    return rows, maps, edges


def _pupil_label(iZ):
    name = NOLL_NAMES.get(iZ)
    return f'Z{iZ} ({name})' if name else f'Z{iZ}'


def page_summary(pdf, met, iZs, prefixes):
    """Residual scatter per pupil Zernike, k<=3 against k<=6."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axes[0]
    width = 0.38
    x = np.arange(len(iZs))
    colors = {'z1toz3': 'tab:orange', 'z1toz6': 'tab:blue'}
    med = {}
    for pi, prefix in enumerate(prefixes):
        vals = [met[(met['prefix'] == prefix) & (met['pupil_j'] == iZ)]
                ['resid_nmad_um'].median() for iZ in iZs]
        med[prefix] = np.array(vals, dtype=float)
        ax.bar(x + (pi - 0.5) * width, vals, width, label=f'{prefix} (k<={PREFIX_MAX_K[prefix]})',
               color=colors.get(prefix), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Z{j}' for j in iZs], fontsize=8)
    ax.set_ylabel('median residual nMAD [µm of wavefront]')
    ax.set_title('DZ fit residual per pupil Zernike — median over visits')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    ax2 = axes[1]
    if 'z1toz3' in med and 'z1toz6' in med:
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = med['z1toz6'] / med['z1toz3']
        ax2.bar(x, ratio, 0.6, color='tab:green', alpha=0.85)
        ax2.axhline(1.0, color='k', lw=0.8, ls='--')
        ax2.set_ylabel('nMAD ratio\nk<=6 / k<=3\n(dimensionless)', fontsize=9)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Z{j}' for j in iZs], fontsize=8)
        ax2.grid(alpha=0.3, axis='y')
        fin = np.isfinite(ratio)
        if fin.any():
            ax2.set_title(f'median ratio over pupil Zernikes = '
                          f'{np.median(ratio[fin]):.3f} (dimensionless; '
                          f'k<=6 residual nMAD over k<=3 residual nMAD)',
                          fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_scale_check(pdf, met, fits_df, names, iZs, prefixes):
    """Recomputed residual scatter against the Huber scale the fit stored."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(prefixes), figsize=(11, 5.2), squeeze=False)
    for pi, prefix in enumerate(prefixes):
        ax = axes[0][pi]
        sub = met[met['prefix'] == prefix]
        xs, ys = [], []
        key_to_row = {(int(d), int(s)): i for i, (d, s) in
                      enumerate(zip(fits_df['day_obs'], fits_df['seq_num']))}
        for iZ in iZs:
            col = f'{prefix}_z{iZ}_scale'
            if col not in names:
                continue
            s_arr = fits_df[col].to_numpy(dtype=float)
            ss = sub[sub['pupil_j'] == iZ]
            for d, sn, rn in zip(ss['day_obs'], ss['seq_num'], ss['resid_nmad_um']):
                r = key_to_row.get((int(d), int(sn)))
                if r is not None and np.isfinite(s_arr[r]) and np.isfinite(rn):
                    xs.append(s_arr[r])
                    ys.append(rn)
        if xs:
            xs, ys = np.array(xs), np.array(ys)
            ax.plot(xs, ys, '.', ms=1.5, alpha=0.25)
            hi = np.nanpercentile(np.concatenate([xs, ys]), 99)
            ax.plot([0, hi], [0, hi], 'r--', lw=1.0, label='1:1')
            r_p = float(np.corrcoef(xs, ys)[0, 1])
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_title(f'{prefix}: Pearson r = {r_p:+.4f}\n'
                         f'(dimensionless; n = {len(xs)} visit-Zernikes)', fontsize=9)
            ax.legend(fontsize=8)
        ax.set_xlabel(f'{prefix}_z<j>_scale, stored Huber scale [µm of wavefront]',
                      fontsize=9)
        ax.set_ylabel('recomputed residual nMAD [µm of wavefront]', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle('Cross-check: recomputed residual against the fit\'s own robust scale',
                 fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_coeff_history(pdf, fits_df, names, iZs, prefix, max_k):
    """Fitted DZ coefficients against date, one page per pupil Zernike."""
    import matplotlib.pyplot as plt

    if 'mjd' in fits_df.columns:
        t = fits_df['mjd'].to_numpy(dtype=float)
        tlabel = 'MJD [d]'
    else:
        t = fits_df['day_obs'].to_numpy(dtype=float)
        tlabel = 'day_obs'

    for iZ in iZs:
        cols = [f'{prefix}_z{iZ}_c{k + 1}' for k in range(max_k)]
        if not any(c in names for c in cols):
            continue
        ncol = 3
        nrow = int(np.ceil(max_k / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow + 1.0),
                                 squeeze=False, sharex=True)
        for k in range(nrow * ncol):
            ax = axes[k // ncol][k % ncol]
            if k >= max_k or cols[k] not in names:
                ax.set_visible(False)
                continue
            y = fits_df[cols[k]].to_numpy(dtype=float)
            good = np.isfinite(t) & np.isfinite(y)
            ax.plot(t[good], y[good], '.', ms=2.0, alpha=0.5)
            ax.axhline(0, color='gray', lw=0.6, alpha=0.7)
            if good.sum() >= 3:
                ax.set_title(f'k={k + 1}:  median {np.median(y[good]):+.4f},  '
                             f'nMAD {nmad(y[good]):.4f} µm', fontsize=9)
            ax.set_ylabel(f'c{k + 1} [µm]', fontsize=9)
            ax.grid(alpha=0.3)
            if k // ncol == nrow - 1:
                ax.set_xlabel(tlabel, fontsize=9)
        fig.suptitle(f'{prefix} DZ coefficients vs date — pupil {_pupil_label(iZ)} '
                     f'({int(np.isfinite(t).sum())} visits)', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def page_coeff_hist(pdf, fits_df, names, iZs, prefix, max_k):
    """Distribution of each fitted DZ coefficient, one page per focal k."""
    import matplotlib.pyplot as plt

    for k in range(max_k):
        ncol = 4
        nrow = int(np.ceil(len(iZs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.6 * nrow + 1.0),
                                 squeeze=False)
        for ai in range(nrow * ncol):
            ax = axes[ai // ncol][ai % ncol]
            if ai >= len(iZs):
                ax.set_visible(False)
                continue
            iZ = iZs[ai]
            col = f'{prefix}_z{iZ}_c{k + 1}'
            if col not in names:
                ax.set_visible(False)
                continue
            y = fits_df[col].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            if len(y) < 5:
                ax.set_visible(False)
                continue
            lo, hi = np.percentile(y, [0.5, 99.5])
            ax.hist(y, bins=60, range=(lo, hi), color='tab:blue', alpha=0.8)
            ax.axvline(0, color='gray', lw=0.6)
            ax.axvline(np.median(y), color='firebrick', lw=1.0)
            ax.set_title(f'Z{iZ}: med {np.median(y):+.3f},\nnMAD {nmad(y):.3f} µm',
                         fontsize=8)
            ax.set_xlabel('[µm of wavefront]', fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle(f'{prefix} coefficient c{k + 1} distribution over visits, '
                     f'per pupil Zernike', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def page_residual_maps(pdf, maps, edges, iZs, prefix, min_count=5):
    """Mean residual over the focal plane, one panel per pupil Zernike."""
    import matplotlib.pyplot as plt

    ncol = 4
    n = min(len(iZs), 12)
    sel = iZs[:n]
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.8 * nrow),
                             squeeze=False)
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    for ai in range(nrow * ncol):
        ax = axes[ai // ncol][ai % ncol]
        if ai >= n:
            ax.set_visible(False)
            continue
        iZ = sel[ai]
        ssum, scnt = maps[(prefix, iZ)]
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(scnt >= min_count, ssum / scnt, np.nan)
        fin = mean[np.isfinite(mean)]
        if len(fin) == 0:
            ax.set_visible(False)
            continue
        v = np.percentile(np.abs(fin), 98)
        im = ax.imshow(mean.T, origin='lower', extent=ext, cmap='RdBu_r',
                       vmin=-v, vmax=v, interpolation='none', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.8, label='µm')
        ax.set_title(f'{_pupil_label(iZ)}\nnMAD {nmad(fin):.4f} µm', fontsize=9)
        ax.set_xlabel(f'thx [deg]', fontsize=8)
        ax.set_ylabel(f'thy [deg]', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f'{prefix} (k<={PREFIX_MAX_K[prefix]}) mean fit residual over the '
                 f'focal plane, averaged over visits', fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_residual_map_compare(pdf, maps, edges, iZs, prefixes, min_count=5,
                              n_show=6):
    """k<=3 and k<=6 residual maps side by side for the low pupil Zernikes."""
    import matplotlib.pyplot as plt

    sel = iZs[:n_show]
    ncol = len(prefixes) + 1
    fig, axes = plt.subplots(len(sel), ncol, figsize=(4.2 * ncol, 3.6 * len(sel)),
                             squeeze=False)
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    order = list(prefixes)
    for ri, iZ in enumerate(sel):
        means = {}
        for prefix in order:
            ssum, scnt = maps[(prefix, iZ)]
            with np.errstate(invalid='ignore', divide='ignore'):
                means[prefix] = np.where(scnt >= min_count, ssum / scnt, np.nan)
        allfin = np.concatenate([m[np.isfinite(m)] for m in means.values()]) \
            if any(np.isfinite(m).any() for m in means.values()) else np.array([])
        if len(allfin) == 0:
            for ci in range(ncol):
                axes[ri][ci].set_visible(False)
            continue
        v = np.percentile(np.abs(allfin), 98)
        for ci, prefix in enumerate(order):
            ax = axes[ri][ci]
            im = ax.imshow(means[prefix].T, origin='lower', extent=ext,
                           cmap='RdBu_r', vmin=-v, vmax=v, interpolation='none',
                           aspect='equal')
            plt.colorbar(im, ax=ax, shrink=0.8, label='µm')
            fin = means[prefix][np.isfinite(means[prefix])]
            ax.set_title(f'{_pupil_label(iZ)} — {prefix}\nnMAD {nmad(fin):.4f} µm',
                         fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('thx [deg]', fontsize=8)
            if ci == 0:
                ax.set_ylabel('thy [deg]', fontsize=8)
        ax = axes[ri][ncol - 1]
        if len(order) == 2:
            diff = means[order[0]] - means[order[1]]
            fin = diff[np.isfinite(diff)]
            if len(fin):
                vd = np.percentile(np.abs(fin), 98)
                im = ax.imshow(diff.T, origin='lower', extent=ext, cmap='PuOr_r',
                               vmin=-vd, vmax=vd, interpolation='none',
                               aspect='equal')
                plt.colorbar(im, ax=ax, shrink=0.8, label='µm')
                ax.set_title(f'{order[0]} - {order[1]}\nnMAD {nmad(fin):.4f} µm',
                             fontsize=9)
                ax.tick_params(labelsize=7)
                ax.set_xlabel('thx [deg]', fontsize=8)
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)
    fig.suptitle('Residual maps: k<=3 against k<=6, and their difference',
                 fontsize=13)
    # Leave room for the suptitle: with many rows, tight_layout alone lets the first
    # row's titles collide with it.
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.30 / len(sel)])
    pdf.savefig(fig)
    plt.close(fig)


def page_coeff_err(pdf, fits_df, names, iZs, prefixes):
    """Formal coefficient errors, and the residual scale, per pupil Zernike."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(prefixes), 1,
                             figsize=(11, 4.2 * len(prefixes)), squeeze=False)
    for pi, prefix in enumerate(prefixes):
        ax = axes[pi][0]
        max_k = PREFIX_MAX_K[prefix]
        x = np.arange(len(iZs))
        for k in range(max_k):
            vals = []
            for iZ in iZs:
                col = f'{prefix}_z{iZ}_c{k + 1}_err'
                if col in names:
                    v = fits_df[col].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    vals.append(np.median(v) if len(v) else np.nan)
                else:
                    vals.append(np.nan)
            ax.plot(x, vals, 'o-', ms=3.5, lw=1.0, label=f'c{k + 1}')
        scale_med = []
        for iZ in iZs:
            col = f'{prefix}_z{iZ}_scale'
            if col in names:
                v = fits_df[col].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                scale_med.append(np.median(v) if len(v) else np.nan)
            else:
                scale_med.append(np.nan)
        ax.plot(x, scale_med, 'k--', lw=1.4, label='robust scale')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Z{j}' for j in iZs], fontsize=8)
        ax.set_ylabel('median [µm of wavefront]', fontsize=9)
        ax.set_yscale('log')
        ax.set_title(f'{prefix}: formal coefficient errors and residual scale',
                     fontsize=10)
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    args = parse_args()
    topic = pathlib.Path(__file__).resolve().parents[2]
    out_root = pathlib.Path(args.output_root) if args.output_root else topic / 'output'
    ps_dir = out_root / args.param_set
    out_dir = (pathlib.Path(args.output_dir) if args.output_dir
               else ps_dir / 'dzfit')
    out_dir.mkdir(parents=True, exist_ok=True)

    donuts_path = ps_dir / 'donuts.parquet'
    fits_path = ps_dir / 'fits.parquet'
    visits_path = ps_dir / 'visits.parquet'
    for p in (donuts_path, fits_path):
        if not p.exists():
            raise SystemExit(f'missing input: {p}')

    print(f'[dz_fit_check] param_set={args.param_set} coord={args.coord_sys}')

    fits_df, names = load_fit_table(fits_path, PREFIX_MAX_K, args.drop_bad_fit)

    noll_arr = None
    if 'nollIndices' in names:
        noll_arr = np.asarray(pq.read_table(str(fits_path), columns=['nollIndices'])
                              .to_pandas()['nollIndices'][0])
    elif visits_path.exists():
        vt = pq.read_table(str(visits_path))
        if 'nollIndices' in vt.schema.names:
            noll_arr = np.asarray(vt.to_pandas()['nollIndices'][0])
    n_zk = len(pq.ParquetFile(str(donuts_path)).read_row_group(
        0, columns=[f'zk_{args.coord_sys}']).to_pandas()[f'zk_{args.coord_sys}'][0])
    iZs, iZidx = derive_noll_indices(n_zk, noll_arr)
    print(f'  pupil Noll indices ({len(iZs)}): {iZs}')

    metrics_path = out_dir / 'dz_fit_check.parquet'
    maps = edges = None
    if args.skip_metrics and metrics_path.exists():
        met = pd.read_parquet(metrics_path)
        print(f'  reusing {metrics_path} ({len(met)} rows); maps unavailable')
    else:
        coeffs_by_visit = coeff_lookup(fits_df, names, PREFIX_MAX_K, iZs)
        rows, maps, edges = stream_residual_metrics(
            donuts_path, args.coord_sys, iZs, iZidx, coeffs_by_visit,
            PREFIX_MAX_K, args.n_map_bins, args.max_visits)
        met = pd.DataFrame(rows)
        if len(met) == 0:
            raise SystemExit('no residual rows computed')
        met.to_parquet(metrics_path)
        print(f'  wrote {metrics_path} ({len(met)} rows)')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = out_dir / 'dz_fit_check.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        page_summary(pdf, met, iZs, PREFIX_MAX_K)
        page_scale_check(pdf, met, fits_df, names, iZs, PREFIX_MAX_K)
        page_coeff_err(pdf, fits_df, names, iZs, PREFIX_MAX_K)
        if maps is not None:
            for prefix in PREFIX_MAX_K:
                page_residual_maps(pdf, maps, edges, iZs, prefix)
            page_residual_map_compare(pdf, maps, edges, iZs, PREFIX_MAX_K)
        for prefix, max_k in PREFIX_MAX_K.items():
            page_coeff_hist(pdf, fits_df, names, iZs, prefix, max_k)
        for prefix, max_k in PREFIX_MAX_K.items():
            page_coeff_history(pdf, fits_df, names, iZs, prefix, max_k)
    print(f'  wrote {pdf_path}')

    # Headline numbers, so the log carries the result without opening the PDF.
    print('\n  median residual nMAD over all pupil Zernikes [µm of wavefront]:')
    for prefix in PREFIX_MAX_K:
        v = met[met['prefix'] == prefix]['resid_nmad_um']
        print(f'    {prefix} (k<={PREFIX_MAX_K[prefix]}): {v.median():.4f}  '
              f'(n = {len(v)} visit-Zernike pairs)')
    print('  Done.')


if __name__ == '__main__':
    main()
