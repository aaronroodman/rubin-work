"""Shared machinery for comparing per-donut wavefront Zernikes between two
processing runs (different code versions or settings).

Backs ``study_compare_donuts.ipynb`` — the consolidation of the former
``study_danish_v0p6_vs_v1``, ``study_binning`` and ``donutalgo_comparison``
notebooks, which all did the same thing: load two donut parquet tables, match
the same physical donuts across them, and compare ``zk`` per Noll index.

Two runs are referenced as side **A** and side **B**.  Inputs are resolved
from a named ``param_set`` (the new ``output/<ps>/{donuts,visits,fits}.parquet``
layout) with an explicit-path fallback for legacy tables.

Donut matching is per-CCD positional: two donuts match when their intra-focal
centroids ``(centroid_x_intra, centroid_y_intra)`` lie within ``tol_pix`` on the
same detector (KDTree).  This is the robust matcher shared by the streaming
notebooks; the old key-based matcher (rounded ``thx/thy``) is not carried over.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.zernike_names import NOLL_NAMES  # noqa: E402


# ----------------------------------------------------------------------
# Input resolution
# ----------------------------------------------------------------------
def visits_sidecar_path(donut_parquet_path):
    """Legacy convention: ``<stem>_visits.parquet`` next to a donut parquet."""
    p = Path(donut_parquet_path)
    return p.with_name(p.stem + '_visits.parquet')


def fits_sidecar_path(donut_parquet_path):
    """Legacy convention: ``<stem>_fits.parquet`` next to a donut parquet."""
    p = Path(donut_parquet_path)
    return p.with_name(p.stem + '_fits.parquet')


def resolve_param_set_paths(param_set, output_root='output'):
    """``(donuts, visits, fits)`` paths for the new ``output/<ps>/`` layout."""
    base = Path(output_root) / param_set
    return (base / 'donuts.parquet',
            base / 'visits.parquet',
            base / 'fits.parquet')


def resolve_side(*, param_set=None, donut=None, visits=None, fits=None,
                 output_root='output'):
    """Resolve one comparison side to ``(donut, visits, fits)`` paths.

    ``param_set`` selects ``output/<ps>/{donuts,visits,fits}.parquet``.  Any of
    ``donut`` / ``visits`` / ``fits`` (str, ``Path``, or list of donut paths for
    a legacy multi-chunk run) override the resolved value.  Without a
    ``param_set``, ``visits`` / ``fits`` fall back to the ``_visits`` / ``_fits``
    sidecar of the (first) donut path.
    """
    if param_set is not None:
        d, v, f = resolve_param_set_paths(param_set, output_root)
    else:
        if donut is None:
            raise ValueError('resolve_side: pass param_set or donut path(s)')
        first = donut[0] if isinstance(donut, (list, tuple)) else donut
        d, v, f = donut, visits_sidecar_path(first), fits_sidecar_path(first)
    if donut is not None:
        d = donut
    if visits is not None:
        v = visits
    if fits is not None:
        f = fits
    return d, v, f


# ----------------------------------------------------------------------
# Visit selection / streaming
# ----------------------------------------------------------------------
def _alt_to_deg(alt_arr):
    a = np.asarray(alt_arr, dtype=float)
    if np.nanmax(np.abs(a)) < 2.0 * np.pi + 1e-3:
        return np.rad2deg(a)
    return a


def select_visits(visits_table, *, rot_max_deg=3.0, program_filter=None):
    """Boolean mask over ``visits_table``: |rotator| cut + optional program."""
    keep = np.ones(len(visits_table), dtype=bool)
    if rot_max_deg is not None and 'rotator_angle' in visits_table.colnames:
        rot = np.asarray(visits_table['rotator_angle'], dtype=float)
        keep &= np.isfinite(rot) & (np.abs(rot) <= rot_max_deg)
    if program_filter is not None and 'science_program' in visits_table.colnames:
        sp = np.asarray(visits_table['science_program']).astype(str)
        if isinstance(program_filter, (list, tuple, set)):
            prog_set = {str(p) for p in program_filter}
            keep &= np.array([s in prog_set for s in sp])
        else:
            keep &= (sp == str(program_filter))
    return keep


def load_visits(visits_paths):
    """Read one visits sidecar, or concatenate a list of them, to a QTable."""
    if isinstance(visits_paths, (list, tuple)):
        tabs = [QTable.read(str(p)) for p in visits_paths if Path(p).exists()]
        if not tabs:
            raise FileNotFoundError(f'No visits sidecars found: {visits_paths}')
        if len(tabs) == 1:
            return tabs[0]
        return QTable(np.concatenate([t.as_array() for t in tabs]),
                      names=tabs[0].colnames)
    return QTable.read(str(visits_paths))


def build_row_group_lookup(parquet_path):
    """``(day_obs, seq_num) -> row_group_idx`` from parquet column statistics.

    Relies on run_mktable writing one row group per visit (with day_obs /
    seq_num column stats), the convention combine_parquets.py preserves.
    """
    pf = pq.ParquetFile(str(parquet_path))
    lookup = {}
    for i in range(pf.num_row_groups):
        meta = pf.metadata.row_group(i)
        d = s = None
        for ci in range(meta.num_columns):
            cmeta = meta.column(ci)
            name = cmeta.path_in_schema
            if name == 'day_obs' and cmeta.statistics is not None:
                d = cmeta.statistics.min
            elif name == 'seq_num' and cmeta.statistics is not None:
                s = cmeta.statistics.min
        if d is not None and s is not None:
            lookup[(int(d), int(s))] = i
    return pf, lookup


class VisitSource:
    """Visit-indexed reader over one or more donut parquet files.

    Pools per-visit row groups from a single combined ``donuts.parquet`` or a
    list of legacy per-chunk files into one ``(day_obs, seq_num)`` lookup, so a
    visit can be streamed without loading the whole table.
    """

    def __init__(self, donut_paths):
        paths = donut_paths if isinstance(donut_paths, (list, tuple)) \
            else [donut_paths]
        self._pf_by_path = {}
        self._visit_to_src = {}
        present = []
        for p in paths:
            if not Path(p).exists():
                print(f'(skip missing donut parquet: {p})')
                continue
            pf, rg = build_row_group_lookup(p)
            self._pf_by_path[str(p)] = pf
            for key, idx in rg.items():
                self._visit_to_src.setdefault(key, (str(p), idx))
            present.append(str(p))
        if not present:
            raise FileNotFoundError(f'No donut parquet files found: {paths}')
        self.paths = present

    def keys(self):
        return set(self._visit_to_src)

    def read(self, day_obs, seq_num):
        """Return one visit's donuts as a DataFrame, or None if absent."""
        key = (int(day_obs), int(seq_num))
        if key not in self._visit_to_src:
            return None
        path, idx = self._visit_to_src[key]
        return self._pf_by_path[path].read_row_group(idx).to_pandas()


def match_donuts_per_ccd(df_a, df_b, *, tol_pix=5.0):
    """Match donuts in ``df_a``/``df_b`` on the same CCD via intra centroids.

    Returns parallel integer index arrays ``(idx_a, idx_b)`` into the two input
    DataFrames such that each pair refers to the same physical donut.
    """
    idx_a_all, idx_b_all = [], []
    det_a = df_a['detector'].to_numpy(dtype=str)
    det_b = df_b['detector'].to_numpy(dtype=str)
    for det in np.unique(det_a):
        ai = np.where(det_a == det)[0]
        bi = np.where(det_b == det)[0]
        if len(ai) == 0 or len(bi) == 0:
            continue
        xa = df_a['centroid_x_intra'].to_numpy(dtype=float)[ai]
        ya = df_a['centroid_y_intra'].to_numpy(dtype=float)[ai]
        xb = df_b['centroid_x_intra'].to_numpy(dtype=float)[bi]
        yb = df_b['centroid_y_intra'].to_numpy(dtype=float)[bi]
        tree = cKDTree(np.column_stack([xa, ya]))
        dist, k = tree.query(np.column_stack([xb, yb]),
                             distance_upper_bound=tol_pix)
        good = np.isfinite(dist) & (dist < tol_pix)
        idx_a_all.append(ai[k[good]])
        idx_b_all.append(bi[good])
    if not idx_a_all:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(idx_a_all), np.concatenate(idx_b_all)


class MatchedDonuts:
    """Accumulated matched-pair arrays from accumulate_matched_donuts.

    Attributes (each row = one matched donut pair):
        iZs                     pupil Noll indices (column order of zk_*)
        zk_a_OCS, zk_b_OCS      (n, nZk) Zernikes, OCS
        zk_a_CCS, zk_b_CCS      (n, nZk) Zernikes, CCS (or None if absent)
        thx_OCS/thy_OCS         (n,) extra-focal field angle [deg], OCS
        thx_CCS/thy_CCS         (n,) extra-focal field angle [deg], CCS (or None)
        detector, day_obs, seq_num
        n_matched_per_visit, n_unmatched_per_visit
        total_matched, total_unmatched
    """

    def delta_OCS(self, idx):
        return self.zk_b_OCS[:, idx] - self.zk_a_OCS[:, idx]

    def delta_CCS(self, idx):
        return self.zk_b_CCS[:, idx] - self.zk_a_CCS[:, idx]


def accumulate_matched_donuts(src_a, src_b, common_visits, *, iZs,
                              tol_pix=5.0, require_matched=True,
                              progress=True):
    """Stream common visits, match donuts per CCD, accumulate paired arrays.

    ``src_a``/``src_b`` are VisitSource objects; ``common_visits`` is an iterable
    of ``(day_obs, seq_num)``.  CCS columns are accumulated when present in the
    data (``zk_CCS`` / ``thx_CCS_extra``); otherwise the ``*_CCS`` attributes are
    None.  Field angles are taken from side A's extra-focal centroid, in degrees.
    """
    try:
        from tqdm.auto import tqdm
    except Exception:                       # tqdm optional
        def tqdm(it, **kw):
            return it

    nZk = len(iZs)
    acc = {k: [] for k in ('zk_a_OCS', 'zk_b_OCS', 'zk_a_CCS', 'zk_b_CCS',
                           'thx_OCS', 'thy_OCS', 'thx_CCS', 'thy_CCS',
                           'det', 'dobs', 'snum')}
    n_matched_per_visit, n_unmatched_per_visit = [], []
    has_ccs = None

    it = tqdm(list(common_visits), desc='matching') if progress \
        else common_visits
    for d, s in it:
        df_a = src_a.read(d, s)
        df_b = src_b.read(d, s)
        if df_a is None or df_b is None:
            continue
        if require_matched and 'matched_intra_extra' in df_a.columns:
            df_a = df_a[df_a['matched_intra_extra'].fillna(False)
                        ].reset_index(drop=True)
            df_b = df_b[df_b['matched_intra_extra'].fillna(False)
                        ].reset_index(drop=True)
        if len(df_a) == 0 or len(df_b) == 0:
            continue
        if has_ccs is None:
            has_ccs = ('zk_CCS' in df_a.columns and 'zk_CCS' in df_b.columns
                       and 'thx_CCS_extra' in df_a.columns)

        i_a, i_b = match_donuts_per_ccd(df_a, df_b, tol_pix=tol_pix)
        n_match = len(i_a)
        n_matched_per_visit.append(n_match)
        n_unmatched_per_visit.append(max(len(df_a), len(df_b)) - n_match)
        if n_match == 0:
            continue

        acc['zk_a_OCS'].append(np.stack(df_a['zk_OCS'].values[i_a]))
        acc['zk_b_OCS'].append(np.stack(df_b['zk_OCS'].values[i_b]))
        acc['thx_OCS'].append(
            np.rad2deg(df_a['thx_OCS_extra'].to_numpy(dtype=float)[i_a]))
        acc['thy_OCS'].append(
            np.rad2deg(df_a['thy_OCS_extra'].to_numpy(dtype=float)[i_a]))
        if has_ccs:
            acc['zk_a_CCS'].append(np.stack(df_a['zk_CCS'].values[i_a]))
            acc['zk_b_CCS'].append(np.stack(df_b['zk_CCS'].values[i_b]))
            acc['thx_CCS'].append(
                np.rad2deg(df_a['thx_CCS_extra'].to_numpy(dtype=float)[i_a]))
            acc['thy_CCS'].append(
                np.rad2deg(df_a['thy_CCS_extra'].to_numpy(dtype=float)[i_a]))
        acc['det'].append(df_a['detector'].to_numpy(dtype=str)[i_a])
        acc['dobs'].append(np.full(n_match, int(d), dtype=int))
        acc['snum'].append(np.full(n_match, int(s), dtype=int))

    def _cat(key, width=None):
        if acc[key]:
            return np.concatenate(acc[key], axis=0)
        return np.empty((0, width)) if width else np.empty(0)

    out = MatchedDonuts()
    out.iZs = list(iZs)
    out.zk_a_OCS = _cat('zk_a_OCS', nZk)
    out.zk_b_OCS = _cat('zk_b_OCS', nZk)
    out.thx_OCS = _cat('thx_OCS')
    out.thy_OCS = _cat('thy_OCS')
    if has_ccs:
        out.zk_a_CCS = _cat('zk_a_CCS', nZk)
        out.zk_b_CCS = _cat('zk_b_CCS', nZk)
        out.thx_CCS = _cat('thx_CCS')
        out.thy_CCS = _cat('thy_CCS')
    else:
        out.zk_a_CCS = out.zk_b_CCS = out.thx_CCS = out.thy_CCS = None
    out.detector = _cat('det').astype(str)
    out.day_obs = _cat('dobs').astype(int)
    out.seq_num = _cat('snum').astype(int)
    out.n_matched_per_visit = n_matched_per_visit
    out.n_unmatched_per_visit = n_unmatched_per_visit
    out.total_matched = int(out.zk_a_OCS.shape[0])
    out.total_unmatched = int(sum(n_unmatched_per_visit))
    return out


def probe_noll_indices(visits_table, donut_parquet):
    """Pupil Noll j list from the visits sidecar, or a probe row group."""
    if 'nollIndices' in visits_table.colnames:
        return [int(j) for j in np.asarray(visits_table['nollIndices'][0]).tolist()]
    first = donut_parquet[0] if isinstance(donut_parquet, (list, tuple)) \
        else donut_parquet
    df = pq.ParquetFile(str(first)).read_row_group(0).to_pandas()
    nZk = len(df['zk_OCS'].iloc[0])
    if nZk == 21:
        return list(range(4, 20)) + list(range(22, 27))
    return list(range(4, 4 + nZk))


# ----------------------------------------------------------------------
# Plotting primitives
# ----------------------------------------------------------------------
def zk_label(j):
    """Pretty label ``Z{j} ({name})`` for a pupil Noll index."""
    return f'Z{j} ({NOLL_NAMES.get(j, "?")})'


def stream_pdf_pages(panels, panels_per_page, ncols, page_size,
                     suptitle_fmt, output_pdf, plot_panel):
    """Stream a list of panel payloads into a multi-page PDF (bounded memory)."""
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    panels = list(panels)
    n_pages = (len(panels) + panels_per_page - 1) // panels_per_page
    with PdfPages(output_pdf) as pdf:
        for page in range(n_pages):
            chunk = panels[page * panels_per_page:(page + 1) * panels_per_page]
            nrows = (len(chunk) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=page_size,
                                     layout='constrained')
            axes = np.atleast_1d(axes).ravel()
            for ax, panel in zip(axes, chunk):
                plot_panel(ax, panel)
            for ax in axes[len(chunk):]:
                ax.set_visible(False)
            fig.suptitle(suptitle_fmt.format(page=page + 1, n_pages=n_pages),
                         fontsize=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'Wrote {n_pages} pages to {output_pdf}')
    return n_pages


def make_density_panel(*, x_plo=2.0, x_phi=98.0, y_plo=1.0, y_phi=99.0,
                       n_bins=100, xlabel='zk_A [μm]',
                       ylabel='zk_B − zk_A [μm]'):
    """Panel: 2-D density of Δ = y − x  vs  x, log color, OLS overlay.

    Payload is ``(j, x, y)`` with x = side-A zk, y = side-B zk.
    """
    def _panel(ax, payload):
        j, x, y = payload
        x = np.asarray(x, float)
        dy = np.asarray(y, float) - x
        mask = np.isfinite(x) & np.isfinite(dy)
        if not np.any(mask):
            ax.set_visible(False)
            return
        x, dy = x[mask], dy[mask]
        x_lo = float(np.nanpercentile(x, x_plo))
        x_hi = float(np.nanpercentile(x, x_phi))
        y_lo = float(np.nanpercentile(dy, y_plo))
        y_hi = float(np.nanpercentile(dy, y_phi))
        x_lo -= 0.02 * max(abs(x_hi - x_lo), 1e-6)
        x_hi += 0.02 * max(abs(x_hi - x_lo), 1e-6)
        y_lo -= 0.02 * max(abs(y_hi - y_lo), 1e-6)
        y_hi += 0.02 * max(abs(y_hi - y_lo), 1e-6)
        xbins = np.linspace(x_lo, x_hi, n_bins + 1)
        ybins = np.linspace(y_lo, y_hi, n_bins + 1)
        counts, xe, ye = np.histogram2d(x, dy, bins=[xbins, ybins])
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('white', alpha=0)
        pcm = ax.pcolormesh(xe, ye, np.ma.masked_where(counts == 0, counts).T,
                            cmap=cmap, norm=LogNorm(vmin=1, vmax=max(counts.max(), 2)),
                            shading='auto')
        plt.colorbar(pcm, ax=ax, shrink=0.85, label='donuts / bin')
        ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.7)
        in_box = (x >= x_lo) & (x <= x_hi) & (dy >= y_lo) & (dy <= y_hi)
        if int(in_box.sum()) >= 5:
            m, b = np.polyfit(x[in_box], dy[in_box], 1)
            ax.plot([x_lo, x_hi], [m * x_lo + b, m * x_hi + b], 'r-', lw=1.2,
                    alpha=0.9)
            r = float(np.corrcoef(x[in_box], dy[in_box])[0, 1])
            med = float(np.median(dy[in_box]))
            mad = float(np.median(np.abs(dy[in_box] - med)))
            label = (f'slope={m:+.4f}\nintercept={b:+.4f} μm\n'
                     f'median Δ={med:+.4f} μm\nσ_MAD={1.4826 * mad:.4f} μm\n'
                     f'r={r:+.4f}\nn={int(in_box.sum())}')
        else:
            label = f'n={int(in_box.sum())}'
        ax.text(0.04, 0.96, label, transform=ax.transAxes, ha='left', va='top',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                      alpha=0.85, lw=0))
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(zk_label(j), fontsize=10)
        ax.tick_params(labelsize=8)
    return _panel


def make_hexbin_panel(*, plo=2.0, phi=98.0, gridsize=60,
                      xlabel='zk_A [μm]', ylabel='zk_B [μm]'):
    """Panel: hexbin density of y vs x with a 1:1 line (donutalgo style).

    Payload is ``(j, x, y)`` with x = side-A zk, y = side-B zk.
    """
    def _panel(ax, payload):
        j, x, y = payload
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            ax.set_visible(False)
            return
        x, y = x[mask], y[mask]
        lo = float(min(np.percentile(x, plo), np.percentile(y, plo)))
        hi = float(max(np.percentile(x, phi), np.percentile(y, phi)))
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1,
                       extent=[lo, hi, lo, hi], bins='log', rasterized=True)
        plt.colorbar(hb, ax=ax, shrink=0.85, label='log10(N)')
        ax.plot([lo, hi], [lo, hi], 'r-', lw=1.0, alpha=0.8)
        diff = y - x
        ax.text(0.03, 0.97, f'Δ={np.mean(diff):.3f}\nσ={np.std(diff):.3f} μm',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(zk_label(j), fontsize=10)
        ax.tick_params(labelsize=8)
    return _panel


def make_diff_hist_panel(*, plo=1.0, phi=99.0, n_bins=100):
    """Panel: log-y histogram of Δ = y − x (donutalgo style).

    Payload is ``(j, x, y)`` with x = side-A zk, y = side-B zk.
    """
    def _panel(ax, payload):
        j, x, y = payload
        diff = np.asarray(y, float) - np.asarray(x, float)
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            ax.set_visible(False)
            return
        lo, hi = np.percentile(diff, [plo, phi])
        ax.hist(diff, bins=n_bins, range=(lo, hi), log=True,
                edgecolor='black', linewidth=0.3, alpha=0.7)
        ax.axvline(0, color='red', lw=0.8, ls='--')
        ax.text(0.97, 0.95, f'Δ={np.mean(diff):.3f}\nσ={np.std(diff):.3f} μm',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Δ [μm]', fontsize=9)
        ax.set_ylabel('count', fontsize=9)
        ax.set_title(zk_label(j), fontsize=10)
        ax.tick_params(labelsize=8)
    return _panel


def make_count_panel(*, fp_radius=1.8, n_bins=61, vmax_shared=None):
    """Panel: matched-donut count map per (thx, thy) bin, log color.

    Payload is ``(coord_label, thx, thy)``.
    """
    xb = np.linspace(-fp_radius, fp_radius, n_bins + 1)

    def _panel(ax, payload):
        coord_label, thx, thy = payload
        counts, _, _ = np.histogram2d(thy, thx, bins=[xb, xb])
        vmax = max(vmax_shared or counts.max(), 2)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('white', alpha=0)
        pcm = ax.pcolormesh(xb, xb, np.ma.masked_where(counts == 0, counts).T,
                            cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax),
                            shading='flat')
        ax.set_aspect('equal')
        ax.set_xlim(xb[0], xb[-1])
        ax.set_ylim(xb[0], xb[-1])
        plt.colorbar(pcm, ax=ax, shrink=0.75, label='matched donuts / bin')
        ax.set_xlabel(f'thy_{coord_label} [deg]', fontsize=9)
        ax.set_ylabel(f'thx_{coord_label} [deg]', fontsize=9)
        ax.set_title(f'matched donut counts — {coord_label}', fontsize=10)
        ax.tick_params(labelsize=8)
    return _panel


def make_map_panel(thx, thy, coord_label, *, fp_radius=1.8, n_bins=61,
                   plo=2.0, phi=98.0, value_label='Δ [μm]', title_suffix=''):
    """Panel: binned-median focal-plane map, diverging color (RdBu_r).

    Payload is ``(j, values)`` (one value per matched donut, aligned to thx/thy).
    """
    xb = np.linspace(-fp_radius, fp_radius, n_bins + 1)

    def _panel(ax, payload):
        j, values = payload
        values = np.asarray(values, float)
        if values.size == 0:
            ax.set_visible(False)
            return
        stat, _, _, _ = binned_statistic_2d(thy, thx, values,
                                             statistic='median', bins=[xb, xb])
        finite = stat[np.isfinite(stat)]
        if finite.size == 0:
            ax.set_visible(False)
            return
        lo = float(np.nanpercentile(finite, plo))
        hi = float(np.nanpercentile(finite, phi))
        vlim = max(abs(lo), abs(hi), 1e-6)
        pcm = ax.pcolormesh(xb, xb, stat.T, cmap='RdBu_r', shading='flat',
                            vmin=-vlim, vmax=vlim)
        ax.set_aspect('equal')
        ax.set_xlim(xb[0], xb[-1])
        ax.set_ylim(xb[0], xb[-1])
        plt.colorbar(pcm, ax=ax, shrink=0.75, label=value_label)
        ax.set_xlabel(f'thy_{coord_label} [deg]', fontsize=9)
        ax.set_ylabel(f'thx_{coord_label} [deg]', fontsize=9)
        ax.set_title(zk_label(j) + title_suffix, fontsize=10)
        ax.tick_params(labelsize=8)
    return _panel
