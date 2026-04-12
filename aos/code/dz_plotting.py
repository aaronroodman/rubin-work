"""Double Zernike plotting and analysis library for Rubin AOS wavefront data.

Contains all plotting functions for fit parameter visualization, single-image
residual maps, trio comparisons, thermal correlation analysis, and DZ
inter-correlation analysis.

Used by both the interactive notebook (intrinsics_plots.ipynb) and the
batch CLI script (run_dz_plots.py).
"""

import glob as glob_module
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from mpl_toolkits import axes_grid1
from scipy.stats import binned_statistic_2d
from astropy.table import QTable, vstack
from pathlib import Path

try:
    from .dz_fitting import focal_plane_zernike_basis, derive_noll_indices
except ImportError:
    from dz_fitting import focal_plane_zernike_basis, derive_noll_indices


# ============================================================
# Data loading helpers
# ============================================================

def discover_input_pairs(input_pattern):
    """Auto-discover HDF5 + fit parquet pairs from a glob pattern.

    Convention: HDF5 files contain donuts+visits tables, fit files are
    named {stem}_fits.parquet alongside the HDF5.

    Parameters
    ----------
    input_pattern : str
        Glob pattern for HDF5 files (e.g. 'output/*_20260315_*.hdf5').

    Returns
    -------
    pairs : list of (hdf5_path, fit_path) tuples
    """
    hdf5_files = sorted(glob_module.glob(input_pattern))
    pairs = []
    for h5 in hdf5_files:
        h5 = Path(h5)
        if not h5.is_file() or h5.suffix != '.hdf5':
            continue
        fit_file = h5.parent / f'{h5.stem}_fits.parquet'
        if fit_file.exists():
            pairs.append((str(h5), str(fit_file)))
        else:
            print(f"Warning: no fit file found for {h5} "
                  f"(expected {fit_file})")
    return pairs


def load_and_concatenate(pairs, coord_sys='OCS'):
    """Load multiple HDF5+fit parquet pairs and concatenate.

    Parameters
    ----------
    pairs : list of (hdf5_path, fit_path) tuples
    coord_sys : str

    Returns
    -------
    aosTable : QTable
        Concatenated donut-level table.
    fit_table : QTable
        Concatenated visit-level fit table.
    iZs : list of int
    iZidx : dict
    """
    aos_tables = []
    fit_tables = []
    ref_iZs = None

    for hdf5_file, fit_file in pairs:
        print(f"Loading: {hdf5_file}")
        aos = QTable.read(hdf5_file, path='donuts')
        print(f"  {len(aos)} donuts")
        ft = QTable.read(fit_file)
        print(f"  {len(ft)} visits from {fit_file}")

        # Derive Noll indices from this chunk
        zk = np.stack(aos[f'zk_{coord_sys}'])
        nZk = zk.shape[1]
        noll_arr = None
        if 'nollIndices' in ft.colnames:
            noll_arr = np.array(ft['nollIndices'][0])
        iZs, iZidx = derive_noll_indices(nZk, noll_arr)

        if ref_iZs is None:
            ref_iZs = iZs
        else:
            if iZs != ref_iZs:
                raise ValueError(f"Noll index mismatch: {iZs} vs {ref_iZs}")

        aos_tables.append(aos)
        fit_tables.append(ft)

    if not aos_tables:
        raise FileNotFoundError("No input pairs found")

    aosTable = vstack(aos_tables) if len(aos_tables) > 1 else aos_tables[0]
    fit_table = vstack(fit_tables) if len(fit_tables) > 1 else fit_tables[0]
    iZidx = {iZ: i for i, iZ in enumerate(ref_iZs)}

    print(f"\nCombined: {len(aosTable)} donuts, {len(fit_table)} visits, "
          f"{len(ref_iZs)} Zernike terms")
    return aosTable, fit_table, ref_iZs, iZidx


def reconstruct_zk_fit(aosTable, fit_table, coord_sys, iZs,
                        prefix='z1toz6', max_focal_noll=6):
    """Reconstruct per-donut fit values from stored coefficients.

    Evaluates the focal-plane Zernike basis at each donut's field angle
    and multiplies by the per-image fit coefficients from fit_table.
    Adds column 'zk_fit_{prefix}' to aosTable in-place.

    Parameters
    ----------
    aosTable : QTable
        Donut-level table with thx/thy columns.
    fit_table : QTable
        Visit-level fit table with coefficient columns.
    coord_sys : str
    iZs : list of int
    prefix : str
    max_focal_noll : int
    """
    day_obs_arr = np.array(aosTable['day_obs'])
    seq_num_arr = np.array(aosTable['seq_num'])
    thx_deg = np.rad2deg(np.array(aosTable[f'thx_{coord_sys}']))
    thy_deg = np.rad2deg(np.array(aosTable[f'thy_{coord_sys}']))

    ft_dobs = np.array(fit_table['day_obs'])
    ft_snum = np.array(fit_table['seq_num'])

    n_donuts = len(aosTable)
    n_zernikes = len(iZs)
    zk_fit = np.zeros((n_donuts, n_zernikes))

    # Build fit_table lookup: (day_obs, seq_num) -> row index
    ft_lookup = {}
    for i in range(len(fit_table)):
        ft_lookup[(int(ft_dobs[i]), int(ft_snum[i]))] = i

    # Get unique images
    images = sorted(set(zip(day_obs_arr.tolist(), seq_num_arr.tolist())))

    for dobs, snum in images:
        mask = (day_obs_arr == dobs) & (seq_num_arr == snum)
        ft_idx = ft_lookup.get((dobs, snum))
        if ft_idx is None:
            continue

        A, _ = focal_plane_zernike_basis(thx_deg[mask], thy_deg[mask],
                                         max_focal_noll)

        for j_idx, iZ in enumerate(iZs):
            coeffs = []
            for ci in range(1, max_focal_noll + 1):
                col = f'{prefix}_z{iZ}_c{ci}'
                if col in fit_table.colnames:
                    coeffs.append(float(fit_table[col][ft_idx]))
                else:
                    coeffs.append(0.0)
            zk_fit[mask, j_idx] = A @ np.array(coeffs)

    aosTable[f'zk_fit_{prefix}'] = list(zk_fit)
    print(f"Reconstructed zk_fit_{prefix}: {n_donuts} donuts, "
          f"{len(images)} images, {n_zernikes} Zernikes")


# ============================================================
# Utility functions
# ============================================================

def get_zernike(table, column_name, iZ, iZs, iZidx):
    """Extract a single Zernike term from an array column."""
    if iZ not in iZidx:
        raise ValueError(f"Zernike Z{iZ} not in table. Available: {iZs}")
    zk_array = np.stack(table[column_name])
    return zk_array[:, iZidx[iZ]]


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


# ============================================================
# Internal helpers
# ============================================================

def _identify_fam_blocks(fit_table, min_block_size=3):
    """Identify FAM visits belonging to contiguous blocks.

    A block is a group of visits within the same day_obs where consecutive
    seq_nums differ by exactly 3 (the FAM triplet spacing). Only blocks
    with at least min_block_size visits are kept.
    """
    day_obs_arr = np.array(fit_table['day_obs'])
    seq_num_arr = np.array(fit_table['seq_num'])
    in_block = np.zeros(len(fit_table), dtype=bool)

    for dobs in sorted(set(day_obs_arr.tolist())):
        day_mask = day_obs_arr == dobs
        day_indices = np.where(day_mask)[0]
        day_seqs = seq_num_arr[day_indices]
        order = np.argsort(day_seqs)
        sorted_indices = day_indices[order]
        sorted_seqs = day_seqs[order]

        blocks = [[0]]
        for k in range(1, len(sorted_seqs)):
            if sorted_seqs[k] - sorted_seqs[k - 1] == 3:
                blocks[-1].append(k)
            else:
                blocks.append([k])

        for block in blocks:
            if len(block) >= min_block_size:
                for k in block:
                    in_block[sorted_indices[k]] = True

    return in_block


def _fmt_angle(v):
    """Format angle for display: 2 sig figs, no scientific notation."""
    if v == 0:
        return '0'
    s = f'{v:.2g}'
    if 'e' in s:
        return f'{v:.0f}'
    return s


def _build_pointing_groups(fit_table_day, visit_info, bin_size=3.0,
                           verbose=False):
    """Build pointing groups by binning (alt, rotAngle) to nearest bin_size degrees."""
    ft_dobs = np.array(fit_table_day['day_obs'])
    ft_snum = np.array(fit_table_day['seq_num'])

    # Check if pointing columns exist (handle both naming conventions)
    has_alt = 'alt' in visit_info.colnames
    rot_col = ('rotAngle' if 'rotAngle' in visit_info.colnames
               else 'rotator_angle' if 'rotator_angle' in visit_info.colnames
               else None)
    has_rot = rot_col is not None

    if not has_alt:
        # No pointing info — put everything in one group
        group_indices = {(0, 0): list(range(len(fit_table_day)))}
        group_labels = {(0, 0): 'all'}
        return group_indices, group_labels

    vi_lookup = {}
    vi_dobs = np.array(visit_info['day_obs'])
    vi_snum = np.array(visit_info['seq_num'])
    vi_az = np.array(visit_info['az']) if 'az' in visit_info.colnames else np.full(len(visit_info), np.nan)
    vi_alt = np.array(visit_info['alt'])
    vi_rot = np.array(visit_info[rot_col]) if has_rot else np.full(len(visit_info), np.nan)
    for i in range(len(visit_info)):
        vi_lookup[(int(vi_dobs[i]), int(vi_snum[i]))] = (
            float(vi_az[i]), float(vi_alt[i]), float(vi_rot[i]))

    group_indices = {}
    group_raw_values = {}
    for idx in range(len(fit_table_day)):
        key = (int(ft_dobs[idx]), int(ft_snum[idx]))
        if key in vi_lookup:
            az, alt, rot = vi_lookup[key]
            az_wrapped = ((az + 180) % 360) - 180
            if np.isnan(alt) or np.isnan(rot):
                gkey = (np.nan, np.nan)
            else:
                gkey = (round(alt / bin_size) * bin_size,
                        round(rot / bin_size) * bin_size)
        else:
            az, alt, rot = np.nan, np.nan, np.nan
            gkey = (np.nan, np.nan)
        group_indices.setdefault(gkey, []).append(idx)
        group_raw_values.setdefault(gkey, []).append((az, alt, rot))

    group_labels = {}
    for gkey in group_indices:
        if np.isnan(gkey[0]):
            group_labels[gkey] = 'unknown'
        else:
            group_labels[gkey] = (f'el={_fmt_angle(gkey[0])} '
                                  f'rot={_fmt_angle(gkey[1])}')

    if verbose:
        print(f"\nPointing groups (bin_size={bin_size}\u00b0):")
        print(f"{'Group':>6s}  {'Label':30s}  {'N_visits':>8s}  "
              f"{'Az range':>12s}  {'El range':>12s}  {'Rot range':>12s}")
        print('-' * 95)
        for gi, gkey in enumerate(sorted(group_indices.keys(),
                                         key=lambda g: group_indices[g][0])):
            n = len(group_indices[gkey])
            raw = np.array(group_raw_values[gkey])
            az_lo, az_hi = raw[:, 0].min(), raw[:, 0].max()
            el_lo, el_hi = raw[:, 1].min(), raw[:, 1].max()
            rot_lo, rot_hi = raw[:, 2].min(), raw[:, 2].max()
            print(f'{gi:6d}  {group_labels[gkey]:30s}  {n:8d}  '
                  f'{az_lo:5.1f}-{az_hi:5.1f}  '
                  f'{el_lo:5.1f}-{el_hi:5.1f}  '
                  f'{rot_lo:6.1f}-{rot_hi:5.1f}')
        print()

    return group_indices, group_labels


# ============================================================
# Core plotting functions
# ============================================================

def plot_fit_params_and_residuals(fit_table_day, aosTable_matched,
                                  plot_mask_day, day_obs_list,
                                  fit_prefix, iZs_fit_plot, iZs_hist,
                                  iZs, iZidx, coord_sys,
                                  visit_info=None,
                                  position_group_tol=3.0,
                                  output_dir='.', show=True):
    """Plot fit coefficients vs image and residual histograms.

    Creates a multi-page PDF with coefficient time series (one page per
    focal coefficient) and residual histograms.

    Parameters
    ----------
    fit_table_day : QTable
        Fit parameters table filtered to selected days.
    aosTable_matched : QTable
        Full matched donut table (with zk_fit column).
    plot_mask_day : ndarray (bool)
        Mask into aosTable_matched for the selected days.
    day_obs_list : list of int
    fit_prefix : str
    iZs_fit_plot : list of int
        Zernike indices for coefficient plots.
    iZs_hist : list of int
        Zernike indices for residual histograms.
    iZs : list of int
        All available Noll indices.
    iZidx : dict
    coord_sys : str
    visit_info : QTable, optional
    position_group_tol : float
    output_dir : str
    show : bool
    """
    if len(day_obs_list) == 1:
        day_label = str(day_obs_list[0])
    elif len(day_obs_list) <= 4:
        day_label = ', '.join(str(d) for d in day_obs_list)
    else:
        day_label = (f'{day_obs_list[0]}...{day_obs_list[-1]} '
                     f'({len(day_obs_list)} days)')

    file_suffix = str(day_obs_list[0]) if len(day_obs_list) == 1 else 'all'

    first_iZ = iZs_fit_plot[0]
    sample_cols = [c for c in fit_table_day.colnames
                   if c.startswith(f'{fit_prefix}_z{first_iZ}_c')
                   and not c.endswith('_err')]
    n_coeffs = len(sample_cols)

    param_labels = ['k1 (piston)', 'k2 (tilt)', 'k3 (tip)',
                    'k4 (defocus)', 'k5 (astig45)', 'k6 (astig0)'][:n_coeffs]
    param_units = ['\u03bcm'] * n_coeffs

    n_plots = len(iZs_fit_plot)
    ncols = 4
    nrows = (n_plots + ncols - 1) // ncols

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'p', 'h']
    if visit_info is not None and len(fit_table_day) > 0:
        group_indices, group_labels = _build_pointing_groups(
            fit_table_day, visit_info, bin_size=position_group_tol)
        sorted_groups = sorted(group_indices.keys(),
                               key=lambda g: group_indices[g][0])
    else:
        group_indices = None

    is_hi = (iZs_fit_plot[0] > 15) if len(iZs_fit_plot) > 0 else False
    hi_suffix = '_hi' if is_hi else ''
    pdf_path = (f'{output_dir}/fit_params_resid_{fit_prefix}'
                f'{hi_suffix}_{file_suffix}.pdf')

    with PdfPages(pdf_path) as pdf:
        image_idx = np.arange(len(fit_table_day))

        for ci in range(n_coeffs):
            cname = f'c{ci + 1}'
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(18, 10 * nrows / 3),
                                     sharex=True)
            if nrows == 1:
                axes = axes[np.newaxis, :]
            fig.suptitle(
                f'{fit_prefix} Fit: {param_labels[ci]} vs Image '
                f'(day_obs: {day_label})', fontsize=14)

            for ax_idx, iZ in enumerate(iZs_fit_plot):
                ax = axes[ax_idx // ncols, ax_idx % ncols]
                col = f'{fit_prefix}_z{iZ}_{cname}'
                err_col = f'{fit_prefix}_z{iZ}_{cname}_err'
                if col not in fit_table_day.colnames:
                    continue
                vals = np.array(fit_table_day[col])
                errs = (np.array(fit_table_day[err_col])
                        if err_col in fit_table_day.colnames else None)

                if group_indices is not None and len(sorted_groups) > 1:
                    ax.plot(image_idx, vals, '-', color='gray',
                            linewidth=0.5, alpha=0.5, zorder=1)
                    for gi, gkey in enumerate(sorted_groups):
                        idxs = np.array(group_indices[gkey])
                        m = markers[gi % len(markers)]
                        label = group_labels[gkey] if ax_idx == 0 else None
                        if errs is not None:
                            ax.errorbar(image_idx[idxs], vals[idxs],
                                        yerr=errs[idxs],
                                        fmt=m, markersize=4, linewidth=0,
                                        elinewidth=0.5, capsize=0, alpha=0.8,
                                        label=label, zorder=2)
                        else:
                            ax.plot(image_idx[idxs], vals[idxs], m,
                                    markersize=4, alpha=0.8,
                                    label=label, zorder=2)
                else:
                    if errs is not None:
                        ax.errorbar(image_idx, vals, yerr=errs,
                                    fmt='o-', markersize=3, linewidth=0.8,
                                    elinewidth=0.5, capsize=0, alpha=0.8)
                    else:
                        ax.plot(image_idx, vals, 'o-', markersize=3)

                ax.set_title(f'Z{iZ}')
                ax.set_ylabel(param_units[ci])
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                if ax_idx // ncols == nrows - 1:
                    ax.set_xlabel('Image index')

                if len(day_obs_list) > 1:
                    ft_dobs = np.array(fit_table_day['day_obs'])
                    for d_idx in range(1, len(ft_dobs)):
                        if ft_dobs[d_idx] != ft_dobs[d_idx - 1]:
                            ax.axvline(d_idx - 0.5, color='black',
                                       linewidth=0.5, linestyle=':',
                                       alpha=0.5, zorder=0)
                            if ax_idx == 0:
                                ax.text(d_idx - 0.5, ax.get_ylim()[1],
                                        str(ft_dobs[d_idx]),
                                        fontsize=5, rotation=90,
                                        va='top', ha='right', alpha=0.6)

            for idx in range(n_plots, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            if group_indices is not None and len(sorted_groups) > 1:
                max_legend = 8
                handles, labels = axes[0, 0].get_legend_handles_labels()
                if len(handles) > max_legend:
                    axes[0, 0].legend(
                        handles[:max_legend], labels[:max_legend],
                        fontsize=6, loc='best', ncol=1,
                        handletextpad=0.3, borderpad=0.3,
                        title=f'+{len(handles) - max_legend} more',
                        title_fontsize=5)
                else:
                    axes[0, 0].legend(fontsize=6, loc='best', ncol=1,
                                      handletextpad=0.3, borderpad=0.3)

            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        # Residual histograms
        zk_data_plot = np.stack(
            aosTable_matched[f'zk_{coord_sys}'])[plot_mask_day]
        zk_model_plot = np.stack(
            aosTable_matched[f'zk_intrinsic_{coord_sys}'])[plot_mask_day]
        fit_col = f'zk_fit_{fit_prefix}'
        if fit_col in aosTable_matched.colnames:
            zk_fit_plot = np.stack(
                aosTable_matched[fit_col])[plot_mask_day]
        else:
            zk_fit_plot = np.stack(
                aosTable_matched['zk_fit'])[plot_mask_day]

        n_hist = len(iZs_hist)
        nrows_h = (n_hist + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows_h, ncols,
                                 figsize=(18, 10 * nrows_h / 3))
        if nrows_h == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            f'{fit_prefix} Fit Residuals: data - model - fit '
            f'(day_obs: {day_label})', fontsize=14)

        hist_range = (-1.0, 1.0)
        n_bins = 100

        for ax_idx, iZ in enumerate(iZs_hist):
            ax = axes[ax_idx // ncols, ax_idx % ncols]
            j = iZidx[iZ]
            resid = (zk_data_plot[:, j] - zk_model_plot[:, j]
                     - zk_fit_plot[:, j])
            n_total = len(resid)
            n_in = np.sum((resid >= hist_range[0])
                          & (resid <= hist_range[1]))
            n_out = n_total - n_in
            ax.hist(resid, bins=n_bins, range=hist_range, log=True,
                    edgecolor='black', linewidth=0.3, alpha=0.7)
            ax.set_title(f'Z{iZ}')
            ax.set_xlabel('\u03bcm')
            ax.set_ylabel('Count')
            ax.set_xlim(hist_range)
            std_val = np.std(resid)
            ax.text(0.97, 0.95,
                    f'\u03c3={std_val:.3f} \u03bcm\n{n_out}/{n_total} outside',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        for idx in range(n_hist, nrows_h * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved: {pdf_path}")


def plot_single_image_residual_grid(aosTable_matched, day_obs, seq_num,
                                     iZs, iZidx, coord_sys,
                                     band='', alt=None, az=None,
                                     rotAngle=None,
                                     fit_table=None,
                                     fit_prefix='z1toz6',
                                     iZs_plot=None,
                                     n_steps=16, statistic='median',
                                     plo=2.0, phi=98.0,
                                     k1_range=1.0, fpradius=1.8,
                                     clim_dict=None,
                                     output_dir='.', show=True):
    """Plot a 3x4 grid of binned residual maps for a single image."""
    if iZs_plot is None:
        iZs_plot = iZs[:12]

    dobs_arr = np.array(aosTable_matched['day_obs'])
    snum_arr = np.array(aosTable_matched['seq_num'])
    img_mask = (dobs_arr == day_obs) & (snum_arr == seq_num)
    n_donuts = np.sum(img_mask)
    if n_donuts == 0:
        return None

    k1_vals = {}
    if fit_table is not None:
        ft_dobs = np.array(fit_table['day_obs'])
        ft_snum = np.array(fit_table['seq_num'])
        ft_mask = (ft_dobs == day_obs) & (ft_snum == seq_num)
        if np.sum(ft_mask) == 1:
            ft_row = fit_table[ft_mask][0]
            for iZ in iZs_plot:
                col = f'{fit_prefix}_z{iZ}_c1'
                if col in fit_table.colnames:
                    k1_vals[iZ] = float(ft_row[col])

    xval = np.rad2deg(
        np.array(aosTable_matched[f'thy_{coord_sys}_extra'])[img_mask])
    yval = np.rad2deg(
        np.array(aosTable_matched[f'thx_{coord_sys}_extra'])[img_mask])
    zk_data_img = np.stack(
        aosTable_matched[f'zk_{coord_sys}'])[img_mask]
    zk_model_img = np.stack(
        aosTable_matched[f'zk_intrinsic_{coord_sys}'])[img_mask]
    fit_col = f'zk_fit_{fit_prefix}'
    if fit_col in aosTable_matched.colnames:
        zk_fit_img = np.stack(aosTable_matched[fit_col])[img_mask]
    else:
        zk_fit_img = np.stack(aosTable_matched['zk_fit'])[img_mask]

    bins_edge = np.linspace(-fpradius, fpradius, n_steps)
    n_rows, n_cols = 3, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))

    band_str = f'  band={band}' if band else ''
    meta_str = ''
    if az is not None:
        meta_str = f'  az={az:.1f}  el={alt:.1f}  rot={rotAngle:.1f}'
    fig.suptitle(
        f'Single-Image Residuals: day_obs={day_obs}  seq_num={seq_num}'
        f'{band_str}{meta_str}  ({n_donuts} donuts)',
        fontsize=13, y=0.98)

    for ax_idx, iZ in enumerate(iZs_plot):
        row = ax_idx // n_cols
        col_idx = ax_idx % n_cols
        ax = axes[row, col_idx]
        j = iZidx[iZ]
        resid = zk_data_img[:, j] - zk_model_img[:, j] - zk_fit_img[:, j]

        stat_val, _, _, _ = binned_statistic_2d(
            xval, yval, resid, statistic=statistic,
            bins=[bins_edge, bins_edge])
        if clim_dict is not None and iZ in clim_dict:
            vmin, vmax = clim_dict[iZ]
        else:
            finite_vals = stat_val[np.isfinite(stat_val)]
            vmin, vmax = (np.percentile(finite_vals, [plo, phi])
                          if len(finite_vals) > 0 else (-1.0, 1.0))

        im = ax.imshow(stat_val.T, origin='lower',
                       extent=[-fpradius, fpradius, -fpradius, fpradius],
                       cmap='RdBu_r', interpolation='none', aspect='equal',
                       vmin=vmin, vmax=vmax)
        add_colorbar(im, aspect=25, pad_fraction=0.3)
        ax.set_title(f'Z{iZ}', fontsize=11, loc='center')
        ax.set_xlim(-fpradius, fpradius)
        ax.set_ylim(-fpradius, fpradius)
        if col_idx == 0:
            ax.set_ylabel(f'thx_{coord_sys} [deg]')
        if row == n_rows - 1:
            ax.set_xlabel(f'thy_{coord_sys} [deg]')

        if iZ in k1_vals:
            k1_val = np.clip(k1_vals[iZ], -k1_range, k1_range)
            bar_anchor, bar_y, bar_h = 0.25, 1.04, 0.03
            bar_dx = k1_val / k1_range * 0.25
            color = 'steelblue' if k1_val >= 0 else 'firebrick'
            rect_x = min(bar_anchor, bar_anchor + bar_dx)
            rect_w = abs(bar_dx)
            if rect_w > 0.001:
                rect = Rectangle(
                    (rect_x, bar_y - bar_h / 2), rect_w, bar_h,
                    transform=ax.transAxes, clip_on=False,
                    facecolor=color, edgecolor='none', alpha=0.7)
                ax.add_patch(rect)
            ax.plot([bar_anchor, bar_anchor],
                    [bar_y - bar_h / 2, bar_y + bar_h / 2],
                    transform=ax.transAxes, color='black', linewidth=0.8,
                    clip_on=False)
            ax.text(bar_anchor + bar_dx, bar_y + bar_h / 2 + 0.01,
                    f'{k1_vals[iZ]:.2f}', transform=ax.transAxes,
                    fontsize=7, ha='center', va='bottom', color=color)

    plt.tight_layout()
    output_file = (f'{output_dir}/single_image_resid_'
                   f'{day_obs}_{seq_num}.jpg')
    fig.savefig(output_file, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_file


def plot_zernike_trio(aosTable_matched, iZ, iZs, iZidx, coord_sys,
                      plo=4.0, phi=96.0, statistic='median',
                      fit_prefix='z1toz3',
                      output_dir='.', date_range_str='', pdf=None,
                      show=True):
    """Create trio of plots for a Zernike term: Data, Model, Data-Model."""
    nsteps = 18 * 4 + 1
    fpradius = 1.8
    xbins = np.linspace(-fpradius, fpradius, nsteps)
    ybins = np.linspace(-fpradius, fpradius, nsteps)

    xval = np.rad2deg(aosTable_matched[f'thy_{coord_sys}_extra'])
    yval = np.rad2deg(aosTable_matched[f'thx_{coord_sys}_extra'])

    zk_data_all = np.stack(aosTable_matched[f'zk_{coord_sys}'])
    fit_col = f'zk_fit_{fit_prefix}'
    if fit_col in aosTable_matched.colnames:
        zk_fit_all = np.stack(aosTable_matched[fit_col])
    else:
        zk_fit_all = np.stack(aosTable_matched['zk_fit'])
    zk_model_all = np.stack(
        aosTable_matched[f'zk_intrinsic_{coord_sys}'])

    zval_data = zk_data_all[:, iZidx[iZ]] - zk_fit_all[:, iZidx[iZ]]
    zval_model = zk_model_all[:, iZidx[iZ]]
    zval_residual = zval_data - zval_model

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, zval, cmap, title_str in [
        (axes[0], zval_data, 'viridis',
         f'Z{iZ} Data (linear fit subtracted)'),
        (axes[1], zval_model, 'viridis', f'Z{iZ} Model Intrinsic'),
        (axes[2], zval_residual, 'RdBu_r', f'Z{iZ} Data - Model'),
    ]:
        stat_val, _, _, _ = binned_statistic_2d(
            xval, yval, zval, statistic=statistic, bins=[xbins, ybins])
        vmin, vmax = np.nanpercentile(zval, [plo, phi])
        im = ax.imshow(stat_val.T, origin='lower',
                       extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                       cmap=cmap, interpolation='none', aspect='equal',
                       vmin=vmin, vmax=vmax)
        add_colorbar(im, label='\u03bcm')
        ax.set_xlabel(f'thy_{coord_sys} [deg]')
        ax.set_ylabel(f'thx_{coord_sys} [deg]')
        ax.set_title(title_str)
        ax.set_aspect('equal')

    title = f'Z{iZ} Comparison ({statistic})'
    if date_range_str:
        title += f' ({date_range_str})'
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    else:
        output_file = f'{output_dir}/Z{iZ}_trio_comparison.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Z{iZ}: Data \u03c3={np.nanstd(zval_data):.2f}, "
          f"Model \u03c3={np.nanstd(zval_model):.2f}, "
          f"Resid \u03c3={np.nanstd(zval_residual):.2f} \u03bcm")


# ============================================================
# Thermal correlation analysis
# ============================================================

DEFAULT_THERMAL_VARS = [
    "cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
    "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t",
    "x_gradient", "y_gradient", "z_gradient", "radial_gradient",
    "tma_truss_temp_pxpy", "tma_truss_temp_mxmy",
]


def dz_col_name(k, j, prefix='z1toz6'):
    """Column name for DZ coefficient (k, j)."""
    return f"{prefix}_z{j}_c{k}"


def plot_thermal_scatter_grid(fit_df, dz_terms, thermal_vars=None,
                              dz_prefix='z1toz6', pdf=None, show=True):
    """For each DZ term, create a page of scatter plots vs thermal vars.

    Parameters
    ----------
    fit_df : DataFrame
    dz_terms : list of (k, j) tuples
    thermal_vars : list of str, optional
    dz_prefix : str
    pdf : PdfPages or None
    show : bool
    """
    if thermal_vars is None:
        thermal_vars = [tv for tv in DEFAULT_THERMAL_VARS
                        if tv in fit_df.columns]

    n_thermal = len(thermal_vars)
    ncols = 4
    nrows = int(np.ceil(n_thermal / ncols))

    for k, j in dz_terms:
        dz_col = dz_col_name(k, j, dz_prefix)
        if dz_col not in fit_df.columns:
            print(f"  Skipping ({k},{j}): column {dz_col} not found")
            continue

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
        fig.suptitle(
            f'DZ coefficient (k={k}, j={j})  =  {dz_col}\n'
            f'vs thermal variables',
            fontsize=13, y=1.01)

        ax_array = axes.ravel()
        for idx, tv in enumerate(thermal_vars):
            ax = ax_array[idx]
            mask = fit_df[dz_col].notna() & fit_df[tv].notna()
            x = fit_df.loc[mask, tv].values
            y = fit_df.loc[mask, dz_col].values

            ax.scatter(x, y, s=12, alpha=0.7, edgecolors='none')

            if len(x) > 2:
                coeffs = np.polyfit(x, y, 1)
                r = np.corrcoef(x, y)[0, 1]
                xfit = np.linspace(x.min(), x.max(), 50)
                ax.plot(xfit, np.polyval(coeffs, xfit), 'r-', lw=1.5,
                        alpha=0.8)
                ax.text(0.05, 0.92, f'r = {r:.3f}',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  alpha=0.8))

            ax.set_xlabel(tv, fontsize=8)
            ax.set_ylabel(f'({k},{j}) [\u03bcm]', fontsize=8)
            ax.tick_params(labelsize=7)

        for idx in range(n_thermal, len(ax_array)):
            ax_array[idx].set_visible(False)

        fig.tight_layout()
        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def run_thermal_pca(fit_df, thermal_vars=None, n_components=5):
    """Standardize thermal variables and run PCA.

    Returns
    -------
    scaler : StandardScaler
    pca : PCA
    pc_scores : ndarray
    valid_mask : ndarray (bool)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if thermal_vars is None:
        thermal_vars = [tv for tv in DEFAULT_THERMAL_VARS
                        if tv in fit_df.columns]

    thermal_df = fit_df[thermal_vars]
    valid_mask = thermal_df.notna().all(axis=1).values

    scaler = StandardScaler()
    thermal_std = scaler.fit_transform(thermal_df.loc[valid_mask].values)

    pca = PCA(n_components=min(n_components, thermal_std.shape[1]))
    pc_scores = pca.fit_transform(thermal_std)

    print(f"PCA on {thermal_std.shape[1]} thermal variables, "
          f"{thermal_std.shape[0]} visits")
    cum_var = 0.0
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cum_var += ev
        print(f"  PC{i + 1}: {ev:.3f}  (cumulative: {cum_var:.3f})")

    return scaler, pca, pc_scores, valid_mask


def plot_pca_loadings(pca, thermal_vars, pdf=None, show=True):
    """Heatmap of PCA loadings (thermal variables onto PCs)."""
    loadings = pd.DataFrame(
        pca.components_.T,
        index=thermal_vars,
        columns=[f'PC{i + 1}' for i in range(pca.n_components_)])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(loadings.values, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1)
    ax.set_xticks(range(pca.n_components_))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticks(range(len(thermal_vars)))
    ax.set_yticklabels(thermal_vars, fontsize=9)
    ax.set_title('PCA Loadings: Thermal Variables onto Principal Components')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Loading')

    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run_pcr_analysis(fit_df, dz_terms, pca, pc_scores, valid_mask,
                     dz_prefix='z1toz6'):
    """Principal Component Regression of DZ coefficients on thermal PCs.

    Returns list of dicts with keys: k, j, dz_col, R2, R2_adj, n, beta.
    """
    X = np.column_stack([np.ones(pc_scores.shape[0]), pc_scores])
    results = []

    for k, j in dz_terms:
        dz_col = dz_col_name(k, j, dz_prefix)
        if dz_col not in fit_df.columns:
            continue

        y = fit_df.loc[valid_mask, dz_col].values
        y_mask = ~np.isnan(y)
        X_fit = X[y_mask]
        y_fit = y[y_mask]

        if len(y_fit) < X_fit.shape[1] + 1:
            continue

        beta, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
        y_pred = X_fit @ beta
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        n = len(y_fit)
        p = X_fit.shape[1] - 1
        r2_adj = (1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
                  if n > p + 1 else r2)

        results.append({
            'k': k, 'j': j, 'dz_col': dz_col,
            'R2': r2, 'R2_adj': r2_adj, 'n': n,
            'beta': beta, 'y_fit': y_fit, 'y_pred': y_pred,
        })

        print(f"({k},{j})  R\u00b2 = {r2:.3f}  R\u00b2_adj = {r2_adj:.3f}  "
              f"(n={n})")

    return results


def plot_pcr_results(pcr_results, pdf=None, show=True):
    """Plot predicted vs actual and residual histograms for PCR results."""
    for res in pcr_results:
        y_fit = res['y_fit']
        y_pred = res['y_pred']
        resid = y_fit - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.scatter(y_fit, y_pred, s=15, alpha=0.7, edgecolors='none')
        lims = [min(y_fit.min(), y_pred.min()),
                max(y_fit.max(), y_pred.max())]
        ax1.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax1.set_xlabel(f'Measured ({res["dz_col"]}) [\u03bcm]')
        ax1.set_ylabel('PCR Predicted [\u03bcm]')
        ax1.set_title(f'(k={res["k"]}, j={res["j"]})  '
                       f'R\u00b2 = {res["R2"]:.3f}')
        ax1.set_aspect('equal', adjustable='datalim')

        ax2.hist(resid, bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residual [\u03bcm]')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Residuals  \u03c3 = {resid.std():.4f} \u03bcm')

        fig.suptitle(
            f'PCR: DZ (k={res["k"]}, j={res["j"]}) from thermal PCs',
            y=1.02)
        fig.tight_layout()

        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_thermal_importance(pcr_results, pca, thermal_vars=None,
                            pdf=None, show=True):
    """Bar charts of effective thermal variable weights per DZ term."""
    if thermal_vars is None:
        thermal_vars = DEFAULT_THERMAL_VARS

    for res in pcr_results:
        pc_betas = res['beta'][1:]
        thermal_weights = (pca.components_[:len(pc_betas)].T
                           @ pc_betas)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        colors = ['steelblue' if w >= 0 else 'salmon'
                  for w in thermal_weights]
        ax.barh(range(len(thermal_vars)), thermal_weights,
                color=colors, edgecolor='black', lw=0.5)
        ax.set_yticks(range(len(thermal_vars)))
        ax.set_yticklabels(thermal_vars, fontsize=9)
        ax.set_xlabel('Effective weight (standardized units)')
        ax.set_title(
            f'Thermal variable importance for DZ '
            f'(k={res["k"]}, j={res["j"]})\n'
            f'R\u00b2_adj = {res["R2_adj"]:.3f}')
        ax.axvline(0, color='black', lw=0.5)
        fig.tight_layout()

        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# DZ inter-correlation analysis
# ============================================================

def get_all_dz_columns(fit_table, prefix='z1toz6'):
    """Find all DZ coefficient columns (excluding _err and _scale).

    Returns
    -------
    dz_cols : list of str
    dz_labels : list of str
        Labels in '(k,j)' format.
    """
    colnames = (fit_table.colnames if hasattr(fit_table, 'colnames')
                else list(fit_table.columns))
    dz_cols = []
    dz_labels = []
    for col in sorted(colnames):
        if not col.startswith(f'{prefix}_z'):
            continue
        if col.endswith('_err') or col.endswith('_scale'):
            continue
        if '_bad_fit' in col:
            continue
        # Parse (k, j) from z1toz6_z{j}_c{k}
        parts = col.replace(f'{prefix}_z', '').split('_c')
        if len(parts) == 2:
            try:
                j = int(parts[0])
                k = int(parts[1])
                dz_cols.append(col)
                dz_labels.append(f'({k},{j})')
            except ValueError:
                continue
    return dz_cols, dz_labels


def compute_dz_correlation_matrix(fit_df, dz_cols):
    """Compute pairwise Pearson correlation matrix for DZ coefficients."""
    data = fit_df[dz_cols].values
    valid = ~np.isnan(data).any(axis=1)
    data_clean = data[valid]
    return np.corrcoef(data_clean, rowvar=False)


def plot_dz_correlation_heatmap(corr_matrix, dz_labels,
                                pdf=None, show=True):
    """Full correlation matrix heatmap with annotations for |r| > 0.5."""
    n = len(dz_labels)
    figsize = max(12, n * 0.15)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.9))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='none')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Pearson r')

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(dz_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(dz_labels, fontsize=5)

    # Annotate cells with |r| > 0.5
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr_matrix[i, j]) > 0.5:
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=4,
                        color='white' if abs(corr_matrix[i, j]) > 0.7
                        else 'black')

    # Add separator lines between pupil Zernike groups
    # Parse j values from labels to find boundaries
    prev_j = None
    for idx, label in enumerate(dz_labels):
        # label is '(k,j)' format
        j_val = int(label.split(',')[1].rstrip(')'))
        if prev_j is not None and j_val != prev_j:
            ax.axhline(idx - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(idx - 0.5, color='black', linewidth=0.5, alpha=0.3)
        prev_j = j_val

    ax.set_title('DZ Coefficient Inter-Correlations', fontsize=14)
    fig.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def get_top_correlated_pairs(corr_matrix, dz_cols, dz_labels, top_n=10):
    """Extract top N most correlated (k,j) pairs by |r|.

    Returns list of (col_i, col_j, label_i, label_j, r_value) sorted
    by |r| descending. Excludes self-correlations.
    """
    n = len(dz_cols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if np.isfinite(r):
                pairs.append((dz_cols[i], dz_cols[j],
                              dz_labels[i], dz_labels[j], r))

    pairs.sort(key=lambda x: abs(x[4]), reverse=True)
    return pairs[:top_n]


def plot_dz_scatter_top_pairs(fit_df, top_pairs, pdf=None, show=True):
    """Scatter plots for the top N most correlated DZ coefficient pairs."""
    n_pairs = len(top_pairs)
    if n_pairs == 0:
        return

    ncols = 2
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for idx, (col_i, col_j, lab_i, lab_j, r_val) in enumerate(top_pairs):
        ax = axes[idx // ncols, idx % ncols]
        mask = fit_df[col_i].notna() & fit_df[col_j].notna()
        x = fit_df.loc[mask, col_i].values
        y = fit_df.loc[mask, col_j].values

        ax.scatter(x, y, s=12, alpha=0.7, edgecolors='none')

        if len(x) > 2:
            coeffs = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 50)
            ax.plot(xfit, np.polyval(coeffs, xfit), 'r-', lw=1.5,
                    alpha=0.8)

        ax.set_xlabel(f'{lab_i} [\u03bcm]', fontsize=9)
        ax.set_ylabel(f'{lab_j} [\u03bcm]', fontsize=9)
        ax.set_title(f'{lab_i} vs {lab_j}  r = {r_val:.3f}', fontsize=10)
        ax.tick_params(labelsize=8)

    # Hide unused
    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Top DZ Coefficient Correlations', fontsize=14)
    fig.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
