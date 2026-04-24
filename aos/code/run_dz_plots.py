#!/usr/bin/env python3
"""Generate Double Zernike analysis plots from pre-computed fit results.

Produces fit parameter PDFs, single-image residual maps (with optional
movie), and trio comparison plots. Designed to run outside Jupyter to
avoid memory limitations.

Usage:
    python run_dz_plots.py input.hdf5 fits.parquet
    python run_dz_plots.py input.hdf5 fits.parquet --no-single-image
    python run_dz_plots.py input.hdf5 fits.parquet --coord-sys CCS
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.table import QTable
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dz_fitting import derive_noll_indices, flag_bad_fits
from dz_plotting import (
    reconstruct_zk_fit,
    _identify_fam_blocks,
    plot_fit_params_and_residuals,
    plot_single_image_residual_grid,
    plot_zernike_trio,
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate DZ analysis plots from pre-computed fits.')
    parser.add_argument('input_file',
                        help='Input HDF5 file with donuts+visits tables')
    parser.add_argument('fit_file',
                        help='Input fit table parquet file (merged fits+visits)')
    parser.add_argument('--coord-sys', default='OCS',
                        choices=['OCS', 'CCS'])
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as fit_file)')
    parser.add_argument('--no-single-image', action='store_true',
                        help='Skip single-image residual maps')
    parser.add_argument('--no-fit-params', action='store_true',
                        help='Skip fit parameter PDFs')
    parser.add_argument('--no-trio', action='store_true',
                        help='Skip trio comparison plots')
    parser.add_argument('--bad-fit-threshold', type=float, default=2.0)
    parser.add_argument('--min-donuts', type=int, default=200)
    parser.add_argument('--date-range-str', default=None)

    args = parser.parse_args()
    coord_sys = args.coord_sys

    # Load donuts table — detect format by suffix.
    #   * .parquet  -> streaming donut parquet produced by run_mktable
    #     (astropy concatenates row groups on read)
    #   * .hdf5     -> legacy single-file HDF5 with path='donuts'
    input_path = Path(args.input_file)
    print(f"Loading donuts: {input_path}")
    if input_path.suffix == '.parquet':
        aosTable = QTable.read(str(input_path))
    else:
        aosTable = QTable.read(str(input_path), path='donuts')
    print(f"  {len(aosTable)} donuts")

    print(f"Loading fits: {args.fit_file}")
    fit_table = QTable.read(args.fit_file)
    print(f"  {len(fit_table)} visits")

    # Derive Noll indices
    zk = np.stack(aosTable[f'zk_{coord_sys}'])
    noll_arr = None
    if 'nollIndices' in fit_table.colnames:
        noll_arr = np.array(fit_table['nollIndices'][0])
    iZs, iZidx = derive_noll_indices(zk.shape[1], noll_arr)
    print(f"Noll indices ({len(iZs)} terms): {iZs}")

    iZs_plot_12 = iZs[:12]
    iZs_plot_hi = iZs[12:]

    # Output directory: default to output/{input_stem}/
    if args.output_dir:
        output_dir = args.output_dir
    else:
        input_stem = Path(args.input_file).stem
        output_dir = str(Path(args.input_file).parent / input_stem)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter to matched donuts
    if 'matched_intra_extra' in aosTable.colnames:
        matched_mask = np.array(aosTable['matched_intra_extra'])
        aosTable_matched = aosTable[matched_mask]
        print(f"Matched donuts: {len(aosTable_matched)}")
    else:
        aosTable_matched = aosTable

    # Reconstruct per-donut fit arrays
    print("\nReconstructing per-donut fit arrays...")
    reconstruct_zk_fit(aosTable_matched, fit_table, coord_sys, iZs,
                       prefix='z1toz3', max_focal_noll=3)
    reconstruct_zk_fit(aosTable_matched, fit_table, coord_sys, iZs,
                       prefix='z1toz6', max_focal_noll=6)
    aosTable_matched['zk_fit'] = aosTable_matched['zk_fit_z1toz3']

    # Date range
    all_day_obs = sorted(set(
        np.array(fit_table['day_obs']).tolist()))
    if args.date_range_str:
        date_range_str = args.date_range_str
    else:
        date_range_str = (f'{all_day_obs[0]}-{all_day_obs[-1]}'
                          if len(all_day_obs) > 1
                          else str(all_day_obs[0]))

    # Build visit_info-like table from fit_table
    # (fit_table has visit_info columns merged in from run_dz_fit)
    visit_info = fit_table

    # Bad-fit filtering
    fit_prefix = 'z1toz6'
    bad_col = f'{fit_prefix}_bad_fit'
    if bad_col in fit_table.colnames:
        ft_good = fit_table[~np.array(fit_table[bad_col])]
    else:
        ft_good = fit_table

    # ================================================================
    # Fit parameter plots (z1toz6 only, all Zernikes in one PDF)
    # ================================================================
    if not args.no_fit_params:
        print(f"\n{'=' * 60}")
        print("Fit parameter plots (z1toz6)")
        print(f"{'=' * 60}")

        # Build donut mask excluding bad fits
        bad_visits = set()
        if bad_col in fit_table.colnames:
            for row in fit_table[np.array(fit_table[bad_col])]:
                bad_visits.add((int(row['day_obs']),
                                int(row['seq_num'])))
        if bad_visits:
            dobs_arr = np.array(aosTable_matched['day_obs'])
            snum_arr = np.array(aosTable_matched['seq_num'])
            good_donut_mask = np.array([
                (int(d), int(s)) not in bad_visits
                for d, s in zip(dobs_arr, snum_arr)])
        else:
            good_donut_mask = np.ones(len(aosTable_matched), dtype=bool)

        # FAM block filter
        block_mask = _identify_fam_blocks(ft_good, min_block_size=3)
        ft_blocks = ft_good[block_mask]
        n_excluded = len(ft_good) - len(ft_blocks)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} non-block FAM visits")

        block_visits = set(zip(
            np.array(ft_blocks['day_obs']).tolist(),
            np.array(ft_blocks['seq_num']).tolist()))
        block_donut_mask = good_donut_mask & np.array([
            (int(d), int(s)) in block_visits
            for d, s in zip(np.array(aosTable_matched['day_obs']),
                           np.array(aosTable_matched['seq_num']))])

        # Single merged PDF over all pupil Zernikes (Z4..Z26), one
        # page per pupil Z; 6 focal k coefficients per page.
        plot_fit_params_and_residuals(
            ft_blocks, aosTable_matched, block_donut_mask,
            day_obs_list=all_day_obs, fit_prefix=fit_prefix,
            iZs_fit_plot=iZs, iZs_hist=iZs,
            iZs=iZs, iZidx=iZidx, coord_sys=coord_sys,
            visit_info=visit_info,
            output_dir=output_dir, show=False)

    # ================================================================
    # Single-image residual maps
    # ================================================================
    if not args.no_single_image:
        print(f"\n{'=' * 60}")
        print("Single-image residual maps")
        print(f"{'=' * 60}")

        band_lookup = {}
        pointing_lookup = {}
        if 'band' in fit_table.colnames:
            for row in fit_table:
                key = (int(row['day_obs']), int(row['seq_num']))
                band_lookup[key] = str(row['band'])
                if 'alt' in fit_table.colnames:
                    ptg = {'alt': float(row['alt'])}
                    if 'az' in fit_table.colnames:
                        ptg['az'] = float(row['az'])
                    rot_col = ('rotAngle' if 'rotAngle' in fit_table.colnames
                               else 'rotator_angle' if 'rotator_angle' in fit_table.colnames
                               else None)
                    if rot_col:
                        ptg['rotAngle'] = float(row[rot_col])
                    pointing_lookup[key] = ptg

        all_images = sorted(set(zip(
            np.array(aosTable_matched['day_obs']).tolist(),
            np.array(aosTable_matched['seq_num']).tolist())))
        print(f"Generating residual maps for {len(all_images)} visits...")

        frame_files = []
        for i, (dobs, snum) in enumerate(all_images):
            band = band_lookup.get((dobs, snum), '')
            ptg = pointing_lookup.get((dobs, snum), {})
            outfile = plot_single_image_residual_grid(
                aosTable_matched, dobs, snum,
                iZs=iZs, iZidx=iZidx, coord_sys=coord_sys,
                band=band,
                alt=ptg.get('alt'), az=ptg.get('az'),
                rotAngle=ptg.get('rotAngle'),
                fit_table=fit_table, fit_prefix='z1toz3',
                iZs_plot=iZs_plot_12,
                output_dir=output_dir, show=False)
            if outfile is not None:
                frame_files.append(outfile)
            if (i + 1) % 50 == 0:
                print(f"  ...processed {i + 1}/{len(all_images)} visits")

        print(f"Generated {len(frame_files)} residual map frames")

        # Build movie
        if len(frame_files) > 1:
            list_file = f'{output_dir}/frame_list.txt'
            with open(list_file, 'w') as f:
                for fpath in frame_files:
                    f.write(f"file '{Path(fpath).name}'\n")
                    f.write("duration 0.5\n")

            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', 'frame_list.txt',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-r', '2', 'single_image_residuals.mp4']

            try:
                result = subprocess.run(
                    ffmpeg_cmd, capture_output=True, text=True,
                    cwd=output_dir)
                if result.returncode == 0:
                    movie_file = f'{output_dir}/single_image_residuals.mp4'
                    print(f"Saved movie: {movie_file}")
                    # Clean up JPEGs
                    for fpath in frame_files:
                        Path(fpath).unlink(missing_ok=True)
                    Path(list_file).unlink(missing_ok=True)
                    print(f"Cleaned up {len(frame_files)} JPEGs")
                else:
                    print(f"ffmpeg failed: {result.stderr[-300:]}")
            except FileNotFoundError:
                print("ffmpeg not found — keeping individual JPEGs")
                print(f"  To create movie: cd {output_dir} && "
                      f"{' '.join(ffmpeg_cmd)}")

    # ================================================================
    # Trio comparison plots
    # ================================================================
    if not args.no_trio:
        print(f"\n{'=' * 60}")
        print("Trio comparison plots")
        print(f"{'=' * 60}")

        # Filter bad fits from donuts
        if bad_col in fit_table.colnames:
            bad_visits = set()
            for row in fit_table[np.array(fit_table[bad_col])]:
                bad_visits.add((int(row['day_obs']),
                                int(row['seq_num'])))
            if bad_visits:
                dobs_arr = np.array(aosTable_matched['day_obs'])
                snum_arr = np.array(aosTable_matched['seq_num'])
                good_mask = np.array([
                    (int(d), int(s)) not in bad_visits
                    for d, s in zip(dobs_arr, snum_arr)])
                aos_good = aosTable_matched[good_mask]
            else:
                aos_good = aosTable_matched
        else:
            aos_good = aosTable_matched

        # z1toz3 trio
        trio_pdf = f'{output_dir}/trio_comparison_all.pdf'
        with PdfPages(trio_pdf) as pdf:
            for iZ in iZs:
                plot_zernike_trio(
                    aos_good, iZ, iZs=iZs, iZidx=iZidx,
                    coord_sys=coord_sys,
                    fit_prefix='z1toz3',
                    date_range_str=date_range_str,
                    pdf=pdf, show=False)
        print(f"Saved: {trio_pdf}")

        # z1toz6 trio
        trio_k6_pdf = f'{output_dir}/trio_comparison_k1to6_all.pdf'
        with PdfPages(trio_k6_pdf) as pdf:
            for iZ in iZs:
                plot_zernike_trio(
                    aos_good, iZ, iZs=iZs, iZidx=iZidx,
                    coord_sys=coord_sys,
                    fit_prefix='z1toz6',
                    date_range_str=date_range_str,
                    pdf=pdf, show=False)
        print(f"Saved: {trio_k6_pdf}")

    print("\nDone.")


if __name__ == '__main__':
    main()
