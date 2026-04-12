#!/usr/bin/env python3
"""Run Double Zernike focal-plane fits on a donut wavefront table.

Usage:
    python run_dz_fit.py input.hdf5
    python run_dz_fit.py input.hdf5 --coord-sys CCS
    python run_dz_fit.py input.hdf5 --output output_fits.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dz_fitting import run_double_zernike_fits


def main():
    parser = argparse.ArgumentParser(
        description='Run Double Zernike focal-plane fits on donut wavefront data.')
    parser.add_argument('input_file',
                        help='Input HDF5 file with donuts+visits tables '
                             '(from intrinsics_mktable)')
    parser.add_argument('--output', default=None,
                        help='Output fit parquet file '
                             '(default: {input_stem}_fits.parquet)')
    parser.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'],
                        help='Coordinate system (default: OCS)')
    parser.add_argument('--bad-fit-threshold', type=float, default=2.0,
                        help='Flag fits with |coeff| > threshold μm (default: 2.0)')
    parser.add_argument('--min-donuts', type=int, default=200,
                        help='Flag fits with fewer donuts (default: 200)')

    args = parser.parse_args()

    run_double_zernike_fits(
        input_file=args.input_file,
        coord_sys=args.coord_sys,
        output_file=args.output,
        bad_fit_threshold=args.bad_fit_threshold,
        min_donuts=args.min_donuts,
    )


if __name__ == '__main__':
    main()
