"""Reduce the unpaired FAM donut table to one row per visit per side of focus.

Writes a small parquet of per-visit medians so the study notebook never reads the 13 GB
donut table. One pass, column projection, one row group (one visit) at a time.

Run from the repo root::

    python aos/unpaired_intra_vs_extra/extract_visit_medians.py \
        --variant danish_1_3_v1000 --build 20261013
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import median_abs_deviation

from rubinwork.products import catalog

# Columns the reduction needs, of the 66 in the donut table.
COLUMNS = (
    'day_obs', 'seq_num', 'detector',
    'thx_OCS', 'thy_OCS',
    'blur_intra', 'blur_extra',
    'zk_OCS_intra', 'zk_OCS_extra',
)

FIELD_RADIUS_LIMIT_DEG = 1.725


def visit_medians(variant, build, field_radius_limit_deg=FIELD_RADIUS_LIMIT_DEG):
    """Median per visit per side of focus, for the blur and the 21 Noll Zernikes.

    Parameters
    ----------
    variant : `str`
        `fam_tables` variant name, e.g. ``'danish_1_3_v1000'``.
    build : `str`
        Build identifier, e.g. ``'20261013'``.
    field_radius_limit_deg : `float`, optional
        Donuts outside this OCS field radius [deg] are dropped.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per visit. ``blur_{intra,extra}`` [arcsec] and
        ``z{j}_{intra,extra}`` [micrometres of wavefront] are the per-side medians over
        the visit's donuts. ``dz{j}_donut_median`` and ``dz{j}_donut_nmad`` [micrometres
        of wavefront], with ``dblur_donut_*`` [arcsec], are the median and robust scatter
        of the **per-donut** intra minus extra difference, each donut being an
        already-matched pair. ``n_donuts`` (dimensionless) counts the pairs used.
    noll : `list` [`int`]
        The Noll indices, in the column order of the stored Zernike vectors.
    """
    data_dir = catalog.path('fam_tables', variant, build)
    donuts = pq.ParquetFile(data_dir / 'donuts.parquet')

    visits = pd.read_parquet(data_dir / 'visits.parquet')
    noll = [int(j) for j in visits['nollIndices'].iloc[0]]

    rows = []
    for group in range(donuts.metadata.num_row_groups):
        chunk = donuts.read_row_group(group, columns=list(COLUMNS)).to_pandas()

        intra = np.stack(chunk['zk_OCS_intra'].to_numpy())
        extra = np.stack(chunk['zk_OCS_extra'].to_numpy())

        radius_deg = np.degrees(np.hypot(chunk['thx_OCS'], chunk['thy_OCS']))
        keep = (
            np.isfinite(intra).all(axis=1)
            & np.isfinite(extra).all(axis=1)
            & np.isfinite(chunk['blur_intra'])
            & np.isfinite(chunk['blur_extra'])
            & (radius_deg <= field_radius_limit_deg)
        )
        if not keep.any():
            continue

        kept = keep.to_numpy()
        record = {
            'day_obs': int(chunk['day_obs'].iloc[0]),
            'seq_num': int(chunk['seq_num'].iloc[0]),
            'n_donuts': int(kept.sum()),
            'blur_intra': float(np.median(chunk.loc[keep, 'blur_intra'])),
            'blur_extra': float(np.median(chunk.loc[keep, 'blur_extra'])),
        }

        # Per-donut difference statistics. Each row of the donut table is already a
        # matched pair -- `pair_sides` joins the two sides on (det_name, donut_id), the
        # Gaia source identifier -- so no positional matching is involved.
        blur_difference = (chunk.loc[keep, 'blur_intra']
                           - chunk.loc[keep, 'blur_extra']).to_numpy()
        record['dblur_donut_median'] = float(np.median(blur_difference))
        record['dblur_donut_nmad'] = float(
            median_abs_deviation(blur_difference, scale='normal'))

        for column, noll_index in enumerate(noll):
            side_intra = intra[kept, column]
            side_extra = extra[kept, column]
            record[f'z{noll_index}_intra'] = float(np.median(side_intra))
            record[f'z{noll_index}_extra'] = float(np.median(side_extra))

            difference = side_intra - side_extra
            record[f'dz{noll_index}_donut_median'] = float(np.median(difference))
            record[f'dz{noll_index}_donut_nmad'] = float(
                median_abs_deviation(difference, scale='normal'))
        rows.append(record)

    return pd.DataFrame(rows), noll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--build', default='20261013')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    table, noll = visit_medians(args.variant, args.build)

    # The quality flag and the night key live in visits.parquet, not the donut table.
    data_dir = catalog.path('fam_tables', args.variant, args.build)
    visits = pd.read_parquet(
        data_dir / 'visits.parquet',
        columns=['day_obs', 'seq_num', 'visit', 'intra_visit', 'extra_visit',
                 'rotator_angle', 'visit_quality_pass', 'median_blur_arcsec',
                 'median_blur_intra_arcsec', 'median_blur_extra_arcsec'],
    )
    table = table.merge(visits, on=['day_obs', 'seq_num'], how='left')

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'visit_medians_{args.variant}.parquet'
    table.to_parquet(out_path, index=False)

    passed = table['visit_quality_pass'].fillna(False)
    print(f'{args.variant}@{args.build}: {len(table)} visits, '
          f'{int(passed.sum())} passing visit_quality_pass, '
          f'{table["day_obs"].nunique()} nights, Noll {noll[0]}..{noll[-1]} '
          f'({len(noll)} terms)')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
