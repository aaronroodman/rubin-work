#!/usr/bin/env python3
"""Ingest the in-focus corner-WFS (cwfs) aggregate tables, paired to FAM triplets.

The corner wavefront sensors (SW0 — the extra-focal inner halves of R00/R04/R40/
R44) see defocused donuts during the *in-focus* exposure of each FAM triplet.
The triplet order is intra, extra, in-focus, and the FAM ``visits.parquet``
seq_num is the EXTRA-focal exposure, so the in-focus exposure is **FAM_seq + 1**
(verified against the Butler: FAM aggregate seqs …122,125,128; cwfs in-focus
seqs …123,126,129).

This walks the already-built FAM ``output/<ps>/visits.parquet``, reads each
in-focus exposure's ``aggregateAOSVisitTableRaw`` from the param_set's
``wfs_collection`` (reusing intrinsics_lib.get_aggregate_zernikes — the cwfs
table has the same schema as FAM, plus per-corner intra/extra positions), tags
each donut with the paired FAM seq_num and the FAM visit's rotator/elevation (so
WFS and FAM share the same rotator binning), and writes:

    output/<param_set>/wfs/donuts.parquet   per-corner WFS donuts: zk_<coord>,
        zk_intrinsic_<coord>, thx/thy_<coord>[_intra/_extra], detector,
        day_obs, seq_num (in-focus), fam_seq_num, rotator_angle, alt
    output/<param_set>/wfs/visits.parquet   one row per in-focus exposure

Validation (default on): mean WFS Zernike_j vs ordinal in-focus image, with the
FAM k=1 (field-mean) DZ coefficient z{prefix}_z{j}_c1 overlaid per image →
output/<param_set>/wfs/wfs_mktable_validation.pdf .  RSP-only (Butler).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import QTable, vstack

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from intrinsics_lib import get_aggregate_zernikes, load_param_sets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    coord = args.coord_sys

    pset = load_param_sets()[args.param_set]
    repo = pset['butler_repo']
    wfs_coll = pset.get('wfs_collection')
    if not wfs_coll:
        raise SystemExit(f'param_set {args.param_set} has no wfs_collection')
    base = Path(args.output_root) / args.param_set
    fam_visits = QTable.read(str(base / 'visits.parquet'))
    print(f'[wfs_mktable] {args.param_set}: {len(fam_visits)} FAM visits; '
          f'cwfs collection {wfs_coll}')

    from lsst.daf.butler import Butler
    from lsst.obs.lsst import LsstCam
    butler = Butler(repo, collections=wfs_coll)
    camera = LsstCam.getCamera()
    has_rot = 'rotator_angle' in fam_visits.colnames
    has_alt = 'alt' in fam_visits.colnames

    # keep a lean, fixed column set — the raw aggregate has many extra scalar
    # columns (e.g. lstsq_cost) whose dtype varies across exposures (float vs
    # all-NaN object), which breaks vstack.  We only need the Zernikes +
    # per-corner positions + detector.
    keep = ['detector', f'zk_{coord}', f'zk_intrinsic_{coord}',
            f'thx_{coord}', f'thy_{coord}',
            f'thx_{coord}_intra', f'thx_{coord}_extra',
            f'thy_{coord}_intra', f'thy_{coord}_extra']
    donut_tabs, vrows, n_miss, noll = [], [], 0, None
    for v in fam_visits:
        d = int(v['day_obs']); fam_s = int(v['seq_num']); infocus = fam_s + 1
        tbl, meta = get_aggregate_zernikes(butler, d, infocus, coord, camera)
        if tbl is None:
            n_miss += 1; continue
        if noll is None and meta.get('nollIndices') is not None:
            noll = [int(x) for x in meta['nollIndices']]
        tbl = tbl[[c for c in keep if c in tbl.colnames]]
        rot = float(v['rotator_angle']) if has_rot else np.nan
        alt = float(v['alt']) if has_alt else np.nan
        tbl['day_obs'] = d; tbl['seq_num'] = infocus; tbl['fam_seq_num'] = fam_s
        tbl['rotator_angle'] = rot; tbl['alt'] = alt
        donut_tabs.append(tbl)
        vrows.append(dict(day_obs=d, seq_num=infocus, fam_seq_num=fam_s,
                          rotator_angle=rot, alt=alt, band=meta.get('band', ''),
                          n_donuts=len(tbl)))
    if not donut_tabs:
        raise RuntimeError('No in-focus cwfs aggregate tables found for any '
                           'FAM visit (FAM_seq+1).')
    donuts = vstack(donut_tabs, metadata_conflicts='silent')
    donuts.meta['nollIndices'] = noll          # Zernike order of zk_<coord>
    out = base / 'wfs'; out.mkdir(parents=True, exist_ok=True)
    donuts.write(str(out / 'donuts.parquet'), format='parquet', overwrite=True)
    vdf = pd.DataFrame(vrows)
    vdf['nollIndices'] = [list(noll) if noll else None] * len(vdf)
    vdf.to_parquet(out / 'visits.parquet')
    print(f'  {len(vrows)} in-focus exposures, {len(donuts)} WFS donuts '
          f'({n_miss} FAM visits had no in-focus cwfs table); wrote wfs/donuts'
          f'.parquet + wfs/visits.parquet')

    if not args.no_plot:
        _validation_plot(donuts, base, out, coord, args.dz_prefix, noll)


def _validation_plot(donuts, base, out, coord, prefix, noll):
    """Mean WFS Zernike_j vs ordinal in-focus image, FAM k=1 (z_j_c1) overlaid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    # read straight from the astropy table (zk_<coord> is a multidim column,
    # so to_pandas() would fail)
    zk = np.array(donuts[f'zk_{coord}'], dtype=float)      # (n_donuts, n_zern)
    if not noll:
        noll = list(range(4, 4 + zk.shape[1]))
    fam_s = np.asarray(donuts['fam_seq_num']).astype(int)
    dobs = np.asarray(donuts['day_obs']).astype(int)
    # per in-focus image (fam_seq) WFS mean
    keys = sorted(set(zip(dobs.tolist(), fam_s.tolist())))
    ordinal = {k: i for i, k in enumerate(keys)}
    wfs_mean = np.full((len(keys), len(noll)), np.nan)
    for i, (d, s) in enumerate(keys):
        m = (dobs == d) & (fam_s == s)
        wfs_mean[i] = np.nanmean(zk[m], axis=0)

    # FAM k=1 (field-mean) per image from fits.parquet, joined by (day_obs, seq)
    fam = None
    fp = base / 'fits.parquet'
    if fp.exists():
        ft = pd.read_parquet(fp)
        fam = {(int(r.day_obs), int(r.seq_num)): r for r in ft.itertuples()}

    ncols = 3
    nrows = (len(noll) + ncols - 1) // ncols
    x = np.arange(len(keys))
    with PdfPages(str(out / 'wfs_mktable_validation.pdf')) as pdf:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.6 * nrows),
                                 layout='constrained', squeeze=False)
        axes = axes.ravel()
        for p, j in enumerate(noll):
            ax = axes[p]
            ax.plot(x, wfs_mean[:, p], 'o', ms=3, color='steelblue',
                    label='WFS mean')
            if fam is not None:
                col = f'{prefix}_z{j}_c1'
                fk = np.array([getattr(fam.get((d, s)), col, np.nan)
                               if fam.get((d, s)) is not None else np.nan
                               for (d, s) in keys])
                ax.plot(x, fk, '-', lw=1.0, color='crimson', alpha=0.8,
                        label='FAM k=1')
            ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)
            ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}', fontsize=8)
            ax.tick_params(labelsize=7)
        for p in range(len(noll), len(axes)):
            axes[p].set_visible(False)
        axes[0].legend(fontsize=7, loc='best')
        for ax in axes[max(0, len(noll) - ncols):len(noll)]:
            ax.set_xlabel('ordinal in-focus image', fontsize=8)
        fig.suptitle('WFS mean Zernike vs ordinal image  (FAM k=1 overlaid)',
                     fontsize=13)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print('  wrote wfs/wfs_mktable_validation.pdf')


if __name__ == '__main__':
    main()
