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

from lsst.ts.intrinsic.wavefront.intrinsics_lib import get_aggregate_zernikes, load_param_sets

# Extra-focal SW0 corner detectors (the aggregate keys each pair under this SW0
# detector name; the per-detector `zernikes` tables live on these detector ids).
AIDONUT_SW0 = {191: 'R00_SW0', 195: 'R04_SW0', 199: 'R40_SW0', 203: 'R44_SW0'}


def _sid(x):
    """donut id as str, or None if masked (`zernikes` ids are <U21 strings)."""
    if x is np.ma.masked or np.ma.is_masked(x):
        return None
    s = str(x)
    return None if s in ('', 'masked', '--') else s


def _xy(q):
    """(x, y) floats from a structured ('x','y') Quantity/void element, unit stripped."""
    return np.array([float(getattr(q['x'], 'value', q['x'])),
                     float(getattr(q['y'], 'value', q['y']))])


def _zk_um(q):
    """Zernike vector -> float array in microns (aggregate zk may be a Quantity)."""
    if hasattr(q, 'unit'):
        import astropy.units as u
        try:
            return np.asarray(q.to_value(u.um), float).ravel()
        except Exception:
            pass
    return np.asarray(getattr(q, 'value', q), float).ravel()


def get_aidonut_zernikes(butler, day_obs, seq_num, coord):
    """ai_donut reader: the collection has no joined aggregateAOSVisitTableRaw, only
    `aggregateZernikesRaw` (zk_{OCS,CCS,NW} + intrinsic, no positions) and per-detector
    `zernikes` (positions: intra/extra_field in deg, centroids in pix).  Join the two on
    (detector, intra/extra donut id) and build the standard donuts schema.

    Position transform (determined empirically, determine_wfs_field_frame.py):
      thx_CCS = deg2rad(field_y), thy_CCS = deg2rad(field_x)   [x<->y swap, deg->rad]
      [thx_OCS, thy_OCS] = R(rotTelPos) . [thx_CCS, thy_CCS]
      paired thx/thy = mean(intra, extra).
    Returns (astropy Table with the keep columns, visit_meta) or (None, None).
    """
    from astropy.table import Table
    try:
        agg = butler.get('aggregateZernikesRaw', day_obs=day_obs, seq_num=seq_num)
    except Exception:
        print(f'DatasetNotFoundError: No aggregateZernikesRaw for '
              f'day_obs={day_obs}, seq_num={seq_num}')
        return None, None
    meta = agg.meta
    theta = float(getattr(meta.get('rotTelPos', 0.0), 'value', meta.get('rotTelPos', 0.0)))
    ct, st = np.cos(theta), np.sin(theta)

    def to_ocs(v):
        return np.array([ct * v[0] - st * v[1], st * v[0] + ct * v[1]])

    # positions per (detector-name, intra_id, extra_id) from the 4 SW0 `zernikes` tables
    pos = {}
    for det_id, det_name in AIDONUT_SW0.items():
        try:
            zt = butler.get('zernikes', day_obs=day_obs, seq_num=seq_num, detector=det_id)
        except Exception:
            continue
        for zr in zt:
            ii, ee = _sid(zr['intra_donut_id']), _sid(zr['extra_donut_id'])
            if ii is None or ee is None:
                continue
            fi, fe = _xy(zr['intra_field']), _xy(zr['extra_field'])           # deg
            if not (np.all(np.isfinite(fi)) and np.all(np.isfinite(fe))):
                continue
            ci, ce = _xy(zr['intra_centroid']), _xy(zr['extra_centroid'])     # pix
            ccs_i = np.deg2rad([fi[1], fi[0]]); ccs_e = np.deg2rad([fe[1], fe[0]])  # swap x,y
            pi, pe = (to_ocs(ccs_i), to_ocs(ccs_e)) if coord == 'OCS' else (ccs_i, ccs_e)
            pos[(det_name, ii, ee)] = (pi, pe, 0.5 * (ccs_i + ccs_e), ci, ce)

    zkc, zic = f'zk_{coord}', f'zk_intrinsic_{coord}'
    rows = []
    for r in agg:
        if not bool(r['used']):
            continue
        p = pos.get((str(r['detector']), _sid(r['intra_donut_id']), _sid(r['extra_donut_id'])))
        if p is None:
            continue
        pi, pe, ccs, ci, ce = p
        pair = 0.5 * (pi + pe)
        rows.append((str(r['detector']), _zk_um(r[zkc]), _zk_um(r[zic]),
                     pair[0], pair[1], pi[0], pe[0], pi[1], pe[1], ccs[0], ccs[1],
                     ci[0], ci[1], ce[0], ce[1]))
    if not rows:
        return None, None

    cols = list(zip(*rows))
    tbl = Table()
    tbl['detector'] = list(cols[0])
    tbl[zkc] = np.array(cols[1]); tbl[zic] = np.array(cols[2])
    tbl[f'thx_{coord}'] = cols[3]; tbl[f'thy_{coord}'] = cols[4]
    tbl[f'thx_{coord}_intra'] = cols[5]; tbl[f'thx_{coord}_extra'] = cols[6]
    tbl[f'thy_{coord}_intra'] = cols[7]; tbl[f'thy_{coord}_extra'] = cols[8]
    tbl['thx_CCS'] = cols[9]; tbl['thy_CCS'] = cols[10]
    tbl['centroid_x_intra'] = cols[11]; tbl['centroid_y_intra'] = cols[12]
    tbl['centroid_x_extra'] = cols[13]; tbl['centroid_y_extra'] = cols[14]
    noll = ([int(x) for x in meta['nollIndices']]
            if meta.get('nollIndices') is not None else None)
    return tbl, dict(band=meta.get('band', ''), nollIndices=noll)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--wfs-name', required=True,
                    help='CWFS variant key in the param_set wfs_collections map; '
                         'also the output subdir output/<ps>/wfs/<wfs-name>/')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    coord = args.coord_sys

    pset = load_param_sets()[args.param_set]
    repo = pset['butler_repo']
    wfs_map = pset.get('wfs_collections') or {}
    if args.wfs_name not in wfs_map:
        raise SystemExit(f'param_set {args.param_set} has no wfs_collections entry '
                         f'{args.wfs_name!r} (have: {list(wfs_map)})')
    entry = wfs_map[args.wfs_name]                  # str, or {collection, seq_offset, dataset_type}
    if isinstance(entry, dict):
        wfs_coll = entry['collection']
        seq_offset = int(entry.get('seq_offset', 1))
        dataset_type = entry.get('dataset_type', 'aggregateAOSVisitTableRaw')
    else:
        wfs_coll, seq_offset = entry, 1             # bare string -> in-focus (fam+1)
        dataset_type = 'aggregateAOSVisitTableRaw'
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
    # also carry CCS positions (for the Z4 CCS MIW component) and the intra/extra
    # centroids (for the per-corner CCD-height Z4) — used by run_wfs_dof_compare
    for extra in ['thx_CCS', 'thy_CCS', 'centroid_x_intra', 'centroid_y_intra',
                  'centroid_x_extra', 'centroid_y_extra']:
        if extra not in keep:
            keep.append(extra)
    donut_tabs, vrows, n_miss, noll = [], [], 0, None
    for v in fam_visits:
        d = int(v['day_obs']); fam_s = int(v['seq_num']); infocus = fam_s + seq_offset
        if dataset_type == 'aggregateZernikesRaw':          # ai_donut: no joined AOS table
            tbl, meta = get_aidonut_zernikes(butler, d, infocus, coord)
        else:
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
    out = base / 'wfs' / args.wfs_name; out.mkdir(parents=True, exist_ok=True)
    donuts.write(str(out / 'donuts.parquet'), format='parquet', overwrite=True)
    vdf = pd.DataFrame(vrows)
    vdf['nollIndices'] = [list(noll) if noll else None] * len(vdf)
    vdf.to_parquet(out / 'visits.parquet')
    print(f'  {len(vrows)} in-focus exposures, {len(donuts)} WFS donuts '
          f'({n_miss} FAM visits had no in-focus cwfs table); wrote wfs/donuts'
          f'.parquet + wfs/visits.parquet')

    if not args.no_plot:
        _validation_plot(donuts, base, out, coord, args.dz_prefix, noll)


def _to_deg(a):
    a = np.asarray(a, float)
    return np.rad2deg(a) if np.nanmax(np.abs(a)) < 0.1 else a


def _fam_radius_median(fam_donuts_path, coord, inner, outer):
    """Median FAM donut zk per (day_obs, seq_num) within the WFS radial shell
    [inner, outer]°.  Streams the (large) FAM donuts.parquet by row batch."""
    import pyarrow.parquet as pq
    from collections import defaultdict
    zkc, txc, tyc = f'zk_{coord}', f'thx_{coord}', f'thy_{coord}'
    pf = pq.ParquetFile(str(fam_donuts_path))
    acc = defaultdict(list)
    for batch in pf.iter_batches(columns=[zkc, txc, tyc, 'day_obs', 'seq_num'],
                                 batch_size=300000):
        bd = batch.to_pandas()
        r = np.hypot(_to_deg(bd[txc]), _to_deg(bd[tyc]))
        shell = (r >= inner) & (r <= outer)
        if not shell.any():
            continue
        sub = bd.loc[shell]
        zk = np.vstack(sub[zkc].values)
        dd = np.asarray(sub['day_obs']).astype(int)
        ss = np.asarray(sub['seq_num']).astype(int)
        for d, s in set(zip(dd.tolist(), ss.tolist())):
            acc[(d, s)].append(zk[(dd == d) & (ss == s)])
    return {k: np.nanmedian(np.vstack(v), axis=0) for k, v in acc.items()}


def _ordinal_page(pdf, x, pts, line, noll, names, suptitle, pts_lab, line_lab):
    import matplotlib.pyplot as plt
    ncols = 3; nrows = (len(noll) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.6 * nrows),
                             layout='constrained', squeeze=False)
    axes = axes.ravel()
    for p, j in enumerate(noll):
        ax = axes[p]
        ax.plot(x, pts[:, p], 'o', ms=3, color='steelblue', label=pts_lab)
        if line is not None:
            ax.plot(x, line[:, p], '-', lw=1.0, color='crimson', alpha=0.8,
                    label=line_lab)
        ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)
        ax.set_title(f'Z{j} {names.get(j, "")}', fontsize=8); ax.tick_params(labelsize=7)
    for p in range(len(noll), len(axes)):
        axes[p].set_visible(False)
    axes[0].legend(fontsize=7, loc='best')
    fig.suptitle(suptitle, fontsize=13)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def _scatter_fit_page(pdf, xby, yby, noll, names, suptitle, xlab, ylab):
    """Per-j scatter of y vs x + OLS fit; returns [(j, slope, offset, r, n)]."""
    import matplotlib.pyplot as plt
    ncols = 3; nrows = (len(noll) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.0 * nrows),
                             layout='constrained', squeeze=False)
    axes = axes.ravel(); rows = []
    for p, j in enumerate(noll):
        ax = axes[p]
        x = xby[:, p]; y = yby[:, p]
        m = np.isfinite(x) & np.isfinite(y)
        ttl = f'Z{j} {names.get(j, "")}'
        if int(m.sum()) > 2:
            ax.scatter(x[m], y[m], s=8, alpha=0.5, edgecolors='none')
            lo = float(np.nanmin([x[m].min(), y[m].min()]))
            hi = float(np.nanmax([x[m].max(), y[m].max()]))
            ax.plot([lo, hi], [lo, hi], 'k:', lw=0.8, alpha=0.6)      # unity
            sl, off = np.polyfit(x[m], y[m], 1)
            xf = np.array([x[m].min(), x[m].max()])
            ax.plot(xf, sl * xf + off, 'r-', lw=1.3, alpha=0.9)
            r = float(np.corrcoef(x[m], y[m])[0, 1])
            rows.append(dict(j=j, slope=sl, offset=off, r=r, n=int(m.sum())))
            ttl += f'\nslope={sl:+.2f} off={off:+.3f}μm r={r:+.2f}'
        ax.set_xlabel(xlab, fontsize=7); ax.set_ylabel(ylab, fontsize=7)
        ax.set_title(ttl, fontsize=7.5); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
    for p in range(len(noll), len(axes)):
        axes[p].set_visible(False)
    fig.suptitle(suptitle, fontsize=13)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    return rows


def _validation_plot(donuts, base, out, coord, prefix, noll):
    """WFS↔FAM validation: (0) WFS mean vs FAM k=1 vs ordinal; (A) scatter +
    linear fit of the same; (B) WFS median vs FAM-donut median *at the WFS
    radius* (the apples-to-apples comparison incl. the measured intrinsic +
    k=1..6 field structure) — vs ordinal and as a scatter + fit."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    zk = np.array(donuts[f'zk_{coord}'], dtype=float)      # (n_donuts, n_zern)
    if not noll:
        noll = list(range(4, 4 + zk.shape[1]))
    fam_s = np.asarray(donuts['fam_seq_num']).astype(int)
    dobs = np.asarray(donuts['day_obs']).astype(int)
    keys = sorted(set(zip(dobs.tolist(), fam_s.tolist())))
    x = np.arange(len(keys))
    wfs_mean = np.full((len(keys), len(noll)), np.nan)
    wfs_med = np.full((len(keys), len(noll)), np.nan)
    for i, (d, s) in enumerate(keys):
        m = (dobs == d) & (fam_s == s)
        wfs_mean[i] = np.nanmean(zk[m], axis=0)
        wfs_med[i] = np.nanmedian(zk[m], axis=0)

    # FAM k=1 (field-mean) per image from fits.parquet
    fam_k1 = np.full((len(keys), len(noll)), np.nan)
    fp = base / 'fits.parquet'
    if fp.exists():
        ft = pd.read_parquet(fp)
        d = {(int(r.day_obs), int(r.seq_num)): r for r in ft.itertuples()}
        for i, (dd, ss) in enumerate(keys):
            r = d.get((dd, ss))
            if r is not None:
                fam_k1[i] = [getattr(r, f'{prefix}_z{j}_c1', np.nan) for j in noll]

    # FAM-donut median at the WFS radius (the careful comparison)
    inner, outer = _wfs_shell()
    fam_rad = np.full((len(keys), len(noll)), np.nan)
    dp = base / 'donuts.parquet'
    if dp.exists():
        print(f'  computing FAM median in WFS shell [{inner:.4f}, {outer}]° ...')
        med = _fam_radius_median(dp, coord, inner, outer)
        for i, k in enumerate(keys):
            if k in med:
                fam_rad[i] = med[k][:len(noll)]

    with PdfPages(str(out / 'wfs_mktable_validation.pdf')) as pdf:
        _ordinal_page(pdf, x, wfs_mean, fam_k1, noll, NOLL_NAMES,
                      'WFS mean Zernike vs ordinal image  (FAM k=1 overlaid)',
                      'WFS mean', 'FAM k=1')
        rA = _scatter_fit_page(pdf, fam_k1, wfs_mean, noll, NOLL_NAMES,
                               'WFS mean vs FAM k=1  (unity dotted, OLS red)',
                               'FAM k=1 [μm]', 'WFS mean [μm]')
        _ordinal_page(pdf, x, wfs_med, fam_rad, noll, NOLL_NAMES,
                      f'WFS median vs FAM-donut median at WFS radius '
                      f'[{inner:.3f}-{outer}°] vs ordinal',
                      'WFS median', 'FAM @WFS-r')
        rB = _scatter_fit_page(pdf, fam_rad, wfs_med, noll, NOLL_NAMES,
                               f'WFS median vs FAM median @WFS radius '
                               f'[{inner:.3f}-{outer}°]  (unity dotted, OLS red)',
                               'FAM @WFS-r median [μm]', 'WFS median [μm]')
    print('  wrote wfs/wfs_mktable_validation.pdf')
    print('  offsets (WFS = slope·FAM + offset):')
    for tag, rr in (('vs k=1', rA), ('vs FAM@WFS-r', rB)):
        s = '  '.join(f'Z{d["j"]}:{d["offset"]:+.3f}' for d in rr if abs(d['offset']) > 0.02)
        print(f'    {tag}: {s or "(all |offset|<0.02)"}')


def _wfs_shell():
    """(inner, outer)° radial shell of the corner WFS — extra-focal SW0 inner
    corner from cameraGeom (RSP), out to the AOS-online 1.725° limit."""
    try:
        from lsst.obs.lsst import LsstCam
        from lsst.ts.intrinsic.wavefront.ccd_height import wfs_field_radius_range
        inner, _ = wfs_field_radius_range(LsstCam.getCamera())
    except Exception:
        inner = 1.5178
    return float(inner), 1.725


if __name__ == '__main__':
    main()
