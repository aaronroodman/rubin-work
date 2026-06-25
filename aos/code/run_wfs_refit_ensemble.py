#!/usr/bin/env python3
"""wfs_refit_ensemble — re-fit corner-WFS donut pairs across a clean ensemble with the
Danish-1.2 triangle model and save the per-pair Zernikes for settings tuning.

Single donuts can't determine the optimal danish settings (alpha, tolerances, pupil) —
changing the per-pixel error model shifts the wavefront with no single-donut truth.  So
this refits an ENSEMBLE with the validated recipe, swept over systematicLossAlpha, and
writes a tidy per-pair Zernike table.  A separate scoring step then tunes alpha against:
  (b) the 50-DOF/34-vmode post-correction v-mode scatter  [primary]
  (a) agreement with the FAM measured wavefront            [cross-check]
Both read the saved parquet, so refits (expensive) and scoring (cheap) are decoupled.

Selects CLEAN, out-of-plane visits (|galactic b| >= --b-min, i-band, near a chosen
elevation) so donut blends in the dense low-b fields don't confound the noise metric.

Validated recipe (see project_wfs_donut_quality_danish12 / the single-donut snippets):
  DonutTriangleFactory with the NEW M1M3 pupil AND matching M1 mask edges
  (R_outer=4.165, R_inner=2.5833); per-donut zkStart += CCD-height Z4 (batoid_rubin);
  DZMultiDonutModel + least_squares(chi, jac, x_scale='jac', tol 1e-3, nfev 100);
  args 2nd = backgroundStd**2; binning=1 (native wep_im); optional systematicLossAlpha.

RSP only: danish>=1.2, lsst.ts.wep, the butler corner-WFS collection, batoid_rubin CCD
height maps, and aos/code/ccd_height.py.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_OUTER, R_INNER = 4.165, 2.5833        # new M1M3 pupil radii (m); also set the M1 mask edges
BINNING, BKG_ORDER = 1, 2
LSTSQ = dict(x_scale='jac', ftol=1e-3, xtol=1e-3, gtol=1e-3, max_nfev=100)
CORNERS = ['R00', 'R04', 'R40', 'R44']
HEIGHT_KEYS = ['fit_success', 'fwhm', 'model_bkg', 'model_dx', 'model_dy', 'model_flux']


def galactic_b(ra_rad, dec_rad):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    return SkyCoord(ra=np.asarray(ra_rad, float) * u.rad,
                    dec=np.asarray(dec_rad, float) * u.rad, frame='icrs').galactic.b.deg


def _alt_deg(a):
    a = np.asarray(a, float)
    return np.rad2deg(a) if np.nanmax(np.abs(a)) < 2 * np.pi + 1e-3 else a


def build_factory_kwargs(inst, rtp_deg):
    """DonutTriangleFactory kwargs with the new M1M3 pupil AND matching M1 mask edges."""
    import copy
    mp = copy.deepcopy(inst.maskParams)
    mp['M1']['outer']['radius'][-1] = R_OUTER
    mp['M1']['inner']['radius'][-1] = R_INNER
    return dict(R_outer=R_OUTER, R_inner=R_INNER, mask_params=mp,
                focal_length=inst.focalLength, pixel_scale=inst.pixelSize * BINNING,
                spider_angle=rtp_deg)


def refit_pair(danish, algo, inst, fkw, noll, row, meta, z4_extra, z4_intra,
               extra_stamp, intra_stamp, alpha):
    """Height-aware Danish-1.2 triangle refit of one intra/extra pair.
    Returns (zkSum_um[n], fwhm, cost, success) or (None, ...) on failure."""
    from scipy.optimize import least_squares
    i4 = noll.index(4)
    zk_int = np.asarray(row['zk_intrinsic_CCS']) * 1e-6
    zkS_e = zk_int.copy(); zkS_e[i4] += z4_extra * 1e-6
    zkS_i = zk_int.copy(); zkS_i[i4] += z4_intra * 1e-6
    ie, ae, ze, bge = algo._prepDanish(image=extra_stamp.wep_im, zkStart=zkS_e,
                                       nollIndices=noll, instrument=inst)
    ii, ai, zi, bgi = algo._prepDanish(image=intra_stamp.wep_im, zkStart=zkS_i,
                                       nollIndices=noll, instrument=inst)
    imgs, sky, zkRefs = [ie, ii], [bge ** 2, bgi ** 2], [ze, zi]
    thxs, thys = [ae[0], ai[0]], [ae[1], ai[1]]
    dz_terms = [(1, j) for j in noll]
    loss_fn = danish.systematic_loss(alpha) if alpha else None
    mkw = {'loss_fn': loss_fn} if loss_fn is not None else {}
    m = danish.DZMultiDonutModel(danish.DonutTriangleFactory(**fkw), z_refs=zkRefs,
                                 dz_terms=dz_terms, field_radius=np.deg2rad(1.81),
                                 thxs=thxs, thys=thys, npix=ie.shape[0],
                                 bkg_order=BKG_ORDER, **mkw)
    x0 = m.pack_params(fluxes=[float(np.clip(np.sum(im), 1e3, 1e8)) for im in imgs],
                       dxs=[0., 0.], dys=[0., 0.], fwhm=1.0,
                       bkgs=[[0.] * m.nbkg] * 2, wavefront_params=[0.] * len(dz_terms))
    bn = m.pack_params(fluxes=[[0., np.inf]] * 2, dxs=[[-np.inf, np.inf]] * 2,
                       dys=[[-np.inf, np.inf]] * 2, fwhm=[0.1, 5.0],
                       bkgs=[[[-np.inf, np.inf]] * m.nbkg] * 2,
                       wavefront_params=[[-np.inf, np.inf]] * len(dz_terms))
    bn = [list(b) for b in zip(*bn)]
    x0 = np.clip(x0, bn[0], bn[1])
    try:
        r = least_squares(m.chi, jac=m.jac, x0=x0, args=(imgs, sky), bounds=bn, **LSTSQ)
    except Exception as e:
        print(f'      fit failed: {type(e).__name__}: {e}')
        return None, np.nan, np.nan, False
    p = m.unpack_params(r.x)
    zk = (np.asarray(p['wavefront_params']) + np.mean([zkS_e, zkS_i], axis=0)) * 1e6
    return zk, float(p['fwhm']), float(r.cost), bool(r.success)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--b-min', type=float, default=20.0, help='keep visits with |gal b| >= this')
    ap.add_argument('--alt-center', type=float, default=70.0)
    ap.add_argument('--alt-tol', type=float, default=2.5)
    ap.add_argument('--alphas', default='0.0,0.01,0.02,0.03,0.05',
                    help='comma list of systematicLossAlpha to sweep')
    ap.add_argument('--max-visits', type=int, default=None)
    ap.add_argument('--collection', default=None)
    ap.add_argument('--butler-repo', default=None)
    ap.add_argument('--height-map-dir', default='~/u/LSST/packages/batoid_rubin_data',
                    help='batoid_rubin ccd_height_map dir (avoids the read-only ensure_data_dir download)')
    args = ap.parse_args()
    coord = args.coord_sys
    alphas = [float(a) for a in args.alphas.split(',')]
    base = Path(args.output_root) / args.param_set
    HMAP = os.path.expanduser(args.height_map_dir)

    import danish
    from lsst.daf.butler import Butler
    from lsst.obs.lsst import LsstCam
    from lsst.ts.wep.estimation import DanishAlgorithm
    from lsst.ts.wep.utils import getTaskInstrument
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import ccd_height as cch

    repo, coll = args.butler_repo, args.collection
    if repo is None or coll is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from intrinsics_lib import load_param_sets
        ps = load_param_sets()[args.param_set]
        repo = repo or ps.get('butler_repo', '/repo/main')
        coll = coll or ps.get('wfs_collection')
    butler = Butler(repo, collections=coll)
    camera = LsstCam.getCamera()
    inst = getTaskInstrument("LSSTCam", "R00_SW0", None)
    algo = DanishAlgorithm()

    # ---- clean (out-of-plane) visits + in-focus seq map ----
    vt = pd.read_parquet(base / 'visits.parquet')
    sel = vt[(vt['band'].astype(str) == 'i')
             & (np.abs(_alt_deg(vt['alt']) - args.alt_center) <= args.alt_tol)].copy()
    sel['gal_b'] = galactic_b(sel['ra'], sel['dec'])
    sel = sel[np.abs(sel['gal_b']) >= args.b_min].sort_values(['day_obs', 'seq_num'])
    wd_meta = pd.read_parquet(base / 'wfs' / 'donuts.parquet',
                              columns=['day_obs', 'seq_num', 'fam_seq_num'])
    infocus = {(int(d), int(f)): int(s) for d, f, s in
               zip(wd_meta['day_obs'], wd_meta['fam_seq_num'], wd_meta['seq_num'])}
    visits = [(int(r.day_obs), int(r.seq_num), float(r.gal_b))
              for r in sel.itertuples() if (int(r.day_obs), int(r.seq_num)) in infocus]
    if args.max_visits:
        visits = visits[:args.max_visits]
    print(f'[wfs_refit_ensemble] {args.param_set}: {len(visits)} clean visits '
          f'(|b|>={args.b_min}, i, alt {args.alt_center}+/-{args.alt_tol}); '
          f'alphas={alphas}')

    rtp_cache, fkw = {}, None
    rows = []
    for (day, fam_seq, gb) in visits:
        seq = infocus[(day, fam_seq)]
        try:
            agg = butler.get('aggregateAOSVisitTableRaw', day_obs=day, seq_num=seq)
        except Exception as e:
            print(f'  visit {day}/{seq}: no agg ({type(e).__name__}); skip'); continue
        noll = [int(x) for x in agg.meta['nollIndices']]
        ei = agg.meta['estimatorInfo']
        det_str = np.asarray(agg['detector']).astype(str)
        for raft in CORNERS:
            det = camera[f'{raft}_SW0'].getId()
            try:
                es = butler.get('donutStampsExtra', day_obs=day, seq_num=seq, detector=det)
                is_ = butler.get('donutStampsIntra', day_obs=day, seq_num=seq, detector=det)
            except Exception:
                continue
            fkw = build_factory_kwargs(inst, np.rad2deg(
                es.metadata["BORESIGHT_PAR_ANGLE_RAD"]
                - es.metadata["BORESIGHT_ROT_ANGLE_RAD"] - np.pi / 2))
            exy = np.array([[s.centroid_position.x, s.centroid_position.y] for s in es])
            ixy = np.array([[s.centroid_position.x, s.centroid_position.y] for s in is_])
            idx = np.where((det_str == f'{raft}_SW0') & np.asarray(agg['used']))[0]
            if len(idx) == 0:
                continue
            # batched per-CCD height Z4 for this corner: 2 calls (SW0/SW1), not per-donut
            cex = [float(agg[i]['centroid_x_extra']) for i in idx]
            cey = [float(agg[i]['centroid_y_extra']) for i in idx]
            cix = [float(agg[i]['centroid_x_intra']) for i in idx]
            ciy = [float(agg[i]['centroid_y_intra']) for i in idx]

            def _hbatch(dn, xs, ys):
                df = pd.DataFrame({'detector': [dn] * len(xs),
                                   'centroid_x_intra': xs, 'centroid_y_intra': ys,
                                   'centroid_x_extra': xs, 'centroid_y_extra': ys})
                return np.asarray(cch.compute_ccd_heights(
                    df, camera, source='batoid_rubin', height_map_dir=HMAP)['Z4_height'], float)
            z4e_arr = _hbatch(f'{raft}_SW0', cex, cey)
            z4i_arr = _hbatch(f'{raft}_SW1', cix, ciy)

            for k, i0 in enumerate(idx):
                row = agg[i0]
                meta = {kk: np.asarray(ei[kk])[i0] for kk in HEIGHT_KEYS}
                ek = int(np.argmin(np.hypot(exy[:, 0] - row['centroid_x_extra'],
                                            exy[:, 1] - row['centroid_y_extra'])))
                ik = int(np.argmin(np.hypot(ixy[:, 0] - row['centroid_x_intra'],
                                            ixy[:, 1] - row['centroid_y_intra'])))
                z4e, z4i = float(z4e_arr[k]), float(z4i_arr[k])
                for alpha in alphas:
                    zk, fwhm, cost, ok = refit_pair(
                        danish, algo, inst, fkw, noll, row, meta, z4e, z4i,
                        es[ek], is_[ik], alpha)
                    if zk is None:
                        continue
                    rec = dict(alpha=alpha, day_obs=day, infocus_seq=seq, fam_seq=fam_seq,
                               gal_b=gb, raft=raft,
                               donut_id_extra=str(row['donut_id_extra']),
                               donut_id_intra=str(row['donut_id_intra']),
                               thx=float(row[f'thx_{coord}']), thy=float(row[f'thy_{coord}']),
                               fwhm=fwhm, cost=cost, success=ok,
                               zk_refit=list(np.asarray(zk, float)),
                               zk_stored=list(np.asarray(row['zk_deviation_CCS'], float)),
                               nollIndices=list(noll))
                    rows.append(rec)
        print(f'  visit {day}/{seq} (b={gb:+.1f}): {sum(r["day_obs"]==day and r["infocus_seq"]==seq for r in rows)} pair-fits so far')

    if not rows:
        sys.exit('No refits produced — check the visit selection / collection.')
    df = pd.DataFrame(rows)
    out = base / 'wfs' / 'wfs_refit_ensemble.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f'  wrote {out}  ({len(df)} pair-fits over {df.donut_id_extra.nunique()} donuts, '
          f'{len(alphas)} alphas)')

    # ---- quick scatter-leads metric: intra-(visit,corner) donut-to-donut RMS vs alpha ----
    zr = np.array(df['zk_refit'].tolist())
    noll0 = df['nollIndices'].iloc[0]
    print('\nintra-(visit,corner) donut-to-donut Zernike scatter [um RMS], pooled:')
    print(f"  {'alpha':>6}" + "".join(f"{'Z'+str(j):>8}" for j in [4, 5, 6, 7, 8]) + f"{'all':>9}")
    for a in alphas:
        m = df['alpha'].values == a
        sub = df[m].copy(); z = zr[m]
        # residual after removing each (visit,corner) mean = within-group scatter
        resid = []
        for _, g in sub.groupby(['day_obs', 'infocus_seq', 'raft']):
            gi = g.index.values
            zz = np.array(g['zk_refit'].tolist())
            if len(zz) >= 2:
                resid.append(zz - zz.mean(0))
        if not resid:
            print(f'  {a:6.2f}  (no multi-donut groups)'); continue
        R = np.vstack(resid)
        rms = np.sqrt(np.nanmean(R ** 2, axis=0))
        print(f'  {a:6.2f}' + "".join(f"{rms[noll0.index(j)]:8.4f}" for j in [4, 5, 6, 7, 8])
              + f"{np.sqrt(np.nanmean(R**2)):9.4f}")
    print('\nNext: score wfs_refit_ensemble.parquet -> FAM agreement + 50/34 v-mode scatter vs alpha.')


if __name__ == '__main__':
    main()
