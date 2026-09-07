#!/usr/bin/env python
"""run_closed_loop — AOS closed-loop simulation of the delivered optical PSF.

Simulates proportional control of the Active Optics System over a sequence of Full
Array Mode (FAM) visits: at each step the corner wavefront recovery estimates the
optical state, the controller subtracts an assumed intrinsic and applies a gain, and the
residual wavefront is rendered to a Point Spread Function (PSF) to give the delivered
full width at half maximum (FWHM).

Knobs that matter physically:

  --gain        proportional control gain (dimensionless)
  --latency     nplustwo: correction from image n applied at n+2, the n+1 measurement
                ignored (realistic); nplusone: applied at n+1 (future goal)
  --intrinsic   what the controller subtracts to estimate degrees of freedom (DOF):
                tabulated = batoid design, miw = Measured Intrinsic Wavefront, none = raw OPD
  --order       ordered = day_obs/seq_num, so groups of same-position visits;
                random = shuffled, i.e. random slewing. Reality is in between.

Usage:
  python code/closedloop/run_closed_loop.py --case loop50
  python code/closedloop/run_closed_loop.py --case loop --gain 0.5 --latency nplusone

Writes output/<param_set>/<mi>/closedloop/closedloop_<case>_<band><suffix>.pdf, where the
suffix records the intrinsic, order, latency and gain, so runs with different control
settings sit side by side.

Requires lsst.ts.ofc (the sensitivity-matrix SVD), lsst.obs.lsst (camera geometry) and
galsim (rendering) -- RSP or an s3df node with the AOS/CWFS environment.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('NUMEXPR_MAX_THREADS', '8')

sys.path.insert(0, str(Path(__file__).resolve().parent))          # same-study siblings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # aos/code (shared)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))      # repo root -> common/
from common.psf_render import (   # noqa: E402
    LAM_NM, build_psf_tools, render_measure)
from psf_maps_lib import (   # noqa: E402
    FP_RADIUS, sample_science_stars, measure_zk,
    psf_page, build_svd, eval_dz_field,
    corner_matrix_at, _design_intrinsic_at)



def loop_corner_data(base_ps, fam_mi_dir, coord, noll, sec, visits, intrinsic):
    """Per-visit 4-corner WFS measurement for the closed loop.  Donuts are selected in the
    FIXED CAMERA (CCS) corner wedges; the OCS measurement medians and OCS corner positions
    are returned (so all rotators are handled from the data, no OCS<->CCS rotation assumed).
    The measured quantity is the wavefront the *controller* sees after subtracting its
    assumed intrinsic:  tabulated -> zk_deviation (zk - design);  miw -> zk - zk_intrinsic_MI;
    none -> raw zk.  Returns {(day,seq): (z (4,nZk), pos (4,2) thx/thy_OCS deg)} or None."""
    import pyarrow.parquet as _pq
    base_cols = ['day_obs', 'seq_num', 'thx_CCS', 'thy_CCS', 'thx_OCS', 'thy_OCS']
    val = f'zk_deviation_{coord}' if intrinsic == 'tabulated' else f'zk_{coord}'
    dd = _pq.read_table(str(base_ps / 'donuts.parquet'), columns=base_cols + [val]).to_pandas()
    vt = _pq.read_table(str(base_ps / 'visits.parquet'), columns=['nollIndices']).to_pandas()
    noll_m = [int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
    im = [noll_m.index(j) for j in noll]
    meas = np.stack(dd[val].values).astype(float)[:, im]
    if intrinsic == 'miw':
        sc = _pq.read_table(str(fam_mi_dir / 'zk_intrinsic.parquet'), columns=['zk_intrinsic_MI']).to_pandas()
        md = _pq.read_schema(str(fam_mi_dir / 'zk_intrinsic.parquet')).metadata or {}
        noll_i = (np.frombuffer(md[b'nollIndices'], dtype=int).tolist() if b'nollIndices' in md else noll_m)
        meas = meas - np.stack(sc['zk_intrinsic_MI'].values).astype(float)[:, [noll_i.index(j) for j in noll]]
    day = dd.day_obs.astype(int).values; seq = dd.seq_num.astype(int).values
    cx = np.rad2deg(dd.thx_CCS.astype(float).values); cy = np.rad2deg(dd.thy_CCS.astype(float).values)
    ox = np.rad2deg(dd.thx_OCS.astype(float).values); oy = np.rad2deg(dd.thy_OCS.astype(float).values)
    rC = np.hypot(cx, cy); azC = np.degrees(np.arctan2(cy, cx)) % 360.0
    half = sec['wfs_azimuth_width_deg'] / 2.0
    rin, rout = sec['wfs_inner_radius_deg'], sec['wfs_outer_radius_deg']
    out = {}
    for r in visits.itertuples():
        d, s = int(r.day_obs), int(r.seq_num); vis = (day == d) & (seq == s)
        z = np.full((4, len(noll)), np.nan); pos = np.full((4, 2), np.nan); ok = True
        for ci, off in enumerate(MIMIC_OFFSETS):
            ctr = (sec['delta_deg'] + off) % 360.0; lo, hi = (ctr - half) % 360.0, (ctr + half) % 360.0
            azin = (azC >= lo) & (azC <= hi) if lo < hi else (azC >= lo) | (azC <= hi)
            w = vis & (rC >= rin) & (rC <= rout) & azin
            if int(w.sum()) < sec['min_donuts_per_wedge']:
                ok = False; break
            z[ci] = np.nanmedian(meas[w], axis=0); pos[ci] = [np.nanmedian(ox[w]), np.nanmedian(oy[w])]
        out[(d, s)] = (z, pos) if ok else None
    print(f'[loop] intrinsic={intrinsic}: {sum(v is not None for v in out.values())}/{len(out)} '
          f'visits with all 4 CCS-corner wedges')
    return out



def run_closed_loop(case, base, args, svd, noll, stars, atm, aper, lam_nm, fwhm_atm, pdf):
    """Design-relative proportional closed loop over a FAM disturbance sequence (all rotators).
    Truth W = deviation-from-design (top-level fits); baseline = design intrinsic per star.
    Each visit i:
      r = W_i - c ;  render PSF from design + r
      (measured images only) y = z_corners_i - B·c ;  A = pinv(B·U_eff)·y
      schedule  c += g·U_eff·A  to take effect at image i+L   (n+2 latency: L=2, every other
      image measured; n+1: L=1, every image).  --intrinsic sets what the controller subtracts."""
    import matplotlib.pyplot as plt
    from run_wfs_mimic import DEFAULT as MD
    sec = {**MD, 'delta_deg': args.mimic_delta}
    prefix, g = args.dz_prefix, args.gain
    fits = pd.read_parquet(base / 'fits.parquet')          # top-level = deviation from design
    fits = (fits.sample(frac=1, random_state=args.seed) if args.order == 'random'
            else fits.sort_values(['day_obs', 'seq_num']))
    if args.max_visits and args.max_visits > 0:
        fits = fits.head(args.max_visits)
    fits = fits.reset_index(drop=True)
    corners = loop_corner_data(base, base / args.mi, args.coord, noll, sec, fits, args.intrinsic)
    design = _design_intrinsic_at(stars, args.band, noll)
    L = 2 if args.latency == 'nplustwo' else 1
    tag = (('50DOF/34vmode' if case.endswith('50') else '22DOF/12vmode')
           + f' g={g} {args.intrinsic} {args.order} {args.latency}')
    c = np.zeros(len(svd.kj_grid)); scheduled = {}; ts = []; optics = []; sample = {}
    step = 0; last_meas = None
    for i, row in fits.iterrows():
        if i in scheduled:
            c = c + scheduled.pop(i)
        W = np.nan_to_num(np.array([float(row.get(f'{prefix}_z{j}_c{k}', np.nan)) for (k, j) in svd.kj_grid]))
        meas = measure_zk(design + eval_dz_field((W - c)[None, :], svd, noll, stars), noll, lam_nm, atm, aper)
        rtp = float(row.get('rotator_angle', np.nan))
        ts.append((step, rtp, float(meas.fwhm.median()), float(meas.e.median())))
        optics.append((step, meas['fwhm'].values - fwhm_atm))
        if i % L == 0:                                     # measured image (stride = latency)
            cm = corners.get((int(row.day_obs), int(row.seq_num)))
            if cm is not None and np.all(np.isfinite(cm[0])):
                z_corners, pos = cm; B = corner_matrix_at(svd, noll, pos)
                A = np.linalg.pinv(B @ svd.U_eff) @ (z_corners.ravel() - B @ c)
                scheduled[i + L] = scheduled.get(i + L, 0.0) + g * (svd.U_eff @ A)
        if step == 0:
            sample[0] = meas
        last_meas = meas; step += 1
        print(f'  [{case}] step {step} {int(row.day_obs)}/{int(row.seq_num)} rtp={rtp:+.0f}: '
              f'medFWHM={meas.fwhm.median():.3f}" e={meas.e.median():.3f}')
    if not ts:
        print(f'[{case}] no usable visits'); return
    sample[ts[-1][0]] = last_meas
    a = np.array([(t[0], t[2], t[3]) for t in ts])
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax[0].plot(a[:, 0], a[:, 1], '-o', ms=3); ax[0].axhline(fwhm_atm, ls=':', color='gray', label='atm')
    ax[0].set_ylabel('median FWHM ["]'); ax[0].legend(fontsize=8); ax[0].set_title(f'Closed loop — {tag}')
    ax[1].plot(a[:, 0], a[:, 2], '-o', ms=3, color='darkorange')
    ax[1].set_ylabel('median e'); ax[1].set_xlabel('visit step')
    pdf.savefig(fig); plt.close(fig)
    burn = args.burn_in
    pool = np.concatenate([o for st, o in optics if st >= burn] or [o for _, o in optics])
    pool = pool[np.isfinite(pool)]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(pool, bins=50, color='slategray')
    ax.text(0.97, 0.95, f'steps≥{burn}\nmean={pool.mean():.3f}"\nrms={pool.std():.3f}"',
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    ax.set_xlabel('optics FWHM contribution = FWHM − atm [arcsec]')
    ax.set_title(f'Closed-loop steady-state optics FWHM — {tag}')
    pdf.savefig(fig); plt.close(fig)
    for st in sorted(sample):
        psf_page(stars, sample[st], f'Closed loop step {st} — {tag} ({args.band})', fwhm_atm, pdf)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ps', default='fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x')
    ap.add_argument('--mi', default='pathA_50_34_i_5rot',
                    help='measured-intrinsic build supplying both the MIW split maps and '
                         'the per-visit FAM fits')
    ap.add_argument('--case', default='loop50', choices=['loop50', 'loop22', 'loop'],
                    help="loop50 = 50-DOF/34-v-mode scheme, loop22 = 22-DOF/12-v-mode, "
                         "loop = both")
    ap.add_argument('--gain', type=float, default=0.3,
                    help='proportional control gain (dimensionless)')
    ap.add_argument('--burn-in', type=int, default=5,
                    help='loop steps to skip for the steady-state histogram')
    ap.add_argument('--intrinsic', default='tabulated',
                    choices=['tabulated', 'miw', 'none'],
                    help='intrinsic the controller subtracts to estimate DOF')
    ap.add_argument('--order', default='ordered', choices=['ordered', 'random'],
                    help='visit order: ordered = day_obs,seq_num; random = shuffled')
    ap.add_argument('--latency', default='nplustwo', choices=['nplustwo', 'nplusone'],
                    help='nplustwo: correction from image n applied at n+2 (realistic); '
                         'nplusone: applied at n+1 (future goal)')
    ap.add_argument('--mimic-delta', type=float, default=0.0,
                    help='azimuth offset of the 4 WFS-mimic corners (deg)')
    ap.add_argument('--band', default='i')
    ap.add_argument('--n-stars', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--day-obs', type=int, default=20260315)
    ap.add_argument('--rot-lim', type=float, default=3.0)
    ap.add_argument('--max-visits', type=int, default=24)
    ap.add_argument('--dz-prefix', default='z1toz6')
    ap.add_argument('--coord', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--height-map-dir',
                    default='~/u/LSST/packages/batoid_rubin_data')
    args = ap.parse_args()

    from lsst.obs.lsst import LsstCam
    camera = LsstCam.getCamera()
    base = Path(args.output_root) / args.ps
    lam_nm = LAM_NM[args.band]
    cases = ['loop50', 'loop22'] if args.case == 'loop' else [args.case]

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    stars = sample_science_stars(camera, args.n_stars, FP_RADIUS, args.seed)
    atm, aper = build_psf_tools(lam_nm)
    hmap = os.path.expanduser(args.height_map_dir)

    # The MIW split maps fix the Noll index set used throughout, and give the
    # atmosphere-only FWHM baseline the loop pages are compared against.
    _miw_zk, noll = miw_zernikes(
        stars, base / args.mi / 'intrinsic_split_maps.parquet', camera, hmap)
    fwhm_atm = render_measure(None, noll, lam_nm, atm, aper,
                              with_optics=False)['fwhm']

    out_dir = base / args.mi / 'closedloop'
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        n_keep = 34 if case.endswith('50') else 12
        n_dof = None if case.endswith('50') else 22
        suffix = (f'_{args.intrinsic}_{args.order}_{args.latency}'
                  f'_g{args.gain:g}')
        out = out_dir / f'closedloop_{case}_{args.band}{suffix}.pdf'
        with PdfPages(str(out)) as pdf:
            run_closed_loop(case, base, args, build_svd(noll, n_keep, n_dof),
                            noll, stars, atm, aper, lam_nm, fwhm_atm, pdf)
        print('wrote', out)


if __name__ == '__main__':
    main()
