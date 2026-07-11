"""Run the v-Mode OptAtmo fit on real PSF-star moments (seq 25 & 28, 20260513).

wavefront = MIW_official(intrinsic, OCS-frame) + G_v @ A  (v-mode amplitudes)
  MIW_official = official ip_isr intrinsicZernikes product (per-detector CCS with
  CCD height folded into Z4), read from the Butler and reconstructed in the OCS
  frame per-detector (miw.MIWCalib).
atmosphere = VonKarman(fwhm, L0=25) sheared by (g1,g2)
Fit A + (fwhm,g1,g2) to the OCS-rotated, per-detector sub-CCD-binned HSM moments.
Outputs the fitted v-mode amplitudes and the wavefront for the corner comparison.
"""
import numpy as np, pandas as pd, jax, jax.numpy as jnp
from scipy.optimize import minimize

from config import load_config, ParamLayout
import fit as fitmod
import data_fit
from model import Forward
from vmode_fit import build_vmode_design, cwfs_vmode_amps
from miw import MIWCalib
from fit_monitor import FitMonitor

import sys as _sys
NPZ = next((a for a in _sys.argv[1:] if a.endswith('.npz')), 'data/ofc_svd_22_12.npz')
# official ip_isr intrinsicZernikes calib, read straight from the Butler
# (per-CCD CCS with CCD height in Z4); tokens: coll=, filt=, repo=
MIW_COLL = next((a.split('=', 1)[1] for a in _sys.argv if a.startswith('coll=')),
                'u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
MIW_FILT = next((a.split('=', 1)[1] for a in _sys.argv if a.startswith('filt=')),
                'i_39')
MIW_REPO = next((a.split('=', 1)[1] for a in _sys.argv if a.startswith('repo=')),
                '/repo/main')
VISITS_PARQUET = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'
DAY = next((int(a.split('=')[1]) for a in _sys.argv if a.startswith('day=')),
           20260513)
_seqs = next((a.split('=')[1] for a in _sys.argv if a.startswith('seqs=')), None)
SEQS = [int(s) for s in _seqs.split(',')] if _seqs else [25, 28]


def visit_of(seq):
    return int(f'{DAY}{seq:05d}')


def rot_for(seq):
    v = pd.read_parquet(VISITS_PARQUET)
    r = v[(v.day_obs == DAY) & (v.seq_num == seq - 1)]
    return float(r.rotator_angle.iloc[0]) if len(r) else 0.0


def main():
    cfg = load_config('config.yaml')
    cfg['geometry']['stamp'] = 24
    cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'VonKarman'
    cfg['atmosphere']['fit'] = ['fwhm', 'g1', 'g2']
    jmax = cfg['geometry']['jmax']
    fit_moments = cfg['fit']['moments']            # from config.yaml (incl. M22)
    weights = cfg['fit'].get('weights', {}) or {}  # relative per-moment weights
    print(f'fit moments: {fit_moments}')
    print(f'weights: {[(m, weights.get(m, 1.0)) for m in fit_moments]}')

    model = fitmod.build_model(cfg)
    miw = MIWCalib(MIW_COLL, physical_filter=MIW_FILT, repo=MIW_REPO)
    n_v = int(np.load(NPZ)['U_eff'].shape[1])      # #v-modes from the SVD file
    vmode_names = [f'v{i+1}' for i in range(n_v)]
    print(f'v-mode fit: {n_v} modes from {NPZ}')
    layout = ParamLayout({**cfg, 'moment_offsets': {'moments': [], 'init': 0}},
                         vmode_names)

    import sys
    SIGN = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    REG = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('reg=')),
               0.0)                            # Tikhonov L2 on v-mode amplitudes
    INIT = next((a.split('=')[1] for a in sys.argv if a.startswith('init=')),
                'zero')                         # v-mode start: zero | cwfs
    OPTICS = next((a.split('=')[1] for a in sys.argv if a.startswith('optics=')),
                  'free')                        # free | fixed (freeze v-modes)
    _parts = ([INIT] if INIT != 'zero' else []) + (['atmonly'] if OPTICS == 'fixed'
                                                   else [])
    tag = ('_' + '_'.join(_parts)) if _parts else ''   # output suffix (no clobber)
    print(f'### rotation sign = {SIGN:+d}, reg_lambda = {REG:g}, '
          f'v-mode init = {INIT}, optics = {OPTICS} ###')
    out = {}
    for seq in SEQS:
        rot = rot_for(seq)
        prep = data_fit.load_and_prep(f'data/psfmoments_{visit_of(seq)}.parquet',
                                      sign=SIGN, rot_deg=rot)
        binned = data_fit.bin_grid(prep, cell_deg=cfg['fit'].get('cell_deg', 0.10),
                                   min_n=cfg['fit'].get('min_n', 3))
        cat = data_fit.to_catalog(binned)
        G_v, _, _ = build_vmode_design(NPZ, cat['thx_deg'], cat['thy_deg'],
                                       jmax, fp_radius=1.75)
        z0 = np.nan_to_num(miw.zernikes(cat['thx_deg'], cat['thy_deg'],
                                        cat['rotator_rad'], jmax,
                                        cat['detector']))
        fwd = Forward(model, layout, z0, G_v, cat['moments'], cat['errors'],
                      fit_moments, weights, reg_lambda=REG)

        vg = jax.jit(jax.value_and_grad(fwd.cost))
        p0 = np.array(layout.initial(), float)
        bnds = list(layout.bounds())
        lo = np.array([b[0] if b[0] is not None else -np.inf for b in bnds])
        hi = np.array([b[1] if b[1] is not None else np.inf for b in bnds])
        A_init = np.zeros(layout.n_dz)
        if INIT == 'cwfs':
            # start the v-modes at the CWFS-expected optical state
            A_init = cwfs_vmode_amps(f'data/cwfs_{visit_of(seq)}.parquet', miw,
                                     NPZ, np.deg2rad(rot), jmax, fp_radius=1.75,
                                     offsets=cfg.get('cwfs', {}).get('offsets', {}))
            p0[layout.i_dz] = A_init
            p0 = np.clip(p0, lo, hi)
            print('  init v-modes from CWFS:', np.round(A_init, 3))
        dz_idx = (list(range(*layout.i_dz.indices(len(p0))))
                  if isinstance(layout.i_dz, slice) else list(np.atleast_1d(layout.i_dz)))
        if OPTICS == 'fixed':
            # freeze the v-modes at their init; fit only the atmosphere
            for i in dz_idx:
                bnds[i] = (float(p0[i]), float(p0[i]))
            print(f'  optics FIXED at init -- fitting only atmosphere '
                  f'{layout.atm_free}')

        mon = FitMonitor(label=f'fit seq{seq}{tag}', verbose=True,
                         checkpoint=f'data/fitprog_{seq}{tag}.npz')
        fun = mon.objective(vg)
        mon.start()
        res = minimize(fun, p0, jac=True, method='L-BFGS-B',
                       bounds=bnds, callback=mon.callback,
                       options={'maxiter': 300})
        mon.stop()
        A = res.x[layout.i_dz]
        atm = {a: res.x[layout.n_dz + i] for i, a in enumerate(layout.atm_free)}
        model_mom = np.array(fwd.moments(jnp.asarray(res.x)))   # (n_cells, 12)
        print(f'\n=== seq {seq} (rot {rot:.1f}): cost={res.fun:.3f} '
              f'nit={res.nit} success={res.success} ===')
        print('  v-mode amps:', np.round(A, 3))
        print('  atm:', {k: round(v, 4) for k, v in atm.items()})
        print('  fit monitor:', mon.summary_line(res))

        atm_idx = [layout.n_dz + i for i in range(len(layout.atm_free))]
        mon.plot(f'output/fitmon_{seq}{tag}.png', res, layout.i_dz, vmode_names,
                 atm_idx, layout.atm_free, reg_lambda=REG,
                 title=f'20260513 seq={seq} (rot {rot:.1f}) init={INIT}')
        st = mon.stats(res)
        np.savez(f'data/vmodefit_{seq}{tag}.npz', A=A,
                 atm=np.array(list(atm.values())),
                 A_init=A_init, init=INIT,
                 thx=cat['thx_deg'], thy=cat['thy_deg'], rot=rot,
                 detector=cat['detector'],
                 data_mom=cat['moments'], model_mom=model_mom,
                 data_err=cat['errors'],
                 mon_costs=st['costs'], mon_params=st['params'],
                 mon_iter_evals=st['iter_evals'], nfev=st['nfev'],
                 njev=st['njev'], nit=st['nit'], fit_time_s=st['time_s'])
        out[seq] = (A, atm)

    # v-mode amplitudes across the two rotators (not expected identical —
    # turbulence stochasticity, esp. overall k=1 Z4-Z8)
    ks = list(out)
    for k in ks:
        print(f'  seq{k}:', np.round(out[k][0], 3))


if __name__ == '__main__':
    main()
