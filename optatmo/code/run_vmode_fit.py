"""Run the v-Mode OptAtmo fit on real PSF-star moments (seq 31 & 34, 20260513).

wavefront = MIW(intrinsic, OCS) + G_v @ A  (12 v-mode amplitudes)
atmosphere = VonKarman(fwhm, L0=25) sheared by (g1,g2)
Fit A + (fwhm,g1,g2) to the OCS-rotated, grid-binned HSM moments.
Outputs the fitted v-mode amplitudes and the wavefront for the corner comparison.
"""
import numpy as np, pandas as pd, jax, jax.numpy as jnp
from scipy.optimize import minimize

from config import load_config, ParamLayout
import fit as fitmod
import data_fit
from model import Forward
from vmode_fit import build_vmode_design
from miw import MIW

import sys as _sys
NPZ = next((a for a in _sys.argv[1:] if a.endswith('.npz')), 'data/ofc_svd_22_12.npz')
MIW_PARQUET = '../aos/calibration/miw/intrinsic_split_maps_v1.parquet'
VISITS_PARQUET = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'
DAY = 20260513


def rot_for(seq):
    v = pd.read_parquet(VISITS_PARQUET)
    r = v[(v.day_obs == DAY) & (v.seq_num == seq - 1)]
    return float(r.rotator_angle.iloc[0]) if len(r) else 0.0


def main():
    cfg = load_config('config.yaml')
    cfg['geometry']['stamp'] = 24
    cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'VonKarman'
    cfg['fit']['moments'] = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03']
    cfg['atmosphere']['fit'] = ['fwhm', 'g1', 'g2']
    jmax = cfg['geometry']['jmax']

    model = fitmod.build_model(cfg)
    miw = MIW(MIW_PARQUET)
    n_v = int(np.load(NPZ)['U_eff'].shape[1])      # #v-modes from the SVD file
    vmode_names = [f'v{i+1}' for i in range(n_v)]
    print(f'v-mode fit: {n_v} modes from {NPZ}')
    layout = ParamLayout({**cfg, 'moment_offsets': {'moments': [], 'init': 0}},
                         vmode_names)

    import sys
    SIGN = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    REG = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('reg=')),
               0.0)                            # Tikhonov L2 on v-mode amplitudes
    print(f'### rotation sign = {SIGN:+d}, reg_lambda = {REG:g} ###')
    out = {}
    for seq in [25, 28]:
        rot = rot_for(seq)
        prep = data_fit.load_and_prep(f'data/psfmoments_{DAY}000{seq}.parquet',
                                      sign=SIGN, rot_deg=rot)
        binned = data_fit.bin_grid(prep, cell_deg=0.30)
        cat = data_fit.to_catalog(binned)
        G_v, _, _ = build_vmode_design(NPZ, cat['thx_deg'], cat['thy_deg'],
                                       jmax, fp_radius=1.75)
        z0 = np.nan_to_num(miw.zernikes(cat['thx_deg'], cat['thy_deg'],
                                        cat['rotator_rad'], jmax))
        fwd = Forward(model, layout, z0, G_v, cat['moments'], cat['errors'],
                      cfg['fit']['moments'], {m: 1.0 for m in cfg['fit']['moments']},
                      reg_lambda=REG)

        vg = jax.jit(jax.value_and_grad(fwd.cost))
        p0 = layout.initial()

        def fun(p):
            v, g = vg(jnp.asarray(p))
            return float(v), np.asarray(g, float)
        res = minimize(fun, p0, jac=True, method='L-BFGS-B',
                       bounds=layout.bounds(), options={'maxiter': 300})
        A = res.x[layout.i_dz]
        atm = {a: res.x[layout.n_dz + i] for i, a in enumerate(layout.atm_free)}
        model_mom = np.array(fwd.moments(jnp.asarray(res.x)))   # (n_cells, 12)
        print(f'\n=== seq {seq} (rot {rot:.1f}): cost={res.fun:.3f} '
              f'nit={res.nit} success={res.success} ===')
        print('  v-mode amps:', np.round(A, 3))
        print('  atm:', {k: round(v, 4) for k, v in atm.items()})
        np.savez(f'data/vmodefit_{seq}.npz', A=A, atm=np.array(list(atm.values())),
                 thx=cat['thx_deg'], thy=cat['thy_deg'], rot=rot,
                 data_mom=cat['moments'], model_mom=model_mom,
                 data_err=cat['errors'])
        out[seq] = (A, atm)

    # v-mode amplitudes across the two rotators (not expected identical —
    # turbulence stochasticity, esp. overall k=1 Z4-Z8)
    ks = list(out)
    for k in ks:
        print(f'  seq{k}:', np.round(out[k][0], 3))


if __name__ == '__main__':
    main()
