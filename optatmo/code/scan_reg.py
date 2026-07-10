"""Scan Tikhonov lambda: interior chi^2 vs corner-blow-up (seq 31), no plots."""
import sys, numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from config import load_config, ParamLayout
import fit as fitmod, data_fit
from model import Forward
from vmode_fit import build_vmode_design, wavefront_at
from miw import MIW
import pandas as pd

NPZ = 'data/ofc_svd_50_34_k6.npz'
DAY = 20260513
cfg = load_config('config.yaml')
cfg['geometry']['stamp'] = 24; cfg['geometry']['oversample'] = 12
cfg['atmosphere']['kernel'] = 'VonKarman'
cfg['fit']['moments'] = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03']
cfg['atmosphere']['fit'] = ['fwhm', 'g1', 'g2']
jmax = cfg['geometry']['jmax']
model = fitmod.build_model(cfg)
miw = MIW('../aos/calibration/miw/intrinsic_split_maps_v1.parquet')
n_v = int(np.load(NPZ)['U_eff'].shape[1])
layout = ParamLayout({**cfg, 'moment_offsets': {'moments': [], 'init': 0}},
                     [f'v{i+1}' for i in range(n_v)])

v = pd.read_parquet('../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet')
rot = float(v[(v.day_obs == DAY) & (v.seq_num == 30)].rotator_angle.iloc[0])
cw = pd.read_parquet(f'data/cwfs_{DAY}00031.parquet'); cw['c'] = cw.detector.str[:3]
corners = [(np.median(cw[cw.c == c].thx_OCS) * 180 / np.pi,
            np.median(cw[cw.c == c].thy_OCS) * 180 / np.pi) for c in ['R00', 'R04', 'R40', 'R44']]
cxs = np.array([c[0] for c in corners]); cys = np.array([c[1] for c in corners])

prep = data_fit.load_and_prep(f'data/psfmoments_{DAY}00031.parquet', sign=1, rot_deg=rot)
binned = data_fit.bin_grid(prep, cell_deg=0.30); cat = data_fit.to_catalog(binned)
G_v = build_vmode_design(NPZ, cat['thx_deg'], cat['thy_deg'], jmax, 1.75)[0]
z0 = np.nan_to_num(miw.zernikes(cat['thx_deg'], cat['thy_deg'], cat['rotator_rad'], jmax))

print(f'{"lambda":>8} {"chi2/n":>9} {"max|A|":>8} {"max|corner dev Zj|":>18}')
for lam in [0.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]:
    fwd = Forward(model, layout, z0, G_v, cat['moments'], cat['errors'],
                  cfg['fit']['moments'], {m: 1.0 for m in cfg['fit']['moments']}, reg_lambda=lam)
    fwd0 = Forward(model, layout, z0, G_v, cat['moments'], cat['errors'],
                   cfg['fit']['moments'], {m: 1.0 for m in cfg['fit']['moments']}, reg_lambda=0.0)
    vg = jax.jit(jax.value_and_grad(fwd.cost))
    r = minimize(lambda p: tuple(np.asarray(x, float) if i else float(x)
                                 for i, x in enumerate(vg(jnp.asarray(p)))),
                 layout.initial(), jac=True, method='L-BFGS-B', options={'maxiter': 300})
    A = r.x[layout.i_dz]
    chi2 = float(fwd0.cost(jnp.asarray(r.x)))
    cornerdev = wavefront_at(A, NPZ, cxs, cys, jmax=jmax, fp_radius=1.75)[:, 4:16]
    print(f'{lam:8.0f} {chi2:9.2f} {np.max(np.abs(A)):8.3f} '
          f'{np.max(np.abs(cornerdev)):18.3f}')
