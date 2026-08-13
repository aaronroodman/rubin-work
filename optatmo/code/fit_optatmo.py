"""fit_optatmo -- the optatmo Optics+Atmosphere v-mode fit driver (staged).

Fits, per in-focus visit, a wavefront = MIW_official(OCS) + G_v @ A (v-mode
amplitudes) plus a VonKarman atmosphere (fwhm, g1, g2) to the OCS-rotated,
per-detector sub-CCD-binned HSM moments.

A fit is a PLAN = an ordered list of STAGES.  Each stage frees a subset of the
v-modes (and optionally the atmosphere) and WARM-STARTS from the previous
stage's optimum; stage 1 starts from --init (zero | cwfs).  Presets:

    full             1 stage: all v-modes + atmosphere            (default)
    focus            1 stage: v1 (focus) + atmosphere only
    focus_then_full  stage1 v1+atm  ->  stage2 warm-start, free all v-modes

All stages' results go into ONE npz per visit (per-stage A / atm / model_mom /
cost), enough to remake any moment map, scatter, or corner plot for either
stage.  Minimizer: --minimizer lbfgsb (scipy L-BFGS-B) | migrad (iminuit).

Outputs are organized per campaign (see campaign.py):
    output/runs/<campaign>/<day>/fits/vmodefit_<seq>.npz
                                 /reports/fitmon_<seq>.png
Inputs (psfmoments/cwfs/visitmeta, SVD) are shared under data/.

    python code/fit_optatmo.py --day 20260513 --seqs 25 28 \
        --campaign focus_then_full --plan focus_then_full --init cwfs \
        --minimizer lbfgsb --coll <MIW_COLL> --filt i_39
"""
import argparse
import os
from collections import namedtuple
from types import SimpleNamespace

import numpy as np, pandas as pd, jax
# float64 is REQUIRED for iminuit MIGRAD: at the cost scale (~1e4) float32 gives
# only ~1e-3 absolute precision, so MIGRAD's small steps read as "no improvement"
# and it stalls at the start.  Also makes L-BFGS-B more accurate; must be set
# before any jax computation is traced.
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.optimize import minimize

import campaign as camp
from config import load_config, ParamLayout
import fit as fitmod
import data_fit
from model import Forward
from vmode_fit import build_vmode_design, cwfs_vmode_amps
from miw import MIWCalib
from fit_monitor import FitMonitor

VISITS = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'

Stage = namedtuple('Stage', 'name free_vmodes free_atm')   # free_vmodes: 'all' | [ints]
PLANS = {
    'full': [Stage('full', 'all', True)],
    'focus': [Stage('focus', [1], True)],
    'focus_then_full': [Stage('focus', [1], True), Stage('full', 'all', True)],
}


def run_migrad(vg, p0, bnds, mon, maxcall=200000):
    """Minimize with iminuit MIGRAD using the jax value_and_grad (exact grads).

    Records every unique evaluation into `mon` and returns a scipy-like result
    (x, fun, nfev, njev, nit, success).  Bounds with lo==hi are fixed params.
    """
    from iminuit import Minuit
    last = {'k': None}

    def _both(par):
        par = np.asarray(par, float)
        key = par.tobytes()
        if key != last['k']:
            v, g = vg(jnp.asarray(par))
            last.update(k=key, v=float(v), g=np.asarray(g, float))
            mon.costs.append(float(v)); mon.params.append(par.copy())
            if mon.verbose:
                print(f'[{mon.label}] eval {len(mon.costs):4d}  cost {float(v):.6g}'
                      f'  t {mon._elapsed():6.0f}s', flush=True)
        return last

    m = Minuit(lambda p: _both(p)['v'], np.asarray(p0, float),
               grad=lambda p: _both(p)['g'])
    m.errordef = 1.0
    # strategy 0 is ESSENTIAL with an exact (jax autodiff) gradient: it trusts the
    # gradient and skips MIGRAD's numerical-Hessian probing (~2*npar evals before
    # the first step), which otherwise makes the fit ~30x slower.  We only want
    # the minimum, not the error matrix.
    m.strategy = 0
    for i, (lo, hi) in enumerate(bnds):
        lo_f = None if (lo is None or not np.isfinite(lo)) else float(lo)
        hi_f = None if (hi is None or not np.isfinite(hi)) else float(hi)
        if lo_f is not None and hi_f is not None and lo_f == hi_f:
            m.values[i] = lo_f; m.fixed[i] = True
        else:
            m.limits[i] = (lo_f, hi_f)
    m.migrad(ncall=maxcall)
    mon.iter_evals.append(len(mon.costs))
    return SimpleNamespace(x=np.array(m.values, float), fun=float(m.fval),
                           nfev=int(m.nfcn), njev=int(m.ngrad), nit=int(m.nfcn),
                           success=bool(m.valid))


def minimize_stage(vg, p, bnds, mon, minimizer):
    if minimizer == 'migrad':
        return run_migrad(vg, p, bnds, mon)
    res = minimize(mon.objective(vg), p, jac=True, method='L-BFGS-B',
                   bounds=bnds, callback=mon.callback, options={'maxiter': 300})
    return res


def rot_for(seq, day):
    """Rotator angle (deg): prefer the per-visit ConsDB meta, else danish table."""
    import os
    visit = int(f'{day}{seq:05d}')
    mp = camp.visitmeta_path(visit)
    if os.path.exists(mp):
        return float(pd.read_parquet(mp).iloc[0]['rot_deg'])
    if os.path.exists(VISITS):
        v = pd.read_parquet(VISITS)
        r = v[(v.day_obs == day) & (v.seq_num == seq - 1)]
        if len(r):
            return float(r.rotator_angle.iloc[0])
    return 0.0


def parse_args():
    ap = argparse.ArgumentParser(description='optatmo staged v-mode fit')
    ap.add_argument('--day', type=int, required=True)
    ap.add_argument('--seqs', type=int, nargs='+', required=True)
    ap.add_argument('--campaign', required=True, help='output run label (required)')
    ap.add_argument('--plan', default='full', choices=list(PLANS))
    ap.add_argument('--init', default='cwfs', choices=['zero', 'cwfs'])
    ap.add_argument('--minimizer', default='lbfgsb', choices=['lbfgsb', 'migrad'])
    ap.add_argument('--svd', default='data/svd/ofc_svd_50_34_k6.npz')
    ap.add_argument('--coll', default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filt', default='i_39')
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--sign', type=int, default=1, help='CCS->OCS rotation sign')
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip visits whose campaign fit npz already exists')
    ap.add_argument('--moffsets', default='off', help='off | higher | all | csv')
    # regularization (off by default; kept as an option)
    ap.add_argument('--reg', type=float, default=None, help='lambda (default: config)')
    ap.add_argument('--regmode', default=None, choices=['vmode', 'wavefront'])
    ap.add_argument('--regform', default=None, choices=['quadratic', 'power', 'hinge'])
    ap.add_argument('--regpow', type=float, default=None)
    ap.add_argument('--regknee', type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config('config.yaml')
    cfg['geometry']['stamp'] = 24
    cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'VonKarman'
    cfg['atmosphere']['fit'] = ['fwhm', 'g1', 'g2']
    jmax = cfg['geometry']['jmax']
    fit_moments = cfg['fit']['moments']
    weights = cfg['fit'].get('weights', {}) or {}

    # optional spatially-constant per-moment offsets
    _higher = [m for m in fit_moments if m not in ('e0', 'e1', 'e2')]
    if args.moffsets in ('off', 'none', ''):
        moff_list = []
    elif args.moffsets == 'higher':
        moff_list = _higher
    elif args.moffsets == 'all':
        moff_list = list(fit_moments)
    else:
        moff_list = [m for m in args.moffsets.split(',') if m]

    # regularization config (CLI overrides config; default off = lambda 0)
    rc = cfg.get('regularization', {}) or {}
    REGMODE = args.regmode or rc.get('mode', 'vmode')
    REGFORM = args.regform or rc.get('form', 'quadratic')
    REGPOW = args.regpow if args.regpow is not None else float(rc.get('power', 2.0))
    REGKNEE = args.regknee if args.regknee is not None else float(rc.get('knee', 1.0))
    _lam_def = float(rc.get('lambda_wavefront', 0.0) if REGMODE == 'wavefront'
                     else rc.get('lambda', 0.0))
    REG = args.reg if args.reg is not None else _lam_def
    reg_w = None
    if REGMODE == 'wavefront':
        reg_w = np.zeros(jmax + 1)
        for j, s in (rc.get('sigma_j', {}) or {}).items():
            if 0 < int(j) <= jmax and s and float(s) > 0:
                reg_w[int(j)] = 1.0 / float(s) ** 2
        if not reg_w.any():
            raise SystemExit('regmode=wavefront but regularization.sigma_j is empty')

    plan = PLANS[args.plan]
    print(f'campaign={args.campaign} day={args.day} plan={args.plan} '
          f'stages={[s.name for s in plan]} init={args.init} '
          f'minimizer={args.minimizer} reg={REG:g}({REGMODE})')
    print(f'fit moments: {fit_moments}  moffsets({args.moffsets}): {moff_list}')

    campn = camp.Campaign(args.campaign, args.day).ensure()
    campn.snapshot_config('config.yaml')
    campn.write_manifest(dict(day=args.day, seqs=args.seqs, plan=args.plan,
                              init=args.init, minimizer=args.minimizer, reg=REG,
                              regmode=REGMODE, regform=REGFORM, regpow=REGPOW,
                              regknee=REGKNEE, moffsets=args.moffsets,
                              svd=args.svd, coll=args.coll, filt=args.filt))

    model = fitmod.build_model(cfg)
    miw = MIWCalib(args.coll, physical_filter=args.filt, repo=args.repo)
    n_v = int(np.load(args.svd)['U_eff'].shape[1])
    vmode_names = [f'v{i+1}' for i in range(n_v)]
    layout = ParamLayout({**cfg, 'moment_offsets': {'moments': moff_list, 'init': 0.0}},
                         vmode_names)
    n_atm = len(layout.atm_free)

    for seq in args.seqs:
        if args.skip_existing and os.path.exists(campn.fit_npz(seq)):
            print(f'  [seq {seq}] skip (exists: {campn.fit_npz(seq)})')
            continue
        fit_one(seq, args, cfg, model, miw, layout, jmax, fit_moments, weights,
                moff_list, plan, n_v, n_atm, vmode_names,
                REG, REGMODE, REGFORM, REGPOW, REGKNEE, reg_w, campn)


def fit_one(seq, args, cfg, model, miw, layout, jmax, fit_moments, weights,
            moff_list, plan, n_v, n_atm, vmode_names,
            REG, REGMODE, REGFORM, REGPOW, REGKNEE, reg_w, campn):
    visit = int(f'{args.day}{seq:05d}')
    rot = rot_for(seq, args.day)
    prep = data_fit.load_and_prep(camp.psfmoments_path(visit),
                                  sign=args.sign, rot_deg=rot)
    binned = data_fit.bin_grid(prep, cell_deg=cfg['fit'].get('cell_deg', 0.10),
                               min_n=cfg['fit'].get('min_n', 3))
    cat = data_fit.to_catalog(binned)
    G_v, _, _ = build_vmode_design(args.svd, cat['thx_deg'], cat['thy_deg'],
                                   jmax, fp_radius=1.75)
    z0 = np.nan_to_num(miw.zernikes(cat['thx_deg'], cat['thy_deg'],
                                    cat['rotator_rad'], jmax, cat['detector']))
    G_reg = None
    if REGMODE == 'wavefront':
        cwo = pd.read_parquet(camp.cwfs_path(visit))
        cwo['corner'] = cwo.detector.str[:3]
        cxs, cys = [], []
        for c in ('R00', 'R04', 'R40', 'R44'):
            sub = cwo[cwo.corner == c]
            if len(sub):
                cxs.append(float(np.median(sub.thx_OCS)) * 180.0 / np.pi)
                cys.append(float(np.median(sub.thy_OCS)) * 180.0 / np.pi)
        G_reg = build_vmode_design(args.svd, np.array(cxs), np.array(cys),
                                   jmax, fp_radius=1.75)[0]
    fwd = Forward(model, layout, z0, G_v, cat['moments'], cat['errors'],
                  fit_moments, weights, reg_lambda=REG, reg_mode=REGMODE,
                  G_reg=G_reg, reg_w=reg_w, reg_form=REGFORM, reg_power=REGPOW,
                  reg_knee=REGKNEE)
    vg = jax.jit(jax.value_and_grad(fwd.cost))

    p0 = np.array(layout.initial(), float)
    base = list(layout.bounds())
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in base])
    hi = np.array([b[1] if b[1] is not None else np.inf for b in base])
    if 'fwhm' in layout.atm_free:
        med = float(np.nanmedian(np.sqrt(np.clip(prep['mom'][:, 0], 0, None)
                                         * np.log(256.0))))
        fw_i = layout.n_dz + layout.atm_free.index('fwhm')
        flo, fhi = layout.atm_bounds['fwhm']
        p0[fw_i] = min(max(med, flo), fhi)
    A_init = np.zeros(layout.n_dz)
    if args.init == 'cwfs':
        A_init = cwfs_vmode_amps(camp.cwfs_path(visit), miw, args.svd,
                                 np.deg2rad(rot), jmax, fp_radius=1.75,
                                 offsets=cfg.get('cwfs', {}).get('offsets', {}))
        p0[layout.i_dz] = A_init
        p0 = np.clip(p0, lo, hi)
    dz_idx = (list(range(*layout.i_dz.indices(len(p0))))
              if isinstance(layout.i_dz, slice) else list(np.atleast_1d(layout.i_dz)))
    atm_idx = [layout.n_dz + i for i in range(n_atm)]

    mon = FitMonitor(label=f'{args.campaign} seq{seq}', verbose=True,
                     checkpoint=campn.fitprog_npz(seq))
    mon.start()
    p = p0.copy()
    stages, stage_evals = [], []
    for st in plan:
        free_v = (set(range(1, n_v + 1)) if st.free_vmodes == 'all'
                  else set(st.free_vmodes))
        b = list(base)
        for k, i in enumerate(dz_idx):
            if (k + 1) not in free_v:
                b[i] = (float(p[i]), float(p[i]))          # freeze at warm-start
        if not st.free_atm:
            for i in atm_idx:
                b[i] = (float(p[i]), float(p[i]))
        res = minimize_stage(vg, p, b, mon, args.minimizer)
        p = np.array(res.x, float)
        stage_evals.append(len(mon.costs))
        chi2_s = float(fwd.chi2(jnp.asarray(p)))
        reg_s = float(fwd.reg(jnp.asarray(p)))
        stages.append(dict(
            name=st.name, A=p[layout.i_dz].copy(),
            atm=np.array([p[layout.n_dz + i] for i in range(n_atm)]),
            offsets=np.array(layout.offset_vector(p)),
            model_mom=np.array(fwd.moments(jnp.asarray(p))),
            chi2=chi2_s, reg=reg_s, cost=chi2_s + reg_s,
            success=bool(res.success),
            mask=np.array([(k + 1) in free_v for k in range(n_v)]),
            res=res))
        print(f'  [seq {seq}] stage {st.name}: free v-modes '
              f'{"all" if st.free_vmodes == "all" else sorted(free_v)}  '
              f'cost={chi2_s + reg_s:.3f} (chi2={chi2_s:.3f}) '
              f'nit={res.nit} ok={res.success}')
    mon.stop()

    S = len(stages)
    st_last = stages[-1]
    mon.plot(campn.fitmon_png(seq), st_last['res'], layout.i_dz, vmode_names,
             atm_idx, layout.atm_free, reg_lambda=REG,
             title=f'{args.day} seq={seq} (rot {rot:.1f}) {args.campaign} '
                   f'plan={args.plan}',
             off_idx=list(range(layout.i_off.start, layout.i_off.stop)),
             off_names=list(layout.offset_moments))
    stt = mon.stats(st_last['res'])
    np.savez(
        campn.fit_npz(seq),
        # shared
        thx=cat['thx_deg'], thy=cat['thy_deg'], detector=cat['detector'], rot=rot,
        data_mom=cat['moments'], data_err=cat['errors'],
        fit_moments=np.array(fit_moments), n_stars=len(prep['thx']),
        n_cells=len(cat['thx_deg']), svd_file=args.svd, init=args.init,
        minimizer=args.minimizer, plan=args.plan, jmax=jmax, A_init=A_init,
        atm_names=np.array(layout.atm_free), offset_moments=np.array(moff_list),
        reg=REG, regmode=REGMODE, regform=REGFORM, regpow=REGPOW, regknee=REGKNEE,
        # per stage
        stage_names=np.array([s['name'] for s in stages]),
        A=np.stack([s['A'] for s in stages]),
        atm=np.stack([s['atm'] for s in stages]),
        offsets=np.stack([s['offsets'] for s in stages]),
        model_mom=np.stack([s['model_mom'] for s in stages]),
        chi2=np.array([s['chi2'] for s in stages]),
        reg_term=np.array([s['reg'] for s in stages]),
        cost=np.array([s['cost'] for s in stages]),
        success=np.array([s['success'] for s in stages]),
        free_vmodes_mask=np.stack([s['mask'] for s in stages]),
        # combined fit-progress trace
        mon_costs=stt['costs'], mon_params=stt['params'],
        mon_iter_evals=stt['iter_evals'], stage_evals=np.array(stage_evals),
        n_stages=S, fit_time_s=stt['time_s'])
    print(f'  [seq {seq}] wrote {campn.fit_npz(seq)}  ({S} stage(s))')


if __name__ == '__main__':
    main()
