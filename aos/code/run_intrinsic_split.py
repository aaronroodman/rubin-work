#!/usr/bin/env python3
"""Decompose the per-rotator-bin measured intrinsic into OCS (telescope-fixed)
and CCS (camera-fixed) components — Phase 2 step 2.

Script port of the decomposition core of intrinsic_camera_telescope_split.ipynb
(which stays for interactive diagnostics).  Reads the per-rotator-bin grids
written by run_build_intrinsic.py, runs the spin-aware OCS/CCS decomposition
(intrinsic_split.py) per Zernike group, and writes a single

    output/<param_set>/<mi_name>/intrinsic_split.parquet

with O_Z{j} (OCS) and C_Z{j} (CCS) sampled at the rot~0 field points, plus a
…_rms.csv of the per-Zernike telescope/camera/residual RMS.  RSP-only
(numpy/scipy + intrinsic_split).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import intrinsic_split as isp
import mi_config as mc

DEFAULT_NOLL = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                22, 23, 24, 25, 26]
DEFAULT_SPLIT = dict(rotation_sign='auto', spin_sign=1, m_max=12,
                     degen_assignment='ocs', ridge=1e-3, hole_dist=0.06,
                     r_min=0.06, r_max=1.75, n_r=80, n_az=180,
                     r_lim_lo=0.1, r_lim_hi=1.6, use_z4_optical=True)


def _map_rms(vals_pol, R, r_lim):
    m = (R >= r_lim[0]) & (R <= r_lim[1])
    v = np.real(vals_pol)[m]
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    sp = {**DEFAULT_SPLIT, **(cfg.get('split') or {})}
    noll_list = cfg.get('noll_list') or DEFAULT_NOLL
    rot_bins = mc.rotator_bins(cfg)
    alt_range = (cfg.get('alt_min_deg'), cfg.get('alt_max_deg'))

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.output_root) / args.param_set / 'fits.parquet'
    fits_table = (pd.read_parquet(fits_path,
                                  columns=['rotator_angle', 'alt', 'day_obs', 'seq_num'])
                  if fits_path.exists() else None)

    # ---- load each rotator-bin grid + its mean rotator angle ----
    dsets, thetas, labels = [], [], []
    for lo, hi in rot_bins:
        gp = base / 'build' / f'rot_{lo:g}_{hi:g}' / 'intrinsic_grid.parquet'
        if not gp.exists():
            raise FileNotFoundError(gp)
        df = pd.read_parquet(gp)
        zk = np.vstack(df['zk'].values)
        nolls = [int(j) for j in np.asarray(df['nollIndices'][0]).tolist()]
        rec = {'thx': df['thx_deg'].values, 'thy': df['thy_deg'].values}
        if 'z4_optical_OCS' in df.columns:
            rec['z4opt'] = df['z4_optical_OCS'].values
        for j in noll_list:
            rec[f'z{j}'] = zk[:, nolls.index(j)]
        dsets.append(rec)
        center = 0.5 * (lo + hi)
        if fits_table is not None:
            mr, *_ = isp.mean_rotator(fits_table, (lo, hi), alt_range)
            theta = mr if np.isfinite(mr) else center
        else:
            theta = center
        thetas.append(theta)
        labels.append(f'rot{theta:+.0f}')
    thetas = np.array(thetas)
    th_rad = np.deg2rad(thetas)
    print(f'[intrinsic_split] {args.param_set}/{args.mi_name}: '
          f'{len(dsets)} rotator bins, thetas={np.round(thetas,2).tolist()}')

    R, A, X, Y = isp.make_polar_grid(r_min=sp['r_min'], r_max=sp['r_max'],
                                     n_r=sp['n_r'], n_az=sp['n_az'])
    r_lim = (sp['r_lim_lo'], sp['r_lim_hi'])

    def sample_field(key):
        maps = []
        for d in dsets:
            v = d[key]
            f = np.isfinite(v)
            maps.append((d['thx'][f], d['thy'][f], v[f]))
        Z, valid, _ = isp.sample_maps_polar(maps, X, Y, hole_dist=sp['hole_dist'])
        return np.nan_to_num(Z), valid

    # field-rotation sense s
    if sp['rotation_sign'] == 'auto':
        _Z, _V = sample_field('z4opt' if (sp['use_z4_optical'] and 'z4opt' in dsets[0])
                              else 'z4')
        _r, _rms = isp.decompose_auto_sign(_Z, th_rad, A, R, r_lim=r_lim, valid=_V,
                                           method='lsq', m_max=sp['m_max'],
                                           ridge=sp['ridge'])
        s = _r['s']
        print(f's (field rotation) = {s:+d}  (Z4 auto: +1 {_rms[1]:.4f}, '
              f'-1 {_rms[-1]:.4f})')
    else:
        s = int(sp['rotation_sign'])
        print(f's = {s:+d} (fixed)')

    # ---- per-group decomposition ----
    groups = isp.group_zernikes(noll_list)
    dec_by_j, metrics = {}, []
    for grp in groups:
        n_spin = sp['spin_sign'] * grp['spin']
        if grp['kind'] == 'single':
            key = ('z4opt' if (grp['j'] == 4 and sp['use_z4_optical']
                               and 'z4opt' in dsets[0]) else f"z{grp['j']}")
            Z, V = sample_field(key)
            dec = isp.decompose_spin_lsq(Z.astype(complex), V, th_rad, A, n_spin=0,
                                         s=s, m_max=sp['m_max'], ridge=sp['ridge'],
                                         degen_assignment=sp['degen_assignment'])
            comps = [(grp['j'], np.real)]
            Zc = Z.astype(complex)
        elif grp['kind'] == 'pair':
            Zc_, Vc_ = sample_field(f"z{grp['j_cos']}")
            Zs_, Vs_ = sample_field(f"z{grp['j_sin']}")
            Zc = Zc_ + 1j * Zs_
            V = Vc_ & Vs_
            dec = isp.decompose_spin_lsq(Zc, V, th_rad, A, n_spin=n_spin, s=s,
                                         m_max=sp['m_max'], ridge=sp['ridge'],
                                         degen_assignment=sp['degen_assignment'])
            comps = [(grp['j_cos'], np.real), (grp['j_sin'], np.imag)]
        else:
            print(f"  (skipping unpaired {grp['label']})")
            continue
        for j, part in comps:
            dec_by_j[j] = (dec, part)
            metrics.append(dict(
                Zernike=f'Z{j}', j=j, spin=abs(dec['n_spin']),
                dataRMS=_map_rms(part(Zc), R, r_lim),
                O_tel=_map_rms(part(dec['O_pol']), R, r_lim),
                C_cam=_map_rms(part(dec['C_pol']), R, r_lim),
                residRMS=_map_rms(part(dec['res']), R, r_lim)))

    metrics_df = pd.DataFrame(metrics).sort_values('j').reset_index(drop=True)
    print(f"  overall |O| tel RMS={np.sqrt((metrics_df['O_tel']**2).mean()):.4f}  "
          f"|C| cam RMS={np.sqrt((metrics_df['C_cam']**2).mean()):.4f} um")

    # ---- resample O / C onto the rot~0 grid; write outputs ----
    i0 = int(np.argmin(np.abs(thetas)))
    thx0, thy0 = dsets[i0]['thx'], dsets[i0]['thy']
    out = {'thx_deg': thx0, 'thy_deg': thy0}
    for j in noll_list:
        if j not in dec_by_j:
            continue
        dec, part = dec_by_j[j]
        out[f'O_Z{j}'] = isp.polar_field_to_points(part(dec['O_pol']), X, Y, thx0, thy0)
        out[f'C_Z{j}'] = isp.polar_field_to_points(part(dec['C_pol']), X, Y, thx0, thy0)
    df_out = pd.DataFrame(out)
    df_out.attrs['rotation_sign'] = int(s)
    df_out.attrs['degen_assignment'] = sp['degen_assignment']
    df_out.attrs['thetas_deg'] = list(np.round(thetas, 3))
    base.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(base / 'intrinsic_split.parquet')
    metrics_df.to_csv(base / 'intrinsic_split_rms.csv', index=False)
    print(f'  wrote intrinsic_split.parquet ({len(df_out)} rows x '
          f'{len(df_out.columns)} cols) + intrinsic_split_rms.csv')

    # ---- persist the complex polar decomposition for per-donut reconstruction
    # (step 3, run_make_intrinsic_sidecar.py).  One (O_pol, C_pol, n_spin, s,
    # part) record per pupil Noll j; pairs duplicate the shared group dec, which
    # is cheap (n_r x n_az complex).  `part` = 0 (real) / 1 (imag).
    jvals, Op, Cp, nsp, ss, prt = [], [], [], [], [], []
    for j in noll_list:
        if j not in dec_by_j:
            continue
        dec, part = dec_by_j[j]
        jvals.append(int(j))
        Op.append(np.asarray(dec['O_pol']))
        Cp.append(np.asarray(dec['C_pol']))
        nsp.append(int(dec.get('n_spin', 0)))
        ss.append(int(dec['s']))
        prt.append(0 if part is np.real else 1)
    np.savez(base / 'intrinsic_split_decomp.npz',
             A=A, X=X, Y=Y, jvals=np.array(jvals, dtype=int),
             O_pol=np.array(Op), C_pol=np.array(Cp),
             n_spin=np.array(nsp, dtype=int), s=np.array(ss, dtype=int),
             part=np.array(prt, dtype=int))
    print(f'  wrote intrinsic_split_decomp.npz ({len(jvals)} Zernikes) '
          f'for per-donut reconstruction')


if __name__ == '__main__':
    main()
