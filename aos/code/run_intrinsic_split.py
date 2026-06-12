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
DEFAULT_SPLIT = dict(rotation_sign=1, spin_sign=1, m_max=12,
                     degen_assignment='ocs', ridge=1e-3, hole_dist=0.06,
                     r_min=0.06, r_max=1.75, n_r=80, n_az=180,
                     r_lim_lo=0.1, r_lim_hi=1.6, use_z4_optical=True)


def _map_rms(vals_pol, R, r_lim):
    # Ring-limited RMS over a (..., n_r, n_az) polar field.  R is the 1-D radial
    # array, so the mask applies to the radial axis (-2), not axis 0 — the data
    # field is stacked per rotator bin (n_bins, n_r, n_az).  Delegate to the lib
    # helper, which masks [..., m, :] and ignores NaNs.
    return isp.residual_rms(np.real(vals_pol), R, r_lim)


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

    # field-rotation sense s (fixed; see split.rotation_sign in mi_config.yaml)
    s = int(sp['rotation_sign'])
    print(f's (field rotation) = {s:+d} (fixed)')

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
            dec_by_j[j] = (dec, part, Zc)
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
        dec, part, _ = dec_by_j[j]
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
        dec, part, _ = dec_by_j[j]
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

    # ---- diagnostic plots (RMS summary + per-Zernike O/C/residual maps) ----
    _write_split_pdf(base / 'intrinsic_split.pdf', metrics_df, dec_by_j,
                     noll_list, thetas, labels, X, Y, R, r_lim, s)
    print('[intrinsic_split] done.')


# ----------------------------------------------------------------------
# diagnostic plots (port of intrinsic_camera_telescope_split.ipynb maps)
# ----------------------------------------------------------------------
def _plot_field(ax, X, Y, vals, title, vlim, levels=21, cmap='RdBu_r'):
    """Filled-contour map of a polar-grid field on its (X, Y) points."""
    import numpy as np
    xr, yr, vr = X.ravel(), Y.ravel(), np.asarray(vals).ravel()
    fin = np.isfinite(vr)
    im = ax.tricontourf(xr[fin], yr[fin], vr[fin],
                        levels=np.linspace(-vlim, vlim, levels),
                        cmap=cmap, extend='both')
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)
    ax.set_xlabel('thx [deg]', fontsize=7); ax.set_ylabel('thy [deg]', fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def _abs_vlim(Z, R, r_lim, pct=98):
    import numpy as np
    m = (R >= r_lim[0]) & (R <= r_lim[1])
    return max(float(np.nanpercentile(np.abs(np.asarray(Z)[..., m, :]), pct)), 1e-6)


def _zernike_page(part, dec, Zc, thetas, labels, X, Y, R, r_lim, zlabel):
    """One page per Zernike: data maps at up to 4 representative rotator bins
    (top row) + O (telescope) / C (camera) / model / residual (bottom row).
    Adapts to any number of rotator bins (the notebook assumed <=5)."""
    import numpy as np
    import matplotlib.pyplot as plt
    O = part(dec['O_pol']); C = part(dec['C_pol'])
    data = part(np.asarray(Zc)); resid = part(dec['res'])
    model = data - resid
    vlim = _abs_vlim(data, R, r_lim)
    n_bins = len(labels)
    sel = sorted(set(np.linspace(0, n_bins - 1, min(4, n_bins)).round().astype(int)))
    mid = n_bins // 2
    fig, axs = plt.subplots(2, 4, figsize=(20, 10.5), layout='constrained')
    flat = axs.flat
    for p, i in enumerate(sel):
        rd = _map_rms(data[i], R, r_lim); rr = _map_rms(resid[i], R, r_lim)
        _plot_field(flat[p], X, Y, data[i],
                    f'data {zlabel} {labels[i]} (rot={thetas[i]:+.0f})\n'
                    f'RMS data={rd:.4f}  res={rr:.4f}', vlim)
    for p in range(len(sel), 4):
        flat[p].set_visible(False)
    _plot_field(flat[4], X, Y, O,
                f'O telescope (OCS)\n|O| RMS={_map_rms(O, R, r_lim):.4f}', vlim)
    _plot_field(flat[5], X, Y, C,
                f'C camera (CCS)\n|C| RMS={_map_rms(C, R, r_lim):.4f}', vlim)
    _plot_field(flat[6], X, Y, model[mid],
                f'model {labels[mid]} (data-res)\n'
                f'RMS={_map_rms(model[mid], R, r_lim):.4f}', vlim)
    im = _plot_field(flat[7], X, Y, resid[mid],
                     f'residual {labels[mid]}\nRMS={_map_rms(resid[mid], R, r_lim):.4f}',
                     vlim)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5, label=f'{zlabel} [μm]')
    fig.suptitle(f'{zlabel}: telescope (OCS) + camera (CCS) split   '
                 f'(spin n={dec["n_spin"]}, s={dec["s"]:+d})', fontsize=13)
    return fig


def _write_split_pdf(path, metrics_df, dec_by_j, noll_list, thetas, labels,
                     X, Y, R, r_lim, s):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    n_pages = 0
    with PdfPages(str(path)) as pdf:
        # telescope (O) vs camera (C) map RMS per Zernike
        d = metrics_df
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(16, 5.5), layout='constrained')
        ax.bar(x - 0.21, d['O_tel'], 0.4, label='|O| telescope', color='steelblue')
        ax.bar(x + 0.21, d['C_cam'], 0.4, label='|C| camera', color='indianred')
        ax.plot(x, d['dataRMS'], 'k_', ms=12, label='data RMS')
        ax.plot(x, d['residRMS'], 'kx', ms=6, label='residual RMS')
        ax.set_xticks(x); ax.set_xticklabels(d['Zernike'], rotation=45)
        ax.set_ylabel('map RMS [μm]')
        ax.set_title('Telescope (O) vs camera (C) map RMS per Zernike  '
                     f'(in-window r={r_lim[0]:.1f}..{r_lim[1]:.1f} deg)')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig); n_pages += 1

        for j in noll_list:
            if j not in dec_by_j:
                continue
            dec, part, Zc = dec_by_j[j]
            fig = _zernike_page(part, dec, Zc, thetas, labels, X, Y, R, r_lim,
                                f'Z{j}')
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig); n_pages += 1
    print(f'  wrote intrinsic_split.pdf ({n_pages} pages)')


if __name__ == '__main__':
    main()
