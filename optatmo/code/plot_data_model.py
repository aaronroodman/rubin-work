"""Data-vs-model validation plots for the v-Mode fit:
  1. FWHM: 2x2-super-pixel-per-CCD colored boxes (median), data | model.
  2. Ellipticity whiskers (dense, ~5x stars), data | model.
  3. Coma & trefoil whiskers/markers (~3x stars), data | model.
  4. Corner bar chart: per-corner Zj deviation, CWFS vs PSF v-mode-fit model.

Usage: python plot_data_model.py <svd_npz>   (defaults to k3 file)
Model moments are recomputed at the plotted positions from the saved fit.
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import load_config
import fit as fitmod
import data_fit, frames
from vmode_fit import model_moments_at, wavefront_at
from miw import MIWCalib

DAY = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('day=')),
           20260513)
LAB = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03', 'M22', 'M31', 'M13', 'M40', 'M04']
MIW_COLL = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('coll=')),
                'u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
MIW_FILT = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('filt=')),
                'i_39')
MIW_REPO = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('repo=')),
                '/repo/main')


def visit_of(seq):
    return int(f'{DAY}{seq:05d}')
VISITS = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'
FP_R = 1.75
NOLL_CWFS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
RAD2DEG = 180.0 / np.pi


def fwhm_of(e0):
    # e0 = HSM-weighted M11 = trace(M)/2 (self-consistent adaptive Gaussian);
    # RubinTV T = trace(M) = 2 e0, FWHM = sqrt(T/2 * ln256) = sqrt(e0 * ln256).
    return np.sqrt(np.clip(e0, 0, None) * np.log(256.0))


def rot_for(seq):
    v = pd.read_parquet(VISITS)
    r = v[(v.day_obs == DAY) & (v.seq_num == seq - 1)]
    return float(r.rotator_angle.iloc[0]) if len(r) else 0.0


def _mapaxis(ax, ttl):
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='r', lw=0.7))
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect('equal')
    ax.set_title(ttl, fontsize=9); ax.tick_params(labelsize=6)
    ax.set_xlabel('OCS x [deg]', fontsize=7); ax.set_ylabel('OCS y [deg]', fontsize=7)


def run(seq, svd_npz, cfg, model, miw, tag=''):
    rot = rot_for(seq)
    # cleaned (MAD-clipped), OCS-rotated stars — same sample as the fit
    prep = data_fit.load_and_prep(f'data/psfmoments_{visit_of(seq)}.parquet',
                                  sign=1, rot_deg=rot)
    thx, thy, mom_ocs = prep['thx'], prep['thy'], prep['mom']
    fit = np.load(f'data/vmodefit_{seq}{tag}.npz')
    A, atm = fit['A'], fit['atm']

    # ---------- 1. FWHM super-pixels (2x2 per CCD) ----------
    sp = (prep['detector'].astype(str) + '_'
          + (prep['x'] // 2048).astype(int).astype(str)
          + (prep['y'] // 2048).astype(int).astype(str))
    g = pd.DataFrame({'sp': sp, 'det': prep['detector'], 'thx': thx, 'thy': thy,
                      'e0': mom_ocs[:, 0]})
    agg = g.groupby('sp').agg(thx=('thx', 'mean'), thy=('thy', 'mean'),
                              det=('det', 'first'),
                              e0=('e0', 'median'), n=('e0', 'size'))
    agg = agg[agg.n >= 3]
    mmom = model_moments_at(model, svd_npz, A, atm, agg.thx.values, agg.thy.values,
                            np.full(len(agg), np.deg2rad(rot)), miw=miw,
                            detector=agg.det.values)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    vlo, vhi = np.percentile(np.r_[fwhm_of(agg.e0.values), fwhm_of(mmom[:, 0])], [2, 98])
    for a, val, tag in [(ax[0], fwhm_of(agg.e0.values), 'DATA'),
                        (ax[1], fwhm_of(mmom[:, 0]), 'MODEL')]:
        _mapaxis(a, f'{tag} FWHM [arcsec] (2x2/CCD super-pixels)')
        s = a.scatter(agg.thx, agg.thy, c=val, s=55, marker='s', cmap='viridis',
                      vmin=vlo, vmax=vhi)
        fig.colorbar(s, ax=a, shrink=0.8)
    fig.suptitle(f'20260513 seq={seq}: FWHM data vs model'); fig.tight_layout()
    fig.savefig(f'output/dm_fwhm_{seq}{tag}.png', dpi=115, bbox_inches='tight'); plt.close(fig)

    # ---------- 2 & 3. whiskers (dense) ----------
    ia0 = LAB.index('e0')

    def angamp(src, ia, ib, kind):
        if kind == 'spin2':            # ellipticity: normalised e1/e0, e2/e0, half-angle
            a1, a2 = src[:, ia] / src[:, ia0], src[:, ib] / src[:, ia0]
            return 0.5 * np.arctan2(a2, a1), np.hypot(a1, a2)
        return np.arctan2(src[:, ib], src[:, ia]), np.hypot(src[:, ia], src[:, ib])

    def whisker_fig(nstar, keypair, kind, fname, ref_len, title):
        idx = np.linspace(0, len(thx) - 1, min(nstar, len(thx))).astype(int)
        mm = model_moments_at(model, svd_npz, A, atm, thx[idx], thy[idx],
                              np.full(len(idx), np.deg2rad(rot)), miw=miw,
                              detector=prep['detector'][idx])
        ia, ib = LAB.index(keypair[0]), LAB.index(keypair[1])
        # scale from the SAME quantity that is plotted: 95th-pct whisker -> ref_len deg
        amp95 = np.percentile(angamp(mom_ocs[idx], ia, ib, kind)[1], 95)
        scale = max(amp95, 1e-9) / ref_len
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        for a, src, tag in [(ax[0], mom_ocs[idx], 'DATA'), (ax[1], mm, 'MODEL')]:
            _mapaxis(a, f'{tag} {title}')
            ang, amp = angamp(src, ia, ib, kind)
            a.quiver(thx[idx], thy[idx], amp * np.cos(ang), amp * np.sin(ang),
                     angles='xy', scale_units='xy', scale=scale, pivot='mid',
                     headlength=0, headaxislength=0, width=0.0035)
        fig.suptitle(f'20260513 seq={seq}: {title} data vs model '
                     f'({len(idx)} stars, 95%-whisker={ref_len:g} deg)')
        fig.tight_layout(); fig.savefig(fname, dpi=115, bbox_inches='tight'); plt.close(fig)

    whisker_fig(1250, ('e1', 'e2'), 'spin2', f'output/dm_ellip_{seq}{tag}.png', 0.30, 'ellipticity')
    whisker_fig(750, ('M21', 'M12'), 'spin1', f'output/dm_coma_{seq}{tag}.png', 0.30, 'coma')
    whisker_fig(750, ('M30', 'M03'), 'spin1', f'output/dm_trefoil_{seq}{tag}.png', 0.30, 'trefoil')
    print(f'seq {seq}: wrote FWHM/ellip/coma/trefoil data-vs-model plots')

    # ---------- 4. corner bar chart: CWFS vs PSF-model deviation ----------
    cw = pd.read_parquet(f'data/cwfs_{visit_of(seq)}.parquet')
    cw['corner'] = cw.detector.str[:3]
    from lsst.obs.lsst import LsstCam
    name2id = {d.getName(): d.getId() for d in LsstCam.getCamera()}
    corners = ['R00', 'R04', 'R40', 'R44']
    njz = min(12, len(NOLL_CWFS))            # compare Noll 4..15
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for c, axc in zip(corners, axes.flat):
        sub = cw[cw.corner == c]
        if len(sub) == 0:
            axc.set_visible(False); continue
        cx = np.median(sub.thx_OCS.values) * RAD2DEG      # rad -> deg
        cy = np.median(sub.thy_OCS.values) * RAD2DEG
        # CWFS deviation relative to the SAME official MIW the PSF fit uses:
        #   dev = total OPD (zk_OCS = ztot) - MIW_official(corner sensor, rotator)
        det_id = name2id.get(sub.detector.iloc[0], -1)
        miwc = np.nan_to_num(
            miw.zernikes(cx, cy, np.deg2rad(rot), 22, [det_id])[0])
        cwfs_z = np.array([np.median(sub[f'ztot_{i}'].values) - miwc[NOLL_CWFS[i]]
                           for i in range(njz)])
        psf_dev = wavefront_at(A, svd_npz, [cx], [cy], jmax=22, fp_radius=FP_R)[0]
        psf_z = np.array([psf_dev[NOLL_CWFS[i]] if NOLL_CWFS[i] <= 22 else np.nan
                          for i in range(njz)])
        nolls = [NOLL_CWFS[i] for i in range(njz)]
        w = 0.4; xloc = np.arange(njz)
        axc.bar(xloc - w / 2, cwfs_z, w, label='CWFS', color='C0')
        axc.bar(xloc + w / 2, psf_z, w, label='PSF v-mode fit', color='C3')
        axc.set_xticks(xloc); axc.set_xticklabels([f'Z{n}' for n in nolls], fontsize=7)
        axc.set_title(f'corner {c}  (x={cx:.2f}, y={cy:.2f} deg)', fontsize=9)
        axc.axhline(0, color='k', lw=0.6); axc.legend(fontsize=7)
        axc.set_ylabel('Zj deviation [µm]', fontsize=8)
    fig.suptitle(f'20260513 seq={seq}: corner-WFS vs PSF v-mode-fit Zj '
                 f'(deviation rel. official MIW)')
    fig.tight_layout(); fig.savefig(f'output/dm_corners_{seq}{tag}.png', dpi=115,
                                    bbox_inches='tight'); plt.close(fig)
    print(f'seq {seq}: wrote corner bar chart')


def main():
    svd_npz = next((a for a in sys.argv[1:] if a.endswith('.npz')),
                   'data/ofc_svd_22_12.npz')
    seqs = next((a.split('=')[1] for a in sys.argv if a.startswith('seqs=')), None)
    seqs = [int(s) for s in seqs.split(',')] if seqs else [25, 28]
    init = next((a.split('=')[1] for a in sys.argv if a.startswith('init=')), 'zero')
    tag = '' if init == 'zero' else f'_{init}'    # match run_vmode_fit output tag
    cfg = load_config('config.yaml')
    cfg['geometry']['stamp'] = 24; cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'VonKarman'
    model = fitmod.build_model(cfg)
    miw = MIWCalib(MIW_COLL, physical_filter=MIW_FILT, repo=MIW_REPO)
    for seq in seqs:
        run(seq, svd_npz, cfg, model, miw, tag=tag)


if __name__ == '__main__':
    main()
