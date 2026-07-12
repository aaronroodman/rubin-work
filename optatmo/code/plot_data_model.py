"""Single multi-page PDF report for the v-Mode OptAtmo fit (one per seq/tag):

  page 1  info: day_obs, seq, visit, Alt/Az/Rot, band, N stars/cells, final cost,
                fit stats, and the options used.
  page 2  FWHM        : data map | model map | model-vs-data scatter.
  page 3  ellipticity : data | model whiskers | e1 & e2 model-vs-data scatters.
  page 4  coma        : data | model whiskers | M21 & M12 scatters.
  page 5  trefoil     : data | model whiskers | M30 & M03 scatters.
  page 6  M22 = <r^4> : data map | model map | scatter (azimuthally-symmetric
                        4th-order moment; NOT M40/M04, which are the spin-4 ones).
  page 7  corners     : per-corner CWFS vs PSF-fit Zj deviation.
  page 8  fit progress: cost / v-modes / atmosphere vs function evaluation.

Writes output/fit_<seq><tag>.pdf.  Model maps/whiskers are recomputed from the
saved fit; the scatters use the fit's binned cells (data_mom vs model_mom).
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages

from config import load_config
import fit as fitmod
import data_fit
from vmode_fit import model_moments_at, wavefront_at
from miw import MIWCalib

DAY = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('day=')), 20260513)
LAB = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03', 'M22', 'M31', 'M13', 'M40', 'M04']
MIW_COLL = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('coll=')),
                'u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
MIW_FILT = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('filt=')), 'i_39')
MIW_REPO = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('repo=')), '/repo/main')
VISITS = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'
FP_R = 1.75
NOLL_CWFS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
RAD2DEG = 180.0 / np.pi


def visit_of(seq):
    return int(f'{DAY}{seq:05d}')


def fwhm_of(e0):
    # e0 = HSM-weighted M11 = trace/2; FWHM = sqrt(e0 * ln256).
    return np.sqrt(np.clip(e0, 0, None) * np.log(256.0))


def visit_info(seq):
    """Alt/Az/rot/band for the triplet (from the FAM-key row = seq-1)."""
    v = pd.read_parquet(VISITS)
    r = v[(v.day_obs == DAY) & (v.seq_num == seq - 1)]
    if not len(r):
        return dict(alt=np.nan, az=np.nan, rot=0.0, band='?')
    r = r.iloc[0]
    return dict(alt=float(r.get('alt', np.nan)), az=float(r.get('az', np.nan)),
                rot=float(r.get('rotator_angle', 0.0)), band=str(r.get('band', '?')))


def _mapaxis(ax, ttl):
    ax.add_patch(Circle((0, 0), FP_R, fill=False, ls='--', color='r', lw=0.7))
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect('equal')
    ax.set_title(ttl, fontsize=9); ax.tick_params(labelsize=6)
    ax.set_xlabel('OCS x [deg]', fontsize=7); ax.set_ylabel('OCS y [deg]', fontsize=7)


def _scatter(ax, d, m, label, unit=''):
    d, m = np.asarray(d), np.asarray(m)
    ax.plot(d, m, '.', ms=3, alpha=0.35, color='C0')
    lo, hi = float(min(d.min(), m.min())), float(max(d.max(), m.max()))
    ax.plot([lo, hi], [lo, hi], 'r-', lw=0.8)
    rms = float(np.sqrt(np.mean((m - d) ** 2)))
    ax.set_xlabel(f'data {label} {unit}'.strip(), fontsize=8)
    ax.set_ylabel(f'model {label} {unit}'.strip(), fontsize=8)
    ax.set_title(f'{label}: model vs data  (rms={rms:.3g}, N={d.size})', fontsize=9)
    ax.set_aspect('equal'); ax.tick_params(labelsize=6)


def _superpix(prep):
    return (prep['detector'].astype(str) + '_'
            + (prep['x'] // 2048).astype(int).astype(str)
            + (prep['y'] // 2048).astype(int).astype(str))


def _angamp(src, ia, ib, kind, ia0):
    if kind == 'spin2':
        a1, a2 = src[:, ia] / src[:, ia0], src[:, ib] / src[:, ia0]
        return 0.5 * np.arctan2(a2, a1), np.hypot(a1, a2)
    return np.arctan2(src[:, ib], src[:, ia]), np.hypot(src[:, ia], src[:, ib])


def _page_scalar(pdf, seq, prep, dcell, mcell, scalarfn, label, unit,
                 model, svd, A, atm, rot, miw, offsets=None):
    """FWHM- / M22-style page: data map | model map | model-vs-data scatter."""
    thx, thy, mom = prep['thx'], prep['thy'], prep['mom']
    g = pd.DataFrame({'sp': _superpix(prep), 'det': prep['detector'],
                      'thx': thx, 'thy': thy, 'val': scalarfn(mom)})
    agg = g.groupby('sp').agg(thx=('thx', 'mean'), thy=('thy', 'mean'),
                              det=('det', 'first'), val=('val', 'median'),
                              n=('val', 'size'))
    agg = agg[agg.n >= 3]
    mm = model_moments_at(model, svd, A, atm, agg.thx.values, agg.thy.values,
                          np.full(len(agg), np.deg2rad(rot)), miw=miw,
                          detector=agg.det.values, offsets=offsets)
    vdat, vmod = agg.val.values, scalarfn(mm)
    vlo, vhi = np.percentile(np.r_[vdat, vmod], [2, 98])
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for a, val, ttl in [(ax[0, 0], vdat, 'DATA'), (ax[0, 1], vmod, 'MODEL')]:
        _mapaxis(a, f'{ttl} {label} [{unit}] (2x2/CCD super-pixels)')
        s = a.scatter(agg.thx, agg.thy, c=val, s=45, marker='s', cmap='viridis',
                      vmin=vlo, vmax=vhi)
        fig.colorbar(s, ax=a, shrink=0.8)
    _scatter(ax[1, 0], scalarfn(dcell), scalarfn(mcell), label, unit)
    ax[1, 1].axis('off')
    fig.suptitle(f'{DAY} seq={seq}: {label}', fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_doublet(pdf, seq, prep, dcell, mcell, keypair, kind, ref_len, title,
                  model, svd, A, atm, rot, miw, nstar=1000, offsets=None):
    thx, thy, mom = prep['thx'], prep['thy'], prep['mom']
    ia0 = LAB.index('e0'); ia, ib = LAB.index(keypair[0]), LAB.index(keypair[1])
    idx = np.linspace(0, len(thx) - 1, min(nstar, len(thx))).astype(int)
    mm = model_moments_at(model, svd, A, atm, thx[idx], thy[idx],
                          np.full(len(idx), np.deg2rad(rot)), miw=miw,
                          detector=prep['detector'][idx], offsets=offsets)
    amp95 = np.percentile(_angamp(mom[idx], ia, ib, kind, ia0)[1], 95)
    scale = max(amp95, 1e-9) / ref_len
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for a, src, ttl in [(ax[0, 0], mom[idx], 'DATA'), (ax[0, 1], mm, 'MODEL')]:
        _mapaxis(a, f'{ttl} {title} (95%-whisker={ref_len:g} deg)')
        ang, amp = _angamp(src, ia, ib, kind, ia0)
        a.quiver(thx[idx], thy[idx], amp * np.cos(ang), amp * np.sin(ang),
                 angles='xy', scale_units='xy', scale=scale, pivot='mid',
                 headlength=0, headaxislength=0, width=0.0035)
    _scatter(ax[1, 0], dcell[:, ia], mcell[:, ia], keypair[0], 'arcsec^n')
    _scatter(ax[1, 1], dcell[:, ib], mcell[:, ib], keypair[1], 'arcsec^n')
    fig.suptitle(f'{DAY} seq={seq}: {title}', fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_corners(pdf, seq, fit, A, svd, rot, miw, cfg):
    cw = pd.read_parquet(f'data/cwfs_{visit_of(seq)}.parquet')
    cw['corner'] = cw.detector.str[:3]
    from lsst.obs.lsst import LsstCam
    name2id = {d.getName(): d.getId() for d in LsstCam.getCamera()}
    offsets = cfg.get('cwfs', {}).get('offsets', {})
    njz = min(12, len(NOLL_CWFS))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for c, axc in zip(['R00', 'R04', 'R40', 'R44'], axes.flat):
        sub = cw[cw.corner == c]
        if len(sub) == 0:
            axc.set_visible(False); continue
        cx = np.median(sub.thx_OCS.values) * RAD2DEG
        cy = np.median(sub.thy_OCS.values) * RAD2DEG
        det_id = name2id.get(sub.detector.iloc[0], -1)
        miwc = np.nan_to_num(miw.zernikes(cx, cy, np.deg2rad(rot), 22, [det_id])[0])
        cwfs_z = np.array([np.median(sub[f'ztot_{i}'].values) - miwc[NOLL_CWFS[i]]
                           - float(offsets.get(NOLL_CWFS[i], 0.0)) for i in range(njz)])
        psf_dev = wavefront_at(A, svd, [cx], [cy], jmax=22, fp_radius=FP_R)[0]
        psf_z = np.array([psf_dev[NOLL_CWFS[i]] if NOLL_CWFS[i] <= 22 else np.nan
                          for i in range(njz)])
        w = 0.4; xloc = np.arange(njz)
        axc.bar(xloc - w / 2, cwfs_z, w, label='CWFS', color='C0')
        axc.bar(xloc + w / 2, psf_z, w, label='PSF v-mode fit', color='C3')
        axc.set_xticks(xloc)
        axc.set_xticklabels([f'Z{NOLL_CWFS[i]}' for i in range(njz)], fontsize=7)
        axc.set_title(f'corner {c}  (x={cx:.2f}, y={cy:.2f})', fontsize=9)
        axc.axhline(0, color='k', lw=0.6); axc.legend(fontsize=7)
        axc.set_ylabel('Zj deviation [µm]', fontsize=8); axc.tick_params(labelsize=6)
    fig.suptitle(f'{DAY} seq={seq}: corner-WFS vs PSF v-mode fit '
                 f'(deviation rel. official MIW, CWFS offset-corrected)', fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_progress(pdf, seq, fit):
    costs = np.asarray(fit['mon_costs'], float)
    P = np.asarray(fit['mon_params'], float)
    ie = np.asarray(fit['mon_iter_evals'], int)
    reg = float(fit['reg']); n_v = len(np.asarray(fit['A']))
    atm_names = [str(x) for x in np.atleast_1d(fit['atm_names'])]
    ev = np.arange(1, len(costs) + 1)
    dz = P[:, :n_v]
    chi2 = costs - reg * np.sum(dz ** 2, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for e in ie:
        for a in ax:
            a.axvline(e, color='0.88', lw=0.5, zorder=0)
    ax[0].semilogy(ev, np.clip(costs, 1e-15, None), '-', color='C0',
                   label='objective (χ²ᵣ+reg)')
    if reg > 0:
        ax[0].semilogy(ev, np.clip(chi2, 1e-15, None), '--', color='C1',
                       label='reduced χ² (data)')
    ax[0].set_xlabel('function evaluation'); ax[0].set_ylabel('cost')
    ax[0].set_title('cost vs evaluation'); ax[0].legend(fontsize=7)
    for i in range(n_v):
        ax[1].plot(ev, dz[:, i], lw=0.7)
    ax[1].set_xlabel('function evaluation'); ax[1].set_ylabel('amplitude')
    ax[1].set_title(f'v-mode amplitudes ({n_v} modes)')
    for j, nm in enumerate(atm_names):
        ax[2].plot(ev, P[:, n_v + j], lw=1.3, label=nm)
    ax[2].set_xlabel('function evaluation'); ax[2].set_title('atmosphere params')
    ax[2].legend(fontsize=8)
    fig.suptitle(f'{DAY} seq={seq}: fit progress', fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_info(pdf, seq, fit, info, cfg):
    atm = np.asarray(fit['atm']); atm_names = [str(x) for x in np.atleast_1d(fit['atm_names'])]
    lines = [
        f'OptAtmo v-Mode fit report', '',
        f'day_obs        : {DAY}',
        f'seq_num        : {seq}      visit: {visit_of(seq)}',
        f'band           : {info["band"]}',
        f'Alt / Az       : {info["alt"]:.2f} / {info["az"]:.2f}  deg',
        f'Rotator        : {float(fit["rot"]):.3f}  deg',
        f'N stars        : {int(fit["n_stars"])}      N binned cells: {int(fit["n_cells"])}',
        '',
        f'final cost     : {float(fit["cost"]):.5g}   (reduced χ² + Tikhonov reg)',
        f'success        : {bool(fit["success"])}',
        f'nfev / njev    : {int(fit["nfev"])} / {int(fit["njev"])}     nit: {int(fit["nit"])}',
        f'fit wall time  : {float(fit["fit_time_s"]):.1f}  s',
        '',
        f'--- options ---',
        f'v-mode init    : {str(fit["init"])}',
        f'optics         : {str(fit["optics"])}  ({"frozen at init" if str(fit["optics"])=="fixed" else "free"})',
        f'Tikhonov reg λ : {float(fit["reg"]):g}',
        f'SVD basis      : {str(fit["svd_file"])}   ({len(np.asarray(fit["A"]))} v-modes)',
        f'fit moments    : {", ".join(str(m) for m in np.atleast_1d(fit["fit_moments"]))}',
        f'atmosphere     : kernel={cfg["atmosphere"]["kernel"]}, L0={cfg["atmosphere"]["L0"]}, '
        f'free={atm_names}',
        f'moment offsets : {[str(m) for m in np.atleast_1d(fit["offset_moments"])] if "offset_moments" in fit.files and np.atleast_1d(fit["offset_moments"]).size else "none"}',
        f'bin cell_deg   : {cfg["fit"].get("cell_deg", 0.1)}   min_n: {cfg["fit"].get("min_n", 3)}',
        f'PSF model      : jmax={cfg["geometry"]["jmax"]}, annular={cfg["geometry"].get("annular")}, '
        f'stamp={cfg["geometry"]["stamp"]}, oversample={cfg["geometry"]["oversample"]}',
        '',
        f'--- fitted atmosphere ---',
        '   ' + '   '.join(f'{n}={v:.4f}' for n, v in zip(atm_names, atm)),
    ]
    if 'offset_moments' in fit.files and np.atleast_1d(fit['offset_moments']).size:
        offv = np.asarray(fit['offsets'])
        oms = [str(m) for m in np.atleast_1d(fit['offset_moments'])]
        lines += ['', f'--- fitted moment offsets ---',
                  '   ' + '   '.join(f'{m}={offv[LAB.index(m)]:.5f}' for m in oms)]
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, '\n'.join(lines), va='top', ha='left',
             family='monospace', fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def run(seq, svd_npz, cfg, model, miw, tag=''):
    fit = np.load(f'data/vmodefit_{seq}{tag}.npz', allow_pickle=False)
    A, atm, rot = fit['A'], fit['atm'], float(fit['rot'])
    dcell, mcell = fit['data_mom'], fit['model_mom']       # binned cells (n,12)
    offsets = fit['offsets'] if 'offsets' in fit.files else None
    info = visit_info(seq)
    prep = data_fit.load_and_prep(f'data/psfmoments_{visit_of(seq)}.parquet',
                                  sign=1, rot_deg=rot)
    out = f'output/fit_{seq}{tag}.pdf'
    with PdfPages(out) as pdf:
        _page_info(pdf, seq, fit, info, cfg)
        _page_scalar(pdf, seq, prep, dcell, mcell, lambda m: fwhm_of(m[:, 0]),
                     'FWHM', 'arcsec', model, svd_npz, A, atm, rot, miw, offsets)
        _page_doublet(pdf, seq, prep, dcell, mcell, ('e1', 'e2'), 'spin2', 0.30,
                      'ellipticity', model, svd_npz, A, atm, rot, miw,
                      nstar=1250, offsets=offsets)
        _page_doublet(pdf, seq, prep, dcell, mcell, ('M21', 'M12'), 'spin1', 0.30,
                      'coma', model, svd_npz, A, atm, rot, miw,
                      nstar=750, offsets=offsets)
        _page_doublet(pdf, seq, prep, dcell, mcell, ('M30', 'M03'), 'spin1', 0.30,
                      'trefoil', model, svd_npz, A, atm, rot, miw,
                      nstar=750, offsets=offsets)
        _page_scalar(pdf, seq, prep, dcell, mcell, lambda m: m[:, LAB.index('M22')],
                     'M22 = <r^4>', 'arcsec^4', model, svd_npz, A, atm, rot, miw,
                     offsets)
        _page_corners(pdf, seq, fit, A, svd_npz, rot, miw, cfg)
        _page_progress(pdf, seq, fit)
    print(f'seq {seq}: wrote {out}')


def main():
    svd_npz = next((a for a in sys.argv[1:] if a.endswith('.npz')),
                   'data/ofc_svd_22_12.npz')
    seqs = next((a.split('=')[1] for a in sys.argv if a.startswith('seqs=')), None)
    seqs = [int(s) for s in seqs.split(',')] if seqs else [25, 28]
    init = next((a.split('=')[1] for a in sys.argv if a.startswith('init=')), 'zero')
    optics = next((a.split('=')[1] for a in sys.argv if a.startswith('optics=')), 'free')
    moff = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('moffsets=')), 'off')
    _p = (([init] if init != 'zero' else [])
          + (['atmonly'] if optics == 'fixed' else [])
          + (['moff'] if moff not in ('off', 'none', '') else []))
    tag = ('_' + '_'.join(_p)) if _p else ''
    cfg = load_config('config.yaml')
    cfg['geometry']['stamp'] = 24; cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'VonKarman'
    model = fitmod.build_model(cfg)
    miw = MIWCalib(MIW_COLL, physical_filter=MIW_FILT, repo=MIW_REPO)
    for seq in seqs:
        run(seq, svd_npz, cfg, model, miw, tag=tag)


if __name__ == '__main__':
    main()
