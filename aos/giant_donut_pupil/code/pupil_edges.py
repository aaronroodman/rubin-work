"""Donut inner and outer edges against azimuth (step 3 of the pupil pass), and the pass's PDF.

Edges are measured on four images per donut side, all about one centre (the
step 1 v4.02 fit centre): the data, the v4.02 and v1000 Z4 fit models
(`z4_fits.py`) and the v4.02 Batoid image (`batoid_images.py`).

Per 2 deg azimuth bin: the radial profile at 0.5 pixel steps, the mean of the
pixels in that bin and radial step. The provisional inner and outer edges are
0.62 r_out and r_out; the flat-top level is the median of the profile between
them. The edges are the outermost and innermost 50 % crossings of the flat top,
interpolated linearly. A bin is flagged ``spider`` when a v4.02 Spider_3D vane
shadow at that rtp crosses it (from danish images with and without the
vanes), ``vignetted`` when the Batoid flat top is below 0.3 of its largest
bin value (the ring is cut away there), and ``masked`` when a neighbour mask
covers the crossing; flagged bins are marked in the plots and kept out of the
edge statistics. Crossings are searched for between 0.4 and 1.15 r_out only,
so a neighbour beyond the edge is not taken as the edge.

Azimuth is in OCS: the CCS stamp angle plus rtp, deg.

Unmodelled surfaces: runs of 3 or more adjacent unflagged bins with
|data - Batoid| > 3 pixels. For each run the clipping surface is the one that
first vignettes the rays `batoid_images.py` traced with the full v4.02 model
that land within 3 pixels outside the Batoid edge in that azimuth range: the
surface that sets the model edge there.

Run, from ``code/``::

    python pupil_edges.py --run ../output/20261018_pupil
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

NBIN = 180
DR_PIX = 0.5
PIXEL_MM = 0.010
OUTLIER_PIX = 3.0
OUTLIER_RUN = 3
EDGE_BAND_PIX = 3.0
DARK_FRAC = 0.3
SEARCH_PIX = (0.4, 1.15)
PUPILS = ('v1000', 'v4.02')
CURVES = {'data': dict(color='k', ls='-'), 'v4.02 model': dict(color='tab:red', ls='-'),
          'Batoid': dict(color='tab:green', ls='-'), 'v1000 model': dict(color='tab:blue', ls='--')}


def edges(image, valid, cx, cy, r_out):
    """Inner and outer 50 % edges per azimuth bin (pixels), CCS azimuth.

    Returns
    -------
    r_in, r_outer : `numpy.ndarray`
        Edge radii, pixels; NaN where no crossing.
    masked_in, masked_out : `numpy.ndarray`
        True where an invalid pixel lies within 1 pixel of the crossing.
    """
    n = image.shape[0]
    yy, xx = np.mgrid[:n, :n]
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    phi = np.mod(np.degrees(np.arctan2(dy, dx)), 360.0)
    rmax = 1.3 * r_out
    sel = r < rmax
    kb = np.minimum((phi[sel] / (360 / NBIN)).astype(int), NBIN - 1)
    kr = (r[sel] / DR_PIX).astype(int)
    nr = int(rmax / DR_PIX) + 1
    flat = kb * nr + kr
    good = valid[sel]
    tot = np.bincount(flat[good], image[sel][good], NBIN * nr).reshape(NBIN, nr)
    cnt = np.bincount(flat[good], minlength=NBIN * nr).reshape(NBIN, nr)
    bad = np.bincount(flat[~good], minlength=NBIN * nr).reshape(NBIN, nr)
    with np.errstate(invalid='ignore'):
        prof = tot / cnt
    rad = (np.arange(nr) + 0.5) * DR_PIX
    r_in = np.full(NBIN, np.nan)
    r_ot = np.full(NBIN, np.nan)
    m_in = np.zeros(NBIN, bool)
    m_ot = np.zeros(NBIN, bool)
    tops = np.full(NBIN, np.nan)
    flat_win = (rad > 0.62 * r_out) & (rad < r_out)
    for b in range(NBIN):
        p = prof[b]
        ok = np.isfinite(p)
        if ok.sum() < 20:
            continue
        p = np.interp(rad, rad[ok], p[ok])
        top = np.median(p[flat_win & ok])
        tops[b] = top
        half = 0.5 * top
        # Search only between 0.4 and 1.15 r_out, so a neighbour's light beyond
        # the edge is not taken as the crossing.
        above = (p > half) & (rad > SEARCH_PIX[0] * r_out) & (rad < SEARCH_PIX[1] * r_out)
        if not above.any():
            continue
        i_out = np.nonzero(above)[0].max()
        i_in = np.nonzero(above)[0].min()
        if 0 < i_in and i_out < nr - 1 and p[i_in - 1] <= half and p[i_out + 1] <= half:
            r_in[b] = rad[i_in - 1] + (half - p[i_in - 1]) / (p[i_in] - p[i_in - 1]) * DR_PIX
            r_ot[b] = rad[i_out] + (p[i_out] - half) / (p[i_out] - p[i_out + 1]) * DR_PIX
            for val, flag in ((r_in[b], m_in), (r_ot[b], m_ot)):
                j = int(val / DR_PIX)
                flag[b] = bad[b, max(j - 2, 0):j + 3].any()
    return r_in, r_ot, m_in, m_ot, tops


def spider_bins(params, rtp_deg, thx, thy, offsets, npix, r_out_pix):
    """Azimuth bins (CCS, about the donut centre) crossed by a v4.02 Spider_3D shadow.

    The shadow is read off two danish images at the batoid intrinsic, with and
    without the vanes, so the pupil-to-image inversion is danish's own. A bin
    is crossed when the vanes remove more than 20 % of the light from any ring
    pixel in it; the neighbouring bins are added for the blur.
    """
    import danish
    from rubinwork.giant_donuts import fit_giant_donut
    import obscuration_v402 as ob
    import z4_fits
    blitz_utils, _ = fit_giant_donut._import_blitz()
    instrument = blitz_utils._INSTRUMENT
    r_outer, r_inner = ob._annulus(params)
    _, _, _, optic = __import__('fit_donuts').pupil('v4.02', None)
    zk = z4_fits.zk_intrinsic(optic, thx, thy, offsets, r_outer, r_inner)
    no_vanes = {k: v for k, v in params.items() if k != 'Spider_3D'}
    imgs = []
    for mp in (params, no_vanes):
        fac = danish.DonutFactory(R_outer=r_outer, R_inner=r_inner, mask_params=mp,
                                  focal_length=instrument.focalLength,
                                  pixel_scale=instrument.pixelSize, spider_angle=rtp_deg)
        imgs.append(np.asarray(fac.image(aberrations=zk, thx=thx, thy=thy, npix=npix)))
    with_v, without = imgs
    n = npix // 2
    yy, xx = np.mgrid[:npix, :npix] - n
    lit = without > 0.5 * np.median(without[without > 0])
    shadow = lit & (with_v < 0.8 * without)
    phi = np.mod(np.degrees(np.arctan2(yy[shadow], xx[shadow])), 360.0)
    hit = np.zeros(NBIN, bool)
    hit[np.minimum((phi / (360 / NBIN)).astype(int), NBIN - 1)] = True
    return hit | np.roll(hit, 1) | np.roll(hit, -1)


def runs(mask, min_len):
    """(start, stop) bin index pairs of circular runs of True of at least `min_len`."""
    if mask.all():
        return [(0, NBIN)]
    roll = np.argmin(mask)
    m = np.roll(mask, -roll)
    out = []
    i = 0
    while i < NBIN:
        if m[i]:
            j = i
            while j < NBIN and m[j]:
                j += 1
            if j - i >= min_len:
                out.append(((i + roll) % NBIN, (j + roll) % NBIN))
            i = j
        else:
            i += 1
    return out


def clip_surface(npz, cx, cy, b0, b1, r_edge, edge):
    """Surface that sets the Batoid edge over bins `b0`..`b1`.

    The full-model `traceFull` rays landing within `EDGE_BAND_PIX` outside the
    Batoid edge (beyond it for the outer edge, inside it for the inner edge);
    the surface that first vignettes most of them.

    Returns
    -------
    surface : `str`
        Leaf name, with the runner-up and its fraction.
    fraction : `float`
        Fraction of those rays the named surface blocks first, dimensionless.
    """
    px, py, first, names = npz['tf_px'], npz['tf_py'], npz['tf_first'], npz['tf_names']
    phi = np.mod(np.degrees(np.arctan2(py - cy, px - cx)), 360.0)
    k = np.minimum((phi / (360 / NBIN)).astype(int), NBIN - 1)
    r = np.hypot(px - cx, py - cy)
    sign = 1.0 if edge == 'outer' else -1.0
    near = np.zeros(len(px), bool)
    for b in np.arange(b0, b0 + ((b1 - b0) % NBIN or NBIN)) % NBIN:
        if np.isfinite(r_edge[b]):
            d = sign * (r - r_edge[b])
            near |= (k == b) & (d > 0) & (d < EDGE_BAND_PIX)
    blocked = near & (first >= 0)
    if not blocked.any():
        return 'none traced', 0.0
    u, c = np.unique(first[blocked], return_counts=True)
    leaf = [str(names[i]).split('.')[-1] for i in u]
    order = np.argsort(c)[::-1]
    frac = c[order] / c.sum()
    main = leaf[order[0]]
    if len(order) > 1:
        main += f' (then {leaf[order[1]]}, {frac[1]:.2f})'
    return main, float(frac[0])


def measure(run, mask_v402):
    """Edges per side and image; returns the per-bin table and the outlier table."""
    import fit_donuts
    z4 = pd.read_parquet(run / 'z4_fits.parquet')
    params = yaml.safe_load(open(mask_v402))
    fit_donuts.pupil('v4.02', mask_v402)
    bins_rows, outliers = [], []
    for (cand, side), grp in z4.groupby(['cand_id', 'side']):
        g = grp.set_index('pupil')
        r4 = g.loc['v4.02']
        st = np.load(run.parent / '20261018_fits' / 'images' / f'{cand}_{side}_stamp.npz')
        img, msk = st['stamp'].astype(float), st['masked'].astype(bool)
        n = img.shape[0]
        cx, cy = n // 2 + r4.dx_pix, n // 2 + r4.dy_pix
        bat = np.load(run / 'images' / f'{cand}_{side}_batoid.npz')
        images = {'data': (img, ~msk),
                  'v4.02 model': (np.load(run / 'images' / f'{cand}_{side}_v4.02_z4.npz')['model'], ~msk),
                  'v1000 model': (np.load(run / 'images' / f'{cand}_{side}_v1000_z4.npz')['model'], ~msk),
                  'Batoid': (bat['batoid'], ~msk)}
        res = {k: edges(a, v, cx, cy, r4.r_out_pix) for k, (a, v) in images.items()}
        thx, thy = np.radians(r4.field_x_ccs_deg), np.radians(r4.field_y_ccs_deg)
        offsets = (r4.det_offset_mm * 1e-3, r4.cam_offset_mm * 1e-3, r4.m2_offset_mm * 1e-3)
        spider = spider_bins(params, r4.rtp_deg, thx, thy, offsets, n, r4.r_out_pix)
        az_ccs = (np.arange(NBIN) + 0.5) * 360 / NBIN
        az_ocs = np.mod(az_ccs + r4.rtp_deg, 360.0)
        d_in, d_out, dm_in, dm_out, _ = res['data']
        top_b = res['Batoid'][4]
        dark = ~(top_b > DARK_FRAC * np.nanmax(top_b))
        dark = dark | np.roll(dark, 1) | np.roll(dark, -1)
        for b in range(NBIN):
            row = dict(cand_id=cand, side=side, bin=b, az_ccs_deg=az_ccs[b], az_ocs_deg=az_ocs[b],
                       spider=bool(spider[b]), vignetted=bool(dark[b]),
                       masked_inner=bool(dm_in[b]),
                       masked_outer=bool(dm_out[b]), field_r_deg=r4.field_r_deg,
                       clean=bool(r4.clean))
            for k, (ri, ro, _, _, _) in res.items():
                tag = k.replace(' ', '_').replace('.', '')
                row[f'r_inner_{tag}_pix'] = ri[b]
                row[f'r_outer_{tag}_pix'] = ro[b]
            bins_rows.append(row)
        for edge, idx, mflag in (('inner', 0, dm_in), ('outer', 1, dm_out)):
            diff = res['data'][idx] - res['Batoid'][idx]
            usable = ~spider & ~dark & ~mflag & np.isfinite(diff)
            for b0, b1 in runs(usable & (np.abs(diff) > OUTLIER_PIX), OUTLIER_RUN):
                sel = np.arange(b0, b0 + ((b1 - b0) % NBIN or NBIN)) % NBIN
                surface, frac = clip_surface(bat, cx, cy, b0, b1, res['Batoid'][idx], edge)
                outliers.append(dict(
                    cand_id=cand, side=side, edge=edge, field_r_deg=r4.field_r_deg,
                    clean=bool(r4.clean),
                    az_ocs_from_deg=float(az_ocs[sel[0]] - 1.0) % 360,
                    az_ocs_to_deg=float(az_ocs[sel[-1]] + 1.0) % 360, n_bins=len(sel),
                    median_offset_pix=float(np.median(diff[sel])),
                    max_offset_pix=float(diff[sel][np.argmax(np.abs(diff[sel]))]),
                    clipping_surface=surface, surface_fraction=frac))
        print(cand, side, flush=True)
    return pd.DataFrame(bins_rows), pd.DataFrame(outliers)


def edge_stats(bins):
    """RMS and max |data - Batoid| and |v4.02 model - Batoid| per side and edge, pixels."""
    rows = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        row = dict(cand_id=cand, side=side, field_r_deg=g.field_r_deg.iloc[0],
                   clean=g.clean.iloc[0])
        for edge in ('inner', 'outer'):
            use = ~g.spider & ~g.vignetted & ~g[f'masked_{edge}']
            bat = g[f'r_{edge}_Batoid_pix']
            for name, col in (('data', f'r_{edge}_data_pix'), ('v402', f'r_{edge}_v402_model_pix')):
                d = (g[col] - bat)[use].dropna()
                row[f'{edge}_{name}_rms_pix'] = float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
                row[f'{edge}_{name}_max_pix'] = float(np.abs(d).max()) if len(d) else np.nan
                row[f'{edge}_n_bins'] = int(len(d))
        rows.append(row)
    return pd.DataFrame(rows)


def stamp_key(ax, rtp_deg):
    """CCS axes and the OCS azimuth-0 direction on a CCS stamp (column = +x, row = +y)."""
    from select_candidates import axis_key
    axis_key(ax, (0.08, 0.08), ccs_x=(1.0, 0.0), ccs_y=(0.0, 1.0), frame='CCS stamp')
    a = np.deg2rad(-rtp_deg)
    ax.annotate('', xy=(0.86 + 0.06 * np.cos(a), 0.08 + 0.06 * np.sin(a)),
                xytext=(0.86, 0.08), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='cyan', lw=1.2))
    ax.text(0.86, 0.04, 'OCS az 0', color='cyan', fontsize=7, transform=ax.transAxes)


def _scales(img, resid_list):
    vmax = np.nanpercentile(img, 99.5)
    rl = max(np.nanpercentile(np.abs(r), 99) for r in resid_list)
    return vmax, rl


def page_z4(pdf, run, cand, side, g):
    st = np.load(run.parent / '20261018_fits' / 'images' / f'{cand}_{side}_stamp.npz')
    img, msk = st['stamp'].astype(float), st['masked'].astype(bool)
    mods = {p: np.load(run / 'images' / f'{cand}_{side}_{p}_z4.npz')['model'] for p in PUPILS}
    res = {p: np.where(msk, np.nan, img - mods[p]) for p in PUPILS}
    vmax, rl = _scales(img, res.values())
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, p in enumerate(PUPILS):
        for k, (a, title, kw) in enumerate((
                (np.where(msk, np.nan, img), 'data', dict(vmin=0, vmax=vmax, cmap='gray')),
                (mods[p], 'Z4 fit model', dict(vmin=0, vmax=vmax, cmap='gray')),
                (res[p], 'data - model', dict(vmin=-rl, vmax=rl, cmap='RdBu_r')))):
            im = axes[i, k].imshow(a, origin='lower', **kw)
            axes[i, k].set_title(f'{p} {title}')
            stamp_key(axes[i, k], g.iloc[0].rtp_deg)
            fig.colorbar(im, ax=axes[i, k], shrink=0.75, label='electrons per pixel')
        rr = g.loc[p]
        axes[0, 3].text(0, 1.0 - 0.5 * i,
                        f'{p}\nresidual RMS {rr.resid_rms_e:.1f} electrons per pixel'
                        f'\nchi2/dof {rr.chi2_per_dof:.3f} (dof {int(rr.dof)})'
                        f'\nZ4 fitted {rr.z4_um:+.2f} µm, intrinsic {rr.z4_ref_um:+.2f} µm, '
                        f'\nZ4 - intrinsic {rr.z4_dev_um:+.3f} µm (µm of wavefront)'
                        f'\ncentre ({rr.dx_pix:+.2f}, {rr.dy_pix:+.2f}) pixels',
                        va='top', fontsize=9, transform=axes[0, 3].transAxes)
    axes[0, 3].axis('off')
    d = np.abs(res['v1000']) - np.abs(res['v4.02'])
    dl = np.nanpercentile(np.abs(d), 99)
    im = axes[1, 3].imshow(d, origin='lower', cmap='PuOr', vmin=-dl, vmax=dl)
    axes[1, 3].set_title('|resid v1000| - |resid v4.02|\n(>0: v4.02 closer)', fontsize=9)
    fig.colorbar(im, ax=axes[1, 3], shrink=0.8, label='electrons per pixel')
    f0 = g.iloc[0]
    fig.suptitle(f'Step 1, Z4-only fit: {cand} {side} {f0.detector}, field radius '
                 f'{f0.field_r_deg:.3f} deg; Noll 5-28 fixed at batoid intrinsic, blur FWHM '
                 f'fixed 1.0 arcsec; CCS stamp; {"clean" if f0.clean else "FLAGGED"}',
                 fontsize=10)
    pdf.savefig(fig, dpi=90)
    plt.close(fig)
    return vmax, rl


def page_batoid(pdf, run, cand, side, b, vmax, rl, save=None):
    st = np.load(run.parent / '20261018_fits' / 'images' / f'{cand}_{side}_stamp.npz')
    img, msk = st['stamp'].astype(float), st['masked'].astype(bool)
    bat = np.load(run / 'images' / f'{cand}_{side}_batoid.npz')['batoid']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    for ax, (a, title, kw) in zip(axes, (
            (np.where(msk, np.nan, img), 'data', dict(vmin=0, vmax=vmax, cmap='gray')),
            (bat, 'Batoid v4.02', dict(vmin=0, vmax=vmax, cmap='gray')),
            (np.where(msk, np.nan, img - bat), 'data - Batoid', dict(vmin=-rl, vmax=rl, cmap='RdBu_r')))):
        im = ax.imshow(a, origin='lower', **kw)
        ax.set_title(title)
        stamp_key(ax, b.get('rtp_deg', 0.0) if hasattr(b, 'get') else 0.0)
        fig.colorbar(im, ax=ax, shrink=0.8, label='electrons per pixel')
    fig.suptitle(
        f'Step 2, Batoid image: {cand} {side}; camera piston {b.piston_um:+.1f} µm '
        f'(Batoid Z4 {b.z4_batoid_um:+.2f} µm vs fitted {b.z4_target_um:+.2f} µm); '
        f'{"FULL model" if b.used_full else "reduced model"} '
        f'({b.n_clippers} clipping surfaces, agreement {b.agreement:.6f}); '
        f'xcorr shift ({b.xcorr_dx_pix:+.2f}, {b.xcorr_dy_pix:+.2f}) pixels; '
        f'residual RMS {b.resid_rms_e:.1f} electrons per pixel', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=90)
    if save:
        fig.savefig(save, dpi=80)
    plt.close(fig)


def _mark(ax, g, flag, color, label):
    for x in g.az_ocs_deg[g[flag]]:
        ax.axvspan(x - 1, x + 1, color=color, alpha=0.25, lw=0)
    ax.plot([], [], color=color, alpha=0.4, lw=6, label=label)


def page_edges(pdf, cand, side, g, outl, save=None):
    g = g.sort_values('az_ocs_deg')
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, edge in zip(axes[:2], ('outer', 'inner')):
        for name, kw in CURVES.items():
            tag = name.replace(' ', '_').replace('.', '')
            ax.plot(g.az_ocs_deg, g[f'r_{edge}_{tag}_pix'], lw=1, label=name, **kw)
        _mark(ax, g, 'spider', 'gold', 'spider bin')
        _mark(ax, g, f'masked_{edge}', 'tab:purple', 'neighbour mask')
        _mark(ax, g, 'vignetted', 'tab:gray', 'vignetted (Batoid)')
        ax.set_ylabel(f'r_{edge} (pixels)')
        sec = ax.secondary_yaxis('right', functions=(lambda r: r * PIXEL_MM, lambda m: m / PIXEL_MM))
        sec.set_ylabel(f'r_{edge} (mm, focal plane)')
        ax.legend(fontsize=7, ncol=6, loc='lower center')
    ax = axes[2]
    for edge, ls in (('outer', '-'), ('inner', ':')):
        use = ~g.spider & ~g.vignetted & ~g[f'masked_{edge}']
        bat = g[f'r_{edge}_Batoid_pix']
        ax.plot(g.az_ocs_deg, np.where(use, g[f'r_{edge}_data_pix'] - bat, np.nan), color='k',
                ls=ls, lw=1, label=f'data - Batoid, {edge}')
        ax.plot(g.az_ocs_deg, np.where(use, g[f'r_{edge}_v402_model_pix'] - bat, np.nan),
                color='tab:red', ls=ls, lw=1, label=f'v4.02 model - Batoid, {edge}')
    for r in outl.itertuples():
        lo, hi = r.az_ocs_from_deg, r.az_ocs_to_deg
        for a, b in ([(lo, hi)] if lo < hi else [(lo, 360.0), (0.0, hi)]):
            ax.axvspan(a, b, color='tab:orange', alpha=0.2)
    ax.axhline(OUTLIER_PIX, color='gray', ls=':')
    ax.axhline(-OUTLIER_PIX, color='gray', ls=':')
    ax.set_ylabel('edge difference (pixels)')
    ax.set_xlabel('azimuth, OCS (deg)')
    ax.set_xlim(0, 360)
    ax.legend(fontsize=7, ncol=4)
    f0 = g.iloc[0]
    fig.suptitle(f'Step 3, edges: {cand} {side}, field radius {f0.field_r_deg:.3f} deg, '
                 f'{"clean" if f0.clean else "flagged"}; 50 % of flat top, 2 deg bins; '
                 'orange: |data - Batoid| > 3 pixels over 3 or more bins', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig, dpi=90)
    if save:
        fig.savefig(save, dpi=110)
    plt.close(fig)


def page_summary(pdf, z4, bat, stats, outl):
    p = z4.pivot_table(index=['cand_id', 'side'], columns='pupil', values='resid_rms_e')
    clean = z4.groupby(['cand_id', 'side']).clean.first()
    ratio = p['v4.02'] / p['v1000']
    v4 = z4[z4.pupil == 'v4.02']
    lines = ['Step 1, Z4-only fits (Noll 5-28 fixed at the batoid intrinsic, blur FWHM 1.0 arcsec)',
             f'  v4.02 lower residual RMS on {(ratio < 1).sum()} of {len(ratio)} donut sides (all), '
             f'{(ratio[clean] < 1).sum()} of {clean.sum()} (clean).',
             f'  median v4.02/v1000 residual RMS (dimensionless): {ratio.median():.4f} all, '
             f'{ratio[clean].median():.4f} clean.',
             f'  Z4 - intrinsic (µm of wavefront), v4.02: {v4.z4_dev_um.min():+.2f} to '
             f'{v4.z4_dev_um.max():+.2f} all; {v4[v4.clean].z4_dev_um.min():+.2f} to '
             f'{v4[v4.clean].z4_dev_um.max():+.2f} clean.', '',
             'Step 2, Batoid images',
             f'  reduced model used on {(~bat.used_full).sum()} of {len(bat)}; agreement '
             f'{bat.agreement.min():.6f} to {bat.agreement.max():.6f} (dimensionless); '
             f'clipping surfaces {bat.n_clippers.min()} to {bat.n_clippers.max()}.',
             f'  camera piston {bat.piston_um.min():+.1f} to {bat.piston_um.max():+.1f} µm; '
             f'xcorr shift |dx| <= {bat.xcorr_dx_pix.abs().max():.2f}, '
             f'|dy| <= {bat.xcorr_dy_pix.abs().max():.2f} pixels.', '']
    head = (f'{"donut side":<14}{"r (deg)":>8} {"cl":<3}{"Z4 v1000":>9}{"Z4 v4.02":>9}'
            f'{"dZ4 v1000":>10}{"dZ4 v4.02":>10}{"chi2/dof v1000":>15}{"v4.02":>7}'
            f'{"RMS v1000":>10}{"v4.02":>7}{"piston":>8}{"dx":>7}{"dy":>7}')
    lines += ['Per side: Z4 and Z4 - intrinsic in µm of wavefront; residual RMS in electrons per '
              'pixel; piston in µm; xcorr shift in pixels', head]
    b = bat.set_index(['cand_id', 'side'])
    for (cand, side), g in z4.groupby(['cand_id', 'side']):
        g = g.set_index('pupil')
        a, c = g.loc['v1000'], g.loc['v4.02']
        bb = b.loc[(cand, side)]
        lines.append(f'{cand + " " + side:<14}{c.field_r_deg:>8.3f} {"y" if c.clean else "-":<3}'
                     f'{a.z4_um:>9.2f}{c.z4_um:>9.2f}{a.z4_dev_um:>10.3f}{c.z4_dev_um:>10.3f}'
                     f'{a.chi2_per_dof:>15.2f}{c.chi2_per_dof:>7.2f}{a.resid_rms_e:>10.1f}'
                     f'{c.resid_rms_e:>7.1f}{bb.piston_um:>8.1f}{bb.xcorr_dx_pix:>7.2f}'
                     f'{bb.xcorr_dy_pix:>7.2f}')
    fig = plt.figure(figsize=(17, 11))
    fig.text(0.02, 0.98, '\n'.join(lines), family='monospace', fontsize=7.5, va='top')
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for i, edge in enumerate(('outer', 'inner')):
        for j, kind in enumerate(('rms', 'max')):
            ax = axes[i, j]
            for name, color in (('data', 'k'), ('v402', 'tab:red')):
                for cl, mk in ((True, 'o'), (False, 'x')):
                    s = stats[stats.clean == cl]
                    ax.scatter(s.field_r_deg, s[f'{edge}_{name}_{kind}_pix'], color=color,
                               marker=mk, s=30,
                               label=f'{"data" if name == "data" else "v4.02 model"} - Batoid, '
                                     f'{"clean" if cl else "flagged"}')
            ax.set_yscale('log')
            ax.set_ylabel(f'{kind.upper()} |edge difference|, {edge} (pixels)')
            ax.axvline(1.725, color='gray', ls=':')
            ax.legend(fontsize=7)
        axes[1, 0].set_xlabel('field radius (deg)')
        axes[1, 1].set_xlabel('field radius (deg)')
    fig.suptitle('Edge differences against the Batoid image, per donut side (spider, '
                 'vignetted and neighbour-masked bins excluded)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    lines = [f'Unmodelled surfaces: |data - Batoid| > {OUTLIER_PIX:.0f} pixels over '
             f'{OUTLIER_RUN} or more adjacent 2 deg bins; clipping surface from the full v4.02 '
             'traceFull rays just outside the Batoid edge (fraction of those rays it blocks first, '
             'dimensionless)', '',
             f'{"donut side":<14}{"r (deg)":>8} {"edge":<6}{"az OCS (deg)":>16}{"bins":>5}'
             f'{"median (pix)":>13}{"max (pix)":>10}  clipping surface']
    for r in outl.sort_values(['field_r_deg', 'cand_id', 'side']).itertuples():
        lines.append(f'{r.cand_id + " " + r.side:<14}{r.field_r_deg:>8.3f} {r.edge:<6}'
                     f'{r.az_ocs_from_deg:>7.0f} to {r.az_ocs_to_deg:>4.0f}{r.n_bins:>5}'
                     f'{r.median_offset_pix:>13.2f}{r.max_offset_pix:>10.2f}  '
                     f'{r.clipping_surface} ({r.surface_fraction:.2f})')
    for i0 in range(0, len(lines), 90):
        fig = plt.figure(figsize=(17, 11))
        fig.text(0.02, 0.98, '\n'.join(lines[i0:i0 + 90]), family='monospace', fontsize=7,
                 va='top')
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--mask-v402', default='../output/20261018_obscuration_v402/'
                    'RubinObsc_v4.02_r_rtpp0_pp0d0.yaml')
    ap.add_argument('--figure-side', default='G0515_extra')
    args = ap.parse_args()
    run = pathlib.Path(args.run).resolve()
    z4 = pd.read_parquet(run / 'z4_fits.parquet')
    bat = pd.read_parquet(run / 'batoid_images.parquet')
    bins, outl = measure(run, args.mask_v402)
    stats = edge_stats(bins)
    for name, df in (('edges_bins', bins), ('edge_stats', stats), ('unmodelled_surfaces', outl)):
        df.to_parquet(run / f'{name}.parquet')
        df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')
    out = run / 'giant_donut_pupil_model.pdf'
    b = bat.set_index(['cand_id', 'side'])
    with PdfPages(out) as pdf:
        for (cand, side), g in z4.groupby(['cand_id', 'side']):
            vmax, rl = page_z4(pdf, run, cand, side, g.set_index('pupil'))
            label = f'{cand}_{side}'
            page_batoid(pdf, run, cand, side, b.loc[(cand, side)], vmax, rl,
                        save=run / f'batoid_{label}.png' if label == args.figure_side else None)
            page_edges(pdf, cand, side, bins[(bins.cand_id == cand) & (bins.side == side)],
                       outl[(outl.cand_id == cand) & (outl.side == side)] if len(outl) else outl,
                       save=run / f'edges_{label}.png' if label == args.figure_side else None)
        page_summary(pdf, z4, bat, stats, outl)
    print('wrote', out)


if __name__ == '__main__':
    main()
