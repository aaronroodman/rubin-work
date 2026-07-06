#!/usr/bin/env python3
"""wfs_fam_compare — per-triplet FAM vs in-focus corner-WFS wavefront, vs azimuth,
with an optional interleaved donut-fit gallery.

A foundational, PER-IMAGE FAM<->WFS consistency check.  For each FAM triplet (near
a chosen rotator angle and elevation; default rotator 0 deg, elevation 70 deg):

  Page 1 — Z5/Z6/Z7/Z8 median measured Zernike vs focal-plane azimuth:
    * FAM  — every science-array donut in the WFS annulus (r in [r_min, r_max]).
    * WFS  — the corner-WFS donuts of the paired in-focus exposure (fam_seq_num).
  Then (if --donut-gallery) one page PER CORNER (R00/R04/R40/R44) of the in-focus
  exposure's USED donut pairs, in the donut_viz RubinTV style: per pair a row of
    [intra: data | model | residual]  [extra: data | model | residual]  [Zernike bars]
  The danish model is re-rendered from the fitted Zernikes (the model is not
  persisted) following lsst-ts/donut_viz PlotDonutFitsTask.getModel.

Reads parquets for the comparison:
    output/<ps>/donuts.parquet, visits.parquet, wfs/donuts.parquet
The gallery additionally needs the butler (corner-WFS collection), danish, and
lsst.ts.wep — RSP only.  Use --max-visits to cap the number of triplets.

Writes output/<ps>/wfs/fam_wfs_triplet_compare.pdf .
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.table import QTable
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_ZK = [5, 6, 7, 8]
SW0 = {191: 'R00', 195: 'R04', 199: 'R40', 203: 'R44'}   # extra-focal SW0 detectors

# donut_viz colour scheme (PlotDonutFitsTask.plotResults), copied verbatim
DV_CMAP = LinearSegmentedColormap.from_list(
    'cyan_white_magenta', list(zip([0.0, 1/11, 1.0],
                                   [(0., 0., 1.), (1., 1., 1.), (1., 0., 0.)])))


# ----------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------
def _alt_to_deg(a):
    """Robustly express altitude in degrees (auto-detect radians)."""
    a = np.asarray(a, dtype=float)
    return np.rad2deg(a) if np.nanmax(np.abs(a)) < 2 * np.pi + 1e-3 else a


def _az_medians(az, vals, azedges, min_n):
    """Median + count of ``vals`` per azimuth bin (NaN-aware)."""
    from scipy.stats import binned_statistic
    fin = np.isfinite(vals)
    if fin.sum() == 0:
        return np.full(len(azedges) - 1, np.nan), np.zeros(len(azedges) - 1, int)
    med, _, _ = binned_statistic(az[fin], vals[fin], 'median', bins=azedges)
    cnt, _, _ = binned_statistic(az[fin], vals[fin], 'count', bins=azedges)
    cnt = cnt.astype(int)
    med[cnt < min_n] = np.nan
    return med, cnt


def _annulus(thx_deg, thy_deg, r_min, r_max):
    r = np.hypot(thx_deg, thy_deg)
    az = np.degrees(np.arctan2(thy_deg, thx_deg)) % 360.0
    return (r >= r_min) & (r <= r_max), az


def comparison_page(pdf, zks, row, fmask, wmask, fam, wfs, jf, jw, args, azedges, azc):
    """One Z5-Z8 FAM-vs-WFS azimuth page for a triplet; returns True if drawn."""
    import matplotlib.pyplot as plt
    try:
        from lsst.ts.intrinsic.wavefront.common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    fa_in, fa_az = _annulus(fam['thx'][fmask], fam['thy'][fmask], args.r_min, args.r_max)
    wa_in, wa_az = _annulus(wfs['thx'][wmask], wfs['thy'][wmask], args.r_min, args.r_max)
    if int(fa_in.sum()) == 0 and int(wa_in.sum()) == 0:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained', sharex=True)
    axes = axes.ravel()
    for p, j in enumerate(zks):
        ax = axes[p]
        if j in jf and fa_in.sum():
            fv = fam['zk'][fmask][fa_in, jf[j]]
            ax.scatter(fa_az[fa_in], fv, s=10, color='steelblue', alpha=0.25, edgecolors='none')
            med, _ = _az_medians(fa_az[fa_in], fv, azedges, args.min_donuts_per_bin)
            ax.plot(azc, med, '-o', color='steelblue', ms=5, lw=1.3,
                    label=f'FAM (n={int(fa_in.sum())})')
        if j in jw and wa_in.sum():
            wv = wfs['zk'][wmask][wa_in, jw[j]]
            ax.scatter(wa_az[wa_in], wv, s=28, color='crimson', alpha=0.5, marker='s', edgecolors='none')
            med, _ = _az_medians(wa_az[wa_in], wv, azedges, args.min_donuts_per_bin)
            ax.plot(azc, med, '-s', color='crimson', ms=6, lw=1.3,
                    label=f'WFS (n={int(wa_in.sum())})')
        ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)
        ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}', fontsize=10)
        ax.set_ylabel(f'Z{j} [μm]', fontsize=9)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60))
        if p >= 2:
            ax.set_xlabel('focal-plane azimuth [deg] (OCS)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f"FAM vs corner-WFS — day_obs {int(row.day_obs)}, FAM seq {int(row.seq_num)} "
                 f"(rotator {row.rotator_angle:.1f} deg, alt {row.alt_deg:.1f} deg)   "
                 f"annulus r=[{args.r_min}, {args.r_max}] deg", fontsize=12)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    return True


# ----------------------------------------------------------------------
# donut model rendering (lifted from donut_viz PlotDonutFitsTask.getModel)
# ----------------------------------------------------------------------
class DonutModeler:
    """Re-render danish donut model images for an intra/extra pair — standalone
    copy of donut_viz PlotDonutFitsTask.getModel (no Task / EFD needed)."""
    KEYS = ['fit_success', 'fwhm', 'model_bkg', 'model_dx', 'model_dy', 'model_flux']

    def __init__(self):
        import yaml
        import danish
        from lsst.ts.wep.estimation import DanishAlgorithm
        from lsst.ts.wep.utils import getTaskInstrument
        self._danish = danish
        self.inst = getTaskInstrument("LSSTCam", "R00_SW0", None)
        self.algo = DanishAlgorithm()
        with open(Path(danish.datadir) / "RubinObsc.yaml") as f:
            self.mask_params = yaml.safe_load(f)
        self.factory = None

    def build_factory(self, stampsExtra):
        md = stampsExtra.metadata
        rtp_deg = np.rad2deg(md["BORESIGHT_PAR_ANGLE_RAD"]
                             - md["BORESIGHT_ROT_ANGLE_RAD"] - np.pi / 2)
        self.factory = self._danish.DonutFactory(
            R_outer=self.inst.radius, R_inner=self.inst.radius * self.inst.obscuration,
            mask_params=self.mask_params, focal_length=self.inst.focalLength,
            pixel_scale=self.inst.pixelSize, spider_angle=rtp_deg)

    def model_pair(self, zk_dev_ccs, zk_int_ccs, noll, meta, extra_stamp, intra_stamp):
        """Return ([extra_img, intra_img], [extra_model, intra_model]) (electrons)."""
        zk_int = np.asarray(zk_int_ccs) * 1e-6
        dz_terms = [(1, j) for j in noll]
        ie, ae, ze, _ = self.algo._prepDanish(image=extra_stamp.wep_im, zkStart=zk_int,
                                              nollIndices=noll, instrument=self.inst)
        ii, ai, zi, _ = self.algo._prepDanish(image=intra_stamp.wep_im, zkStart=zk_int,
                                              nollIndices=noll, instrument=self.inst)
        if meta['fit_success'] <= 0:
            return [ie, ii], [np.zeros_like(ie), np.zeros_like(ii)]
        zk_fit = np.asarray(zk_dev_ccs) * 1e-6 - zk_int
        nbkg = np.asarray(meta['model_bkg']).shape[1]
        bkg_order = int(np.sqrt(9 + 8 * (nbkg - 1)) - 3) // 2
        model = self._danish.DZMultiDonutModel(
            self.factory, z_refs=[ze, zi], dz_terms=dz_terms, field_radius=np.deg2rad(1.81),
            thxs=[ae[0], ai[0]], thys=[ae[1], ai[1]], npix=ie.shape[0], bkg_order=bkg_order)
        bkgs = [tuple(b) for b in meta['model_bkg']]
        mdl = model.model(meta['model_flux'], meta['model_dx'], meta['model_dy'],
                          meta['fwhm'], zk_fit, bkgs=bkgs)
        return [ie, ii], mdl


def _draw_zern_bars(ax, noll, zk, used, ymin, ymax):
    """Per-donut Zernike bar chart with donut_viz mode-group background bands."""
    ax.bar(noll, zk, color='k')
    ax.axhline(0, color='k', lw=0.5)
    for j in [4, 11, 22]:
        ax.axvspan(j - 0.5, j + 0.5, color='red', alpha=0.2, ec='none')
    for j in [5, 12, 23]:
        ax.axvspan(j - 0.5, j + 1.5, color='orange', alpha=0.2, ec='none')
    for j in [7, 16]:
        ax.axvspan(j - 0.5, j + 1.5, color='yellow', alpha=0.2, ec='none')
    for j in [9, 18]:
        ax.axvspan(j - 0.5, j + 1.5, color='green', alpha=0.2, ec='none')
    for j in [14, 25]:
        ax.axvspan(j - 0.5, j + 1.5, color='blue', alpha=0.2, ec='none')
    ax.axvspan(19.5, 21.5, color='indigo', alpha=0.2, ec='none')
    ax.axvspan(26.5, 28.5, color='violet', alpha=0.2, ec='none')
    ax.set_ylim(ymin, ymax); ax.set_xlim(3.5, 28.5)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28])
    ax.yaxis.tick_right()                      # y labels on the right, clear of the abutting image
    ax.tick_params(labelsize=6)
    ax.spines['right'].set_edgecolor('green' if used else 'red')
    ax.spines['right'].set_linewidth(3)


def corner_gallery_page(pdf, butler, modeler, day_obs, seq, fam_seq, agg, noll, ei,
                        det, raft, args):
    """One page: USED donut pairs of one corner, data|model|resid (intra & extra) + bars."""
    import matplotlib.pyplot as plt
    es = butler.get('donutStampsExtra', day_obs=day_obs, seq_num=seq, detector=det)
    is_ = butler.get('donutStampsIntra', day_obs=day_obs, seq_num=seq, detector=det)
    detmask = (np.asarray(agg['detector']).astype(str) == f'{raft}_SW0') & np.asarray(agg['used'])
    idx = np.where(detmask)[0][:args.n_per_corner]
    if len(idx) == 0:
        return False
    ex_xy = np.array([[s.centroid_position.x, s.centroid_position.y] for s in es])
    in_xy = np.array([[s.centroid_position.x, s.centroid_position.y] for s in is_])
    zkcol = f'zk_{args.bar_frame}'
    nrow = len(idx)
    fig, axs = plt.subplots(nrow, 7, squeeze=False, sharex='col',
                            figsize=(11.5, 1.30 * nrow + 0.8),
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 2.6],
                                         'wspace': 0.0, 'hspace': 0.0})
    titles = ['intra data', 'intra model', 'intra resid',
              'extra data', 'extra model', 'extra resid', f'Z ({args.bar_frame}) [μm]']
    for c in range(7):
        axs[0][c].set_title(titles[c], fontsize=8)
    for rr, i0 in enumerate(idx):
        row = agg[i0]
        meta = {k: np.asarray(ei[k])[i0] for k in DonutModeler.KEYS}
        ej = int(np.argmin(np.hypot(ex_xy[:, 0] - row['centroid_x_extra'],
                                    ex_xy[:, 1] - row['centroid_y_extra'])))
        ij = int(np.argmin(np.hypot(in_xy[:, 0] - row['centroid_x_intra'],
                                    in_xy[:, 1] - row['centroid_y_intra'])))
        try:
            imgs, models = modeler.model_pair(row['zk_deviation_CCS'], row['zk_intrinsic_CCS'],
                                              noll, meta, es[ej], is_[ij])
        except Exception as e:
            for c in range(6):
                axs[rr][c].text(0.5, 0.5, 'model failed', fontsize=6, ha='center')
            imgs = [es[ej].stamp_im.image.array, is_[ij].stamp_im.image.array]
            models = [np.zeros_like(imgs[0]), np.zeros_like(imgs[1])]
        ex_img, in_img = imgs[0], imgs[1]
        ex_mdl, in_mdl = models[0], models[1]
        # unit-sum normalize (donut_viz), then per-pair vmax from the intra data
        def _n(a):
            s = np.sum(a)
            return a / s if s > 0 else a
        in_img, ex_img = _n(in_img), _n(ex_img)
        in_mdl, ex_mdl = _n(in_mdl), _n(ex_mdl)
        vmax = float(np.nanquantile(in_img, 0.99)) or 1e-9
        panels = [(in_img, 'd'), (in_mdl, 'd'), (in_img - in_mdl, 'r'),
                  (ex_img, 'd'), (ex_mdl, 'd'), (ex_img - ex_mdl, 'r')]
        for c, (img, kind) in enumerate(panels):
            ax = axs[rr][c]; ax.set_xticks([]); ax.set_yticks([])
            if kind == 'd':
                ax.imshow(img, origin='lower', aspect='auto', cmap=DV_CMAP,
                          vmin=-vmax / 10, vmax=vmax)
            else:
                ax.imshow(img, origin='lower', aspect='auto', cmap='bwr',
                          vmin=-vmax / 3, vmax=vmax / 3)
        res = float(np.sum(np.abs(in_img - in_mdl)))
        axs[rr][0].text(0.04, 0.9, f"blur {meta['fwhm']:.2f}", transform=axs[rr][0].transAxes,
                        fontsize=6, color='k')
        axs[rr][2].text(0.04, 0.9, f"res {res:.3f}", transform=axs[rr][2].transAxes, fontsize=6)
        _draw_zern_bars(axs[rr][6], noll, np.asarray(row[zkcol], float),
                        bool(row['used']), args.zk_ymin, args.zk_ymax)
    axs[-1][6].set_xlabel('Noll index', fontsize=7)
    fig.suptitle(f"{raft} (det {det}) donut fits — day_obs {day_obs}, in-focus seq {seq} "
                 f"(FAM seq {fam_seq})   [{nrow} used pairs]", fontsize=11)
    fh = 1.30 * nrow + 0.8
    fig.subplots_adjust(left=0.03, right=0.95, top=1.0 - 0.55 / fh, bottom=0.30 / fh)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    return True


def _resolve_butler(args):
    repo, coll = args.butler_repo, args.collection
    if repo is None or coll is None:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import load_param_sets
        ps = load_param_sets()[args.param_set]
        repo = repo or ps.get('butler_repo', '/repo/main')
        coll = coll or ps.get('wfs_collection')
    return repo, coll


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--r-min', type=float, default=1.5178)
    ap.add_argument('--r-max', type=float, default=1.725)
    ap.add_argument('--rot-center', type=float, default=0.0)
    ap.add_argument('--rot-tol', type=float, default=3.0)
    ap.add_argument('--alt-center', type=float, default=70.0)
    ap.add_argument('--alt-tol', type=float, default=2.0)
    ap.add_argument('--az-bin-deg', type=float, default=30.0)
    ap.add_argument('--zernikes', default='5,6,7,8')
    ap.add_argument('--min-donuts-per-bin', type=int, default=1)
    ap.add_argument('--max-visits', type=int, default=None,
                    help='process at most this many FAM triplets (seq_num)')
    # donut-fit gallery (RSP: needs butler + danish + lsst.ts.wep)
    ap.add_argument('--donut-gallery', dest='gallery', action='store_true', default=True)
    ap.add_argument('--no-donut-gallery', dest='gallery', action='store_false')
    ap.add_argument('--n-per-corner', type=int, default=8,
                    help='max USED donut pairs shown per corner in the gallery')
    ap.add_argument('--bar-frame', default='CCS', choices=['CCS', 'OCS'],
                    help='frame for the per-donut Zernike bar chart (CCS matches RubinTV)')
    ap.add_argument('--zk-ymin', type=float, default=-1.0)
    ap.add_argument('--zk-ymax', type=float, default=1.0)
    ap.add_argument('--collection', default=None, help='corner-WFS butler collection (auto)')
    ap.add_argument('--butler-repo', default=None)
    args = ap.parse_args()
    coord = args.coord_sys
    zks = [int(x) for x in args.zernikes.split(',')]
    base = Path(args.output_root) / args.param_set

    # ---- FAM per-donut measured wavefront ----
    dd = pq.read_table(str(base / 'donuts.parquet')).to_pandas()
    fam = dict(zk=np.stack(dd[f'zk_{coord}'].values).astype(float),
               thx=np.rad2deg(np.asarray(dd[f'thx_{coord}'], float)),
               thy=np.rad2deg(np.asarray(dd[f'thy_{coord}'], float)))
    fam_day = np.asarray(dd['day_obs']).astype(int)
    fam_seq = np.asarray(dd['seq_num']).astype(int)

    vt = pq.read_table(str(base / 'visits.parquet')).to_pandas()
    noll_fam = ([int(x) for x in np.asarray(vt['nollIndices'].iloc[0])]
                if 'nollIndices' in vt.columns else list(range(4, 4 + fam['zk'].shape[1])))
    vt = vt.copy()
    vt['alt_deg'] = _alt_to_deg(vt['alt']) if 'alt' in vt.columns else np.nan

    # ---- corner-WFS measured wavefront (+ in-focus seq map) ----
    wd = QTable.read(str(base / 'wfs' / 'donuts.parquet'))
    noll_wfs = [int(x) for x in wd.meta.get('nollIndices')]
    wfs = dict(zk=np.array(wd[f'zk_{coord}'], dtype=float),
               thx=np.rad2deg(np.asarray(wd[f'thx_{coord}'], float)),
               thy=np.rad2deg(np.asarray(wd[f'thy_{coord}'], float)))
    wfs_day = np.asarray(wd['day_obs']).astype(int)
    wfs_fam = np.asarray(wd['fam_seq_num']).astype(int)
    wfs_infocus = np.asarray(wd['seq_num']).astype(int)
    infocus_map = {(int(d), int(f)): int(s)
                   for d, f, s in zip(wfs_day, wfs_fam, wfs_infocus)}

    jf = {j: noll_fam.index(j) for j in zks if j in noll_fam}
    jw = {j: noll_wfs.index(j) for j in zks if j in noll_wfs}

    sel = vt[(np.abs(vt['rotator_angle'] - args.rot_center) <= args.rot_tol)
             & (np.abs(vt['alt_deg'] - args.alt_center) <= args.alt_tol)]
    sel = sel.sort_values(['day_obs', 'seq_num'])
    if args.max_visits is not None:
        sel = sel.head(args.max_visits)
    print(f'[wfs_fam_compare] {args.param_set}: {len(sel)} FAM triplets near '
          f'rotator {args.rot_center}+/-{args.rot_tol}, alt {args.alt_center}+/-{args.alt_tol}'
          f'{f" (capped at {args.max_visits})" if args.max_visits else ""}')

    azedges = np.arange(0.0, 360.0 + 1e-6, args.az_bin_deg)
    azc = 0.5 * (azedges[:-1] + azedges[1:])
    out = base / 'wfs' / 'fam_wfs_triplet_compare.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- gallery setup (butler + danish) ----
    butler = modeler = None
    if args.gallery:
        try:
            from lsst.daf.butler import Butler
            repo, coll = _resolve_butler(args)
            butler = Butler(repo, collections=coll)
            modeler = DonutModeler()
            print(f'  donut gallery ON (collection {coll})')
        except Exception as e:
            print(f'  donut gallery DISABLED: {type(e).__name__}: {e}')
            butler = modeler = None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_cmp = n_gal = 0
    with PdfPages(str(out)) as pdf:
        for row in sel.itertuples():
            d, s = int(row.day_obs), int(row.seq_num)
            fmask = (fam_day == d) & (fam_seq == s)
            wmask = (wfs_day == d) & (wfs_fam == s)
            if comparison_page(pdf, zks, row, fmask, wmask, fam, wfs, jf, jw, args, azedges, azc):
                n_cmp += 1
            if butler is not None and (d, s) in infocus_map:
                infocus = infocus_map[(d, s)]
                try:
                    agg = butler.get('aggregateAOSVisitTableRaw', day_obs=d, seq_num=infocus)
                    noll = [int(x) for x in agg.meta['nollIndices']]
                    ei = agg.meta['estimatorInfo']
                    modeler.build_factory(butler.get('donutStampsExtra', day_obs=d,
                                                     seq_num=infocus, detector=191))
                    for det, raft in SW0.items():
                        if corner_gallery_page(pdf, butler, modeler, d, infocus, s,
                                               agg, noll, ei, det, raft, args):
                            n_gal += 1
                except Exception as e:
                    print(f'  gallery skipped for seq {s} (in-focus {infocus}): '
                          f'{type(e).__name__}: {e}')
    print(f'  wrote wfs/fam_wfs_triplet_compare.pdf ({n_cmp} comparison + {n_gal} gallery pages)')


if __name__ == '__main__':
    main()
