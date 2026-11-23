"""Pairs review: focal-plane, fit-gallery and edge PDFs per intra/extra pair, and summaries.

The final review of this study on the v4.02 masks (`pairs_review_plan.md`). A pair is an
intra-focal and an extra-focal exposure whose absolute Trim differs only in camera z and
M2 z (dof0, dof5; −8000 µm intra − extra). Only existing fits are used: fit c with
v4.02g and v4.02s at 1.6-1.725 deg (science) and 1.4-1.75 deg (corner), and v4.02g at
0.25-0.5 deg (science), where v4.02s is the same mask.

Steps:

- ``models`` (batch): render the fit c model of every gallery side, both masks, into
  ``models/<source>/<key>_<side>.npz`` of a new ``pairs_review`` study run. Needs the
  0.25-0.5 deg ``psi_labels`` runs (``psi_labels.py --band 0.25 0.5``) to exist.
- ``pdfs`` (local, ``--out <run>``): per pair ``focal_plane.pdf``, ``fit_gallery.pdf`` and
  ``edges.pdf``; then ``summary.pdf`` (pairs table, edge misses and spider offsets against
  rotator angle) and ``conclusions.pdf`` (each study conclusion with its supporting pages).

Positions on the focal-plane pages are cameraGeom FOCAL_PLANE mm (DVCS, the DM view).
Stamps are CCS pixels, origin lower left.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pairs_review.py --steps models --workers 60
    python aos/giant_donut_pupil/code/pairs_review.py --steps pdfs --out 20261120_pairs_review

With ``--mask-set v4speed``: the v4_speed fits of `v4speed_fits.py`, science sides only,
one mask, and a first summary page of the donut sample in CCS field angle.
"""
import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import psi_labels as pl  # noqa: E402
import psi_edges_report as per  # noqa: E402
import psi_gallery as pg  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
OUT = pl.STUDY_OUT
FWHM_ARCSEC = 1.0  # blur of the fits
OUTER = 'outer'  # science 1.6-1.725 deg and corner 1.4-1.75 deg, both masks
INNER = 'inner'  # science 0.25-0.5 deg, v4.02g
INNER_BAND = (0.25, 0.5)
OUTER_BAND = (1.6, 1.725)
CORNER_BAND = (1.4, 1.75)
MASKS = ('v4.02g', 'v4.02s')
BASE = 'v4.02g'  # the mask of the 0.25-0.5 deg band, the side lists and the exposure table
CORNERS = True
# Mask sets (``--mask-set``): the v4.02 review of 2026-11-20, and the v4_speed rerun
# (science sides only, outer-band labels from ``psi_labels.py --band 1.6 1.725``).
MASK_SETS = {'v402': dict(masks=('v4.02g', 'v4.02s'), corners=True, label='v4.02 masks'),
             'v4speed': dict(masks=('v4_speed',), corners=False, label='v4_speed mask')}
LABEL = MASK_SETS['v402']['label']


def configure(mask_set):
    global MASKS, BASE, CORNERS, LABEL, SIL
    m = MASK_SETS[mask_set]
    MASKS, BASE, CORNERS, LABEL = m['masks'], m['masks'][0], m['corners'], m['label']
    SIL = 'Camera silhouette' if CORNERS else 'CameraBody'
PIXEL_SCALE_ARCSEC = 0.2  # LSSTCam, 10 µm pixels
BAND_COLOURS = {INNER: 'tab:orange', 'science outer': 'red', 'corner outer': 'magenta'}


@dataclasses.dataclass
class Pair:
    name: str
    day_obs: int
    intra: int
    extra: int
    run: str
    fit: dict  # mask -> fit directory
    cache: str  # 4x4-binned ISR cache, <cache>/<visit>/
    candidates: str  # selected-candidate table with a visit column

    def visits(self):
        return {'intra': self.intra, 'extra': self.extra}


N1, N7, N8 = '20261023_pupil6', '20261118_night2_20261107', '20261118_night2_20261108'
PAIRS = [
    Pair('pair0_20251023_340i_337e', 20251023, 2025102300340, 2025102300337, N1,
         {'v4.02g': 'isr_phys', 'v4.02s': 'isr_phys_v402s', 'v4_speed': 'v4speed'}, '20261018_isr_bin4',
         '20261020_candidates_v2/candidates_v2.parquet'),
    Pair('pair1_20261107_366i_388e', 20261107, 2026110700366, 2026110700388, N7,
         {'v4.02g': 'v402g', 'v4.02s': 'v402s', 'v4_speed': 'v4speed'}, f'{N7}/isr_bin4', f'{N7}/candidates.parquet'),
    Pair('pair2_20261108_014i_036e', 20261108, 2026110800014, 2026110800036, N8,
         {'v4.02g': 'v402g', 'v4.02s': 'v402s', 'v4_speed': 'v4speed'}, f'{N8}/isr_bin4', f'{N8}/candidates.parquet'),
    Pair('pair3_20261108_152i_147e', 20261108, 2026110800152, 2026110800147, N8,
         {'v4.02g': 'v402g', 'v4.02s': 'v402s', 'v4_speed': 'v4speed'}, f'{N8}/isr_bin4', f'{N8}/candidates.parquet'),
    Pair('pair4_20261108_168i_190e', 20261108, 2026110800168, 2026110800190, N8,
         {'v4.02g': 'v402g', 'v4.02s': 'v402s', 'v4_speed': 'v4speed'}, f'{N8}/isr_bin4', f'{N8}/candidates.parquet'),
]


def label_runs(run, fit_dir, which):
    """Newest ``psi_labels`` run of a source: the outer bands, or the 0.25-0.5 deg band."""
    tag = f'psi_labels_{run}_{fit_dir}'
    band = INNER_BAND if which == INNER else None if CORNERS else OUTER_BAND
    if band is not None:
        tag += '_band_' + '_'.join(f'{v:g}'.replace('.', 'p') for v in band)
    found = sorted(p.name for p in OUT.glob(f'2*_{tag}') if (p / 'labels.parquet').exists())
    if not found:
        raise FileNotFoundError(f'no {tag} run with labels.parquet')
    return found[-1]


_LOADED = {}


def source(run, fit_dir):
    """(sides, bins, mask run) of one fit directory, cached."""
    if (run, fit_dir) not in _LOADED:
        _LOADED[(run, fit_dir)] = pl._load(run, fit_dir)
    return _LOADED[(run, fit_dir)]


def pair_sides(p, mask, which):
    """Fitted sides of a pair in a band set, restricted to those with element labels."""
    c, _, _ = source(p.run, p.fit[mask])
    c = c.copy()
    if 'visit' not in c:  # pass 6: one visit per side of focus
        c['visit'] = c.side.map(p.visits())
    c = c[[v == p.visits().get(s) for v, s in zip(c.visit, c.side)]]
    if which == INNER:
        c = c[(c.sensor_class == 'science') & c.field_r_deg.between(*INNER_BAND)]
    else:
        c = pl.select(c, band=None if CORNERS else OUTER_BAND)
    lab = pd.read_parquet(OUT / label_runs(p.run, p.fit[mask], which) / 'labels.parquet',
                          columns=['key', 'side'])
    c = c.merge(lab.drop_duplicates())
    c['set'] = which
    return c.sort_values(['side', 'field_az_ocs_deg'], ascending=[False, True])


def model_path(out, run, fit_dir, key, side):
    return out / 'models' / f'{run}__{fit_dir}' / f'{key.replace(":", "_")}_{side}.npz'


def load_model(out, run, fit_dir, key, side):
    return np.load(model_path(out, run, fit_dir, key, side))['model'].astype(float)


# ---------------------------------------------------------------- models (batch)

def _render(task):
    row, path = task
    model = pl.fit_model(row, pl.mask_for(row), FWHM_ARCSEC)
    np.savez_compressed(path, model=model.astype(np.float32))
    print(f'{row.key} {row.side} done', flush=True)
    return path


def step_models(out, workers):
    """Render every gallery model, one pool per fit directory (each has its own masks)."""
    import multiprocessing as mp
    todo = {}
    for p in PAIRS:
        for mask in MASKS:
            for which in (OUTER, INNER):
                if which == INNER and mask != BASE:
                    continue
                for r in pair_sides(p, mask, which).itertuples(index=False):
                    path = model_path(out, p.run, p.fit[mask], r.key, r.side)
                    if not path.exists():
                        todo.setdefault((p.run, p.fit[mask]), {})[path] = pd.Series(r._asdict())
    for (run, fit_dir), tasks in todo.items():
        _, _, mask_run = source(run, fit_dir)
        (out / 'models' / f'{run}__{fit_dir}').mkdir(parents=True, exist_ok=True)
        print(f'{run} {fit_dir}: {len(tasks)} models', flush=True)
        with mp.get_context('spawn').Pool(max(1, min(workers, len(tasks))), initializer=pl._init,
                                          initargs=tuple(map(str, mask_run)),
                                          maxtasksperchild=8) as pool:
            list(pool.imap_unordered(_render, [(r, path) for path, r in tasks.items()]))


# ---------------------------------------------------------------- focal_plane.pdf

def _fp_mm(camera, det, x, y):
    import lsst.geom
    from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
    q = camera[det].getTransform(PIXELS, FOCAL_PLANE).applyForward(lsst.geom.Point2D(x, y))
    return q.getX(), q.getY()


def _field_circle_mm(camera, r_deg):
    from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE
    import lsst.geom
    t = camera.getTransform(FIELD_ANGLE, FOCAL_PLANE)
    q = t.applyForward(lsst.geom.Point2D(np.radians(r_deg), 0.0))
    return float(np.hypot(q.getX(), q.getY()))


def used_donuts(p, camera):
    """Every side of the pair used in a gallery or edge page, with DVCS mm and colour key."""
    rows = []
    for which in (OUTER, INNER):
        c = pair_sides(p, BASE, which)
        for r in c.itertuples(index=False):
            s = _side_row(p, r.key, r.side)
            x, y = _fp_mm(camera, s.detector, float(s.x_pix), float(s.y_pix))
            kind = INNER if which == INNER else (
                'science outer' if r.sensor_class == 'science' else 'corner outer')
            rows.append(dict(key=r.key, side=r.side, visit=r.visit, detector=s.detector,
                             fp_x_mm=x, fp_y_mm=y, r_out_mm=0.01 * float(r.r_out_pix),
                             field_r_deg=r.field_r_deg, mag=float(s.mag), kind=kind))
    return pd.DataFrame(rows)


_SIDES = {}


def _side_row(p, key, side):
    if p.run not in _SIDES:
        f = OUT / p.run / p.fit[BASE] / 'sides.parquet'
        f = f if f.exists() else OUT / p.run / 'sides.parquet'
        _SIDES[p.run] = pd.read_parquet(f).set_index(['key', 'side'])
    return _SIDES[p.run].loc[(key, side)]


def _markers(ax, cand, used, label=False):
    for r in cand.itertuples(index=False):
        ax.add_patch(Circle((r.fp_x_mm, r.fp_y_mm), r.r_out_mm if np.isfinite(r.r_out_mm) else 3.4,
                            fill=False, lw=0.3, color='0.7'))
    for r in used.itertuples(index=False):
        ax.add_patch(Circle((r.fp_x_mm, r.fp_y_mm), r.r_out_mm, fill=False, lw=1.2,
                            color=BAND_COLOURS[r.kind]))
        if label:
            ax.text(r.fp_x_mm, r.fp_y_mm - r.r_out_mm - 0.4,
                    f'{r.key.split(":")[0]}\n{r.field_r_deg:.3f} deg, mag {r.mag:.1f}',
                    fontsize=5, color=BAND_COLOURS[r.kind], ha='center', va='top')


def _fp_legend(ax, used):
    names = {INNER: 'used, science 0.25-0.5 deg', 'science outer': 'used, science 1.6-1.725 deg',
             'corner outer': 'used, corner 1.4-1.75 deg'}
    h = [Patch(fill=False, edgecolor=BAND_COLOURS[k], label=f'{names[k]}: {(used.kind == k).sum()}')
         for k in names]
    h.append(Patch(fill=False, edgecolor='0.7', label='other selected donuts'))
    ax.legend(handles=h, loc='upper right', fontsize=7, framealpha=0.85)


def focal_plane_pdf(p, path, camera, info):
    import select_candidates as sc
    import selection_pages as pages
    import candidates as cd
    from lsst.afw.cameraGeom import FOCAL_PLANE
    used_all = used_donuts(p, camera)
    cand_all = pd.read_parquet(OUT / p.candidates)
    radii = [_field_circle_mm(camera, r) for r in (0.25, 0.5, 1.6, 1.725)]
    with PdfPages(path) as pdf:
        for side in ('intra', 'extra'):
            visit = p.visits()[side]
            used = used_all[used_all.visit == visit]
            cand = cand_all[(cand_all.visit == visit) & cand_all.selected.astype(bool)]
            images, _ = sc.load_cache(OUT / p.cache / str(visit), camera)
            names = sorted(images)
            bkg = {n: cd.background_map(im, block=128, percentile=10.0) for n, im in images.items()}
            mos, geom = pages.raft_mosaic(camera, images, bkg, names, sc.MOSAIC_BIN)
            fig, ax = plt.subplots(figsize=(11, 11))
            sc.show_mosaic(ax, mos, geom, pages.STRETCH_E)
            pages.draw_ccd_edges(ax, camera, names, lw=0.3)
            for r_mm, r_deg in zip(radii, (0.25, 0.5, 1.6, 1.725)):
                ax.add_patch(Circle((0, 0), r_mm, fill=False, ls='--', lw=0.5, color='yellow'))
                ax.text(0, r_mm + 2, f'{r_deg} deg', color='yellow', fontsize=6, ha='center')
            _markers(ax, cand, used)
            _fp_legend(ax, used)
            ax.set_title(f'{p.name}: visit {visit}, {side}-focal, {info[visit]}; '
                         f'mosaic binned {sc.MOSAIC_BIN}x{sc.MOSAIC_BIN}', fontsize=9)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            t = used.groupby(['kind', 'detector']).size().rename('donuts').reset_index()
            pages._text_page(pdf, f'{p.name}: visit {visit} ({side}-focal), donuts used '
                             '(donut counts)', t.to_string(index=False), 8)
            for raft in sorted({n[:3] for n in names}):
                rn = [n for n in names if n.startswith(raft)]
                rmos, rgeom = pages.raft_mosaic(camera, images, bkg, rn, sc.RAFT_BIN)
                fig, ax = plt.subplots(figsize=(11, 11))
                sc.show_mosaic(ax, rmos, rgeom, pages.STRETCH_E)
                pages.draw_ccd_edges(ax, camera, rn, lw=0.8)
                for n in rn:
                    c = camera[n].getCenter(FOCAL_PLANE)
                    ax.text(c.getX(), c.getY(), n[4:], color='cyan', fontsize=7, alpha=0.6,
                            ha='center')
                _markers(ax, cand[cand.detector.str.startswith(raft)],
                         used[used.detector.str.startswith(raft)], label=True)
                ax.set_title(f'{p.name}: visit {visit} ({side}-focal) raft {raft}; '
                             f'binned {sc.RAFT_BIN}x{sc.RAFT_BIN}', fontsize=9)
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
            print(p.name, visit, 'focal plane done', flush=True)


# ---------------------------------------------------------------- fit_gallery.pdf

def _model_panels(ax_m, ax_r, row, g, img, msk, model):
    """One mask's model (edges by element, data edges white) and its residual."""
    top = np.nanpercentile(img[~msk], 99.0)
    ax_m.imshow(model, origin='lower', interpolation='none', cmap='gray', vmin=0, vmax=top)
    h = ax_r.imshow(np.where(msk, np.nan, (img - model) / float(row.noise_e)), origin='lower',
                    interpolation='none', cmap='RdBu_r', vmin=-pg.RESID_SIGMA, vmax=pg.RESID_SIGMA)
    plt.colorbar(h, ax=ax_r, fraction=0.046, pad=0.02, label='(data − model) / noise')
    for e in ('inner', 'outer'):
        cols = [per.COLOURS.get(v, 'k') if isinstance(v, str) else (0, 0, 0, 0)
                for v in g[f'element_{e}']]
        pg._edges(ax_m, g, row.cx_a, row.cy_a, f'r_{e}_c_pix', cols, s=3, lw=0)
        pg._edges(ax_m, g, row.cx_a, row.cy_a, f'r_{e}_data_pix', color='w', ms=1.0)
    d = (g[['r_inner_data_pix', 'r_outer_data_pix']].to_numpy()
         - g[['r_inner_c_pix', 'r_outer_c_pix']].to_numpy())
    ax_m.set_title(f'{row["mask"]}: chi2/dof {row.chi2_dof:.2f} (dof {int(row.dof)})', fontsize=7)
    ax_r.set_title(f'{row["mask"]}: median data − model edge, inner {np.nanmedian(d[:, 0]):+.2f}, '
                   f'outer {np.nanmedian(d[:, 1]):+.2f} pixels', fontsize=7)


def _data_panel(ax, row, g, img, msk):
    top = np.nanpercentile(img[~msk], 99.0)
    ax.imshow(np.where(msk, np.nan, img), origin='lower', interpolation='none', cmap='gray',
              vmin=0, vmax=top)
    for e in ('inner', 'outer'):
        pg._edges(ax, g, row.cx_a, row.cy_a, f'r_{e}_data_pix', color='w', ms=1.0)
    a = np.radians(row.field_az_ocs_deg + 180.0 - row.rtp_deg)
    n = img.shape[0]
    ax.annotate('', xy=(row.cx_a + 0.47 * n * np.cos(a), row.cy_a + 0.47 * n * np.sin(a)),
                xytext=(row.cx_a, row.cy_a),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8))
    ax.set_ylabel(f'{row.key.split(":")[0]}\n{row.side}, {row.sensor_class}\n'
                  f'field {row.field_r_deg:.3f} deg\nOCS az {row.field_az_ocs_deg:.0f} deg\n'
                  f'rtp {row.rtp_deg:.2f} deg', fontsize=7)
    ax.set_title('data (electrons)', fontsize=7)


def _bins_with_labels(p, mask, which, key, side):
    _, b, _ = source(p.run, p.fit[mask])
    lab = _labels(p.run, p.fit[mask], which)
    g = b[(b.key == key) & (b.side == side)]
    return g.merge(lab[(lab.key == key) & (lab.side == side)], on=['key', 'side', 'bin']).sort_values('bin')


_LAB = {}


def _labels(run, fit_dir, which):
    k = (run, fit_dir, which)
    if k not in _LAB:
        _LAB[k] = pl.consistent_kind(
            pd.read_parquet(OUT / label_runs(run, fit_dir, which) / 'labels.parquet'))
    return _LAB[k]


def fit_gallery_pdf(p, path, out):
    rows_per_page = pg.ROWS_PER_PAGE
    with PdfPages(path) as pdf:
        for which, masks in ((OUTER, MASKS), (INNER, (BASE,))):
            sides = {m: pair_sides(p, m, which).assign(mask=m).set_index(['key', 'side'])
                     for m in masks}
            order = list(sides[BASE].index)
            for side in ('intra', 'extra'):
                keys = [k for k in order if k[1] == side and all(k in sides[m].index for m in masks)]
                for i0 in range(0, len(keys), rows_per_page):
                    chunk = keys[i0:i0 + rows_per_page]
                    ncol = 1 + 2 * len(masks)
                    fig, axes = plt.subplots(rows_per_page, ncol,
                                             figsize=(4.0 * ncol, 4.0 * rows_per_page),
                                             squeeze=False)
                    used = set()
                    for k, (key, sd) in enumerate(chunk):
                        img, msk = pg.stamp(OUT / p.run, key)
                        r0 = pd.Series(sides[BASE].loc[(key, sd)]).copy()
                        r0['key'], r0['side'] = key, sd
                        g0 = _bins_with_labels(p, BASE, which, key, sd)
                        _data_panel(axes[k, 0], r0, g0, img, msk)
                        for j, m in enumerate(masks):
                            r = pd.Series(sides[m].loc[(key, sd)]).copy()
                            r['key'], r['side'] = key, sd
                            g = g0 if m == BASE else _bins_with_labels(p, m, which, key, sd)
                            used |= set(g.element_inner.dropna()) | set(g.element_outer.dropna())
                            model = load_model(out, p.run, p.fit[m], key, sd)
                            _model_panels(axes[k, 1 + 2 * j], axes[k, 2 + 2 * j], r, g, img, msk,
                                          model)
                    for ax in axes.ravel():
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if not ax.images:
                            ax.axis('off')
                    if which == OUTER:
                        band = ('science 1.6-1.725 deg and corner 1.4-1.75 deg' if CORNERS
                                else f'science 1.6-1.725 deg, {BASE}')
                    elif len(MASKS) > 1:
                        band = 'science 0.25-0.5 deg (v4.02g; v4.02s is the same mask here)'
                    else:
                        band = f'science 0.25-0.5 deg, {BASE}'
                    handles = [Patch(color=per.COLOURS[u], label=per.NAMES.get(u, u))
                               for u in per.COLOURS if u in used]
                    fig.legend(handles=handles, loc='upper center',
                               ncol=min(len(handles), 2 * ncol),
                               fontsize=7, bbox_to_anchor=(0.5, 0.955))
                    fig.suptitle(f'{p.name}, visit {p.visits()[side]} ({side}-focal), {band}. '
                                 'Fit c models; model edges coloured by mask element, data edges '
                                 'white; yellow arrow: psi 0 (toward the field centre); CCS, '
                                 'origin lower left', fontsize=9, y=0.975)
                    fig.tight_layout(rect=(0, 0, 1, 0.94))
                    pdf.savefig(fig, dpi=90)
                    plt.close(fig)
    print(p.name, 'gallery done', flush=True)


# ---------------------------------------------------------------- edges.pdf

def edges_pdf(p, path, info):
    """The psi_edges_report pages for the pair, grouped by band, mask and side."""
    groups = [(OUTER, m, ('science', s), OUTER_BAND) for m in MASKS for s in ('intra', 'extra')]
    if CORNERS:
        groups += [(OUTER, m, (k, s), CORNER_BAND) for m in MASKS for k in ('SW0', 'SW1')
                   for s in ('intra', 'extra')]
    groups += [(INNER, BASE, ('science', s), INNER_BAND) for s in ('intra', 'extra')]
    tables = []
    with PdfPages(path) as pdf:
        for which, mask, (cls, side), band in groups:
            fd = p.fit[mask]
            key = (p.run, fd, which, 'edges')
            if key not in _LOADED:
                _LOADED[key] = per.load(p.run, fd, [label_runs(p.run, fd, which)])
            b = _LOADED[key]
            visit = p.visits()[side]
            keys = set(pair_sides(p, mask, which).key)
            g = b[b.key.isin(keys) & (b.sensor_class == cls) & (b.side == side)
                  & b.field_r_deg.between(*band)]
            if g[['key', 'side']].drop_duplicates().shape[0] < 3:
                continue
            title = (f'{p.name} {mask} {cls} {side}, {band[0]}-{band[1]} deg: visit {visit}, '
                     f'{info[visit]}')
            cons = per.page_psi(pdf, g, title)
            t = per.element_table(cons, title)
            per.page_table(pdf, t, title)
            tables.append(t.assign(pair=p.name, mask=mask, sensor_class=cls, side=side,
                                   band=f'{band[0]}-{band[1]}', visit=visit))
    t = pd.concat(tables, ignore_index=True)
    t.to_csv(path.with_name('elements.csv'), index=False, float_format='%.4g')
    print(p.name, 'edges done', flush=True)
    return t


# ---------------------------------------------------------------- summary.pdf

def exposure_info():
    """Per visit: rotator angle (deg), band, exposure (s), altitude (deg), Trim (µm)."""
    from rubinwork.products import value_added as efd_db
    rows = []
    for p in PAIRS:
        c, _, _ = source(p.run, p.fit[BASE])
        e = (pd.read_parquet(OUT / p.run / 'exposures.parquet').set_index('visit_id')
             if (OUT / p.run / 'exposures.parquet').exists() else None)
        for side, visit in p.visits().items():
            cc = c[c.visit == visit] if 'visit' in c else c[c.side == side]
            r = dict(pair=p.name, side=side, visit=visit, day_obs=p.day_obs, seq_num=visit % 100000,
                     rtp_deg=float(np.median(cc.rtp_deg)))
            if e is not None:
                x = e.loc[visit]
                r.update(band=x.band, exp_time_s=float(x.exp_time), altitude_deg=float(x.altitude))
            else:
                r.update(band='r', exp_time_s=60.0, altitude_deg=np.nan)
            rows.append(r)
    t = pd.DataFrame(rows)
    con = efd_db.connect()
    q = ' or '.join(f'(day_obs={d} and seq_num={s})' for d, s in zip(t.day_obs, t.seq_num))
    tr = con.execute(f'select day_obs, seq_num, dof0, dof1, dof2, dof5, dof6, dof7 from '
                     f'visit_telemetry where {q}').fetchdf()
    tr = tr.rename(columns={'dof0': 'm2_dz_um', 'dof1': 'm2_dx_um', 'dof2': 'm2_dy_um',
                            'dof5': 'cam_dz_um', 'dof6': 'cam_dx_um', 'dof7': 'cam_dy_um'})
    return t.merge(tr, on=['day_obs', 'seq_num'], how='left')


def _info_text(t):
    return {r.visit: f'rotator {r.rtp_deg:.2f} deg, {r.band} band, {r.exp_time_s:.0f} s'
            for r in t.itertuples()}


ELEMENTS = [('outer', 'M1Baffle1 (clear)'), ('outer', 'M3 outer (clear)'),
            ('inner', 'M1 inner (obsc.)'), ('inner', 'M2 inner (obsc.)'),
            ('inner', 'CameraSilhouette (obsc.)'),
            ('inner', 'CameraSilhouette, v4.02s refit (obsc.)')]


def _element_rows(el, info):
    """Bin-weighted median data − model per visit, mask, edge and element."""
    t = el[el.sensor_class == 'science'].copy()
    t['w'] = t.n_bins * t.dmm_median_pix
    g = t.groupby(['pair', 'mask', 'band', 'side', 'visit', 'edge', 'element'], as_index=False)
    a = g.agg(w=('w', 'sum'), n_bins=('n_bins', 'sum'))
    a['dmm_pix'] = a.w / a.n_bins
    return a.merge(info[['visit', 'rtp_deg', 'cam_dx_um', 'cam_dy_um']], on='visit')


def _elements_for(a):
    """(edge, element) panels: `ELEMENTS` for v4.02; for one mask, the 9 with most bins."""
    if len(MASKS) > 1:
        return [x for x in ELEMENTS if ((a.edge == x[0]) & (a.element == x[1])).any()]
    t = (a[a.element != per.NAMES[pl.TIE]].groupby(['edge', 'element']).n_bins.sum()
         .sort_values(ascending=False))
    return sorted(t.index[:9], key=lambda x: (x[0] != 'outer', -t[x]))


def page_elements(pdf, a, band, title):
    a = a[a.band == band]
    els = _elements_for(a)
    shape = (2, 3) if len(MASKS) > 1 else (3, 3)
    fig, axes = plt.subplots(*shape, figsize=(16, 4.75 * shape[0]), squeeze=False)
    for ax, (edge, name) in zip(axes.ravel(), els):
        g = a[(a.edge == edge) & (a.element == name)]
        for (side, mask), h in g.groupby(['side', 'mask']):
            ax.plot(h.rtp_deg, h.dmm_pix, 'o' if mask == BASE else 's',
                    color='tab:blue' if side == 'intra' else 'tab:red',
                    mfc=None if mask == BASE else 'none', ms=6,
                    label=f'{side}, {mask}, {int(h.n_bins.sum())} bins')
            for r in h.itertuples():
                ax.annotate(r.pair[4], (r.rtp_deg, r.dmm_pix), fontsize=6, xytext=(3, 3),
                            textcoords='offset points')
        ax.axhline(0, color='0.5', lw=0.5)
        ax.set_title(f'{edge} edge set by {name}', fontsize=9)
        ax.set_xlabel('physical_rotator_angle (deg)')
        ax.set_ylabel('median data − model edge (pixels)')
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(els):]:
        ax.axis('off')
    fig.suptitle(title + '. One point per visit: the bin-weighted median over the psi ranges '
                 'the element sets; label = pair number', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def spider_table(info):
    """Per science side: spider translation and rotation, inner-edge dy, vane width."""
    import night2_report as nr
    if BASE == 'v4.02g':
        parts = [nr.night1()]
        parts += [pd.read_parquet(OUT / run / 'per_side.parquet') for run in (N7, N8)]
    else:  # v4speed_fits.py per_side
        parts = [pd.read_parquet(OUT / run / PAIRS[0].fit[BASE] / 'per_side.parquet')
                 for run in (N1, N7, N8)]
    t = pd.concat(parts, ignore_index=True)
    t = t[(t.sensor_class == 'science') & t.visit.isin(info.visit)]
    return t.merge(info[['visit', 'pair', 'cam_dx_um', 'cam_dy_um']], on='visit')


def rotator_decomposition(t, side):
    """Huber fit [dx, dy]_OCS = a + R(rtp) b per side (pixels): a fixed in OCS, b in CCS."""
    import statsmodels.api as sm
    g = t[t.side == side].dropna(subset=['dx_pix', 'dy_pix', 'rtp_deg'])
    r = np.radians(g.rtp_deg.to_numpy())
    c, s = np.cos(r), np.sin(r)
    one, zero = np.ones_like(c), np.zeros_like(c)
    y = np.concatenate([g.dx_pix.to_numpy(), g.dy_pix.to_numpy()])
    out = {'n_sides': len(g)}
    designs = {'both': np.vstack([np.column_stack([one, zero, c, -s]),
                                  np.column_stack([zero, one, s, c])]),
               'ocs': np.vstack([np.column_stack([one, zero]), np.column_stack([zero, one])]),
               'ccs': np.vstack([np.column_stack([c, -s]), np.column_stack([s, c])])}
    for name, X in designs.items():
        f = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        out[name] = (f.params, f.bse, float(np.median(np.abs(f.resid))))
    return out


def _curve(params, model, rtp_deg):
    r = np.radians(rtp_deg)
    c, s = np.cos(r), np.sin(r)
    if model == 'ocs':
        return np.full_like(r, params[0]), np.full_like(r, params[1])
    if model == 'ccs':
        return params[0] * c - params[1] * s, params[0] * s + params[1] * c
    return (params[0] + params[2] * c - params[3] * s, params[1] + params[2] * s + params[3] * c)


def page_spider(pdf, t):
    import night2_report as nr
    v = []
    for (visit, side), g in t.groupby(['visit', 'side']):
        r = dict(visit=visit, side=side, pair=g.pair.iloc[0], rtp_deg=g.rtp_deg.median(),
                 cam_dy_um=g.cam_dy_um.iloc[0], cam_dx_um=g.cam_dx_um.iloc[0])
        for q in ('dx_pix', 'dy_pix', 'inner_dy_pix', 'delta_corr_deg', 'vane_width_mm'):
            s = nr.stat(g[q]) if q in g else dict(median=np.nan, err=np.nan, n=0)
            r[q], r[q + '_err'], r[q + '_n'] = s['median'], s['err'], s['n']
        v.append(r)
    v = pd.DataFrame(v)
    fits = {s: rotator_decomposition(t, s) for s in ('intra', 'extra')}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    grid = np.linspace(-5, 60, 200)
    panels = [('dx_pix', 'spider translation dx, OCS image (pixels)'),
              ('dy_pix', 'spider translation dy, OCS image (pixels)'),
              ('inner_dy_pix', 'inner-edge circle dy, data − model, OCS (pixels)'),
              ('delta_corr_deg', 'spider rotation δ, recovery-corrected (deg)'),
              ('vane_width_mm', 'free vane width, central band (mm in the pupil; CAD 50)')]
    colours = {'intra': 'tab:blue', 'extra': 'tab:red'}
    for ax, (q, lab) in zip(axes.ravel(), panels):
        for side, h in v.groupby('side'):
            ax.errorbar(h.rtp_deg, h[q], yerr=h[q + '_err'], fmt='o', color=colours[side],
                        ms=5, capsize=2, label=side)
            for r in h.itertuples():
                ax.annotate(r.pair[4], (r.rtp_deg, getattr(r, q)), fontsize=6, xytext=(3, 3),
                            textcoords='offset points')
        if q in ('dx_pix', 'dy_pix'):
            j = 0 if q == 'dx_pix' else 1
            for side in ('intra', 'extra'):
                for model, ls in (('ocs', '--'), ('ccs', ':')):
                    ax.plot(grid, _curve(fits[side][model][0], model, grid)[j], ls,
                            color=colours[side], lw=0.9,
                            label=f'{side}: fixed in {model.upper()}' if q == 'dx_pix' else None)
        ax.axhline(0, color='0.6', lw=0.4)
        ax.set_xlabel('physical_rotator_angle (deg)')
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(fontsize=6)
    ax = axes[1, 2]
    for side, h in v.groupby('side'):
        ax.errorbar(h.cam_dy_um / 1000, h.dy_pix, yerr=h.dy_pix_err, fmt='o', color=colours[side],
                    ms=5, capsize=2, label=side)
    ax.set_xlabel('camera dy Trim (mm)')
    ax.set_ylabel('spider translation dy, OCS image (pixels)', fontsize=8)
    ax.legend(fontsize=7)
    lines = []
    for side in ('intra', 'extra'):
        f = fits[side]
        pa, ea, ra = f['both']
        lines.append(f'{side} ({f["n_sides"]} sides): OCS-fixed a = ({pa[0]:+.2f} ± {ea[0]:.2f}, '
                     f'{pa[1]:+.2f} ± {ea[1]:.2f}), CCS-fixed b = ({pa[2]:+.2f} ± {ea[2]:.2f}, '
                     f'{pa[3]:+.2f} ± {ea[3]:.2f}) pixels; median |residual| OCS only '
                     f'{f["ocs"][2]:.2f}, CCS only {f["ccs"][2]:.2f}, both {ra:.2f} pixels')
    fig.suptitle('Spider shadow offsets against rotator angle, science sides, one point per visit '
                 '(median ± error of the median). A cause fixed to the telescope stays put in OCS '
                 '(dashed); one fixed to the camera turns with the rotator (dotted).\nHuber fit '
                 '[dx, dy]_OCS = a + R(rtp) b over the sides: ' + '\n'.join(lines), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)
    return v, fits


SIL = 'Camera silhouette'


def page_silhouette(pdf, t):
    """Silhouette inner-edge miss with v4.02s against camera decenter and rotator angle."""
    import night2_report as nr
    if 'sil_miss_pix' not in t:
        return
    v = []
    for (visit, side), g in t[t.field_r_deg.between(*OUTER_BAND)].groupby(['visit', 'side']):
        s = nr.stat(g.sil_miss_pix)
        v.append(dict(visit=visit, side=side, pair=g.pair.iloc[0], rtp_deg=g.rtp_deg.median(),
                      cam_dx_um=g.cam_dx_um.iloc[0], cam_dy_um=g.cam_dy_um.iloc[0],
                      miss=s['median'], err=s['err'], n=s['n']))
    v = pd.DataFrame(v)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (x, lab, sc_) in zip(axes, (('cam_dy_um', 'camera dy Trim (mm)', 1e-3),
                                        ('cam_dx_um', 'camera dx Trim (mm)', 1e-3),
                                        ('rtp_deg', 'physical_rotator_angle (deg)', 1.0))):
        for side, h in v.groupby('side'):
            ax.errorbar(h[x] * sc_, h.miss, yerr=h.err, fmt='o', capsize=2,
                        color='tab:blue' if side == 'intra' else 'tab:red', label=side)
            for r in h.itertuples():
                ax.annotate(r.pair[4], (getattr(r, x) * sc_, r.miss), fontsize=6, xytext=(3, 3),
                            textcoords='offset points')
        ax.axhline(0, color='0.6', lw=0.4)
        ax.set_xlabel(lab)
        ax.set_ylabel(f'{SIL} inner-edge data − model, {MASKS[-1]} (pixels)')
        ax.legend(fontsize=7)
    fig.suptitle(f'{SIL} inner-edge miss ({MASKS[-1]}), science sides 1.6-1.725 deg, psi '
                 f'{nr.PSI_RANGES}, one point per visit (median ± error); label = pair number',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def page_sample(pdf, info):
    """Field positions (CCS) of the fitted science donuts, one panel per visit."""
    colours = {'0.25-0.5 deg (used)': 'tab:orange', '1.6-1.725 deg (used)': 'red',
               'other fitted': '0.6'}
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(4.2 * len(PAIRS), 10.0), squeeze=False)
    rtp = info.set_index('visit').rtp_deg
    for j, p in enumerate(PAIRS):
        c, _, _ = source(p.run, p.fit[BASE])
        c = c[c.sensor_class == 'science'].copy()
        if 'visit' not in c:
            c['visit'] = c.side.map(p.visits())
        used = {w: set(zip(*pair_sides(p, BASE, w)[['key', 'side']].to_numpy().T))
                for w in (OUTER, INNER)}
        for i, side in enumerate(('intra', 'extra')):
            ax = axes[i, j]
            visit = p.visits()[side]
            g = c[(c.visit == visit) & (c.side == side)]
            n = {}
            for r in g.itertuples():
                k = (r.key, r.side)
                kind = ('1.6-1.725 deg (used)' if k in used[OUTER] else
                        '0.25-0.5 deg (used)' if k in used[INNER] else 'other fitted')
                n[kind] = n.get(kind, 0) + 1
                # donut outer radius in field angle: pixels × pixel scale
                rad = float(r.r_out_pix) * PIXEL_SCALE_ARCSEC / 3600.0
                ax.add_patch(Circle((r.field_x_ccs_deg, r.field_y_ccs_deg), rad,
                                    color=colours[kind], alpha=0.8, lw=0))
            for rr in (1.6, 1.725):
                ax.add_patch(Circle((0, 0), rr, fill=False, lw=0.4, color='k'))
            ax.set_xlim(-1.85, 1.85)
            ax.set_ylim(-1.85, 1.85)
            ax.set_aspect('equal')
            ax.set_xlabel('thx, CCS (deg)', fontsize=8)
            ax.set_ylabel('thy, CCS (deg)', fontsize=8)
            ax.tick_params(labelsize=7)
            counts = ', '.join(f'{v} {k.split(" (")[0]}' for k, v in sorted(n.items()))
            ax.set_title(f'{p.name.split("_")[0]} visit {visit}, {side}\nrotator '
                         f'{rtp.get(visit, np.nan):.1f} deg; {len(g)} donuts: {counts}',
                         fontsize=7)
    h = [Patch(color=v, label=k) for k, v in colours.items()]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f'Giant-donut sample, {BASE} fits (fit_ok), science sensors: field position in '
                 'CCS, marker = donut outer radius; circles at 1.6 and 1.725 deg field radius',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
    pdf.savefig(fig)
    plt.close(fig)


def summary_pdf(path, info, elements):
    with PdfPages(path) as pdf:
        if len(MASKS) == 1:
            page_sample(pdf, info)
        cols = ['pair', 'side', 'visit', 'rtp_deg', 'band', 'exp_time_s', 'altitude_deg',
                'm2_dz_um', 'cam_dz_um', 'm2_dx_um', 'm2_dy_um', 'cam_dx_um', 'cam_dy_um']
        txt = info[cols].round(1).to_string(index=False)
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.axis('off')
        ax.text(0, 1, 'Pairs: intra and extra exposures whose absolute Trim differs only in '
                'camera z and M2 z (value_added visit_telemetry). Rotator angle '
                'physical_rotator_angle (deg); Trim in µm.\n\n' + txt, family='monospace',
                fontsize=8, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        a = _element_rows(elements, info)
        a.to_csv(path.with_name('elements_by_visit.csv'), index=False, float_format='%.4g')
        masks = (' and '.join(f'{m} ({"filled" if m == BASE else "open"})' for m in MASKS)
                 if len(MASKS) > 1 else BASE)
        page_elements(pdf, a, '1.6-1.725', 'Median data − model edge per mask element, science '
                      f'sides 1.6-1.725 deg, {masks}')
        page_elements(pdf, a, '0.25-0.5', 'Median data − model edge per mask element, science '
                      f'sides 0.25-0.5 deg, {BASE}')
        t = spider_table(info)
        v, fits = page_spider(pdf, t)
        v.to_csv(path.with_name('spider_by_visit.csv'), index=False, float_format='%.4g')
        page_silhouette(pdf, t)
    print('wrote', path, flush=True)


# ---------------------------------------------------------------- conclusions.pdf

S = 'summary.pdf'
CONCLUSIONS = [
    ('Settled: v4.02 fits better than v1000',
     'Residual RMS is lower with v4.02 on 16 of 16 clean sides; median ratio v4.02 / v1000 0.94 '
     '(dimensionless). Most of the gain is the spider, which v1000 places 50 mm off the CAD shadow.',
     [('20261018_fits/giant_donut_pupil_results.pdf', [1, 2]),
      ('20261018_spider_validation/spider_validation.pdf', [1])]),
    ('Settled: no wavefront trend inside 1.725 deg; the outer-band Z5-Z7 is pupil bias',
     'Inside 1.725 deg the fitted Z5-Z8 deviations stay within ±3 µm of wavefront, with no trend '
     'in field. The 1-3.9 µm Z5-Z7 of the outer band is pupil bias: the central-band medians cost '
     'only 6 % in chi2/dof there.',
     [('20261018_fits/giant_donut_pupil_results.pdf', [5, 6]),
      ('20261021_pupil4/pupil_pass4.pdf', [10, 11, 12])]),
    ('Settled: the +3.3 pixel edge-centre shift looks like coma but is not',
     'The inner-minus-outer edge-centre shift reads as Z7 +0.7 to +0.8 µm, but whole-donut fits '
     'with Z7 free find +0.02 to +0.24 µm, and fit d (edges only) fits the donut worse.',
     [('20261019_pupil2/giant_donut_pupil_pass2.pdf', [1, 3]),
      ('20261020_pupil3/pupil_pass3.pdf', [105, 106, 107])]),
    ('Settled: r_inner / r_outer is 0.0104 above the model; diffraction gives about half',
     'Median over science donuts 0.0104 (dimensionless; nMAD 0.0048), flat in field. Batoid '
     'Huygens sims move the edges by about half the data miss, with the same signs.',
     [('20261020_pupil3/pupil_pass3.pdf', [101, 102]),
      ('20261023_pupil6/pupil_pass6.pdf', [44, 45])]),
    ('Settled: the v4.02s camera silhouette matches Batoid',
     'danish − Batoid inner edge 0.0 pixels median in silhouette-set bins (worst −0.52), '
     'against −1.76 to −5.26 pixels with v4.02g.',
     [('20261023_pupil6/pupil_pass6.pdf', [39, 40])]),
    ('Settled: the spider offset is a translation, not a rotation',
     'dy −1.65 / −2.07 pixels OCS image (extra / intra) removes the m = 1 term. It is not danish '
     'bias (sims), the TEA (centred to 1.5-2 mm), the detector, ISR or the rotator.',
     [('20261022_pupil5/pupil_pass5.pdf', [54, 55, 56]),
      ('20261023_pupil6/pupil_pass6.pdf', [3, 4, 5, 7, 11])]),
    ('Settled: the vane shadows are wider than opaque 50 mm vanes',
     'Free vane width 102.8 / 61.3 mm (extra / intra) against CAD 50 mm; sims return 49.7 / '
     '49.9 mm. A +0.10 to +0.15 µm phase strip along the vanes gives the extra / intra difference.',
     [('20261023_pupil6/pupil_pass6.pdf', [42, 43, 45]), (S, [4])]),
    ('Settled: the outer edges repeat across nights and pairs',
     'Both clear apertures (M1Baffle1, M3 outer) sit 0.2-0.7 pixels inside the model on both '
     'sides of focus, on every pair; diffraction gives −0.37 / −0.28 pixels of it.',
     [(S, [2, 3])]),
    ('Settled: M1 inner and M3 outer rims',
     'Where M1 inner sets the edge the data sit +0.9 to +1.4 pixels outside the model '
     'extra-focal and −0.3 to −0.4 inside intra-focal, the signs of the M1 roll-off. Where M3 '
     'outer sets the edge there is an intra-focal rim (1.33-1.38 over the plateau, '
     'dimensionless) and none extra-focal: a turned-up edge.',
     [('20261118_rim_analysis/rim_analysis.pdf', [1, 2, 3, 4, 5])]),
    ('Open: the cause of the common spider translation and inner-edge shift',
     'Spider dy −1.65 / −2.07 pixels and inner-edge dy +2.1 / +1.8 pixels OCS image on night 1. '
     'The rotator-angle test on the pairs separates a telescope-fixed from a camera-fixed cause.',
     [(S, [4]), ('20261023_pupil6/pupil_pass6.pdf', [46])]),
    ('Open: the camera silhouette misses with v4.02s',
     'Intra +1.93 ± 0.39 pixels on night 1. Extra-focal the miss may grow with the camera dy '
     'decenter (−0.65, +1.25, +2.18 pixels at 0, −2.1, −2.2 to −2.8 mm).',
     [('20261023_pupil6/pupil_pass6.pdf', [39, 41]), (S, [5])]),
    ('Open: M2 inner intra-focal +1 to +2 pixels on every night',
     'The M2 inner edge sits +1 to +2 pixels outside the model intra-focal and near zero '
     'extra-focal: a static pupil error would not change sign with focus.',
     [(S, [2])]),
    ('Open: TMASection and M1Baffle1 edges the mask lacks',
     'danish reproduces its own Batoid sims to the noise below 1.4 deg, but chi2/dof 2.1-2.8 '
     'remains at 1.69-1.70 deg on rays first clipped by TMASection and M1Baffle1.',
     [('20261023_pupil6/pupil_pass6.pdf', [12, 13])]),
    ('Open: the residual spider rotation',
     'δ −0.026 / +0.087 deg (extra / intra) with physical_rotator_angle, after the sim recovery '
     'correction; intra − extra 0.11 deg is unexplained.',
     [('20261023_pupil6/pupil_pass6.pdf', [4, 5]), (S, [4])]),
]


# The v4_speed rerun (run 20261122_pairs_review_v4speed); chi2/dof numbers are from the
# matched-side comparison in the study Log of 2026-11-22.
CONCLUSIONS_V4SPEED = [
    ('Settled: v4_speed fits as well as v4.02s, no better',
     'Fit c chi2/dof, median over matched science sides, v4_speed / v4.02s: 0.9997 (night 1, '
     '173 sides), 1.0004 (20261107, 143), 1.0008 (20261108, 461) (dimensionless; dof about '
     '7.3e5 per side). Against v4.02g 0.996-0.999, lower on 59-71 % of sides. No field band '
     'differs: 0.999-1.003 from 0 to 1.725 deg.',
     [(S, [1, 2])]),
    ('Settled: the outer edges still sit inside the model',
     'M1Baffle1 −0.40 to −0.84 pixels at 1.6-1.725 deg and −0.42 to −0.81 at 0.25-0.5 deg; '
     'M3 outer +0.08 to −0.62 pixels; every pair, both sides of focus. Within 0.3 pixels of '
     'v4.02s.',
     [(S, [3, 4])]),
    ('Settled: near the field centre the inner edge sits outside the model',
     'M1 inner at 0.25-0.5 deg: +0.51 to +0.81 pixels extra, +0.31 to +0.59 intra, every pair '
     '(v4.02g +0.35 to +0.77). The r_inner / r_outer excess holds with v4_speed.',
     [(S, [4])]),
    ('Open: the CameraBody miss follows the camera dy decenter, as with v4.02s',
     'Inner edge set by CameraBody, 1.6-1.725 deg: extra −0.36 pixels at camera dy 0 mm, '
     '+1.33 at −2.06 mm, +2.32 at −2.23 mm, +2.30 and +2.72 at −2.82 mm; intra +1.52 at 0 mm, '
     '−0.56 to 0.00 on night 3. Against v4.02s: 0.39-0.56 pixels '
     'lower intra, 0.08-0.33 higher extra. The mask assumes a centred camera.',
     [(S, [3, 6])]),
    ('Open: the M1 inner and M2 inner signs flip with focus on night 1',
     '1.6-1.725 deg, night 1; night 3 range (pixels): M1 inner extra +1.30; +0.16 to +0.46, '
     'intra −0.25; +0.32 to +0.52. M2 inner intra +1.68; +0.71 to +1.04, extra −0.59; +0.04 '
     'to +0.43. Same pattern as v4.02s, within 0.13 (M1) and 0.46 (M2) pixels.',
     [(S, [3])]),
    ('Open: the spider translation is unchanged',
     'dy −1.76 / −1.95 pixels OCS image (extra / intra) on night 1, −0.58 to −1.67 on night 3; '
     'within 0.12 pixels of v4.02. Huber fit [dx, dy]_OCS = a + R(rtp) b: median |residual| '
     'telescope-fixed 0.33 / 0.36, camera-fixed 0.39 / 0.27, both 0.33 / 0.23 pixels (intra / '
     'extra), against per-visit errors 0.03-0.07. Neither cause fits.',
     [(S, [5])]),
    ('Settled: the vane widths are unchanged',
     'Free vane width, central band: extra 103.1 mm (night 1), 107.2-107.5 mm, and two night 3 '
     'visits at the 120 mm grid edge; intra 61.2 and 63.5-67.6 mm. CAD 50 mm. Within 0.4 mm '
     'of v4.02.',
     [(S, [5])]),
    ('Open: the outer-band chi2/dof excess remains',
     'Median chi2/dof 5.9 at 1.68-1.725 deg against 3.0 at 0-0.8 deg, with v4_speed and v4.02s '
     'alike (ratio 1.003). The v4_speed telescope elements (LowerMidLevel, LowerTopEnd, '
     'UpperTopEnd, M1Baffle2) set no edge bin in the science sides out to 1.725 deg.',
     [(S, [3])]),
]


def conclusions_pdf(path, out):
    import fitz
    doc = fitz.open()
    srcs = {}
    for title, text, refs in (CONCLUSIONS if len(MASKS) > 1 else CONCLUSIONS_V4SPEED):
        page = doc.new_page(width=792, height=612)
        src = '\n'.join(f'{f} p{",".join(map(str, pp))}' for f, pp in refs)
        if len(MASKS) == 1:
            page.insert_textbox(fitz.Rect(54, 30, 738, 56), "Draft for Aaron's review",
                                fontsize=10, color=(0.8, 0, 0))
        page.insert_textbox(fitz.Rect(54, 60, 738, 160), title, fontsize=18)
        page.insert_textbox(fitz.Rect(54, 170, 738, 420), text, fontsize=12)
        page.insert_textbox(fitz.Rect(54, 440, 738, 580), 'Supporting pages that follow:\n' + src,
                            fontsize=9)
        for f, pp in refs:
            if f not in srcs:
                srcs[f] = fitz.open(out / S if f == S else OUT / f)
            for k in pp:
                doc.insert_pdf(srcs[f], from_page=k - 1, to_page=k - 1)
    # pages copied from one source share its fonts and images: write them once
    doc.save(path, garbage=4, deflate=True)
    print('wrote', path, flush=True)


README = """# Pairs review, {label}

> **Status:** built {date} by `code/pairs_review.py` · plan: `pairs_review_plan.md`

- `conclusions.pdf`: each study conclusion, then the pages that support it.
- `summary.pdf`: the pairs table; median data − model edge per mask element against rotator
  angle (1.6-1.725 deg, 0.25-0.5 deg); spider offsets against rotator angle; the camera
  silhouette miss against camera decenter.
- `pair<N>_<day_obs>_<intra>i_<extra>e/`:
  - `focal_plane.pdf`: per visit the binned ISR mosaic, a count page, one page per raft;
    donuts used circled by band, other selected donuts grey (DVCS).
  - `fit_gallery.pdf`: data, fit c model and residual per side; 1.6-1.725 deg{gallery}.
  - `edges.pdf`: the psi edge pages and element tables per band, mask and side;
    `elements.csv` the tables.
- `models/`: the rendered fit c models the galleries use.
"""


def step_pdfs(out, pairs):
    import datetime
    from lsst.obs.lsst import LsstCam
    camera = LsstCam.getCamera()
    info = exposure_info()
    info.to_csv(out / 'pairs.csv', index=False, float_format='%.4g')
    text = _info_text(info)
    els = []
    for p in PAIRS:
        if pairs and p.name.split('_')[0] not in pairs:
            continue
        d = out / p.name
        d.mkdir(parents=True, exist_ok=True)
        els.append(edges_pdf(p, d / 'edges.pdf', text))
        fit_gallery_pdf(p, d / 'fit_gallery.pdf', out)
        focal_plane_pdf(p, d / 'focal_plane.pdf', camera, text)
    if not pairs:
        summary_pdf(out / 'summary.pdf', info, pd.concat(els, ignore_index=True))
        if len(MASKS) > 1 or CONCLUSIONS_V4SPEED:
            conclusions_pdf(out / 'conclusions.pdf', out)
        gallery = (' (and corner 1.4-1.75 deg) with v4.02g and v4.02s side by side, then '
                   '0.25-0.5 deg with v4.02g' if CORNERS else f', then 0.25-0.5 deg, {BASE}')
        readme = README.format(date=datetime.date.today().isoformat(), label=LABEL,
                               gallery=gallery)
        if len(MASKS) == 1:
            readme = readme.replace('- `summary.pdf`: the pairs table;',
                                    '- `summary.pdf`: the donut sample in CCS field angle; the '
                                    'pairs table;').replace('camera\n  silhouette', 'CameraBody')
        (out / 'README.md').write_text(readme)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--steps', default='models,pdfs')
    p.add_argument('--out', default=None, help='existing pairs_review run under output/')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--pairs', default=None, help='e.g. pair0,pair2: only these, no summaries')
    p.add_argument('--mask-set', default='v402', choices=sorted(MASK_SETS))
    a = p.parse_args()
    configure(a.mask_set)
    if a.out:
        out = OUT / a.out
    else:
        from rubinwork.common import study_run
        out = study_run(STUDY, 'pairs_review' + ('' if a.mask_set == 'v402' else
                                                 f'_{a.mask_set}')).path
    out.mkdir(parents=True, exist_ok=True)
    steps = a.steps.split(',')
    if 'models' in steps:
        step_models(out, a.workers)
    if 'pdfs' in steps:
        step_pdfs(out, a.pairs.split(',') if a.pairs else None)


if __name__ == '__main__':
    main()
