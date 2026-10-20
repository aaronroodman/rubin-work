"""Hand labelling of giant donuts, the truth set for the v2 selection.

One page per (visit, raft): the background-subtracted raft image from the 4x4 ISR
cache, with markers for every the_monster star at 7-14.5 AB mag (by magnitude bin)
and every peak no catalog star explains. Clicking within 0.3 r_out of a marker or
an earlier click cycles its label::

    unlabelled -> good -> marginal -> blended -> not_donut -> unlabelled

Clicking elsewhere adds a donut the algorithm missed, labelled ``good``; cycling an
added donut back to unlabelled removes it. Every click rewrites
``hand_labels.csv``, so a session can stop at any time and resume.

A page marked done means every donut on it is labelled: unlabelled markers there
are then not donuts worth scoring and missed donuts have been added.

Labels:

- ``good``: isolated, whole ring clean, usable for a pupil fit;
- ``marginal``: usable, but a neighbour's halo, an edge or a faint overlap touches it;
- ``blended``: overlapped by another donut, not usable;
- ``not_donut``: no donut there (a catalog star with no visible ring, a spurious peak).
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd

import candidates as cd
import fp_mosaic
import select_candidates as sc
import selection_pages as pages

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from lsst.afw.cameraGeom import FOCAL_PLANE

__all__ = ['LABELS', 'PAGES', 'Labeller']

LABELS = ('unlabelled', 'good', 'marginal', 'blended', 'not_donut')
LABEL_COLOURS = {'good': '#1b9e77', 'marginal': '#e6ab02', 'blended': '#d95f02',
                 'not_donut': '#e41a1c'}
RAFTS = ('R22', 'R31', 'R13', 'R10', 'R40', 'R04')
PAGES = [(visit, raft) for visit in (2025102300337, 2025102300340) for raft in RAFTS]
CLICK_ROUT = 0.3
COLUMNS = ['visit', 'raft', 'detector', 'item', 'refcat_id', 'peak_id', 'fp_x_mm', 'fp_y_mm',
           'x_pix', 'y_pix', 'mag_best', 'label', 'added', 'updated']


class Labeller:
    """State and drawing of the labelling session.

    Parameters
    ----------
    diag_dir : `pathlib.Path`
        The ``selection_diag`` run: ``stars.parquet``, ``peaks.parquet``; labels
        are written here.
    study_dir : `pathlib.Path`
        The study directory, for the ISR caches.
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.
    """

    def __init__(self, diag_dir, study_dir, camera):
        import selection_v2 as sv
        self.diag_dir = pathlib.Path(diag_dir)
        self.study_dir = pathlib.Path(study_dir)
        self.camera = camera
        self.caches = sv.CACHES
        self.stars = pd.read_parquet(self.diag_dir / 'stars.parquet')
        self.peaks = pd.read_parquet(self.diag_dir / 'peaks.parquet')
        self.label_path = self.diag_dir / 'hand_labels.csv'
        self.done_path = self.diag_dir / 'hand_labels_pages.json'
        if self.label_path.exists():
            self.labels = pd.read_csv(self.label_path)
        else:
            self.labels = pd.DataFrame(columns=COLUMNS)
        self.done = json.loads(self.done_path.read_text()) if self.done_path.exists() else {}
        self.stretch_e = pages.STRETCH_E
        self.page = None

    # -- items -----------------------------------------------------------------
    def items(self, visit, raft):
        """Markers of one page: stars, then peaks no star explains."""
        s = self.stars[(self.stars.visit == visit) & self.stars.detector.str.startswith(raft)
                       & self.stars.mag_best.between(sc.MAG_MIN, sc.MAG_MAX)]
        p = self.peaks[(self.peaks.visit == visit) & self.peaks.detector.str.startswith(raft)
                       & self.peaks.unmatched & (self.peaks.pass_rel | self.peaks.pass_sigma)]
        rows = [dict(item='star', refcat_id=int(r.refcat_id), peak_id=-1, detector=r.detector,
                     fp_x_mm=r.fp_x_mm, fp_y_mm=r.fp_y_mm, x_pix=r.x_pix, y_pix=r.y_pix,
                     mag_best=r.mag_best, r_out_mm=r.r_out_pix * 0.01) for r in s.itertuples()]
        rows += [dict(item='peak', refcat_id=-1, peak_id=int(r.peak_id), detector=r.detector,
                      fp_x_mm=r.fp_x_mm, fp_y_mm=r.fp_y_mm, x_pix=r.x_pix, y_pix=r.y_pix,
                      mag_best=np.nan, r_out_mm=r.r_out_pix * 0.01) for r in p.itertuples()]
        return pd.DataFrame(rows)

    def page_labels(self, visit, raft):
        return self.labels[(self.labels.visit == visit) & (self.labels.raft == raft)]

    def _key(self, row):
        if row['item'] == 'star':
            return ('star', int(row['refcat_id']))
        if row['item'] == 'peak':
            return ('peak', int(row['peak_id']))
        return ('added', round(float(row['fp_x_mm']), 3), round(float(row['fp_y_mm']), 3))

    def label_of(self, visit, raft, row):
        lab = self.page_labels(visit, raft)
        key = self._key(row)
        for r in lab.itertuples():
            if self._key(r._asdict()) == key:
                return r.label
        return 'unlabelled'

    def set_label(self, visit, raft, row, label):
        """Set one item's label and rewrite the CSV."""
        key = self._key(row)
        keep = [self._key(r._asdict()) != key or r.visit != visit
                for r in self.labels.itertuples()]
        self.labels = self.labels[keep]
        if label != 'unlabelled':
            new = {c: row.get(c, np.nan) for c in COLUMNS}
            new.update(visit=visit, raft=raft, label=label, added=row['item'] == 'added',
                       updated=time.strftime('%Y-%m-%dT%H:%M:%S'))
            self.labels = pd.concat([self.labels, pd.DataFrame([new])], ignore_index=True)
        self.labels.to_csv(self.label_path, index=False, float_format='%.3f')

    def set_done(self, visit, raft, done):
        self.done[f'{visit}_{raft}'] = bool(done)
        self.done_path.write_text(json.dumps(self.done, indent=1))

    def is_done(self, visit, raft):
        return self.done.get(f'{visit}_{raft}', False)

    # -- clicks ----------------------------------------------------------------
    def click(self, visit, raft, x_mm, y_mm):
        """Cycle the label nearest a click, or add a missed donut.

        Returns
        -------
        message : `str`
        """
        items = self.items(visit, raft)
        added = self.page_labels(visit, raft)
        added = added[added['added'].astype(bool)].copy()
        cands = []
        r_out = float(items.r_out_mm.median()) if len(items) else 3.4
        for _, r in items.iterrows():
            cands.append((np.hypot(r.fp_x_mm - x_mm, r.fp_y_mm - y_mm) / r.r_out_mm, r.to_dict()))
        for _, r in added.iterrows():
            d = dict(r)
            d['item'] = 'added'
            cands.append((np.hypot(r.fp_x_mm - x_mm, r.fp_y_mm - y_mm) / r_out, d))
        best = min(cands, key=lambda c: c[0]) if cands else (np.inf, None)
        if best[0] < CLICK_ROUT:
            row = best[1]
            current = self.label_of(visit, raft, row)
            label = LABELS[(LABELS.index(current) + 1) % len(LABELS)]
            self.set_label(visit, raft, row, label)
            name = row['refcat_id'] if row['item'] == 'star' else row['item']
            return f'{row["item"]} {name}: {current} -> {label}'
        detector = self.detector_at(raft, x_mm, y_mm)
        if detector is None:
            return 'click off silicon: nothing added'
        px, py = cd.focal_to_pixels(self.camera[detector], [x_mm], [y_mm])
        row = dict(item='added', refcat_id=-1, peak_id=-1, detector=detector, fp_x_mm=x_mm,
                   fp_y_mm=y_mm, x_pix=float(px[0]), y_pix=float(py[0]), mag_best=np.nan)
        self.set_label(visit, raft, row, 'good')
        return f'added a missed donut on {detector} at ({x_mm:.2f}, {y_mm:.2f}) mm: good'

    def detector_at(self, raft, x_mm, y_mm):
        for name in self.raft_names(raft):
            box = self.camera[name].getBBox()
            px, py = cd.focal_to_pixels(self.camera[name], [x_mm], [y_mm])
            if box.getMinX() <= px[0] <= box.getMaxX() and box.getMinY() <= py[0] <= box.getMaxY():
                return name
        return None

    # -- images ----------------------------------------------------------------
    def raft_names(self, raft):
        visit = PAGES[0][0]
        cache = self.study_dir / 'output' / self.caches[visit] / str(visit)
        return sorted(p.stem for p in cache.glob(f'{raft}_*.npy'))

    def raft_image(self, visit, raft):
        cache = self.study_dir / 'output' / self.caches[visit] / str(visit)
        sub = {}
        for name in self.raft_names(raft):
            image = np.load(cache / f'{name}.npy').astype(float)
            sub[name] = image - cd.background_map(image, block=128, percentile=10.0)
        return fp_mosaic.make_mosaic(self.camera, sub, cd.CACHE_BIN, sc.RAFT_BIN, buffer_size=4)

    # -- drawing ---------------------------------------------------------------
    def draw(self, ax, visit, raft, mosaic=None, geometry=None):
        """Draw one page on ``ax``; returns the artists the next redraw clears."""
        if mosaic is not None:
            ax.clear()
            ny, nx = mosaic.shape
            x0, y0 = geometry.to_mm(-0.5, -0.5)
            x1, y1 = geometry.to_mm(nx - 0.5, ny - 0.5)
            self._image = ax.imshow(np.arcsinh(mosaic / self.stretch_e), origin='lower', cmap='gray',
                                    extent=(x0, x1, y0, y1), vmin=-0.5, vmax=np.arcsinh(20.0),
                                    interpolation='none')
            self._mosaic = mosaic
            pages.draw_ccd_edges(ax, self.camera, self.raft_names(raft), lw=0.8)
            for n in self.raft_names(raft):
                c = self.camera[n].getCenter(FOCAL_PLANE)
                ax.text(c.getX(), c.getY(), n[4:], color='cyan', fontsize=8, alpha=0.5, ha='center')
            ax.set_xlabel('focal-plane x, DVCS (DM view) [mm] = CCS y')
            ax.set_ylabel('focal-plane y, DVCS (DM view) [mm] = CCS x')
        for artist in getattr(self, '_overlay', []):
            artist.remove()
        self._overlay = []
        items = self.items(visit, raft)
        for lo, hi, marker, colour in pages.MAG_BIN_STYLE:
            sel = items[(items.item == 'star') & items.mag_best.between(lo, hi)]
            self._overlay.append(ax.scatter(sel.fp_x_mm, sel.fp_y_mm, s=40, marker=marker,
                                            facecolors='none', edgecolors=colour, lw=0.8,
                                            label=f'star {lo:g}-{hi:g} mag: {len(sel)}'))
        pk = items[items.item == 'peak']
        self._overlay.append(ax.scatter(pk.fp_x_mm, pk.fp_y_mm, s=50, marker='x', c='red', lw=1.0,
                                        label=f'peak, no star within 0.5 r_out: {len(pk)}'))
        lab = self.page_labels(visit, raft)
        r_out = float(items.r_out_mm.median()) if len(items) else 3.4
        r_by_key = {self._key(r): r['r_out_mm'] for _, r in items.iterrows()}
        for _, r in lab.iterrows():
            d = dict(r)
            d['item'] = 'added' if bool(r['added']) else r['item']
            radius = r_by_key.get(self._key(d), r_out)
            patch = Circle((r.fp_x_mm, r.fp_y_mm), radius, fill=False, lw=1.6,
                           color=LABEL_COLOURS[r.label], ls='--' if bool(r['added']) else '-')
            ax.add_patch(patch)
            self._overlay.append(patch)
        counts = lab.label.value_counts().to_dict()
        for name, colour in LABEL_COLOURS.items():
            self._overlay.append(ax.plot([], [], 'o', mfc='none', color=colour,
                                         label=f'{name}: {counts.get(name, 0)}')[0])
        self._overlay.append(ax.legend(loc='upper right', fontsize=7, framealpha=0.7))
        n_lab = len(lab)
        done = ' [done]' if self.is_done(visit, raft) else ''
        ax.set_title(f'{visit} ({"extra" if visit % 10 in (7, 8) else "intra"}-focal) raft '
                     f'{raft}: {len(items)} markers, {n_lab} labelled{done}', fontsize=10)

    def set_stretch(self, stretch_e):
        self.stretch_e = stretch_e
        if getattr(self, '_image', None) is not None:
            self._image.set_data(np.arcsinh(self._mosaic / stretch_e))


def make_widget(labeller, figsize=(10, 10)):
    """The ipywidgets/ipympl labelling interface. Needs ``%matplotlib widget``."""
    import ipywidgets as w
    names = [f'{v} {r}' for v, r in PAGES]
    page_box = w.Dropdown(options=list(range(len(PAGES))), value=0,
                          description='page')
    page_box.options = [(n, i) for i, n in enumerate(names)]
    prev_b, next_b = w.Button(description='< prev'), w.Button(description='next >')
    done_b = w.Checkbox(description='page done (every donut labelled)')
    stretch = w.FloatLogSlider(value=labeller.stretch_e, base=10, min=1, max=4, step=0.05,
                               description='stretch e-')
    status = w.HTML()
    with plt.ioff():
        fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.header_visible = False
    state = {}

    def show(i):
        visit, raft = PAGES[i]
        mosaic, geometry = labeller.raft_image(visit, raft)
        labeller.draw(ax, visit, raft, mosaic, geometry)
        state['page'] = (visit, raft)
        done_b.value = labeller.is_done(visit, raft)
        fig.canvas.draw_idle()
        status.value = f'{visit} {raft}: click a marker to cycle its label; click empty sky ' \
                       f'on a donut to add it. Labels in {labeller.label_path}'

    def on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        toolbar = getattr(fig.canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', '') not in ('', None):
            return
        visit, raft = state['page']
        status.value = labeller.click(visit, raft, event.xdata, event.ydata)
        labeller.draw(ax, visit, raft)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    page_box.observe(lambda ch: show(ch['new']), names='value')
    prev_b.on_click(lambda _: setattr(page_box, 'value', max(0, page_box.value - 1)))
    next_b.on_click(lambda _: setattr(page_box, 'value', min(len(PAGES) - 1, page_box.value + 1)))

    def on_done(ch):
        if 'page' in state:
            labeller.set_done(*state['page'], ch['new'])
            labeller.draw(ax, *state['page'])
            fig.canvas.draw_idle()
    done_b.observe(on_done, names='value')

    def on_stretch(ch):
        labeller.set_stretch(ch['new'])
        fig.canvas.draw_idle()
    stretch.observe(on_stretch, names='value')
    show(0)
    return w.VBox([w.HBox([prev_b, page_box, next_b, done_b]), stretch, status, fig.canvas])
