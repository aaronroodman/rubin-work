"""Hand review of the giant-donut selection, the truth set for scoring it.

One page per (visit, raft): the background-subtracted raft image from the 4x4 ISR
cache, with the CCD edges and two markers:

- ``^`` on every catalog star the current selection classes isolated
  (``iso_class == 'isolated'`` in ``stars.parquet``);
- ``v`` on every good peak (``good_peak`` in ``peak_metrics.parquet``, from
  ``peak_quality.py``): a ``threshold_rel = 0.3`` peak with its whole donut on
  silicon, not saturated, ``annulus_ratio < 0.05`` and ``overlap_frac < 0.02``.

A star and a good peak within 0.3 r_out are one donut, drawn with both markers
offset. Every marked donut starts unreviewed. Clicking within 0.3 r_out of one
toggles it to bad and back. Clicking empty sky adds a missed good donut; clicking
an added donut removes it. "Done: the rest are good" sets every unreviewed donut on
the page to good and marks the page done; on a done page a bad donut toggles back to
good. So only the bad and the missed donuts need a click.

Every click rewrites ``hand_labels_v2.csv`` and ``hand_labels_v2_pages.json`` in the
diag run, so a session can stop and resume. Unreviewed donuts are not written.
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

__all__ = ['PAGES', 'COLUMNS', 'Labeller', 'make_widget']

RAFTS = ('R22', 'R31', 'R13', 'R10', 'R40', 'R04')
PAGES = [(visit, raft) for visit in (2025102300337, 2025102300340) for raft in RAFTS]
MATCH_ROUT = 0.3
CLICK_ROUT = 0.3
# Offset of the ^ and v markers of a donut that has both, in r_out.
BOTH_OFFSET_ROUT = 0.15
LABEL_COLOURS = {'unreviewed': '#ffff33', 'good': '#1b9e77', 'bad': '#e41a1c',
                 'missed': '#00e5ff'}
COLUMNS = ['visit', 'raft', 'detector', 'x_pix', 'y_pix', 'fp_x_mm', 'fp_y_mm', 'refcat_id',
           'peak_id', 'star_isolated', 'good_peak', 'label', 'updated']


def match_marked(stars, peaks, match_rout=MATCH_ROUT):
    """Marked donuts of one page: isolated stars and good peaks, matched by position.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Star rows classed isolated (``stars.parquet`` columns).
    peaks : `pandas.DataFrame`
        Good-peak rows (``peak_metrics.parquet`` columns).
    match_rout : `float`, optional
        Match radius, in units of r_out (dimensionless).

    Returns
    -------
    marked : `pandas.DataFrame`
        One row per donut: ``key``, ``detector``, positions (star position where
        there is a star), ``refcat_id`` and ``peak_id`` (-1 where absent),
        ``star_isolated``, ``good_peak`` and ``r_out_mm``.
    """
    rows, used = [], set()
    p_xy = peaks[['fp_x_mm', 'fp_y_mm']].to_numpy()
    p_ids = peaks.peak_id.to_numpy()
    for s in stars.itertuples():
        r_mm = s.r_out_pix * 0.01
        peak_id = -1
        if len(p_xy):
            d = np.hypot(p_xy[:, 0] - s.fp_x_mm, p_xy[:, 1] - s.fp_y_mm)
            d[[i for i, pid in enumerate(p_ids) if pid in used]] = np.inf
            j = int(np.argmin(d))
            if d[j] < match_rout * r_mm:
                peak_id = int(p_ids[j])
                used.add(peak_id)
        rows.append(dict(key=f's{int(s.refcat_id)}', detector=s.detector, x_pix=s.x_pix,
                         y_pix=s.y_pix, fp_x_mm=s.fp_x_mm, fp_y_mm=s.fp_y_mm,
                         refcat_id=int(s.refcat_id), peak_id=peak_id, star_isolated=True,
                         good_peak=peak_id >= 0, r_out_mm=r_mm))
    for p in peaks.itertuples():
        if int(p.peak_id) in used:
            continue
        rows.append(dict(key=f'p{int(p.peak_id)}', detector=p.detector, x_pix=p.x_pix,
                         y_pix=p.y_pix, fp_x_mm=p.fp_x_mm, fp_y_mm=p.fp_y_mm,
                         refcat_id=int(p.match_refcat_id), peak_id=int(p.peak_id),
                         star_isolated=False, good_peak=True, r_out_mm=p.r_out_pix * 0.01))
    columns = ['key', 'detector', 'x_pix', 'y_pix', 'fp_x_mm', 'fp_y_mm', 'refcat_id',
               'peak_id', 'star_isolated', 'good_peak', 'r_out_mm']
    return pd.DataFrame(rows, columns=columns)


class Labeller:
    """State and drawing of the review session.

    Parameters
    ----------
    diag_dir : `pathlib.Path`
        The ``selection_diag`` run: ``stars.parquet``; labels are written here.
    metrics_dir : `pathlib.Path`
        The ``peak_quality.py`` run: ``peak_metrics.parquet``.
    study_dir : `pathlib.Path`
        The study directory, for the ISR caches.
    camera : `lsst.afw.cameraGeom.Camera`
        Camera geometry.
    """

    def __init__(self, diag_dir, metrics_dir, study_dir, camera):
        import selection_v2 as sv
        self.diag_dir = pathlib.Path(diag_dir)
        self.study_dir = pathlib.Path(study_dir)
        self.camera = camera
        self.caches = sv.CACHES
        self.stars = pd.read_parquet(self.diag_dir / 'stars.parquet')
        self.peaks = pd.read_parquet(pathlib.Path(metrics_dir) / 'peak_metrics.parquet')
        self.label_path = self.diag_dir / 'hand_labels_v2.csv'
        self.done_path = self.diag_dir / 'hand_labels_v2_pages.json'
        if self.label_path.exists():
            self.labels = pd.read_csv(self.label_path)
        else:
            self.labels = pd.DataFrame(columns=COLUMNS)
        self.done = json.loads(self.done_path.read_text()) if self.done_path.exists() else {}
        self.stretch_e = pages.STRETCH_E
        self._marked = {}

    # -- items -----------------------------------------------------------------
    def marked(self, visit, raft):
        """The marked donuts of a page (cached)."""
        if (visit, raft) not in self._marked:
            s = self.stars[(self.stars.visit == visit) & self.stars.detector.str.startswith(raft)
                           & (self.stars.iso_class == 'isolated')]
            p = self.peaks[(self.peaks.visit == visit) & self.peaks.detector.str.startswith(raft)
                           & self.peaks.good_peak]
            self._marked[(visit, raft)] = match_marked(s, p)
        return self._marked[(visit, raft)]

    def page_labels(self, visit, raft):
        return self.labels[(self.labels.visit == visit) & (self.labels.raft == raft)]

    @staticmethod
    def _label_key(r):
        if int(r['refcat_id']) >= 0 and bool(r['star_isolated']):
            return f's{int(r["refcat_id"])}'
        if int(r['peak_id']) >= 0:
            return f'p{int(r["peak_id"])}'
        return f'a{float(r["fp_x_mm"]):.3f}_{float(r["fp_y_mm"]):.3f}'

    def label_of(self, visit, raft, key):
        for _, r in self.page_labels(visit, raft).iterrows():
            if r['label'] != 'missed' and self._label_key(r) == key:
                return r['label']
        return 'unreviewed'

    def _write(self):
        self.labels.to_csv(self.label_path, index=False, float_format='%.3f')

    def _drop(self, visit, raft, key):
        keep = [not (r['visit'] == visit and r['raft'] == raft and self._label_key(r) == key)
                for _, r in self.labels.iterrows()]
        self.labels = self.labels[keep].reset_index(drop=True)

    def _add(self, visit, raft, row, label):
        new = {c: row.get(c, np.nan) for c in COLUMNS}
        new.update(visit=visit, raft=raft, label=label,
                   updated=time.strftime('%Y-%m-%dT%H:%M:%S'))
        self.labels = pd.concat([self.labels, pd.DataFrame([new], columns=COLUMNS)],
                                ignore_index=True)

    def set_label(self, visit, raft, item, label):
        """Set a marked donut's label (``unreviewed`` removes it) and rewrite the CSV."""
        self._drop(visit, raft, item['key'])
        if label != 'unreviewed':
            self._add(visit, raft, item, label)
        self._write()

    def is_done(self, visit, raft):
        return self.done.get(f'{visit}_{raft}', False)

    def finish_page(self, visit, raft):
        """Every unreviewed marked donut on the page to good; mark the page done."""
        n = 0
        for _, item in self.marked(visit, raft).iterrows():
            if self.label_of(visit, raft, item['key']) == 'unreviewed':
                self._add(visit, raft, item.to_dict(), 'good')
                n += 1
        self._write()
        self.done[f'{visit}_{raft}'] = True
        self.done_path.write_text(json.dumps(self.done, indent=1))
        return n

    # -- clicks ----------------------------------------------------------------
    def click(self, visit, raft, x_mm, y_mm):
        """Toggle the marked donut at a click, remove an added one, or add a missed one.

        Returns
        -------
        message : `str`
        """
        marked = self.marked(visit, raft)
        lab = self.page_labels(visit, raft)
        added = lab[lab.label == 'missed']
        r_default = float(marked.r_out_mm.median()) if len(marked) else 3.4
        best = (np.inf, None, None)
        for _, r in marked.iterrows():
            d = np.hypot(r.fp_x_mm - x_mm, r.fp_y_mm - y_mm) / r.r_out_mm
            if d < best[0]:
                best = (d, 'marked', r.to_dict())
        for _, r in added.iterrows():
            d = np.hypot(r.fp_x_mm - x_mm, r.fp_y_mm - y_mm) / r_default
            if d < best[0]:
                best = (d, 'added', r.to_dict())
        if best[0] < CLICK_ROUT:
            kind, item = best[1], best[2]
            if kind == 'added':
                self._drop(visit, raft, self._label_key(item))
                self._write()
                return f'removed the added donut at ({item["fp_x_mm"]:.2f}, {item["fp_y_mm"]:.2f}) mm'
            current = self.label_of(visit, raft, item['key'])
            if current == 'bad':
                label = 'good' if self.is_done(visit, raft) else 'unreviewed'
            else:
                label = 'bad'
            self.set_label(visit, raft, item, label)
            return f'{item["key"]} on {item["detector"]}: {current} -> {label}'
        detector = self.detector_at(raft, x_mm, y_mm)
        if detector is None:
            return 'click off silicon: nothing added'
        px, py = cd.focal_to_pixels(self.camera[detector], [x_mm], [y_mm])
        row = dict(detector=detector, x_pix=float(px[0]), y_pix=float(py[0]), fp_x_mm=x_mm,
                   fp_y_mm=y_mm, refcat_id=-1, peak_id=-1, star_isolated=False, good_peak=False)
        self._add(visit, raft, row, 'missed')
        self._write()
        return f'added a missed donut on {detector} at ({x_mm:.2f}, {y_mm:.2f}) mm'

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
        """Draw one page on ``ax``; the overlay is redrawn on every call."""
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
        marked = self.marked(visit, raft)
        labels = np.array([self.label_of(visit, raft, k) for k in marked.key], dtype=object)
        colours = [LABEL_COLOURS[lab] for lab in labels]
        both = (marked.star_isolated & marked.good_peak).to_numpy()
        dy = np.where(both, BOTH_OFFSET_ROUT * marked.r_out_mm, 0.0)
        star = marked.star_isolated.to_numpy()
        peak = marked.good_peak.to_numpy()
        if star.any():
            self._overlay.append(ax.scatter(
                marked.fp_x_mm[star], (marked.fp_y_mm + dy)[star], s=70, marker='^',
                c=[colours[i] for i in np.where(star)[0]], edgecolors='black', lw=0.5))
        if peak.any():
            self._overlay.append(ax.scatter(
                marked.fp_x_mm[peak], (marked.fp_y_mm - dy)[peak], s=70, marker='v',
                c=[colours[i] for i in np.where(peak)[0]], edgecolors='black', lw=0.5))
        lab = self.page_labels(visit, raft)
        added = lab[lab.label == 'missed']
        r_default = float(marked.r_out_mm.median()) if len(marked) else 3.4
        for _, r in added.iterrows():
            patch = Circle((r.fp_x_mm, r.fp_y_mm), r_default, fill=False, lw=1.4, ls='--',
                           color=LABEL_COLOURS['missed'])
            ax.add_patch(patch)
            self._overlay.append(patch)
        # Legend handles: marker meaning, then label colours with counts.
        n_star_only = int((star & ~peak).sum())
        n_peak_only = int((peak & ~star).sum())
        handles = [
            ax.scatter([], [], s=70, marker='^', c='white', edgecolors='black',
                       label=f'^ star classed isolated: {int(star.sum())}'),
            ax.scatter([], [], s=70, marker='v', c='white', edgecolors='black',
                       label=f'v good peak: {int(peak.sum())}'),
            ax.plot([], [], ' ', label=f'both {int(both.sum())}, ^ only {n_star_only}, '
                                       f'v only {n_peak_only}')[0]]
        counts = pd.Series(labels).value_counts().to_dict()
        for name in ('unreviewed', 'good', 'bad'):
            handles.append(ax.scatter([], [], s=50, marker='s', c=LABEL_COLOURS[name],
                                      label=f'{name}: {counts.get(name, 0)}'))
        handles.append(ax.plot([], [], '--', color=LABEL_COLOURS['missed'],
                               label=f'missed (added): {len(added)}')[0])
        self._overlay += handles
        self._overlay.append(ax.legend(handles=handles, loc='upper left',
                                       bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                                       fontsize=8, framealpha=1.0))
        done = ' [done]' if self.is_done(visit, raft) else ''
        ax.set_title(f'{visit} ({"extra" if visit % 10 in (7, 8) else "intra"}-focal) raft '
                     f'{raft}: {len(marked)} marked donuts{done}', fontsize=10)

    def set_stretch(self, stretch_e):
        self.stretch_e = stretch_e
        if getattr(self, '_image', None) is not None:
            self._image.set_data(np.arcsinh(self._mosaic / stretch_e))


def make_figure(figsize=(13.5, 10)):
    """Figure with room right of the image axes for the legend."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.07, right=0.72, bottom=0.07, top=0.95)
    return fig, ax


def make_widget(labeller, figsize=(13.5, 10)):
    """The ipywidgets/ipympl review interface. Needs ``%matplotlib widget``."""
    import ipywidgets as w
    # select_candidates and selection_pages set Agg on import, which undoes %matplotlib widget.
    plt.switch_backend('module://ipympl.backend_nbagg')
    names = [f'{v} {r}' for v, r in PAGES]
    page_box = w.Dropdown(options=[(n, i) for i, n in enumerate(names)], value=0,
                          description='page')
    prev_b, next_b = w.Button(description='< prev'), w.Button(description='next >')
    done_b = w.Button(description='Done: the rest are good', button_style='success',
                      layout=w.Layout(width='220px'))
    stretch = w.FloatLogSlider(value=labeller.stretch_e, base=10, min=1, max=4, step=0.05,
                               description='stretch e-')
    status = w.HTML()
    with plt.ioff():
        fig, ax = make_figure(figsize)
    fig.canvas.header_visible = False
    state = {}

    def show(i):
        visit, raft = PAGES[i]
        mosaic, geometry = labeller.raft_image(visit, raft)
        labeller.draw(ax, visit, raft, mosaic, geometry)
        state['page'] = (visit, raft)
        fig.canvas.draw_idle()
        status.value = f'{visit} {raft}: click a bad marked donut; click a missed donut ' \
                       f'to add it; then Done. Labels in {labeller.label_path}'

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

    def on_done(_):
        visit, raft = state['page']
        n = labeller.finish_page(visit, raft)
        labeller.draw(ax, visit, raft)
        fig.canvas.draw_idle()
        status.value = f'{visit} {raft} done: {n} unreviewed donuts set to good'

    fig.canvas.mpl_connect('button_press_event', on_click)
    page_box.observe(lambda ch: show(ch['new']), names='value')
    prev_b.on_click(lambda _: setattr(page_box, 'value', max(0, page_box.value - 1)))
    next_b.on_click(lambda _: setattr(page_box, 'value', min(len(PAGES) - 1, page_box.value + 1)))
    done_b.on_click(on_done)

    def on_stretch(ch):
        labeller.set_stretch(ch['new'])
        fig.canvas.draw_idle()
    stretch.observe(on_stretch, names='value')
    show(0)
    return w.VBox([w.HBox([prev_b, page_box, next_b, done_b]), stretch, status, fig.canvas])
