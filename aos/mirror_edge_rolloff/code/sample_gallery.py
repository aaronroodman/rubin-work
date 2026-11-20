"""Focal-plane gallery of the fit sample: one page per exposure, each donut stamp at its place.

The sample is the ``model_profiles`` run's ``sides.csv`` (science sides at 1.6-1.725 deg
that pass fit c). Each page draws the LSSTCam detector outlines in CCS mm and every
sample stamp of the exposure at true size (10 µm pixels), centred on its detector
position. Stamps are CCS-oriented (`giant_donuts` convention), and CCS x, y are
FOCAL_PLANE y, x. Images are embedded at full resolution (``interpolation='none'``), so
zooming in on the PDF shows every pixel.

Writes ``sample_gallery.pdf`` into a new study run.

Usage, from the repo root::

    python aos/mirror_edge_rolloff/code/sample_gallery.py --models 20261118_model_profiles
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_rolloff as fr  # noqa: E402

PIXEL_MM = 0.010
MM_PER_DEG = 1e3 * 10.312 * np.pi / 180  # nominal plate scale, for the position check


def detector_outlines(camera):
    """Corners of every science and corner detector in CCS mm, with names."""
    from lsst.afw.cameraGeom import FOCAL_PLANE, DetectorType
    out = []
    for det in camera:
        if det.getType() not in (DetectorType.SCIENCE, DetectorType.WAVEFRONT):
            continue
        c = np.array([[p.getX(), p.getY()] for p in det.getCorners(FOCAL_PLANE)])
        out.append((det.getName(), c[:, [1, 0]]))  # FOCAL_PLANE (x, y) -> CCS (y, x)
    return out


def stamp_center_mm(camera, row):
    """CCS mm of a stamp centre from its detector pixel position."""
    import lsst.geom
    from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
    det = camera[row.detector]
    p = det.getTransform(PIXELS, FOCAL_PLANE).applyForward(lsst.geom.Point2D(row.x_pix, row.y_pix))
    return p.getY(), p.getX()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from lsst.obs.lsst import LsstCam
    from rubinwork.common import study_run
    from rubinwork.products import giant_donuts
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--models', required=True, help='model_profiles run directory name')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    S = pd.read_csv(fr._study_out() / a.models / 'sides.csv')
    exp = giant_donuts.load(fr.VARIANT)['exposures'].set_index('visit')
    camera = LsstCam.getCamera()
    outlines = detector_outlines(camera)
    out = pathlib.Path(a.out) if a.out else study_run(fr.STUDY, 'sample_gallery').path
    out.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    with PdfPages(out / 'sample_gallery.pdf') as pdf:
        for visit, g in S.groupby('visit'):
            e = exp.loc[visit]
            fig, ax = plt.subplots(figsize=(14, 14))
            for name, c in outlines:
                ax.add_patch(plt.Polygon(c, closed=True, fill=False, lw=0.15, color='0.4'))
                ax.text(*c.mean(0), name, fontsize=1.5, color='0.6', ha='center', va='center')
            for row in g.itertuples():
                img, msk, _ = giant_donuts.stamp(row.key, fr.VARIANT)
                x0, y0 = stamp_center_mm(camera, row)
                worst = max(worst, np.hypot(x0 - row.field_x_ccs_deg * MM_PER_DEG,
                                            y0 - row.field_y_ccs_deg * MM_PER_DEG))
                n = img.shape[0]
                h = n * PIXEL_MM / 2
                vals = img[~msk.astype(bool)]
                lo, hi = np.percentile(vals, [1, 99.7])
                ax.imshow(img, origin='lower', cmap='gray', vmin=lo, vmax=hi,
                          interpolation='none', extent=(x0 - h, x0 + h, y0 - h, y0 + h), zorder=2)
                ax.text(x0, y0 - h - 0.5, f'{row.detector} {row.star}\nr {row.field_r_deg:.3f} deg, '
                        f'mag {row.mag:.1f}', fontsize=2, ha='center', va='top', zorder=3)
            ax.set_xlim(-330, 330)
            ax.set_ylim(-330, 330)
            ax.set_aspect('equal')
            ax.set_xlabel('CCS x (mm)')
            ax.set_ylabel('CCS y (mm)')
            ax.tick_params(labelsize=6)
            ax.set_title(f'visit {visit} (night {int(e.night)}, seq {int(e.seq_num)}), {e.side}-focal, '
                         f'M2 and camera z {e.step_mm:+.1f} mm, rotator {e.rtp_deg:.2f} deg; '
                         f'{len(g)} sample donuts at true size (zoom in)', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
            print(visit, len(g), flush=True)
    print(f'largest stamp-centre offset from field position x plate scale: {worst:.2f} mm')
    print('wrote', out / 'sample_gallery.pdf', flush=True)


if __name__ == '__main__':
    main()
