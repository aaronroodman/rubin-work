"""Diagnostic pages for the v2 giant-donut selection.

Per visit: the full focal-plane mosaic, a tally page, a checks page, one page per
raft and one per CCD. Every page draws the CCD edges, every the_monster star at
7-14.5 AB mag with one marker per magnitude bin, every peak (peaks with no catalog
star within 0.5 r_out drawn apart), and the quality circles of the scored stars.
Positions are cameraGeom FOCAL_PLANE mm, the DM (DVCS) view.
"""
import json

import numpy as np

import candidates as cd
import fp_mosaic
import select_candidates as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from lsst.afw.cameraGeom import FOCAL_PLANE  # noqa: E402

__all__ = ['MAG_BIN_STYLE', 'draw_overlays', 'raft_mosaic', 'write_diag_pdf']

# (lo, hi, marker, colour)
MAG_BIN_STYLE = ((7.0, 9.0, '*', '#e7298a'), (9.0, 11.0, 'D', '#66a61e'),
                 (11.0, 13.0, 's', '#1f78b4'), (13.0, 14.5, '^', '#a6cee3'))
REASON_SHORT = {'kept': 'K', 'edge': 'E', 'no_r_flux': 'N', 'out_of_mag': 'M',
                'blitz_neighbour': 'B', 'annulus': 'A', 'overlap': 'O'}
STRETCH_E = 150.0


def mag_bin_counts(stars):
    m = stars.mag_best.to_numpy()
    return [int(((m >= lo) & (m < hi) | ((hi == 14.5) & (m == hi))).sum())
            for lo, hi, _, _ in MAG_BIN_STYLE]


def draw_ccd_edges(ax, camera, names, colour='#00bcd4', lw=0.4):
    for name in names:
        corners = [(c.getX(), c.getY()) for c in camera[name].getCorners(FOCAL_PLANE)]
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec=colour, lw=lw, alpha=0.7))


def draw_overlays(ax, stars, peaks, size=12, circles=True, labels=None, counts=True):
    """Star markers by magnitude bin, peaks, and quality circles.

    Parameters
    ----------
    labels : `str` or `None`, optional
        ``'reason'`` writes the reason code by each star; ``'id'`` the star id.
    """
    counts_by_bin = mag_bin_counts(stars)
    for (lo, hi, marker, colour), n in zip(MAG_BIN_STYLE, counts_by_bin):
        sel = stars[(stars.mag_best >= lo) & (stars.mag_best <= hi)]
        ax.scatter(sel.fp_x_mm, sel.fp_y_mm, s=size, marker=marker, facecolors='none',
                   edgecolors=colour, lw=0.6,
                   label=f'star {lo:g}-{hi:g} mag: {n}' if counts else None)
    matched = peaks[~peaks.unmatched]
    unmatched = peaks[peaks.unmatched]
    ax.scatter(matched.fp_x_mm, matched.fp_y_mm, s=size * 0.6, marker='+', c='yellow', lw=0.6,
               label=f'peak, star within 0.5 r_out: {len(matched)}')
    ax.scatter(unmatched.fp_x_mm, unmatched.fp_y_mm, s=size * 1.2, marker='x', c='red', lw=0.8,
               label=f'peak, no star within 0.5 r_out: {len(unmatched)}')
    if circles:
        for _, r in stars[stars.iso_class != ''].iterrows():
            ax.add_patch(Circle((r.fp_x_mm, r.fp_y_mm), r.r_out_pix * 0.01, fill=False, lw=0.6,
                                color=sc.CLASS_COLOURS[r.iso_class]))
        for name, colour in sc.CLASS_COLOURS.items():
            n = int((stars.iso_class == name).sum())
            ax.plot([], [], 'o', mfc='none', color=colour, label=f'{name}: {n}')
    if labels:
        for _, r in stars.iterrows():
            text = REASON_SHORT[r.reason] if labels == 'reason' else r.star_id[1:]
            ax.annotate(text, (r.fp_x_mm, r.fp_y_mm), xytext=(3, 3), textcoords='offset points',
                        fontsize=5, color='white')


def raft_mosaic(camera, images, backgrounds, names, bin_size):
    """Background-subtracted mosaic of some detectors."""
    sub = {n: images[n] - backgrounds[n] for n in names}
    return fp_mosaic.make_mosaic(camera, sub, cd.CACHE_BIN, bin_size, buffer_size=4)


def _legend(ax, fontsize=6):
    ax.legend(loc='upper right', fontsize=fontsize, framealpha=0.8, markerscale=1.0)


def _text_page(pdf, title, text, fontsize=7):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0, 1, title + '\n\n' + text, family='monospace', fontsize=fontsize, va='top')
    pdf.savefig(fig)
    plt.close(fig)


def checks_text(check, visit):
    lines = []
    for rec in check['peak_threshold_counts']:
        if rec['visit'] == visit:
            lines.append(f"  {rec['sensor_class']:8s} both {rec.get('both', 0):4d}  "
                         f"rel_only {rec.get('rel_only', 0):4d}  sigma_only {rec.get('sigma_only', 0):4d}")
    lines = ['Peak threshold: threshold_rel 0.3 against 5 nMAD of the score map (peak counts)']\
        + lines
    lines += ['', 'Blitz separation, all visits in this run (star counts):']
    for key in ('blitz_rejected', 'blitz_rejected_no_relevant_within_2rout',
                'blitz_rejected_no_dmag2_within_2rout', 'blitz_rejected_image_clean',
                'blitz_rejected_no_relevant_within_2rout_and_clean'):
        lines.append(f'  {key}: {check[key]}')
    lines.append(f"  reimplementation against blitz: {check['blitz_reimpl_mismatch']} mismatches "
                 f"of {check['blitz_reimpl_eligible']} eligible stars")
    lines += ['', f"no_r_flux stars: {check['no_r_flux']}, best magnitude from "
              f"{check['no_r_flux_by_source']}",
              f"  colour term r - G vs BP - RP: {json.dumps(check['colour_term'])}"]
    lines += ['', 'Reason codes: K kept, E edge, N no_r_flux, M out_of_mag, B blitz_neighbour, '
              'A annulus, O overlap']
    return '\n'.join(lines)


def write_diag_pdf(path, visit, side, stars, peaks, counts, check, images, backgrounds, camera):
    """Mosaic, tally, checks, raft and CCD pages of one visit."""
    names = sorted(images)
    with PdfPages(path) as pdf:
        mosaic, geometry = raft_mosaic(camera, images, backgrounds, names, sc.MOSAIC_BIN)
        fig, ax = plt.subplots(figsize=(11, 11))
        sc.show_mosaic(ax, mosaic, geometry, STRETCH_E)
        draw_ccd_edges(ax, camera, names, lw=0.3)
        draw_overlays(ax, stars, peaks, size=4)
        _legend(ax)
        ax.set_title(f'{visit} ({side}-focal): the_monster 7-14.5 AB mag, peaks and quality '
                     f'circles; mosaic binned {sc.MOSAIC_BIN}x{sc.MOSAIC_BIN}', fontsize=9)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        cols = ['sensor_class', 'catalog', 'on_silicon', 'full_on_silicon', 'with_r',
                'matched_peak', 'isolated', 'marginal', 'blended', 'saturated', 'peaks_rel',
                'peaks_rel_unmatched', 'peaks_sigma', 'peaks_sigma_unmatched']
        _text_page(pdf, f'{visit} ({side}-focal): tally, star and peak counts per sensor class\n'
                   'catalog: the_monster at 7-14.5 AB mag (best r) in the camera footprint; '
                   'on_silicon: centre on a CCD;\nfull_on_silicon: whole donut on the CCD; '
                   'with_r: ComCam r flux and 7-14.5 mag; matched_peak: a threshold_rel peak\n'
                   'within 0.25 r_out. Classes are the 20261018 rule. peaks_*_unmatched: '
                   'no catalog star (to 18.5 mag) within 0.5 r_out.',
                   counts[cols].to_string(index=False), fontsize=8)
        _text_page(pdf, f'{visit} ({side}-focal): checks', checks_text(check, visit))
        if visit == 2025102300340:
            lost = [r for r in check['R31_intra_clean_not_kept']]
            text = '\n'.join(json.dumps(r) for r in lost) or 'none'
            _text_page(pdf, 'R31 intra: image-clean donuts (annulus_ratio < 0.05, overlap_frac '
                       '< 0.02) that the baseline drops, with the reason code.\n'
                       'Distances in r_out (dimensionless), magnitudes AB mag.', text, fontsize=6)

        for raft in sorted({n[:3] for n in names}):
            rnames = [n for n in names if n.startswith(raft)]
            rmos, rgeom = raft_mosaic(camera, images, backgrounds, rnames, sc.RAFT_BIN)
            fig, ax = plt.subplots(figsize=(11, 11))
            sc.show_mosaic(ax, rmos, rgeom, STRETCH_E)
            draw_ccd_edges(ax, camera, rnames, lw=0.8)
            rs = stars[stars.detector.str.startswith(raft)]
            rp = peaks[peaks.detector.str.startswith(raft)]
            draw_overlays(ax, rs, rp, size=30, labels='reason')
            for n in rnames:
                c = camera[n].getCenter(FOCAL_PLANE)
                ax.text(c.getX(), c.getY(), n[4:], color='cyan', fontsize=7, alpha=0.6,
                        ha='center')
            _legend(ax)
            ax.set_title(f'{visit} ({side}-focal) raft {raft}; reason codes by each star; '
                         f'binned {sc.RAFT_BIN}x{sc.RAFT_BIN}', fontsize=9)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        for name in names:
            cmos, cgeom = raft_mosaic(camera, images, backgrounds, [name], cd.CACHE_BIN)
            fig, ax = plt.subplots(figsize=(9, 9))
            sc.show_mosaic(ax, cmos, cgeom, STRETCH_E)
            draw_ccd_edges(ax, camera, [name], lw=1.0)
            cs = stars[stars.detector == name]
            cp = peaks[peaks.detector == name]
            draw_overlays(ax, cs, cp, size=50, labels='reason')
            _legend(ax)
            ax.set_title(f'{visit} ({side}-focal) {name}; binned {cd.CACHE_BIN}x{cd.CACHE_BIN}',
                         fontsize=9)
            pdf.savefig(fig, dpi=90)
            plt.close(fig)
