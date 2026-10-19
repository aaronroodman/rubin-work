"""Spider-shadow check: danish ``Spider_3D`` vanes against batoid-traced spider planes.

For each danish file (v1000, v4.02) and each configuration (field radius along +x,
offsets (detector, camera, M2)), rays on the stop plane are classified twice:
blocked by the danish vanes (rotator 0 deg), and blocked by the batoid spider
planes alone (v4.02 mirrors and lenses with no apertures, plus the
``SpiderFlat``/``SpiderInclined`` planes). The v1000 batoid model has no spider,
so both files are compared with the v4.02 CAD spider.

Per arm (45, 135, -135, -45 deg) and side of the arm (the two plates of a member,
0.70 m apart), the shadow centre perpendicular to the arm is the mean
perpendicular coordinate of blocked rays in 0.1 m bins along the arm. The offset
is the median over bins of danish minus batoid. The strip agreement is the
fraction of rays within one vane width (0.050 m) of the batoid shadow centre on
which danish and batoid agree.

Run::

    python aos/giant_donut_pupil/code/spider_validation.py --mask <v4.02 yaml>
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import obscuration_v402 as ob  # noqa: E402

VANE_WIDTH_M = 0.050
FIELDS_DEG = (0.0, 1.0, 1.75)
OFFSETS_MM = ((0.0, 0.0, 0.0), (0.0, 4.0, 4.0), (0.0, -4.0, -4.0))
ARMS_DEG = (45.0, 135.0, -135.0, -45.0)
# Mean spider height above the stop plane, m; only used to centre the windows.
Z_MEAN_M = 6.7
R_PUPIL_M = (2.58, 4.18)


def shadow_offsets(u, v, b_blocked, d_blocked, theta_deg):
    """Per-plate shadow offsets and the strip agreement.

    Returns
    -------
    plates : `list` [`dict`]
        Per arm and side: ``offset_m`` (danish minus batoid shadow centre,
        perpendicular to the arm, m, stop plane), ``n_bins``.
    agreement : `float`
        Fraction of rays within one vane width of a batoid shadow centre where
        danish and batoid agree (dimensionless).
    """
    r = np.hypot(u, v)
    annulus = (r > R_PUPIL_M[0]) & (r < R_PUPIL_M[1])
    shift = Z_MEAN_M * np.tan(np.deg2rad(theta_deg))
    plates = []
    strip = np.zeros(len(u), bool)
    bins = np.arange(R_PUPIL_M[0], R_PUPIL_M[1] + 1e-9, 0.1)
    for arm in ARMS_DEG:
        a = np.deg2rad(arm)
        s = u * np.cos(a) + v * np.sin(a)
        t = -u * np.sin(a) + v * np.cos(a)
        t0 = -shift * np.sin(a)
        for side in (-1, 1):
            win = annulus & (s > 0) & (side * (t - t0) > 0.15) & (side * (t - t0) < 0.6)
            k = np.digitize(s, bins)
            diffs = []
            for i in range(1, len(bins)):
                sel = win & (k == i)
                bb, dd = sel & b_blocked, sel & d_blocked
                if bb.sum() < 5 or dd.sum() < 5:
                    continue
                tb = np.mean(t[bb])
                diffs.append(np.mean(t[dd]) - tb)
                strip |= sel & (np.abs(t - tb) < VANE_WIDTH_M)
            plates.append(dict(arm_deg=arm, side=side, n_bins=len(diffs),
                               offset_m=float(np.median(diffs)) if diffs else np.nan))
    agreement = float(np.mean((b_blocked == d_blocked)[strip])) if strip.any() else np.nan
    return plates, agreement


def run(mask_files, nrad=400, naz=3000):
    optic = ob.planes_optic(ob.SPIDER_PREFIXES)
    rows, maps = [], {}
    for th in FIELDS_DEG:
        for off in OFFSETS_MM:
            u, v, kept = ob.batoid_keep(optic, off, th, nrad=nrad, naz=naz)
            b_blocked = ~kept
            for name, params in mask_files.items():
                vanes = {'Spider_3D': params['Spider_3D']}
                d_blocked = ~ob.danish_keep(vanes, u, v, np.deg2rad(th), 0.0)
                plates, agree = shadow_offsets(u, v, b_blocked, d_blocked, th)
                off_m = np.array([p['offset_m'] for p in plates])
                row = dict(model=name, field_deg=th, det_mm=off[0], cam_mm=off[1],
                           m2_mm=off[2], max_abs_offset_mm=float(np.nanmax(np.abs(off_m)) * 1e3),
                           median_abs_offset_mm=float(np.nanmedian(np.abs(off_m)) * 1e3),
                           strip_agreement=agree, plates=plates)
                row['max_abs_offset_over_width'] = row['max_abs_offset_mm'] / (VANE_WIDTH_M * 1e3)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'plates'}), flush=True)
                if off == (0.0, 4.0, 4.0):
                    maps[(name, th)] = (u, v, b_blocked, d_blocked)
    return rows, maps


def plot_overlays(maps, pdf, models):
    fig, axes = plt.subplots(len(models), len(FIELDS_DEG), figsize=(15, 5 * len(models)))
    for i, name in enumerate(models):
        for j, th in enumerate(FIELDS_DEG):
            ax = axes[i, j]
            u, v, bb, dd = maps[(name, th)]
            r = np.hypot(u, v)
            sel = (r > R_PUPIL_M[0]) & (r < R_PUPIL_M[1])
            for mask, colour, label in ((bb & dd, 'k', 'both block'),
                                        (bb & ~dd, 'tab:blue', 'batoid only'),
                                        (dd & ~bb, 'tab:red', 'danish only')):
                m = sel & mask
                ax.scatter(u[m], v[m], s=0.05, c=colour, label=label, rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlim(-4.3, 4.3)
            ax.set_ylim(-4.3, 4.3)
            ax.set_title(f'{name} vanes vs batoid CAD spider, field radius {th} deg (+x)')
            ax.set_xlabel('stop-plane u (m)')
            ax.set_ylabel('stop-plane v (m)')
            if i == 0 and j == 0:
                ax.legend(markerscale=60, loc='upper left', fontsize=8)
    fig.suptitle('Spider shadow on the stop plane, offsets (detector, camera, M2) = '
                 '(0, +4, +4) mm, rotator 0 deg; pupil annulus 2.58-4.18 m')
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mask', required=True, help='v4.02 danish obscuration yaml')
    ap.add_argument('--nrad', type=int, default=400)
    ap.add_argument('--naz', type=int, default=3000)
    args = ap.parse_args()
    import yaml
    from rubinwork.common import study_run
    run_dir = study_run(ob.STUDY, 'spider_validation')
    models = {'v1000': ob.load_danish(),
              'v4.02': yaml.safe_load(open(args.mask))}
    rows, maps = run(models, nrad=args.nrad, naz=args.naz)
    (run_dir.path / 'spider_validation.json').write_text(json.dumps(rows, indent=1))
    with PdfPages(run_dir.path / 'spider_validation.pdf') as pdf:
        plot_overlays(maps, pdf, list(models))
    print('done', run_dir.path)


if __name__ == '__main__':
    main()
