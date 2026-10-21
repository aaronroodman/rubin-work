"""The pupil-pass-3 donut sample (step 4 of the 2026-10-21 plan) and its focal-plane map.

From the final candidates (``candidates_v2.parquet``), 2025102300337 (extra) and
2025102300340 (intra), field radius 1.75 deg at most:

- science: the 12 science donuts of the 20261018 sample where they still pass,
  replacements at a comparable field radius where not, 4 more at 0-0.8 deg and
  1 more at 0.8-1.3 deg, so every band spans at least three OCS azimuth
  quadrants. Both sides of focus for every science donut;
- corner WFS: every selected corner donut side within 1.75 deg.

Field angles: CCS ``(FIELD_ANGLE y, FIELD_ANGLE x)``, rotated to OCS by the
rotator angle as `fit_donuts` does.

Writes ``sample.parquet``, ``sample.csv`` and ``sample_map.png`` into a new
``pupil3`` run.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pupil_sample.py --candidates 20261020_candidates_v2
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import selection_v2 as sv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

VISITS = {'extra': 2025102300337, 'intra': 2025102300340}
FIELD_MAX_DEG = 1.75
# Rotator angle as blitz computes it (`fit_donuts.rtp_rad`), deg; from stamps.csv of
# run 20261018_fits.
RTP_DEG = {2025102300337: -0.19894913612764337, 2025102300340: -0.039196}
BANDS_DEG = (0.0, 0.8, 1.3, FIELD_MAX_DEG)
# refcat_id: (sample id, why). The 20261018 ids are kept where the donut still passes.
SCIENCE = {
    5116330175897800704: ('G0005', 'kept'),
    5116616465533584128: ('G0160', 'kept'),
    5068608970366388736: ('G0329', 'kept'),
    5117053040369192704: ('G0491', 'kept'),
    5068066017780627072: ('G0515', 'kept'),
    5020021139218063872: ('G0664', 'kept'),
    5116373610903311488: ('P1064', 'replaces G0249 (1.05 deg): labelled bad on 337'),
    5068652401076227328: ('P1676', 'replaces G0681 (1.76 deg): beyond 1.75 deg'),
    5117347361595190144: ('P1693', 'replaces G0684 (1.77 deg): beyond 1.75 deg'),
    5021160198905070720: ('P1727', 'replaces G0714 (1.86 deg): beyond 1.75 deg'),
    5068137245517790336: ('P1742', 'replaces G0732 (1.92 deg): beyond 1.75 deg'),
    5117811492938760320: ('P1656', 'replaces G0737 (1.98 deg): beyond 1.75 deg'),
    5116537846156682368: ('P0255', 'added, 0-0.8 deg'),
    5116357079572940288: ('P0361', 'added, 0-0.8 deg'),
    5116303276518918272: ('P0482', 'added, 0-0.8 deg'),
    5116414052314958720: ('P0659', 'added, 0-0.8 deg'),
    5116972088825666560: ('P1149', 'added, 0.8-1.3 deg, fourth quadrant'),
}


def add_ocs(t):
    """OCS field angle, radius and azimuth, deg."""
    rtp = np.radians(t.visit.map(RTP_DEG).to_numpy())
    x_ccs, y_ccs = t.field_y_deg.to_numpy(), t.field_x_deg.to_numpy()
    t['field_x_ccs_deg'], t['field_y_ccs_deg'] = x_ccs, y_ccs
    t['field_x_ocs_deg'] = np.cos(rtp) * x_ccs - np.sin(rtp) * y_ccs
    t['field_y_ocs_deg'] = np.sin(rtp) * x_ccs + np.cos(rtp) * y_ccs
    t['field_az_ocs_deg'] = np.mod(np.degrees(np.arctan2(t.field_y_ocs_deg, t.field_x_ocs_deg)),
                                   360.0)
    t['rtp_deg'] = t.visit.map(RTP_DEG)
    return t


def build(cand):
    t = cand[cand.selected & cand.visit.isin(VISITS.values()) & (cand.field_r_deg <= FIELD_MAX_DEG)
             & (cand.hand_label != 'bad')].copy()
    t = add_ocs(t)
    sci = t[t.refcat_id.isin(SCIENCE)].copy()
    sci['sample_id'] = sci.refcat_id.map(lambda r: SCIENCE[r][0])
    sci['why'] = sci.refcat_id.map(lambda r: SCIENCE[r][1])
    corner = t[t.sensor_class != 'science'].copy()
    corner['sample_id'] = ['C' + d[:3] + d[-3:] + f'_{i}' for i, d in
                           zip(corner.groupby('detector').cumcount(), corner.detector)]
    # Corner donuts seen on both visits share one id.
    first = corner.drop_duplicates('refcat_id').set_index('refcat_id').sample_id
    corner.loc[corner.refcat_id >= 0, 'sample_id'] = corner.refcat_id.map(first)
    corner['why'] = 'corner WFS, passes, within 1.75 deg'
    s = pd.concat([sci, corner], ignore_index=True)
    s['side'] = s.visit.map({v: k for k, v in VISITS.items()})
    s['band'] = pd.cut(s.field_r_deg, BANDS_DEG).astype(str)
    s['quadrant'] = (s.field_az_ocs_deg // 90).astype(int)
    missing = set(SCIENCE) - set(sci.refcat_id)
    if missing:
        raise RuntimeError(f'science donuts not in the candidates: {missing}')
    return s.sort_values(['field_r_deg', 'sample_id', 'side']).reset_index(drop=True)


def sample_map(s, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    one = s.drop_duplicates('sample_id')
    sc = ax.scatter(one.field_x_ocs_deg, one.field_y_ocs_deg, c=one.field_r_deg, cmap='viridis',
                    vmin=0, vmax=FIELD_MAX_DEG, s=60,
                    marker='o', edgecolors='k')
    for marker, side in (('^', 'extra'), ('v', 'intra')):
        c = s[(s.sensor_class != 'science') & (s.side == side)]
        ax.scatter(c.field_x_ocs_deg, c.field_y_ocs_deg, marker=marker, s=25, facecolors='none',
                   edgecolors='r', label=f'corner, {side}-focal only' if side == 'extra' else
                   f'corner, {side}-focal')
    for _, r in one.iterrows():
        ax.annotate(r.sample_id, (r.field_x_ocs_deg, r.field_y_ocs_deg), xytext=(4, 4),
                    textcoords='offset points', fontsize=7)
    for rad in BANDS_DEG[1:]:
        ax.add_patch(plt.Circle((0, 0), rad, fill=False, ls='--', lw=0.7, color='0.5'))
    ax.axhline(0, color='0.8', lw=0.5)
    ax.axvline(0, color='0.8', lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xlabel('field x, OCS (deg)')
    ax.set_ylabel('field y, OCS (deg)')
    fig.colorbar(sc, ax=ax, shrink=0.8, label='field radius (deg)')
    ax.set_title(f'Pupil pass 3 sample: {s.sample_id.nunique()} donuts, {len(s)} sides; '
                 'dashed circles 0.8, 1.3, 1.75 deg', fontsize=9)
    ax.legend(fontsize=7, loc='lower left')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--candidates', required=True, help='candidates_v2 run under output/')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    cand = pd.read_parquet(study_dir / 'output' / args.candidates / 'candidates_v2.parquet')
    s = build(cand)
    run = study_run(sv.STUDY, 'pupil3')
    s.to_parquet(run.path / 'sample.parquet')
    s.to_csv(run.path / 'sample.csv', index=False, float_format='%.4f')
    sample_map(s, run.path / 'sample_map.png')
    pd.set_option('display.width', 200)
    print(s[['sample_id', 'side', 'detector', 'mag', 'marker', 'field_r_deg', 'field_az_ocs_deg',
             'band', 'quadrant', 'hand_label', 'why']].round(3).to_string(index=False))
    print(s.drop_duplicates(['sample_id']).groupby(['band', 'quadrant']).size().unstack(fill_value=0))
    print('done', run.path)


if __name__ == '__main__':
    main()
