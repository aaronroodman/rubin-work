"""Pupil pass 2: the analysis steps that read the tables and images, and the pass's PDF.

Steps (``--steps``, comma separated, run in this order):

* ``coma`` (2): each clean side's data − Batoid edge offsets (pass 1 edges)
  converted to Z7, Z8 by the inner-minus-outer centre shift and the
  `coma_sensitivity` matrix, against the Z4 + Z7 + Z8 danish fits
  (``fits_v402g``) and the Batoid coma of the seq 324 state at each field.
* ``edges`` (3b, 4, 6): edges of every image type per side, both the 50 %
  crossings of `pupil_edges.edges` and the robust edges of `robust_edges`,
  in one per-bin table; sector offsets, radius ratios, edge statistics and
  the unmodelled-surface table from it.
* ``cases`` (5): the 10 Batoid against danish cases and the pattern table.
* ``pdf`` (7): the PDF.

Run, from ``code/``::

    python pass2_report.py --run ../output/20261019_pupil2 --steps coma,edges,cases,pdf
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PASS1 = '20261018_pupil'
FITS = '20261018_fits'
STATE_SEQ = 324
SECTORS = np.arange(0, 360, 45)


def _key(df):
    return df.cand_id + '_' + df.side


def usable(g, edge):
    return ~g.spider & ~g.vignetted & ~g[f'masked_{edge}']


def centre_shift(g, a, b):
    """Inner-minus-outer circle-centre shift of edges `a` minus edges `b`, pixels (CCS x, y)."""
    import coma_sensitivity as cs
    az = g.az_ccs_deg.to_numpy()
    out = []
    for edge in ('inner', 'outer'):
        d = (g[f'r_{edge}_{a}_pix'] - g[f'r_{edge}_{b}_pix']).where(usable(g, edge)).to_numpy()
        out.append(cs.circle_centre(d, az)[:2])
    return out[0][0] - out[1][0], out[0][1] - out[1][1]


def state_coma(run, z4):
    """Batoid Z7, Z8 change (µm of wavefront) from the seq 324 state at each side's field and offsets."""
    import fit_donuts as fd
    import giant_state as gs
    import obscuration_v402 as ob
    import z4_fits
    nominal = ob.v402_optic(spider=False)
    optics = {tag: gs.apply_dof(nominal, gs.state_dof(run, STATE_SEQ, trim_tilts=tt))
              for tag, tt in (('state', False), ('state_trim', True))}
    _, r_outer, r_inner, _ = fd.pupil('v4.02', str(fd.ob.pathlib.Path(__file__).resolve().parents[1]
                                                    / 'output' / '20261018_obscuration_v402'
                                                    / 'RubinObsc_v4.02_r_rtpp0_pp0d0.yaml'))
    rows = []
    for r in z4.itertuples():
        thx, thy = np.radians(r.field_x_ccs_deg), np.radians(r.field_y_ccs_deg)
        off = (r.det_offset_mm * 1e-3, r.cam_offset_mm * 1e-3, r.m2_offset_mm * 1e-3)
        z0 = z4_fits.zk_intrinsic(nominal, thx, thy, off, r_outer, r_inner)
        row = dict(key=r.key)
        for tag, o in optics.items():
            dz = (z4_fits.zk_intrinsic(o, thx, thy, off, r_outer, r_inner) - z0) * 1e6
            row.update({f'{tag}_z{j}_um': dz[j] for j in (4, 5, 6, 7, 8)})
        rows.append(row)
    return pd.DataFrame(rows).set_index('key')


def coma(run):
    """Step 2 table ``coma_conversion``: one row per donut side."""
    import coma_sensitivity as cs
    out = run.parent
    bins = pd.read_parquet(out / PASS1 / 'edges_bins.parquet')
    z4 = pd.read_parquet(out / PASS1 / 'z4_fits.parquet')
    z4 = z4[z4.pupil == 'v4.02'].assign(key=lambda d: _key(d)).set_index('key', drop=False)
    sens = pd.read_parquet(run / 'coma_sensitivity.parquet').set_index('side')
    med = sens.median(numeric_only=True)
    fits = pd.read_parquet(run / 'fits_v402g.parquet')
    fits = fits[fits.fit == 'z478'].assign(key=lambda d: _key(d)).set_index('key')
    st = state_coma(run, z4)
    rows = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        key = f'{cand}_{side}'
        s = sens.loc[key] if key in sens.index else med
        dx, dy = centre_shift(g, 'data', 'Batoid')
        m = np.array([[s.z7_rel_dx_pix_per_um, s.z8_rel_dx_pix_per_um],
                      [s.z7_rel_dy_pix_per_um, s.z8_rel_dy_pix_per_um]])
        z7, z8 = np.linalg.solve(m, [dx, dy])
        mdx, mdy = centre_shift(g, 'v402_model', 'Batoid')
        rows.append(dict(cand_id=cand, side=side, field_r_deg=g.field_r_deg.iloc[0],
                         clean=bool(g.clean.iloc[0]), own_sensitivity=key in sens.index,
                         rel_dx_pix=dx, rel_dy_pix=dy, z7_edges_um=z7, z8_edges_um=z8,
                         model_rel_dx_pix=mdx, model_rel_dy_pix=mdy,
                         z7_fit_um=fits.loc[key, 'z7_dev_um'], z8_fit_um=fits.loc[key, 'z8_dev_um'],
                         **st.loc[key].to_dict()))
    df = pd.DataFrame(rows)
    df.to_parquet(run / 'coma_conversion.parquet')
    df.to_csv(run / 'coma_conversion.csv', index=False, float_format='%.4g')
    from rubinwork.common.utils import nmad
    cols = ['rel_dx_pix', 'rel_dy_pix', 'z7_edges_um', 'z8_edges_um', 'z7_fit_um', 'z8_fit_um',
            'state_z7_um', 'state_z8_um', 'state_trim_z7_um', 'state_trim_z8_um']
    summ = []
    for (clean, side), g in df.groupby(['clean', 'side']):
        for c in cols:
            summ.append(dict(clean=clean, side=side, quantity=c, n=len(g),
                             median=g[c].median(), nmad=nmad(g[c].to_numpy())))
    summ = pd.DataFrame(summ)
    summ.to_csv(run / 'coma_summary.csv', index=False, float_format='%.4g')
    print(summ[summ.clean].pivot(index='quantity', columns='side', values=['median', 'nmad'])
          .round(3).to_string())
    return df


IMAGES = {  # image type: (file suffix, array key, run; None = this run); data is the stamp
    'data': None,
    'Batoid v4.02': ('batoid', 'batoid', PASS1),
    'Batoid v4.02 state': ('batoid_state', 'batoid', None),
    'Batoid v4.02 state, no Trim': ('batoid_state0', 'batoid', None),
    'Batoid v1000': ('batoid_v1000', 'batoid', None),
    'danish v4.02': ('v4.02_z4', 'model', PASS1),
    'danish v4.02g': ('v4.02g_z4', 'model', None),
    'danish v4.02g Z478': ('v4.02g_z478', 'model', None),
    'danish v4.02g state': ('v4.02s_z4', 'model', None),
    'danish v1000': ('v1000_z4', 'model', PASS1),
}


def tag(name):
    return name.replace(' ', '_').replace('.', '').replace(',', '')


def _side_edges(task):
    """Per-bin edges of every image type for one side: 50 % crossing and robust."""
    import pupil_edges as pe
    import robust_edges as re_
    run, cand, side, r4, flags = task
    run = pathlib.Path(run)
    st = np.load(run.parent / FITS / 'images' / f'{cand}_{side}_stamp.npz')
    img, msk = st['stamp'].astype(float), st['masked'].astype(bool)
    n = img.shape[0]
    cx, cy = n // 2 + r4['dx_pix'], n // 2 + r4['dy_pix']
    rows = dict(flags)
    fl_in = flags['spider'] | flags['vignetted'] | flags['masked_inner']
    fl_out = flags['spider'] | flags['vignetted'] | flags['masked_outer']
    for name, spec in IMAGES.items():
        if spec is None:
            a = img
        else:
            suffix, arr, src = spec
            path = (run.parent / src if src else run) / 'images' / f'{cand}_{side}_{suffix}.npz'
            if not path.exists():
                continue
            a = np.load(path)[arr].astype(float)
        # Model images are noiseless; the data's noise keeps the same thresholds.
        r_in, r_ot, _, _, _ = pe.edges(a, ~msk, cx, cy, r4['r_out_pix'])
        t = tag(name)
        rows[f'r_inner_{t}_pix'] = r_in
        rows[f'r_outer_{t}_pix'] = r_ot
        rob = re_.robust_edges(a, ~msk, cx, cy, r4['r_out_pix'], r4['bkg_std_e'], fl_in, fl_out)
        for edge in ('inner', 'outer'):
            rows[f'rr_{edge}_{t}_pix'] = rob[f'r_{edge}']
            rows[f'rr_{edge}_{t}_raw_pix'] = rob[f'r_{edge}_raw']
            rows[f'q_{edge}_{t}'] = rob[f'q_{edge}']
            rows[f'fit_{edge}_{t}_pix'] = rob[f'fit_{edge}']
    print(cand, side, flush=True)
    return pd.DataFrame(rows)


def edges(run, workers):
    """Per-bin table ``edges_bins_pass2``: every side and image type."""
    import multiprocessing as mp
    out = run.parent
    old = pd.read_parquet(out / PASS1 / 'edges_bins.parquet')
    z4 = pd.read_parquet(out / PASS1 / 'z4_fits.parquet')
    z4 = z4[z4.pupil == 'v4.02'].assign(key=lambda d: _key(d)).set_index('key')
    keep = ['cand_id', 'side', 'bin', 'az_ccs_deg', 'az_ocs_deg', 'spider', 'vignetted',
            'masked_inner', 'masked_outer', 'field_r_deg', 'clean']
    tasks = []
    for (cand, side), g in old.groupby(['cand_id', 'side']):
        g = g.sort_values('bin')
        tasks.append((str(run), cand, side, z4.loc[f'{cand}_{side}'].to_dict(),
                      {k: g[k].to_numpy() for k in keep}))
    with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
        parts = pool.map(_side_edges, tasks, chunksize=1)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(run / 'edges_bins_pass2.parquet')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--steps', default='coma,edges,cases,pdf')
    ap.add_argument('--workers', type=int, default=38)
    args = ap.parse_args()
    run = pathlib.Path(args.run).resolve()
    pd.set_option('display.width', 220)
    steps = args.steps.split(',')
    if 'coma' in steps:
        coma(run)
    if 'edges' in steps:
        edges(run, args.workers)


if __name__ == '__main__':
    main()
