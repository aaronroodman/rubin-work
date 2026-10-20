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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--steps', default='coma,edges,cases,pdf')
    args = ap.parse_args()
    run = pathlib.Path(args.run).resolve()
    pd.set_option('display.width', 220)
    steps = args.steps.split(',')
    if 'coma' in steps:
        coma(run)


if __name__ == '__main__':
    main()
