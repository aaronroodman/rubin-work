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


def edge_col(df, edge, name, robust=True):
    """Edge radius column of image type `name`: robust (kept bins) or 50 % crossing, pixels."""
    t = tag(name)
    return df[f'rr_{edge}_{t}_pix'] if robust else df[f'r_{edge}_{t}_pix']


def sectors(bins, model, robust=True):
    """Median data − `model` inner and outer edge per 45 deg OCS sector, clean sides, pixels."""
    rows = []
    for edge in ('inner', 'outer'):
        u = bins[bins.clean & usable(bins, edge)]
        d = edge_col(u, edge, 'data', robust) - edge_col(u, edge, model, robust)
        sec = (u.az_ocs_deg // 45 * 45).astype(int)
        for (side, s), v in d.groupby([u.side, sec]):
            rows.append(dict(model=model, robust=robust, edge=edge, side=side, sector_deg=s,
                             median_pix=v.median(), n_bins=int(v.notna().sum())))
    return pd.DataFrame(rows)


RATIO_MODELS = ('data', 'Batoid v4.02', 'danish v4.02', 'danish v4.02g', 'danish v1000',
                'Batoid v1000', 'Batoid v4.02 state', 'Batoid v4.02 state, no Trim')


def radius_ratio(bins):
    """r_inner / r_outer per bin (robust edges): per side, its median and spread over azimuth."""
    from rubinwork.common.utils import nmad
    rows = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        use = usable(g, 'inner') & usable(g, 'outer')
        for m in RATIO_MODELS:
            if f'rr_inner_{tag(m)}_pix' not in g:
                continue
            q = (edge_col(g, 'inner', m) / edge_col(g, 'outer', m))[use].dropna()
            if len(q) < 10:
                continue
            rows.append(dict(cand_id=cand, side=side, field_r_deg=g.field_r_deg.iloc[0],
                             clean=bool(g.clean.iloc[0]), model=m, n_bins=len(q),
                             median=q.median(), nmad=nmad(q.to_numpy()),
                             p95_minus_p5=q.quantile(0.95) - q.quantile(0.05)))
    return pd.DataFrame(rows)


def edge_stats_robust(bins, ref='Batoid v4.02'):
    """RMS and max |data − ref| and |danish v4.02g − ref| per side and edge (robust edges), pixels."""
    rows = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        row = dict(cand_id=cand, side=side, field_r_deg=g.field_r_deg.iloc[0], clean=g.clean.iloc[0])
        for edge in ('inner', 'outer'):
            u = usable(g, edge)
            r0 = g[f'rr_{edge}_{tag(ref)}_raw_pix']
            for name, m in (('data', 'data'), ('v402g', 'danish v4.02g'), ('v402', 'danish v4.02')):
                d = (edge_col(g, edge, m) - r0)[u].dropna()
                row[f'{edge}_{name}_rms_pix'] = float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
                row[f'{edge}_{name}_max_pix'] = float(np.abs(d).max()) if len(d) else np.nan
            row[f'{edge}_n_kept'] = int(np.isfinite(edge_col(g, edge, 'data'))[u].sum())
            row[f'{edge}_n_usable'] = int(u.sum())
            row[f'{edge}_n_pass1'] = int(np.isfinite(g[f'r_{edge}_data_pix'])[u].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def unmodelled_robust(run, bins):
    """Runs of 3 or more kept bins with |data − Batoid v4.02| > 3 pixels (robust edges), with
    the Batoid clipping surface, as `pupil_edges.measure` does for the pass 1 edges."""
    import pupil_edges as pe
    z4 = pd.read_parquet(run.parent / PASS1 / 'z4_fits.parquet')
    z4 = z4[z4.pupil == 'v4.02'].assign(key=lambda d: _key(d)).set_index('key')
    rows = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        g = g.sort_values('bin')
        r4 = z4.loc[f'{cand}_{side}']
        npz = np.load(run.parent / PASS1 / 'images' / f'{cand}_{side}_batoid.npz')
        n = int(2 * r4.stamp_half_pix + 1)
        cx, cy = n // 2 + r4.dx_pix, n // 2 + r4.dy_pix
        for edge in ('inner', 'outer'):
            ref = g[f'rr_{edge}_Batoid_v402_raw_pix'].to_numpy()
            diff = (edge_col(g, edge, 'data') - g[f'rr_{edge}_Batoid_v402_raw_pix']).to_numpy()
            ok = usable(g, edge).to_numpy() & np.isfinite(diff)
            for b0, b1 in pe.runs(ok & (np.abs(diff) > pe.OUTLIER_PIX), pe.OUTLIER_RUN):
                sel = np.arange(b0, b0 + ((b1 - b0) % pe.NBIN or pe.NBIN)) % pe.NBIN
                surface, frac = pe.clip_surface(npz, cx, cy, b0, b1, ref, edge)
                az = g.az_ocs_deg.to_numpy()
                rows.append(dict(cand_id=cand, side=side, edge=edge, field_r_deg=r4.field_r_deg,
                                 clean=bool(r4.clean), az_ocs_from_deg=float(az[sel[0]] - 1) % 360,
                                 az_ocs_to_deg=float(az[sel[-1]] + 1) % 360, n_bins=len(sel),
                                 median_offset_pix=float(np.median(diff[sel])),
                                 clipping_surface=surface.split(' (')[0], surface_fraction=frac))
    return pd.DataFrame(rows)


def write(run, name, df):
    df.to_parquet(run / f'{name}.parquet')
    df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')


def analyse_edges(run):
    """Tables of steps 3b, 4 and 6 from ``edges_bins_pass2``."""
    bins = pd.read_parquet(run / 'edges_bins_pass2.parquet')
    sec = pd.concat([sectors(bins, m, rb) for m in ('Batoid v4.02', 'Batoid v4.02 state',
                                                    'Batoid v4.02 state, no Trim')
                     for rb in (False, True) if f'r_inner_{tag(m)}_pix' in bins], ignore_index=True)
    write(run, 'sector_offsets', sec)
    print(sec[sec.edge == 'inner'].pivot_table(index=['model', 'robust', 'side'],
                                               columns='sector_deg', values='median_pix')
          .round(2).to_string())
    rr = radius_ratio(bins)
    write(run, 'radius_ratio', rr)
    print(rr[rr.clean].groupby('model')[['median', 'nmad', 'p95_minus_p5']].median().round(4)
          .to_string())
    st = edge_stats_robust(bins)
    write(run, 'edge_stats_robust', st)
    um = unmodelled_robust(run, bins)
    write(run, 'unmodelled_surfaces_robust', um)
    kept = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        for m in IMAGES:
            if f'q_inner_{tag(m)}' not in g:
                continue
            for edge in ('inner', 'outer'):
                q = g[f'q_{edge}_{tag(m)}']
                kept.append(dict(cand_id=cand, side=side, clean=bool(g.clean.iloc[0]),
                                 field_r_deg=g.field_r_deg.iloc[0], image=m, edge=edge,
                                 kept_frac=float(np.isin(q, ('ok', 'multi')).mean()),
                                 kept_of_usable=float(np.isin(q[usable(g, edge)], ('ok', 'multi'))
                                                      .mean()),
                                 pass1_of_usable=float(np.isfinite(
                                     g[f'r_{edge}_{tag(m)}_pix'][usable(g, edge)]).mean()),
                                 **{f'n_{k}': int((q == k).sum()) for k in
                                    ('ok', 'multi', 'flagged', 'low_top', 'no_crossing',
                                     'outlier')}))
    write(run, 'robust_kept', pd.DataFrame(kept))


# Aaron's list (OCS azimuth ranges, deg). G0429 and G0492 are not among the 38 fitted
# sides (both blended); G0249 and G0592 are the fitted sides with those digits.
CASES = [
    ('G0429', ('extra', 'intra'), 'outer', 80, 200, 'Batoid vs danish v4.02'),
    ('G0492', ('extra', 'intra'), 'inner', 330, 350, 'danish v4.02 vs the others'),
    ('G0491', ('extra', 'intra'), 'inner', 150, 280, 'danish v4.02; a cut into the aperture'),
    ('G0515', ('extra', 'intra'), 'inner', 30, 50, 'danish v4.02'),
    ('G0618', ('extra', 'intra'), 'inner', 125, 145, 'danish v4.02'),
    ('G0663', ('extra',), 'outer', 220, 240, 'danish v1000; another cut'),
    ('G0663', ('intra',), 'outer', 35, 50, 'danish v1000 and Batoid vs danish v4.02 and data'),
    ('G0664', ('intra',), 'outer', 170, 190, 'danish v1000 and Batoid vs danish v4.02 and data'),
    ('G0676', ('extra',), 'outer', 40, 60, 'danish v1000 and v4.02 vs Batoid and data'),
    ('G0678', ('intra',), 'outer', 310, 330, 'danish v1000 and v4.02 vs Batoid and data'),
]
SUBSTITUTE = {'G0429': 'G0249', 'G0492': 'G0592'}
COMPARE = ('data', 'Batoid v4.02', 'danish v4.02', 'danish v4.02g', 'danish v1000',
           'Batoid v1000')
CLOSEST = ('Batoid v4.02', 'danish v4.02g', 'danish v1000')
# Batoid surface (leaf-name prefix) -> how the danish v4.02 / v4.02g mask represents it.
DANISH_REP = [
    ('CamAperture', 'none: in-camera openings dropped from the mask'),
    ('CameraSilhouette', 'one CameraSilhouette circle per field angle (union of the planes)'),
    ('Spider', 'Spider_3D vanes'),
    ('M1Baffle1', 'M1Baffle1 outer circle'),
    ('M1', 'M1 inner circle'),
    ('M2', 'M2 inner and outer circles'),
    ('M3', 'M3 inner and outer circles'),
    ('Filter', 'Filter_entrance outer circle, beyond its thetaMin'),
    ('L3', 'L3_entrance outer circle, beyond its thetaMin'),
]


def danish_rep(surface):
    for prefix, rep in DANISH_REP:
        if surface.startswith(prefix):
            return rep
    return 'none: not in the mask'


def bin_surfaces(npz, cx, cy, r_edge, edge):
    """Per-bin Batoid clipping surface (most common first blocker within 3 pixels outside the edge)."""
    import pupil_edges as pe
    px, py, first, names = npz['tf_px'], npz['tf_py'], npz['tf_first'], npz['tf_names']
    phi = np.mod(np.degrees(np.arctan2(py - cy, px - cx)), 360.0)
    k = np.minimum((phi / (360 / pe.NBIN)).astype(int), pe.NBIN - 1)
    d = (np.hypot(px - cx, py - cy) - r_edge[k]) * (1.0 if edge == 'outer' else -1.0)
    near = np.isfinite(d) & (d > 0) & (d < pe.EDGE_BAND_PIX) & (first >= 0)
    leaf = np.array([str(n).split('.')[-1] for n in names])
    out = np.array(['none traced'] * pe.NBIN, dtype=object)
    for b in np.unique(k[near]):
        u, c = np.unique(first[near & (k == b)], return_counts=True)
        out[b] = leaf[u[np.argmax(c)]]
    return out


def _group(surface):
    for prefix, _ in DANISH_REP:
        if str(surface).startswith(prefix):
            return prefix
    return str(surface)


def cases(run):
    """Step 5: the case table ``cases`` and the per-bin pattern tables."""
    import yaml
    import obscuration_v402g as og
    bins = pd.read_parquet(run / 'edges_bins_pass2.parquet')
    z4 = pd.read_parquet(run.parent / PASS1 / 'z4_fits.parquet')
    z4 = z4[z4.pupil == 'v4.02'].assign(key=lambda d: _key(d)).set_index('key')
    cuts = pd.read_parquet(run / 'camaperture_cuts.parquet')
    surf_rows = []
    for (cand, side), g in bins.groupby(['cand_id', 'side']):
        g = g.sort_values('bin')
        r4 = z4.loc[f'{cand}_{side}']
        npz = np.load(run.parent / PASS1 / 'images' / f'{cand}_{side}_batoid.npz')
        n = int(2 * r4.stamp_half_pix + 1)
        cx, cy = n // 2 + r4.dx_pix, n // 2 + r4.dy_pix
        field_az = np.degrees(np.arctan2(r4.field_y_ccs_deg, r4.field_x_ccs_deg))
        for edge in ('inner', 'outer'):
            ref = g[f'rr_{edge}_Batoid_v402_raw_pix'].to_numpy()
            surf = bin_surfaces(npz, cx, cy, ref, edge)
            rel = (g.az_ccs_deg.to_numpy() - field_az + 180) % 360 - 180
            row = dict(cand_id=cand, side=side, edge=edge, bin=g.bin.to_numpy(),
                       az_ocs_deg=g.az_ocs_deg.to_numpy(), az_from_radial_deg=rel,
                       field_r_deg=r4.field_r_deg, clean=bool(r4.clean),
                       usable=usable(g, edge).to_numpy(), surface=surf)
            data = edge_col(g, edge, 'data').to_numpy()
            data_raw = g[f'rr_{edge}_data_raw_pix'].to_numpy()
            for m in COMPARE[1:]:
                t = tag(m)
                if f'rr_{edge}_{t}_raw_pix' in g:
                    row[f'd_{t}_pix'] = g[f'rr_{edge}_{t}_raw_pix'].to_numpy() - data
                    # Unfiltered: raw crossings of both, for the narrow case ranges.
                    row[f'draw_{t}_pix'] = g[f'rr_{edge}_{t}_raw_pix'].to_numpy() - data_raw
            surf_rows.append(pd.DataFrame(row))
    per_bin = pd.concat(surf_rows, ignore_index=True)
    per_bin['surface_group'] = per_bin.surface.map(_group)
    cols = [f'd_{tag(m)}_pix' for m in CLOSEST]
    ok = per_bin.usable & per_bin[cols].notna().all(axis=1)
    per_bin['closest'] = np.where(ok, np.array(CLOSEST)[np.argmin(np.abs(per_bin[cols].fillna(1e9)
                                                                         .to_numpy()), axis=1)], '')
    write(run, 'pattern_bins', per_bin)
    pat = per_bin[ok].assign(
        field_band=pd.cut(per_bin.field_r_deg, [0, 1.0, 1.6, 1.725, 1.9, 2.0]).astype(str),
        radial_band=pd.cut(per_bin.az_from_radial_deg.abs(), [0, 45, 90, 135, 180],
                           include_lowest=True).astype(str))
    tabs = []
    for by in ('field_band', 'radial_band', 'side', 'edge', 'surface_group'):
        for key, g in pat.groupby(by):
            row = dict(by=by, value=str(key), n_bins=len(g))
            for m in CLOSEST:
                row[f'closest_frac_{tag(m)}'] = float((g.closest == m).mean())
                row[f'median_abs_d_{tag(m)}_pix'] = float(g[f'd_{tag(m)}_pix'].abs().median())
            tabs.append(row)
    pattern = pd.DataFrame(tabs)
    write(run, 'pattern_table', pattern)
    print(pattern.round(3).to_string())

    rows = []
    for cand, sides, edge, a0, a1, note in CASES:
        used = SUBSTITUTE.get(cand, cand)
        for side in sides:
            g = per_bin[(per_bin.cand_id == used) & (per_bin.side == side) & (per_bin.edge == edge)]
            w = g[(g.az_ocs_deg >= a0) & (g.az_ocs_deg <= a1)]
            r4 = z4.loc[f'{used}_{side}']
            off = (r4.det_offset_mm * 1e-3, r4.cam_offset_mm * 1e-3, r4.m2_offset_mm * 1e-3)
            mask = yaml.safe_load(og.mask_path(run, off).read_text())
            active = sorted(f'{k}.{e}' for k, v in mask.items() if k != 'Spider_3D'
                            for e, p in v.items()
                            if p['thetaMin'] <= r4.field_r_deg <= p['thetaMax'])
            sv = w.surface[w.surface != 'none traced']
            top = sv.value_counts()
            surface = top.index[0] if len(top) else 'none traced'
            c = cuts[(cuts.cam_mm == r4.cam_offset_mm)]
            c = c.iloc[(c.field_r_deg - r4.field_r_deg).abs().argsort()[:8]]
            row = dict(case=cand, cand_id=used, side=side, edge=edge, az_from_deg=a0, az_to_deg=a1,
                       note=note, field_r_deg=r4.field_r_deg, clean=bool(r4.clean),
                       n_bins=len(w), n_usable=int(w.usable.sum()), surface=surface,
                       surface_frac=float(top.iloc[0] / top.sum()) if len(top) else np.nan,
                       other_surfaces=';'.join(f'{k}:{v}' for k, v in top.iloc[1:4].items()),
                       danish_representation=danish_rep(surface),
                       v402g_active=';'.join(active),
                       camaperture_unique_max=float(c.unique_CamAperture.max()),
                       silhouette_unique_max=float(c.unique_CameraSilhouette.max()),
                       filter_unique_max=float(c.unique_Filter_entrance.max()))
            # Raw crossings over every bin of the range: the robust cut and the
            # spider flag leave too few bins in 10-20 deg ranges.
            row['n_raw'] = int(w['draw_Batoid_v402_pix'].notna().sum())
            for m in COMPARE[1:]:
                col = f'draw_{tag(m)}_pix'
                if col in w:
                    row[f'{tag(m)}_minus_data_pix'] = float(w[col].median())
                    row[f'{tag(m)}_minus_batoid_pix'] = float(
                        (w[col] - w['draw_Batoid_v402_pix']).median())
            rows.append(row)
    cs = pd.DataFrame(rows)
    write(run, 'cases', cs)
    print(cs[['case', 'side', 'edge', 'surface', 'n_raw'] + [c for c in cs if c.endswith('_pix')]]
          .round(2).to_string())
    return cs, pattern


# Fixed categorical order (validated default palette); each image type keeps its colour.
COLORS = {'data': '#000000', 'Batoid v4.02': '#2a78d6', 'Batoid v4.02 state': '#eb6834',
          'danish v4.02': '#1baf7a', 'danish v4.02g': '#eda100', 'danish v1000': '#e87ba4',
          'Batoid v1000': '#008300', 'Batoid v4.02 state, no Trim': '#7a5195'}


def text_page(pdf, title, lines, size=8):
    # Fit the monospace block to the page: about 0.6 em per character, 1.2 em per line.
    width = max((len(x) for x in lines), default=1)
    size = min(12.0, 17 * 72 * 0.95 / (0.6 * width), 11 * 72 * 0.88 / (1.25 * max(len(lines), 1)))
    fig = plt.figure(figsize=(17, 11))
    fig.text(0.02, 0.97, title, fontsize=14, va='top', weight='bold')
    fig.text(0.02, 0.93, '\n'.join(lines), family='monospace', fontsize=size, va='top')
    pdf.savefig(fig)
    plt.close(fig)


def table_lines(df, fmt='%.3g'):
    return df.to_string(float_format=lambda v: fmt % v).split('\n')


def page_state(pdf, run):
    import json
    meta = json.loads((run / f'state{STATE_SEQ}.json').read_text())
    zk = pd.read_parquet(run / f'state{STATE_SEQ}_zk.parquet')
    dof = pd.read_parquet(run / f'state{STATE_SEQ}_dof.parquet').value.to_numpy().copy()
    dof[[3, 4, 8, 9]] *= 3600
    trim = pd.read_parquet(run / f'trim_check_{STATE_SEQ}.parquet')
    act = pd.read_parquet(run / f'hexapod_actual_{STATE_SEQ}.parquet')
    names = ['M2 dz', 'M2 dx', 'M2 dy', 'M2 rx', 'M2 ry', 'cam dz', 'cam dx', 'cam dy',
             'cam rx', 'cam ry']
    lines = [f'Visit {meta["visit_id"]} (seq {STATE_SEQ}, last visit of infocus_initial_alignment); '
             f'value_added optical_state {meta["variant"]}', f'db {meta["db"]}',
             f'corner residual RMS {meta["resid_rms_um"]:.3f} µm of wavefront; rotator '
             f'{meta["rotator_angle_deg"]:.3f} deg; elevation {meta["elevation_deg"]:.2f} deg', '',
             'Corner deviation from the batoid intrinsic, µm of wavefront (OCS):']
    lines += table_lines(zk[zk.noll.isin([5, 6, 7, 8])].pivot(index='corner', columns='noll',
                                                              values='deviation_um'), '%+.3f')
    lines += ['', 'optical_state rigid-body DOF (µm; arcsec for rx, ry):']
    lines += ['  ' + '  '.join(f'{n} {v:+.2f}' for n, v in zip(names, dof[:10]))]
    lines += ['', 'Trim at 337/340 minus Trim at 324 (µm; arcsec):']
    lines += table_lines(trim[['side', 'seq', 'dof', 'unit', 'trim_ref', 'trim_minus_ref']], '%+.3f')
    lines += ['', 'EFD actual minus demand, max |.| per visit:']
    lines += table_lines(act.groupby(['seq', 'unit']).actual_minus_demand_maxabs.max().unstack(),
                         '%.3f')
    lines += ['', 'Batoid DOF of step 3b: v4.02s = this optical_state + the 337/340 Trim tilts '
              '(M2 rx +10.10, ry -84.97; camera rx -26.55, ry +6.89 arcsec); v4.02s0 = '
              'optical_state only.']
    text_page(pdf, 'Section 1: the telescope state, seq 324', lines)


def page_coma(pdf, run):
    df = pd.read_parquet(run / 'coma_conversion.parquet')
    summ = pd.read_csv(run / 'coma_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, j in zip(axes, (7, 8)):
        for side, mk in (('extra', 'o'), ('intra', 's')):
            g = df[df.clean & (df.side == side)]
            ax.plot(g.field_r_deg, g[f'z{j}_edges_um'], mk, color='#2a78d6', ms=8,
                    label=f'from edges, {side}')
            ax.plot(g.field_r_deg, g[f'z{j}_fit_um'], mk, color='#eb6834', ms=8, mfc='none',
                    label=f'danish Z4+Z7+Z8 fit, {side}')
            ax.plot(g.field_r_deg, g[f'state_z{j}_um'], mk, color='#1baf7a', ms=6,
                    label=f'324 state, {side}')
            ax.plot(g.field_r_deg, g[f'state_trim_z{j}_um'], mk, color='#eda100', ms=6,
                    label=f'324 state + Trim tilts, {side}')
        ax.axhline(0, color='0.7', lw=0.8)
        ax.set_xlabel('field radius (deg)')
        ax.set_ylabel(f'Z{j} (µm of wavefront)')
        ax.set_title(f'Z{j}, clean sides')
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Section 2: coma from the inner-minus-outer edge-centre shift (data − Batoid v4.02, '
                 'pass 1 edges); +1 µm Z7 = +4.45 pixels along +y CCS, +1 µm Z8 = +4.45 pixels '
                 'along +x CCS', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    s = summ[summ.clean].pivot(index='quantity', columns='side', values=['median', 'nmad'])
    lines = ['Clean sides: median and nMAD; Z in µm of wavefront, shifts in pixels (CCS)'] + \
        table_lines(s, '%+.3f') + ['', 'Per side:'] + table_lines(
            df[['cand_id', 'side', 'field_r_deg', 'clean', 'rel_dx_pix', 'rel_dy_pix', 'z7_edges_um',
                'z8_edges_um', 'z7_fit_um', 'z8_fit_um', 'state_z7_um', 'state_z8_um',
                'state_trim_z7_um', 'state_trim_z8_um']], '%+.2f')
    text_page(pdf, 'Section 2: coma table', lines, size=7)


def page_sectors(pdf, run):
    sec = pd.read_parquet(run / 'sector_offsets.parquet')
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    for ax, side in zip(axes, ('extra', 'intra')):
        for (model, robust), g in sec[(sec.edge == 'inner') & (sec.side == side)].groupby(
                ['model', 'robust']):
            g = g.sort_values('sector_deg')
            ax.plot(g.sector_deg + 22.5, g.median_pix, 'o-' if robust else 'o--',
                    color=COLORS.get(model, '0.5'), label=f'{model}, {"robust" if robust else "50 %"}')
        ax.axhline(0, color='0.7', lw=0.8)
        ax.set_xlabel('OCS azimuth, sector centre (deg)')
        ax.set_title(f'inner edge, data − model, clean sides, {side}')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('median data − model (pixels)')
    fig.suptitle('Section 3: 45 deg sector inner-edge offsets before (Batoid v4.02) and after '
                 '(with the 324 state) step 3b', fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    p = sec.pivot_table(index=['edge', 'model', 'robust', 'side'], columns='sector_deg',
                        values='median_pix')
    text_page(pdf, 'Section 3: sector offsets, median data − model, pixels', table_lines(p, '%+.2f'))
    zs = run / 'z4_fits_state.parquet'
    if zs.exists():
        z = pd.read_parquet(zs)
        g = pd.read_parquet(run / 'fits_v402g.parquet')
        g = g[g.fit == 'z4'].assign(pupil='v4.02g')
        both = pd.concat([z, g]).pivot_table(index=['cand_id', 'side'], columns='pupil',
                                             values=['resid_rms_e', 'z4_dev_um'])
        lines = ['Z4-only fits, v4.02g mask: residual RMS (electrons per pixel) and Z4 − intrinsic '
                 '(µm of wavefront), nominal v4.02 intrinsic against the 324-state intrinsics'] + \
            table_lines(both, '%.2f')
        text_page(pdf, 'Section 3: Z4-only fits with the state', lines, size=7)


def page_ratio(pdf, run):
    rr = pd.read_parquet(run / 'radius_ratio.parquet')
    bins = pd.read_parquet(run / 'edges_bins_pass2.parquet')
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    ax = axes[0]
    for m in RATIO_MODELS:
        g = rr[rr.clean & (rr.model == m)].sort_values('field_r_deg')
        if len(g):
            ax.plot(g.field_r_deg, g['median'], 'o', color=COLORS.get(m, '0.5'), label=m, ms=6)
    ax.set_xlabel('field radius (deg)')
    ax.set_ylabel('median r_inner / r_outer (dimensionless)')
    ax.legend(fontsize=7)
    ax.set_title('per side, clean sides')
    ax = axes[1]
    g = bins[(bins.cand_id == 'G0515') & (bins.side == 'extra')].sort_values('az_ocs_deg')
    use = usable(g, 'inner') & usable(g, 'outer')
    for m in RATIO_MODELS:
        if f'rr_inner_{tag(m)}_pix' in g:
            q = (edge_col(g, 'inner', m) / edge_col(g, 'outer', m)).where(use)
            ax.plot(g.az_ocs_deg, q, '-', color=COLORS.get(m, '0.5'), lw=1.5, label=m)
    ax.set_xlabel('OCS azimuth (deg)')
    ax.set_ylabel('r_inner / r_outer (dimensionless)')
    ax.set_title('G0515 extra, against azimuth (robust edges)')
    ax.legend(fontsize=7)
    fig.suptitle('Section 4: the radius ratio r_inner / r_outer', fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    t = rr[rr.clean].groupby('model')[['median', 'nmad', 'p95_minus_p5']].median()
    text_page(pdf, 'Section 4: radius ratio, median over clean sides of the per-side statistic '
              '(dimensionless)', table_lines(t, '%.4f'))


def page_cases(pdf, run):
    cs = pd.read_parquet(run / 'cases.parquet')
    pat = pd.read_parquet(run / 'pattern_table.parquet')
    cols = ['case', 'cand_id', 'side', 'edge', 'az_from_deg', 'az_to_deg', 'field_r_deg',
            'surface', 'danish_representation'] + [c for c in cs if c.endswith('minus_batoid_pix')]
    lines = ['Model minus Batoid v4.02, median of raw 50 % crossings over the range, pixels. '
             'G0429 -> G0249 and G0492 -> G0592 (G0429, G0492 were never fitted).'] + \
        table_lines(cs[cols].rename(columns=lambda c: c.replace('_minus_batoid_pix', '')), '%+.2f')
    text_page(pdf, 'Section 5: the 10 cases', lines, size=6.5)
    lines = ['Fraction of usable bins where each model is closest to the data (robust data edge); '
             'median |model − data|, pixels'] + table_lines(pat.set_index(['by', 'value']), '%.3f')
    text_page(pdf, 'Section 5: pattern table', lines, size=7)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
    for ax, by in zip(axes, ('field_band', 'radial_band', 'surface_group')):
        g = pat[pat.by == by]
        x = np.arange(len(g))
        for k, m in enumerate(CLOSEST):
            ax.bar(x + (k - 1) * 0.27, g[f'closest_frac_{tag(m)}'], 0.25, color=COLORS[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(g.value, rotation=45, ha='right', fontsize=8)
        ax.set_title(by.replace('_', ' '))
    axes[0].set_ylabel('fraction of bins closest to the data (dimensionless)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Section 5: which model is closest to the data', fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_side_edges(pdf, bins, cand, side):
    g = bins[(bins.cand_id == cand) & (bins.side == side)].sort_values('az_ocs_deg')
    fig, axes = plt.subplots(2, 1, figsize=(17, 10), sharex=True)
    for ax, edge in zip(axes, ('outer', 'inner')):
        for m in ('data', 'Batoid v4.02', 'Batoid v4.02 state', 'danish v4.02g', 'danish v1000',
                  'Batoid v1000'):
            t = tag(m)
            if f'rr_{edge}_{t}_raw_pix' not in g:
                continue
            q = g[f'q_{edge}_{t}']
            keep = np.isin(q, ('ok', 'multi'))
            raw = g[f'rr_{edge}_{t}_raw_pix']
            ax.plot(g.az_ocs_deg[keep], raw[keep], 'o', ms=3, color=COLORS[m], label=m)
            ax.plot(g.az_ocs_deg[~keep], raw[~keep], 'o', ms=3, mfc='none', color=COLORS[m])
            ax.plot(g.az_ocs_deg, g[f'fit_{edge}_{t}_pix'], '-', lw=0.8, color=COLORS[m])
        for x in g.az_ocs_deg[g.spider]:
            ax.axvspan(x - 1, x + 1, color='gold', alpha=0.25, lw=0)
        ax.set_ylabel(f'r_{edge} (pixels)')
        ax.legend(fontsize=7, ncol=6)
    axes[1].set_xlabel('OCS azimuth (deg)')
    f0 = g.iloc[0]
    fig.suptitle(f'Section 6, robust edges: {cand} {side}, field radius {f0.field_r_deg:.3f} deg, '
                 f'{"clean" if f0.clean else "flagged"}; filled = kept, hollow = rejected; line = '
                 'circle + harmonics to m = 4 (Huber); gold = spider', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def page_robust(pdf, run):
    kept = pd.read_parquet(run / 'robust_kept.parquet')
    st = pd.read_parquet(run / 'edge_stats_robust.parquet')
    old = pd.read_parquet(run.parent / PASS1 / 'edge_stats.parquet')
    um = pd.read_parquet(run / 'unmodelled_surfaces_robust.parquet')
    k = kept[kept.image == 'data'].pivot_table(index=['cand_id', 'side', 'clean'], columns='edge',
                                               values=['kept_of_usable', 'pass1_of_usable'])
    text_page(pdf, 'Section 6: robust edges, fraction of usable bins with an edge (data)',
              table_lines(k, '%.3f'), size=7)
    m = st.merge(old[['cand_id', 'side', 'inner_data_rms_pix', 'outer_data_rms_pix']],
                 on=['cand_id', 'side'], suffixes=('', '_pass1'))
    cols = ['cand_id', 'side', 'field_r_deg', 'clean', 'inner_data_rms_pix', 'inner_data_rms_pix_pass1',
            'outer_data_rms_pix', 'outer_data_rms_pix_pass1', 'inner_v402g_rms_pix', 'outer_v402g_rms_pix']
    text_page(pdf, 'Section 6: edge statistics, RMS against Batoid v4.02, robust vs pass 1, pixels',
              table_lines(m[cols], '%.2f'), size=7)
    t = um.groupby(['clipping_surface', 'edge']).agg(runs=('n_bins', 'size'), bins=('n_bins', 'sum'),
                                                     median_offset_pix=('median_offset_pix',
                                                                        'median'))
    text_page(pdf, 'Section 6: unmodelled surfaces with robust edges (runs of 3+ bins, '
              '|data − Batoid| > 3 pixels)', table_lines(t.sort_values('bins', ascending=False),
                                                         '%+.2f'))


def page_summary(pdf, lines):
    text_page(pdf, 'Pupil pass 2: summary', lines, size=9)


def make_pdf(run, summary_lines):
    from matplotlib.backends.backend_pdf import PdfPages
    bins = pd.read_parquet(run / 'edges_bins_pass2.parquet')
    out = run / 'giant_donut_pupil_pass2.pdf'
    with PdfPages(out) as pdf:
        page_summary(pdf, summary_lines)
        page_state(pdf, run)
        page_coma(pdf, run)
        page_sectors(pdf, run)
        page_ratio(pdf, run)
        page_cases(pdf, run)
        page_robust(pdf, run)
        for (cand, side), _ in bins.groupby(['cand_id', 'side']):
            page_side_edges(pdf, bins, cand, side)
    print('wrote', out)


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
    if 'edges' in steps or 'stats' in steps:
        analyse_edges(run)
    if 'cases' in steps:
        cases(run)
    if 'pdf' in steps:
        summary = run / 'summary.txt'
        make_pdf(run, summary.read_text().split('\n') if summary.exists() else [])


if __name__ == '__main__':
    main()
