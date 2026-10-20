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


if __name__ == '__main__':
    main()
