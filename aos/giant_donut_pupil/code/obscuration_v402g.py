"""danish obscuration files for v4.02 fitted at the giant-donut configurations (v4.02g).

`obscuration_v402.py` fits the circles and the camera silhouette on the nominal
v4.02 optic. The giant donuts are taken with the camera and M2 both moved
±4 mm in z, so the shadows of the camera elements and of M2 land elsewhere on
the stop plane. Here the same fit (`fitMaskModel` on the reduced v4.02, one
silhouette circle per field angle, CAD spider vanes) is run on v4.02 with the
(detector, camera, M2) z offsets applied. Only the z offsets are applied: no
Trim, no LUT and no measured optical state, since the AOS Trim returns M2 and
the camera to the as-built alignment the model already carries.

Also here:

* `validate`: agreement of v4.02 (nominal fit) and v4.02g with a Batoid trace
  of the full v4.02 at the same offsets, at field radii 0-1.9 deg and 8
  azimuths, since the as-built decentres break the azimuthal symmetry
  `fitMaskModel` assumes.
* `unique_cuts_field`: the "clip no ray" check of the in-camera openings
  (``CamAperture_*``), over the full field and all azimuths at the giant
  configurations.
* `edge_shift`: the move of each fitted edge circle between the nominal and
  giant fits, mm on the stop plane.

Run, from ``code/``::

    python obscuration_v402g.py --run ../output/20261019_pupil2
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import obscuration_v402 as ob  # noqa: E402  (puts the blitz checkout first on sys.path)

STUDY = 'aos/giant_donut_pupil'
NOMINAL_MASK = pathlib.Path(__file__).resolve().parents[1] / 'output' / \
    '20261018_obscuration_v402' / 'RubinObsc_v4.02_r_rtpp0_pp0d0.yaml'
# (detector, camera, M2) z offsets, mm, positive extra-focal.
CONFIGS_MM = ((0.0, 4.0, 4.0), (0.0, -4.0, -4.0), (1.5, 4.0, 4.0), (-1.5, 4.0, 4.0),
              (1.5, -4.0, -4.0), (-1.5, -4.0, -4.0))
VALIDATION_RADII_DEG = (0.0, 1.0, 1.5, 1.65, 1.75, 1.9)
VALIDATION_AZ_DEG = tuple(range(0, 360, 45))
CUT_RADII_DEG = (0.5, 1.0, 1.25, 1.5, 1.65, 1.75, 1.85, 1.95)
CUT_AZ_DEG = tuple(range(0, 360, 30))

__all__ = ['mask_name', 'mask_path', 'shift', 'fit_config', 'validate', 'unique_cuts_field',
           'edge_shift']


def mask_name(offsets_mm):
    """File name of the v4.02g mask for (detector, camera, M2) offsets in mm."""
    d, c, m = offsets_mm
    return f'RubinObsc_v4.02g_r_det{d:+.1f}_cam{c:+.1f}_m2{m:+.1f}.yaml'


def mask_path(run_dir, offsets_m):
    """Path of the v4.02g mask for a donut side's offsets, given in m."""
    mm = tuple(round(o * 1e3, 1) + 0.0 for o in offsets_m)
    return pathlib.Path(run_dir) / 'masks' / mask_name(mm)


def shift(optic, offsets_mm):
    """Shift Detector, LSSTCamera and M2 in z by `offsets_mm`."""
    for name, dz in zip(('Detector', 'LSSTCamera', 'M2'), offsets_mm):
        if dz:
            optic = optic.withGloballyShiftedOptic(name, [0.0, 0.0, dz * 1e-3])
    return optic


def fit_config(offsets_mm, dtheta=0.02, theta_max=2.0):
    """`obscuration_v402.build`, with the optic shifted by `offsets_mm`.

    Returns
    -------
    params : `dict`
        danish ``mask_params``.
    silhouette : `list` [`dict`]
        Per field angle, the silhouette circle and its RMS (m).
    """
    from lsst.ts.wep.utils import maskUtils
    maskUtils._fitCircle = ob._fit_circle_fixed
    optic = shift(ob.reduced_v402(), offsets_mm)
    fit, _ = maskUtils.fitMaskModel(optic, wavelength=ob.WAVELENGTH_M, deg=3,
                                    thetaMax=theta_max, dTheta=dtheta)
    fit = {k: v for k, v in fit.items()
           if any(np.isfinite(e['thetaMin']) for e in v.values())}
    for v in fit.values():
        for e in [k for k, x in v.items() if not np.isfinite(x['thetaMin'])]:
            v.pop(e)
    pruned = maskUtils.pruneMaskModel(fit)
    params = {item: {edge: dict(clear=bool(e['clear']), thetaMin=float(e['thetaMin']),
                                thetaMax=float(e['thetaMax']),
                                center=[float(c) for c in e['center']],
                                radius=[float(c) for c in e['radius']])
                     for edge, e in edges.items()}
              for item, edges in pruned.items()}

    sil_optic = shift(ob.planes_optic(('CameraSilhouette',)), offsets_mm)
    thetas = np.arange(0.0, 2.0001, 0.1)
    table = []
    for th in thetas:
        x, y = ob.silhouette_edge(sil_optic, th)
        c, r, rms = ob._circle_fit(x, y)
        table.append(dict(theta_deg=float(th), center_m=c, radius_m=r, rms_m=rms))
    params['CameraSilhouette'] = {'outer': dict(
        clear=False, thetaMin=0.0, thetaMax=float(thetas[-1]),
        center=[float(v) for v in np.polyfit(thetas, [t['center_m'] for t in table], 3)],
        radius=[float(v) for v in np.polyfit(thetas, [t['radius_m'] for t in table], 3)])}
    z_stop = float(ob._v402_dict()['opticalSystem']['stopSurface']['coordSys']['z'])
    params['Spider_3D'] = ob.cad_spider_vanes(z_stop)
    return params, table


def _rays(optic, thx, thy, nrad=150, naz=1000, outer=4.3):
    import batoid
    return batoid.RayVector.asPolar(optic=optic, wavelength=ob.WAVELENGTH_M, theta_x=thx,
                                    theta_y=thy, inner=0.0, outer=outer, nrad=nrad, naz=naz)


def batoid_keep(optic, offsets_mm, thx, thy, nrad=150, naz=1000):
    """Stop-plane coordinates (m) and Batoid transmission at field (thx, thy), rad."""
    rays = _rays(optic, thx, thy, nrad, naz)
    u, v = ob._stop_xy(optic, rays)
    out = shift(optic, offsets_mm).trace(rays.copy())
    return u, v, ~out.vignetted & ~out.failed


def _validate_one(task):
    offsets_mm, r_deg, az_deg, masks = task
    optic = ob.v402_optic()
    thx = np.deg2rad(r_deg * np.cos(np.deg2rad(az_deg)))
    thy = np.deg2rad(r_deg * np.sin(np.deg2rad(az_deg)))
    u, v, kept = batoid_keep(optic, offsets_mm, thx, thy)
    rows = []
    for name, params in masks.items():
        r_outer, r_inner = ob._annulus(params)
        dk = ob.danish_keep(params, u, v, thx, thy, spider=True, r_outer=r_outer, r_inner=r_inner)
        n_t = max(kept.sum(), 1)
        rows.append(dict(mask=name, det_mm=offsets_mm[0], cam_mm=offsets_mm[1],
                         m2_mm=offsets_mm[2], field_r_deg=r_deg, field_az_deg=az_deg,
                         xor_over_T=float(np.sum(dk != kept) / n_t),
                         danish_extra=float(np.sum(dk & ~kept) / n_t),
                         danish_missing=float(np.sum(~dk & kept) / n_t)))
    return rows


def validate(masks_by_config, nominal, workers):
    """Agreement table of the nominal and giant masks with the full v4.02 trace.

    ``xor_over_T`` is the number of stop-plane rays on which danish and Batoid
    disagree over the number Batoid transmits; ``danish_extra`` and
    ``danish_missing`` split it into rays only danish keeps and only Batoid keeps.
    All three are dimensionless.
    """
    import multiprocessing as mp
    tasks = [(off, r, az, {'v4.02': nominal, 'v4.02g': masks_by_config[off]})
             for off in masks_by_config for r in VALIDATION_RADII_DEG
             for az in (VALIDATION_AZ_DEG if r > 0 else (0,))]
    with mp.get_context('fork').Pool(workers) as pool:
        out = pool.map(_validate_one, tasks, chunksize=1)
    return pd.DataFrame([row for rows in out for row in rows])


def _cuts_one(task):
    """Fraction of transmitted rays that only the CamAperture planes (or the silhouette) stop."""
    import re
    offsets_mm, r_deg, az_deg = task
    optic = ob.v402_optic()
    shifted = shift(optic, offsets_mm)
    thx = np.deg2rad(r_deg * np.cos(np.deg2rad(az_deg)))
    thy = np.deg2rad(r_deg * np.sin(np.deg2rad(az_deg)))
    trace = shifted.traceFull(_rays(optic, thx, thy, nrad=100, naz=650, outer=4.25))
    groups = {}
    for key, value in trace.items():
        item = shifted.itemDict[shifted._names[key]]
        if item.obscuration is None:
            continue
        r = value['out']
        group = re.sub(r'_z\d.*$', '', re.sub(r'\d{3}_(top|bottom)$', '', key.split('.')[-1]))
        groups[group] = groups.get(group, False) | item.obscuration.contains(r.x, r.y) | r.failed
    any_block = np.logical_or.reduce(list(groups.values()))
    n_t = max((~any_block).sum(), 1)
    row = dict(det_mm=offsets_mm[0], cam_mm=offsets_mm[1], m2_mm=offsets_mm[2],
               field_r_deg=r_deg, field_az_deg=az_deg, n_transmitted=int(n_t))
    for g in ('CamAperture', 'CameraSilhouette', 'Filter_entrance', 'L3_exit'):
        if g in groups:
            others = np.logical_or.reduce([c for h, c in groups.items() if h != g])
            row[f'unique_{g}'] = float((groups[g] & ~others).sum() / n_t)
            row[f'blocks_{g}'] = float(groups[g].sum() / n_t)
    return row


def unique_cuts_field(workers):
    """The in-camera-opening check over the full field, both giant configurations."""
    import multiprocessing as mp
    tasks = [(off, r, az) for off in CONFIGS_MM[:2] for r in CUT_RADII_DEG for az in CUT_AZ_DEG]
    with mp.get_context('fork').Pool(workers) as pool:
        return pd.DataFrame(pool.map(_cuts_one, tasks, chunksize=1))


def edge_shift(nominal, giant, offsets_mm, thetas=(0.0, 0.5, 1.0, 1.5, 1.65, 1.75, 1.9)):
    """Centre and radius of each edge circle, giant minus nominal, mm on the stop plane."""
    rows = []
    for item, edges in giant.items():
        if item == 'Spider_3D':
            continue
        for edge, e in edges.items():
            n = nominal.get(item, {}).get(edge)
            for th in thetas:
                row = dict(det_mm=offsets_mm[0], cam_mm=offsets_mm[1], m2_mm=offsets_mm[2],
                           item=item, edge=edge, field_deg=th,
                           active_giant=bool(e['thetaMin'] <= th <= e['thetaMax']))
                row['center_giant_mm'] = float(np.polyval(e['center'], th)) * 1e3
                row['radius_giant_mm'] = float(np.polyval(e['radius'], th)) * 1e3
                if n is not None:
                    row['active_nominal'] = bool(n['thetaMin'] <= th <= n['thetaMax'])
                    row['center_nominal_mm'] = float(np.polyval(n['center'], th)) * 1e3
                    row['radius_nominal_mm'] = float(np.polyval(n['radius'], th)) * 1e3
                    row['d_center_mm'] = row['center_giant_mm'] - row['center_nominal_mm']
                    row['d_radius_mm'] = row['radius_giant_mm'] - row['radius_nominal_mm']
                rows.append(row)
    return pd.DataFrame(rows)


def _fit_work(offsets_mm):
    t0 = time.time()
    params, table = fit_config(offsets_mm)
    print(json.dumps(dict(offsets_mm=offsets_mm, items=sorted(params),
                          s=round(time.time() - t0))), flush=True)
    return offsets_mm, params, table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, help='the pupil2 run directory')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--skip-fit', action='store_true', help='reuse masks/ already written')
    args = ap.parse_args()
    import multiprocessing as mp
    run = pathlib.Path(args.run).resolve()
    (run / 'masks').mkdir(parents=True, exist_ok=True)
    nominal = yaml.safe_load(NOMINAL_MASK.read_text())

    if not args.skip_fit:
        with mp.get_context('fork').Pool(len(CONFIGS_MM)) as pool:
            fits = pool.map(_fit_work, CONFIGS_MM, chunksize=1)
        sil = []
        for off, params, table in fits:
            (run / 'masks' / mask_name(off)).write_text(
                yaml.safe_dump(params, sort_keys=False, default_flow_style=None))
            sil += [dict(det_mm=off[0], cam_mm=off[1], m2_mm=off[2], **t) for t in table]
        pd.DataFrame(sil).to_csv(run / 'v402g_silhouette.csv', index=False, float_format='%.6g')
    masks = {off: yaml.safe_load((run / 'masks' / mask_name(off)).read_text()) for off in CONFIGS_MM}

    shifts = pd.concat([edge_shift(nominal, masks[off], off) for off in CONFIGS_MM])
    for name, df in (('v402g_edge_shift', shifts),):
        df.to_parquet(run / f'{name}.parquet')
        df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')
    print('edge shifts written', flush=True)

    val = validate(masks, nominal, args.workers)
    val.to_parquet(run / 'v402g_validation.parquet')
    val.to_csv(run / 'v402g_validation.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 200)
    print(val.pivot_table(index=['det_mm', 'cam_mm', 'field_r_deg'], columns='mask',
                          values='xor_over_T', aggfunc=['median', 'max']).round(4).to_string(),
          flush=True)

    cuts = unique_cuts_field(args.workers)
    cuts.to_parquet(run / 'camaperture_cuts.parquet')
    cuts.to_csv(run / 'camaperture_cuts.csv', index=False, float_format='%.5g')
    print(cuts.groupby(['cam_mm', 'field_r_deg'])[[c for c in cuts if c.startswith('unique_')]]
          .max().round(5).to_string())
    print('done', run)


if __name__ == '__main__':
    main()
