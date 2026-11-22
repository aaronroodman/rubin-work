"""Build one ``bounce_aba`` variant: A–B–A Δ, recovery and MC errors, four tables.

Reads the DZ fit table named by the variant (``fam_tables`` for batoid, ``miw`` for an
MIW), applies the bounce visit selection, fits the noise model, runs
`rubinwork.bounce.aba.run`, and writes the tables and the manifest. Nothing here moves
``current``; that is ``python -m rubinwork.products.catalog set-current``.

    python -m rubinwork.products.bounce_aba.builders.build --variant d13-batoid-22_12
"""
import argparse
import datetime
import sys

import pandas as pd

from rubinwork.bounce import aba
from rubinwork.bounce import analysis as ba
from rubinwork.bounce import legs as bl
from rubinwork.common.utils import repo_root

from ... import catalog, manifest
from .. import reader


def compute(variant, fits_build=catalog.CURRENT):
    """Run the reduction of one variant in memory.

    Parameters
    ----------
    variant : `str`
        Variant name.
    fits_build : `str`, optional
        Build of the upstream fit-table product; ``current`` by default.

    Returns
    -------
    tables : `dict` [`str`, `pandas.DataFrame`]
        The four tables of `reader.TABLES`.
    config : `dict`
        The manifest ``config``: the variant entry, the resolved keys and the fitted
        noise model.
    inputs : `dict`
        ``{product: "variant@build"}`` of the fit table read.
    """
    from lsst.ts.intrinsic.wavefront.ofc_svd import project_dz_table

    vc = reader.variant_config(variant)
    param_set, mi_name = reader.mi_keys(variant)
    cfg = bl.resolve_config(param_set, mi_name,
                            analysis_config=repo_root() / 'aos' / 'analysis_config.yaml')
    prod, pvar = reader.fits_input(variant)
    src = catalog.path(prod, pvar, fits_build)
    ft = ba.load_fit_table(src / 'fits.parquet', cfg, min_detectors=int(vc['min_detectors']))
    bounce = next(x for x in cfg['bounces'] if x['name'] == vc['bounce'])
    visits, pairs, legs = aba.leg_tables(ft, bounce, cfg['fit_prefix'], list(vc['day_obs']))
    rec = aba.make_recovery(vc['scheme'], cfg)
    W = project_dz_table(ft, cfg['fit_prefix'], rec.svd)[3]
    i_red = rec.kj.index(tuple(vc['red_term']))
    wn = aba.white_noise(W, pairs)
    rn = aba.red_noise(W, visits, i_red, wn['sigma_w'][i_red], list(vc['lag_bins_min']),
                       float(vc['t0_min']))
    res = aba.run(W, visits, legs, rec, wn, rn, i_red, n_mc=int(vc['n_mc']),
                  seed=int(vc['seed']), t0_min=float(vc['t0_min']))

    nt, nm = len(rec.kj), rec.to_vmode.shape[1]
    leg_rows, dz_rows, st_rows = [], [], []
    for L, r in zip(legs.itertuples(), res):
        base = {'night': int(L.night), 'leg': L.leg, 'b_elev_deg': float(L.b_value_deg)}
        p = float((r['dz'] ** 2).sum())
        leg_rows.append({**base, 'n_A': r['n_A'], 'n_B': r['n_B'], 'n_aba': r['n_aba'],
                         'dz_power_um2': p,
                         'frac_power_left': float((r['resid'] ** 2).sum()) / p,
                         'resid_chi2': float(((r['resid'] / r['resid_err']) ** 2).sum()),
                         'resid_chi2_dof': nt - nm})
        for a, (k, j) in enumerate(rec.kj):
            dz_rows.append({**base, 'k': k, 'j': j, 'name': f'DZ({k},{j})',
                            'delta': r['dz'][a], 'err': r['dz_err'][a],
                            'fit': r['fit'][a], 'fit_err': r['fit_err'][a],
                            'resid': r['resid'][a], 'resid_err': r['resid_err'][a]})
        for a, i in enumerate(rec.dof_index):
            st_rows.append({**base, 'kind': 'dof', 'index': i, 'name': rec.labels[a],
                            'unit': rec.units[a], 'delta': r['dof'][a], 'err': r['dof_err'][a]})
        for m in range(nm):
            st_rows.append({**base, 'kind': 'vmode', 'index': m + 1, 'name': f'v{m + 1}',
                            'unit': 'dimensionless', 'delta': r['vmode'][m],
                            'err': r['vmode_err'][m]})
    noise = pd.DataFrame({'k': [k for k, _ in rec.kj], 'j': [j for _, j in rec.kj],
                          'name': [f'DZ({k},{j})' for k, j in rec.kj],
                          'sigma_w_um': wn['sigma_w']})
    tables = {'aba_legs': pd.DataFrame(leg_rows), 'aba_dz': pd.DataFrame(dz_rows),
              'aba_state': pd.DataFrame(st_rows), 'aba_noise': noise}
    config = {**vc, 'variant': variant, 'param_set': param_set, 'mi_name': mi_name,
              'n_visits_after_cut': len(ft), 'n_terms': nt, 'n_dof': len(rec.dof_index),
              'n_modes': nm,
              'noise_model': {'white_shrink_ledoit_wolf': wn['shrink'],
                              'white_n_pairs': wn['n_pairs'], 'white_n_groups': wn['n_groups'],
                              'red_sigma_r_um': rn['sigma_r'],
                              'red_sigma_r_err_um': rn['sigma_r_err'],
                              'red_alpha': rn['alpha'], 'red_alpha_err': rn['alpha_err'],
                              'red_chi2': rn['chi2'], 'red_dof': rn['dof']}}
    return tables, config, {prod: f'{pvar}@{src.name}'}


def build(variant, build=None, fits_build=catalog.CURRENT, overwrite=False):
    """Compute one variant and write its build directory and manifest.

    Returns
    -------
    build_dir : `pathlib.Path`
    """
    build = build or datetime.datetime.now(datetime.UTC).strftime('%Y%m%d')
    out = reader.build_dir(variant, build)
    if (out / 'manifest.json').exists() and not overwrite:
        raise FileExistsError(f'{out} already has a manifest; pass --overwrite')
    tables, config, inputs = compute(variant, fits_build)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f'{name}.parquet', index=False)
    manifest.write(out, product=reader.PRODUCT, variant=variant, build=build,
                   config=config, inputs=inputs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m rubinwork.products.bounce_aba.builders.build',
                                 description='Build one bounce_aba variant.')
    ap.add_argument('--variant', required=True)
    ap.add_argument('--build', default=None, help='Build name (default: today, YYYYMMDD)')
    ap.add_argument('--fits-build', default=catalog.CURRENT,
                    help='Build of the upstream fam_tables or miw variant')
    ap.add_argument('--overwrite', action='store_true',
                    help='Rewrite a build that already has a manifest')
    args = ap.parse_args(argv)
    out = build(args.variant, args.build, args.fits_build, args.overwrite)
    legs = pd.read_parquet(out / 'aba_legs.parquet')
    print(f'wrote {out}: {len(legs)} (night, leg)')
    print(legs[['night', 'leg', 'n_aba', 'frac_power_left']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
