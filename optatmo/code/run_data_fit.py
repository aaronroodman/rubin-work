"""First data fit: seq 31 & 34 (20260513), compare fitted DZ to CWFS."""
import numpy as np, pandas as pd
from config import load_config
import data_fit

VISITS = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'
DAY = 20260513


def rot_for(infocus_seq):
    v = pd.read_parquet(VISITS)
    r = v[(v.day_obs == DAY) & (v.seq_num == infocus_seq - 1)]
    return float(r.rotator_angle.iloc[0]) if len(r) else np.nan


def main():
    cfg = load_config('config.yaml')
    # faster model grid for the fit (moments are scale-independent)
    cfg['geometry']['stamp'] = 24
    cfg['geometry']['oversample'] = 12
    cfg['atmosphere']['kernel'] = 'Kolmogorov'
    cfg['dz_terms'] = [
        {'pupil': 4, 'focal': [1]},
        {'pupil': 5, 'focal': [1, 2, 3]}, {'pupil': 6, 'focal': [1, 2, 3]},
        {'pupil': 7, 'focal': [1, 2, 3]}, {'pupil': 8, 'focal': [1, 2, 3]},
        {'pupil': 9, 'focal': [1]}, {'pupil': 10, 'focal': [1]},
        {'pupil': 11, 'focal': [1]}]
    cfg['atmosphere']['fit'] = ['fwhm', 'g1', 'g2']
    cfg['fit']['moments'] = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03']
    cfg['fit']['weights'] = {m: 1.0 for m in cfg['fit']['moments']}

    out = {}
    for seq in [31, 34]:
        rot = rot_for(seq)
        print(f'\n========== seq {seq} (rotator {rot:.2f} deg) ==========')
        pq = f'data/psfmoments_{DAY}000{seq}.parquet'
        fitted, cw, res, layout = data_fit.fit_visit(
            cfg, pq, DAY, seq, sign=1, cell_deg=0.30, rot_deg=rot)
        out[seq] = (fitted, cw, layout)

    # OCS consistency: fitted DZ should agree between the two rotators
    print('\n===== OCS DZ consistency (seq31 vs seq34) & CWFS =====')
    lay = out[31][2]
    print(f'{"coeff":8} {"fit31":>9} {"fit34":>9} {"CWFS31":>9} {"CWFS34":>9}')
    for n in lay.dz_names:
        f31, c31, _ = out[31]
        f34, c34, _ = out[34]
        print(f'{n:8} {f31[n]:9.4f} {f34[n]:9.4f} '
              f'{(c31 or {}).get(n, np.nan):9.4f} {(c34 or {}).get(n, np.nan):9.4f}')


if __name__ == '__main__':
    main()
