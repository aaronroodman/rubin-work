"""Score giant-donut selection variants against the hand labels.

Truth comes from the pages marked done in ``hand_labels_pages.json``. Every
labelled item on those pages is a truth donut: catalog stars by refcat_id,
unexplained peaks by peak_id, and added donuts matched to the nearest catalog star
(to 18.5 AB mag) within 0.3 r_out, if any.

Each variant classes every star row isolated / marginal / blended / saturated (or
leaves it unscored) and keeps the isolated ones. Per sensor class and overall:

- completeness: good-labelled donuts kept, over all good-labelled donuts;
- purity: kept donuts labelled good, over kept donuts that carry a label;
- the confusion matrix, hand label against the variant's class.

The variants change one thing each from the 20261018 baseline; ``combined`` takes
every change that raises completeness without lowering purity, and every image
check (fit residual, bleed) that raises purity without lowering completeness.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/score_selection.py --diag 20261020_selection_diag
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import select_candidates as sc  # noqa: E402

TRUTH_LABELS = ('good', 'marginal', 'blended', 'not_donut')
CLASSES = ('isolated', 'marginal', 'blended', 'saturated', 'unscored')
MATCH_ROUT = 0.3
# Variant parameters.
SEPARATION_ROUT = 2.0
NBR_ROUT_D4 = 2.0
BLEED_MAX = 0.5
# The residual cut sits at this quantile of the good-labelled donuts' residuals.
RESID_QUANTILE = 0.98
PROMOTERS = ('dmag4', 'sigma5', 'sep_rout', 'best_mag')
DEMOTERS = ('fit_resid', 'saturation_bleed')


def classify(t, catalog_ok, annulus, overlap, saturated, scored):
    """Class per row from a catalog verdict and the image metrics."""
    clean = (annulus < sc.ANNULUS_CLEAN) & (overlap < sc.OVERLAP_CLEAN)
    marg = (annulus < sc.ANNULUS_MARGINAL) & (overlap < sc.OVERLAP_MARGINAL)
    image = np.where(clean, 'clean', np.where(marg, 'marginal', 'contaminated'))
    cls = np.full(len(t), 'blended', dtype=object)
    cls[catalog_ok & (image == 'clean')] = 'isolated'
    cls[(catalog_ok & (image == 'marginal')) | (~catalog_ok & (image == 'clean'))] = 'marginal'
    cls[saturated] = 'saturated'
    cls[~scored] = 'unscored'
    return cls


def variants(t, resid_max):
    """Every variant's class column."""
    in_r = t.full_on_silicon & np.isfinite(t.mag_r) & t.mag_r.between(sc.MAG_MIN, sc.MAG_MAX)
    in_best = t.full_on_silicon & t.mag_best.between(sc.MAG_MIN, sc.MAG_MAX)
    blitz = t.blitz_isolated.to_numpy(bool)
    ann, ovl = t.annulus_ratio.to_numpy(), t.overlap_frac.to_numpy()
    sat = (t.ring_p99_e > sc.SATURATION_E).to_numpy()
    out = {'baseline': classify(t, blitz, ann, ovl, sat, in_r)}
    out['dmag4'] = classify(t, (t.nbr_dist_rout >= NBR_ROUT_D4).to_numpy(), ann,
                            t.overlap_frac_d4.to_numpy(), sat, in_r)
    out['sigma5'] = classify(t, blitz, ann, t.overlap_frac_sigma.to_numpy(), sat, in_r)
    out['sep_rout'] = classify(t, (t.nbr2r_dist_rout >= SEPARATION_ROUT).to_numpy(), ann, ovl,
                               sat, in_r)
    out['best_mag'] = classify(t, t.blitz_isolated_best.to_numpy(bool), ann,
                               t.overlap_frac_best.to_numpy(), sat, in_best)
    fit_bad = ~(t.annulus_resid < resid_max).to_numpy()
    base = out['baseline'].copy()
    base[(base == 'isolated') & fit_bad] = 'marginal'
    out['fit_resid'] = base
    bleed = sat | (t.bleed_ratio > BLEED_MAX).to_numpy()
    out['saturation_bleed'] = classify(t, blitz, ann, ovl, bleed, in_r)
    return out


def truth_table(stars, labels, pages_done):
    """One row per truth donut, with the star row it matches (``row``, -1 if none)."""
    done = {k for k, v in pages_done.items() if v}
    lab = labels[[f'{v}_{r}' in done for v, r in zip(labels.visit, labels.raft)]].copy()
    lab['row'] = -1
    idx = {(v, rid): i for i, (v, rid) in enumerate(zip(stars.visit, stars.refcat_id))}
    for i, r in lab.iterrows():
        if r['item'] == 'star':
            lab.at[i, 'row'] = idx.get((r.visit, int(r.refcat_id)), -1)
            continue
        # Peaks and added donuts: nearest star row within MATCH_ROUT r_out.
        s = stars[(stars.visit == r.visit) & (stars.detector == r.detector)]
        if len(s):
            d = np.hypot(s.fp_x_mm - r.fp_x_mm, s.fp_y_mm - r.fp_y_mm) / (s.r_out_pix * 0.01)
            j = int(np.argmin(d.to_numpy()))
            if d.iloc[j] < MATCH_ROUT:
                lab.at[i, 'row'] = s.index[j]
    lab['sensor_class'] = [stars.sensor_class.loc[row] if row >= 0 else
                           ('science' if 'SW' not in det else det[-3:])
                           for row, det in zip(lab.row, lab.detector)]
    return lab


def score(truth, cls):
    """Completeness, purity and the confusion matrix for one class column."""
    got = np.array([cls[row] if row >= 0 else 'unscored' for row in truth.row])
    out = {}
    for group, mask in [('all', np.ones(len(truth), bool))] + [
            (c, (truth.sensor_class == c).to_numpy()) for c in ('science', 'SW0', 'SW1')]:
        lab, g = truth.label.to_numpy()[mask], got[mask]
        good = lab == 'good'
        kept = g == 'isolated'
        out[group] = dict(
            n_truth=int(mask.sum()), n_good=int(good.sum()), n_kept=int(kept.sum()),
            n_kept_good=int((good & kept).sum()),
            completeness=float((good & kept).sum() / good.sum()) if good.sum() else np.nan,
            purity=float((good & kept).sum() / kept.sum()) if kept.sum() else np.nan,
            confusion={t: {c: int(((lab == t) & (g == c)).sum()) for c in CLASSES}
                       for t in TRUTH_LABELS})
    return out


def run(diag_dir):
    stars = pd.read_parquet(diag_dir / 'stars.parquet')
    labels = pd.read_csv(diag_dir / 'hand_labels.csv')
    pages_done = json.loads((diag_dir / 'hand_labels_pages.json').read_text())
    truth = truth_table(stars, labels, pages_done)
    good_rows = truth.row[(truth.label == 'good') & (truth.row >= 0)]
    resid_max = float(np.nanquantile(stars.annulus_resid.loc[good_rows], RESID_QUANTILE)) \
        if len(good_rows) else np.inf
    classes = variants(stars, resid_max)
    scores = {name: score(truth, cls) for name, cls in classes.items()}
    base = scores['baseline']['all']
    chosen = []
    for name, s in scores.items():
        c, p = s['all']['completeness'], s['all']['purity']
        if name in PROMOTERS and c > base['completeness'] and p >= base['purity']:
            chosen.append(name)
        # The two image checks only demote, so they cannot raise completeness;
        # they are kept if they raise purity at no cost in completeness.
        if name in DEMOTERS and p > base['purity'] and c >= base['completeness']:
            chosen.append(name)
    classes['combined'] = combine(stars, chosen, resid_max)
    scores['combined'] = score(truth, classes['combined'])
    return stars, truth, classes, scores, chosen, resid_max


def combine(t, chosen, resid_max):
    """The baseline with every chosen change applied together."""
    use_best = 'best_mag' in chosen
    scored = t.full_on_silicon & (t.mag_best.between(sc.MAG_MIN, sc.MAG_MAX) if use_best else
                                  (np.isfinite(t.mag_r) & t.mag_r.between(sc.MAG_MIN, sc.MAG_MAX)))
    if 'dmag4' in chosen:
        catalog_ok = (t.nbr_dist_rout >= NBR_ROUT_D4).to_numpy()
    elif 'sep_rout' in chosen:
        catalog_ok = ((t.nbr2b_dist_rout if use_best else t.nbr2r_dist_rout)
                      >= SEPARATION_ROUT).to_numpy()
    else:
        catalog_ok = (t.blitz_isolated_best if use_best else t.blitz_isolated).to_numpy(bool)
    if 'dmag4' in chosen:
        overlap = t.overlap_frac_d4
    elif 'sigma5' in chosen:
        overlap = t.overlap_frac_sigma
    elif use_best:
        overlap = t.overlap_frac_best
    else:
        overlap = t.overlap_frac
    sat = (t.ring_p99_e > sc.SATURATION_E).to_numpy()
    if 'saturation_bleed' in chosen:
        sat = sat | (t.bleed_ratio > BLEED_MAX).to_numpy()
    cls = classify(t, catalog_ok, t.annulus_ratio.to_numpy(), overlap.to_numpy(), sat,
                   scored.to_numpy())
    if 'fit_resid' in chosen:
        cls[(cls == 'isolated') & ~(t.annulus_resid < resid_max).to_numpy()] = 'marginal'
    return cls


def summary_frame(scores):
    rows = []
    for name, s in scores.items():
        for group, v in s.items():
            rows.append(dict(variant=name, group=group, n_truth=v['n_truth'], n_good=v['n_good'],
                             n_kept=v['n_kept'], n_kept_good=v['n_kept_good'],
                             completeness=v['completeness'], purity=v['purity']))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--diag', required=True, help='selection_diag run under output/')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    diag_dir = study_dir / 'output' / args.diag
    stars, truth, classes, scores, chosen, resid_max = run(diag_dir)
    table = summary_frame(scores)
    table.to_csv(diag_dir / 'score_summary.csv', index=False, float_format='%.3f')
    (diag_dir / 'score_detail.json').write_text(json.dumps(dict(
        chosen=chosen, resid_max=resid_max, scores=scores), indent=1, default=str))
    truth.to_csv(diag_dir / 'truth_matched.csv', index=False, float_format='%.3f')
    print(table[table.group == 'all'].round(3).to_string(index=False))
    print('chosen', chosen, 'annulus_resid max', round(resid_max, 3))


if __name__ == '__main__':
    main()
