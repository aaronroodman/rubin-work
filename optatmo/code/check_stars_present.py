"""Check whether the PSF-star dataset is present for a block's visits on given
nights, reporting present-vs-expected per night (RUN ON USDF).

The optatmo fit needs `single_visit_star` (and, for stamps,
`preliminary_visit_image`) from a DRP collection (default
LSSTCam/runs/nightlyValidation).  This enumerates the block's visits for each
night from the Butler visit dimension (the denominator) and how many have the
dataset (the numerator), so you can scope which nights/visits are fittable.

    python code/check_stars_present.py --days 20260331 20260512 --block BLOCK-T698
    python code/check_stars_present.py --days 20260512 --block BLOCK-T698 \
        --reason fbs_driven_aos_stability_test --list-missing
    python code/check_stars_present.py --days 20260512 --dataset preliminary_visit_image
"""
import argparse

from lsst.daf.butler import Butler


def _where(block, reason):
    w = ["instrument='LSSTCam'", "visit.day_obs=day"]
    bind = {}
    if block:
        w.append("visit.science_program=blk"); bind['blk'] = block
    if reason:
        w.append("visit.observation_reason=rsn"); bind['rsn'] = reason
    return " and ".join(w), bind


def expected_visits(b, day, block, reason):
    """All visits of the block/reason on this night (Butler visit dimension)."""
    where, bind = _where(block, reason)
    bind['day'] = day
    try:
        recs = b.registry.queryDimensionRecords('visit', where=where, bind=bind)
        return sorted(r.id for r in recs)
    except Exception as e:                    # science_program/reason not queryable
        if block or reason:
            print(f'  (visit metadata filter failed [{e.__class__.__name__}]; '
                  f'falling back to day-only)')
            recs = b.registry.queryDimensionRecords(
                'visit', where="instrument='LSSTCam' and visit.day_obs=day",
                bind={'day': day})
            return sorted(r.id for r in recs)
        raise


def present_visits(b, day, block, reason, dataset, collection):
    where, bind = _where(block, reason)
    bind['day'] = day
    try:
        refs = b.registry.queryDatasets(dataset, collections=collection,
                                        findFirst=True, where=where, bind=bind)
    except Exception:
        refs = b.registry.queryDatasets(
            dataset, collections=collection, findFirst=True,
            where="instrument='LSSTCam' and visit.day_obs=day", bind={'day': day})
    return sorted({r.dataId['visit'] for r in refs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, nargs='+', required=True)
    ap.add_argument('--block', default=None, help='science_program, e.g. BLOCK-T698')
    ap.add_argument('--reason', default=None, help='observation_reason filter')
    ap.add_argument('--dataset', default='single_visit_star')
    ap.add_argument('--collection', default='LSSTCam/runs/nightlyValidation')
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--list-missing', action='store_true')
    ap.add_argument('--list-present', action='store_true')
    args = ap.parse_args()

    b = Butler(args.repo)
    tag = ' '.join(x for x in (args.block, args.reason) if x) or 'all visits'
    print(f'{args.dataset} in {args.collection}  ({tag})\n')
    for day in args.days:
        exp = expected_visits(b, day, args.block, args.reason)
        pres = present_visits(b, day, args.block, args.reason, args.dataset,
                              args.collection)
        exp_s, pres_s = set(exp), set(pres)
        missing = sorted(exp_s - pres_s)
        # datasets present but not in the expected set (e.g. block filter mismatch)
        extra = sorted(pres_s - exp_s)
        rng = f'{exp[0]}..{exp[-1]}' if exp else '-'
        print(f'{day}: {len(pres_s & exp_s)}/{len(exp)} visits have {args.dataset}'
              f'   (expected range {rng}; {len(missing)} missing'
              + (f', {len(extra)} present outside expected set' if extra else '') + ')')
        if args.list_missing and missing:
            print('   missing:', missing)
        if args.list_present and pres:
            print('   present:', sorted(pres_s & exp_s))


if __name__ == '__main__':
    main()
