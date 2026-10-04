#!/usr/bin/env python3
"""run_blitz_mktable — build Danish 1.2-schema wavefront tables from Danish 1.3 blitz
unpaired Full Array Mode (FAM) results.

The Danish 1.3 "blitz" pipeline writes one ``donutBlitzFamResults`` table per FAM visit,
holding one row per donut per exposure — both sides of focus in one table, "unpaired".
Every downstream stage in this topic instead expects the paired Danish 1.2 schema that
``run_mktable`` from the ``ts_intrinsic_wavefront`` package produces: a ``donuts.parquet``
with one row group per visit, a ``visits.parquet`` sidecar, and from those a
``fits.parquet`` of Double Zernike (DZ) coefficients. This script produces the first two
so that the existing DZ fitting machinery runs on the blitz data unchanged, then optionally
runs that fit.

The recast itself — the join of the two sides of focus, the Zernike index selection, the
one-row-group-per-visit writer — lives in `blitz_reader`, which is written to be liftable
into the external package. This script is the repository-specific wrapper: it resolves a
``param_set`` from ``aos/param_sets.yaml``, resolves the output directory through
``output_paths``, and merges the per-visit fields the blitz metadata cannot supply from the
Consolidated Database (ConsDB).

Intrinsic wavefront
-------------------
The DZ fit uses the **default Batoid intrinsic wavefront carried in the blitz dataset
type** — the ``zk_intrinsic_ocs`` column, traceable through the table's Butler input
provenance to the ``intrinsicZernikes`` calibration run. No Measured Intrinsic Wavefront
(MIW) sidecar is passed, so `lsst.ts.intrinsic.wavefront.dz_fitting` takes the intrinsic
straight from the donut table. The calibration run collection is recorded in
``provenance.yaml`` beside the output tables.

Outputs, under ``aos/output/fam_processing/<dir_name>/``:

  ``donuts.parquet``     one row per paired donut, one row group per visit
  ``visits.parquet``     one row per visit, the 19 columns ``run_mktable`` produces
  ``fits.parquet``       the k=1..3 and k=1..6 DZ fit results (with ``--fit``)
  ``provenance.yaml``    collection, dataset type, intrinsic calibration run, versions

Usage:
  # One visit, to validate
  python code/fam_processing/run_blitz_mktable.py --param-set danish_1_3_test \\
      --visits 2026031500122 --fit --overwrite

  # One night
  python code/fam_processing/run_blitz_mktable.py --param-set danish_1_3_test \\
      --day-obs 20260315 --fit --overwrite

  # Everything in the collection
  python code/fam_processing/run_blitz_mktable.py --param-set danish_1_3_test \\
      --fit --overwrite

Key arguments:
  ``--param-set``   entry in ``aos/param_sets.yaml`` giving the Butler repo and collection
  ``--visits``      explicit visit identifiers of FAM extra-focal exposures
  ``--day-obs``     restrict to one or more nights, as ``YYYYMMDD``
  ``--max-visits``  process at most this many visits, dimensionless, for a quick check
  ``--mode``        ``mean`` (default), ``intra`` or ``extra`` — see `blitz_reader`
  ``--fit``         run the DZ fit after writing the tables
  ``--no-consdb``   skip the ConsDB merge, leaving ``ra``/``dec``/``band``/
                    ``science_program``/``rotator_angle`` unpopulated

Notes
-----
Needs the Butler and, unless ``--no-consdb`` is passed, ConsDB — so it runs on the Rubin
Science Platform (RSP) or an interactive USDF node, **not** on a batch compute node.

A visit whose blitz table is missing, empty, or has no star fitted on both sides of focus
is skipped with a message and counted; it does not abort the run.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # aos/code -> flat modules
sys.path.insert(0, str(Path(__file__).resolve().parent))       # this study's modules

from output_paths import study_dir  # noqa: E402
import blitz_reader as br  # noqa: E402

TOPIC = Path(__file__).resolve().parents[2]

# Per-visit quality-cut thresholds, matching the run_mktable defaults so that a blitz
# table and a Danish 1.2 table are cut the same way.
DEFAULT_MIN_DONUTS_PER_VISIT = 500
DEFAULT_MIN_DONUTS_PER_DETECTOR = 3
DEFAULT_MIN_DETECTORS_PER_VISIT = 170
DEFAULT_MAX_MEDIAN_BLUR_ARCSEC = 1.2

# Flag a visit whose camera rotator angle exceeds this, in deg, as run_mktable does.
DEFAULT_ROTATOR_THRESHOLD_DEG = 90.0

DEFAULT_CONSDB_URL = 'https://usdf-rsp.slac.stanford.edu/consdb'

# The 19 columns run_mktable writes into visits.parquet before any telemetry is attached,
# in its order, followed by this reader's additions.  run_attach_telemetry.py adds the
# rest later.
VISITS_COLUMNS = (
    'day_obs', 'seq_num', 'visit', 'skyAngle', 'ra', 'dec', 'az', 'alt', 'band', 'mjd',
    'nollIndices', 'n_donuts', 'n_detectors', 'n_detectors_with_min_donuts',
    'median_blur_arcsec', 'science_program', 'rotator_angle', 'rotator_flagged',
    'visit_quality_pass',
    # Beyond the run_mktable schema: the per-side blur medians, so the intra/extra
    # difference is available per visit without re-reading the blitz collection.
    'median_blur_intra_arcsec', 'median_blur_extra_arcsec',
    # The two exposure ids of the FAM pair. `visit` is the extra-focal one, so without
    # `intra_visit` there is no key to join per-side telemetry on.
    'intra_visit', 'extra_visit',
)


def parse_args(argv=None):
    """Command-line interface.

    Returns
    -------
    args : `argparse.Namespace`
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build Danish 1.2-schema wavefront tables from Danish 1.3 blitz '
                    'unpaired FAM results.')
    parser.add_argument('--param-set', required=True,
                        help='Entry in aos/param_sets.yaml giving butler_repo and '
                             'fam_collections')
    parser.add_argument('--visits', nargs='+', type=int, default=None,
                        help='Visit identifiers of FAM extra-focal exposures to process '
                             '(default: every visit in the collection)')
    parser.add_argument('--day-obs', nargs='+', type=int, default=None,
                        help='Restrict to these nights, as YYYYMMDD')
    parser.add_argument('--max-visits', type=int, default=None,
                        help='Process at most this many visits (dimensionless), for a '
                             'quick check')
    parser.add_argument('--mode', default='mean', choices=['mean', 'intra', 'extra'],
                        help="Pairing mode: 'mean' of the two sides of focus (default), "
                             "or a single unpaired side")
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: '
                             'output/fam_processing/<dir_name>/ under aos/)')
    parser.add_argument('--dataset-type', default=br.BLITZ_FAM_DATASET,
                        help=f'Butler dataset type (default: {br.BLITZ_FAM_DATASET})')
    parser.add_argument('--fit', action='store_true',
                        help='Run the k=1..3 and k=1..6 DZ fits after writing the tables')
    parser.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'],
                        help='Coordinate system the DZ fit reads (default: OCS, which is '
                             'what the v-mode recovery requires)')
    parser.add_argument('--no-consdb', action='store_true',
                        help='Skip the ConsDB merge; ra, dec, band, science_program and '
                             'rotator_angle are left unpopulated')
    parser.add_argument('--consdb-url', default=DEFAULT_CONSDB_URL,
                        help=f'ConsDB URL (default: {DEFAULT_CONSDB_URL}; token read '
                             'from ~/.lsst/consdb_token)')
    parser.add_argument('--min-donuts-per-visit', type=int,
                        default=DEFAULT_MIN_DONUTS_PER_VISIT,
                        help=f'Minimum donuts per visit, dimensionless (default '
                             f'{DEFAULT_MIN_DONUTS_PER_VISIT}; 0 disables)')
    parser.add_argument('--min-donuts-per-detector', type=int,
                        default=DEFAULT_MIN_DONUTS_PER_DETECTOR,
                        help=f'Per-detector donut floor used to count covered detectors, '
                             f'dimensionless (default {DEFAULT_MIN_DONUTS_PER_DETECTOR})')
    parser.add_argument('--min-detectors-per-visit', type=int,
                        default=DEFAULT_MIN_DETECTORS_PER_VISIT,
                        help=f'Minimum covered detectors per visit, dimensionless '
                             f'(default {DEFAULT_MIN_DETECTORS_PER_VISIT}; 0 disables)')
    parser.add_argument('--max-median-blur-arcsec', type=float,
                        default=DEFAULT_MAX_MEDIAN_BLUR_ARCSEC,
                        help=f'Maximum per-visit median donut blur in arcsec (default '
                             f'{DEFAULT_MAX_MEDIAN_BLUR_ARCSEC}; 0 disables)')
    parser.add_argument('--rotator-threshold-deg', type=float,
                        default=DEFAULT_ROTATOR_THRESHOLD_DEG,
                        help=f'Flag visits whose rotator angle exceeds this, in deg '
                             f'(default {DEFAULT_ROTATOR_THRESHOLD_DEG})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output parquet files')
    return parser.parse_args(argv)


def load_param_set(param_set):
    """Read one entry from ``aos/param_sets.yaml``.

    Parameters
    ----------
    param_set : `str`
        The entry key.

    Returns
    -------
    entry : `dict`
        The entry, which must carry ``butler_repo`` and ``fam_collections``.

    Raises
    ------
    KeyError
        If the entry is absent.
    """
    doc = yaml.safe_load((TOPIC / 'param_sets.yaml').read_text())
    doc = doc.get('param_sets', doc)
    if param_set not in doc:
        raise KeyError(f'unknown param_set {param_set!r}; available: '
                       f'{sorted(k for k in doc if isinstance(doc[k], dict))}')
    entry = dict(doc[param_set])
    for key in ('butler_repo', 'fam_collections'):
        if key not in entry:
            raise KeyError(f'param_set {param_set!r} has no {key!r}')
    return entry


def select_visits(butler, collections, dataset_type, visits=None, day_obs=None,
                  max_visits=None):
    """Resolve which visits to process.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler on the repository holding the collection.
    collections : `list` of `str`
        Butler collection(s) to query.
    dataset_type : `str`
        Blitz dataset type name.
    visits : `list` of `int`, optional
        Explicit visit identifiers. When given, the registry is still queried so that a
        visit absent from the collection is reported rather than failing later.
    day_obs : `list` of `int`, optional
        Restrict to these nights, as ``YYYYMMDD``.
    max_visits : `int`, optional
        Keep at most this many, dimensionless, taking the earliest.

    Returns
    -------
    selected : `list` of `int`
        Visit identifiers in ascending order.

    Notes
    -----
    ``day_obs`` is derived as ``visit // 100000``, which is the Rubin visit-identifier
    convention ``day_obs * 100000 + seq_num``.
    """
    available = sorted({int(ref.dataId['visit']) for ref in
                        butler.registry.queryDatasets(dataset_type,
                                                      collections=collections)})
    print(f'  {len(available)} visits of {dataset_type} in the collection, '
          f'{available[0]} to {available[-1]}')
    selected = available
    if visits is not None:
        wanted = [int(v) for v in visits]
        missing = sorted(set(wanted) - set(available))
        if missing:
            print(f'  WARNING: {len(missing)} requested visits are not in the '
                  f'collection and are skipped: {missing}')
        selected = [v for v in available if v in set(wanted)]
    if day_obs is not None:
        nights = {int(d) for d in day_obs}
        selected = [v for v in selected if v // 100000 in nights]
    if max_visits is not None:
        selected = selected[:int(max_visits)]
    return selected


def query_consdb(day_obs_values, consdb_url):
    """Per-visit fields the blitz metadata cannot supply, from ConsDB.

    Parameters
    ----------
    day_obs_values : `iterable` of `int`
        Nights to query, as ``YYYYMMDD``.
    consdb_url : `str`
        ConsDB base URL. A token from ``~/.lsst/consdb_token`` is embedded when the URL
        carries no credentials.

    Returns
    -------
    records : `dict`
        Maps visit identifier to a `dict` with ``ra`` and ``dec`` in **radians**,
        ``band``, ``science_program``, ``mjd`` in days (mid-exposure) and
        ``rotator_angle`` in deg.

    Notes
    -----
    ConsDB returns ``s_ra`` and ``s_dec`` in **deg**, while the Danish 1.2
    ``visits.parquet`` stores ``ra`` and ``dec`` in **radians** — verified on
    ``day_obs`` 20260315 ``seq_num`` 122, where the Danish 1.2 ``ra`` of 1.934339 rad is
    110.8294552 deg against a ConsDB ``s_ra`` of 110.8294610 deg. The conversion happens
    here.

    ``rotator_angle`` is ConsDB ``physical_rotator_angle``, not the blitz ``rot_tel_pos``
    and not ``boresightRotAngle``.
    """
    os.environ.setdefault('no_proxy', '')
    if '.consdb' not in os.environ['no_proxy']:
        os.environ['no_proxy'] += ',.consdb'
    from lsst.summit.utils import ConsDbClient

    url = consdb_url
    if '@' not in url and 'consdb-pq.consdb' not in url:
        token_file = Path.home() / '.lsst' / 'consdb_token'
        if token_file.exists():
            url = url.replace('://', f'://user:{token_file.read_text().strip()}@', 1)

    nights = sorted({int(d) for d in day_obs_values})
    day_list = ', '.join(str(d) for d in nights)
    query = f'''
        SELECT v1.visit_id, v1.s_ra, v1.s_dec, v1.band, v1.science_program,
               v1.exp_midpt_mjd, ql.physical_rotator_angle
        FROM cdb_lsstcam.visit1 v1
        LEFT JOIN cdb_lsstcam.visit1_quicklook ql ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs IN ({day_list})
    '''
    frame = ConsDbClient(url).query(query).to_pandas()
    print(f'  ConsDB returned {len(frame)} visit rows over {len(nights)} nights')

    records = {}
    for row in frame.itertuples(index=False):
        records[int(row.visit_id)] = {
            'ra': float(np.deg2rad(row.s_ra)) if row.s_ra is not None
            and np.isfinite(row.s_ra) else float('nan'),
            'dec': float(np.deg2rad(row.s_dec)) if row.s_dec is not None
            and np.isfinite(row.s_dec) else float('nan'),
            'band': '' if row.band is None else str(row.band),
            'science_program': '' if row.science_program is None
            else str(row.science_program),
            'mjd': float(row.exp_midpt_mjd) if row.exp_midpt_mjd is not None
            else float('nan'),
            'rotator_angle': (float(row.physical_rotator_angle)
                              if row.physical_rotator_angle is not None
                              and np.isfinite(row.physical_rotator_angle)
                              else float('nan')),
        }
    return records


def quality_mask(records, min_donuts_per_visit, min_detectors_per_visit,
                 max_median_blur_arcsec):
    """Standard per-visit quality cuts, as `run_mktable` applies them.

    Parameters
    ----------
    records : `list` of `dict`
        Per-visit records carrying ``n_donuts``, ``n_detectors_with_min_donuts`` and
        ``median_blur_arcsec``.
    min_donuts_per_visit : `int` or `None`
        Minimum donuts per visit, dimensionless. `None` disables the cut.
    min_detectors_per_visit : `int` or `None`
        Minimum covered detectors per visit, dimensionless. `None` disables the cut.
    max_median_blur_arcsec : `float` or `None`
        Maximum per-visit median donut blur, in arcsec. `None` disables the cut.

    Returns
    -------
    mask : `numpy.ndarray`
        Boolean, True where the visit passes every enabled cut.

    Notes
    -----
    Duplicates ``lsst.ts.intrinsic.wavefront.intrinsics_lib.quality_visit_mask`` rather
    than calling it, because that function takes an `astropy.table.QTable` and is applied
    here to plain records before the table is built. The three thresholds and their
    comparison directions are kept identical so that a blitz visit and a Danish 1.2 visit
    are cut the same way; the recorded ``visit_quality_pass`` column is what the DZ fitter
    reads.
    """
    n_donuts = np.array([r['n_donuts'] for r in records], dtype=float)
    n_detectors = np.array([r['n_detectors_with_min_donuts'] for r in records],
                           dtype=float)
    blur = np.array([r['median_blur_arcsec'] for r in records], dtype=float)
    mask = np.ones(len(records), dtype=bool)
    if min_donuts_per_visit:
        cut = n_donuts >= min_donuts_per_visit
        print(f'  n_donuts >= {min_donuts_per_visit} (dimensionless): drops '
              f'{int((~cut).sum())} visits')
        mask &= cut
    if min_detectors_per_visit:
        cut = n_detectors >= min_detectors_per_visit
        print(f'  n_detectors_with_min_donuts >= {min_detectors_per_visit} '
              f'(dimensionless): drops {int((~cut).sum())} visits')
        mask &= cut
    if max_median_blur_arcsec and max_median_blur_arcsec > 0:
        with np.errstate(invalid='ignore'):
            cut = blur <= max_median_blur_arcsec
        print(f'  median_blur_arcsec <= {max_median_blur_arcsec} arcsec: drops '
              f'{int((~cut).sum())} visits')
        mask &= cut
    print(f'  {int(mask.sum())} of {len(records)} visits pass every enabled cut')
    return mask


def build_tables(args):
    """Read the blitz collection and write ``donuts.parquet`` and ``visits.parquet``.

    Parameters
    ----------
    args : `argparse.Namespace`
        Parsed command line.

    Returns
    -------
    result : `dict`
        ``out_dir`` (`pathlib.Path`), ``donuts_file``, ``visits_file``,
        ``n_visits_written`` and ``n_donuts_written`` (dimensionless counts),
        ``row_group_report`` (the output of `blitz_reader.verify_row_groups`) and
        ``provenance`` (a `dict` written to ``provenance.yaml``).
    """
    from lsst.daf.butler import Butler

    entry = load_param_set(args.param_set)
    collections = list(entry['fam_collections'])
    out_dir = (Path(args.out_dir) if args.out_dir
               else TOPIC / study_dir('fam_processing', args.param_set))
    out_dir.mkdir(parents=True, exist_ok=True)
    donuts_file = out_dir / 'donuts.parquet'
    visits_file = out_dir / 'visits.parquet'

    existing = [p for p in (donuts_file, visits_file) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            'refusing to overwrite existing output; pass --overwrite:\n  '
            + '\n  '.join(str(p) for p in existing))

    print(f'Blitz mktable: param_set {args.param_set}, mode {args.mode}')
    print(f'  Butler repo: {entry["butler_repo"]}')
    print(f'  Collections: {collections}')
    print(f'  Output dir : {out_dir}')

    butler = Butler(entry['butler_repo'])
    visits = select_visits(butler, collections, args.dataset_type,
                           visits=args.visits, day_obs=args.day_obs,
                           max_visits=args.max_visits)
    if not visits:
        raise RuntimeError('no visits selected')
    print(f'  Processing {len(visits)} visits (dimensionless count)')

    records, provenance_seen = [], {}
    n_skipped, n_failed = 0, 0
    with br.BlitzDonutWriter(donuts_file) as writer:
        for n_done, visit in enumerate(visits, start=1):
            day_obs, seq_num = divmod(int(visit), 100000)
            try:
                table = br.read_blitz_visit(butler, visit, collections,
                                            dataset_type=args.dataset_type)
                meta = dict(table.meta)
                arrow_table, stats = br.donut_table(table, meta, day_obs, seq_num,
                                                    mode=args.mode)
            except Exception as exc:
                print(f'  [{n_done}/{len(visits)}] visit {visit}: FAILED — '
                      f'{type(exc).__name__}: {exc}')
                n_failed += 1
                continue
            if arrow_table.num_rows == 0:
                print(f'  [{n_done}/{len(visits)}] visit {visit}: no donut paired on '
                      'both sides of focus, skipped')
                n_skipped += 1
                continue
            if not writer.write_visit(arrow_table):
                n_skipped += 1
                continue

            record = br.visit_meta_record(meta, day_obs, seq_num, visit)
            record.update(br.visit_metrics(
                arrow_table, min_donuts_per_detector=args.min_donuts_per_detector))
            records.append(record)
            if not provenance_seen:
                provenance_seen = br.intrinsic_provenance(meta)
            if n_done % 25 == 0 or n_done == len(visits) or n_done <= 3:
                print(f'  [{n_done}/{len(visits)}] visit {visit}: '
                      f'{stats["n_donuts"]} paired donuts of {stats["n_intra_ok"]} '
                      f'intra and {stats["n_extra_ok"]} extra fitted rows '
                      f'(dimensionless counts), {record["n_detectors"]} detectors')

    print(f'\nWrote {writer.n_visits} visits, {writer.n_rows} donut rows '
          f'(dimensionless counts) to {donuts_file}')
    if n_skipped or n_failed:
        print(f'  skipped {n_skipped} visits, {n_failed} failed to read '
              '(dimensionless counts)')
    if writer.n_visits == 0:
        raise RuntimeError('no visit produced any donut row')

    # The contract the streaming DZ fitter depends on.  Checked before the fit, because a
    # violation makes the fit silently return an empty table.
    expected = {(int(r['day_obs']), int(r['seq_num'])) for r in records}
    report = br.verify_row_groups(donuts_file, expected_visits=expected)
    print(f'  row-group contract: {report["n_row_groups"]} row groups for '
          f'{len(expected)} visits (dimensionless counts), '
          f'{report["n_missing_statistics"]} lacking day_obs/seq_num statistics')

    # Fields the blitz metadata cannot supply.
    if args.no_consdb:
        print('\n--consdb skipped: ra, dec, band, science_program, rotator_angle '
              'left unpopulated')
        consdb = {}
    else:
        print('\nQuerying ConsDB for ra, dec, band, science_program, rotator_angle')
        consdb = query_consdb({r['day_obs'] for r in records}, args.consdb_url)
    n_matched = 0
    for record in records:
        extra = consdb.get(int(record['visit']))
        if extra is None:
            record.update(ra=float('nan'), dec=float('nan'), band='',
                          science_program='', rotator_angle=float('nan'))
            continue
        n_matched += 1
        record['ra'] = extra['ra']
        record['dec'] = extra['dec']
        record['band'] = extra['band']
        record['science_program'] = extra['science_program']
        record['rotator_angle'] = extra['rotator_angle']
        # ConsDB exp_midpt_mjd is the same quantity the Danish 1.2 visits table stores,
        # so prefer it over the blitz `date` for consistency across the two tables.
        if np.isfinite(extra['mjd']):
            record['mjd'] = extra['mjd']
    if not args.no_consdb:
        print(f'  matched {n_matched} of {len(records)} visits in ConsDB '
              '(dimensionless counts)')

    rotator = np.array([r['rotator_angle'] for r in records], dtype=float)
    with np.errstate(invalid='ignore'):
        flagged = np.abs(rotator) > args.rotator_threshold_deg
    for record, flag in zip(records, flagged):
        record['rotator_flagged'] = bool(flag)
    print(f'  rotator_flagged (|rotator_angle| > {args.rotator_threshold_deg} deg): '
          f'{int(flagged.sum())} of {len(records)} visits')

    print('\n--- Per-visit quality cuts ---')
    passes = quality_mask(records,
                          args.min_donuts_per_visit or None,
                          args.min_detectors_per_visit or None,
                          args.max_median_blur_arcsec)
    for record, flag in zip(records, passes):
        record['visit_quality_pass'] = bool(flag)

    visit_table = QTable()
    for column in VISITS_COLUMNS:
        visit_table[column] = [r[column] for r in records]
    visit_table.meta['min_donuts_per_visit'] = int(args.min_donuts_per_visit)
    visit_table.meta['min_donuts_per_detector'] = int(args.min_donuts_per_detector)
    visit_table.meta['min_detectors_per_visit'] = int(args.min_detectors_per_visit)
    visit_table.meta['max_median_blur_arcsec'] = float(args.max_median_blur_arcsec)
    visit_table.write(str(visits_file), format='parquet', overwrite=True)
    print(f'\nWrote {len(visit_table)} visit rows, {len(visit_table.colnames)} columns '
          f'to {visits_file}')

    provenance = {
        'param_set': args.param_set,
        'butler_repo': entry['butler_repo'],
        'fam_collections': collections,
        'dataset_type': args.dataset_type,
        'pairing_mode': args.mode,
        'coord_sys': args.coord_sys,
        'intrinsic': 'default Batoid intrinsic carried in the blitz dataset type '
                     '(zk_intrinsic column); no measured-intrinsic sidecar',
        'input_run_collections': provenance_seen,
        'n_visits': int(writer.n_visits),
        'n_donuts': int(writer.n_rows),
        'ts_wep_version': records[0]['ts_wep_version'],
        'danish_version': records[0]['danish_version'],
        'batoid_version': records[0]['batoid_version'],
    }
    intrinsic_runs = list(provenance_seen.get('intrinsicZernikes', []))
    provenance['intrinsic_zernikes_run'] = intrinsic_runs
    # default_flow_style=False alone still emits a YAML anchor when two keys share one
    # list object; copy the list above and disable aliases so the file is plain YAML.
    dumper = yaml.SafeDumper
    dumper.ignore_aliases = lambda *_args: True
    (out_dir / 'provenance.yaml').write_text(
        yaml.dump(provenance, Dumper=dumper, default_flow_style=False,
                  sort_keys=False))
    print(f'  intrinsicZernikes calibration run(s): {intrinsic_runs}')
    print(f'  provenance: {out_dir / "provenance.yaml"}')

    return {
        'out_dir': out_dir,
        'donuts_file': donuts_file,
        'visits_file': visits_file,
        'n_visits_written': int(writer.n_visits),
        'n_donuts_written': int(writer.n_rows),
        'row_group_report': report,
        'provenance': provenance,
    }


def run_fits(donuts_file, visits_file, coord_sys='OCS'):
    """Run the k=1..3 and k=1..6 Double Zernike fits on the written tables.

    Parameters
    ----------
    donuts_file : `pathlib.Path`
        The donut parquet, one row group per visit.
    visits_file : `pathlib.Path`
        The visits sidecar.
    coord_sys : `str`, optional
        ``'OCS'`` or ``'CCS'``. Default ``'OCS'``.

    Returns
    -------
    fits_file : `pathlib.Path`
        The written ``fits.parquet``.

    Notes
    -----
    ``intrinsic_sidecar`` is deliberately left at `None`, so the fit uses the default
    Batoid intrinsic wavefront carried in the donut table's ``zk_intrinsic_<coord_sys>``
    column rather than a Measured Intrinsic Wavefront build.

    The fitter reads the donut table's field angles and applies `numpy.rad2deg`, so the
    stored ``thx_<coord_sys>`` / ``thy_<coord_sys>`` must be in radians, which is what
    `blitz_reader.donut_table` writes.
    """
    from lsst.ts.intrinsic.wavefront.dz_fitting import run_double_zernike_fits

    fits_file = Path(donuts_file).parent / 'fits.parquet'
    print(f'\n=== Double Zernike fits, {coord_sys}, default Batoid intrinsic ===')
    run_double_zernike_fits(str(donuts_file), coord_sys=coord_sys,
                            output_file=str(fits_file),
                            visits_file=str(visits_file),
                            intrinsic_sidecar=None)
    table = pq.read_table(str(fits_file))
    print(f'Wrote {table.num_rows} fit rows, {table.num_columns} columns to {fits_file}')
    return fits_file


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    result = build_tables(args)
    if args.fit:
        run_fits(result['donuts_file'], result['visits_file'],
                 coord_sys=args.coord_sys)
    return 0


if __name__ == '__main__':
    sys.exit(main())
