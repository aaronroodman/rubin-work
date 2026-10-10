#!/usr/bin/env python3
"""run_attach_telemetry — attach ALL per-visit telemetry in one pass.

Replaces the split between ``run_backfill_thermal.py`` (rewrote each per-chunk
``visits.parquet`` in place) and ``run_backfill_camera_telemetry.py`` (wrote per-chunk
sidecars and merged only into the *combined* table). That split is why re-running
``combine_visits`` silently drops the 28 ``cam_*`` columns: the combined file was the only
place they lived. Here every quantity lands in one per-chunk ``telemetry.parquet``
sidecar, which is the single source of truth, and a merge step joins it into both the
per-chunk and the combined ``visits.parquet``. A re-combine can then always be repaired by
re-running ``--merge`` alone, and a failed fetch never damages an expensive ``mktable``
output.

Sources, ConsDB first and the EFD only where ConsDB cannot answer -- see
``docs/telemetry.md`` for the measurements behind each choice:

===================== ========= ==========================================================
group                 source    why
===================== ========= ==========================================================
``thermal``           ConsDB    ESS temps, deltas, TMA truss: 88.6% on FAM exposures
``gradients``         EFD       M1M3 spatial gradients are not in the transform
``wind``              ConsDB    88.6% and currently thrown away entirely
``camera``            EFD       camera-body temperatures are not in ConsDB at all
``lut``               ConsDB    M1M3 elevation + M2 gravity LUT join 400/400 sampled
``hexlut``            EFD       hexapod LUT: ConsDB covers 7.7% (cam) / 0.0% (M2) of cwfs
``trim``              EFD       0% on FAM (cwfs) exposures in ConsDB -- no ConsDB path
``tweak``             derived   no topic and no property exists; differenced from Trim
===================== ========= ==========================================================

Three distinct DOF quantities are attached, and they are **not** interchangeable:

===================== ==================== ===================================================
quantity              columns              meaning
===================== ==================== ===================================================
LUT (hexapod)         ``lut_dof0..9``      baseline from the hexapod elevation/rotator/filter
                                           lookup, ``MTHexapod.logevent_compensationOffset``
LUT (mirror)          ``m1m3elev_*``,      axial forces from the M1M3 elevation and M2 gravity
                      ``m2grav_*``         LUTs, in N (never changed from mirror-lab values)
Trim                  ``dof0..49``         accumulated AOS offset *from* the LUT,
                                           ``MTAOS.logevent_degreeOfFreedom.aggregatedDoF``
Tweak                 ``tweak_dof0..49``   the per-iteration correction, ``Trim_i - Trim_(i-1)``
===================== ==================== ===================================================

A physical hexapod position is LUT + Trim; neither alone is the position.

Telemetry that predates its deployment is filled with NaN, which is expected rather than a
failure: much of this was added gradually through commissioning in 2025.

**Trim needs the as-of-time EFD lookup, not a ConsDB join.** ConsDB's
``mt_logevent_aggregated_dof`` is populated on ``img_type='science'`` exposures and
essentially never on the ``cwfs`` exposures FAM uses (6 ever, repo-wide, against 46562
science). The MTAOS events do fire during FAM -- 52 in one FAM hour versus 31 in the
following science hour -- so the data exists; it just is not attached to these exposures.

**Tweak** is derived, not fetched: ``Tweak_i = Trim_i - Trim_(i-1)``. Where consecutive
visits share one ``degreeOfFreedom`` event the AOS applied no new correction, and Tweak is
**0.0** -- a real measurement of "no correction", not missing data. NaN is reserved for
genuinely unknown values: the first visit of a chunk, or a visit whose Trim or source event
id could not be resolved. The ``event_id`` array returned by
``common.dof_telemetry.fetch_aggregated_dof_for_visits`` is what distinguishes the two
cases.

Needs a node where both the EFD and ConsDB resolve -- the RSP terminal or a
slaciana/slacrd interactive node, NOT a batch compute node.

Usage:
  # one chunk, every group
  python -m rubinwork.products.fam_tables.builders.run_attach_telemetry --param-set <ps> --chunk 20260713_20260713
  # all chunks of a param_set, then merge into per-chunk + combined visits
  python -m rubinwork.products.fam_tables.builders.run_attach_telemetry --param-set <ps> --all-chunks --merge
  # re-merge only, from existing sidecars (the fix after a combine_visits re-run)
  python -m rubinwork.products.fam_tables.builders.run_attach_telemetry --param-set <ps> --merge --skip-fetch
  # pick groups
  python -m rubinwork.products.fam_tables.builders.run_attach_telemetry --param-set <ps> --chunk <c> --groups trim,lut
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rubinwork.common.dof_telemetry import (
    derive_tweak, fetch_aggregated_dof_for_visits, fetch_hexapod_lut_for_visits,
    fetch_lut_forces)
from rubinwork.common.telemetry_clients import make_consdb_client, make_efd_client
from rubinwork.common.visit_telemetry import WIND_COLS, fetch_camera, fetch_wind

GROUPS = ('thermal', 'gradients', 'wind', 'camera', 'lut', 'hexlut', 'trim', 'tweak')

# Hexapod LUT axis order, matching the first 10 entries of the 50-DOF OFC state
# (m2HexPos 0-4, camHexPos 5-9). z/x/y in µm, u/v in deg -- note the OFC state uses
# arcsec for the angular axes, so lut_dof3/4/8/9 are NOT directly comparable to dof3/4/8/9
# without a unit conversion; the dz axes lut_dof0 and lut_dof5 are.
HEXLUT_COLS = [f'lut_dof{k}' for k in range(10)]

# ESS / truss / gradient columns kept from the package's get_thermal_data. The ~180
# per-thermocouple m1m3_tc_* and per-cell m1m3_dt_* columns are dropped: they bloat the
# table without adding analysis value.
CORE_THERMAL = [
    'cam_air_temp', 'm2_air_temp', 'm1m3_air_temp', 'outside_temp',
    'm2_delta_t', 'cam_m1m3_delta_t', 'dome_delta_t',
    'tma_truss_temp_pxpy', 'tma_truss_temp_mxmy',
]
GRADIENT_COLS = ['x_gradient', 'y_gradient', 'z_gradient', 'radial_gradient']


def _visit_keys(visits_path):
    """Return the (day_obs, seq_num, visit, mjd) frame for a visits.parquet."""
    have = set(pq.ParquetFile(str(visits_path)).schema_arrow.names)
    cols = [c for c in ('day_obs', 'seq_num', 'visit', 'mjd') if c in have]
    return pq.read_table(str(visits_path), columns=cols).to_pandas()


def _close_efd(efd):
    """Close an EFD client's aiohttp session, ignoring any failure to do so.

    Notes
    -----
    Reaching for ``efd.influx_client`` emits a deprecation warning on this stack, so the
    session is found by scanning ``__dict__`` instead of by attribute name. Without this,
    asyncio prints "Unclosed client session" at interpreter shutdown.
    """
    import asyncio
    seen = []
    for holder in list(vars(efd).values()):
        sess = getattr(holder, '_session', None)
        if sess is not None and not getattr(sess, 'closed', True):
            seen.append(sess)
    for sess in seen:
        try:
            asyncio.run(sess.close())
        except Exception:
            pass


def fetch_for_chunk(chunk_dir, groups, cdb, efd, consdb_url, verbose=True):
    """Fetch the requested groups for one chunk and write its telemetry.parquet sidecar.

    Returns
    -------
    n_rows : `int`
        Rows written.
    written : `list` [`str`]
        Column names written, excluding the join keys.
    """
    vp = Path(chunk_dir) / 'visits.parquet'
    if not vp.exists():
        print(f'  {chunk_dir.name}: no visits.parquet, skipping')
        return 0, []
    keys = _visit_keys(vp)
    n = len(keys)
    out = keys.copy()
    vids = [int(v) for v in keys['visit']] if 'visit' in keys else []
    if verbose:
        print(f'  {Path(chunk_dir).name}: {n} visits')

    if 'wind' in groups and vids:
        w = fetch_wind(cdb, vids)
        out = out.merge(w, on='visit', how='left')
        got = [c for c in WIND_COLS.values() if c in out]
        if verbose and got:
            f = float(np.isfinite(out[got[0]].to_numpy(dtype=float,
                                                       na_value=np.nan)).mean())
            print(f'    wind: {len(got)} cols, {100 * f:.1f}% finite')

    if 'lut' in groups and vids:
        lut = fetch_lut_forces(cdb, vids)
        if len(lut.columns) > 1:
            out = out.merge(lut, on='visit', how='left')
            if verbose:
                print(f'    lut: {len(lut.columns) - 1} axial-force cols')

    if 'hexlut' in groups:
        from astropy.table import QTable
        try:
            # The hexapod LUT (MTHexapod.logevent_compensationOffset) is the baseline the
            # Trim is an offset from, so a physical hexapod position needs both. ConsDB's
            # *_compensation_offset_z covers only 7.7% (camera) and 0.0% (M2) of cwfs
            # exposures, so this is EFD-only -- see docs/telemetry.md.
            hlut, hinfo = fetch_hexapod_lut_for_visits(
                QTable.from_pandas(keys), efd_client=efd, consdb_client=cdb,
                consdb_url=consdb_url)
            out = pd.concat([out, pd.DataFrame(hlut, columns=HEXLUT_COLS,
                                               index=out.index)], axis=1)
            if verbose:
                print(f"    hexlut: {hinfo.get('n_lut')}/{n} visits with all 10 axes")
        except Exception as e:
            print(f'    hexlut FAILED: {type(e).__name__}: {str(e)[:200]}')

    if 'trim' in groups or 'tweak' in groups:
        from astropy.table import QTable
        try:
            # The fetcher indexes fit_table[...] and tests `in fit_table.colnames`, so it
            # wants an astropy table rather than a DataFrame.
            trim, info = fetch_aggregated_dof_for_visits(
                QTable.from_pandas(keys), efd_client=efd, consdb_client=cdb,
                consdb_url=consdb_url)
            # One concat rather than 50 inserts: a per-column assign fragments the
            # frame and pandas warns about it.
            out = pd.concat([out, pd.DataFrame(
                trim, columns=[f'dof{k}' for k in range(trim.shape[1])],
                index=out.index)], axis=1)
            if verbose:
                nfin = int(np.isfinite(trim[:, 0]).sum())
                print(f"    trim: {nfin}/{n} visits anchored "
                      f"(obs_start={info.get('n_obs_start')}, "
                      f"mjd_fallback={info.get('n_mjd_fallback')})")
            if 'tweak' in groups:
                # 'event_id' (singular) is the key fetch_aggregated_dof_for_visits sets;
                # it carries the per-visit source degreeOfFreedom event visitId.
                ev = np.asarray(info.get('event_id', np.full(n, np.nan)), dtype=float)
                out['dof_event_id'] = ev
                tw = derive_tweak(trim, ev)
                out = pd.concat([out, pd.DataFrame(
                    tw, columns=[f'tweak_dof{k}' for k in range(tw.shape[1])],
                    index=out.index)], axis=1)
                if verbose:
                    fin = np.isfinite(tw[:, 0])
                    nz = int((fin & (tw[:, 0] != 0.0)).sum())
                    print(f'    tweak: derived, {int(fin.sum())}/{n} visits known '
                          f'({nz} with a non-zero correction, '
                          f'{int(fin.sum()) - nz} with none applied)')
        except Exception as e:
            print(f'    trim/tweak FAILED: {type(e).__name__}: {str(e)[:140]}')

    if 'camera' in groups and 'mjd' in keys:
        try:
            cam = fetch_camera(keys)
            out = out.merge(cam, on=['day_obs', 'seq_num'], how='left')
            got = [c for c in out.columns if c.startswith('cam_') and c != 'cam_n_samp']
            if verbose and got:
                f = float(np.isfinite(out['cam_AverageTemp'].to_numpy(
                    dtype=float, na_value=np.nan)).mean())
                print(f'    camera: {len(got)} cols, cam_AverageTemp {100 * f:.1f}% finite')
        except Exception as e:
            print(f'    camera FAILED: {type(e).__name__}: {str(e)[:140]}')

    if 'thermal' in groups or 'gradients' in groups:
        try:
            _attach_thermal(out, keys, cdb, efd, groups, verbose)
        except Exception as e:
            print(f'    thermal FAILED: {type(e).__name__}: {str(e)[:140]}')

    # Preserve columns an earlier run fetched. `out` holds only the groups requested this
    # time, so writing it directly would drop every other group's columns from the
    # sidecar -- and the sidecar is the source of truth the merge step reads. Columns
    # fetched now win; columns only in the old sidecar are carried forward.
    sidecar = Path(chunk_dir) / 'telemetry.parquet'
    if sidecar.exists():
        old = pq.read_table(str(sidecar)).to_pandas()
        on = [c for c in ('day_obs', 'seq_num') if c in old.columns and c in out.columns]
        if not on:
            on = [c for c in ('visit',) if c in old.columns and c in out.columns]
        if on:
            carry = [c for c in old.columns if c not in out.columns]
            if carry:
                out = out.merge(old[on + carry], on=on, how='left')
                if verbose:
                    print(f'    carried forward {len(carry)} existing sidecar cols')
        else:
            print('    WARNING: cannot key the existing sidecar; not carrying it forward')
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), str(sidecar))
    cols = [c for c in out.columns if c not in ('day_obs', 'seq_num', 'visit', 'mjd')]
    if verbose:
        print(f'    wrote {sidecar.name}: {len(out)} rows, {len(cols)} telemetry cols')
    return len(out), cols


def _attach_thermal(out, keys, cdb, efd, groups, verbose):
    """Merge the package's thermal product into `out`, in place.

    Uses ``intrinsics_lib.get_thermal_data`` rather than reimplementing the query, so the
    columns match what ``mktable`` itself would have written.
    """
    import asyncio
    from astropy.table import QTable
    from lsst.ts.intrinsic.wavefront.intrinsics_lib import get_thermal_data
    vi = QTable.from_pandas(keys)
    df = asyncio.run(get_thermal_data(cdb, efd, vi))
    if df is None or not len(df):
        print('    thermal: no rows returned')
        return
    want = []
    if 'thermal' in groups:
        want += CORE_THERMAL
    if 'gradients' in groups:
        want += GRADIENT_COLS
    keep = ['day_obs', 'seq_num'] + [c for c in want if c in df.columns]
    merged = out.merge(df[keep], on=['day_obs', 'seq_num'], how='left')
    for c in keep[2:]:
        out[c] = merged[c].to_numpy()
    if verbose:
        got = [c for c in want if c in out]
        print(f'    thermal/gradients: {len(got)} cols')


def merge_sidecars(base, verbose=True, chunks_dir=None):
    """Join every chunk's telemetry.parquet into the per-chunk and combined visits tables.

    The sidecars are the source of truth, so this is idempotent and is the repair step
    after a ``combine_visits`` re-run drops the telemetry columns.

    Parameters
    ----------
    base : `pathlib.Path`
        Directory holding the combined ``visits.parquet``.
    verbose : `bool`, optional
        Print one line per chunk merged.
    chunks_dir : `pathlib.Path`, optional
        Directory holding the per-chunk subdirectories. Defaults to
        ``base / 'chunks'``.
    """
    chunks_dir = Path(chunks_dir) if chunks_dir else base / 'chunks'
    chunks = sorted(p for p in chunks_dir.iterdir() if p.is_dir())
    frames = []
    for ch in chunks:
        sc = ch / 'telemetry.parquet'
        vp = ch / 'visits.parquet'
        if not sc.exists() or not vp.exists():
            continue
        tel = pq.read_table(str(sc)).to_pandas()
        frames.append(tel)
        # per-chunk visits.parquet
        vis = pq.read_table(str(vp)).to_pandas()
        newc = [c for c in tel.columns if c not in ('day_obs', 'seq_num', 'visit', 'mjd')]
        vis = vis.drop(columns=[c for c in newc if c in vis.columns], errors='ignore')
        on = ['day_obs', 'seq_num'] if {'day_obs', 'seq_num'} <= set(vis.columns) else ['visit']
        vis = vis.merge(tel[on + newc], on=on, how='left')
        pq.write_table(pa.Table.from_pandas(vis, preserve_index=False), str(vp))
        if verbose:
            print(f'  merged {len(newc)} cols into {ch.name}/visits.parquet')
    if not frames:
        print('  no sidecars found -- nothing to merge')
        return
    allt = pd.concat(frames, ignore_index=True)
    cvp = base / 'visits.parquet'
    if cvp.exists():
        vis = pq.read_table(str(cvp)).to_pandas()
        newc = [c for c in allt.columns if c not in ('day_obs', 'seq_num', 'visit', 'mjd')]
        vis = vis.drop(columns=[c for c in newc if c in vis.columns], errors='ignore')
        on = ['day_obs', 'seq_num'] if {'day_obs', 'seq_num'} <= set(vis.columns) else ['visit']
        vis = vis.merge(allt[on + newc], on=on, how='left')
        pq.write_table(pa.Table.from_pandas(vis, preserve_index=False), str(cvp))
        print(f'  merged {len(newc)} cols into the combined visits.parquet '
              f'({len(vis)} rows)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--chunk', default=None, help='one chunk directory name')
    ap.add_argument('--all-chunks', action='store_true')
    ap.add_argument('--groups', default='all',
                    help=f'comma-separated subset of {",".join(GROUPS)}, or "all"')
    ap.add_argument('--merge', action='store_true',
                    help='join the sidecars into per-chunk and combined visits.parquet')
    ap.add_argument('--skip-fetch', action='store_true',
                    help='merge only, from existing sidecars')
    ap.add_argument('--output-root', default='output')
    # The caller (the Snakefile) owns the output layout and may name directories
    # differently from the param_set key, so the phase-1 table directory can be
    # supplied directly rather than derived from it.
    ap.add_argument('--out-dir', default=None,
                    help='dir holding the phase-1 chunks/ and visits.parquet '
                         '(read and updated in place; default: output/<ps>)')
    ap.add_argument('--chunks-dir', default=None,
                    help="dir holding the per-chunk subdirectories "
                         "(default: <out-dir>/chunks). The product build keeps "
                         "them under the build's _work/, away from the tables.")
    ap.add_argument('--consdb-url', default='auto')
    ap.add_argument('--efd', default='usdf_efd')
    args = ap.parse_args()

    groups = GROUPS if args.groups == 'all' else tuple(
        g.strip() for g in args.groups.split(','))
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f'unknown group(s) {bad}; choose from {GROUPS}')

    base = (Path(args.out_dir) if args.out_dir
            else Path(args.output_root) / args.param_set)
    if not base.is_dir():
        ap.error(f'no such phase-1 table dir: {base}')
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else base / 'chunks'

    if not args.skip_fetch:
        if args.all_chunks:
            chunks = sorted(p for p in chunks_dir.iterdir() if p.is_dir())
        elif args.chunk:
            chunks = [chunks_dir / args.chunk]
        else:
            ap.error('give --chunk, --all-chunks, or --skip-fetch')
        print(f'[attach_telemetry] {args.param_set}: {len(chunks)} chunk(s), '
              f'groups={",".join(groups)}')
        cdb = make_consdb_client(args.consdb_url)
        efd = make_efd_client(args.efd) if (
            {'trim', 'tweak', 'camera', 'gradients', 'thermal'} & set(groups)) else None
        for ch in chunks:
            fetch_for_chunk(ch, groups, cdb, efd, args.consdb_url)

        # The EFD client holds an aiohttp session; closing it avoids the
        # "Unclosed client session" error asyncio prints at interpreter shutdown.
        if efd is not None:
            _close_efd(efd)

    if args.merge:
        print('[attach_telemetry] merging sidecars')
        merge_sidecars(base, chunks_dir=chunks_dir)


if __name__ == '__main__':
    main()
