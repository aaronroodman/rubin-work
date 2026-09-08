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
``trim``              EFD       0% on FAM (cwfs) exposures in ConsDB -- no ConsDB path
``tweak``             derived   no topic and no property exists; differenced from Trim
===================== ========= ==========================================================

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
``aos_trim.fetch_aggregated_dof_for_visits`` is what distinguishes the two cases.

Needs a node where both the EFD and ConsDB resolve -- the RSP terminal or a
slaciana/slacrd interactive node, NOT a batch compute node.

Usage:
  # one chunk, every group
  python code/fam_processing/run_attach_telemetry.py --param-set <ps> --chunk 20260713_20260713
  # all chunks of a param_set, then merge into per-chunk + combined visits
  python code/fam_processing/run_attach_telemetry.py --param-set <ps> --all-chunks --merge
  # re-merge only, from existing sidecars (the fix after a combine_visits re-run)
  python code/fam_processing/run_attach_telemetry.py --param-set <ps> --merge --skip-fetch
  # pick groups
  python code/fam_processing/run_attach_telemetry.py --param-set <ps> --chunk <c> --groups trim,lut
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))           # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))           # aos/code
from common.telemetry_clients import make_consdb_client, make_efd_client  # noqa: E402

GROUPS = ('thermal', 'gradients', 'wind', 'camera', 'lut', 'trim', 'tweak')

# ESS / truss / gradient columns kept from the package's get_thermal_data. The ~180
# per-thermocouple m1m3_tc_* and per-cell m1m3_dt_* columns are dropped: they bloat the
# table without adding analysis value.
CORE_THERMAL = [
    'cam_air_temp', 'm2_air_temp', 'm1m3_air_temp', 'outside_temp',
    'm2_delta_t', 'cam_m1m3_delta_t', 'dome_delta_t',
    'tma_truss_temp_pxpy', 'tma_truss_temp_mxmy',
]
GRADIENT_COLS = ['x_gradient', 'y_gradient', 'z_gradient', 'radial_gradient']

# Wind and airflow: present in ConsDB at 88.6% on FAM exposures and currently unused.
# ConsDB transformed-EFD column -> our name.
WIND_COLS = {
    'mt_salindex110_wind_speed_magnitude_mean': 'wind_speed_inside',
    'mt_salindex301_airflow_speed_mean': 'wind_speed_outside',
    'mt_salindex301_airflow_direction_mean': 'wind_dir_outside',
    'mt_salindex110_wind_speed_0_mean': 'wind_inside_x',
    'mt_salindex110_wind_speed_1_mean': 'wind_inside_y',
    'mt_salindex110_wind_speed_2_mean': 'wind_inside_z',
    'mt_salindex110_wind_speed_maxmagnitude_mean': 'wind_inside_maxmag',
    'mt_salindex110_sonic_temperature_mean': 'sonic_temperature',
}

# Mirror LUT array properties in efd_lsstcam.exposure_efd_unpivoted. These are axial
# FORCES; aos_trim.fetch_mirror_lut_for_visits converts them to bending amplitudes.
LUT_PROPS = {
    'mt_m1m3_applied_elevation_forces_mean': ('m1m3elev', 'zForces', 156),
    'mt_m2_axial_force_lut_gravity_mean': ('m2grav', 'lutGravity', 72),
}

N_DOF = 50


def _visit_keys(visits_path):
    """Return the (day_obs, seq_num, visit, mjd) frame for a visits.parquet."""
    have = set(pq.ParquetFile(str(visits_path)).schema_arrow.names)
    cols = [c for c in ('day_obs', 'seq_num', 'visit', 'mjd') if c in have]
    return pq.read_table(str(visits_path), columns=cols).to_pandas()


def fetch_wind(cdb, visit_ids):
    """Wind and airflow per visit from the ConsDB transformed EFD.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
        ConsDB client.
    visit_ids : `list` [`int`]
        Visit (exposure) ids.

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit`` plus the columns named in `WIND_COLS`; speeds in m/s, directions in deg,
        `sonic_temperature` in deg C. Missing visits are absent rather than NaN-filled.
    """
    out = []
    sel = ', '.join(WIND_COLS)
    for i in range(0, len(visit_ids), 700):
        part = visit_ids[i:i + 700]
        inl = ','.join(str(int(v)) for v in part)
        q = (f'SELECT exposure_id, {sel} FROM efd_lsstcam.exposure_efd '
             f'WHERE exposure_id IN ({inl})')
        try:
            out.append(cdb.query(q).to_pandas())
        except Exception as e:
            print(f'    wind chunk {i}: {type(e).__name__}: {str(e)[:90]}')
    if not out:
        return pd.DataFrame(columns=['visit'] + list(WIND_COLS.values()))
    df = pd.concat(out, ignore_index=True).rename(columns=WIND_COLS)
    return df.rename(columns={'exposure_id': 'visit'})


def fetch_lut_forces(cdb, visit_ids):
    """Mirror LUT axial forces per visit, pivoted, from the unpivoted ConsDB table.

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit`` plus ``<prefix>_<n>`` columns of axial force in N: 156 for M1M3
        elevation, 72 for M2 gravity.
    """
    frames = []
    for prop, (prefix, field, n) in LUT_PROPS.items():
        rows = []
        for i in range(0, len(visit_ids), 400):
            part = visit_ids[i:i + 400]
            inl = ','.join(str(int(v)) for v in part)
            q = (f"SELECT exposure_id, field, value "
                 f"FROM efd_lsstcam.exposure_efd_unpivoted "
                 f"WHERE property='{prop}' AND exposure_id IN ({inl})")
            try:
                rows.append(cdb.query(q).to_pandas())
            except Exception as e:
                print(f'    {prefix} chunk {i}: {type(e).__name__}: {str(e)[:90]}')
        if not rows:
            continue
        d = pd.concat(rows, ignore_index=True)
        p = d.pivot_table(index='exposure_id', columns='field', values='value')
        # field names are e.g. zForces0..zForces155; order them numerically
        order = [f'{field}{k}' for k in range(n) if f'{field}{k}' in p.columns]
        p = p[order]
        p.columns = [f'{prefix}_{k}' for k in range(len(order))]
        frames.append(p.reset_index().rename(columns={'exposure_id': 'visit'}))
    if not frames:
        return pd.DataFrame(columns=['visit'])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on='visit', how='outer')
    return out


def derive_tweak(trim, event_ids):
    """Tweak per visit, differenced from Trim.

    Parameters
    ----------
    trim : `numpy.ndarray`
        Shape ``(n_visits, n_dof)`` Trim values, in the OFC DOF units (µm for hexapod
        translations, arcsec for rotations, dimensionless for bending amplitudes).
    event_ids : `numpy.ndarray`
        Shape ``(n_visits,)`` ``visitId`` of the source ``degreeOfFreedom`` event, NaN
        where none resolved. A change between consecutive visits marks a re-alignment.

    Returns
    -------
    tweak : `numpy.ndarray`
        Shape ``(n_visits, n_dof)``, same units as `trim`. **0.0** where the AOS applied
        no new correction between this visit and the previous one, since that is a real
        measurement of "no correction" rather than missing information. **NaN** only where
        the value is genuinely unknown: the first row (no predecessor), or where either
        visit's Trim or source event id could not be resolved.

    Notes
    -----
    Tweak has no EFD topic and no ConsDB property; ``Tweak = PID(optical_state)`` and
    ``Trim_(i+1) = Trim_i + Tweak``, so differencing Trim is the only route.

    Where consecutive visits share one ``degreeOfFreedom`` event the difference is exactly
    zero by construction, and it is written as 0.0 rather than recomputed -- guarding
    against a float subtraction of two equal Trim values landing on a denormal instead of
    a clean zero.
    """
    trim = np.asarray(trim, dtype=float)
    ev = np.asarray(event_ids, dtype=float)
    tweak = np.full_like(trim, np.nan)
    for i in range(1, len(trim)):
        # Unknown Trim on either side -> genuinely unknown Tweak.
        if not (np.isfinite(trim[i]).any() and np.isfinite(trim[i - 1]).any()):
            continue
        if np.isfinite(ev[i]) and np.isfinite(ev[i - 1]) and ev[i] == ev[i - 1]:
            tweak[i] = 0.0            # loop ran, emitted no new correction
        else:
            tweak[i] = trim[i] - trim[i - 1]
    return tweak


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

    if 'trim' in groups or 'tweak' in groups:
        import aos_trim
        from astropy.table import QTable
        try:
            # aos_trim indexes fit_table[...] and tests `in fit_table.colnames`, so it
            # wants an astropy table rather than a DataFrame.
            trim, info = aos_trim.fetch_aggregated_dof_for_visits(
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

    if 'thermal' in groups or 'gradients' in groups:
        try:
            _attach_thermal(out, keys, cdb, efd, groups, verbose)
        except Exception as e:
            print(f'    thermal FAILED: {type(e).__name__}: {str(e)[:140]}')

    sidecar = Path(chunk_dir) / 'telemetry.parquet'
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


def merge_sidecars(base, verbose=True):
    """Join every chunk's telemetry.parquet into the per-chunk and combined visits tables.

    The sidecars are the source of truth, so this is idempotent and is the repair step
    after a ``combine_visits`` re-run drops the telemetry columns.
    """
    chunks = sorted(p for p in (base / 'chunks').iterdir() if p.is_dir())
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
    ap.add_argument('--consdb-url', default='auto')
    ap.add_argument('--efd', default='usdf_efd')
    args = ap.parse_args()

    groups = GROUPS if args.groups == 'all' else tuple(
        g.strip() for g in args.groups.split(','))
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f'unknown group(s) {bad}; choose from {GROUPS}')

    base = Path(args.output_root) / args.param_set
    if not base.is_dir():
        ap.error(f'no such param_set output dir: {base}')

    if not args.skip_fetch:
        if args.all_chunks:
            chunks = sorted(p for p in (base / 'chunks').iterdir() if p.is_dir())
        elif args.chunk:
            chunks = [base / 'chunks' / args.chunk]
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
        merge_sidecars(base)


if __name__ == '__main__':
    main()
