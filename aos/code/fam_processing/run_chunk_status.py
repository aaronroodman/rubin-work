#!/usr/bin/env python3
"""run_chunk_status — all-chunks status review for one param_set, in one PDF.

The per-chunk ``*_visit_quality.pdf`` files show one chunk at a time. This gives the view
across every chunk of a ``param_set``, which is what is needed to answer "is the collected
sample complete and self-consistent?".

Five pages:

  1  **Chunk inventory** — visits, donuts and fits per chunk, the ``day_obs`` span of each,
     and whether the per-chunk tables sum to the combined ones. A mismatch means a chunk
     was rebuilt without re-running the combine.
  2  **Coverage** — visits per ``day_obs``, and the band and science-program mix per chunk.
  3  **Elevation x rotator coverage** — a text histogram over all chunks, in deg.
  4  **Telemetry completeness** — finite fraction per telemetry column per chunk. This is
     how a chunk built with ``mktable --no-thermal`` is spotted: its thermal columns are
     either absent or all-NaN.
  5  **Degree-of-freedom (DOF) presence** — whether commanded-DOF columns (Trim, Tweak,
     hexapod and mirror LUT) exist at all, and which fetcher would supply each. They are
     currently absent from every table; see
     ``docs/status/dof_telemetry_availability.md``.

Reads only the parquet tables, so it needs no Butler, EFD or ConsDB access and runs
anywhere the output tree exists.

Writes ``output/<param_set>/fam_processing/chunk_status.{pdf,parquet}`` — under
``<param_set>/`` because nothing here depends on which Measured Intrinsic Wavefront build
was used.

Usage:
  python code/fam_processing/run_chunk_status.py --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x
  python code/fam_processing/run_chunk_status.py --param-set all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
from common.utils import alt_to_deg, centered_edges, text_hist2d  # noqa: E402

# Telemetry columns mktable is expected to merge into visits.parquet. Grouped so a page
# can say which group is missing rather than listing 13 names.
TELEMETRY_GROUPS = {
    'ESS air temps': ['cam_air_temp', 'm2_air_temp', 'm1m3_air_temp', 'outside_temp'],
    'derived deltas': ['m2_delta_t', 'dome_delta_t', 'cam_m1m3_delta_t'],
    'M1M3 gradients': ['x_gradient', 'y_gradient', 'z_gradient', 'radial_gradient'],
    'TMA truss': ['tma_truss_temp_pxpy', 'tma_truss_temp_mxmy'],
    # Camera-body temps live ONLY in the combined visits.parquet: the backfill writes
    # per-chunk camera_telemetry.parquet sidecars and --merge joins them into the
    # combined file, leaving the per-chunk visits.parquet pristine. So this group is
    # reported from the combined table, not per chunk.
    'camera body': ['cam_AverageTemp', 'cam_AmbAirtemp'],
}
COMBINED_ONLY_GROUPS = ('camera body',)

# Commanded-DOF columns, none of which the pipeline currently writes. Each entry names the
# quantity, the column prefix a backfill would use, and where it would come from.
DOF_EXPECTED = [
    ('Trim (accumulated offset)', 'dof',
     'EFD MTAOS.logevent_degreeOfFreedom aggregatedDoF0..49',
     'aos_trim.fetch_aggregated_dof_for_visits'),
    ('Tweak (per-iteration correction)', 'tweak_dof',
     'not retrievable -- derive by differencing Trim across a re-alignment',
     'derived from Trim + event_ids'),
    ('Hexapod LUT', 'lut_dof',
     'EFD MTHexapod.logevent_compensationOffset (dof0-9)',
     'aos_trim.fetch_hexapod_lut_for_visits'),
    ('Mirror LUT (M1M3 + M2 bending)', 'lut_dof',
     'EFD MTM1M3 elevation zForces + MTM2 lutGravity -> bending (dof10-49)',
     'aos_trim.fetch_mirror_lut_for_visits'),
]


def _table_cols(path):
    """Column names of a parquet file, or None if it does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return list(pq.ParquetFile(str(p)).schema_arrow.names)


def _nrows(path):
    """Row count of a parquet file without reading it, or None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    return int(pq.ParquetFile(str(p)).metadata.num_rows)


def _finite_fraction(path, cols):
    """Fraction of finite values per column, as {col: frac}; missing cols are omitted.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Parquet file to read.
    cols : `list` [`str`]
        Candidate column names.

    Returns
    -------
    frac : `dict` [`str`, `float`]
        Column name to finite fraction (dimensionless, 0-1).
    """
    have = _table_cols(path)
    if have is None:
        return {}
    want = [c for c in cols if c in have]
    if not want:
        return {}
    df = pq.read_table(str(path), columns=want).to_pandas()
    out = {}
    for c in want:
        v = df[c].to_numpy(dtype=float, na_value=np.nan)
        out[c] = float(np.isfinite(v).mean()) if v.size else 0.0
    return out


def gather(ps, out_root, chunks):
    """Collect per-chunk status for one param_set.

    Parameters
    ----------
    ps : `str`
        param_set name.
    out_root : `pathlib.Path`
        Output root, normally ``output``.
    chunks : `list` [`str`]
        Chunk directory names, e.g. ``20260315_20260327``.

    Returns
    -------
    rows : `list` [`dict`]
        One dict per chunk, with row counts, day_obs span, and telemetry fractions.
    combined : `dict`
        Row counts and columns of the three combined tables.
    """
    base = out_root / ps
    rows = []
    for ch in chunks:
        d = base / 'chunks' / ch
        rec = {'param_set': ps, 'chunk': ch,
               'n_visits': _nrows(d / 'visits.parquet'),
               'n_donuts': _nrows(d / 'donuts.parquet'),
               'n_fits': _nrows(d / 'fits.parquet')}
        vp = d / 'visits.parquet'
        cols = _table_cols(vp)
        rec['visits_cols'] = len(cols) if cols else 0
        if cols and 'day_obs' in cols:
            dv = pq.read_table(str(vp), columns=['day_obs']).to_pandas()['day_obs']
            dv = dv.to_numpy(dtype=float)
            dv = dv[np.isfinite(dv)]
            rec['day_obs_min'] = int(dv.min()) if dv.size else None
            rec['day_obs_max'] = int(dv.max()) if dv.size else None
        else:
            rec['day_obs_min'] = rec['day_obs_max'] = None
        # nollIndices consistency within the chunk
        rec['noll'] = None
        if cols and 'nollIndices' in cols:
            ni = pq.read_table(str(vp), columns=['nollIndices']).to_pandas()['nollIndices']
            sets = {tuple(int(x) for x in v) for v in ni if v is not None}
            rec['n_noll_sets'] = len(sets)
            if len(sets) == 1:
                rec['noll'] = next(iter(sets))
        else:
            rec['n_noll_sets'] = 0
        for gname, gcols in TELEMETRY_GROUPS.items():
            if gname in COMBINED_ONLY_GROUPS:
                continue                      # reported from the combined table instead
            fr = _finite_fraction(vp, gcols)
            rec[f'tel::{gname}'] = (float(np.mean(list(fr.values()))) if fr else None)
        # camera_telemetry.parquet is the per-chunk source of truth for cam_* columns
        rec['has_camera_sidecar'] = (d / 'camera_telemetry.parquet').exists()
        rows.append(rec)

    combined = {}
    for name in ('visits', 'fits', 'donuts'):
        p = base / f'{name}.parquet'
        combined[name] = {'n': _nrows(p), 'cols': _table_cols(p)}
    return rows, combined


def _page_inventory(pdf, ps, rows, combined):
    """Page 1: per-chunk counts and the per-chunk vs combined consistency check."""
    fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.34 * len(rows) + 3.2)), dpi=150)
    ax.axis('off')
    head = ['chunk', 'day_obs span', 'visits', 'donuts', 'fits', 'v.cols', 'noll sets']
    body = []
    for r in rows:
        span = ('--' if r['day_obs_min'] is None
                else f"{r['day_obs_min']}-{r['day_obs_max']}")
        body.append([r['chunk'], span,
                     '--' if r['n_visits'] is None else f"{r['n_visits']}",
                     '--' if r['n_donuts'] is None else f"{r['n_donuts']}",
                     '--' if r['n_fits'] is None else f"{r['n_fits']}",
                     f"{r['visits_cols']}", f"{r['n_noll_sets']}"])
    sums = {k: sum(r[f'n_{k}'] or 0 for r in rows) for k in ('visits', 'donuts', 'fits')}
    body.append(['CHUNK SUM', '', f"{sums['visits']}", f"{sums['donuts']}",
                 f"{sums['fits']}", '', ''])
    body.append(['COMBINED', '',
                 f"{combined['visits']['n']}", f"{combined['donuts']['n']}",
                 f"{combined['fits']['n']}",
                 f"{len(combined['visits']['cols'] or [])}", ''])
    t = ax.table(cellText=body, colLabels=head, loc='upper center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.25)
    # Flag the consistency check in the title rather than making the reader subtract.
    bad = [k for k in ('visits', 'donuts', 'fits')
           if combined[k]['n'] is not None and sums[k] != combined[k]['n']]
    verdict = ('chunk sums match the combined tables' if not bad
               else 'MISMATCH in ' + ', '.join(bad) + ' -- re-run the combine')
    ax.set_title(f'{ps}\nchunk inventory ({len(rows)} chunks) -- {verdict}',
                 fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_coverage(pdf, ps, base):
    """Page 2: visits per day_obs, and the band / program mix."""
    vp = base / 'visits.parquet'
    cols = _table_cols(vp) or []
    want = [c for c in ('day_obs', 'band', 'science_program') if c in cols]
    if not want:
        return
    df = pq.read_table(str(vp), columns=want).to_pandas()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), dpi=150)
    if 'day_obs' in df:
        d = df['day_obs'].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        u, c = np.unique(d.astype(int), return_counts=True)
        axes[0].bar(range(len(u)), c, width=0.9)
        step = max(1, len(u) // 28)
        axes[0].set_xticks(range(0, len(u), step))
        axes[0].set_xticklabels([str(x) for x in u[::step]], rotation=90, fontsize=6)
        axes[0].set_xlabel('day_obs'); axes[0].set_ylabel('visits')
        axes[0].set_title(f'visits per night ({len(u)} nights, {len(d)} visits)',
                          fontsize=10)
        axes[0].grid(alpha=0.3, axis='y')
    for ax, key, lab in ((axes[1], 'band', 'band'),
                         (axes[2], 'science_program', 'science program')):
        if key not in df:
            ax.set_visible(False); continue
        v = df[key].astype(str)
        u, c = np.unique(v, return_counts=True)
        order = np.argsort(-c)
        u, c = u[order][:24], c[order][:24]
        ax.bar(range(len(u)), c)
        ax.set_xticks(range(len(u)))
        ax.set_xticklabels(u, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('visits'); ax.set_title(f'visits per {lab}', fontsize=10)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle(f'{ps} -- coverage', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _page_elev_rot(pdf, ps, base, elev_bin, rot_bin):
    """Page 3: elevation x rotator coverage over all chunks, in deg."""
    vp = base / 'visits.parquet'
    cols = _table_cols(vp) or []
    if not {'alt', 'rotator_angle'} <= set(cols):
        return
    df = pq.read_table(str(vp), columns=['alt', 'rotator_angle']).to_pandas()
    elev = alt_to_deg(df['alt'].to_numpy(dtype=float))
    rot = df['rotator_angle'].to_numpy(dtype=float)
    m = np.isfinite(elev) & np.isfinite(rot)
    elev, rot = elev[m], rot[m]
    if elev.size == 0:
        return
    re_ = centered_edges(np.nanmin(rot), np.nanmax(rot), rot_bin)
    ee = centered_edges(np.nanmin(elev), np.nanmax(elev), elev_bin)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.30 * (len(re_) - 1) + 3.0),
                                    max(6.0, 0.30 * (len(ee) - 1) + 2.5)), dpi=150)
    text_hist2d(rot, elev, ax=ax, xbins=re_, ybins=ee, fontsize=7)
    ax.set_xlabel('rotator angle (deg)'); ax.set_ylabel('elevation (deg)')
    ax.set_title(f'{ps}\nelevation x rotator coverage, all chunks '
                 f'({elev.size} visits; bins {elev_bin:g} x {rot_bin:g} deg)',
                 fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_telemetry(pdf, ps, rows):
    """Page 4: finite fraction per telemetry group per chunk."""
    gnames = [g for g in TELEMETRY_GROUPS if g not in COMBINED_ONLY_GROUPS]
    M = np.full((len(rows), len(gnames)), np.nan)
    for i, r in enumerate(rows):
        for j, g in enumerate(gnames):
            v = r.get(f'tel::{g}')
            if v is not None:
                M[i, j] = v
    fig, ax = plt.subplots(figsize=(max(8.0, 1.7 * len(gnames) + 3.5),
                                    max(4.0, 0.34 * len(rows) + 2.6)), dpi=150)
    im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(gnames)))
    ax.set_xticklabels(gnames, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['chunk'] for r in rows], fontsize=7)
    for i in range(len(rows)):
        for j in range(len(gnames)):
            txt = '--' if not np.isfinite(M[i, j]) else f'{100 * M[i, j]:.0f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='k' if (np.isfinite(M[i, j]) and M[i, j] > 0.35) else 'w')
    fig.colorbar(im, ax=ax, shrink=0.8, label='finite fraction (%)')
    n_side = sum(1 for r in rows if r.get('has_camera_sidecar'))
    ax.set_title(f'{ps}\ntelemetry completeness per chunk  '
                 f'("--" = column absent; low % = built with --no-thermal)\n'
                 f'camera-body temps are combined-table only: '
                 f'{n_side}/{len(rows)} chunks have camera_telemetry.parquet',
                 fontsize=9)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _page_dof(pdf, ps, combined):
    """Page 5: are the commanded-DOF columns present anywhere?"""
    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=150)
    ax.axis('off')
    vcols = set(combined['visits']['cols'] or [])
    fcols = set(combined['fits']['cols'] or [])
    n_cam = sum(1 for c in vcols if c.startswith('cam_'))
    body = []
    for label, prefix, source, fetcher in DOF_EXPECTED:
        hits_v = sorted(c for c in vcols if c.startswith(prefix))
        hits_f = sorted(c for c in fcols if c.startswith(prefix))
        body.append([label,
                     f'{len(hits_v)} col' if hits_v else 'ABSENT',
                     f'{len(hits_f)} col' if hits_f else 'ABSENT',
                     source, fetcher])
    t = ax.table(cellText=body,
                 colLabels=['quantity', 'in visits', 'in fits', 'source', 'fetcher'],
                 loc='upper center', cellLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(6.5); t.scale(1, 1.5)
    ax.set_title(
        f'{ps}\ncommanded degree-of-freedom (DOF) columns\n'
        'Trim/Tweak/LUT are not written by mktable; see '
        'docs/status/dof_telemetry_availability.md\n'
        f'(for contrast, camera-body telemetry IS merged: {n_cam} cam_* columns '
        'in the combined visits table)', fontsize=9)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def run(ps, out_root, chunks, elev_bin, rot_bin):
    """Build the status PDF and parquet for one param_set. Returns the row dicts."""
    base = Path(out_root) / ps
    rows, combined = gather(ps, Path(out_root), chunks)
    out_dir = base / 'fam_processing'
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / 'chunk_status.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        _page_inventory(pdf, ps, rows, combined)
        _page_coverage(pdf, ps, base)
        _page_elev_rot(pdf, ps, base, elev_bin, rot_bin)
        _page_telemetry(pdf, ps, rows)
        _page_dof(pdf, ps, combined)

    # Machine-readable sidecar: drop the tuple/None-valued helper columns.
    import pyarrow as pa
    flat = []
    for r in rows:
        q = {k: v for k, v in r.items() if k != 'noll'}
        q['noll'] = (','.join(str(x) for x in r['noll']) if r['noll'] else '')
        flat.append(q)
    keys = list(flat[0]) if flat else []
    tbl = pa.table({k: pa.array([f.get(k) for f in flat]) for k in keys})
    pq.write_table(tbl, str(out_dir / 'chunk_status.parquet'))
    print(f'  wrote {pdf_path}')
    print(f'  wrote {out_dir / "chunk_status.parquet"}  ({len(rows)} chunks)')
    return rows, combined


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True,
                    help="param_set name, or 'all' for every one with an output dir")
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--snake-config', default='snake_config.yaml')
    ap.add_argument('--elev-bin', type=float, default=10.0,
                    help='elevation bin width in deg')
    ap.add_argument('--rot-bin', type=float, default=15.0,
                    help='rotator-angle bin width in deg')
    args = ap.parse_args()

    out_root = Path(args.output_root)
    if args.param_set == 'all':
        names = sorted(p.name for p in out_root.iterdir()
                       if p.is_dir() and (p / 'chunks').is_dir())
    else:
        names = [args.param_set]

    cfg = {}
    cp = Path(args.snake_config)
    if cp.exists():
        cfg = yaml.safe_load(cp.read_text()) or {}

    for ps in names:
        # Chunk list comes from the directories actually on disk, not the config, so a
        # chunk built and later dropped from the config still shows up.
        cdir = out_root / ps / 'chunks'
        if not cdir.is_dir():
            print(f'  {ps}: no chunks/ directory, skipping')
            continue
        chunks = sorted(p.name for p in cdir.iterdir() if p.is_dir())
        if not chunks:
            print(f'  {ps}: no chunks, skipping')
            continue
        print(f'[chunk_status] {ps}: {len(chunks)} chunks')
        run(ps, out_root, chunks, args.elev_bin, args.rot_bin)


if __name__ == '__main__':
    main()
