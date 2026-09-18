#!/usr/bin/env python
"""Change in focus — v-mode 1 — against exposure sequence number inside FAM blocks.

Measures how the uniform-defocus error drifts *within* a single Full Array Mode (FAM) block:
a run of ``acq, cwfs, cwfs`` triplets taken at one fixed pointing over tens of minutes. The
in-focus ``acq`` visit of each triplet carries a Corner Wavefront Sensor (CWFS) optical state
in the Consolidated Database (ConsDB), so v-mode 1 — essentially uniform defocus — can be read
per triplet and followed across the block.

The question matters because a FAM coadd averages the wavefront over a whole block, so any
within-block focus drift enters the coadd as a systematic. It is also a timescale the
``science_lut`` thermal model was never fitted on: that model is fitted and scored between
nights, where the telemetry actually moves.

Response, identical to ``science_lut`` so the two cannot drift apart::

    response [um of equivalent hexapod dz] = (v1_trim + MEASURED_SIGN * v1) / v1_per_um_dz

The hexapod look-up-table (LUT) baseline is deliberately excluded, as there, so the commanded
elevation dependence does not enter. The thermal correction is the ``science_lut`` model
**applied**, never refitted on FAM data.

Usage
-----
    python code/fam_focus/run_fam_focus.py
    python code/fam_focus/run_fam_focus.py --cache output/fam_focus/fam_acq.parquet
    python code/fam_focus/run_fam_focus.py --pointing-tol 5.0 --free-y

Needs ConsDB, so it runs on the Rubin Science Platform (RSP) or USDF only — unlike
``run_science_lut_analysis.py``, which reads parquet alone. ``truss_temp_mean_c`` is derived
inside `common.efd_db.join_consdb` from two Telescope Mount Assembly (TMA) truss thermometers
rather than stored, so there is no offline route to it. ``--cache`` makes the network cost
one-time.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))                     # repo root -> common/
sys.path.insert(0, str(_HERE.parents[2]))                     # aos/code -> flat modules
sys.path.insert(0, str(_HERE.parents[1] / 'science_lut'))     # the science_lut study

from common import efd_db                                                 # noqa: E402
from common.utils import nmad                                            # noqa: E402
import run_science_lut_analysis as A                                     # noqa: E402
from run_science_lut import LUT_EPOCH_OFFSET_NIGHTS, MEASURED_SIGN       # noqa: E402

# --------------------------------------------------------------------------- constants

#: FAM triplet programs. ConsDB returns HTTP 500 on ``LIKE '%T614%'`` against
#: ``science_program``, so the names are matched exactly; extend this tuple rather than
#: reaching for a wildcard.
DEFAULT_PROGRAMS = ('BLOCK-T614_triplets', 'BLOCK-T614')

#: First night to consider. Earlier nights predate the triplet program as it now runs.
DEFAULT_DAY_OBS_MIN = 20251101

#: Visits per selected set — 12 triplets, one ``acq`` each.
DEFAULT_SET_SIZE = 12

#: ``seq_num`` step between consecutive ``acq`` visits of a triplet sequence: acq, cwfs, cwfs.
DEFAULT_SEQ_STEP = 3

#: Pointing tolerance [deg] on altitude, azimuth and rotator within one block. Measured: the
#: selected sets hold pointing to about 0.01 deg, so 2.0 and the ``coadd`` study's 5.0 give the
#: same sets.
DEFAULT_POINTING_TOL = 2.0

#: A new block starts once ``seq_num`` reaches this far past the block's first visit. A
#: 12-triplet block spans exactly 33, so 36 keeps it whole and splits a back-to-back repeat.
DEFAULT_MAX_SEQ_SPAN = 36.0

#: The four M1M3 thermal gradient columns, read from ``visit_telemetry``.
GRADIENT_COLS = ('m1m3_x_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
                 'm1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m')

#: Panels per page of the per-set drift plots, as 4 columns by 3 rows.
PANELS_PER_PAGE = 12

#: Colours for the two series drawn on every panel.
RAW_COLOR = '#1f77b4'
CORR_COLOR = '#d62728'


# --------------------------------------------------------------------------- selection

def _wrapdiff(a, b):
    """Circular difference between two angles [deg].

    Parameters
    ----------
    a, b : `float`
        Angles in deg.

    Returns
    -------
    d : `float`
        Smallest absolute difference in deg, in [0, 180].
    """
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def assign_blocks(df, pointing_tol=DEFAULT_POINTING_TOL, max_seq_span=DEFAULT_MAX_SEQ_SPAN):
    """Label contiguous fixed-pointing blocks, the `coadd` study's greedy pointing-set walk.

    Within one ``(science_program, day_obs)`` the visits are walked in ``seq_num`` order and a
    new block starts when ``seq_num`` reaches `max_seq_span` past the block's first visit, or
    when altitude, azimuth or rotator angle drifts beyond `pointing_tol` from the block's first
    visit. Missing triplets inside the span are tolerated.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must carry ``science_program``, ``day_obs``, ``seq_num``, ``altitude_deg`` [deg],
        ``azimuth_deg_consdb`` [deg] and ``rotator_angle_deg`` [deg].
    pointing_tol : `float`, optional
        Tolerance in deg, applied to all three angles.
    max_seq_span : `float`, optional
        Maximum ``seq_num`` span of one block, dimensionless (a sequence-number difference).

    Returns
    -------
    df : `pandas.DataFrame`
        A copy sorted by ``(science_program, day_obs, seq_num)`` with an integer ``block``
        column; rows lacking any pointing angle are dropped.

    Notes
    -----
    The `coadd` implementation reads altitude and azimuth in **radians** from the FAM donut
    table and converts; the ConsDB columns used here are already in deg, so there is no
    conversion. Azimuth is compared with `_wrapdiff` so a block spanning 360 deg does not split.
    """
    v = df.dropna(subset=['altitude_deg', 'azimuth_deg_consdb', 'rotator_angle_deg']).copy()
    v = v.sort_values(['science_program', 'day_obs', 'seq_num']).reset_index(drop=True)

    block = np.full(len(v), -1, dtype=int)
    seq = v['seq_num'].to_numpy(float)
    alt = v['altitude_deg'].to_numpy(float)
    az = np.mod(v['azimuth_deg_consdb'].to_numpy(float), 360.0)
    rot = v['rotator_angle_deg'].to_numpy(float)

    nb = 0
    for _, pos in v.groupby(['science_program', 'day_obs']).indices.items():
        pos = np.sort(np.asarray(pos))
        cur, start_seq, ref = None, None, None
        for p in pos:
            new = (start_seq is None
                   or (seq[p] - start_seq) >= max_seq_span
                   or abs(alt[p] - ref[0]) > pointing_tol
                   or _wrapdiff(az[p], ref[1]) > pointing_tol
                   or abs(rot[p] - ref[2]) > pointing_tol)
            if new:
                cur = nb
                nb += 1
                start_seq = seq[p]
                ref = (alt[p], az[p], rot[p])
            block[p] = cur
    v['block'] = block
    return v


def select_sets(df, set_size=DEFAULT_SET_SIZE, seq_step=DEFAULT_SEQ_STEP,
                drop_lut_epoch=True, verbose=True):
    """Keep only blocks that are a clean run of `set_size` triplets.

    Three cuts, in order: exactly `set_size` visits in the block; a constant ``seq_num`` step
    of `seq_step`, validating the ``acq, cwfs, cwfs`` structure; and — by default — no night in
    `LUT_EPOCH_OFFSET_NIGHTS`, which ran a different hexapod look-up-table configuration and so
    sits thousands of µm from the rest.

    Parameters
    ----------
    df : `pandas.DataFrame`
        An `assign_blocks` result.
    set_size : `int`, optional
    seq_step : `int`, optional
        Required ``seq_num`` step, dimensionless. None skips the check.
    drop_lut_epoch : `bool`, optional
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        The kept visits, with ``set_id`` numbering the surviving sets from 0 in
        ``(day_obs, seq_num)`` order.
    info : `dict`
        Counts at each cut, for the validation page.
    """
    sized = [b for b, g in df[df.block >= 0].groupby('block') if len(g) == set_size]
    n_sized = len(sized)

    kept, n_bad_step = [], 0
    for b in sized:
        s = np.sort(df.loc[df.block == b, 'seq_num'].to_numpy(int))
        if seq_step is not None and not np.all(np.diff(s) == seq_step):
            n_bad_step += 1
            continue
        kept.append(b)

    out = df[df.block.isin(kept)].copy()
    n_before_epoch, nights_before = len(kept), out.day_obs.nunique()
    dropped_nights = sorted(set(int(d) for d in out.day_obs.unique())
                            & set(LUT_EPOCH_OFFSET_NIGHTS))
    if drop_lut_epoch and dropped_nights:
        bad = out.day_obs.isin(dropped_nights)
        n_dropped_sets = out.loc[bad, 'block'].nunique()
        out = out[~bad].copy()
    else:
        n_dropped_sets, dropped_nights = 0, []

    order = (out.groupby('block')[['day_obs', 'seq_num']].min()
             .sort_values(['day_obs', 'seq_num']).index.tolist())
    out['set_id'] = out.block.map({b: i for i, b in enumerate(order)})
    out = out.sort_values(['set_id', 'seq_num']).reset_index(drop=True)

    info = dict(n_blocks=int(df[df.block >= 0].block.nunique()), n_sized=n_sized,
                n_bad_step=n_bad_step, n_before_epoch=n_before_epoch,
                nights_before=int(nights_before), n_dropped_sets=int(n_dropped_sets),
                dropped_nights=dropped_nights, n_sets=int(out.set_id.nunique()),
                n_visits=len(out), n_nights=int(out.day_obs.nunique()),
                set_size=set_size, seq_step=seq_step)
    if verbose:
        print(f'blocks: {info["n_blocks"]} -> exactly {set_size} visits: {n_sized}'
              f' -> seq step {seq_step}: {n_before_epoch} (rejected {n_bad_step})')
        if n_dropped_sets:
            print(f'  dropped {n_dropped_sets} set(s) on LUT-epoch nights '
                  f'{dropped_nights} (--keep-lut-epoch-offset-nights to restore)')
        print(f'selected: {info["n_sets"]} sets, {info["n_visits"]} visits, '
              f'{info["n_nights"]} nights')
    return out, info


# --------------------------------------------------------------------------- assembly

def load_acq(variant=A.DEFAULT_VARIANT, day_obs_min=DEFAULT_DAY_OBS_MIN,
             programs=DEFAULT_PROGRAMS, consdb_url='auto', db_path=None, verbose=True):
    """Assemble the in-focus FAM ``acq`` visits with their optical state and telemetry.

    Reads ``visit_telemetry`` and the ``optical_state`` variant from the value-added database
    offline, then makes **one live ConsDB call** for the exposure metadata and the truss
    temperature.

    Parameters
    ----------
    variant : `str`, optional
        ``optical_state`` variant id.
    day_obs_min : `int`, optional
        First night, as ``YYYYMMDD``.
    programs : `sequence` [`str`], optional
        ``science_program`` values to keep.
    consdb_url : `str`, optional
    db_path : `str`, optional
    verbose : `bool`, optional

    Returns
    -------
    df : `pandas.DataFrame`
        One row per ``acq`` visit: identity, pointing [deg], band, the v-mode-1 components,
        ``truss_temp_mean_c`` [°C] and the four M1M3 gradients [°C per m].

    Notes
    -----
    ``truss_temp_mean_c`` is derived inside `common.efd_db.join_consdb`, not stored, so this
    function cannot run offline. `common.efd_db.visits` is queried for the gradients only —
    asking it for the truss temperature raises `duckdb.BinderException`.
    """
    cols = [c for c in GRADIENT_COLS if c in {c0[0] for c0 in efd_db.all_columns()}]
    df = efd_db.visits(day_obs_range=(day_obs_min, None), columns=cols, db_path=db_path)
    if verbose:
        print(f'visit_telemetry: n = {len(df)} visits, {df.day_obs.nunique()} nights')

    df = efd_db.join_consdb(df, groups=('meta', 'thermal'), consdb_url=consdb_url)
    if verbose:
        print(f'after join_consdb: {len(df)} rows, {len(df.columns)} columns')

    for c in df.columns:
        if c in ('band', 'img_type', 'science_program', 'obs_start'):
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df[df['img_type'] == 'acq'].copy()
    df = df[df['science_program'].isin(list(programs))].copy()
    if verbose:
        print(f"img_type = 'acq' in {list(programs)}: n = {len(df)}, "
              f'{df.day_obs.nunique()} nights')
        print(df.groupby('science_program').size().to_string())

    st = efd_db.optical_state(variant, day_obs_range=(day_obs_min, None), wide=True,
                              db_path=db_path)
    keep = ['visit_id'] + [c for c in ('v1', 'v1_lut', 'v1_trim') if c in st.columns]
    n_before = len(df)
    df = df.merge(st[keep], on='visit_id', how='inner')
    if verbose:
        print(f'with a valid {variant} optical state: n = {len(df)} of {n_before}, '
              f'{df.day_obs.nunique()} nights')
    return df


def attach_response(df, v1_per_um_dz, features, full):
    """Add the response, the thermal prediction and the corrected response.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying ``v1_trim``, ``v1`` and the model's feature columns.
    v1_per_um_dz : `float`
        Dimensionless v-mode-1 amplitude per µm of total hexapod dz travel.
    features : `list` [`str`]
        The fitted feature order.
    full : `dict`
        A `run_science_lut_analysis.fit_full` result; ``full['model']`` is applied unchanged.

    Returns
    -------
    df : `pandas.DataFrame`
        A copy with ``y``, ``pred`` and ``y_corr``, all in µm of equivalent hexapod dz
        (0.5 µm on each hexapod), and ``n_feat_missing`` counting NaN features per visit.
    """
    out = df.copy()
    out['y'] = (out['v1_trim'] + MEASURED_SIGN * out['v1']) / v1_per_um_dz
    X = out[list(features)].to_numpy(float)
    out['n_feat_missing'] = np.isnan(X).sum(axis=1)
    out['pred'] = full['model'].predict(X)
    out['y_corr'] = out['y'] - out['pred']
    return out


def set_summary(df, verbose=True):
    """One row per set: pointing, band, and the within-set scatter of both responses.

    Parameters
    ----------
    df : `pandas.DataFrame`
        An `attach_response` result carrying ``set_id``.
    verbose : `bool`, optional

    Returns
    -------
    out : `pandas.DataFrame`
        Per set: ``day_obs``, ``seq_num`` range, mean pointing [deg], band, ``n``, and for the
        raw and corrected response the median, peak-to-peak, standard deviation and drift
        (last − first), all in µm of equivalent hexapod dz.
    """
    rows = []
    for sid, g in df.groupby('set_id'):
        g = g.sort_values('seq_num')
        r = dict(set_id=int(sid), day_obs=int(g.day_obs.iloc[0]),
                 seq_first=int(g.seq_num.min()), seq_last=int(g.seq_num.max()),
                 n=len(g), band=g.band.mode().iat[0] if len(g.band.mode()) else '?',
                 altitude_deg=float(g.altitude_deg.mean()),
                 azimuth_deg=float(g.azimuth_deg_consdb.mean()),
                 rotator_angle_deg=float(g.rotator_angle_deg.mean()),
                 truss_p2p_c=float(np.ptp(g.truss_temp_mean_c.dropna()))
                 if g.truss_temp_mean_c.notna().any() else np.nan,
                 pred_p2p_um=float(np.ptp(g.pred.dropna())) if g.pred.notna().any() else np.nan)
        for tag, col in (('raw', 'y'), ('corr', 'y_corr')):
            v = g[col].dropna().to_numpy(float)
            r[f'{tag}_median_um'] = float(np.median(v)) if len(v) else np.nan
            r[f'{tag}_p2p_um'] = float(np.ptp(v)) if len(v) else np.nan
            r[f'{tag}_std_um'] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            r[f'{tag}_drift_um'] = float(v[-1] - v[0]) if len(v) > 1 else np.nan
        for c in GRADIENT_COLS:
            if c in g.columns and g[c].notna().any():
                r[f'{c}_p2p'] = float(np.ptp(g[c].dropna()))
        rows.append(r)
    out = pd.DataFrame(rows).sort_values('set_id').reset_index(drop=True)
    if verbose and len(out):
        n_better = int((out.corr_std_um < out.raw_std_um).sum())
        n_both = int((out.corr_std_um.notna() & out.raw_std_um.notna()).sum())
        print(f'\nwithin-set scatter [um of equivalent hexapod dz], median over '
              f'{len(out)} sets:')
        print(f'  uncorrected : p2p {out.raw_p2p_um.median():.1f}  '
              f'std {out.raw_std_um.median():.1f}')
        print(f'  corrected   : p2p {out.corr_p2p_um.median():.1f}  '
              f'std {out.corr_std_um.median():.1f}')
        print(f'  ratio corrected/uncorrected (dimensionless): '
              f'p2p {out.corr_p2p_um.median() / out.raw_p2p_um.median():.2f}  '
              f'std {out.corr_std_um.median() / out.raw_std_um.median():.2f}')
        print(f'  correction reduces the within-set std in {n_better} of {n_both} sets')
    return out


# --------------------------------------------------------------------------- pages

def page_opening(pdf, info, sets, df, variant, v1_per_um_dz, full, features, args):
    """Opening description: goal, response, sample and finding, over two pages.

    Split in two because one `page_text` sheet cannot hold all five blocks: the page spaces
    blocks by line count and the finding table would run off the bottom.
    """
    med_raw_p2p = sets.raw_p2p_um.median()
    med_corr_p2p = sets.corr_p2p_um.median()
    med_raw_std = sets.raw_std_um.median()
    med_corr_std = sets.corr_std_um.median()
    n_better = int((sets.corr_std_um < sets.raw_std_um).sum())
    n_both = int((sets.corr_std_um.notna() & sets.raw_std_um.notna()).sum())
    truss_c = dict(zip(full['features'], full['coef'])).get('truss_temp_mean_c', float('nan'))

    page1 = [
        ('The question', (
            'A Full Array Mode (FAM) block is a run of acq, cwfs, cwfs triplets at one fixed\n'
            'pointing over tens of minutes. This study measures how the uniform-defocus error\n'
            'drifts across such a block, from the in-focus acq visit of each triplet.\n\n'
            'It matters twice over: a FAM coadd averages the wavefront over a whole block, so\n'
            'a within-block drift enters the coadd as a systematic; and the science_lut thermal\n'
            'model is fitted between nights, so this is a timescale it never saw.')),
        ('What is measured', (
            'The Consolidated Database (ConsDB) records the optical state recovered at the four\n'
            'Corner Wavefront Sensors (CWFS) for every acq visit. The quantity is v-mode 1 of\n'
            'the Active Optics System (AOS) sensitivity matrix -- essentially uniform defocus:\n\n'
            '    response = v1(commanded Trim) - v1(measured state)\n\n'
            'the focus error the closed loop had accumulated but not yet corrected, expressed\n'
            'as equivalent hexapod dz [um]: total defocus travel, 0.5 um on the camera hexapod\n'
            'and 0.5 um on the M2 hexapod. The hexapod look-up-table (LUT) baseline is left out,\n'
            'as in science_lut, so the commanded elevation dependence does not enter.\n\n'
            f'    v1_per_um_dz = {v1_per_um_dz:.6e} dimensionless v-mode-1 amplitude\n'
            '                   per um of total dz travel')),
        ('Selection', (
            f'img_type = acq in {", ".join(args.programs)},\n'
            f'day_obs >= {args.day_obs_min}, with a valid {variant} optical state.\n\n'
            f'Contiguous blocks at fixed pointing, by the coadd study\'s greedy pointing-set\n'
            f'walk: altitude, azimuth and rotator constant to {args.pointing_tol:.1f} deg, and a\n'
            f'seq_num span under {args.max_seq_span:.0f}. Kept only where a block holds exactly\n'
            f'{info["set_size"]} visits with a constant seq_num step of {info["seq_step"]}.\n\n'
            f'    {info["n_sets"]} sets, {info["n_visits"]} visits, {info["n_nights"]} nights')),
    ]
    page2 = [
        ('The thermal correction is applied, not refitted', (
            'The correction is the science_lut model -- one band-independent Huber robust linear\n'
            'fit on the Telescope Mount Assembly (TMA) truss temperature and the four M1M3\n'
            'thermal gradients -- fitted on the science exposures and applied unchanged here.\n'
            f'Its truss coefficient is {truss_c:+.2f} um of equivalent hexapod dz per deg C.\n'
            'Nothing is fitted to FAM data.')),
        ('What it finds', (
            'The within-block drift is real: median within-set peak-to-peak\n'
            f'{med_raw_p2p:.1f} um of equivalent hexapod dz, against a science_lut out-of-fold\n'
            'residual normalized median absolute deviation (nMAD) of 61.3 um on the same\n'
            'quantity between nights.\n\n'
            'The thermal correction makes within-block scatter WORSE, not better:\n\n'
            '    quantity, median over sets   uncorrected   corrected   ratio [dimensionless,\n'
            '                                                            corr over uncorr]\n'
            f'    peak-to-peak [um dz]         {med_raw_p2p:11.1f}   {med_corr_p2p:9.1f}   '
            f'{med_corr_p2p / med_raw_p2p:.2f}\n'
            f'    standard deviation [um dz]   {med_raw_std:11.1f}   {med_corr_std:9.1f}   '
            f'{med_corr_std / med_raw_std:.2f}\n\n'
            f'It reduces the within-set standard deviation in only {n_better} of {n_both} sets.\n'
            'The last page shows why: inside one block the truss temperature moves only a few\n'
            'hundredths of a deg C, and the large between-nights coefficients turn that into a\n'
            'prediction swing of the same order as the signal, uncorrelated with it. The model\n'
            'describes night-to-night thermal drift, not what happens inside one block.')),
    ]
    A.page_text(pdf, 'FAM focus drift within a block', page1,
                subtitle=f'variant {variant}')
    A.page_text(pdf, 'FAM focus drift within a block', page2,
                subtitle='the correction, and what the measurement finds')


def page_selection(pdf, sets, info):
    """Validation: which nights the selected sets come from, and their pointing."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), height_ratios=[2.0, 1.0, 1.0])
    dates = A.day_obs_to_date(sets.day_obs.to_numpy())

    ax = axes[0]
    for band in A.BAND_ORDER:
        m = (sets.band == band).to_numpy()
        if not m.any():
            continue
        ax.scatter(dates[m], sets.set_id.to_numpy()[m], s=42,
                   color=A.BAND_COLORS.get(band, 'gray'), label=f'{band} (n={int(m.sum())})',
                   edgecolor='k', linewidth=0.4, zorder=3)
    ax.set_ylabel('set index')
    ax.set_title(f'{info["n_sets"]} selected sets of {info["set_size"]} FAM triplets '
                 f'over {info["n_nights"]} nights, coloured by band')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=6, loc='upper left')

    ax = axes[1]
    per_night = sets.groupby('day_obs').size()
    ax.bar(A.day_obs_to_date(per_night.index.to_numpy()), per_night.to_numpy(),
           width=0.8, color='#4c72b0')
    ax.set_ylabel('sets per night')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(dates, sets.altitude_deg, s=26, label='elevation [deg]', color='#55a868')
    ax.scatter(dates, sets.rotator_angle_deg, s=26, marker='^',
               label='camera rotator [deg]', color='#c44e52')
    ax.set_ylabel('angle [deg]')
    ax.set_xlabel('day_obs')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    drop = (f'; dropped {info["n_dropped_sets"]} LUT-epoch set(s) on '
            f'{info["dropped_nights"]}' if info['n_dropped_sets'] else '')
    fig.text(0.06, 0.012,
             f'{info["n_blocks"]} blocks -> {info["n_sized"]} with exactly {info["set_size"]} '
             f'visits -> {info["n_before_epoch"]} with seq_num step {info["seq_step"]} '
             f'(rejected {info["n_bad_step"]}){drop}. '
             f'Pointing tolerance {info.get("pointing_tol", float("nan")):.1f} deg on altitude, '
             f'azimuth and camera rotator.', fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def page_set_table(pdf, sets):
    """Per-set table: pointing, band and the scatter of both responses."""
    per = 44
    for i in range(0, len(sets), per):
        chunk = sets.iloc[i:i + per]
        lines = []
        head = (f'{"set":>4} {"day_obs":>9} {"seq0":>5} {"n":>3} {"bnd":>4} {"elev":>6} '
                f'{"az":>7} {"rot":>7} {"med":>9} {"p2p":>7} {"std":>7} '
                f'{"p2p_c":>7} {"std_c":>7} {"drift":>8}')
        lines.append(head)
        for _, r in chunk.iterrows():
            lines.append(f'{int(r.set_id):>4} {int(r.day_obs):>9} {int(r.seq_first):>5} '
                         f'{int(r.n):>3} {str(r.band):>4} {r.altitude_deg:>6.1f} '
                         f'{r.azimuth_deg:>7.1f} {r.rotator_angle_deg:>7.1f} '
                         f'{r.raw_median_um:>+9.1f} {r.raw_p2p_um:>7.1f} {r.raw_std_um:>7.1f} '
                         f'{r.corr_p2p_um:>7.1f} {r.corr_std_um:>7.1f} '
                         f'{r.raw_drift_um:>+8.1f}')
        body = '\n'.join(lines)
        A.page_text(pdf, 'Selected sets' + (f' ({i + 1}-{min(i + per, len(sets))})'
                                            if len(sets) > per else ''),
                    [('med, p2p, std, drift [um equiv hexapod dz]; _c = corrected; '
                      'elev, az, rot [deg]', body)])


def page_set_panels(pdf, df, sets, panels=PANELS_PER_PAGE, free_y=False):
    """Per-set drift panels: both responses against ``seq_num``, medians removed.

    Each series is offset to its own within-set median, so a panel shows the drift rather than
    the set's offset; the two medians are printed in the panel title.
    """
    import matplotlib.pyplot as plt

    ids = sets.set_id.tolist()
    if not free_y:
        span = []
        for sid in ids:
            g = df[df.set_id == sid]
            for col in ('y', 'y_corr'):
                v = g[col].dropna().to_numpy(float)
                if len(v):
                    span.append(np.ptp(v - np.median(v)))
        half = 0.5 * float(np.nanpercentile(span, 90)) * 1.25 if span else 1.0
        ylim = (-half, half)

    ncol, nrow = 4, 3
    for start in range(0, len(ids), panels):
        chunk = ids[start:start + panels]
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 8.5), sharex=False,
                                 sharey=not free_y)
        axes = np.atleast_1d(axes).ravel()
        for ax, sid in zip(axes, chunk):
            g = df[df.set_id == sid].sort_values('seq_num')
            row = sets[sets.set_id == sid].iloc[0]
            s = g.seq_num.to_numpy(float) - g.seq_num.min()
            for col, color, lab in (('y', RAW_COLOR, 'raw'),
                                    ('y_corr', CORR_COLOR, 'thermally corrected')):
                v = g[col].to_numpy(float)
                if np.isfinite(v).any():
                    ax.plot(s, v - np.nanmedian(v), 'o-', ms=3.4, lw=1.1, color=color,
                            label=lab)
            ax.axhline(0.0, color='k', lw=0.6, alpha=0.5)
            ax.set_title(f'set {int(sid)}  {int(row.day_obs)}  {row.band}  '
                         f'elev {row.altitude_deg:.0f} deg\n'
                         f'med raw {row.raw_median_um:+.0f}, p2p {row.raw_p2p_um:.0f} / '
                         f'corr {row.corr_p2p_um:.0f} um', fontsize=7.4)
            ax.grid(alpha=0.3)
            if not free_y:
                ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
        for ax in axes[len(chunk):]:
            ax.axis('off')
        for k, ax in enumerate(axes[:len(chunk)]):
            if k % ncol == 0:
                ax.set_ylabel('v1 - set median\n[um equiv hexapod dz]', fontsize=7.5)
            if k >= len(chunk) - ncol:
                ax.set_xlabel('seq_num - first [dimensionless]', fontsize=7.5)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False)
        fig.suptitle(f'Focus drift within each FAM set, sets {chunk[0]}-{chunk[-1]} '
                     f'of {ids[-1]}', fontsize=12)
        fig.tight_layout(rect=(0, 0.035, 1, 0.965))
        pdf.savefig(fig)
        plt.close(fig)


def page_summary(pdf, sets, df, full):
    """Closing page: the scatter comparison and the feature motion explaining it."""
    import matplotlib.pyplot as plt

    n_better = int((sets.corr_std_um < sets.raw_std_um).sum())
    n_both = int((sets.corr_std_um.notna() & sets.raw_std_um.notna()).sum())
    coef = dict(zip(full['features'], full['coef']))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    lim = (0.0, float(np.nanpercentile(
        np.concatenate([sets.raw_p2p_um.dropna(), sets.corr_p2p_um.dropna()]), 98)) * 1.1)
    ax.scatter(sets.raw_p2p_um, sets.corr_p2p_um, s=30, color='#4c72b0',
               edgecolor='k', linewidth=0.4)
    ax.plot(lim, lim, 'k--', lw=1, label='1:1 (no change)')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel('uncorrected within-set p2p [um equiv hexapod dz]')
    ax.set_ylabel('corrected within-set p2p [um equiv hexapod dz]')
    ax.set_title(f'Above the line = correction made it worse\n'
                 f'{n_both - n_better} of {n_both} sets above', fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bins = np.linspace(0.0, lim[1], 26)
    ax.hist(sets.raw_p2p_um.dropna(), bins=bins, histtype='step', lw=1.6, color=RAW_COLOR,
            label=f'uncorrected, median {sets.raw_p2p_um.median():.1f} um')
    ax.hist(sets.corr_p2p_um.dropna(), bins=bins, histtype='step', lw=1.6, color=CORR_COLOR,
            label=f'corrected, median {sets.corr_p2p_um.median():.1f} um')
    ax.set_xlabel('within-set p2p [um equiv hexapod dz]')
    ax.set_ylabel('sets')
    ax.set_title('Within-set peak-to-peak', fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(sets.pred_p2p_um, sets.raw_p2p_um, s=30, color='#55a868',
               edgecolor='k', linewidth=0.4)
    ok = sets.pred_p2p_um.notna() & sets.raw_p2p_um.notna()
    r_p = (sets.loc[ok, 'pred_p2p_um'].corr(sets.loc[ok, 'raw_p2p_um'])
           if ok.sum() > 2 else float('nan'))
    r_s = (sets.loc[ok, 'pred_p2p_um'].corr(sets.loc[ok, 'raw_p2p_um'], method='spearman')
           if ok.sum() > 2 else float('nan'))
    ax.set_xlabel('model prediction p2p within set [um equiv hexapod dz]')
    ax.set_ylabel('response p2p within set [um equiv hexapod dz]')
    ax.set_title(f'If the correction worked these would track\n'
                 f'Pearson r {r_p:+.3f}, Spearman rho {r_s:+.3f} '
                 f'(dimensionless, n = {int(ok.sum())})', fontsize=9.5)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(sets.raw_drift_um.dropna(), bins=24, color='#8172b2')
    ax.axvline(0.0, color='k', lw=1)
    ax.set_xlabel('drift, last - first [um equiv hexapod dz]')
    ax.set_ylabel('sets')
    ax.set_title(f'Drift is not monotonic: median '
                 f'{sets.raw_drift_um.median():+.1f} um, nMAD '
                 f'{nmad(sets.raw_drift_um.dropna().to_numpy(float)):.1f} um', fontsize=9.5)
    ax.grid(alpha=0.3)

    fig.suptitle('The thermal correction increases within-block scatter', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    pdf.savefig(fig)
    plt.close(fig)

    lines = [f'{"quantity":42s} {"median within-set p2p":>22s}  unit']
    lines.append('-' * 78)
    rows = [('response (Trim - measured)', sets.raw_p2p_um.median(),
             'um equiv hexapod dz'),
            ('model prediction', sets.pred_p2p_um.median(), 'um equiv hexapod dz'),
            ('truss temperature', sets.truss_p2p_c.median(), 'deg C')]
    for c in GRADIENT_COLS:
        k = f'{c}_p2p'
        if k in sets.columns:
            rows.append((c, sets[k].median(), 'deg C per m'))
    for name, val, unit in rows:
        lines.append(f'{name:42s} {val:>22.4f}  {unit}')

    ratio = sets.pred_p2p_um.median() / sets.raw_p2p_um.median()
    coef_lines = '\n'.join(
        f'  {f:32s} {coef[f]:+10.2f} um of equivalent hexapod dz per {A.FEATURE_UNITS.get(f, "?")}'
        for f in full['features'])

    A.page_text(pdf, 'Why the correction does not help inside a block',
                [('Within-set motion of the response and of every model input',
                  '\n'.join(lines)),
                 ('The prediction swings as much as the signal, uncorrelated with it',
                  f'prediction p2p over response p2p = {ratio:.2f} (dimensionless,\n'
                  f'median within-set prediction peak-to-peak over median within-set\n'
                  f'response peak-to-peak)\n\n'
                  f'The correction reduces the within-set standard deviation in only\n'
                  f'{n_better} of {n_both} sets; the median ratio is '
                  f'{sets.corr_std_um.median() / sets.raw_std_um.median():.2f} (dimensionless,\n'
                  f'corrected standard deviation over uncorrected).'),
                 ('The science_lut coefficients doing the amplifying',
                  coef_lines + '\n\n'
                  'Fitted between nights, where the truss temperature moves degrees rather\n'
                  'than hundredths of a degree. Inside one block the inputs barely move, so\n'
                  'the large coefficients turn telemetry noise into a prediction swing of the\n'
                  'same order as the drift being measured.')])


# --------------------------------------------------------------------------- driver

def build_pdf(out_pdf, df, sets, info, variant, v1_per_um_dz, full, features, args):
    """Write the document.

    Parameters
    ----------
    out_pdf : `pathlib.Path`
    df, sets, info : see `attach_response`, `set_summary`, `select_sets`
    variant : `str`
    v1_per_um_dz : `float`
    full : `dict`
    features : `list` [`str`]
    args : `argparse.Namespace`
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        page_opening(pdf, info, sets, df, variant, v1_per_um_dz, full, features, args)
        page_selection(pdf, sets, info)
        page_set_table(pdf, sets)
        page_set_panels(pdf, df, sets, free_y=args.free_y)
        page_summary(pdf, sets, df, full)
        d = pdf.infodict()
        d['Title'] = 'FAM focus drift within a block'
        d['Subject'] = f'v-mode 1 against seq_num, variant {variant}'
    print(f'wrote {out_pdf}')


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--variant', default=A.DEFAULT_VARIANT, help='optical_state variant id')
    p.add_argument('--day-obs-min', type=int, default=DEFAULT_DAY_OBS_MIN,
                   help='first night as YYYYMMDD')
    p.add_argument('--programs', nargs='+', default=list(DEFAULT_PROGRAMS),
                   help='science_program values holding the FAM triplets')
    p.add_argument('--set-size', type=int, default=DEFAULT_SET_SIZE,
                   help='triplets per selected set')
    p.add_argument('--seq-step', type=int, default=DEFAULT_SEQ_STEP,
                   help='required seq_num step between acq visits; 0 disables the check')
    p.add_argument('--pointing-tol', type=float, default=DEFAULT_POINTING_TOL,
                   help='altitude/azimuth/rotator tolerance within a block [deg]')
    p.add_argument('--max-seq-span', type=float, default=DEFAULT_MAX_SEQ_SPAN,
                   help='maximum seq_num span of one block')
    p.add_argument('--keep-lut-epoch-offset-nights', action='store_true',
                   help='keep the nights running a different hexapod LUT configuration')
    p.add_argument('--free-y', action='store_true',
                   help='autoscale each panel instead of sharing one y-range')
    p.add_argument('--science-lut-dir', default=None,
                   help='directory holding science_lut.parquet (default aos/output/science_lut)')
    p.add_argument('--out-dir', default=None,
                   help='output directory (default aos/output/fam_focus)')
    p.add_argument('--cache', default=None,
                   help='parquet path for the assembled acq table; read if present, else written')
    p.add_argument('--db-path', default=None, help='value-added database path')
    p.add_argument('--consdb-url', default='auto')
    args = p.parse_args(argv)

    aos = _HERE.parents[2]
    out_dir = pathlib.Path(args.science_lut_dir) if args.science_lut_dir \
        else aos / 'output' / 'science_lut'
    dest = pathlib.Path(args.out_dir) if args.out_dir else aos / 'output' / 'fam_focus'
    dest.mkdir(parents=True, exist_ok=True)

    v1_per_um_dz = A.v1_per_um_dz_value()
    print()

    # The science_lut model, fitted on science exposures and applied unchanged below.
    print('--- the science_lut thermal model (fitted on science exposures) ---')
    df_sci, features = A.load_target(out_dir, variant=args.variant,
                                     v1_per_um_dz=v1_per_um_dz,
                                     features=A.resolve_features(A.DEFAULT_FEATURES),
                                     verbose=True)
    full = A.fit_full(df_sci, features, model='huber', verbose=True)

    print('\n--- the FAM acq sample ---')
    cache = pathlib.Path(args.cache) if args.cache else None
    if cache and cache.exists():
        acq = pd.read_parquet(cache)
        print(f'read {len(acq)} rows from {cache}')
    else:
        acq = load_acq(variant=args.variant, day_obs_min=args.day_obs_min,
                       programs=args.programs, consdb_url=args.consdb_url,
                       db_path=args.db_path)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            acq.to_parquet(cache, index=False)
            print(f'cached {len(acq)} rows to {cache}')

    print('\n--- block selection ---')
    blocks = assign_blocks(acq, pointing_tol=args.pointing_tol,
                           max_seq_span=args.max_seq_span)
    sel, info = select_sets(blocks, set_size=args.set_size,
                            seq_step=args.seq_step or None,
                            drop_lut_epoch=not args.keep_lut_epoch_offset_nights)
    info['pointing_tol'] = args.pointing_tol
    if not len(sel):
        raise SystemExit('no sets selected; loosen --pointing-tol or --set-size')

    # Every selected set must be a clean triplet run -- assert, do not merely print.
    for sid, g in sel.groupby('set_id'):
        s = np.sort(g.seq_num.to_numpy(int))
        assert len(s) == args.set_size, f'set {sid}: {len(s)} visits'
        if args.seq_step:
            assert np.all(np.diff(s) == args.seq_step), f'set {sid}: seq step {np.diff(s)}'
            assert s[-1] - s[0] == args.seq_step * (args.set_size - 1), f'set {sid}: span'

    sel = attach_response(sel, v1_per_um_dz, features, full)
    n_missing = int((sel.n_feat_missing > 0).sum())
    if n_missing:
        print(f'note: {n_missing} of {len(sel)} visits have at least one NaN feature; the '
              f'model pipeline imputes them with the training median')
    sets = set_summary(sel, verbose=True)

    sel.to_parquet(dest / 'fam_focus_visits.parquet', index=False)
    sets.to_parquet(dest / 'fam_focus_sets.parquet', index=False)
    print(f'\nwrote {dest / "fam_focus_visits.parquet"} ({len(sel)} rows)')
    print(f'wrote {dest / "fam_focus_sets.parquet"} ({len(sets)} rows)')

    build_pdf(dest / 'fam_focus.pdf', sel, sets, info, args.variant, v1_per_um_dz,
              full, features, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
