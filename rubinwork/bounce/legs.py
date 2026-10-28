"""Bounce-test leg definitions, pairing and paired-difference reduction.

A library: it computes and returns, and writes no data of its own.  FAM-triplet
telescope-position bounce tests move the telescope between a reference (A) set
and a throw (B) set within one night, so the optical-state change between the
two positions is measured as a time-ordered paired difference rather than as a
difference of nightly means.  This module holds that reduction — leg selection,
A/B pair formation, the per-(k, j) and per-DOF paired Δ with robust (MAD)
errors, and the regularized-recovery solvers' Δ — for both
`rubinwork.products.bounce_tables` (which writes the tables) and the `bounce`
study in `aos/` (which renders the PDFs).

The ~20 plotters that used to sit beside these functions stay with the study, in
`aos/bounce/code/bounce_lib.py`, which re-exports everything here so its own
callers are unaffected.

Notes
-----
Nothing here imports `matplotlib`, so a builder can run it headless.  The
regularized-inversion solvers come from `rubinwork.smatrix.regularized_inversion`,
where the `regularized_inversion` study derives and validates them; they are
deliberately not copied, so the two cannot drift.
"""
import numpy as np
import pandas as pd

from rubinwork.common.utils import alt_to_deg as _alt_to_deg

# The bounce legs of the two FAM telescope-position blocks, as the study ran
# them: BLOCK-T720 throws elevation at fixed rotator angle, BLOCK-T724 throws
# the camera rotator at fixed elevation.  Elevation and rotator windows are in
# deg (OCS).  `analysis_config.yaml`'s `bounce:` section overrides this, so
# these are the fallback rather than what a configured build uses.
DEFAULT_BOUNCES = [
    {'name': 'T720_elevation', 'description': 'Elevation 40 - 70 deg, rotator ~ 0',
     'program': 'BLOCK-T720', 'default_scheme': 'rbr',
     'reference': {'label': 'Elev=70', 'alt_range': [67.0, 73.0], 'rotator_range': [-3.0, 3.0]},
     'comparisons': [{'label': 'Elev=40', 'alt_range': [37.0, 43.0], 'rotator_range': [-3.0, 3.0]}]},
    {'name': 'T724_rotator', 'description': 'Rotator 60 - 0 deg, elevation ~ 70',
     'program': 'BLOCK-T724',
     # rotator bounce: only the camera hexapod moves, so also evaluate a 5-DOF /
     # 5-v-mode (Camera-hexapod-only) correction alongside the full 50/34.
     'camera_hexapod_only': True, 'default_scheme': '5_5',
     'reference': {'label': 'Rot=0', 'rotator_range': [-3.0, 3.0], 'alt_range': [67.0, 73.0]},
     'comparisons': [{'label': 'Rot=60', 'rotator_range': [57.0, 63.0], 'alt_range': [67.0, 73.0]}]},
]
# Camera-hexapod DOF indices in LABELS_50DOF (Cam_dz/dx/dy/rx/ry) for the 5/5 scheme.
CAM_HEX_DOF = [5, 6, 7, 8, 9]
# The AOS reduced DOF set for the 22/12 scheme: 10 rigid-body, the first 7 M1M3
# bending modes, the first 5 M2 bending modes.  An *index set*, NOT the first 22
# contiguous indices -- see the root CLAUDE.md's terminology section.
# Passed to build_ofc_svd as an explicit list, which carries it on OFCSvd.dof_idx.
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))

# The full bounce configuration schema: every knob both halves read, with the
# settings the study ran.  `analysis_config.yaml`'s `bounce:` section overrides
# entries of it; the plotting knobs are read only by the study half.
DEFAULT = dict(
    fit_prefix='z1toz6', focal_k_range=[1, 2, 3, 4, 5, 6], pupil_j_range=None,
    bounces=DEFAULT_BOUNCES, n_dof=50, n_keep=34, ofc_normalization_yaml=None,
    heatmap_vlim_um=None, heatmap_cell_fontsize=7, sig_vlim=5.0,
    pass_nsigma_threshold=3.5, pass_delta_threshold_um=0.1,
    pass_sigma_only_threshold=5.0, cross_scatter_zoom_um=0.1,
    ordinal_j_per_page=7, night_min_visits=3,
    # A/B position identification tolerances, in deg. Elevation centers 70 and
    # 75 are only 5 deg apart, so the half-width must stay below 2.5 deg;
    # measured BLOCK-T720/T724 positions sit within 0.3 deg of a center.
    ab_elev_halfwidth_deg=2.0, ab_rot_halfwidth_deg=2.0,
    vmode_ncols=5, vmode_rows_per_page=7, dof_ncols=2, dof_rows_per_page=5,
    add_dof_trim=False, trim_efd_topic='lsst.sal.MTAOS.logevent_degreeOfFreedom',
    trim_consdb_url='https://usdf-rsp.slac.stanford.edu/consdb',
    trim_time_col='mjd', trim_mjd_scale='tai',
    # Range-Bounded Recovery (RBR): the range-penalized alternative to the
    # truncated recovery, reported alongside it.  kappa is the ratio
    # |d_j| / r_j (dimensionless, recovered amplitude over allowed range) at
    # which the penalty reaches unit weight; power sets how fast it climbs
    # (penalty ~ ratio ** (2 * power)).  The defaults are the setting the
    # smatrix `regularized_inversion` study found best or near-best on five of
    # the six bounce legs.
    rbr_enable=True, rbr_kappa=4.0, rbr_power=3,
    # The RBR scheme's own mode count, barrier weight (µm^2 of wavefront) and
    # whether it solves over every DOF with the barrier as the only regularizer.
    # 50 / 1.0 / True is the 50/50 RBR the measured intrinsic is built with.
    rbr_n_keep=50, rbr_penalty_weight=1.0, rbr_full_dof=True,
    # The 22-DOF / 12-v-mode reduced scheme the AOS is expected to operate in.
    # Only the mode count is configurable: the 22 DOF are an *index set*, fixed in
    # code as `DOF22`, not a count that could be read from here.
    reduced_enable=True, n_keep_reduced=12,
    # The quadratic motion penalty the ts_ofc Optimal Integral Controller (OIC)
    # already carries, scored as a fourth scheme.  `oic_rho` is the OIC's
    # `motion_penalty`, dimensionless.  ts_ofc ships 0.0, i.e. the penalty is
    # inactive as delivered, so a value must be chosen; this one is *consumed*,
    # and the scan it comes from is in
    # smatrix/regularized_inversion/regularized_inversion.md.  Do not re-derive it here.
    oic_enable=True, oic_rho=1.0e-3,
    # LUT fits of the default scheme's paired-Δ DOF against the B-set position:
    # a polynomial with no constant term in (x - reference), so the correction is
    # zero at the reference (elevation 70 deg, camera rotator angle 0 deg).
    lut_order_elevation=2, lut_order_rotator=1,
    lut_ref_elevation_deg=70.0, lut_ref_rotator_deg=0.0,
    lut_range_elevation_deg=[30.0, 75.0], lut_range_rotator_deg=[0.0, 60.0],
    lut_nsigma=3.0)

# Recovery schemes by `default_scheme` key: the per-leg Δ key in the analysis
# result and the per-(night, leg) FWHM column.
SCHEME_KEYS = {
    '50_34': ('dof_deltas', 'fwhm_after_default'),
    'rbr': ('rbr_deltas', 'fwhm_after_rbr'),
    '22_12': ('dof22_deltas', 'fwhm_after_22_12'),
    'oic': ('dof_oic_deltas', 'fwhm_after_oic'),
    '5_5': ('dof5_deltas', 'fwhm_after_5_5'),
}

RBR_METHOD_NAME = 'Range-Bounded Recovery (RBR)'


def scheme_label(key, cfg):
    """Legend name of a recovery scheme, with its mode counts and knobs.

    Parameters
    ----------
    key : `str`
        A `SCHEME_KEYS` key.
    cfg : `dict`
        The resolved bounce configuration.

    Returns
    -------
    label : `str`
    """
    if key == 'rbr':
        return (f'RBR {cfg["n_dof"]}/{cfg["rbr_n_keep"]} '
                f'(kappa={cfg["rbr_kappa"]:g}, power={cfg["rbr_power"]}, '
                f'both dimensionless)')
    if key == '22_12':
        return f'{len(DOF22)}/{cfg["n_keep_reduced"]} reduced DOF set'
    if key == 'oic':
        return f'OIC {cfg["n_dof"]}/{cfg["n_keep"]} (rho={cfg["oic_rho"]:g} dimensionless)'
    if key == '5_5':
        return '5/5 camera hexapod'
    return f'{cfg["n_dof"]}/{cfg["n_keep"]} truncated'


def default_scheme(bounce):
    """The `SCHEME_KEYS` key a bounce is read in: 5/5 if camera-hexapod-only, else RBR."""
    key = str(bounce.get('default_scheme')
              or ('5_5' if bounce.get('camera_hexapod_only') else 'rbr'))
    if key not in SCHEME_KEYS:
        raise ValueError(f'{bounce.get("name")}: unknown default_scheme {key!r}; '
                         f'known: {sorted(SCHEME_KEYS)}')
    return key
RBR_DEFAULTS = {'kappa': 4.0, 'power': 3}


def rbr_module():
    """The shared regularized-inversion solvers.

    Returns
    -------
    mod : `module`
        `rubinwork.smatrix.regularized_inversion`, supplying `invert_truncated`,
        `invert_range_penalty`, `invert_oic`, `oic_authority`,
        `achieved_residual` and `dof_range_vector`.

    Notes
    -----
    The solvers are deliberately *not* copied here: the methods are derived and
    validated in the `smatrix` topic's `regularized_inversion` study, and a
    second copy would be free to drift from it.
    """
    from rubinwork.smatrix import regularized_inversion as ri
    return ri
def filter_visits(fit_table, *, alt_range=None, rotator_range=None,
                  day_obs_range=None, seq_num_range=None,
                  program=None, mask=None):
    """Build a boolean mask for visits matching every supplied criterion.

    All ranges are inclusive.  Missing keys are unrestricted.

    `program` matches the `science_program` column (case-sensitive
    exact match).  Pass a single string or a list/tuple of strings;
    the result is the union of all matches.  A visit with no program
    (missing or empty) never matches.  Raises `KeyError` if the table
    has no `science_program` column.
    """
    n = len(fit_table)
    keep = np.ones(n, dtype=bool)
    if alt_range is not None and 'alt' in fit_table.colnames:
        alt_deg = _alt_to_deg(fit_table['alt'])
        keep &= (alt_deg >= alt_range[0]) & (alt_deg <= alt_range[1])
    if rotator_range is not None and 'rotator_angle' in fit_table.colnames:
        rot = np.asarray(fit_table['rotator_angle'], dtype=float)
        keep &= (rot >= rotator_range[0]) & (rot <= rotator_range[1])
    if day_obs_range is not None and 'day_obs' in fit_table.colnames:
        d = np.asarray(fit_table['day_obs']).astype(int)
        keep &= (d >= int(day_obs_range[0])) & (d <= int(day_obs_range[1]))
    if seq_num_range is not None and 'seq_num' in fit_table.colnames:
        s = np.asarray(fit_table['seq_num']).astype(int)
        keep &= (s >= int(seq_num_range[0])) & (s <= int(seq_num_range[1]))
    if program is not None:
        if 'science_program' not in fit_table.colnames:
            raise KeyError("program filter requested but fit_table has no "
                           "'science_program' column")
        sp = np.asarray(fit_table['science_program']).astype(str)
        if isinstance(program, (list, tuple, set)):
            prog_set = {str(p) for p in program}
            keep &= np.array([s in prog_set for s in sp])
        else:
            keep &= (sp == str(program))
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    return keep


def visit_list_str(fit_table, mask, max_per_day=20):
    """Pretty-print summary of which (day_obs, seq_num) visits survive a mask.

    Returns a string with one line per day_obs.  Truncates the seq_num
    list if it has more than `max_per_day` entries.
    """
    if not np.any(mask):
        return '    (no visits)'
    sub = fit_table[mask]
    dobs = np.asarray(sub['day_obs']).astype(int)
    snum = np.asarray(sub['seq_num']).astype(int)
    order = np.lexsort((snum, dobs))
    dobs = dobs[order]; snum = snum[order]
    lines = []
    for d in sorted(set(dobs.tolist())):
        s_list = sorted(snum[dobs == d].tolist())
        if len(s_list) > max_per_day:
            shown = (', '.join(str(x) for x in s_list[:max_per_day // 2])
                     + ', …, '
                     + ', '.join(str(x) for x in s_list[-max_per_day // 2:]))
            lines.append(f'    {d}: {len(s_list):3d} seq_num — '
                         f'[{shown}]')
        else:
            lines.append(f'    {d}: {len(s_list):3d} seq_num — '
                         f'{s_list}')
    return '\n'.join(lines)


def stats_per_kj(fit_table, mask, prefix, k_range, j_range):
    """Per-(k, j) median, robust RMS (1.4826*MAD), SEM of the median, n.

    SEM (standard error of the median) for normally distributed values
    is `1.2533 * sigma / sqrt(n)`; we use the MAD-based sigma estimate.
    """
    sub = fit_table[mask]
    out = {}
    for j in j_range:
        for k in k_range:
            col = f'{prefix}_z{j}_c{k}'
            if col not in sub.colnames:
                continue
            vals = np.asarray(sub[col], dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(len(vals))
            if n < 3:
                out[(int(k), int(j))] = {
                    'median': np.nan, 'sigma_mad': np.nan,
                    'sem': np.nan, 'n': n}
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            sigma_mad = 1.4826 * mad
            sem = 1.2533 * sigma_mad / np.sqrt(n)
            out[(int(k), int(j))] = {
                'median': med, 'sigma_mad': sigma_mad,
                'sem': sem, 'n': n}
    return out


def diff_stats(stats_comp, stats_ref):
    """Difference (comparison - reference) per (k, j) with quadrature errors."""
    out = {}
    keys = set(stats_comp.keys()) & set(stats_ref.keys())
    for kj in keys:
        a = stats_comp[kj]; b = stats_ref[kj]
        if not (np.isfinite(a['median']) and np.isfinite(b['median'])):
            out[kj] = {'delta': np.nan, 'err': np.nan,
                       'sig': np.nan,
                       'n_comp': a['n'], 'n_ref': b['n']}
            continue
        delta = a['median'] - b['median']
        err = float(np.sqrt(a['sem'] ** 2 + b['sem'] ** 2))
        sig = delta / err if err > 0 else np.nan
        out[kj] = {'delta': delta, 'err': err, 'sig': sig,
                   'n_comp': a['n'], 'n_ref': b['n']}
    return out


def _kj_to_array(stats_dict, k_list, j_list, key):
    """Pack one statistic into a (n_k, n_j) array for imshow."""
    Z = np.full((len(k_list), len(j_list)), np.nan)
    for (k, j), s in stats_dict.items():
        if k in k_list and j in j_list:
            Z[k_list.index(k), j_list.index(j)] = s.get(key, np.nan)
    return Z


def to_long_df(stats, bounce_name, ref_label, comp_label,
                ref_stats, comp_stats, night='all'):
    """Long-format DataFrame for one bounce comparison."""
    rows = []
    for (k, j), d in sorted(stats.items()):
        r = ref_stats.get((k, j), {})
        c = comp_stats.get((k, j), {})
        rows.append({
            'bounce':      bounce_name,
            'night':       str(night),
            'reference':   ref_label,
            'comparison':  comp_label,
            'k': int(k), 'j': int(j),
            'n_ref':       int(r.get('n', 0)),
            'ref_median':  float(r.get('median', np.nan)),
            'ref_sigma':   float(r.get('sigma_mad', np.nan)),
            'ref_sem':     float(r.get('sem', np.nan)),
            'n_comp':      int(c.get('n', 0)),
            'comp_median': float(c.get('median', np.nan)),
            'comp_sigma':  float(c.get('sigma_mad', np.nan)),
            'comp_sem':    float(c.get('sem', np.nan)),
            'delta':       float(d.get('delta', np.nan)),
            'delta_err':   float(d.get('err', np.nan)),
            'significance': float(d.get('sig', np.nan)),
        })
    return pd.DataFrame(rows)


def bounce_program_mask(fit_table, b):
    """Boolean mask for all visits in a bounce's science_program(s)."""
    p = b.get('program')
    if isinstance(p, (list, tuple, set)):
        progs = [str(x) for x in p]
    elif p is not None:
        progs = [str(p)]
    else:
        return np.ones(len(fit_table), dtype=bool)
    return filter_visits(fit_table, program=progs)


def ab_position_table(fit_table, b, elev_halfwidth_deg=None,
                      rot_halfwidth_deg=None, min_block=2):
    """Identify the A and B telescope positions of a bounce, per night.

    A bounce alternates between a reference position **A** (BLOCK-T720:
    elevation 70 deg, rotator 0 deg) and a comparison position **B**. This
    walks each night's visits in `seq_num` order, groups them into contiguous
    blocks of one classified (elevation, rotator) position, and reports the
    block's extent. A night may contain more than one distinct B position —
    20260713 throws to elevation 75 deg and then to 30 deg, each interleaved
    with its own elevation 70 deg reference block.

    Positions are classified onto the `intrinsics_lib` grid: elevation
    centers (30, 40, 50, 60, 70, 75) deg and rotator centers (-60 … 60) deg in
    15 deg steps. A visit further than the half-width from every center is
    reported as `unclassified`, which is what flags a contiguous block with no
    clear value for the B set.

    Parameters
    ----------
    fit_table : `astropy.table.Table`
        Fit table with `day_obs`, `seq_num`, `alt` (deg or rad),
        `rotator_angle` (deg), and optionally `band`.
    b : `dict`
        Bounce definition; only `program` is used, to select the visits.
    elev_halfwidth_deg : `float`, optional
        Elevation acceptance half-width in degrees. Defaults to the
        `intrinsics_lib` value (2.0 deg). Values for BLOCK-T720 / T724 sit
        within 0.3 deg of a center, so 2 deg is comfortable.
    rot_halfwidth_deg : `float`, optional
        Rotator acceptance half-width in degrees. Default 2.0 deg.
    min_block : `int`, optional
        A (night, position) set holding fewer than this many visits is still
        reported but marked `short`.

    Returns
    -------
    rows : `list` of `dict`
        One row per (day_obs, position) set, in (day_obs, seq_lo) order, with
        keys day_obs, position, elev_deg, rot_deg, seq_lo, seq_hi, n, n_blocks,
        band, alt_min_deg, alt_max_deg, rot_min_deg, rot_max_deg,
        unclassified, short, and `role` ('A' for the night's most-populated
        position, else 'B').

    Notes
    -----
    A bounce alternates A/B on consecutive visits, so the contiguous-block
    walk finds many single-visit blocks. Those are aggregated into one row per
    distinct (night, position); `n_blocks` records how many times the
    telescope returned to that position during the night.
    """
    hw_e = 2.0 if elev_halfwidth_deg is None else float(elev_halfwidth_deg)
    hw_r = 2.0 if rot_halfwidth_deg is None else float(rot_halfwidth_deg)
    rot_centers = np.arange(-60.0, 60.0 + 1e-9, 15.0)
    elev_centers = np.array([30.0, 40.0, 50.0, 60.0, 70.0, 75.0])

    def _snap(val, centers, hw):
        if not np.isfinite(val):
            return None
        d = np.abs(centers - val)
        i = int(np.argmin(d))
        return float(centers[i]) if d[i] <= hw else None

    pmask = bounce_program_mask(fit_table, b)
    if not np.any(pmask):
        return []
    ft = fit_table[pmask]
    dobs = np.asarray(ft['day_obs']).astype(int)
    snum = np.asarray(ft['seq_num']).astype(int)
    alt = _alt_to_deg(ft['alt'])
    rot = (np.asarray(ft['rotator_angle'], dtype=float)
           if 'rotator_angle' in ft.colnames else np.full(len(ft), np.nan))
    band = (np.asarray(ft['band']).astype(str) if 'band' in ft.colnames
            else np.full(len(ft), ''))

    rows = []
    for d in sorted(set(dobs.tolist())):
        sel = np.where(dobs == d)[0]
        sel = sel[np.argsort(snum[sel])]
        cur = None
        for i in sel:
            e = _snap(float(alt[i]), elev_centers, hw_e)
            r = _snap(float(rot[i]), rot_centers, hw_r)
            key = (e, r)
            if cur is not None and cur['key'] == key:
                cur['idx'].append(i)
                continue
            if cur is not None:
                rows.append(cur)
            cur = {'day_obs': d, 'key': key, 'idx': [i]}
        if cur is not None:
            rows.append(cur)

    # Aggregate the contiguous blocks into one row per (night, position): the
    # bounce alternates A/B every visit, so the raw blocks are the alternation
    # pattern, not the positions.
    agg = {}
    for blk in rows:
        e, r = blk['key']
        key = (blk['day_obs'], e, r)
        rec = agg.setdefault(key, {'idx': [], 'n_blocks': 0})
        rec['idx'].extend(blk['idx'])
        rec['n_blocks'] += 1

    out = []
    for (d, e, r), rec in agg.items():
        idx = np.array(sorted(rec['idx']))
        unclass = (e is None) or (r is None)
        pos = 'unclassified' if unclass else f'{int(e)}/{int(r)}'
        bands = sorted(set(band[idx].tolist()))
        out.append({
            'day_obs': d, 'position': pos,
            'elev_deg': e, 'rot_deg': r,
            'seq_lo': int(snum[idx].min()), 'seq_hi': int(snum[idx].max()),
            'n': int(len(idx)), 'n_blocks': int(rec['n_blocks']),
            'band': ','.join(b0 for b0 in bands if b0),
            'alt_min_deg': float(np.nanmin(alt[idx])),
            'alt_max_deg': float(np.nanmax(alt[idx])),
            'rot_min_deg': float(np.nanmin(rot[idx])),
            'rot_max_deg': float(np.nanmax(rot[idx])),
            'unclassified': bool(unclass),
            'short': bool(len(idx) < min_block),
        })
    out.sort(key=lambda row: (row['day_obs'], row['seq_lo']))

    # Role: per night, the position holding the most visits is the reference A.
    for d in sorted(set(row['day_obs'] for row in out)):
        nres = {}
        for row in out:
            if row['day_obs'] == d and not row['unclassified']:
                nres[row['position']] = nres.get(row['position'], 0) + row['n']
        a_pos = max(nres, key=nres.get) if nres else None
        for row in out:
            if row['day_obs'] == d:
                row['role'] = ('A' if row['position'] == a_pos
                               else ('?' if row['unclassified'] else 'B'))
    return out


def run_bounce(fit_table, b, prefix, k_list, j_list, day_obs=None,
               trim_segment=None):
    """Reference/comparison per-(k, j) stats and *paired* Δ for one bounce.

    The Δ is computed with the paired-difference method (see
    `form_pairs` / `paired_delta`): reference and comparison visits are
    paired in time order (never across a day_obs boundary), Δ is the
    median of the per-pair (comp − ref) differences, and the error is
    the scaled-MAD robust RMS of those differences / √n_pairs.  This
    absorbs slow time variation without modelling it.  `stats_per_kj`
    is still computed per setting for descriptive medians/RMS in the
    long table.  If `day_obs` is given, restrict to that single night.

    Returns {'ref_stats','ref_n','ref_mask',
             'comparisons': {label: {'comp_stats','comp_n','deltas',
                                     'comp_mask','pairs'}}}.
    """
    program = b.get('program')
    ref = b['reference']
    extra = {} if day_obs is None else {'day_obs_range': (int(day_obs), int(day_obs))}

    ref_kwargs = {k: v for k, v in ref.items() if k != 'label'}
    ref_kwargs.setdefault('program', program)
    ref_kwargs.update(extra)
    ref_mask = filter_visits(fit_table, **ref_kwargs)
    ref_stats = stats_per_kj(fit_table, ref_mask, prefix, k_list, j_list)

    comps = {}
    for comp in b['comparisons']:
        ck = {k: v for k, v in comp.items() if k != 'label'}
        ck.setdefault('program', program)
        ck.update(extra)
        cm = filter_visits(fit_table, **ck)
        cs = stats_per_kj(fit_table, cm, prefix, k_list, j_list)
        pairs = form_pairs(fit_table, ref_mask, cm, segment=trim_segment)
        comps[comp['label']] = {
            'comp_stats': cs, 'comp_n': int(cm.sum()),
            'deltas': paired_deltas_kj(fit_table, prefix, k_list, j_list, pairs),
            'comp_mask': cm, 'pairs': pairs,
        }
    return {'ref_stats': ref_stats, 'ref_n': int(ref_mask.sum()),
            'ref_mask': ref_mask, 'comparisons': comps}


def bounce_nights(fit_table, b, prefix, k_list, j_list, min_visits=3,
                  trim_segment=None):
    """Distinct day_obs nights where the reference and the comparison legs
    present that night have >= min_visits, with per-night run_bounce results.

    A night qualifies on the comparison legs it actually populates, not on
    every leg the bounce defines: a multi-leg bounce need not exercise every
    leg every night (e.g. the BLOCK-T720 elevation sweep throws to 40 deg in
    April/May and to 60 / 50 / 30 / 75 deg on individual July nights).  At
    least one leg must be populated, and every populated leg must clear
    min_visits; legs with no visits that night are simply absent from the
    per-night result.

    Returns {night: run_bounce_result} for qualifying nights (sorted).
    """
    program = b.get('program')
    pmask = filter_visits(fit_table, program=program)
    if not np.any(pmask):
        return {}
    nights = sorted(set(np.asarray(fit_table['day_obs'])[pmask]
                        .astype(int).tolist()))
    out = {}
    for d in nights:
        rb = run_bounce(fit_table, b, prefix, k_list, j_list, day_obs=d,
                        trim_segment=trim_segment)
        present = [c['comp_n'] for c in rb['comparisons'].values()
                   if c['comp_n'] > 0]
        ok = (rb['ref_n'] >= min_visits and bool(present)
              and all(n >= min_visits for n in present))
        if ok:
            out[d] = rb
    return out


def diff_of_deltas(deltas_a, deltas_b):
    """Difference of two Δ-DZ dicts (night A − night B) per (k, j), with
    quadrature-combined errors and significance."""
    out = {}
    for kj in set(deltas_a) & set(deltas_b):
        a, b = deltas_a[kj], deltas_b[kj]
        if not (np.isfinite(a.get('delta', np.nan))
                and np.isfinite(b.get('delta', np.nan))):
            out[kj] = {'delta': np.nan, 'err': np.nan, 'sig': np.nan}
            continue
        d = a['delta'] - b['delta']
        e = float(np.sqrt(a.get('err', np.nan) ** 2 + b.get('err', np.nan) ** 2))
        out[kj] = {'delta': d, 'err': e,
                   'sig': (d / e if e > 0 else np.nan)}
    return out


def passing_terms(deltas, delta_th, nsigma_th, sigma_only_th=None):
    """Set of (k, j) passing the significance cut.

    A term passes if  (|Δ| > delta_th AND |Δ/σ| > nsigma_th)  OR
    (sigma_only_th is not None AND |Δ/σ| > sigma_only_th).
    """
    out = set()
    for (k, j), d in deltas.items():
        dl = d.get('delta', np.nan); sg = d.get('sig', np.nan)
        if not (np.isfinite(dl) and np.isfinite(sg)):
            continue
        cutA = abs(dl) > delta_th and abs(sg) > nsigma_th
        cutB = sigma_only_th is not None and abs(sg) > sigma_only_th
        if cutA or cutB:
            out.add((k, j))
    return out


def form_pairs(fit_table, ref_mask, comp_mask, segment=None):
    """Time-ordered, non-overlapping (reference, comparison) visit pairs.

    Walks all reference+comparison visits in (day_obs, seq_num) order
    and greedily pairs each visit with the nearest following visit of
    the opposite setting, never crossing a day_obs boundary.  A run of
    same-setting visits keeps only the most recent unpaired one.

    `segment` (optional, aligned to `fit_table` rows) adds a second
    no-cross boundary: a pair is only formed when both visits share the
    same segment label.  Used to avoid pairing across an AOS Trim
    re-alignment (a change in the degreeOfFreedom event id) within a
    night.  Returns (ref_row, comp_row) integer row indices.
    """
    n = len(fit_table)
    setting = np.zeros(n, dtype=int)
    setting[np.asarray(ref_mask, bool)] = -1
    setting[np.asarray(comp_mask, bool)] = 1          # comp wins any overlap
    active = np.nonzero(setting != 0)[0]
    if active.size == 0:
        return []
    dobs = np.asarray(fit_table['day_obs']).astype(int)
    snum = np.asarray(fit_table['seq_num']).astype(int)
    seg = (np.asarray(segment) if segment is not None
           else np.zeros(n, dtype=int))
    order = active[np.lexsort((snum[active], dobs[active]))]
    pairs = []
    pend = None
    for i in order:
        if (pend is not None and dobs[i] == dobs[pend]
                and seg[i] == seg[pend]
                and setting[i] == -setting[pend]):
            r = pend if setting[pend] == -1 else i
            c = i if setting[i] == 1 else pend
            pairs.append((int(r), int(c)))
            pend = None
        else:
            pend = i
    return pairs


def paired_delta(values, pairs):
    """Paired-difference Δ for one quantity over (ref, comp) pairs.

    Δ = median(comp − ref) over pairs; err = (1.4826·MAD of the per-pair
    differences) / √n_pairs; sig = Δ / err.  Returns
    {delta, err, sig, n}  (n = number of finite pairs).
    """
    v = np.asarray(values, dtype=float)
    if not pairs:
        return {'delta': np.nan, 'err': np.nan, 'sig': np.nan, 'n': 0}
    diffs = np.array([v[c] - v[r] for (r, c) in pairs], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = int(diffs.size)
    if n < 1:
        return {'delta': np.nan, 'err': np.nan, 'sig': np.nan, 'n': 0}
    delta = float(np.median(diffs))
    sigma_mad = 1.4826 * float(np.median(np.abs(diffs - delta)))
    err = 1.2533 * sigma_mad / np.sqrt(n)        # SEM of a *median* (not mean)
    sig = delta / err if err > 0 else np.nan
    return {'delta': delta, 'err': float(err), 'sig': sig, 'n': n}


def paired_deltas_kj(fit_table, prefix, k_list, j_list, pairs):
    """Paired Δ per (k, j) for the DZ coefficients.  {(k, j): {...}}."""
    out = {}
    for j in j_list:
        for k in k_list:
            col = f'{prefix}_z{j}_c{k}'
            if col not in fit_table.colnames:
                continue
            out[(int(k), int(j))] = paired_delta(
                np.asarray(fit_table[col], dtype=float), pairs)
    return out


def paired_deltas_matrix(value_matrix, pairs, keys=None):
    """Paired Δ for each column of `value_matrix` (n_visits, n_q),
    aligned row-for-row to the fit_table the pairs index into.
    Returns {key: {delta,err,sig,n}} with key = keys[i] or int i."""
    M = np.asarray(value_matrix, dtype=float)
    nq = M.shape[1]
    ks = list(range(nq)) if keys is None else list(keys)
    return {ks[i]: paired_delta(M[:, i], pairs) for i in range(nq)}


def solver_dof_per_pair(W_all, pairs, svd, solver):
    """Recover DOF from each pair's Δ wavefront with one recovery scheme.

    Parameters
    ----------
    W_all : `numpy.ndarray`, (n_visits, n_kj)
        Per-visit DZ wavefront in µm of wavefront, row-aligned to the
        fit_table the pairs index into (the `_W` from `project_dz_table`).
    pairs : `list` of `tuple`
        `(ref_row, comp_row)` index pairs.
    svd : `OFCSvd`
        The scheme's own decomposition.  Its `n_dof` sets the width of the
        returned array, so a 22-DOF or 5-DOF scheme returns that many columns,
        in that scheme's own DOF order.
    solver : `callable`
        Maps `(dW, svd)` to recovered DOF, each in its own unit (µm for
        translations and bending-mode amplitudes, arcsec for rotations).

    Returns
    -------
    D : `numpy.ndarray`, (n_pairs, n_dof)
        Recovered DOF for each pair's Δ wavefront, in each DOF's own unit.
        Rows whose Δ wavefront is not finite are NaN, as are rows whose solve
        raised.

    Notes
    -----
    Each pair is inverted independently, because the regularized inversions are
    *nonlinear* and so do not commute with the median over pairs.  Cost is one
    solve per pair (a few tens of IRLS iterations on a 126 x 50 system for RBR,
    one linear solve for the truncated and OIC schemes — negligible here).
    """
    W = np.asarray(W_all, dtype=float)
    out = np.full((len(pairs), int(svd.V.shape[0])), np.nan)
    for i, (ref, comp) in enumerate(pairs):
        dW = W[comp] - W[ref]
        if not np.all(np.isfinite(dW)):
            continue
        try:
            out[i] = solver(dW, svd)
        except Exception:
            continue
    return out


def solver_deltas(W_all, pairs, svd, solver, keys=None):
    """Median / median-SEM DOF Δ over pairs for one recovery scheme.

    Same return form and same error definition as `paired_deltas_matrix`, so a
    regularized scheme's Δ is directly comparable to the default recovery's.

    Parameters
    ----------
    W_all : `numpy.ndarray`, (n_visits, n_kj)
        Per-visit DZ wavefront, µm of wavefront.
    pairs : `list` of `tuple`
        `(ref_row, comp_row)` index pairs.
    svd : `OFCSvd`
        The scheme's own decomposition.
    solver : `callable`
        Maps `(dW, svd)` to recovered DOF in each DOF's own unit.
    keys : `iterable` [`int`], optional
        Global DOF index per column of the scheme's DOF vector, e.g.
        `svd.dof_idx` for a reduced-DOF scheme.  `None` keys by column number,
        which is correct only for a 50-DOF scheme.

    Returns
    -------
    deltas : `dict`
        `{dof_index: {'delta', 'err', 'sig', 'n'}}`, `delta` in each DOF's own
        unit (µm or arcsec), `err` the SEM of the median over pairs, `sig` the
        dimensionless ratio `delta / err`.
    """
    if not pairs:
        return {}
    D = solver_dof_per_pair(W_all, pairs, svd, solver)
    ks = list(range(D.shape[1])) if keys is None else list(keys)
    out = {}
    for i in range(D.shape[1]):
        v = D[:, i]
        v = v[np.isfinite(v)]
        n = int(v.size)
        if n < 1:
            out[ks[i]] = {'delta': np.nan, 'err': np.nan, 'sig': np.nan, 'n': 0}
            continue
        delta = float(np.median(v))
        sigma_mad = 1.4826 * float(np.median(np.abs(v - delta)))
        err = 1.2533 * sigma_mad / np.sqrt(n)    # SEM of a median, as elsewhere
        out[ks[i]] = {'delta': delta, 'err': float(err),
                      'sig': (delta / err if err > 0 else np.nan), 'n': n}
    return out


def rbr_dof_per_pair(W_all, pairs, svd, ranges, *, kappa=None, power=None):
    """Recover DOF from each pair's Δ wavefront with RBR.

    Thin wrapper binding the RBR penalty shape onto `solver_dof_per_pair`.

    Parameters
    ----------
    W_all : `numpy.ndarray`, (n_visits, n_kj)
        Per-visit DZ wavefront in µm of wavefront.
    pairs : `list` of `tuple`
        `(ref_row, comp_row)` index pairs.
    svd : `OFCSvd`
        The 50-DOF / 34-v-mode decomposition the default recovery uses.
    ranges : `array_like`, (n_dof,)
        Allowed range `r_j` per DOF, each in that DOF's own unit (µm for
        translations and bending-mode amplitudes, arcsec for rotations).
    kappa : `float`, optional
        Dimensionless ratio `|d_j| / r_j` at which the penalty reaches unity
        weight.  Defaults to `RBR_DEFAULTS['kappa']`.
    power : `int`, optional
        Penalty exponent p, dimensionless; the penalty goes as the ratio to the
        `2 * p`.  Defaults to `RBR_DEFAULTS['power']`.

    Returns
    -------
    D : `numpy.ndarray`, (n_pairs, n_dof)
        RBR-recovered DOF per pair, in each DOF's own unit.
    """
    ri = rbr_module()
    kap = float(RBR_DEFAULTS['kappa'] if kappa is None else kappa)
    pw = int(RBR_DEFAULTS['power'] if power is None else power)
    r = np.asarray(ranges, dtype=float)
    return solver_dof_per_pair(
        W_all, pairs, svd,
        lambda dW, s: ri.invert_range_penalty(dW, s, r, kappa=kap, power=pw))


def rbr_deltas(W_all, pairs, svd, ranges, *, kappa=None, power=None,
               keys=None):
    """Median / median-SEM RBR DOF Δ over pairs.

    Thin wrapper binding the RBR penalty shape onto `solver_deltas`; see that
    function for the return form and the error definition.

    Returns
    -------
    deltas : `dict`
        `{dof_index: {'delta', 'err', 'sig', 'n'}}`, `delta` in each DOF's own
        unit (µm or arcsec), `err` the SEM of the median over pairs.
    """
    ri = rbr_module()
    kap = float(RBR_DEFAULTS['kappa'] if kappa is None else kappa)
    pw = int(RBR_DEFAULTS['power'] if power is None else power)
    r = np.asarray(ranges, dtype=float)
    return solver_deltas(
        W_all, pairs, svd,
        lambda dW, s: ri.invert_range_penalty(dW, s, r, kappa=kap, power=pw),
        keys=keys)


def leg_b_value(label):
    """Numeric B-set position parsed out of a comparison-leg label, in deg.

    Leg labels are written `Elev=40` or `Rot=60`, i.e. the axis name and the
    nominal B position in degrees.  Returns the number so a leg can be used as
    an x-axis coordinate rather than only as a categorical label.

    Parameters
    ----------
    label : `str`
        Comparison-leg label, e.g. `'Elev=40'` or `'Rot=60'`.

    Returns
    -------
    value : `float`
        The B-set elevation or camera-rotator angle in deg, or NaN if the
        label does not carry a number.
    """
    import re
    m = re.search(r'(-?\d+(?:\.\d+)?)', str(label))
    return float(m.group(1)) if m else float('nan')


def leg_axis_name(bounce):
    """Which telescope axis a bounce throws along — 'Elevation' or
    'Rotator angle' — inferred from its comparison-leg labels.

    Parameters
    ----------
    bounce : `dict`
        One bounce config entry, with a `comparisons` list of labelled legs.

    Returns
    -------
    name : `str`
        Axis name for an x-axis label; `'B set'` if the labels are not
        recognized.  The unit is deg in every case.
    """
    labs = ' '.join(str(c.get('label', '')) for c in bounce.get('comparisons', []))
    low = labs.lower()
    if 'elev' in low or 'alt' in low:
        return 'Elevation'
    if 'rot' in low:
        return 'Rotator angle'
    return 'B set'


def leg_pointing(fit_table, mask, day_obs=None):
    """Median measured pointing of the visits selected by a mask.

    The configured leg windows give only nominal positions, so the actual
    telescope position is taken from the visits themselves.  Restricting to one
    `day_obs` gives that night's position on the leg, which is what a per-night
    row should carry.

    Parameters
    ----------
    fit_table : `astropy.table.Table`
        The DZ fit table, carrying `alt`, `rotator_angle`, `day_obs` and
        (optionally) `science_program`.
    mask : `array_like` of `bool`
        Selects the visits on one leg.
    day_obs : `int`, optional
        Restrict to this night.  `None` pools every night on the leg.

    Returns
    -------
    pointing : `dict`
        `elevation_deg` and `rot_angle_deg` as medians in deg, `n_visits` as an
        `int`, and `block` as the `science_program` string (or a `+`-joined list
        if the selection spans more than one, which should not happen).  The
        angles are NaN when the selection is empty.
    """
    m = np.asarray(mask, dtype=bool).copy()
    if day_obs is not None and 'day_obs' in fit_table.colnames:
        m &= np.asarray(fit_table['day_obs']).astype(int) == int(day_obs)
    out = {'elevation_deg': float('nan'), 'rot_angle_deg': float('nan'),
           'n_visits': int(m.sum()), 'block': ''}
    if not m.any():
        return out
    if 'alt' in fit_table.colnames:
        out['elevation_deg'] = float(np.median(_alt_to_deg(fit_table['alt'])[m]))
    if 'rotator_angle' in fit_table.colnames:
        out['rot_angle_deg'] = float(np.median(
            np.asarray(fit_table['rotator_angle'], dtype=float)[m]))
    if 'science_program' in fit_table.colnames:
        progs = sorted(set(np.asarray(fit_table['science_program']).astype(str)[m]))
        out['block'] = '+'.join(progs)
    return out


def leg_night_coverage(bounce_results):
    """Which nights back each (bounce, comparison leg), as table rows.

    Answers "which nights have results for each value of the B conditions" —
    the precondition for a night-vs-night comparison, which needs 2 or more
    nights on the same leg.

    Parameters
    ----------
    bounce_results : `dict`
        `{bounce_name: bounce_result}` as assembled by `run_bounce.py`, whose
        per-leg blocks carry `deltas_by_night` and `comp_n`.

    Returns
    -------
    rows : `list` [`dict`]
        One row per (bounce, leg) with keys `bounce`, `comparison`,
        `b_value_deg`, `n_visits`, `n_nights`, `nights`, `n_pairs` and
        `scatter_pages` — the latter the number of night pairs
        `n_nights * (n_nights - 1) / 2` that a night-vs-night scatter can
        draw, which is 0 when only one night is available.
    """
    rows = []
    for name, br in bounce_results.items():
        for clabel, cb in br['comparisons'].items():
            nts = sorted(int(d) for d in cb.get('deltas_by_night', {}))
            n_nt = len(nts)
            rows.append({
                'bounce': name, 'comparison': clabel,
                'b_value_deg': leg_b_value(clabel),
                'n_visits': int(cb.get('comp_n', 0)),
                'n_pairs': len(cb.get('pairs', [])),
                'n_nights': n_nt,
                'nights': ', '.join(str(d) for d in nts) if nts else '-',
                'scatter_pages': n_nt * (n_nt - 1) // 2,
            })
    return rows


def resolve_config(param_set, mi_name, *, analysis_config=None, rbr_kappa=None,
                   rbr_power=None, no_rbr=False):
    """The resolved bounce configuration: `DEFAULT` under the config overrides.

    Parameters
    ----------
    param_set : `str`
        The long FAM param_set key `analysis_config.yaml`'s ``bounce:`` section
        is keyed on.
    mi_name : `str`
        The measured-intrinsic-wavefront settings entry name.
    analysis_config : `str` or `pathlib.Path`, optional
        Path to `analysis_config.yaml`.  `None` leaves
        `lsst.ts.intrinsic.wavefront.mi_config.analysis_section` to its default,
        which is relative to the current working directory.
    rbr_kappa : `float`, optional
        Override the Range-Bounded Recovery (RBR) ratio |d_j|/r_j (dimensionless)
        at which the penalty reaches unit weight.
    rbr_power : `int`, optional
        Override the RBR penalty exponent (dimensionless).
    no_rbr : `bool`, optional
        Skip RBR and report only the default truncated recovery.

    Returns
    -------
    cfg : `dict`
        Every knob the analysis and the plotters read.

    Notes
    -----
    `analysis_section` returns only the overrides, so `DEFAULT` carries the
    unoverridden legs.  The live override for the `danish_1_2` param_set
    redefines the BLOCK-T720 legs as a five-point elevation sweep, so a build
    that failed to resolve it would silently run one leg instead of five.
    """
    from pathlib import Path

    from lsst.ts.intrinsic.wavefront import mi_config as mc

    cfg = {**DEFAULT, **mc.analysis_section(
        'bounce', param_set, mi_name,
        config_path=(Path(analysis_config) if analysis_config else None))}
    if rbr_kappa is not None:
        cfg['rbr_kappa'] = float(rbr_kappa)
    if rbr_power is not None:
        cfg['rbr_power'] = int(rbr_power)
    if no_rbr:
        cfg['rbr_enable'] = False
    return cfg
