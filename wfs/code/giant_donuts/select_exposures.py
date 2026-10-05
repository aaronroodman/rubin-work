"""Giant-donut exposure selection from the hexapod dz Trim.

Selects the 8 mm giant-donut exposures of a night without relying on the
free-text ``observation_reason`` label, by reading the aggregated degree-of-freedom
(DOF) Trim from the value-added database.  ``dof0`` is the M2 hexapod dz and
``dof5`` is the camera hexapod dz, both in µm (the ordering is documented in
`common/dof_telemetry.py`); the remaining 48 DOF must stay at the night's baseline
for an exposure to count as a clean giant donut.

The label is unreliable in both directions.  On ``day_obs`` 20251023 the two
exposures labelled ``intra_8mm_m1m3_b4`` show no bending-mode motion in the Trim at
all, and ``seq_num`` 339 sits at the giant intra Trim state while belonging to no
BLOCK-T626 sequence.  Selection is therefore Trim-first, with the exposure metadata
used only to reject exposures that the Trim alone would wrongly admit.

Import by putting the repository root on the path::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'value_added' / 'code'))
import efd_db  # noqa: E402

# Index of the hexapod dz degrees of freedom within the 50-DOF Trim vector.
DOF_M2_DZ = 0
DOF_CAM_DZ = 5

# A giant donut is nominally 8 mm of defocus.  Exposures are accepted on the
# magnitude of the dz excursion from the night's baseline, in µm.
GIANT_DZ_MIN_UM = 3000.0

# Tolerance, in µm, on a DOF being "unmoved" from the night's baseline.  The Trim
# is a commanded quantity and repeats exactly, so this only absorbs float noise.
DOF_QUIET_TOL_UM = 1.0

__all__ = [
    'DOF_M2_DZ', 'DOF_CAM_DZ', 'GIANT_DZ_MIN_UM',
    'baseline_trim', 'classify_defocus', 'select_giant_donuts',
]


def _dof_columns(n_dof=50):
    """Trim column names, in DOF order."""
    return [f'dof{i}' for i in range(n_dof)]


def baseline_trim(df, n_dof=50):
    """Per-night baseline Trim vector, taken as the per-DOF median.

    The median is the right estimator here because the defocal excursions are a
    minority of a night's exposures and are symmetric about the baseline, so they
    cancel in rank rather than pulling the centre as a mean would.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Visit rows for a single ``day_obs``, carrying the Trim ``dof0..dof49``.
    n_dof : `int`, optional
        Number of Trim degrees of freedom.

    Returns
    -------
    baseline : `numpy.ndarray`
        Shape ``(n_dof,)`` baseline Trim, in the OFC DOF units (µm for the
        hexapod dz terms).
    """
    return df[_dof_columns(n_dof)].median().to_numpy(dtype=float)


def classify_defocus(df, baseline=None, n_dof=50, dz_min_um=GIANT_DZ_MIN_UM,
                     quiet_tol_um=DOF_QUIET_TOL_UM):
    """Classify each exposure's defocal state from its Trim excursion.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Visit rows for a single ``day_obs``, carrying the Trim ``dof0..dof49``.
    baseline : `numpy.ndarray`, optional
        Baseline Trim in the DOF units; computed by `baseline_trim` if omitted.
    n_dof : `int`, optional
        Number of Trim degrees of freedom.
    dz_min_um : `float`, optional
        Minimum magnitude of the total dz excursion, in µm, for an exposure to
        be called giant.
    quiet_tol_um : `float`, optional
        Tolerance, in µm, on a non-dz DOF being unmoved from the baseline.

    Returns
    -------
    out : `pandas.DataFrame`
        Copy of `df` with the defocus columns appended: ``cam_dz_um`` and
        ``m2_dz_um`` (excursion from baseline, µm), ``total_dz_um`` (their sum),
        ``side`` (``'intra'``, ``'extra'`` or ``'focus'``), ``other_dof_max_um``
        (largest excursion among the 48 non-dz DOF) and ``dz_only`` (whether all
        of those stayed within `quiet_tol_um`).
    """
    cols = _dof_columns(n_dof)
    if baseline is None:
        baseline = baseline_trim(df, n_dof=n_dof)

    dev = df[cols].to_numpy(dtype=float) - baseline[None, :]
    cam_dz = dev[:, DOF_CAM_DZ]
    m2_dz = dev[:, DOF_M2_DZ]

    other = np.delete(dev, [DOF_M2_DZ, DOF_CAM_DZ], axis=1)
    other_max = np.nanmax(np.abs(other), axis=1)

    total_dz = cam_dz + m2_dz
    side = np.where(total_dz > dz_min_um, 'extra',
                    np.where(total_dz < -dz_min_um, 'intra', 'focus'))

    out = df.copy()
    out['cam_dz_um'] = cam_dz
    out['m2_dz_um'] = m2_dz
    out['total_dz_um'] = total_dz
    out['side'] = side
    out['other_dof_max_um'] = other_max
    out['dz_only'] = other_max <= quiet_tol_um
    return out


def select_giant_donuts(day_obs, require_dz_only=True, require_both_sides=True,
                        exp_time_min_sec=None, science_program=None,
                        dz_min_um=GIANT_DZ_MIN_UM, con=None, db_path=None):
    """Select a night's clean giant-donut exposures, label-independently.

    Parameters
    ----------
    day_obs : `int`
        Observing night, as ``YYYYMMDD``.
    require_dz_only : `bool`, optional
        Keep only exposures whose non-dz DOF are all at the night's baseline.
        Rejects Hartmann and bending-mode tests taken at the same defocus.
    require_both_sides : `bool`, optional
        Raise if the surviving sample lacks either side of focus.  The
        intra/extra Z11 split is undefined without both.
    exp_time_min_sec : `float`, optional
        Minimum exposure time, in seconds.  Giant-donut sequences use a longer
        exposure than the in-focus images bracketing them, so this separates an
        exposure that merely shares the defocal Trim state from one belonging to
        the giant sequence.
    science_program : `str`, optional
        Required ``science_program`` (the BLOCK label), applied after the Trim
        cut as a rejection filter only.
    dz_min_um : `float`, optional
        Minimum magnitude of the total dz excursion, in µm.
    con, db_path : optional
        Passed to `efd_db.visits`.

    Returns
    -------
    sel : `pandas.DataFrame`
        The selected exposures with the defocus columns of `classify_defocus`
        and the joined ConsDB metadata, sorted by ``seq_num``.

    Raises
    ------
    ValueError
        If `require_both_sides` and only one side of focus survives.
    """
    df = efd_db.visits(day_obs_range=(day_obs, day_obs), con=con, db_path=db_path)
    if df.empty:
        raise ValueError(f'no visits in the value-added database for day_obs {day_obs}')

    cls = classify_defocus(df, dz_min_um=dz_min_um)
    sel = cls[cls.side != 'focus'].copy()

    if require_dz_only:
        sel = sel[sel.dz_only]

    # ConsDB carries band, rotator angle, exposure time and the program label;
    # the rotator angle is needed for the spider model, so join before filtering.
    sel = efd_db.join_consdb(sel)

    if exp_time_min_sec is not None and 'exp_time_sec' in sel.columns:
        sel = sel[sel.exp_time_sec >= exp_time_min_sec]
    if science_program is not None and 'science_program' in sel.columns:
        sel = sel[sel.science_program == science_program]

    sel = sel.sort_values('seq_num').reset_index(drop=True)

    if require_both_sides:
        sides = set(sel.side.unique())
        if not {'intra', 'extra'} <= sides:
            raise ValueError(
                f'day_obs {day_obs} has only {sorted(sides)} after selection; '
                'the intra/extra Z11 split needs both sides of focus'
            )
    return sel
