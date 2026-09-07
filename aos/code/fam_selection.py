"""Selection of Full Array Mode (FAM) visits for downstream analysis, and the
column-name helper for the Double Zernike (DZ) fit tables.

`fam_quality_selection` is the single place that decides which FAM visits are good
enough to use. Call it rather than re-deriving a cut, so that every analysis of a given
`fits.parquet` works from the same sample.
"""

import re


def dz_coeff_columns(df, prefix):
    """Return the DZ coefficient column names in `df` carrying `prefix`, in table order.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A DZ fit table.
    prefix : `str`
        Coefficient-family prefix, e.g. ``'dz'`` or ``'z1toz6'``. Regex-escaped, so it
        is treated literally.

    Returns
    -------
    columns : `list` [`str`]
        Matching column names, in the order they appear in `df.columns`. Empty if none
        match — callers that require coefficients should check for that themselves.
    """
    pat = re.compile(rf'^{re.escape(prefix)}_z\d+_c\d+$')
    return [c for c in df.columns if pat.match(c)]


def fam_quality_selection(df, prefix='z1toz6', max_coeff_um=None,
                          max_blur_arcsec=None, verbose=True):
    """Select the FAM visits whose wavefront fits are good enough to analyse.

    Applies up to three cuts, in this order:

    1. **Fit failure** — always applied. Any visit with a true value in *any*
       ``*bad_fit`` column is dropped. These are usually visits with too few donuts to
       constrain the k=1..6 focal-plane terms.
    2. **Maximum DZ coefficient** — optional. Drops a visit if any of its DZ
       coefficients exceeds `max_coeff_um` in absolute value.
    3. **Maximum median donut blur** — optional. Drops a visit whose
       ``median_blur_arcsec`` exceeds `max_blur_arcsec`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A per-visit DZ fit table, e.g. from ``output/<ps>/<mi>/fits.parquet``.
    prefix : `str`, optional
        DZ coefficient-family prefix, used both to find the coefficient columns and to
        prefer a ``<prefix>_bad_fit`` column. Default ``'z1toz6'``.
    max_coeff_um : `float` or `None`, optional
        Maximum absolute DZ coefficient, in µm of wavefront. `None` (default) skips
        this cut.
    max_blur_arcsec : `float` or `None`, optional
        Maximum per-visit median donut blur, in arcsec. `None` (default) skips this
        cut.
    verbose : `bool`, optional
        Print a one-line count per cut applied.

    Returns
    -------
    selected : `pandas.DataFrame`
        A copy of `df` holding only the visits that pass. Index values are preserved,
        so the caller can align against the original table.

    Notes
    -----
    The two optional cuts default to off so that adding a cut is always an explicit
    choice, visible at the call site.

    Every ``*bad_fit`` column is combined with a logical OR rather than trusting one.
    In the tables checked on 2026-09-07 (`pathA_50_34_i_5rot`, 1126 visits)
    ``bad_fit``, ``z1toz6_bad_fit`` and ``z1toz3_bad_fit`` flagged the same 25 visits,
    but nothing guarantees that, and a visit failing any fit variant is not usable.

    A missing ``median_blur_arcsec`` column with `max_blur_arcsec` set raises
    `KeyError`, rather than silently skipping the requested cut.
    """
    n0 = len(df)
    out = df

    bad_cols = [c for c in out.columns if c.endswith('bad_fit')]
    if bad_cols:
        # Prefer the prefix-specific flag first for reporting, but drop on the union.
        keep = ~out[bad_cols].astype(bool).any(axis=1)
        n_bad = int((~keep).sum())
        out = out[keep].copy()
        if verbose and n_bad:
            print(f'  fam selection: dropped {n_bad} visit(s) flagged '
                  f'{"/".join(bad_cols)}')
    elif verbose:
        print('  fam selection: no *bad_fit column found — no fit-failure cut applied')

    if max_coeff_um:
        cols = dz_coeff_columns(out, prefix)
        if cols:
            keep = ~out[cols].abs().gt(max_coeff_um).any(axis=1)
            n_cut = int((~keep).sum())
            out = out[keep].copy()
            if verbose:
                print(f'  fam selection: dropped {n_cut} visit(s) with any '
                      f'|{prefix} DZ coefficient| > {max_coeff_um} um')
        elif verbose:
            print(f'  fam selection: no {prefix} DZ coefficient columns — '
                  'no coefficient cut applied')

    if max_blur_arcsec:
        keep = ~(out['median_blur_arcsec'] > max_blur_arcsec)
        n_cut = int((~keep).sum())
        out = out[keep].copy()
        if verbose:
            print(f'  fam selection: dropped {n_cut} visit(s) with '
                  f'median_blur_arcsec > {max_blur_arcsec} arcsec')

    if verbose:
        print(f'  fam selection: {len(out)}/{n0} visits selected')
    return out
