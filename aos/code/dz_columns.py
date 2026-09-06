"""Column-name helpers for the Double Zernike (DZ) fit tables.

The per-visit DZ fit tables name each coefficient ``<prefix>_z<j>_c<k>``, for pupil
Noll index ``j`` and focal-plane order index ``k`` — for example ``dz_z5_c1``. Several
analyses need the list of those columns for a given prefix, so the pattern lives here
rather than being re-expressed per script.
"""

import re


def dz_coeff_columns(df, prefix):
    """Return the DZ coefficient column names in `df` carrying `prefix`, in table order.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A DZ fit table.
    prefix : `str`
        Coefficient-family prefix, e.g. ``'dz'`` or ``'dz_resid'``. Regex-escaped, so
        it is treated literally.

    Returns
    -------
    columns : `list` [`str`]
        Matching column names, in the order they appear in `df.columns`. Empty if none
        match — callers that require coefficients should check for that themselves.
    """
    pat = re.compile(rf'^{re.escape(prefix)}_z\d+_c\d+$')
    return [c for c in df.columns if pat.match(c)]
