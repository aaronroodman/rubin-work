---
name: rubin-docstrings
description: Writing or updating Python docstrings or module headers in rubin-work (Rubin DM numpydoc standard, units and frames).
---

# Docstring standard for rubin-work

The Writing rules in the root `CLAUDE.md` (plain, short, direct) apply here too.

## Docstrings — Rubin DM / numpydoc
Follow the [LSST DM docstring standard](https://developer.lsst.io/python/numpydoc.html):
numpydoc sections, with **types in backticks**. Section order is short summary,
extended summary, `Parameters`, `Returns`/`Yields`, `Raises`, `See Also`, `Notes`,
`Examples` — include only the ones that apply.

```python
def nmad(x, min_n=3):
    """Normalized median absolute deviation — a robust sigma estimate.

    Extended description if the one-liner is not enough.

    Parameters
    ----------
    x : `array_like`
        Values in any single unit; the result carries that same unit.
    min_n : `int`, optional
        Return NaN if fewer than this many finite values remain.

    Returns
    -------
    sigma : `float`
        Robust scatter in the units of `x`, or NaN if under-determined.
    """
```

Points to get right in this repo:

- **Every module opens with a docstring** giving a short description of what the code
  does. For a runnable script, include the invocation and its key arguments.
- **Units belong in the docstring**, on every physical parameter and return value — µm
  of wavefront, deg, arcsec, or an explicit "dimensionless" with the ratio named. This
  is the same rule as the Reporting numbers rule in the root `CLAUDE.md`; a docstring is where a
  reader looks first.
- Say which **frame** an angle or Zernike is in (OCS/CCS) when it matters.
- Note real **failure modes** in `Notes` rather than leaving them implicit — e.g.
  `common/utils.alt_to_deg` documents that degree values below 6.28 are misread as
  radians.
- Existing code is inconsistent (numpydoc appears in only a handful of files, and there
  is no Google-style `Args:` anywhere). Bring a file up to this standard when working in
  it; do not launch a repo-wide reformat as a side quest.
