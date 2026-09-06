---
name: docstring-standard-dm
description: rubin-work adopts the Rubin DM numpydoc docstring standard — types in backticks, units and frame on every physical quantity
metadata:
  type: feedback
---

Python docstrings in `rubin-work` follow the **Rubin DM standard**
(https://developer.lsst.io/python/numpydoc.html): numpydoc sections with **types in
backticks**, and named `Returns` entries.

**Why:** Aaron noted (2026-09-06) that the codebase has *no* standard header format —
he wants a short description of the code's function plus arguments with types, close to
the Rubin DM convention. Surveyed at the time: numpydoc `Parameters` blocks appeared in
only 3 of 58 files in `aos/code/`, and Google-style `Args:` in zero. So DM/numpydoc is
formalizing what little convention exists rather than importing a foreign one.

Section order: short summary, extended summary, `Parameters`, `Returns`/`Yields`,
`Raises`, `See Also`, `Notes`, `Examples` — only those that apply.

```python
def nmad(x, min_n=3):
    """Normalized median absolute deviation — a robust sigma estimate.

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

**How to apply:**
- Every module gets a docstring; runnable scripts include the invocation and key args.
- **Units on every physical parameter and return** — µm of wavefront, deg, arcsec, or an
  explicit "dimensionless" with the ratio named. Same rule as
  [[reporting-units-standard]]; the docstring is where a reader looks first.
- State the **frame** (OCS/CCS) for angles and Zernikes where it matters
  ([[frame-conventions-ccs-ocs]]).
- Put real failure modes in `Notes` rather than leaving them implicit — e.g.
  `common/utils.alt_to_deg` documents that degree values below 6.28 are misread as
  radians.
- **Bring a file up to standard when working in it.** Do not launch a repo-wide
  reformat as a side quest.

Aaron is doing a file-by-file review of `aos/code/` for both `common/` candidacy and
readability, with this standard as part of it. The open items are tracked in
`aos/docs/status/code_review_backlog.md`. Also in the root `CLAUDE.md` under
"Docstrings — Rubin DM / numpydoc", so it loads every session.
