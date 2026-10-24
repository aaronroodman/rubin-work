"""Compatibility shim: ``efd_db`` is now the ``value_added`` product's reader.

Phase 3a of the products/libraries/studies reorganization moved this module to
``rubinwork/products/value_added/reader.py``. It reads the value-added telemetry
database, which several topics consume, so it is a product reader.

The shim binds the real module's namespace here and registers the real module under this
path's name, so a bare-name ``import efd_db`` after a ``sys.path`` insert of
``value_added/code`` gives the **same objects, not copies**. That matters more here than
for any other shim in the reorganization: two scripts and three notebooks import it this
way, and two of them pass the connection it opens to each other, which copies would
break.

Callers kept working by this shim:

* ``aos/code/cwfs_lut/cwfs_lut_lib.py``
* ``aos/code/cwfs_lut/run_cwfs_lut.py``
* ``aos/notebooks/bounce/bending_mode_test_lut_trim.ipynb``
* ``aos/notebooks/bounce/_exectest.ipynb``
* ``aos/notebooks/fam_processing/blur_intra_extra_offset.ipynb``

New code imports ``rubinwork.products.value_added`` and calls ``visits``,
``optical_state``, ``fam_dz``, ``m1m3_thermal_r2`` or ``join_consdb`` on it, or opens a
catalog build with ``connect()``. The shim is removed in phase 5
(``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.value_added import reader as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
