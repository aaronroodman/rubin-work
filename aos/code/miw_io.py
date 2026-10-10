"""Compatibility shim: ``miw_io`` became the ``miw`` product's reader.

Phase 3a of the products/libraries/studies reorganization moved this module to
``rubinwork/products/miw/reader.py``, re-exported from ``rubinwork.products.miw``. It is a
pure reader of the Measured Intrinsic Wavefront (MIW) field maps the ``intrinsic_split``
step writes, so it belongs to the product that writes them.

The shim binds the real module's namespace here and registers the real module under this
path's name, so the bare-name ``from miw_io import load_miw`` in the four
``aos/code/static_optics/`` scripts and the ``JS_DEFAULT`` import in the four
``aos/code/miw/`` comparison scripts give the same objects, not copies.

``load_miw`` is kept as an alias of the reader's ``load_maps``, which is the name that says
what it reads. New code imports ``rubinwork.products.miw`` and resolves the parquet path
through ``rubinwork.products.catalog`` rather than building one. The shim is removed in
phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.miw import reader as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
