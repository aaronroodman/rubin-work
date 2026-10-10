"""Compatibility shim: ``blitz_reader`` moved into the ``fam_tables`` product.

Phase 2 of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/fam_tables/builders/blitz_reader.py``. The shim binds the
real module's namespace here and registers the real module under this path's
name, so a bare-name ``import blitz_reader`` after a ``sys.path`` insert of
``aos/code/fam_processing`` gives the same objects, not copies.

New code imports
``rubinwork.products.fam_tables.builders.blitz_reader``. Run it as
``python -m rubinwork.products.fam_tables.builders.blitz_reader``. The shim is
removed in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.fam_tables.builders import blitz_reader as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

