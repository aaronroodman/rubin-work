"""Compatibility shim: ``run_blitz_mktable`` moved into the ``fam_tables`` product.

Phase 2 of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/fam_tables/builders/run_blitz_mktable.py``. The shim binds the
real module's namespace here and registers the real module under this path's
name, so a bare-name ``import run_blitz_mktable`` after a ``sys.path`` insert of
``aos/code/fam_processing`` gives the same objects, not copies.

New code imports
``rubinwork.products.fam_tables.builders.run_blitz_mktable``. Run it as
``python -m rubinwork.products.fam_tables.builders.run_blitz_mktable``. The shim is
removed in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.fam_tables.builders import run_blitz_mktable as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
