"""Compatibility shim: ``build_efd_db`` moved into the ``value_added`` product.

Phase 3a of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/value_added/builders/build_efd_db.py``. The value-added telemetry database it
writes is read by several topics, so it is a product, not a study.

The shim binds the real module's namespace here and registers the real module under this
path's name, so a bare-name ``import build_efd_db`` after a ``sys.path`` insert of
``value_added/code`` gives the **same objects, not copies**.

New code imports ``rubinwork.products.value_added.builders.build_efd_db``. Run it as
``python -m rubinwork.products.value_added.builders.build_efd_db``. The shim is removed in
phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.value_added.builders import build_efd_db as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
