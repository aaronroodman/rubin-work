"""Compatibility shim: ``build_m1m3_thermal_r2`` moved into the ``value_added`` product.

Phase 3a of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/value_added/builders/build_m1m3_thermal_r2.py``. The value-added telemetry database it
writes is read by several topics, so it is a product, not a study.

The shim binds the real module's namespace here and registers the real module under this
path's name, so a bare-name ``import build_m1m3_thermal_r2`` after a ``sys.path`` insert of
``value_added/code`` gives the **same objects, not copies**.

New code imports ``rubinwork.products.value_added.builders.build_m1m3_thermal_r2``. Run it as
``python -m rubinwork.products.value_added.builders.build_m1m3_thermal_r2``. The shim is removed in
phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.value_added.builders import build_m1m3_thermal_r2 as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
