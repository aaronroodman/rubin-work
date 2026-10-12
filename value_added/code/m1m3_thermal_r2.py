"""Compatibility shim: ``m1m3_thermal_r2`` moved into the ``value_added`` product.

Phase 3a of the products/libraries/studies reorganization moved this computation module to
``rubinwork/products/value_added/builders/m1m3_thermal_r2.py``. The value-added telemetry gradients it
computes feed a table that is read by several topics, so it is a product, not a study.

The shim binds the real module's namespace here and registers the real module under this
path's name, so a bare-name ``import m1m3_thermal_r2`` after a ``sys.path`` insert of
``value_added/code`` gives the **same objects, not copies**.

It has no ``main``: it is the pure computation behind
``build_m1m3_thermal_r2``, its only importer, which is why it stayed inside the product
rather than being promoted to a library (a library is code *other work* imports).

New code imports ``rubinwork.products.value_added.builders.m1m3_thermal_r2``. The shim is
removed in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.value_added.builders import m1m3_thermal_r2 as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
