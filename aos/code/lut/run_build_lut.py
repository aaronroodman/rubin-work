"""Compatibility shim: ``run_build_lut`` moved into the ``dof_lut`` product.

Phase 3a of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/dof_lut/builders/run_build_lut.py``. The averaged
degree-of-freedom look-up table it writes is read by work outside this directory, so it
is a product, not a study.

The shim binds the real module's namespace here and registers the real module under this
path's name, so a bare-name ``import run_build_lut`` after a ``sys.path`` insert of
``aos/code/lut`` gives the same objects, not copies.

New code imports ``rubinwork.products.dof_lut.builders.run_build_lut``, or reads a build
through ``rubinwork.products.dof_lut.load``. Run it as
``python -m rubinwork.products.dof_lut.builders.run_build_lut``. The shim is removed in
phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.dof_lut.builders import run_build_lut as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

if __name__ == "__main__":
    raise SystemExit(_real.main())
