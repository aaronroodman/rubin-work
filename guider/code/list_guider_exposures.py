"""Compatibility shim: ``list_guider_exposures`` moved into the ``guider_moments`` product.

Phase 3a of the products/libraries/studies reorganization moved this builder to
``rubinwork/products/guider_moments/builders/list_guider_exposures.py``. The shim binds the
real module's namespace here and registers the real module under this path's name, so a
bare-name import after a ``sys.path`` insert of ``guider/code`` gives the same objects,
not copies, and ``python guider/code/list_guider_exposures.py ...`` still runs.

Run it as ``python -m rubinwork.products.guider_moments.builders.list_guider_exposures``.
The shim is removed in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.guider_moments.builders import list_guider_exposures as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
