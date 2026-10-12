"""Compatibility shim: ``guiderMoments`` moved into the ``guider_moments`` product.

Phase 3a of the products/libraries/studies reorganization moved this module to
``rubinwork/products/guider_moments/guiderMoments.py``. It carries ``SCHEMA_VERSION``,
which the plan's product table names as the product's variant (``v6`` is "the current
``guiderMoments`` schema"), so the schema and the code that writes it stay together inside
the product rather than becoming a separate library.

The shim binds the real module's namespace here and registers the real module under this
path's name, so the bare-name ``from guiderMoments import measureStampMoments,
MomentConfig, _FWHM_PER_SIGMA`` in the atmosphere study (``guider_atmo_compare.py``) gives
the same objects, not copies.

New code imports ``rubinwork.products.guider_moments.guiderMoments``. The shim is removed
in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.products.guider_moments import guiderMoments as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
