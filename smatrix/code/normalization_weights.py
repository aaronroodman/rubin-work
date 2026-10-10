"""Compatibility shim: ``normalization_weights`` is now ``rubinwork.smatrix.normalization_weights``.

Phase 1 of the products/libraries/studies reorganization moved this library into
the installable ``rubinwork`` package. Callers import it by bare module name after
a ``sys.path`` insert of ``smatrix/code``, so the shim has to be a module at the
old path; it registers the real module under the bare name, so the objects are
shared rather than copied.

New code should import ``rubinwork.smatrix.normalization_weights`` directly. The shim is removed in
phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork.smatrix import normalization_weights as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
