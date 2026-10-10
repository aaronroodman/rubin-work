"""Compatibility shim: ``aos_state`` is now ``rubinwork.aos_state``.

Phase 1 of the products/libraries/studies reorganization moved this library into
the installable ``rubinwork`` package. Callers import it by bare module name after
a ``sys.path`` insert of ``aos/code``, so the shim has to be a module at the old
path. It binds the real module's namespace here and registers the real module
under the bare name, so ``import aos_state`` and
``from aos_state import StateEstimator`` both give the same objects as
``rubinwork.aos_state`` -- not copies.

New code should import ``rubinwork.aos_state`` directly. The shim is removed in
phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork import aos_state as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
