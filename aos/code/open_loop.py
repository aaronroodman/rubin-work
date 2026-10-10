"""Compatibility shim: ``open_loop`` is now ``rubinwork.open_loop``.

See the sibling ``aos_state.py`` shim for why this is a module at the old path
rather than a package redirect. New code should import ``rubinwork.open_loop``.
The shim is removed in phase 5 (``notes/status/organization_plan.md``).
"""
import sys

from rubinwork import open_loop as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
