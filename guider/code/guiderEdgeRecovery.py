"""Compatibility shim: ``guiderEdgeRecovery`` became ``rubinwork.guider_edge_recovery``.

Phase 3a of the products/libraries/studies reorganization moved this module to
``rubinwork/guider_edge_recovery.py``. It is a **library** by the plan's section 2:
`makeTrackerConfig` is imported by the ``guider_moments`` product builder *and* by two
study runners (``run_guider_movie.py``, ``run_guider_rubintv_plots.py``), and it writes no
data of its own.

The shim binds the real module's namespace here and registers the real module under this
path's name, so the bare-name ``from guiderEdgeRecovery import makeTrackerConfig`` in the
two study runners gives the same objects, not copies.

New code imports ``rubinwork.guider_edge_recovery``. The shim is removed in phase 5
(``notes/status/organization_plan.md``).
"""
import sys

from rubinwork import guider_edge_recovery as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
