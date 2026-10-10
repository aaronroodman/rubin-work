"""Compatibility shim: ``common`` is now ``rubinwork.common``.

Phase 1 of the products/libraries/studies reorganization moved these modules
into the installable ``rubinwork`` package. This shim keeps every existing
``from common.X import Y`` working until phase 5 rewrites the imports. It
registers ``rubinwork.common`` and each of its submodules under the old
``common.*`` names, so ``import common.utils``, ``from common import utils``
and ``from common.utils import nmad`` all resolve to the same module objects as
the new names -- not to copies.

New code should import ``rubinwork.common`` directly.

Note: in ``aos/code``, ``common`` in an import usually means
``lsst.ts.intrinsic.wavefront.common``, an external package. That is a
different, fully qualified name and is unaffected by this shim.
"""

import importlib
import sys

_SUBMODULES = [
    "FocalPlaneInterpolator",
    "consdb_efd",
    "dof_telemetry",
    "ess_telemetry",
    "psf_moments_consdb",
    "psf_render",
    "telemetry_clients",
    "utils",
    "visit_telemetry",
]

_pkg = importlib.import_module("rubinwork.common")

# Make this module a view of the real package: the same __path__ lets
# `import common.<anything>` find submodules not listed above.
__path__ = list(_pkg.__path__)

for _name in _SUBMODULES:
    try:
        _mod = importlib.import_module(f"rubinwork.common.{_name}")
    except ImportError:
        # An optional dependency is missing; let `import common.<name>` raise
        # the same error it would have raised before the move.
        continue
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

del _pkg
