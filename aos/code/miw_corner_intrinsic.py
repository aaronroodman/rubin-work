"""Compatibility shim: ``miw_corner_intrinsic`` became the ``rubinwork.miw_corner`` library.

Phase 3a of the products/libraries/studies reorganization moved this module to
``rubinwork/miw_corner.py``. It is a **library**: it evaluates a Measured Intrinsic
Wavefront (MIW) decomposition it is handed at the four corner-wavefront-sensor field
points, writing no data of its own. The MIW itself is the ``miw`` product.

The shim binds the real module's namespace here and registers the real module under this
path's name, so the bare-name ``from miw_corner_intrinsic import MiwCornerLookup`` in
``value_added/code/build_optical_state.py`` gives the same objects, not copies.

One behaviour change came with the move, and it is a fix rather than a rename:
``decomp_path`` built ``output/<param_set>/<mi_name>/``, a layout the tree has not had
since the ``dir_name`` change, so every default lookup missed. It now builds
``output/miw/<ps dir_name>_<mi dir_name>/``, which resolves. `MiwCornerLookup` also takes
``path=`` now, which is the preferred route: resolve a ``miw`` product build through
`rubinwork.products.catalog.path` instead of rebuilding an old-tree path.

New code imports ``rubinwork.miw_corner``. The shim is removed in phase 5
(``notes/status/organization_plan.md``).
"""
import sys

from rubinwork import miw_corner as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
