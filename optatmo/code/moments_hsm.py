"""Compatibility shim: ``moments_hsm`` became the ``rubinwork.hsm_moments`` library.

Phase 3a of the products/libraries/studies reorganization moved this module to
``rubinwork/hsm_moments.py``. It is a **library** by the plan's section 2: three scripts in
two topics import it and it writes no data of its own.

The shim binds the real module's namespace here and registers the real module under this
path's name, so a bare-name ``from moments_hsm import measure_hsm_moments`` after a
``sys.path`` insert of ``optatmo/code`` gives the same objects, not copies.

Why it moved: the guider side reached it through
``sys.path.insert(__file__/../../optatmo/code)``, a relative hop between topics that no
static import check can see and that breaks silently when either file moves. The
matched-estimator property is the point of the coupling — the guider and the adjacent-CCD
PSF stars must use the *same* galsim-HSM estimator or the comparison is confounded — and a
library import makes that explicit rather than fragile.

New code imports ``rubinwork.hsm_moments``. The shim is removed in phase 5
(``notes/status/organization_plan.md``).
"""
import sys

from rubinwork import hsm_moments as _real

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})
__all__ = getattr(_real, "__all__", [k for k in vars(_real) if not k.startswith("_")])
sys.modules[__name__] = _real
