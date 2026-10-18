"""Open Loop Reproduction (OLR) core — **superseded, retained as a guard**.

The implementation moved to **``aos/code/open_loop.py``** (item 2), which is where the
v-modes, DOF sets and per-corner recovery already live. The per-visit open-loop state for
every science and acquisition visit is now built into the value-added database by
``value_added/code/build_optical_state.py`` and read back as the ``_olr`` columns of
``optical_state``; this topic is being repurposed for *analysis* of that stored OLR.

The old functions are **not** re-exported, because the replacement is not a drop-in: the
sign was wrong here. ``apply_trim(..., subtract=False)`` *added* the Trim wavefront back,
whereas the open-loop reconstruction *subtracts* it — the Trim is applied with the opposite
sign in order to drive the measured deviation toward zero, so ``Trim - Deviation`` is the
visit's optical state and the open-loop state is its negative, ``Deviation - Trim``. A
caller silently picking up the old behaviour would get sign-flipped Zernikes, which is why
this module raises instead of forwarding.

The archived original is ``scratch/archive/item2/olr_superseded.py``, kept only so
pre-item-2 numbers can be reproduced; it lists the other three defects that were fixed in
the move (a hand-written duplicate of ``DOF_SETS['standard_22']``, a bare ``OFCData`` on the
obsolete normalization, and field angles labelled CCS at rotator zero where OCS is
required).

Note that ``run_olr.py``'s identity check ``olr_deviation == olr_opd - intrinsic`` never
caught the sign error: the intrinsic cancels from both sides, so the identity holds either
way. It is carried into the replacement as a basis and corner-ordering check only.
"""

#: Replaced names -> what to call instead. ``run_olr.py`` and the ``olr`` Snakemake rule
#: still import these, so the message has to name the replacement rather than let the
#: failure surface as a bare ImportError that looks like a missing file.
_MOVED = {
    'build_olr_sensitivity_matrix': 'open_loop.olr_sensitivity_matrix(state_estimator)',
    'apply_trim':
        'open_loop.olr_zernikes(zk_deviation, dof_trim, sens_mat, state_estimator), '
        'which SUBTRACTS the Trim wavefront where apply_trim(subtract=False) added it',
    'extract_olr':
        'value_added/code/build_optical_state.py, which writes the per-visit open-loop '
        'state to the optical_state table; or efd_db.optical_state(variant) to read it back',
    'DEFAULT_DOF_INDICES':
        "aos_state.DOF_SETS['standard_22'] -- this constant was a hand-written duplicate "
        'of it, and the replacement reads the active DOF off the state estimator instead',
    'DEFAULT_TRUNCATION': 'the n_modes argument of aos_state.make_state_estimator',
    'DEFAULT_ZN_SELECTED': 'aos_state.ZK_NOLL',
}

# Still true of the instrument rather than of the superseded implementation, so these stay.
SENSOR_IDS = [191, 195, 199, 203]
CORNER_DETNAMES = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]
CORNER_NAMES = ["R00", "R04", "R40", "R44"]


def _moved_message(name):
    """The replacement message for a superseded name."""
    return (
        f'olr.{name} was superseded by item 2 and deliberately not re-exported: the OLR '
        f'sign was wrong here (the Trim wavefront was added where it must be subtracted), '
        f'so forwarding would hand back sign-flipped numbers. Use {_MOVED[name]}. The '
        f'archived original is scratch/archive/item2/olr_superseded.py.')


def _make_guard(name):
    """Build a stub that imports cleanly but raises with the replacement when called.

    Notes
    -----
    A module-level ``__getattr__`` (PEP 562) is the usual idiom here, and is what
    ``aos/code/aos_state.py`` uses for its retired SVD helpers. It is **not** enough on its
    own: for ``from olr import apply_trim`` CPython converts the `AttributeError` into an
    `ImportError` and **discards the message**, so the caller sees only "cannot import
    name", with no pointer to the replacement. Measured on this interpreter, including
    against the existing `aos_state` guard, whose docstring claims otherwise.

    Binding real callables instead means the import succeeds and the named error arrives at
    the call site, where it can still be acted on. ``__getattr__`` is kept below for any
    name not bound here.
    """
    def _guard(*_args, **_kwargs):
        raise NotImplementedError(_moved_message(name))
    _guard.__name__ = name
    _guard.__doc__ = f'Superseded. {_moved_message(name)}'
    return _guard


# Bound as callables so the import succeeds and the message reaches the call site; see
# _make_guard. The three non-callable constants raise on use through __getattr__ instead,
# since a stub function would be a type error rather than a clear one.
build_olr_sensitivity_matrix = _make_guard('build_olr_sensitivity_matrix')
apply_trim = _make_guard('apply_trim')
extract_olr = _make_guard('extract_olr')


def __getattr__(name):
    """Raise a named error for a moved constant; normal AttributeError otherwise.

    Parameters
    ----------
    name : `str`
        Attribute requested from this module.

    Raises
    ------
    AttributeError
        Always. For a name in `_MOVED` the message gives the replacement. Note that for
        ``from olr import DEFAULT_DOF_INDICES`` CPython rewrites this into an `ImportError`
        and drops the message — which is why the three *functions* above are bound as real
        callables rather than left to this hook.
    """
    if name in _MOVED:
        raise AttributeError(_moved_message(name))
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
