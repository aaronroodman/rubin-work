"""Studies: find, parse and check every ``<topic>/<study>/study.md``.

A study is a directory ``<topic>/<study>/`` whose ``study.md`` opens with YAML front
matter (``study``, ``topic``, ``status``, ``question``, ``answer``, ``uses``,
``started``) followed by the sections Question, Current answer and Log. This package
reads those files for the ``/start-study``, ``/wrap`` and ``/review`` skills and writes
``STUDIES.md``::

    python -m rubinwork.studies index

It is tooling over the repository and the product catalog, not a library: nothing in a
study or a product imports it.
"""

from .core import (  # noqa: F401
    STATUSES, REQUIRED_KEYS, Study, Problem, UsesEntry, repo_root, find_studies,
    load_study, parse_front_matter, parse_uses, format_uses, validate,
)
