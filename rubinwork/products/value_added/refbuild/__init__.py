"""Reference-build helpers for checking a code change against a frozen build.

Not part of a build. Used by the phase 3 checks of
``notes/status/organization_plan.md``: rerun a builder against a **scratch** copy of a
reference database and compare at zero tolerance, with the build-time timestamp columns
on an expected-difference list. The live database is never written.
"""
