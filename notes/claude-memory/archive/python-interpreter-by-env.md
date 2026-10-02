---
name: python-interpreter-by-env
description: Which Python to invoke — MacPorts on the laptop, bare `python` on the RSP/USDF
metadata:
  type: feedback
---

On the **laptop** use MacPorts `/opt/local/bin/python3` (per CLAUDE.md). On the
**RSP / USDF** just use `python` — the LSST stack interpreter is on PATH there,
and the MacPorts binary does not exist on those machines.

**Why:** CLAUDE.md says "always use MacPorts Python", but that rule is
laptop-scoped; applying it to code that runs on the RSP would fail.

**How to apply:** For commands/scripts intended to run on USDF (extraction,
Butler, fits — see [[usdf-access-slacrd]], [[usdf-mount-paths]]) write `python`.
Reserve `/opt/local/bin/python3` for local laptop analysis.
