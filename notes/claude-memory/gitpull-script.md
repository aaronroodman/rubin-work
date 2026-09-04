---
name: gitpull-script
description: Aaron syncs repos with his own gitpull script, not raw `git pull`
metadata:
  type: feedback
---

Aaron uses a personal **gitpull script** to update repos; he does not want to be
handed explicit `git pull` commands when a sync is needed.

**Why:** it's his standard workflow (wraps the pull, likely with his path/mount
conventions — see [[usdf-mount-paths]]).

**How to apply:** when telling him to update code (e.g. on USDF/slaciana before a
run), say "run your gitpull script" instead of `git pull`. My own commits/pushes
to the repo still use git directly.
