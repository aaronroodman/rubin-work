---
name: ask-before-slac
description: Ask before connecting to SLAC/USDF computers; prefer giving runnable snippets
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cc746a2f-9746-41cd-90db-f7d33ae6f183
---

Explicitly ask the user before connecting to the SLAC computers (ssh `slacrd`, USDF, `/repo/main` Butler, etc.). Do not open an ssh/USDF connection without first getting the user's OK.

**Why:** The user asked for this control "at least for the time being" (2026-07-10).

**How to apply:** When USDF work is needed, either ask first, or hand the user a runnable snippet to execute themselves (the user's established preference — see [[usdf-access-slacrd]]). Local work in `rubin-work/` needs no permission.
