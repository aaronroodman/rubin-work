---
study: aos_openloop
topic: nightlyiq
status: dropped
question: "Can the AOS closed-loop PID be replayed from the corner Zernikes, as a check before open-loop reconstruction?"
answer: "Dropped: superseded by rubinwork.open_loop."
uses: {}  # none
started: 2026-03-11
---

# `aos_openloop`: replaying the closed-loop PID

> **Notebook:** `notebooks/aos_openloop.ipynb`

## Question

Can the AOS closed-loop PID be replayed from the corner Zernikes, as a check before open-loop reconstruction?

## Current answer

Dropped: superseded by rubinwork.open_loop.

## Log

### 2026-07-27 — backfilled from docs and history

Backfilled 2026-10-18 from `nightlyiq/README.md` (commits `9de682e`, `d2e395a`). Built the sensitivity-matrix SVD, projected the 4-corner Zernikes onto v-modes, ran a per-v-mode PID and compared with the applied DOF corrections. Dropped in the phase 5 triage: `rubinwork.open_loop` and the value-added `optical_state` columns replace it.
