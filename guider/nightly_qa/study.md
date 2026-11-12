---
study: nightly_qa
topic: guider
status: active
question: 'What does each night''s guider data look like: the per-day QA PDF, the movies and the RubinTV-style review plots?'
answer: ''
uses: {}
started: 2026-11-12
---

# `nightly_qa`

## Question

What does each night's guider data look like: the per-day QA PDF, the movies and the RubinTV-style review plots?

## Current answer

None yet.

## Log

### 2026-11-12 — owner of the guider Snakefile's QA rules

Created to own the per-night QA that `guider/Snakefile` writes: `plots` (the per-day QA
PDF, from the `guider_moments` product), `movie`/`movies` and `rtvplot`/`rtvplots`
(RubinTV-style movies and review plots for every `movie_stride`-th visit). Those rules and
`run_snake.sh` now write to `output/night_<dayObs>/` here (`QA_ROOT`). The existing files
for nights 20260512–20260712 (47 files, 306 MB) were copied from the old
`guider/output/night_*` tree.
