# Claude working-memory snapshot

A **copy** of Claude's private working memory from
`~/.claude/projects/-Users-roodman/memory/`, snapshotted **2026-09-04** so it is
versioned and shareable. Not the live files — the originals stay in `~/.claude/` and
will drift from these; re-copy to refresh.

## What these are

Notes-to-self, not documentation. They are written in Claude's own shorthand, are
deliberately terse, and record things like "mistakes I made and must not repeat".
Useful for continuity and for auditing how a conclusion was reached; not intended as
a primary reference.

**For handing this work to another person or another assistant, use
[`../../aos/MIW_INVESTIGATION_HANDOFF.md`](../../aos/MIW_INVESTIGATION_HANDOFF.md)
instead.** That document covers the same ground written for an outside reader, with
full context, units on every quantity, and an explicit list of retracted claims.

## Contents

| file | what it holds |
|---|---|
| `MEMORY.md` | the index loaded at session start — one line per memory, covering AOS, guider, camera and frame-convention topics |
| `reporting-units-standard.md` | standing requirement: every number carries its quantity name and units (or "dimensionless" with the ratio named) |
| `miw-focal-order-truncation.md` | 83% of MIW power sits above the k<=6 focal orders the build fits — reframes every DZ-subspace analysis |
| `miw-coadd-retrieval-bias.md` | the coadd-vs-MIW investigation: the retrieval-bias hypothesis, the L estimator (L == 0 iff unbiased), the Sigma_emp error model, and corrections made along the way |

## Known gaps in this snapshot

`[[double-bracket]]` cross-references point to other memory files. Four of the seven
referenced here are **not** in this snapshot, so those links dangle:

- `miw-products-and-m3-backprojection` — which MIW file to use, the back-projection
  and joint static-fit results
- `sparse-fit-sensitivity` — the sparse donut-fit study, Piece 1 and observability
- `frame-conventions-ccs-ocs` — CCS vs OCS, rotator angle source
- `camera-temperature-sources` — ConsDB vs EFD camera thermal telemetry

The first two are directly relevant to the MIW work; add them if this snapshot is
meant to stand alone. Everything they cover that matters to the current
investigation is already restated in `MIW_INVESTIGATION_HANDOFF.md`.
