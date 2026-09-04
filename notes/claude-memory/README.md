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

Every **AOS / wavefront** memory is here, and the cross-references between them all
resolve within this directory.

| file | what it holds |
|---|---|
| `MEMORY.md` | the index loaded at session start — one line per memory |
| `reporting-units-standard.md` | standing requirement: every number carries its quantity name and units (or "dimensionless" with the ratio named) |
| `miw-focal-order-truncation.md` | 83% of MIW power sits above the k<=6 focal orders the build fits — reframes every DZ-subspace analysis |
| `miw-coadd-retrieval-bias.md` | the coadd-vs-MIW investigation: the retrieval-bias hypothesis, the L estimator (L == 0 iff unbiased), the Sigma_emp error model, and corrections made along the way |
| `miw-products-and-m3-backprojection.md` | which MIW file to use, the back-projection and joint static-fit results, the 5rot build selection |
| `sparse-fit-sensitivity.md` | the sparse donut-fit study: Piece 1 (primary<->secondary degeneracy) and the DOF-observability results |
| `frame-conventions-ccs-ocs.md` | CCS vs OCS, rotator angle source, what moves with the mirrors |
| `camera-temperature-sources.md` | ConsDB vs EFD camera thermal telemetry, the `MTCamera.utiltrunk_body` topic |
| `z11-intra-extra-mystery.md` | unexplained Z11/Z14 intra- vs extra-focal split in Danish unpaired CWFS — **open question, not a known effect** |

`z11-intra-extra-mystery.md` was not in the original request but is included because
it is the last AOS cross-reference and is directly pertinent: an intra/extra
asymmetry in the donut fit is itself a candidate retrieval-bias signature, which is
the hypothesis under test in `miw-coadd-retrieval-bias.md`.

## Known gaps

- `MEMORY.md` is the **full** index, so it lists memories that are not snapshotted
  here — the guider-topic files (`guider-*`, `deploy-summit-guider-branch`) and
  `aos-private-consdb-cache`. Those are unrelated to this investigation and were
  deliberately left out; the index lines describing them still resolve to nothing.
- `z11-intra-extra-mystery.md` links to `[[danish-tarts-compare-notebook]]`, which
  does not exist as a memory file even upstream — it is a forward reference marking
  something worth writing later, not an error.
