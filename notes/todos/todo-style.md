# Writing style for `todo-ideas.md`

> **Status:** current · **Last updated:** 2026-09-29 · **Kind:** working state (style rules)

Rules for how an item in [todo-ideas.md](todo-ideas.md) is written, derived from Aaron's
edit of item 1, which is the worked example. All five items follow these rules; editing this
file is how the rules change.

## The shape of an item

Four parts, in this order.

1. **`## N. Headline`** — a sentence naming what gets compared, built or measured.
2. **Status line** — `**Status:** <state> · **Blocked on:** <thing>`, one line, no prose.
3. **The idea, in one short paragraph** — what gets done, on what data. No history, no
   implementation, no paths, no file names, no numbers from a probe.
4. **`**Goals:**`** — one or two sentences on what the work determines or validates.
5. **`<details>` block** — everything else, collapsed.

## The introduction carries no history and no implementation

This is the rule that matters most. The paragraph above the fold says *what the item is*,
in the terms the work would be described to a colleague. Specifically it does **not**
carry:

- **History** — what already exists, what was tried, what a previous version did, which
  notebook it generalizes, what is currently missing. None of that introduces the item.
- **Implementation** — which function matches donuts, what a matcher is written for, what
  needs generalizing, which file a reader lives in.
- **The constraint that shapes the result** — even a genuinely important one. If the
  interesting fact is that no method covers all three nights, that is a *detail*, and it
  belongs in the collections table and the open questions where it is actionable.
- **Numbers from a probe.**

What it does carry: the comparison or measurement, the methods or data involved, and the
`day_obs` or data set it runs on.

> **Example.** "Compare wavefront retrieval methods using regular FBS visits from several
> typical nights. Methods will include the current default Danish 1.2 paired, Danish 1.2
> unpaired, Danish 1.3 unpaired, Danish 1.3 unpaired with an updated pupil model, TARTS
> and possibly AIdonut. The day_obs being processed are 20260512, 20260513 and 20260713."

Note what that does *not* say: nothing about `compare_donuts.py`, nothing about the
two-sided matcher, nothing about coverage gaps, nothing about naming the study.

## `**Goals:**`, not "why it matters"

One or two sentences on what the work establishes. Phrase it as what gets determined or
validated, not as downstream consequence.

> "Determine the consistency or lack thereof between methods and validate the new methods."

## Inside `<details>`

Four subsections, in this order. Drop any that is empty. An item may add its own factual
sections — `### Data selection`, `### The bug`, `### What the probe found`,
`### The two optical-model variants` — but they go **between collections and scope**, never
after it, so `### Scope` and `### Open questions` are always the last two.

### `### Known collections`

**A table of what is present, and nothing more.** One row per method or data set, with the
collection path and the version fields. Coverage is a column, not a commentary.

What does not belong here: whether a collection's coverage is convenient, what it implies
for the comparison, whether versions skew, what to do about any of it. Those are open
questions. A sentence that begins "so" or "which means" is in the wrong section.

A short paragraph after the table is fine for facts a table cannot hold — that a
collection does not exist, that one carries a different dataset type. Still facts, not
implications.

### `### Existing machinery to build on`

A table of code and outputs that already exist, with paths, plus a short note on any
behaviour that constrains the work. This is where implementation detail is welcome.

### `### Scope`

**Bare deliverables, imperative, one line each.** No rationale, no trade-off, no "decide
whether". A scope line states what will be done; anything not yet decided is an open
question instead.

### `### Open questions`

Everything undecided, each as a **numbered bolded question followed by the options**, and
each followed by an answer line Aaron edits in place. The section opens with the same two
lines in every item:

```markdown
Answer by replacing the `_unanswered_` on the `**A:**` line. An answered question stays
here as the record of the decision.

**Q1. Short question in bold?** The options, in one or two sentences — what each choice
buys and what it costs.

**A:** _unanswered_
```

Numbering is per item and starts at Q1. An answered question is **not deleted** — the `**A:**`
line becomes the record of the decision, which is why the questions are numbered and stay
in place.

The test for what belongs here: if a line cannot be acted on without someone choosing
first, it is an open question, not scope. Anything that in an earlier draft appeared as
"decide whether X or Y", "pick one and be explicit", or "name it when the scope settles"
lives here.

## Sentence-level rules

From `rubin-work/CLAUDE.md`, and they apply here too.

- **Every number carries its quantity name and units**, or an explicit "dimensionless"
  with the ratio named.
- **Define acronyms on first use per item** — an item is read on its own, so FAM, CWFS,
  MIW, DZ, OFC, DOF, FBS, EFD, ConsDB each get expanded once in the item that uses them.
- **State outstanding work as a fact about the current state**, not as a plan or a
  scolding.
- **No meta-commentary about the file** — not "this item is large", not "see below".
- **Bold only the labels and genuine findings.** Four bolded phrases in one sentence read
  as none.

## Length

Above the fold: the headline, the status line, a short paragraph, and `**Goals:**`. Item 1
is 6 lines and that is the target, not a floor to grow from. The `<details>` block is as
long as it needs to be — it is a store, not a document.
