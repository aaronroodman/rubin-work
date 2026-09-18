# Running multiple Claude Code sessions on one repository

> **Status:** current · **Last updated:** 2026-09-18 · **Kind:** reference (working practice)

How to get more work done in parallel on `rubin-work` without sessions colliding.
Verified against Claude Code **2.1.273** on S3DF, which is what is installed here.

## Contents

- [The problem: what actually collides](#the-problem-what-actually-collides)
- [Pattern 1 — one session per topic, same working tree](#pattern-1--one-session-per-topic-same-working-tree)
- [Pattern 2 — git worktrees, for parallel edits](#pattern-2--git-worktrees-for-parallel-edits)
- [Pattern 3 — subagents inside one session](#pattern-3--subagents-inside-one-session)
- [Managing sessions](#managing-sessions)
- [What to use for this repository](#what-to-use-for-this-repository)

## The problem: what actually collides

Two Claude sessions in the same directory share one working tree and one git index.
Claude Code does **not** lock files between sessions. So what collides is:

- **The git index.** If session A stages files and session B runs `git commit`, B
  commits A's staged work. This is the most likely way to lose or scramble work.
- **The working tree.** Both sessions see and can write the same files. Two edits to one
  file race; last writer wins, silently.
- **The branch.** One `HEAD` for both, so neither can be on a different branch.

What does **not** collide: each session's conversation and transcript are independent,
stored per session under `~/.claude/projects/<project>/`.

The practical consequence: **same-tree parallel sessions are safe when they touch
disjoint files, and unsafe otherwise.** That matches how you are already working — this
session was told to stay off the other session's paths — but it depends on discipline
rather than enforcement.

## Pattern 1 — one session per topic, same working tree

The simplest approach, and adequate when the topics are genuinely independent, which in
this repository they usually are (`CLAUDE.md` calls the topics "mostly independent lines
of work").

```bash
tmux new -s aos            # window 1
cd ~/notebooks/rubin-work && claude

tmux new -s guider         # window 2
cd ~/notebooks/rubin-work && claude
```

Rules that make it safe:

1. **One topic per session, stated in the first prompt.** "Work only in `guider/`."
2. **Name every session immediately:** `/rename guider-atmo-study`. Without a name you
   cannot tell two sessions apart in the resume picker.
3. **Only one session commits at a time.** This is the real hazard. Either designate one
   session as the committer, or have each session commit *named paths* rather than `-A`:
   `git commit -- guider/` instead of `git add -A && git commit`.
4. **Never run `gitpull` while another session has uncommitted work** — it stashes and
   rebases the shared tree underneath the other session.

## Pattern 2 — git worktrees, for parallel edits

When two sessions must edit the *same* files, or when you want each on its own branch,
give each its own working tree. The repository is currently single-worktree
(`git worktree list` shows only the main checkout).

Claude Code has this built in:

```bash
cd ~/notebooks/rubin-work
claude --worktree science-lut-refactor
```

This creates a worktree under `.claude/worktrees/`, on a new branch, and puts the session
in it. Claude Code then **blocks that session from editing the main checkout**, which is
the enforcement Pattern 1 lacks. Add `--tmux` to have it create the tmux session too:

```bash
claude --worktree science-lut-refactor --tmux=classic
```

Or drive it by hand, which is worth knowing since it is plain git:

```bash
git worktree add ~/wt/science-lut -b science-lut-refactor
cd ~/wt/science-lut && claude
# when done
git worktree remove ~/wt/science-lut
```

Two caveats for this repository:

- **`.claude/worktrees/` is not in `.gitignore`.** Add it before using `--worktree`, or
  the worktree shows up as untracked and `gitpull` may complain.
- **`output/` is a symlink in `aos/` and `blocks/`.** A new worktree gets a *real* empty
  directory there instead, so an analysis run inside a worktree writes to the wrong place
  or fails. Either work on code only in worktrees, or re-link with
  `common/scripts/relink_output_dirs.sh`. This is the main reason worktrees suit
  refactoring and documentation work better than analysis runs here.

## Pattern 3 — subagents inside one session

Within a single session, delegate independent read-heavy work to subagents. They run in
parallel, each with its own context, and only their conclusions come back — which is why
the audit behind the reorganization plan cost this session very little context despite
reading most of the repository.

This is the highest-leverage option for **survey and review** work: "audit `optatmo/`
for study seams" and "audit `smatrix/`" are independent and can run at once. It does not
help for editing, since subagents writing to the same tree have the same collision
problem, and a defined agent can be given `isolation: worktree` if it must write.

You can define reusable agents in `.claude/agents/*.md`. A useful one here would be a
read-only auditor pinned to this repository's conventions — worth adding if the Part C
review proceeds study by study, since the same prompt gets reused a dozen times.

## Managing sessions

Verified flags on 2.1.273:

| command | what it does |
|---|---|
| `/rename <name>` | name the current session — do this first, always |
| `claude --resume` | interactive picker of sessions for this directory |
| `claude --resume <term>` | picker filtered by a search term (session ID or text) |
| `claude --continue` | reopens the **most recent** session in this directory — avoid with several sessions open; it will grab the wrong one |
| `claude --bg "<prompt>"` | run a session detached; prints an id |
| `claude agents` | dashboard of background sessions; attach, rename, stop |
| `claude attach <id>` / `logs <id>` / `stop <id>` | drive one background session |
| `/sessions` | skill listing past sessions with dates and first prompts |

`--continue` versus `--resume` is the one to get right: with two or more sessions per
directory, always use `--resume` and pick from the list.

Sessions can also message each other (`ListAgents` to discover, then ask Claude to send
a message), which is useful for "I have finished with `common/efd_db.py`, it is yours".
It is coordination, not synchronization — a message cannot approve a permission or
release a file lock in the other session.

## What to use for this repository

For 2–4 parallel lines of work here:

1. **Default to Pattern 1** — one tmux window and one named session per topic, with the
   commit discipline above. The topics are independent, so collisions are rare, and this
   avoids the `output/` symlink problem entirely.
2. **Use a worktree (Pattern 2) for any refactor that rewrites many files** — the
   `science_lut/` extraction is the obvious candidate. It touches 9,476 lines, and doing
   it on a branch in its own tree means an analysis session can keep running on `main`
   undisturbed.
3. **Use subagents (Pattern 3) freely for read-only survey and review**, which is most of
   the code-review plan.
4. **Keep one session as the designated committer** when several are live, or scope every
   commit to explicit paths.

The workflow / "ultracode" orchestration tool also exists and can fan out dozens of
agents, but it is token-expensive and aimed at problems like "audit 100 files in
parallel". The Part C review is a dozen study-sized units that each need your judgement,
so ordinary sessions plus subagents fit it better.
