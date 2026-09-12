# The 3-Agent Build Process

> **This is execution tooling, not the architecture.** It's "Layer 0" of
> the context model described in
> [`../architecture/CONTEXT_ARCHITECTURE.md`](../architecture/CONTEXT_ARCHITECTURE.md)
> — the mechanism used to build any single component *after* the domain
> architecture ([`../architecture/DOMAIN_ARCHITECTURE.md`](../architecture/DOMAIN_ARCHITECTURE.md))
> has defined what that component is and why it exists. Read the
> architecture docs first.

This project is the first in a series built using a repeatable, three-role AI
agent workflow instead of a single agent doing everything end to end. The
goal is separation of concerns that mirrors a real engineering team: nobody
writes code against a spec they wrote themselves five minutes ago, and nobody
reviews their own homework.

## Roles

1. **Specifier** (`specifier.md`) — Never writes code. Produces `SPEC.md`
   (what's being built, scope, explicit non-goals) and `QA_PLAN.md`
   (acceptance criteria, test plan, quality gates) *before* implementation
   starts.
2. **Implementor** (`implementor.md`) — Never invents scope. Builds strictly
   against `SPEC.md`, writes the accompanying tests, and stops at the
   boundary the Specifier drew.
3. **Reviewer / QA** (`reviewer.md`) — Never fixes its own findings silently.
   Runs the quality gates from `QA_PLAN.md` (lint, type-check, tests,
   coverage), checks every acceptance criterion explicitly, and writes
   `QA_REPORT.md` with a pass/fail per item plus any defects found.

Each role is a separate agent invocation with only the artifacts it needs
(not the full prior conversation), so its output has to stand on its own —
the same discipline as handing a ticket to a different engineer.

## Why this is worth writing about

- **Traceable provenance**: every file in `docs/spec/` and `docs/qa/` is a
  timestamped, committed artifact — not a claim after the fact.
- **Real quality gates**: CI enforces the same checks the Reviewer agent
  runs locally (`.github/workflows/ci.yml`), so "the QA agent approved this"
  is independently verifiable by anyone reading the repo.
- **Reusable across projects**: the role definitions in this folder are
  project-agnostic. The next project in the series reuses `specifier.md`,
  `implementor.md`, and `reviewer.md` unchanged — only `docs/spec/SPEC.md`
  and `docs/qa/QA_REPORT.md` change per project.

## Artifact flow

```
Specifier   -->  docs/spec/SPEC.md, docs/spec/QA_PLAN.md
Implementor -->  src/, tests/         (reads SPEC.md)
Reviewer    -->  docs/qa/QA_REPORT.md (reads QA_PLAN.md, runs gates, reads the diff)
```

If the Reviewer finds blocking defects, they go back to the Implementor with
the specific QA_PLAN item that failed — never a silent fix by whoever notices
first.
