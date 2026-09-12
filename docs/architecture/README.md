# Architecture

Read in this order:

1. [`DOMAIN_ARCHITECTURE.md`](DOMAIN_ARCHITECTURE.md) — what a fraud
   detection platform is, at enterprise scope: components, data flows,
   detection strategy, decisioning, governance, compliance considerations.
   Section 14 is the honest map of what's actually built versus designed.
2. [`GLOSSARY.md`](GLOSSARY.md) — shared domain vocabulary, referenced by
   every component spec.
3. [`CONTEXT_ARCHITECTURE.md`](CONTEXT_ARCHITECTURE.md) — how these two
   documents function as shared context for AI agents building any one
   component of the platform, and how that generalizes to every future
   project in this series.

These two concerns are deliberately separate documents: `DOMAIN_ARCHITECTURE.md`
is about fraud detection; `CONTEXT_ARCHITECTURE.md` is about the AI build
process and contains nothing fraud-specific. The next project in the
series replaces the first, reuses the second unchanged.
