# IEEE UAV/AAV/ISAC/MEC/RL Phrase Patterns

## Source Boundary

This file is a compatibility wrapper for the shared SCI manuscript language bank. For new work, read:

- `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\source-boundary.md`
- `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\section-function-patterns.md`
- `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\grammar-patterns.md`
- `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\claim-strength-lexicon.md`
- `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\domain-ieee-uav-isac-mec-rl.md`

This reference abstracts phrase behavior from a local review of 11 IEEE-series UAV/AAV/ISAC/MEC/RL papers supplied by Kevin on 2026-05-17. It records rhetorical patterns, not reusable source text. Do not copy long sentences from those papers.

Use this reference for top-journal phrase refinement in UAV, AAV, ISAC, MEC, RIS, beamforming, trajectory optimization, MARL/DRL, and wireless-sensing manuscripts.

## Section-Function Patterns

### Abstract And Problem Framing

Prefer a compact chain:

1. application setting
2. technical bottleneck
3. jointly optimized variables or modeled phenomena
4. method class
5. evidence type and benchmark relation

Safe phrase functions:

- introduce a system setting without hype
- state the coupled design variables
- identify why the problem is difficult
- state the proposed method as a response to that difficulty
- report result evidence with benchmark scope

Avoid generic openings such as "with the rapid development of" unless the journal style or section truly needs broad context.

### Challenge Framing

For optimization-heavy IEEE writing, articulate challenge sources through coupling, constraints, non-convexity, uncertainty, partial observability, multi-agent scalability, or mixed discrete-continuous decisions.

Use measured language:

- "The resulting problem is challenging because ..."
- "The coupling among ... prevents direct decomposition."
- "The mixed variables and constraints make the problem non-convex."
- "The dynamic environment requires an adaptive policy rather than a static rule."

Do not write "intractable" unless complexity or proof supports it.

### Problem Formulation Prose

Before a formal problem statement, explain:

- decision variables
- objective
- constraints
- time horizon or scenario index
- what each major term represents

After a formal problem statement, explain:

- why standard convex methods do not directly apply
- which variables are fixed or updated in each subproblem
- which approximation or learning mechanism is admitted by evidence

### Contribution Phrasing

Prefer contribution bullets that each include:

- contribution object
- technical mechanism
- evidence or role in the paper
- boundary

Avoid contribution bullets that are only topic labels. Avoid claiming novelty unless the nearest-neighbor literature check supports it.

### Method Transition

Use transitions that reflect scientific logic:

- from system model to optimization problem
- from problem hardness to decomposition or learning
- from algorithm design to convergence, complexity, or feasibility boundary
- from simulation setup to benchmark comparison

Avoid process chronology such as "we first write code" or internal workflow wording.

### Result Interpretation

IEEE result phrasing usually ties a result to:

- metric
- benchmark
- scenario or constraint
- evidence type
- limitation if applicable

Prefer measured verbs such as "show", "indicate", "validate", "illustrate", "reduce", "improve", or "achieve under the tested settings". Use "demonstrate" only when the evidence is direct and sufficiently strong.

Do not write robustness, superiority, scalability, or real-world readiness unless the evidence gate admits those claims.

## Rewrite Checklist

- Preserve every number, metric, comparator, and scenario boundary.
- Replace vague "good/effective/superior" wording with benchmark-grounded phrasing.
- Make the section function visible within the first sentence of the paragraph.
- Keep formulas and symbols unchanged.
- Keep phrase polishing subordinate to claim-evidence boundaries.
