# IEEE Wireless 2025-2026 Formal Block Style

Use this reference when drafting or auditing TWC/TCOM/JSAC-style equations, pseudocode, definitions, assumptions, theorem-style blocks, proof sketches, and complexity statements.

## Corpus Boundary

- Source route: local Zotero Desktop API, read-only metadata pagination plus indexed full-text reads.
- Corpus status: `coverage-incomplete`.
- Local coverage on 2026-06-20: 29 matched 2025-2026 TWC/TCOM/JSAC items, 29 attachments, 28 indexed full texts.
- Local journal/year counts: TWC 2025 = 21, TWC 2026 = 3, TCOM 2025 = 1, TCOM 2026 = 0, JSAC 2025 = 4, JSAC 2026 = 0.
- This is available-corpus style guidance only. It is not all-paper coverage, citation evidence, formula content, proof authority, or claim approval.

Do not copy source equations, pseudocode, proof text, source-paper sentences, paper-specific variables, numerical results, baselines, or novelty claims.

## Equation Practices

- Treat each displayed equation or compact equation family as a reader-facing method contract with a clear role: model definition, objective, constraint, update rule, metric, reward/loss, readout, or complexity/accounting.
- Introduce dense notation before or immediately after the displayed block; avoid orphaned symbols that only become clear several paragraphs later.
- Prefer objective-plus-constraint grouping only when the target manuscript actually defines the variables, objective, feasible set, and constraint families.
- Use paired or compact equations only when the text says why the equations are paired, sequential, or jointly define one construct.
- Put claim-boundary prose near formulas that could be misread as proof, convergence, optimality, calibrated physical meaning, deployment evidence, robustness, or superiority.

## Pseudocode Practices

- Add an algorithm block when reproducibility depends on initialization, loop order, update sequencing, stopping criteria, or fixed-variable scheduling.
- Each algorithm should expose inputs, outputs, initialization, repeated updates, termination, and final returned object.
- Keep algorithm variables aligned with accepted equations. If an equation variable is not actually used in the procedure, do not force it into pseudocode.
- Put complexity or accounting text near algorithms only when loop dimensions, dominant operations, or per-iteration costs are supported.
- Avoid algorithm blocks for ordinary coding workflow, file handling, plotting, or prompt/agent process.

## Theorem-Style Practices

- Use definitions and assumptions to reduce symbol burden before dense math; do not inflate them into theorem ceremony.
- Use propositions, lemmas, theorems, and proof sketches only when the target manuscript contains explicit conditions, a precise statement, and a proof path.
- Downgrade empirical observations, implementation bookkeeping, obvious boundedness, or restated definitions into prose, inline math, or equations.
- Do not create convergence, optimality, feasibility, robustness, superiority, or generalization claims from corpus style patterns.
- A theorem-style block is reader-usable only if the surrounding text explains why the result matters for the method or claim boundary.

## Closeout Matrix Additions

For every accepted or rejected formal block, record:

- block role: equation / pseudocode / definition / assumption / proposition / lemma / theorem / proof / complexity
- source in target manuscript: method, code, data, derivation, cited theory, or existing evidence
- corpus-style influence: equation practice / pseudocode practice / theorem-style practice / none
- boundary: style guidance only; not evidence or claim approval
- decision: keep / merge / move / downgrade / reject / defer

## Upgrade Trigger

Read this reference for TWC/TCOM/JSAC-style formalization, IEEE wireless optimization, ISAC/RIS/SAGIN, UAV/MEC, DRL/MARL, algorithm-heavy methods, theorem-style requests, or when the user asks for formula/pseudocode/theorem-block quality aligned with recent top IEEE wireless papers.
