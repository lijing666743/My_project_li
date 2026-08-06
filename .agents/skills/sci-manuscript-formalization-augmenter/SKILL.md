---
name: sci-manuscript-formalization-augmenter
description: "Proactively discover and maximize reader-usable formalization opportunities in SCI manuscripts with evidence-grounded equations, notation tables, variable definitions, problem formulations, objectives, constraints, algorithm blocks, complexity statements, definitions, assumptions, propositions, lemmas, theorems, and proof sketches. Use during manuscript drafting, methods/results formalization, LaTeX manuscript work, reviewer revision, or whenever manuscript writing may benefit from rigorous formulas, algorithms, or theorem-style blocks. The skill actively surfaces candidate formal blocks before applying evidence, claim, and prose-rhythm gates. Every accepted formal block must be evidence-grounded, used by the text, symbol-defined, prose-integrated, and LaTeX-compilable; do not add decorative, unsupported, or formula-stacked formalism."
---

# SCI Manuscript Formalization Augmenter

## Encoding Route

When creating or auditing formulas, notation tables, algorithms, theorem blocks, Markdown, or LaTeX, read `C:\Users\didgm\.codex\skills\.shared\utf8-encoding-contract.md` in addition to the mathematical notation contract. Encoding status is separate from mathematical validity and must be reported when text artifacts are changed.

## Purpose

Maximize reader-usable formalization density in SCI manuscripts without inventing theory or evidence. Default to a proactive formal opportunity sweep followed by a maximal valid formalization pass: discover broadly, draft the useful candidates, and explicitly merge, move, downgrade, or reject unsupported, decorative, formula-stacked, or reader-disruptive candidates.

Bias toward finding more valid formalization opportunities before filtering them. The skill should be opportunistic in discovery and strict in acceptance.

## When To Use

Use when working on:

- methods, system model, problem formulation, algorithm design, experiments, results interpretation, or theory sections
- LaTeX manuscript drafting or revision
- reviewer requests for rigor, reproducibility, mathematical clarity, or algorithmic detail
- user asks for formulas, algorithms, theorems, definitions, complexity, or proof sketches
- broad manuscript writing where equations, algorithms, or theorem-style blocks may be valid even if the user did not explicitly ask

Coordinate with:

- `journal-manuscript-gatekeeper` before broad claim upgrades
- `sci-manuscript-phrase-refiner` after formal blocks are accepted and surrounding prose needs phrase refinement or academic compression
- `sci-latex-output` for LaTeX integration
- `sci-verification` before claiming formal content is complete or supported

## Formal Block Inventory

Always check these candidates:

- notation table
- variable definition
- system model equation
- metric equation
- loss or reward equation
- DL+RL state, action, transition, Q-update, policy/objective, and gradient-boundary equations when the method combines deep learning with reinforcement learning
- DL/RL cross-paradigm rescue blocks when a single-paradigm baseline is extended with a minimal complementary component: representation-to-state mapping, policy-controlled loss weighting, sample selection policy, adaptive resource/resource-allocation rule, supervised warm-start, joint objective, and baseline/hybrid/ablation algorithm block
- IEEE UAV/AAV/ISAC/MEC/RL system model, objective/constraint, non-convex problem statement, decomposition, alternating optimization, DRL/MARL algorithm, and complexity/convergence-boundary formalization when applicable
- objective function
- constraint set
- algorithm block
- complexity statement
- definition
- assumption block
- proposition
- lemma
- theorem
- proof sketch

## Proactive Formal Opportunity Bias

Unless the user explicitly limits the task to pure language polishing, assume manuscript prose may hide formalizable content.

Actively scan for:

- equations for metrics, losses, rewards, objectives, utilities, scores, readouts, updates, and transformations
- objective/constraint sets for prose feasibility conditions, resource budgets, service floors, and design tradeoffs
- algorithm blocks for repeated procedures, training loops, inference procedures, selection rules, repair steps, and readout protocols
- notation tables or definitions for repeated variables, indices, sets, states, actions, and protocol terms
- assumption, proposition, lemma, theorem, or proof-sketch candidates only when conditions, statement, use, and proof path are visible
- complexity, convergence-boundary, or scaling statements when loops, dimensions, or computational costs are defined enough

Record candidates before rejecting them. Reader-flow risk changes the placement decision (`merge/group`, `move`, `downgrade`, or `reject`); it is not a reason to silently skip an opportunity.

## Effectiveness Gate

A formal block is valid only if it has:

- purpose in the manuscript
- source in method, data, code, experiment, cited theory, or derivation
- defined symbols
- notation category, font policy, units, indices, and project-level exceptions
- no unsupported claim upgrade
- textual reference before or after the block
- prose rhythm: a motivation before the block or group, an interpretation after it, and enough transition language to prevent equation dumping
- placement that supports reader flow; compact paired equations are allowed only when group-level prose explains their relationship
- LaTeX representation that can compile in the target manuscript style

Candidate discovery is intentionally broad; final acceptance remains evidence-gated. Do not use readability concerns to justify under-formalized method prose when a compact, well-explained equation, algorithm, definition, or theorem-style block would improve reproducibility or scientific precision.

## Workflow

1. Identify the manuscript section and approved claim boundary.
2. Mandatory boundary read: on every invocation, read `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\README.md` and `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\source-boundary.md` before selecting any language-bank module.
3. Extract method variables, steps, assumptions, constraints, metrics, losses, rewards, objectives, and evidence.
4. Run the Proactive Formal Opportunity Sweep from `references/proactive-formal-opportunity-sweep.md` unless the scope is explicitly language-only.
5. Run the maximal valid formalization pass from `references/maximal-valid-formalization-pass.md`.
6. Apply the Formula-Prose Rhythm Contract from `references/formula-prose-integration-contract.md` before accepting block placement; build a formal-prose integration matrix for every displayed equation, algorithm, definition, theorem-style block, and compact formula group.
7. For DL+RL methods, read `references/dl-rl-formalization.md` and check whether the manuscript can support `state`, `action`, `reward`, transition, update target, supervised loss, joint objective, and algorithm blocks.
8. For UAV/AAV/ISAC/MEC/RL manuscripts, read `references/ieee-uav-formalization-patterns.md` and check whether the section can support an IEEE-style system model, objective/constraint set, decomposition, algorithm block, and complexity/convergence-boundary wording.
   When propagation, CSI, fading, blockage, Doppler, arrays, RIS, near-field, satellite, or interference equations are material, also read `references/wireless-channel-formalization.md` and require the target repo/manuscript `Channel Modeling Decision Record`; the shared channel bank guides structure but does not supply unsupported equations or parameters.
9. When drafting prose around a formal problem, algorithm block, complexity statement, or result interpretation, read the relevant shared language-bank section-function/domain module and `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\equation-context-phrasing.md`; use the language bank only for surrounding prose, problem-formulation wording, algorithm-transition wording, complexity-boundary wording, and result-interpretation wording, not for formulas, algorithms, theorem-style block content, evidence, or claim approval.
10. For wireless optimization, ISAC/RIS/SAGIN, UAV/MEC, and DRL/MARL manuscripts, read `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\formalization-patterns-from-zotero.md` and `zotero-derived-formalization-lexicon.md` as additional formalization-slot guidance, while keeping all actual formal blocks evidence-grounded in the target manuscript.
11. For TWC/TCOM/JSAC-style formal blocks, recent IEEE wireless optimization, algorithm-heavy ISAC/RIS/SAGIN/UAV/MEC/DRL/MARL methods, or user requests about formulas, pseudocode, and theorem blocks aligned with recent top IEEE wireless papers, read `references/ieee-wireless-2025-2026-formal-block-style.md`. Treat it as available Zotero corpus style guidance only, not all-paper coverage, evidence, formula content, proof authority, or claim approval.
12. For any formula, notation table, algorithm, theorem-style block, wireless channel chain, or RL formulation, read the shared mathematical-notation-governance reference. Record notation class, font policy, units, index policy, operator policy, first definition, exception, and verification status; treat the reference as formatting guidance only.
13. Build a formal opportunity expansion matrix and a formalization coverage matrix for all candidate block types.
14. Draft accepted blocks with purpose, source, symbols, placement, LaTeX, and pre/post prose plan.
15. Apply `references/formal-block-closeout-matrix.md` after drafting or auditing a manuscript section, and classify each block as `keep`, `downgrade`, `merge/group`, `move`, `defer`, or `reject`.
16. Downgrade mathematically true but manuscript-trivial theorem/proposition/lemma blocks into prose or equations when they add ceremony without scientific value.
17. Reject decorative, unsupported, unused, over-strong, or prose-disruptive candidates with explicit reasons.
18. Hand phrase and surrounding-prose refinement to `sci-manuscript-phrase-refiner`.
19. Hand LaTeX integration to `sci-latex-output` when editing `.tex`.
20. Hand completion claims to `sci-verification`.
21. When drafting Markdown, keep code identifiers in backticks and mathematical symbols or formulas in `$...$` / `$$...$$`.

## Persistent Blocker Deep Research Escalation

If equation, algorithm, definition, theorem-condition, proof-source, or formal-prose decisions remain unresolved after the risk-adaptive attempt threshold, read `C:\Users\didgm\.codex\skills\.shared\deep-research-escalation-contract.md`. Include the inline method context, symbols, assumptions, source gap, target reader need, and blocked claim boundary; research cannot authorize unsupported formalism.
## Guardrails

- Do not convert empirical observations into theorems unless proof conditions are available.
- Do not invent assumptions that make a theorem true but are absent from the method.
- Do not add unused equations or algorithms.
- Do not add more formalism than the target venue can read.
- Do not strengthen novelty, convergence, optimality, robustness, generalization, or superiority claims without evidence.
- Do not wrap mathematical symbols or formulas in Markdown code spans.

## Output Shape

```md
## Formal Opportunity Expansion Matrix
| Detected prose cue | Candidate block type | Why formalization helps | Draftable content | Evidence/source | Reader-usable plan | Decision |
|---|---|---|---|---|---|---|

## Missed-Opportunity Check
- Metrics / losses / rewards / objectives checked:
- Constraints / states / actions / protocols checked:
- Algorithms / procedures / readouts checked:
- Definition / assumption / theorem-style candidates checked:
- Complexity / convergence-boundary candidates checked:

## Formalization Coverage Matrix
| Candidate block | Accepted / Rejected | Purpose | Evidence/source | Placement | Rejection reason |
|---|---|---|---|---|---|

## Accepted Blocks
- Type:
- Purpose:
- Source/evidence:
- Symbols:
- Placement:
- LaTeX:

## Rejected Blocks
- Candidate:
- Rejection reason:

## Formal Block Closeout Matrix
| Block | Decision: keep / downgrade / reject | Reason | Replacement if downgraded | Verification need |
|---|---|---|---|---|

## Formal-Prose Integration Matrix
| Block/group | Role | Pre-block motivation | Post-block interpretation | Symbol burden | Back-to-back formula risk | Action |
|---|---|---|---|---|---|---|

## Equation Role Map
- Definition / objective / constraint / protocol / readout / complexity / theorem-style property:
- Linked figure/table/algorithm/result:
- Claim boundary:

## Pre/Post Prose Plan
- Required motivation sentence:
- Required interpretation sentence:
- Paired-equation or compact-group explanation:
- Phrase-refiner handoff:

## Downgrade / Merge / Move Candidates
- Candidate:
- Reason:
- Safer replacement:

## Prose Integration Notes
- Required surrounding text:
- Terms needing manuscript phrase refinement:
- Shared language-bank boundary read:
- Selected language-bank modules, or `none selected`:

## Verification Needs
- Compile check:
- Claim/evidence check:
- Missing definitions:
```

## References

- Mathematical notation governance: C:\Users\didgm\.codex\skills\.shared\mathematical-notation-governance.md

- Maximal pass: `references/maximal-valid-formalization-pass.md`
- Proactive opportunity sweep: `references/proactive-formal-opportunity-sweep.md`
- Formula-prose rhythm contract: `references/formula-prose-integration-contract.md`
- Formalization rules: `references/formalization-rules.md`
- Validity gate: `references/formal-block-validity-gate.md`
- Formal block closeout matrix: `references/formal-block-closeout-matrix.md`
- Markdown technical notation: `references/markdown-technical-notation.md`
- LaTeX templates: `assets/formal_blocks_template.tex`
- DL+RL formalization: `references/dl-rl-formalization.md`
- IEEE UAV/AAV/ISAC/MEC/RL formalization: `references/ieee-uav-formalization-patterns.md`
- IEEE wireless 2025-2026 formal block style: `references/ieee-wireless-2025-2026-formal-block-style.md`
- Wireless channel formalization: `references/wireless-channel-formalization.md`
- Shared SCI language bank boundary read for surrounding prose only: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\README.md`
- Shared SCI language bank source boundary: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\source-boundary.md`
- Shared equation-context phrasing: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\equation-context-phrasing.md`
- Zotero-derived formalization slots: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\formalization-patterns-from-zotero.md`
- Zotero-derived formalization lexicon: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\zotero-derived-formalization-lexicon.md`
