# Equation-Adjacent Phrase Polishing

Use this when manuscript prose immediately before or after formulas, algorithms, definitions, assumptions, propositions, lemmas, theorems, proofs, or complexity statements needs refinement.

This is a prose-only pass. It cannot change formal content.

## Protected Content

Protect without modification unless a separate verified formalization pass explicitly authorizes changes:

- equation, algorithm, theorem-style, table, and figure labels
- mathematical symbols, variables, indices, constraints, objective functions, losses, rewards, and update rules
- algorithm logic, theorem statements, assumptions, proof steps, and complexity expressions
- numerical values, comparator names, citations, figure/table references, and claim boundaries

## Allowed Edits

- Replace abrupt formula introductions with motivation sentences.
- Convert implementation or workflow residue into manuscript-facing explanation.
- Add concise post-equation interpretation when the formula role is underspecified.
- Add transition text between paired equations or formula families.
- Make claim-boundary wording explicit when a formal block might be overread.
- Compress repetitive "where" clauses if symbols are already defined in a notation table.

## Formalization Handback Triggers

Stop phrase-only smoothing for the affected span and route to `sci-manuscript-formalization-augmenter` when prose:

- defines a metric, score, objective, utility, loss, reward, constraint, or update rule without a formal block
- describes a repeated method, training loop, inference path, selection rule, repair rule, readout protocol, or matching procedure
- introduces state, action, transition, policy, buffer, target, gradient boundary, or adaptive control behavior
- states a bounded property, monotonic relation, feasibility condition, convergence boundary, equivalence, or theorem-like claim
- makes complexity, scaling, runtime, memory, or parameter-count claims with enough structure to formalize
- repeats symbols or term families that would benefit from a notation table, definition, or compact equation family

The phrase refiner may propose a short handback note, but it must not draft or alter the formal block itself unless explicitly operating through the formalization skill.

## Required Report

| Formal item | Protected content | Prose edited | Purpose of edit | Claim-strength change |
|---|---|---|---|---|

| Handback candidate | Formalization cue | Suggested candidate type | Reason phrase-only editing is insufficient |
|---|---|---|---|

If the formula body, algorithm logic, or theorem statement appears wrong, stop the phrase pass and route the issue to `sci-manuscript-formalization-augmenter` plus `sci-verification`.
