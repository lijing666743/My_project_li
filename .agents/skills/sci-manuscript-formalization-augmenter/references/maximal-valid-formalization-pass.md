# Maximal Valid Formalization Pass

Default to finding all effective formalization opportunities. The goal is high reader-usable formal density, not decorative math or formula stacking.

Maximal means broad candidate discovery before strict acceptance. It does not mean inserting every candidate into the main text.

## Pass Order

0. Opportunity sweep: list formalizable cues before deciding acceptance, placement, or rejection.
1. Notation: identify repeated variables, sets, indices, parameters, metrics, and functions.
2. Problem formulation: identify inputs, outputs, state/action spaces, constraints, and objective.
3. Equations: convert prose definitions of metrics, losses, rewards, scores, transformations, and update rules into candidate equations.
4. Algorithms: convert reproducible multi-step methods, training loops, inference procedures, or selection rules into candidate algorithm blocks.
5. Complexity: add candidate complexity statements when algorithmic loops, model size, or computational costs are defined enough.
6. Definitions and assumptions: formalize repeated technical terms and scope limits.
7. Propositions, lemmas, theorems: add candidates only when a precise statement and proof conditions exist.
8. Proof sketches: include candidates only when the proof can be derived from stated assumptions or cited theory.
9. Formula-prose rhythm: for every accepted displayed block or compact group, plan pre-block motivation, post-block interpretation, symbol-definition handling, downstream use, and claim-boundary wording.

## Coverage Matrix Requirement

For every candidate type, mark:
- accepted
- merge/group
- move to Methods, SI, appendix, table, or supplementary derivation
- downgrade to prose, inline math, notation note, or algorithm step
- defer because evidence, definitions, placement, or source confirmation is missing
- rejected as decorative
- rejected as unsupported
- rejected as too strong for the claim gate
- rejected as redundant with existing formal content
- rejected or moved because it creates formula stacking, excessive symbol burden, or poor reader flow

## Missed-Opportunity Check

Before closeout, explicitly check whether the section still has prose-only:

- metrics, losses, rewards, objectives, utilities, scores, or readouts that should become equations
- constraints, states, actions, transitions, protocols, or feasibility rules that should become formal definitions or constraints
- repeated procedures, training loops, inference paths, or selection rules that should become algorithm blocks
- bounded properties or conditions that may justify definition, assumption, proposition, lemma, theorem, or proof-sketch candidates
- complexity, scaling, runtime, memory, or convergence-boundary claims that should be formalized or downgraded

## Theorem Threshold

Theorem-style blocks require:
- explicit assumptions
- a precise statement
- proof path or cited theoretical basis
- no dependence on unverified empirical observations
- textual use in the manuscript

If any item is missing, reject the theorem and consider an assumption, definition, or empirical observation instead.
