# Formal Block Closeout Matrix

Use this after a formalization pass or before final polishing. The goal is not to keep every possible formal block; the goal is to keep only useful, supported, manuscript-facing formalism.

## Decisions

- `keep`: the block is useful, cited by surrounding prose, symbol-defined, evidence-grounded, prose-integrated, and LaTeX-checkable.
- `merge/group`: adjacent formulas are valid only as one conceptual family and need group-level motivation and interpretation.
- `move`: the block is useful but too heavy for the current narrative position; move it to Methods, SI, appendix, table, or notation note.
- `downgrade`: the block is mathematically true but too trivial, ceremonial, or disruptive as theorem/proposition/lemma; convert it to prose, a simple equation, a definition, or a notation note.
- `defer`: the candidate appears useful but lacks evidence, definitions, placement confirmation, or source checking; keep it in the opportunity matrix for follow-up rather than inserting it.
- `reject`: the block is unsupported, decorative, unused, over-strong, unproven, or not aligned with the manuscript claim gate.

## Matrix Columns

| Block | Type | Decision | Reason | Prose/rhythm issue | Replacement if downgraded/moved | Verification need |
|---|---|---|---|---|---|---|

## Downgrade Triggers

- The statement is a restatement of a definition or metric.
- The proof is tautological or just follows from notation.
- The theorem-like form implies a theoretical contribution that the paper does not need or support.
- The block distracts from the method or results more than it clarifies them.
- The block is valid but creates a dense run of formulas without enough motivation, interpretation, or transition prose.

## Reject Triggers

- Missing assumptions or proof conditions.
- Empirical observation is presented as theorem.
- Symbols are undefined or not used elsewhere.
- The block would strengthen convergence, optimality, robustness, superiority, or generalization beyond evidence.
- The block cannot be explained with a clear manuscript role and claim boundary.
