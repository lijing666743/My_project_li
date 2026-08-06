# Formula-Prose Integration Contract

Use this after the proactive formal opportunity sweep and before a candidate block is accepted into a manuscript.

The goal is not dense math. The goal is reader-usable formalization: every displayed block must earn its space and be stitched into the surrounding scientific argument.

## Discovery Before Filtering

Do not let reader-flow concerns suppress candidate discovery. First record the candidate and its source; then decide whether it should be kept, merged, moved, downgraded, deferred, or rejected.

Reader-heavy candidates should normally become `merge/group`, `move`, `downgrade`, or `defer` decisions rather than silent omissions. Accepted formal density should be high when blocks are evidence-grounded, useful, and prose-usable.

## Required Fields

For each displayed equation, algorithm, definition, assumption, theorem-style block, proof, complexity statement, or compact group, record:

- role: definition, objective, constraint, protocol, readout, update rule, complexity, or bounded formal property
- source: method, code, data, derivation, cited theory, or accepted manuscript protocol
- pre-block motivation: why the reader needs this block now
- post-block interpretation: what the block means and how it is used next
- symbol burden: whether symbols are defined locally, in a notation table, or still missing
- placement: main text, Methods, SI/appendix, table/notation note, or reject
- claim boundary: what the block does not prove or imply
- rhythm risk: whether the block creates back-to-back formula stacking, oversized symbol load, or theorem ceremony

## Decisions

- `keep`: formal block is supported, useful, introduced, interpreted, and referenced.
- `merge/group`: adjacent formulas form one conceptual family and should be explained by group-level prose.
- `downgrade`: a displayed block should become prose, inline math, a table row, or a notation note.
- `move`: a valid but heavy block belongs in Methods, SI, appendix, or a supplementary derivation.
- `defer`: a plausible candidate needs more evidence, definitions, placement confirmation, or source checking before insertion.
- `reject`: block is unsupported, decorative, unused, claim-strengthening, or reader-disruptive.

## Rhythm Rules

- A single displayed equation may be followed by another displayed equation only when the text explicitly says they are paired, sequential, or jointly define one construct.
- Long constraint/objective/KPI families should be introduced as a family before the block and interpreted as a family after the block.
- Theorem-style blocks require more than truth: they must add manuscript value beyond a definition restatement.
- A formula that mainly documents implementation bookkeeping should usually become a Methods-level protocol sentence or table row.
- Claim-boundary prose must be adjacent when a formula could be misread as a proof, guarantee, calibrated physical model, deployment claim, or superiority claim.

## Output Matrix

| Block/group | Role | Pre-block motivation | Post-block interpretation | Symbol burden | Rhythm risk | Decision |
|---|---|---|---|---|---|---|

If a block cannot be placed in this matrix, it is not ready for manuscript insertion.
