# Proactive Formal Opportunity Sweep

Use this before deciding that a manuscript section is sufficiently formalized.

Default posture: discover broadly, accept strictly. The sweep should surface every plausible equation, algorithm, definition, assumption, proposition, lemma, theorem, proof-sketch, complexity, and readout opportunity before the reader-usable gate decides placement.

## Trigger Cues

Scan prose for:

- repeated variables, sets, indices, dimensions, states, actions, parameters, metrics, or protocol terms
- prose definitions of a metric, score, utility, loss, reward, objective, transformation, update, or readout
- prose constraints, feasibility rules, service floors, resource budgets, fairness requirements, safety boundaries, or tradeoff conditions
- multi-step procedures, training loops, inference paths, selection rules, repair rules, matching rules, or repeated update cycles
- state/action/transition/reward language in DL/RL, DRL, MARL, adaptive-control, scheduling, allocation, or policy-learning methods
- theorem-like claims about monotonicity, boundedness, feasibility, convergence boundary, equivalence, dominance under conditions, or complexity
- complexity, scaling, runtime, memory, or parameter-count claims with defined loop or dimension structure
- repeated references to a downstream figure, table, ablation, benchmark, or protocol that would benefit from a formal readout definition

## Candidate Types

Consider:

- displayed equation
- objective and constraint set
- notation table
- definition
- assumption
- algorithm block
- proposition, lemma, theorem, or proof sketch
- complexity statement
- convergence-boundary statement
- evaluation/readout protocol

## Decision Rule

Reader-flow risk changes the decision; it does not erase the candidate. Mark each candidate as:

- `keep`: supported and reader-usable in the current location
- `merge/group`: valid but should be explained as a conceptual family
- `move`: valid but belongs in Methods, SI, appendix, or a supplementary derivation
- `downgrade`: useful concept but better as prose, inline math, table row, notation note, or algorithm step
- `defer`: potentially useful but missing evidence, definitions, placement, or source confirmation
- `reject`: unsupported, decorative, unused, over-strong, or scientifically misleading

## Guardrails

- Do not convert empirical observations into theorem-style blocks.
- Do not invent assumptions, objectives, constraints, or proofs to make a candidate formalizable.
- Do not use theorem language for implementation bookkeeping or obvious definitions.
- Do not accept a formal block only because it increases density.
- Do not skip a valid formal opportunity only because dense math would require prose work; instead, plan the surrounding prose or move the block.

## Required Output

| Detected prose cue | Candidate block type | Why formalization helps | Draftable content | Evidence/source | Reader-usable plan | Decision |
|---|---|---|---|---|---|---|

Close the sweep with a missed-opportunity check across metrics/objectives, constraints/states/actions, algorithms/procedures, definition/theorem-style candidates, and complexity/readout candidates.
