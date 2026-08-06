# IEEE UAV/AAV/ISAC/MEC/RL Formalization Patterns

## Source Boundary

This reference abstracts formalization behavior from a local review of 11 user-supplied IEEE-series UAV/AAV/ISAC/MEC/RL papers from 2024-2025. It records reusable formalization slots, not copied equations or source text.

## Common Formalization Slots

Use these slots only when the manuscript evidence supports them.

### System Model

Check whether the section needs:

- entity sets, such as UAVs/AAVs, users, sensors, targets, RIS elements, edge servers, or tasks
- time slots, trajectory states, channel states, sensing states, or queue states
- communication, sensing, computation, energy, or mobility variables
- coupling statements between trajectory, association, beamforming, resource allocation, offloading, and policy variables

### Problem Formulation

For optimization-heavy manuscripts, check:

- objective function
- decision-variable list
- constraint set
- time horizon or episode horizon
- discrete-continuous coupling
- source of non-convexity or mixed-integer hardness
- feasibility or relaxation boundary

Do not state non-convexity, NP-hardness, or global optimality unless the method or cited theory supports it.

### Algorithm Block

Candidate algorithm blocks include:

- alternating optimization
- successive convex approximation
- matching-assisted optimization
- DRL/MARL policy update
- digital-twin-assisted training loop
- trajectory/resource/beamforming update sequence
- simulation-only evaluation loop

Each algorithm block must specify input, output, initialization, update order, stop condition, and what is fixed while each subproblem is solved.

### DL/RL Or MARL Contract

For learning-based methods, require:

- state
- action
- reward
- transition or episode update
- policy or Q-value update
- replay/target-network status if applicable
- training/inference separation
- seed or `seed_set` boundary for robustness claims

### Complexity And Convergence Boundary

Prefer bounded wording:

- computational complexity per iteration or per episode
- convergence to a stationary point only when conditions support it
- empirical convergence when only simulation curves exist
- no global-optimality claim for non-convex DRL or alternating heuristics unless formally proven

## Closeout Rule

After drafting formal blocks, hand surrounding prose to `sci-manuscript-phrase-refiner` and completion claims to `sci-verification`.
