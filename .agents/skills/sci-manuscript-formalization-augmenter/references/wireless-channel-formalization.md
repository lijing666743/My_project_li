# Wireless Channel Formalization

Use with `sci-wireless-channel-modeling` and the target manuscript/repo evidence.

## Required Equation Order

1. coordinate frame and link geometry;
2. per-link large-scale and propagation-state terms;
3. small-scale fading or path-sum model;
4. array, RIS/STAR-RIS, polarization, or near-field response;
5. delay, Doppler, and CSI estimate/error;
6. composite channel;
7. received signal, interference, and noise;
8. SNR/SINR/rate or other KPI;
9. optimization/RL state, action, reward, and objective.

## Acceptance Gate

Every displayed channel equation needs:

- a role in the model;
- defined symbols and units;
- a source or derivation anchor;
- an implementation mapping;
- surrounding prose;
- a validation or simplification boundary.

Do not derive an optimization objective from undefined channel terms. Do not add a complex fading, blockage, Doppler, or surface model only to increase formal density.

