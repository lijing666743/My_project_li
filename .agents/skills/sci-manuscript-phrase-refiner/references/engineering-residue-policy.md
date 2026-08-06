# Engineering Residue Policy

Engineering residue is internal implementation or Vibe Coding language that should not leak into final SCI manuscript prose unless it is a legitimate reproducibility detail.

## High-Risk Residue

Flag by default:
- internal labels such as `phase 3w`, `stage 1d`, `wave 02`, `P1v3`, `run_id`, `claim_gate`, `current_state`, `next_task`
- agent/process terms such as prompt, Codex, Gemini, ChatGPT, Jarvis, subagent, automation, handoff, closeout
- repo artifact terms such as logs, dashboard, manifest, artifact path, local script, notebook draft
- unqualified `pipeline` when it means an internal workflow rather than a scientific process
- `\texttt{}` wrapping internal labels, workflow names, or project artifacts
- OS paths, local folders, cache paths, or private workspace names

## Allowed Technical Use

Allow when context clearly requires it:
- real software packages, APIs, commands, functions, class names, config keys, or file formats
- reproducibility-focused implementation sections or appendices
- standard technical terms such as training pipeline, inference pipeline, data-processing pipeline, or rendering pipeline when they describe the actual scientific system
- `\texttt{}` for real code identifiers, commands, package names, or file paths in technical reproducibility contexts

## Replacement Strategy

- Replace internal chronology with scientific chronology.
- Replace repo artifacts with paper-level evidence surfaces.
- Replace prompt/agent/process wording with method, experiment, or analysis wording.
- Replace unqualified `pipeline` with framework, workflow, processing chain, model, training procedure, inference process, or evaluation protocol as appropriate.
- If a term is ambiguous, keep the scientific meaning and list the ambiguity under verification needs.
