---
name: sci-image2-diagram-drafter
description: Generate logic-coupled, module-interaction dense, math-light, icon-minimized SCI diagram draft images with Codex imagegen / gpt-image-2, using the shared Image2 contract and illustration-bank boundary reads. Use for scientific mechanism, architecture, framework, module-interaction, training-loop, inference-pipeline, closed-loop learning/control, and Visio-redraw diagram drafts that emphasize macro regions, data/control flow, branches, loops, feedback, and academic labels. Generate a companion white-background shape/container board only when requested, object icons are unavoidable, or output mode is main-plus-board. Do not use for scenario figures, graphical abstracts, TikZ/PGFPlots, Python data plots, icon-only assets, transparent-background icon requests, or LaTeX integration.
---

# SCI Image2 Diagram Drafter

## Purpose

Create a high-information-density SCI mechanism, architecture, framework, workflow, module-interaction, training-pipeline, inference-pipeline, hybrid knowledge-source coupling, or closed-loop learning/control draft image for manual Visio redraw.

## Workflow

1. Read the shared contract: [../.shared/sci-image2-drafter-common.md](../.shared/sci-image2-drafter-common.md).
   For paper framework, architecture, method, mechanism, pipeline, module-interaction, training-loop, inference, or system/data-flow diagrams, also read [../.shared/sci-image2-paper-framework-studio-distillation.md](../.shared/sci-image2-paper-framework-studio-distillation.md). Use it only as local source-faithful prompt-audit and human-in-the-loop staging discipline; do not route to or inherit response rules from `paper-framework-figure-studio-pro`.
2. Mandatory boundary read: read [../.shared/sci-image2-illustration-style-bank/README.md](../.shared/sci-image2-illustration-style-bank/README.md) and [../.shared/sci-image2-illustration-style-bank/source-boundary.md](../.shared/sci-image2-illustration-style-bank/source-boundary.md) before drafting any prompt.
3. Conditional module read for diagrams: read [../.shared/sci-image2-illustration-style-bank/diagram-visual-grammar.md](../.shared/sci-image2-illustration-style-bank/diagram-visual-grammar.md), [../.shared/sci-image2-illustration-style-bank/illustration-archetypes.md](../.shared/sci-image2-illustration-style-bank/illustration-archetypes.md), [../.shared/sci-image2-illustration-style-bank/palette-linework-policy.md](../.shared/sci-image2-illustration-style-bank/palette-linework-policy.md), [../.shared/sci-image2-illustration-style-bank/prompt-fragments.md](../.shared/sci-image2-illustration-style-bank/prompt-fragments.md), and [../.shared/sci-image2-illustration-style-bank/negative-prompt-policy.md](../.shared/sci-image2-illustration-style-bank/negative-prompt-policy.md) when they match the task. Read [../.shared/sci-image2-illustration-style-bank/icon-vocabulary-and-board-policy.md](../.shared/sci-image2-illustration-style-bank/icon-vocabulary-and-board-policy.md) only when an object icon is unavoidable or a companion board is requested.
4. Lock the diagram boundary from repo/manuscript evidence: modules, submodules, inputs, outputs, data flow, control flow, training/inference path, branches, loops, feedback, interfaces, and math-light concept placeholders only when needed.
5. Select the output mode:
   - `main-only`: use when the user explicitly asks for no icon/shape/container board, when primitive shapes are sufficient, or when a board would add process clutter.
   - `main-plus-board`: use when the user asks for a board, object icons are unavoidable, or a Visio redraw needs separated reusable shapes.
   - Record the mode and the reason. In `main-only`, board prompt, board image generation, and board QA are `not applicable`.
6. Select visual policy switches before drafting:
   - `legend-free`: use local path labels and consistent line styles instead of a standalone legend or visual key.
   - `no-numbered-callouts`: use macro-region titles, flow direction, lanes, and section bands instead of numbered badges.
   - `academic-label-only`: suppress engineering terms, code identifiers, raw CSV/log labels, file paths, prompt/agent/audit/package wording, snake_case, and camelCase.
   - `journal-palette-flex`: use a Nature/Science-compatible, colorblind-safe, low-saturation palette with clear hierarchy rather than a fixed palette.
7. Build the content inventory: `must-show`, `should-show`, `optional`, and `forbidden`.
8. Build the module/interaction inventory: module hierarchy, submodules, program stages, branch points, update loops, input/output ports, state transitions, data/control arrows, and feedback relations.
9. Select the diagram composition mode: `module-interaction`, `hybrid knowledge-source coupling`, `closed-loop learning/control`, or `multi-stage pipeline with evaluation strip`. Do not force closed-loop or hybrid coupling when the source material is a plain architecture.
10. Run an IEEE-style commonality gap pass when a draft or prompt is clean but flat: add nested macro regions, record/batch/gate primitives, parallel lanes, ports, branch points, and feedback loops before adding icons, equations, or decorative density.
11. Build the logic-coupling inventory: macro regions, knowledge sources, loop direction, arrow semantics, state/data repositories, buffers, model/repository containers, constraint/evaluation outputs, bottom evidence/requirement strips, and repeated-process semantics.
12. Build the program-detail inventory: algorithmic steps rewritten as academic process labels, not source-code names or implementation variables.
13. Build the engineering-residue suppression list: code variable names, function names, class names, file paths, snake_case, camelCase, CLI/config keys, internal pipeline labels, prompt/agent/audit/package wording, and raw data-column labels that must not appear in the image.
14. Build the math suppression plan: avoid formulas and Greek-heavy symbols by default; if math is essential, use short academic placeholders such as `Objective term`, `Update step`, `Constraint check`, or `Policy evaluation`, and put exact LaTeX only in the Visio checklist.
15. Build the minimal primitive-shape inventory: default to rectangles, rounded rectangles, lanes, arrows, braces, ports, grouping boxes, dashed macro-region frames, database cylinders, vertical state ribbons, evaluation ovals, constraint braces, loop arrows, gates, and simple model insets. Add numbered callouts, compact legends, or object icons only when allowed by user constraints and needed for readability.
16. Select the Zotero-derived illustration archetype and visual grammar family; use them only as abstract prompt guidance, never as source-figure copying or evidence.
17. Draft the main prompt with [templates/image2_prompt.md](templates/image2_prompt.md), using the selected output mode, visual policy switches, macro-region layout, logic-coupling inventory, dense module lanes, program-detail flow, data/control arrows, branches, loops, feedback, academic labels, math-light placeholders, selected illustration grammar, and minimal primitive-shape inventory.
18. Use the system `imagegen` skill to generate one main draft unless the user asks for variants.
19. If output mode is `main-plus-board`, draft the minimal white-background shape/container-board prompt with [templates/icon_board_prompt.md](templates/icon_board_prompt.md), using the same minimal shape/container inventory, then use the system `imagegen` skill to generate one pure-white-background `#FFFFFF` board PNG.
20. Provide the shared output contract. Always include illustration bank boundary read, selected modules, selected archetype, source-copying risk check, composition mode, output mode, visual policy switches, `Logic-Coupling Inventory`, `Macro-Region Map`, loop/iteration semantics, module/interaction inventory, program-detail inventory, engineering-residue suppression list, math suppression plan, Visio redraw checklist, post-generation QA, and omitted/compressed content. When the framework-studio distillation applies, also include `Source-Faithful Prompt Audit`, `Visible Text Whitelist`, `Connector And Edge-Label Plan`, `Candidate/Asset Ledger`, and `Human Decision Boundary`. Include `Arrow Semantics Legend` and board QA only when a legend or board is allowed and generated; otherwise write `not applicable` with the reason.
21. If the user explicitly asks for a transparent PNG icon board, optionally run `../.shared/scripts/ensure_icon_board_transparency.py` as extra postprocessing and report manual QA risk.
22. If fallback is needed, use [templates/notebooklm_fallback_prompt.md](templates/notebooklm_fallback_prompt.md) and list `Sources to upload`.

## Diagram-Specific Slots

Default to Logic-Coupled Diagram Mode. Prioritize a readable scientific story built from macro regions, logic-coupling inventory, module hierarchy, submodules, data flow, control flow, training/inference path, input/output blocks, branch/loop/update structure, feedback loops, state/data repositories, evaluation/requirement strips, state transitions, and key interfaces.

Do not default to equations, mathematical symbols, code variables, legends, numbered callouts, or complex object icons. Express program details as concise academic labels such as `State encoding`, `Policy evaluation`, `Action selection`, `Experience update`, `Constraint check`, `Resource allocation`, or `Performance assessment`.

Prefer Zotero-derived abstract diagram archetypes such as module-interaction workflow, hybrid knowledge-source coupling, closed-loop learning/control, multi-stage pipeline with evaluation strip, joint optimization framework, and multi-agent control scene when the evidence boundary supports them. Use hybrid illustration plus legend/board only when the user allows those aids.

Never absorb reference images by copying their content. Absorb only abstract organization principles: macro-region framing, knowledge-source coupling, closed-loop arrows, repository primitives, arrow semantics, evaluation/requirement strips, and density through interaction.

For complex paper-framework diagrams, use the distilled human-in-the-loop staging model: lock source foundation, prepare a source-faithful prompt package, generate only the requested draft and optional board, write a concise issue ledger, then stop at Kevin's human decision boundary.

## Routing Boundaries

- Use `sci-image2-scenario-drafter` for system model, problem setting, network scenario, application environment, macro scenario, or entities-and-constraints overview figures.
- Use `sci-image2-infographic-drafter` for graphical abstracts, visual summaries, paper highlights, cover or TOC-like images, and research-story infographics.
- Use `sci-tikz-publication-figures` for TikZ, PGFPlots, standalone LaTeX figure sources, or LaTeX figure code.
- Use `sci-figures-python` for data-backed plots.
- Use system `imagegen` directly for standalone icon-only assets or transparent-background icon requests; this skill owns companion white-background boards only when `main-plus-board` mode is selected.
- Use `sci-latex-output` for manuscript integration.
- Use `sci-verification` before claiming the draft is evidence-complete.
