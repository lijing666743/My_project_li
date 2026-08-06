# Image2 Draft Prompt

Use case: module-interaction-dense-diagram
Asset type: high-information-density SCI diagram draft for manual Visio redraw
Output mode: [main-only / main-plus-board]
Primary request: Create a dense but redrawable flat vector-like scientific [mechanism/workflow/architecture/framework/module interaction/training pipeline/inference pipeline/concept] diagram about [topic].
Audience: SCI manuscript reviewers and domain researchers.

Evidence boundary: Show only these supported modules, submodules, program stages, inputs, outputs, data flows, control flows, branch points, update loops, feedback relations, interfaces, and academic process steps: [supported facts]. Do not add new results, causal claims, performance rankings, statistical significance, journal logos, unsupported mechanisms, or implementation details not present in the source evidence.

Zotero-derived illustration grammar: Boundary files read: `sci-image2-illustration-style-bank/README.md` and `source-boundary.md`. Selected modules: [diagram-visual-grammar / illustration-archetypes / palette-linework-policy / prompt-fragments / negative-prompt-policy]. Selected composition mode: [module-interaction / hybrid knowledge-source coupling / closed-loop learning-control / multi-stage pipeline with evaluation strip]. Selected archetype: [pipeline/workflow / layered architecture / joint optimization framework / multi-agent control scene / hybrid knowledge-source coupling / closed-loop learning or decision]. Use these only as abstract top-journal visual grammar; do not copy any source-paper figure, caption, layout, values, formulas, colors, source-specific terms, or conclusions.

Output-mode policy: If `main-only`, generate exactly one main diagram and do not generate or depict a separate icon board, shape board, container board, or companion asset. If `main-plus-board`, generate the main diagram first and use the companion board template only after the main image is accepted.

Visual policy switches:
- Legend policy: [legend-free / compact legend allowed]. If legend-free, do not draw any standalone arrow legend, visual key, line-style explanation box, or legend panel; express arrow meaning through local path labels, line styles, color consistency, and region placement.
- Callout policy: [no-numbered-callouts / numbered callouts allowed]. If no-numbered-callouts, do not draw numbered circles, badge labels, or 1/2/3/4 markers; use macro-region titles, flow direction, lanes, and section bands.
- Palette policy: [journal-palette-flex / fixed palette]. If journal-palette-flex, use a Nature/Science-compatible, colorblind-safe, low-saturation palette with clear hierarchy and no red-green dependency.
- Visible label policy: [academic-label-only / ordinary concise labels]. If academic-label-only, suppress code variables, file names, function names, class names, snake_case, camelCase, raw CSV/log labels, local paths, prompt/agent/audit/package wording, and internal workflow labels.

Module/interaction inventory: [module hierarchy, submodules, lanes, program stages, input/output ports, branch points, update loops, state transitions, data/control arrows, feedback relations]. This inventory is the main source of density.

Logic-coupling inventory: [macro regions], [knowledge sources: data / environment / model / physics or rules / evidence], [state or data repositories], [buffers or memory stores], [model or target containers], [constraint/evaluation outputs], [loop direction], [repeated-process semantics], [bottom evidence or requirement strip]. Use this inventory to make the diagram a logic-coupled scientific figure rather than an independent module list.

IEEE-style commonality gap pass: If the diagram would otherwise become a clean but flat box list, increase density through nested macro regions, record stores, record/batch stacks, gates, ports, parallel lanes, branch points, feedback loops, and bottom metric/evidence strips. Do not increase density through object icons, dense formulas, copied source structures, or decorative neural-network ornaments.

Visual logic composition plan: Organize the figure into 3-5 macro regions with clear scientific roles. Recommended patterns: [source inputs -> feature/representation -> predictive/model layer -> rule/mechanism layer -> evaluation/requirement strip] or [environment/state -> policy/model evaluation -> action/decision -> repository/buffer -> loss/update/target -> feedback]. Choose only the pattern supported by the source evidence.

Macro-region layout: [region names and roles]. Use dashed or lightly tinted frames, swimlanes, grouped panels, or bottom/side strips to show region membership. Do not force a left-input/middle-model/right-output template if the source logic is different.

Coupling semantics: Define arrow roles before drawing: [primary state/action flow], [data flow], [control/update flow], [diagnostic/audit flow], [feedback loop], [evidence/requirement link]. Use consistent arrow colors/weights/styles. Include a legend only when the selected legend policy allows it; otherwise use local labels and consistent line styles.

Closed-loop / repeated-process notation: If the method is iterative or learning-based, show only evidence-supported loops such as replay/buffer update, target/model update, diagnostic feedback, or repeated scheduling/control. Do not imply online control, convergence, deployment readiness, or validated adaptation unless explicitly supported.

Container and repository primitives: Use Visio-redrawable containers such as database cylinders, repository blocks, dashed macro-region frames, vertical state ribbons, evaluation ovals, constraint braces, decision diamonds, gates, ports, and simplified 3-layer model insets when they help explain logic coupling. Avoid complex decorative neural networks or source-specific object icons.

Program-detail inventory: [algorithmic or procedural details rewritten as concise academic labels]. Use academic labels such as `State encoding`, `Feature extraction`, `Policy evaluation`, `Action selection`, `Experience update`, `Constraint check`, `Resource allocation`, `Model refinement`, `Performance assessment`, or other source-supported phrases.

Academic label rewrite: Convert code variables, file names, function names, class names, snake_case, camelCase, config keys, and internal pipeline labels into short scholarly phrases. Do not display implementation identifiers, function names, class names, file paths, CLI options, raw variable names, prompt labels, agent labels, artifact/audit/package labels, or raw data-column names.

Math-light policy: Do not draw dense formulas, Greek-heavy equations, matrices, summations, or exact LaTeX in the image. If a mathematical idea is necessary, use a short academic placeholder such as `Objective term`, `Update step`, `Constraint check`, `Policy evaluation`, or `Loss balancing`. Exact formulas will be listed separately in the Visio checklist, not rendered in the raster draft.

Icon-minimized policy: Use primitive Visio-redrawable shapes by default: rectangles, rounded rectangles, lanes, arrows, braces, ports, grouping boxes, dashed macro-region frames, database cylinders, vertical state ribbons, evaluation ovals, constraint braces, loop arrows, gates, and simple model insets. Use numbered callouts, legend markers, or object icons only when allowed by the selected policies and needed for readability.

Information density target: Maximize effective SCI information density through module hierarchy, interaction density, program-detail flow, branch/loop/update structure, compact grouping, record/batch/gate primitives, ports, and lanes. Do not simplify into a sparse poster-like diagram unless the user explicitly requests low density. Use modules and interactions as the density mechanism, not formulas or object artwork.

Canvas: [16:9 landscape / 4:3 landscape / wide multi-panel]. Use strict alignment, compact whitespace, and clear grouping. Avoid large empty regions unless they serve an explicit visual hierarchy.

Dense visual structure: [macro-region map and reading order]. Include slots for [major regions], [major modules], [submodules], [knowledge sources], [repositories/buffers], [data flow arrows], [control/update arrows], [branch points], [update loop], [feedback loop], [input/output blocks], and [bottom evidence/requirement strip]. Include [legend/caption strip] or [numbered callouts] only when allowed by the selected policies. Make every arrow evidence-supported.

Shape-first visual grammar: Use lanes, nested boxes, grouped regions, ports, braces, gates, record stacks, and arrows before using icons. Every element should be easy to redraw in Visio with basic shapes.

Labels: Use short English academic labels plus optional secondary micro-labels where readable: [labels]. Keep labels large enough for projection and Visio tracing. Avoid code identifiers and formula-like text.

Style: Flat publication-style vector diagram, Nature/Science-compatible accessible palette, color-blind safe, no red-green dependency, dark neutral text, high contrast, simple linework, low-saturation tints, and clear hierarchy.

Negative prompt: Avoid copied source-paper figure, source-diagram imitation, long caption reproduction, paper-specific layout imitation, copying the visual structure of reference images, source-specific terms not present in current evidence, photorealism, 3D, complex icons, decorative AI brain, server rack, drone/RIS/antenna icons unless unavoidable, standalone legend when legend-free, numbered callouts when no-numbered-callouts, separate board when main-only, tiny text, watermarks, logos, invented values, unsupported arrows, unsupported superiority or robustness claims, code identifiers, snake_case, camelCase, raw variable names, Greek-heavy formulas, exact equations, excessive empty whitespace, or deleting important supported elements for cleanliness.

Post-generation QA before selecting final asset: scan the generated draft for claim-risk labels or visuals such as `best`, `ideal`, `optimal`, `optimum`, `upper-envelope`, `superior`, `robust`, `generalizable`, `guarantee`, deployment readiness, online rerouting, route-switching resilience, physical Wh savings, ranking badges, significance marks, or trophy-like visuals. Reference objects such as oracle/envelope/baseline/diagnostic scaffold must be rendered as bounded references, not as best or optimal mechanisms. Reject or regenerate any draft that violates these rules, and record selected asset, rejected asset, rejection reason, claim-risk visual scan, board QA status or `not applicable`, canvas normalization, regeneration notes, and known manual-fix items.
