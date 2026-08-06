# NotebookLM Fallback Prompt

Create a high-information-density scientific diagram draft for manual Visio redraw about [topic].

Objective: [one-sentence purpose].

Evidence boundary: Use only the uploaded sources. Show only these supported modules, submodules, program stages, inputs, outputs, data/control flows, branches, loops, feedback relations, interfaces, and academic process steps: [facts]. Do not add unsupported results, causal claims, numerical values, statistical significance, performance superiority, journal branding, implementation variables, or code identifiers.

Illustration style boundary: Use only abstract top-journal visual grammar from the Image2 illustration style bank. Do not copy any source-paper figure, caption, layout, values, formulas, or conclusions.

Required dense content slots:
- Must-show modules/submodules: [must-show]
- Program-detail steps rewritten as academic labels: [program-detail inventory]
- Data/control flows, branches, loops, and feedback relations: [arrows]
- Inputs, outputs, interfaces, ports, and state transitions: [interfaces]
- Legend and callout policy: [legend-free / compact legend allowed; no-numbered-callouts / numbered callouts allowed]. If legend-free or no-numbered-callouts is selected, use local path labels, macro-region titles, flow direction, grouping boxes, lanes, ports, and section bands instead of standalone legends or numbered badges.

Engineering-residue suppression: Do not show code variable names, file names, function names, class names, snake_case, camelCase, CLI/config keys, or internal pipeline labels. Convert them to concise academic labels.

Math-light policy: Do not render dense formulas, Greek-heavy equations, matrices, summations, or exact LaTeX. If a mathematical idea is essential, show only short placeholders such as `Objective term`, `Update step`, `Constraint check`, or `Policy evaluation`.

Canvas and layout: Use [ratio/orientation]. Organize the diagram into [number] dense but readable lanes or panels: [module list]. Use strict alignment, compact whitespace, and a clear reading order.

Visual structure: Use redrawable primitive shapes, arrows, lanes, brackets, grouping boxes, callouts, ports, and small insets. Avoid complex icons, photorealism, 3D, decorative clutter, and tiny unreadable labels.

Do not simplify away: [critical modules, interactions, branches, loops, and feedback relations that must remain visible].

Required labels: [short academic labels].

Style and palette: Publication-style, accessible color-blind safe palette, avoid red-green dependency, dark neutral text preferred, color used for boxes and arrows.

Final QA: Check every label, arrow, relation, and branch against the sources. Keep high effective information density through module and interaction structure while preserving manual Visio redrawability.
