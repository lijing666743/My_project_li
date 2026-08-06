# Minimal Shape/Container Board Prompt

Use this template only when output mode is `main-plus-board`. Do not use it when the user asks for `main-only`, no icon board, no shape board, no container board, or when primitive shapes in the main draft are sufficient and a companion board would add process clutter.

Use case: diagram-companion-minimal-shape-container-board
Asset type: pure-white-background PNG board for manual Visio splitting and redraw. The board must use a white background, not transparency.

Create one PNG board on a solid pure white `#FFFFFF` background containing the minimal reusable shape, container, and relation primitives needed by this SCI logic-coupled diagram draft: [minimal shape/container inventory].

Default content: Include primitive redraw elements first: arrow styles, lane markers, rounded module blocks, port markers, brace markers, numbered callout circles, grouping boxes, dashed macro-region frames, database cylinders, repository blocks, vertical state ribbons, evaluation ovals, constraint braces, decision diamonds, thick loop arrows, thin data arrows, dashed diagnostic arrows, feedback-loop arrows, and legend markers. If the inventory says `No complex reusable icons required`, draw only the reusable shape/container primitives and relation markers.

Model inset exception: If a neural/model inset is necessary, use only a simplified 3-layer mini-network made of a few circles and thin lines. Do not draw a complex decorative neural network.

Object icon exception: Include object icons only when the main diagram cannot express the concept with primitive shapes or containers. If included, keep them minimal, flat, and easy to trace.

Zotero-derived style boundary: Align with the selected diagram archetype and palette family, but do not copy any source-paper icon, source figure, caption, layout, value, formula, or conclusion.

Layout: Arrange items in a clean grid or loose array. Items must not overlap. Leave large padding around every item so the user can manually crop or trace them in Visio.

Style: Flat vector-like scientific shape primitives, simple redrawable geometry, consistent line weight, consistent corner radius, consistent perspective, Nature/Science-compatible accessible palette, color-blind safe, high contrast.

Content rules: Include only items from the minimal shape/container inventory. Repeated identical primitives appear once. If line style, direction, role, state, container type, or color semantics differ, draw them as separate variants.

Background: solid pure white `#FFFFFF`. Do not draw checkerboard pattern, gray-white grid, simulated transparency, transparency preview, paper texture, or background panel.

Avoid: text labels, formulas, titles, legends, captions, callout text, copied source-paper icons, source-figure imitation, source-diagram imitation, watermarks, logos, photorealism, 3D, AI brain icons, server racks, drone/RIS/antenna icons unless unavoidable, shadows, gradients, decorative textures, tiny strokes, overlapping objects, merged groups, or unsupported icons.
