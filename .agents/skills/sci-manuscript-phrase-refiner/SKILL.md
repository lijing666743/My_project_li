---
name: sci-manuscript-phrase-refiner
description: Refine SCI manuscript wording after claim and polishing gates permit rewriting. Use for IEEE/top-journal phrase refinement, section-function phrasing, equation-adjacent prose refinement, semantic-preserving academic English, engineering-residue cleanup, phase/stage/pipeline/texttt cleanup, redundancy compression, contribution/result phrasing, and preserving core scientific content while improving scholarly register. Do not use for claim approval, evidence invention, broad unsupported rewriting, LaTeX integration, or adding/changing equations/algorithms/theorems; route those to manuscript gate, formalization, LaTeX, or verification skills.
---

# SCI Manuscript Phrase Refiner

## Purpose

Rewrite manuscript text into concise, professional, top-journal SCI English while preserving meaning, evidence boundaries, citations, formulas, numbers, comparators, limitations, and section function.

This skill owns phrase-level manuscript polishing, including IEEE-style rhetorical patterns for problem framing, challenge framing, contribution phrasing, problem-formulation prose, algorithm-transition prose, benchmark comparison, simulation-result interpretation, and limitation wording.

This skill also owns prose-only refinement around equations, algorithms, and theorem-style blocks after `sci-manuscript-formalization-augmenter` has accepted the formal content. It may improve motivation, transition, interpretation, and claim-boundary wording, but it must not change formulas, labels, symbols, algorithm logic, or theorem content.

Treat engineering residue as a submission risk unless it is a legitimate technical term in context.

## Required Workflow

1. Confirm rewrite permission and target-journal scope status:
   - If the target journal is explicitly out-of-scope, stop journal-targeted polishing and report a FATAL gate.
   - If scope is unverified, do not grant submission-readiness or desk-rejection-proof wording; route to scope verification.
   - Read C:\Users\didgm\.codex\skills\.shared\manuscript-governance\journal-scope-first-gate.md when a target journal is specified.
   - If broad gate status is unknown, route first to `supervisory-polishing-gate` or `journal-manuscript-gatekeeper`.
   - If the user explicitly asks for a tiny language-only cleanup on already gated text, proceed and state the limited scope.
2. Mandatory boundary read: on every invocation, read `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\README.md` and `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\source-boundary.md` before selecting any language-bank module. The shared bank guides rhetorical, grammar, section-function, claim-strength, and engineering-residue wording; it is not evidence and does not approve claims.
3. Extract non-negotiable content:
   - Claims, variables, equations, numerical values, comparator names, citations, limitations, scope boundaries, and figure/table references.
4. Diagnose the section function:
   - Identify whether the text is abstract, introduction, related work, contribution list, system model, problem formulation, method transition, algorithm explanation, result interpretation, discussion, limitation, or conclusion.
   - Read shared bank files `section-function-patterns.md`, `grammar-patterns.md`, `claim-strength-lexicon.md`, and domain modules such as `domain-ieee-uav-isac-mec-rl.md`, `domain-ieee-wireless-optimization.md`, or `domain-isac-ris-sagin.md` when the section or domain matches.
   - When formulas, algorithms, or theorem-style blocks are adjacent to the target prose, read `references/equation-adjacent-phrase-polishing.md` and shared bank file `equation-context-phrasing.md`; refine only the prose that motivates, links, interprets, or bounds the formal block.
   - If phrase refinement exposes a metric, objective, loss, reward, constraint, repeated procedure, theorem-like property, complexity claim, or formal protocol that should be formalized, stop local smoothing for that span and hand it to `sci-manuscript-formalization-augmenter`; do not hide under-formalization with fluent prose.
   - For broad top-journal vocabulary coverage, read `zotero-derived-domain-vocabulary.md`, `zotero-derived-high-frequency-lexicon.md`, `zotero-derived-academic-moves.md`, `zotero-derived-sentence-skeletons.md`, `zotero-derived-grammar-patterns.md`, and `zotero-derived-verb-object-patterns.md`.
   - Read `zotero-corpus-index.md` only to understand corpus provenance, collection coverage, and metadata limitations; never use it as evidence.
   - Read `references/ieee-uav-phrase-patterns.md` only as the local compatibility wrapper for the shared IEEE UAV/AAV/ISAC/MEC/RL language-bank module.
   - Read `references/nature-phrase-adaptation-boundary.md` only when Nature-family phrasing is explicitly requested and `sci-nature-manuscript-writer` has already locked structure, story, allocation, and claim strength.
5. Check engineering residue:
   - Read `references/engineering-residue-policy.md`.
   - Read shared bank file `engineering-residue-replacements.md` when manuscript-facing replacements need cross-project consistency.
   - Flag internal phase/stage/wave labels, repo artifacts, prompt/agent wording, dubious `pipeline`, and suspicious `\texttt{}` usage.
6. Apply academic compression:
   - Read `references/academic-compression-contract.md`.
   - Remove redundant, promotional, procedural, or meta-writing without deleting scientific meaning.
7. Calibrate claim strength and phrase register:
   - Read `references/claim-strength-phrase-policy.md`.
   - Read shared bank file `claim-strength-lexicon.md` when claim-strength wording must align across projects.
   - Prefer measured verbs and benchmark-grounded phrasing.
   - Do not introduce robustness, optimality, superiority, deployment, generalization, or novelty claims not already permitted by the gate.
8. Preserve semantics:
   - Read `references/semantic-preservation-checklist.md`.
   - Do not change claim strength unless the user requested safe downgrading or the gate requires it.
9. If the final language pass will be done in web ChatGPT or another external UI:
   - Use `templates/web_polish_packet.md`.
   - Produce a controlled prompt with protected claims, numbers, formulas, citations, figure references, terminology, and forbidden edits.
   - Require the returned polished text to go through `sci-verification` before acceptance.
10. Return a phrase refinement report:
   - Use `templates/refinement_report.md`.
   - Include phrase risks, section-function diagnosis, engineering residue replacements, protected content, and verification needs.

## Output Contract

Every substantive rewrite must include:

- phrase risks and section-function diagnosis
- shared language-bank boundary read
- selected shared language-bank modules, or `none selected`
- protected claims, numbers, formulas, citations, comparators, and limitations
- engineering residue replacement table
- equation-adjacent changes when formulas, algorithms, or theorem-style blocks are in scope
- formalization handback candidates, or `none`
- compressed or deleted non-core wording
- refined manuscript text
- claim-strength changes if any
- unchanged items with reasons
- verification needs
- external web-polish packet when requested or when the workflow explicitly hands final wording to web ChatGPT

## Guardrails

- Do not invent evidence, citations, experiments, datasets, metrics, or journal-specific claims.
- Do not delete core content to make the paragraph shorter.
- Do not hide uncertainty, limitations, or weak evidence behind fluent prose.
- Do not turn implementation workflow labels into scientific method names.
- Do not rewrite formulas, citation keys, labels, or numbers unless the user explicitly asks and verification is possible.
- Do not add equations, algorithms, definitions, assumptions, propositions, lemmas, theorems, or proof sketches; route those to `sci-manuscript-formalization-augmenter`.
- Do not smooth over under-formalized method prose; if a sentence defines a metric, objective, constraint, protocol, repeated procedure, theorem-like property, or complexity claim, report a formalization handback candidate.
- Do not alter algorithm logic, constraints, variable definitions, or mathematical symbols while polishing surrounding prose.
- Do not hide a formula-stack problem with fluent prose; if the formal block itself needs merging, moving, downgrading, or rejection, hand it back to `sci-manuscript-formalization-augmenter`.
- Do not use `\texttt{}` for internal workflow remnants in final manuscript prose.
- Do not treat external web polishing as evidence review; it is a language-only step and must be re-verified on return.
- Do not copy long sentences from source papers; use corpus-derived rhetorical patterns only.

## References

- Journal scope-first gate: C:\Users\didgm\.codex\skills\.shared\manuscript-governance\journal-scope-first-gate.md

- Academic compression: `references/academic-compression-contract.md`
- Equation-adjacent phrase polishing: `references/equation-adjacent-phrase-polishing.md`
- Engineering residue: `references/engineering-residue-policy.md`
- IEEE/top-journal phrase patterns: `references/ieee-uav-phrase-patterns.md`
- Nature-family phrase adaptation boundary: `references/nature-phrase-adaptation-boundary.md`
- Shared SCI language bank mandatory boundary read: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\README.md`
- Shared SCI language bank source boundary: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\source-boundary.md`
- Shared equation-context phrasing: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\equation-context-phrasing.md`
- Zotero corpus index: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\zotero-corpus-index.md`
- Zotero-derived high-frequency lexicon: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\zotero-derived-high-frequency-lexicon.md`
- Zotero-derived sentence skeletons: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\zotero-derived-sentence-skeletons.md`
- Zotero-derived grammar patterns: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\zotero-derived-grammar-patterns.md`
- Zotero-derived verb-object patterns: `C:\Users\didgm\.codex\skills\.shared\sci-manuscript-language-bank\zotero-derived-verb-object-patterns.md`
- Claim-strength phrasing: `references/claim-strength-phrase-policy.md`
- Corpus extraction boundary: `references/corpus-extraction-boundary.md`
- Semantic preservation: `references/semantic-preservation-checklist.md`
- Output template: `templates/refinement_report.md`
- Web polish packet template: `templates/web_polish_packet.md`
