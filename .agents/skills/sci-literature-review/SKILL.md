---
name: sci-literature-review
description: "Use for SCI-oriented literature review work: search strategy design, source triage, DOI/source verification, related-work synthesis, citation risk checks, and manuscript evidence alignment. Trigger when the user asks for literature review, related work, citation validation, bibliography triage, research gap analysis, or source-grounded positioning. Do not use for unsupported claim polishing or generic text beautification."
---

# SCI Literature Review

## Core Rules

- Treat literature review as evidence work, not paragraph generation.
- Use primary sources, official records, DOI pages, journal pages, PubMed, Crossref, Semantic Scholar, arXiv, or the user's supplied PDFs/metadata.
- Browse for literature, citation, DOI, journal, or current-source questions unless the user explicitly says not to.
- Never invent references, authors, years, venues, DOI values, sample sizes, datasets, or findings.
- For Chinese literature, provide CNKI/Wanfang/VIP search strategies and ask the user to supply records or PDFs if direct access is unavailable.
- When journal selection or journal fitness is in scope, read C:\Users\didgm\.codex\skills\.shared\manuscript-governance\journal-scope-first-gate.md and verify official aims-and-scope evidence before ranking journals.
- If repo governance files exist, read `knowledge/wiki/current_state.md` and `knowledge/frameworks/claim_gate.md` before drafting review claims.

## Workflow

1. Define review scope and, when a target journal is part of the task, lock its scope status before broad literature ranking: research question, domain, years, inclusion/exclusion criteria, and target claim.
2. Build search queries in English and, when useful, Chinese.
3. Search or inspect supplied sources. Record source, title, authors, year, venue, DOI/URL, and why it matters. Read `references/citation-candidate-export-boundary.md` when using Crossref/Zotero/export-style metadata or when preparing citation candidates for ENW/RIS/BibTeX-like handoff.
4. Triage papers by role: foundational, direct baseline, method family, dataset/task, limitation, or competing claim.
5. Synthesize by theme rather than listing papers one by one.
6. Mark every uncertain citation or unsupported claim as blocked.
7. Update or propose updates to `knowledge/frameworks/claim_gate.md` when the review changes what the paper can safely claim.

## Output Shape

Prefer this compact shape:

```md
## Search Scope
- Topic:
- Date range:
- Inclusion rules:
- Exclusion rules:

## Source Table
| Role | Citation | Source/DOI | Evidence Used | Risk |
| --- | --- | --- | --- | --- |

## Synthesis
[Theme-based synthesis grounded in verified sources.]

## Claim Impact
- Supported:
- Still blocked:
- Needs user/source verification:
```

## Safety Boundaries

- Do not run local scholar scripts or write bibliography files unless the user explicitly asks.
- Do not use broad web snippets as final citation evidence when a DOI, publisher, PubMed, arXiv, or user-supplied PDF is available.
- Do not upgrade a literature gap into novelty unless competing work and baseline coverage have been checked.
- Do not cite a paper from memory.

## References

- Search and citation rules: `references/search-and-citation-rules.md`
- Citation candidate/export boundary: `references/citation-candidate-export-boundary.md`
- Upstream attribution: `references/upstream-attribution.md`
