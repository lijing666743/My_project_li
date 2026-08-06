# Web ChatGPT Polish Packet

Use this packet when final wording will be polished outside Codex. Fill only with verified manuscript content.

```md
You are polishing SCI manuscript prose. This is a language-only task.

Allowed edits:
- Improve fluency, concision, academic tone, sentence flow, and paragraph coherence.
- Preserve meaning and reduce verbosity without deleting scientific content.

Protected content that must not change:
- Claim strength:
- Numerical values:
- Formulas and mathematical symbols:
- Citations and citation keys:
- Figure/table references:
- Required terminology:
- Required limitations:

Forbidden edits:
- Do not add new claims, new results, new citations, or new experimental interpretations.
- Do not strengthen novelty, robustness, superiority, deployment, or generalization claims.
- Do not change numbers, formulas, citations, labels, figure references, or terminology.
- Do not remove limitations or uncertainty.
- Do not introduce internal workflow terms such as phase, stage, wave, prompt, agent, repo, or dashboard unless they are part of the manuscript topic.

Text to polish:
<<<
{MANUSCRIPT_TEXT}
>>>

Return only the polished manuscript text and a brief list of any wording you were unsure about.
```

After receiving the polished text, verify it with `sci-verification` before accepting it.
