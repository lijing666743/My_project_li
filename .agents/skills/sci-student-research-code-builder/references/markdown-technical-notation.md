# Markdown Technical Notation

Use this rule when writing or reviewing Markdown.

## Core Rule

- Use backticks for code identifiers: file names, paths, functions, classes, parameter names, config keys, command snippets, CSV columns, and literal code values.
- Use LaTeX math delimiters for mathematical symbols and formulas: `$...$` for inline math and `$$...$$` for display math.

## Examples

- Code identifier: `epsilon`, `alpha`, `loss`, `main.py`, `src/config.py`
- Mathematical symbol: $\epsilon$, $\alpha$, $\mathcal{L}$, $Q(s,a)$
- Mixed concept: code variable `epsilon` implements exploration rate $\epsilon$, with default $\epsilon=0.1$.

## Avoid

- Do not write `` `\epsilon` ``.
- Do not write `` `Q(s,a)` `` when it is a mathematical function.
- Do not write `` `epsilon=0.1` `` when the expression is a mathematical statement; write $\epsilon=0.1$.
- Do not edit fenced code blocks to satisfy prose notation rules.
