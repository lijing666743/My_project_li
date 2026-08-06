# External Adaptation Notes

This skill supersedes the previous local `sci-notebooklm-diagram-prompt` skill. The workflow is now image2-first, with NotebookLM as fallback only.

## Sources

- `baoyu-diagram` source repository: https://github.com/JimLiu/baoyu-skills
- Previous local adaptation: `sci-notebooklm-diagram-prompt`

## Adaptation Policy

- Do not run or install the original third-party skill.
- Preserve the idea of high-quality scientific diagram prompting, but use Codex's local repo/manuscript context and imagegen first.
- Keep NotebookLM fallback prompts source-grounded and include `Sources to upload`.

