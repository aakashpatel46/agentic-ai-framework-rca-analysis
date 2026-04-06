# Agent2

Agent2 is an isolated validation and enrichment agent.

Input:
- Issue key from Agent1 (or manual CLI) via `--issue-key` or `AGENT2_INPUT_ISSUE_KEY`

Behavior:
- Fetches full ticket details from source (`jira` currently)
- Downloads attachments under `validation_enrichment/attachments/<issue-key>/`
- Categorizes ticket using OpenAI (first call; falls back to heuristic if API key is missing)
- Applies category-based required-information rules (`validation_enrichment/rules.json`)
- If missing information exists, runs a second OpenAI call using category prompts (`validation_enrichment/category_prompts.txt`)
- Outputs enriched JSON result to `validation_enrichment/output/<issue-key>.json`

Run:

```powershell
.\venv\Scripts\python -m validation_enrichment.agent2 --issue-key ABC-123
```

