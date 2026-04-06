# Agent2

Agent2 is an isolated validation and enrichment agent.

Input:
- Issue key from Agent1 (or manual CLI) via `--issue-key` or `AGENT2_INPUT_ISSUE_KEY`

Behavior:
- Fetches full ticket details from source (`jira` currently)
- Downloads attachments under `agent2/attachments/<issue-key>/`
- Categorizes ticket using OpenAI (first call; falls back to heuristic if API key is missing)
- Applies category-based required-information rules (`agent2/rules.json`)
- If missing information exists, runs a second OpenAI call using category prompts (`agent2/category_prompts.txt`)
- Outputs enriched JSON result to `agent2/output/<issue-key>.json`

Run:

```powershell
.\venv\Scripts\python -m agent2.agent2 --issue-key ABC-123
```
