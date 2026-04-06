# Agent1

Agent1 is the first independent agent. It listens for newly created Jira tickets and triggers its handler for each new issue.

## Run

Set env vars:

- `AGENT1_SOURCE=jira`
- `JIRA_BASE_URL=https://<your-domain>.atlassian.net`
- `JIRA_EMAIL=<jira-user-email>`
- `JIRA_API_TOKEN=<jira-api-token>`
- `JIRA_PROJECT_KEY=<project-key>`
- `JIRA_JQL_EXTRA=<optional additional JQL filter>`
- `AGENT1_POLL_INTERVAL=30` (optional)
- `AGENT1_STATE_FILE=.agent_state/agent1_checkpoint.txt` (optional)

Run:

```powershell
.\venv\Scripts\python -m agent1.agent1
```

## Extension

To add another source later, implement `EventSource` in `agent1/source.py` and wire it in `build_source_from_env()` in `agent1/agent1.py`.
