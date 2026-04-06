Agent1:
-> Triggered when Jira ticket is created
-> Gives Jira ticket as output.

Agent2: 
-> Takes Jira ticket number as input
-> Downloads raw information and attachments
-> Search through information and attachments
-> Output : Missing Inforamtion/ (Metadata + attachments)
-> No triage done by the agent






Points : 
Version may be in description/ custom field
What if we got missing information that may not be needed for triaging.










Agent2 Flow (Input -> Output)

Input received
Agent2 takes issue_key from:
CLI: --issue-key <KEY> or
env: AGENT2_INPUT_ISSUE_KEY
Loads config from .env.
Source setup
Reads AGENT2_SOURCE (currently jira).
Builds Jira client using:
JIRA_BASE_URL
JIRA_EMAIL
JIRA_API_TOKEN
Ticket fetch
Fetches full Jira issue for the given key.
Also reads custom fields if configured:
JIRA_STEPS_TO_REPRO_FIELD
JIRA_AFFECTS_VERSION_FIELD
Attachment download
Downloads all ticket attachments to:
agent2/attachments/<issue_key>/ (default)
or AGENT2_ATTACHMENT_ROOT if set.
Collects absolute file paths.
First OpenAI call (categorization)
Categorizes ticket into one of categories from agent2/rules.json.
If OPENAI_API_KEY missing, falls back to heuristic category logic.
Rule-based validation
Loads required fields for chosen category from agent2/rules.json.
Checks availability using:
ticket fields (summary, description, steps custom field, version custom field, etc.)
attachment names (for logs, etc.)
Produces initial missing_information.
Second OpenAI call (only if missing exists)
Triggered only when missing_information is non-empty.
Uses universal secondary prompt from agent2/category_prompts.txt.
Sends only:
raw_ticket
attachment_names
missing_information
Purpose: double-check whether any “missing” item is actually present.
Build final output
Creates result object with:
category
required info
available info
missing info
attachment paths
secondary call details (secondary_openai_called, secondary_enrichment)
raw ticket details
Persist output
Writes JSON to:
agent2/output/<issue_key>.json (default)
or AGENT2_OUTPUT_FILE if set.
Console output
Prints:
processed issue key
output JSON path
attachment paths
category
whether secondary OpenAI was called
missing info summary










Documentation : 
Easy updates to existing index
incremental indexing