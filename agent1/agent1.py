from __future__ import annotations

import os
import subprocess
import sys

from agent1.env_loader import load_env_file
from agent1.event_listener import EventListener, build_checkpoint_store_from_env
from agent1.jira_source import JiraSource
from agent1.models import TicketCreatedEvent


def handle_ticket_created(event: TicketCreatedEvent) -> None:
    print(
        f"[agent1] New ticket detected: {event.ticket_key} "
        f"(project={event.project_key}) summary={event.summary}"
    )
    if os.getenv("AGENT1_TRIGGER_AGENT2", "false").strip().lower() == "true":
        print(f"[agent1] Triggering agent2 for issue: {event.ticket_key}")
        subprocess.run(
            [sys.executable, "-m", "agent2.agent2", "--issue-key", event.ticket_key],
            check=False,
        )


def build_source_from_env():
    source_type = os.getenv("AGENT1_SOURCE", "jira").strip().lower()
    if source_type == "jira":
        return JiraSource.from_env()

    raise ValueError(f"Unsupported AGENT1_SOURCE '{source_type}'. Supported: jira")


def main() -> None:
    load_env_file(".env")
    source = build_source_from_env()
    listener = EventListener(
        source=source,
        on_event=handle_ticket_created,
        checkpoint_store=build_checkpoint_store_from_env(),
        poll_interval_seconds=int(os.getenv("AGENT1_POLL_INTERVAL", "30")),
    )
    listener.run_forever()


if __name__ == "__main__":
    main()
