from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rca_synthesis.agent4 import Agent4RcaSynthesizer, load_env_file


def _read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    load_env_file(".env")

    ticket_path = Path("dummy_tests/current_ticket.json")
    agent3_output_path = Path("multi_source_analysis/output/JMCH-DUMMY-3001.json")
    if not agent3_output_path.exists():
        agent3_output_path = Path("dummy_tests/output/agent3_dummy_output.json")

    if not ticket_path.exists():
        raise FileNotFoundError(f"Missing ticket input file: {ticket_path}")
    if not agent3_output_path.exists():
        raise FileNotFoundError(
            "Missing Agent3 output file. Run agent3 first to generate one under "
            "'multi_source_analysis/output/' or 'dummy_tests/output/'."
        )

    ticket = _read_json_file(ticket_path)
    agent3_output = _read_json_file(agent3_output_path)

    synthesizer = Agent4RcaSynthesizer.from_env()
    detailed = synthesizer.synthesize(ticket=ticket, agent3_output=agent3_output)
    result = {
        "agent": "agent4",
        "input_ticket": ticket,
        "agent3_output_file": str(agent3_output_path.resolve()),
        "detailed_rca_and_fix": detailed,
    }

    output_path = Path("dummy_tests/output/agent4_dummy_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[dummy_tests] Agent4 run complete")
    print(f"[dummy_tests] Output JSON: {output_path.resolve()}")
    print(f"[dummy_tests] Source Agent3 JSON: {agent3_output_path.resolve()}")


if __name__ == "__main__":
    main()


