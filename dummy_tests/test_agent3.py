from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_source_analysis.agent3 import Agent3Analyzer, load_env_file


def main() -> None:
    load_env_file(".env")

    # Dummy current ticket payload for manual Agent3 validation.
    current_ticket = {
        "Issue key": "JMCH-DUMMY-3001",
        "Summary": "POS terminal crashes during refund flow for guest checkout orders",
        "Description": (
            "Store reports intermittent crash while processing refunds for guest checkout transactions. "
            "Issue appears after selecting original receipt and submitting refund."
        ),
        "Priority": "High",
        "Issue Type": "Bug",
        "Components": ["point-of-sale", "refund-service"],
        "Labels": ["refund", "guest-checkout", "production"],
        "Organizations": ["Retail Ops"],
        "Environment": "Production",
        "Affects Version/s": ["250.1"],
        "Steps to Reproduce": [
            "Open POS and search for a guest checkout receipt",
            "Click refund and confirm amount",
            "Observe crash after confirmation",
        ],
        "Error Message": "NullPointerException in refund handler",
        "Comments": [
            "Issue started after weekend deployment.",
            "Does not reproduce for registered users.",
        ],
    }

    analyzer = Agent3Analyzer.from_env()
    result = analyzer.analyze_ticket(current_ticket)

    output_path = Path("dummy_tests/output/agent3_dummy_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[dummy_tests] Agent3 run complete")
    print(f"[dummy_tests] Output JSON: {output_path.resolve()}")
    print(f"[dummy_tests] Similar tickets: {len(result['similar_tickets_top3'])}")
    print(f"[dummy_tests] Documentation matches: {len(result['documentation_related'])}")


if __name__ == "__main__":
    main()

