from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_reporting.agent5 import Agent5RiskAnalyzer, load_env_file


def main() -> None:
    load_env_file(".env")

    agent4_output_path = Path("rca_synthesis/output/JMCH-DUMMY-3001.json")
    if not agent4_output_path.exists():
        agent4_output_path = Path("dummy_tests/output/agent4_dummy_output.json")
    if not agent4_output_path.exists():
        raise FileNotFoundError(
            "Missing Agent4 output file. Run agent4 first to generate one under "
            "'rca_synthesis/output/' or 'dummy_tests/output/'."
        )

    agent4_output = json.loads(agent4_output_path.read_text(encoding="utf-8"))
    analyzer = Agent5RiskAnalyzer.from_env()
    result = analyzer.analyze(agent4_output=agent4_output)

    output_path = Path("dummy_tests/output/agent5_dummy_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[dummy_tests] Agent5 run complete")
    print(f"[dummy_tests] Source Agent4 JSON: {agent4_output_path.resolve()}")
    print(f"[dummy_tests] Output JSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()


