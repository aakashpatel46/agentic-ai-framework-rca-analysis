from __future__ import annotations

import json
from pathlib import Path

from agent2.agent2 import _build_source_from_env
from agent2.category_prompt_config import CategoryPromptConfig
from agent2.env_loader import load_env_file
from agent2.openai_categorizer import OpenAICategorizer
from agent2.processor import ValidationEnrichmentProcessor
from agent2.rules import RuleSet

# Hardcoded issue key for manual testing
ISSUE_KEY = "JMCH-1802"


def main() -> None:
    load_env_file(".env")

    source = _build_source_from_env()
    rules = RuleSet.from_file("agent2/rules.json")
    prompt_config = CategoryPromptConfig.from_file("agent2/category_prompts.txt")
    categorizer = OpenAICategorizer(rules)
    processor = ValidationEnrichmentProcessor(
        source=source,
        rules=rules,
        categorizer=categorizer,
        prompt_config=prompt_config,
    )

    attachment_root = f"agent2/attachments/{ISSUE_KEY}"
    result = processor.process_issue(issue_key=ISSUE_KEY, attachment_root=attachment_root)

    output_path = Path(f"agent2/output/{ISSUE_KEY}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = result.to_dict()
    output_payload["attachment_paths"] = [str(Path(path).resolve()) for path in result.attachment_paths]
    output_payload["output_json_path"] = str(output_path.resolve())
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"[dummy_tests] Agent2 processed issue: {ISSUE_KEY}")
    print(f"[dummy_tests] Output JSON: {output_path.resolve()}")
    print(f"[dummy_tests] Attachments downloaded: {len(result.attachment_paths)}")
    for attachment_path in output_payload["attachment_paths"]:
        print(f"[dummy_tests] Attachment path: {attachment_path}")
    print(f"[dummy_tests] Secondary OpenAI called: {result.secondary_openai_called}")
    if result.missing_information:
        print(f"[dummy_tests] Missing information: {', '.join(result.missing_information)}")
    else:
        print("[dummy_tests] All required information is available")


if __name__ == "__main__":
    main()
