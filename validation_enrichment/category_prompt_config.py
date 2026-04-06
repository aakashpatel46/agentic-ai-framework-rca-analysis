from __future__ import annotations

import json
from pathlib import Path


class CategoryPromptConfig:
    def __init__(self, default_secondary_prompt: str, category_secondary_prompts: dict[str, str] | None = None) -> None:
        self.default_secondary_prompt = default_secondary_prompt
        self.category_secondary_prompts = category_secondary_prompts or {}

    @classmethod
    def from_file(cls, path: str) -> "CategoryPromptConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        default_prompt = str(payload.get("default_secondary_prompt", "")).strip()
        category_prompts_raw = payload.get("category_secondary_prompts", {})
        if not isinstance(category_prompts_raw, dict):
            raise ValueError("category_secondary_prompts must be an object")

        category_prompts: dict[str, str] = {}
        for category, prompt in category_prompts_raw.items():
            category_prompts[str(category)] = str(prompt).strip()

        return cls(default_secondary_prompt=default_prompt, category_secondary_prompts=category_prompts)

    def secondary_prompt_for(self, category: str) -> str:
        _ = category
        return self.default_secondary_prompt
