from __future__ import annotations

import json
from pathlib import Path


class RuleSet:
    def __init__(self, categories: dict[str, list[str]]) -> None:
        self.categories = categories

    @classmethod
    def from_file(cls, path: str) -> "RuleSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        categories = payload.get("categories", {})
        if not isinstance(categories, dict) or not categories:
            raise ValueError("rules file must contain a non-empty 'categories' object")

        normalized: dict[str, list[str]] = {}
        for category, requirements in categories.items():
            if not isinstance(requirements, list):
                raise ValueError(f"Invalid requirements list for category '{category}'")
            normalized[str(category)] = [str(item).strip() for item in requirements if str(item).strip()]
        return cls(categories=normalized)

    def requirements_for(self, category: str) -> list[str]:
        if category in self.categories:
            return self.categories[category]
        return self.categories.get("other", [])

    def category_names(self) -> list[str]:
        return list(self.categories.keys())
