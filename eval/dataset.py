"""Load and validate the evaluation dataset."""

from __future__ import annotations

import json
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "questions.json")


class DatasetError(ValueError):
    pass


def load_dataset(path: str = DEFAULT_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    validate_dataset(data)
    return data


def validate_dataset(data: dict[str, Any]) -> None:
    """Fail loudly on a malformed dataset rather than silently scoring nothing."""
    if not isinstance(data, dict):
        raise DatasetError("dataset must be a JSON object")
    for section in ("answerable", "refusal"):
        if section not in data:
            raise DatasetError(f"missing section: {section}")
        if not isinstance(data[section], list):
            raise DatasetError(f"section {section} must be a list")

    seen: set[str] = set()
    for case in data["answerable"] + data["refusal"]:
        if "id" not in case or "question" not in case:
            raise DatasetError(f"case missing id/question: {case!r}")
        if case["id"] in seen:
            raise DatasetError(f"duplicate case id: {case['id']}")
        seen.add(case["id"])
        for entry in case.get("relevant") or []:
            if "doc_id" not in entry or "pages" not in entry:
                raise DatasetError(
                    f"case {case['id']}: 'relevant' entries need doc_id and pages"
                )
            if not all(isinstance(p, int) and p >= 1 for p in entry["pages"]):
                raise DatasetError(
                    f"case {case['id']}: pages must be 1-based integers"
                )

    for case in data["refusal"]:
        if case.get("expected_refusal") is not True:
            raise DatasetError(
                f"refusal case {case['id']} must set expected_refusal: true"
            )


def labelled_cases(data: dict[str, Any]) -> list[dict]:
    """Answerable cases that actually have ground truth."""
    return [c for c in data["answerable"] if c.get("relevant")]


def unlabelled_cases(data: dict[str, Any]) -> list[dict]:
    return [c for c in data["answerable"] if not c.get("relevant")]
