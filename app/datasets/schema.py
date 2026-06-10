"""
Task dataclass + JSONL loader.

Why a dataclass:
- The original JSONL row is a raw dict, which has no schema, no defaults, no
  validation, and no metadata slot. A Task abstraction lets callers:
    * type-check fields (id is str, not int; expected_answer_keywords is a list[str])
    * carry run-time config (max_steps, timeout_seconds) the runner should respect
    * carry grading hints (difficulty, category) for downstream analytics
    * carry opaque metadata (source URL, license, sample_index) without polluting
      the core fields

This file is intentionally a leaf: no FastAPI / SQLAlchemy / LangGraph dependency,
so it can be imported from anywhere (CLI, evaluator, runner, future web UI).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Allowed values for task_type / expected_tool. Kept here (not in models.py) so the
# schema module has zero coupling to the FastAPI app.
TASK_TYPES = {"tool_calling", "calculator", "rag_qa", "other"}
TOOL_NAMES = {"weather_tool", "calculator_tool", "search_tool"}


@dataclass
class Task:
    """A single benchmark case, parsed from JSONL and enriched with run-time config."""

    task_id: str
    query: str
    task_type: str = "other"
    description: str = ""
    expected_tool: str = ""
    expected_answer_keywords: list[str] = field(default_factory=list)
    # Run-time config (per-case overrides)
    max_steps: int = 5
    timeout_seconds: float = 30.0
    # Grading / analytics hints
    difficulty: str = "medium"  # easy / medium / hard
    category: str = ""          # e.g. "math" / "weather" / "lookup"
    # Opaque key-value bag for source attribution, sample index, license, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Soft validation: don't raise, but flag in metadata for downstream use.
        if self.task_type not in TASK_TYPES:
            self.metadata.setdefault("unexpected_task_type", self.task_type)
        if self.expected_tool and self.expected_tool not in TOOL_NAMES:
            self.metadata.setdefault("unexpected_tool_name", self.expected_tool)

    # ------------------------------------------------------------------
    # Legacy compatibility: the existing runner / API still operate on dicts.
    # These helpers let callers go Task -> dict and dict -> Task without losing
    # the new fields (they're round-tripped through the JSONL "metadata" key).
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.task_id,
            "task_type": self.task_type,
            "query": self.query,
            "expected_tool": self.expected_tool,
            "expected_answer_keywords": list(self.expected_answer_keywords),
        }
        if self.description:
            d["description"] = self.description
        if self.max_steps != 5:
            d["max_steps"] = self.max_steps
        if self.timeout_seconds != 30.0:
            d["timeout_seconds"] = self.timeout_seconds
        if self.difficulty != "medium":
            d["difficulty"] = self.difficulty
        if self.category:
            d["category"] = self.category
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


def task_from_raw(raw: dict) -> Task:
    """Coerce a JSONL row into a Task. Missing fields take defaults; extras go to metadata."""
    if "id" not in raw and "task_id" not in raw:
        raise ValueError(f"Task row missing 'id' / 'task_id': {raw!r}")
    known = {
        "id", "task_id", "query", "task_type", "description",
        "expected_tool", "expected_answer_keywords",
        "max_steps", "timeout_seconds", "difficulty", "category", "metadata",
    }
    extras = {k: v for k, v in raw.items() if k not in known}
    return Task(
        task_id=raw.get("id") or raw["task_id"],
        query=raw["query"],
        task_type=raw.get("task_type", "other"),
        description=raw.get("description", ""),
        expected_tool=raw.get("expected_tool", ""),
        expected_answer_keywords=list(raw.get("expected_answer_keywords", []) or []),
        max_steps=int(raw.get("max_steps", 5)),
        timeout_seconds=float(raw.get("timeout_seconds", 30.0)),
        difficulty=raw.get("difficulty", "medium"),
        category=raw.get("category", ""),
        metadata={**extras, **(raw.get("metadata") or {})},
    )


def load_tasks(jsonl_path: str | Path) -> list[Task]:
    """Read a JSONL file and return a list of Tasks. Blank lines are skipped."""
    p = Path(jsonl_path)
    out: list[Task] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{i} is not valid JSON: {e}") from e
        out.append(task_from_raw(row))
    return out


def dump_tasks(tasks: Iterable[Task], jsonl_path: str | Path) -> int:
    """Write a list of Tasks back to a JSONL file. Returns the number of rows written."""
    p = Path(jsonl_path)
    rows = [json.dumps(t.to_dict(), ensure_ascii=False) for t in tasks]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


__all__ = ["Task", "TASK_TYPES", "TOOL_NAMES", "task_from_raw", "load_tasks", "dump_tasks"]
