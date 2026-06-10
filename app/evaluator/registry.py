"""
Grader plugin registry.

Why a registry:
- The original code hard-codes two judges: keyword_judge and llm_judge (in judge.py).
  Adding a new grader (e.g. numeric tolerance, regex, embedding similarity, LLM rubric)
  meant editing the runner's grading code. The registry pattern decouples the set of
  available graders from the call site.
- Registry design:
    * Grader subclasses implement .name, .score(answer, task, trace) -> float
      and optionally .explain(answer, task, trace) -> str for human-readable output.
    * The @register(name) decorator adds the grader to a process-global dict.
    * get(name) returns an instance; list_graders() returns registered names.

Built-in graders cover the four most common flavors:
    * ExactMatch        — case-insensitive, whitespace-stripped whole-string equality
    * KeywordMatch      — at least `min_ratio` of expected keywords appear
    * NumericTolerance  — extract numbers, return 1.0 if any expected number is close
    * LLMJudge          — placeholder, falls back to KeywordMatch until a real LLM is wired
    * Efficiency        — grades on latency vs. timeout (faster -> higher)
    * ToolAccuracy      — 1.0 if expected_tool in actual_tools, else 0.0
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional


# ---- Registry core --------------------------------------------------------------

_REGISTRY: dict[str, type["Grader"]] = {}


def register(name: str) -> Callable[[type["Grader"]], type["Grader"]]:
    """Class decorator. Register a Grader subclass under the given name."""
    def deco(cls: type["Grader"]) -> type["Grader"]:
        if name in _REGISTRY:
            raise ValueError(f"Grader '{name}' already registered by {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        cls.name = name  # type: ignore[assignment]
        return cls
    return deco


def get(name: str) -> "Grader":
    """Instantiate a registered grader by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown grader '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_graders() -> list[str]:
    """Return the names of all registered graders (insertion order)."""
    return list(_REGISTRY.keys())


# ---- Grader base ---------------------------------------------------------------

@dataclass
class Grader:
    """Base class. Subclasses must set .name and implement .score."""

    name: ClassVar[str] = ""

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        """Return a score in [0.0, 1.0]. 1.0 = perfect, 0.0 = wrong."""
        raise NotImplementedError

    def explain(self, answer: str, task: Any, trace: Any = None) -> str:
        """Optional human-readable explanation, for diagnostic reports."""
        return f"{self.name}: score={self.score(answer, task, trace):.3f}"


# ---- Built-in graders ----------------------------------------------------------

@register("exact_match")
class ExactMatchGrader(Grader):
    """Case-insensitive, whitespace-stripped whole-string equality."""

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        expected = task.expected_answer_keywords or [""]
        target = expected[0]
        return 1.0 if (answer or "").strip().lower() == (target or "").strip().lower() else 0.0

    def explain(self, answer: str, task: Any, trace: Any = None) -> str:
        s = self.score(answer, task, trace)
        return f"exact_match: expected='{task.expected_answer_keywords}' got='{(answer or '')[:80]}' -> {s}"


@register("keyword_match")
class KeywordMatchGrader(Grader):
    """At least `min_ratio` of the expected keywords must appear in the answer (case-insensitive)."""

    min_ratio: float = 0.5

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        kws = task.expected_answer_keywords or []
        if not kws:
            return 0.0
        ans_l = (answer or "").lower()
        hits = sum(1 for k in kws if (k or "").lower() in ans_l)
        ratio = hits / len(kws)
        return 1.0 if ratio >= self.min_ratio else ratio

    def explain(self, answer: str, task: Any, trace: Any = None) -> str:
        kws = task.expected_answer_keywords or []
        ans_l = (answer or "").lower()
        hits = [k for k in kws if (k or "").lower() in ans_l]
        return f"keyword_match: hit {len(hits)}/{len(kws)} ({hits}) of {kws}"


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@register("numeric_tolerance")
class NumericToleranceGrader(Grader):
    """
    Extract numbers from answer and expected, return 1.0 if any expected number appears
    in the answer within absolute or relative tolerance. Useful for math word problems
    where the LLM may write "the answer is 2" vs. "approximately 2.0".
    """

    abs_tol: float = 1e-3
    rel_tol: float = 1e-3

    @staticmethod
    def _extract(text: str) -> list[float]:
        return [float(m.group(0)) for m in _NUMBER_RE.finditer(text or "")]

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        expected = self._extract(" ".join(task.expected_answer_keywords or []))
        got = self._extract(answer)
        if not expected or not got:
            return 0.0
        for e in expected:
            for g in got:
                if math.isclose(e, g, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    return 1.0
        # Partial credit: closest distance, normalized
        diffs = [min(abs(e - g) for g in got) for e in expected]
        best = min(diffs)
        scale = max(abs(x) for x in expected + got) or 1.0
        return max(0.0, 1.0 - best / scale)


@register("llm_judge")
@dataclass
class LLMJudgeGrader(Grader):
    """
    Placeholder. In a real deployment this would call OpenAI / Qwen / DeepSeek with
    a rubric prompt. For now it falls back to KeywordMatchGrader so the registry
    still produces a meaningful score through the full pipeline.
    """
    # NOTE: must use `Optional[...]` and a sentinel default. If we annotate as
    # `Grader` and default to None, dataclass treats the class itself as a default
    # factory candidate and you end up with self.fallback being a Field descriptor.
    # Also: this subclass needs its own @dataclass so the new `fallback` field gets
    # a proper __init__ entry and __post_init__ is called.
    fallback: Optional[Grader] = None

    def __post_init__(self) -> None:
        if self.fallback is None:
            # Lazy-instantiate to avoid circular import
            self.fallback = KeywordMatchGrader()

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        # TODO: wire a real LLM call. Until then the fallback keeps the pipeline working.
        return self.fallback.score(answer, task, trace)

    def explain(self, answer: str, task: Any, trace: Any = None) -> str:
        return f"llm_judge (fallback=keyword_match): {self.fallback.explain(answer, task, trace)}"


@register("efficiency")
class EfficiencyGrader(Grader):
    """
    Score based on how much of the timeout budget the run consumed.
    Faster runs get higher scores. Independent of correctness — combine with another
    grader if you want a single scalar (e.g., min(correctness, efficiency)).
    """

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        if trace is None or not hasattr(trace, "latency_ms") or not getattr(task, "timeout_seconds", None):
            return 0.5  # neutral if we have no signal
        used = trace.latency_ms / (task.timeout_seconds * 1000.0)
        # Linear: used=0 -> 1.0, used>=1 -> 0.0
        return max(0.0, 1.0 - used)


@register("tool_accuracy")
class ToolAccuracyGrader(Grader):
    """
    1.0 if the expected_tool appears in the trace's actual_tools, else 0.0.
    Mirrors the original metric so the registry can replace the hard-coded call.
    """

    def score(self, answer: str, task: Any, trace: Any = None) -> float:
        if trace is None or not hasattr(trace, "actual_tools"):
            return 0.0
        return 1.0 if task.expected_tool in (trace.actual_tools or []) else 0.0


__all__ = [
    "Grader",
    "register", "get", "list_graders",
    "ExactMatchGrader", "KeywordMatchGrader", "NumericToleranceGrader",
    "LLMJudgeGrader", "EfficiencyGrader", "ToolAccuracyGrader",
]
