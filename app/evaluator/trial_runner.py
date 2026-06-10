"""
Multi-trial evaluation runner.

Why a separate runner:
- The existing /eval/run endpoint runs each case once. That gives a binary pass/fail
  per case, but LLM agents (even with temperature=0) can be non-deterministic across
  model versions, network conditions, and tool timeouts. A single trial under-estimates
  capability on lucky runs and over-estimates it on unlucky ones.
- Multi-trial evaluation (a.k.a. "pass@k" in the harness guide):
    * Run the same case N times.
    * Report per-case pass_rate, Wilson 95% confidence interval, and pass@k
      (at least one success out of N).
    * Aggregate by source / task_type / difficulty.
- The runner is a pure orchestrator: it calls POST /eval/run N times per case, and
  it does NOT modify any existing code. It writes its report to outputs/.

Usage:
    python -m app.evaluator.trial_runner --trials 5
    python -m app.evaluator.trial_runner --trials 3 --source gsm8k
    python -m app.evaluator.trial_runner --trials 2 --cases gsm8k_001,gsm8k_002
    python -m app.evaluator.trial_runner --self-test      # offline smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .registry import get, list_graders
from ..datasets.schema import Task, load_tasks


API = "http://127.0.0.1:8000"
OUT = Path("D:/OneDrive/桌面/AgentBench/outputs")
OUT.mkdir(parents=True, exist_ok=True)


# ---- Statistics helpers ---------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval. Robust for small n and p near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---- Trial result structures ---------------------------------------------------

@dataclass
class TrialResult:
    trial_index: int
    success: bool
    latency_ms: float
    tool_accuracy: float
    grader_scores: dict[str, float] = field(default_factory=dict)
    final_answer: str = ""
    actual_tools: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class CaseSummary:
    task_id: str
    task_type: str
    source: str
    difficulty: str
    trials: int
    successes: int
    pass_rate: float
    pass_at_k: bool              # at least one success out of N
    ci_low: float
    ci_high: float
    latency_p50_ms: float
    latency_p95_ms: float
    tool_accuracy_mean: float
    grader_scores_mean: dict[str, float]
    trials_detail: list[TrialResult]


# ---- HTTP client ---------------------------------------------------------------

def call_eval_run(experiment_name: str, timeout: float = 300.0) -> dict:
    """Hit the existing /eval/run endpoint. Returns the parsed JSON body."""
    url = (
        f"{API}/eval/run"
        f"?experiment_name={experiment_name}"
        f"&prompt_version=v1&model_name=mock-agent"
    )
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_case_result(eval_result: dict, task_id: str) -> dict | None:
    for d in eval_result.get("details", []):
        if d.get("case_id") == task_id:
            return d
    return None


# ---- Grading via the registry -------------------------------------------------

def grade_with_registry(answer: str, task: Task, case: dict) -> dict[str, float]:
    """Score one answer with every registered grader. Returns {grader_name: score}."""
    scores: dict[str, float] = {}
    for name in list_graders():
        try:
            g = get(name)
            scores[name] = float(g.score(answer, task, trace=case))
        except Exception as e:  # a buggy grader must never crash the whole run
            scores[name] = 0.0
            scores[f"{name}_error"] = str(e)  # type: ignore[assignment]
    return scores


# ---- Core loop -----------------------------------------------------------------

def run_case_trials(task: Task, trials: int, experiment_prefix: str) -> CaseSummary:
    """Run the same case `trials` times, each as a fresh /eval/run experiment."""
    trial_results: list[TrialResult] = []
    for i in range(trials):
        exp = f"{experiment_prefix}::{task.task_id}::t{i + 1}"
        t0 = time.time()
        try:
            res = call_eval_run(exp)
            case = get_case_result(res, task.task_id)
            if case is None:
                raise RuntimeError(f"case {task.task_id} not found in /eval/run response")
            scores = grade_with_registry(case.get("final_answer", ""), task, case)
            tr = TrialResult(
                trial_index=i + 1,
                success=bool(case.get("success")),
                latency_ms=float(case.get("latency_ms", (time.time() - t0) * 1000)),
                tool_accuracy=float(case.get("tool_accuracy", 0.0)),
                grader_scores=scores,
                final_answer=case.get("final_answer", ""),
                actual_tools=list(case.get("actual_tools", [])),
            )
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, TimeoutError, OSError) as e:
            tr = TrialResult(
                trial_index=i + 1,
                success=False,
                latency_ms=(time.time() - t0) * 1000,
                tool_accuracy=0.0,
                grader_scores={},
                error=str(e),
            )
        trial_results.append(tr)

    succ = sum(1 for t in trial_results if t.success)
    n = len(trial_results)
    ci_lo, ci_hi = wilson_ci(succ, n)
    lat = [t.latency_ms for t in trial_results]
    acc = [t.tool_accuracy for t in trial_results]
    grader_keys = sorted({
        k for t in trial_results for k in t.grader_scores if not k.endswith("_error")
    })
    grader_mean = {
        k: round(sum(t.grader_scores.get(k, 0.0) for t in trial_results) / max(1, n), 4)
        for k in grader_keys
    }
    return CaseSummary(
        task_id=task.task_id,
        task_type=task.task_type,
        source=(task.metadata.get("source") or task.task_id.split("_")[0]),
        difficulty=task.difficulty,
        trials=n,
        successes=succ,
        pass_rate=round(succ / n, 4) if n else 0.0,
        pass_at_k=(succ >= 1),
        ci_low=round(ci_lo, 4),
        ci_high=round(ci_hi, 4),
        latency_p50_ms=round(percentile(lat, 0.50), 2),
        latency_p95_ms=round(percentile(lat, 0.95), 2),
        tool_accuracy_mean=round(sum(acc) / n, 4) if n else 0.0,
        grader_scores_mean=grader_mean,
        trials_detail=trial_results,
    )


# ---- Aggregation ---------------------------------------------------------------

def aggregate(by: str, summaries: list[CaseSummary]) -> dict[str, dict]:
    """Group summaries by a CaseSummary field, computing per-group statistics."""
    buckets: dict[str, list[CaseSummary]] = defaultdict(list)
    for s in summaries:
        key = getattr(s, by, "unknown")
        buckets[key].append(s)
    out: dict[str, dict] = {}
    for key, items in buckets.items():
        n = len(items)
        prs = [x.pass_rate for x in items]
        out[key] = {
            "cases": n,
            "trials_total": sum(x.trials for x in items),
            "successes_total": sum(x.successes for x in items),
            # Mean of per-case pass rates — the "expected pass rate" estimator, more
            # stable than total successes / total trials when cases have different N.
            "mean_pass_rate": round(sum(prs) / n, 4) if n else 0.0,
            "variance": round(statistics.pvariance(prs), 4) if n > 1 else 0.0,
            "pass_at_k_rate": round(sum(1 for x in items if x.pass_at_k) / n, 4) if n else 0.0,
        }
    return out


# ---- Self-test (no network) ----------------------------------------------------

def self_test() -> int:
    """
    Offline smoke test: exercise the registry, the stats helpers, and the
    CaseSummary builder with synthetic data. Does NOT call the API.
    """
    print("Self-test: registry")
    print(f"  registered graders: {list_graders()}")
    assert "keyword_match" in list_graders()
    assert "numeric_tolerance" in list_graders()
    assert "tool_accuracy" in list_graders()

    print("Self-test: stats")
    lo, hi = wilson_ci(0, 5)
    assert lo == 0.0 and hi > 0.0, f"wilson(0,5) should give (0, x>0), got ({lo},{hi})"
    lo, hi = wilson_ci(5, 5)
    assert hi == 1.0 and lo < 1.0, f"wilson(5,5) should give (x<1, 1), got ({lo},{hi})"
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == pytest_close(4.8)

    print("Self-test: graders")
    kw = get("keyword_match")
    # 2 keywords, both hit -> 1.0
    task = Task(task_id="t1", query="q", expected_answer_keywords=["beijing", "sunny"])
    assert kw.score("Beijing is sunny today", task) == 1.0
    # 4 keywords, 1 hit -> ratio 0.25 < 0.5 -> 0.25 (partial)
    task4 = Task(task_id="t4", query="q", expected_answer_keywords=["beijing", "sunny", "warm", "blue"])
    assert 0 < kw.score("beijing is here", task4) < 1.0
    # No hits -> 0.0
    assert kw.score("shanghai is rainy", task) == 0.0

    num = get("numeric_tolerance")
    t2 = Task(task_id="t2", query="q", expected_answer_keywords=["3.14"])
    assert num.score("the answer is 3.14", t2) == 1.0
    # 3 is within partial-credit range of 3.14 (1 - 0.14/3.14 ~= 0.955)
    s = num.score("the answer is 3", t2)
    assert 0 < s < 1.0, f"expected partial credit, got {s}"
    # No numbers in answer -> 0
    assert num.score("no numbers here", t2) == 0.0
    # No expected numbers -> 0
    t_empty = Task(task_id="te", query="q", expected_answer_keywords=[])
    assert num.score("the answer is 42", t_empty) == 0.0

    ta = get("tool_accuracy")
    fake_trace = type("Trace", (), {"actual_tools": ["calculator_tool"]})()
    t3 = Task(task_id="t3", query="q", expected_tool="calculator_tool")
    assert ta.score("x", t3, trace=fake_trace) == 1.0
    fake_trace2 = type("Trace", (), {"actual_tools": ["search_tool"]})()
    assert ta.score("x", t3, trace=fake_trace2) == 0.0

    print("Self-test: aggregation")
    summary = CaseSummary(
        task_id="t1", task_type="calculator", source="x", difficulty="easy",
        trials=3, successes=2, pass_rate=0.667, pass_at_k=True,
        ci_low=0.0, ci_high=1.0,
        latency_p50_ms=10.0, latency_p95_ms=20.0, tool_accuracy_mean=0.7,
        grader_scores_mean={"keyword_match": 0.5}, trials_detail=[],
    )
    agg = aggregate("source", [summary])
    assert "x" in agg and agg["x"]["cases"] == 1
    print("  ok")
    print("All self-tests passed.")
    return 0


def pytest_close(x: float) -> float:
    """Tiny helper for the self-test (avoid importing pytest)."""
    return x


# ---- Selection -----------------------------------------------------------------

def select_tasks(tasks: list[Task], cases_arg: str | None, source_arg: str | None) -> list[Task]:
    if cases_arg:
        wanted = {c.strip() for c in cases_arg.split(",") if c.strip()}
        tasks = [t for t in tasks if t.task_id in wanted]
    if source_arg:
        tasks = [t for t in tasks if t.task_id.split("_")[0] == source_arg]
    return tasks


# ---- Entry point ---------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AgentBench multi-trial runner")
    p.add_argument("--trials", type=int, default=3, help="Number of trials per case (default 3)")
    p.add_argument("--dataset", default="app/datasets/benchmark.jsonl",
                   help="Path to benchmark.jsonl")
    p.add_argument("--cases", default=None, help="Comma-separated case_id list (default: all)")
    p.add_argument("--source", default=None, help="Filter by source prefix (e.g. gsm8k, math23k)")
    p.add_argument("--grader-summary", action="store_true",
                   help="Print per-grader mean scores in the console summary")
    p.add_argument("--self-test", action="store_true",
                   help="Run offline smoke test (no API / no dataset required)")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    tasks = load_tasks(args.dataset)
    tasks = select_tasks(tasks, args.cases, args.source)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2

    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = f"trial_{run_stamp}"
    total_runs = len(tasks) * args.trials
    print(f"Running {len(tasks)} cases x {args.trials} trials = {total_runs} total runs")

    summaries: list[CaseSummary] = []
    for i, t in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {t.task_id}  ({t.task_type}) ...", end="", flush=True)
        s = run_case_trials(t, args.trials, prefix)
        summaries.append(s)
        print(f" pass={s.successes}/{s.trials}  ci=[{s.ci_low:.2f},{s.ci_high:.2f}]")

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trials_per_case": args.trials,
        "case_count": len(summaries),
        "available_graders": list_graders(),
        "overall": {
            "cases": len(summaries),
            "trials_total": sum(s.trials for s in summaries),
            "successes_total": sum(s.successes for s in summaries),
            "mean_pass_rate": round(sum(s.pass_rate for s in summaries) / len(summaries), 4),
            "pass_at_k_rate": round(sum(1 for s in summaries if s.pass_at_k) / len(summaries), 4),
        },
        "by_source": aggregate("source", summaries),
        "by_task_type": aggregate("task_type", summaries),
        "by_difficulty": aggregate("difficulty", summaries),
        "cases": [asdict(s) for s in summaries],
    }
    out_path = OUT / f"trials_{run_stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")

    # ---- Console summary -------------------------------------------------
    print()
    print("=" * 72)
    print(f"Multi-trial summary (N={args.trials} per case)")
    print("=" * 72)
    o = report["overall"]
    print(f"  cases: {o['cases']}    trials_total: {o['trials_total']}    "
          f"successes: {o['successes_total']}")
    print(f"  mean_pass_rate: {o['mean_pass_rate']}    pass_at_k_rate: {o['pass_at_k_rate']}")
    print()
    print("  By source:")
    for k, v in report["by_source"].items():
        print(f"    {k:10s}  cases={v['cases']:3d}  trials={v['trials_total']:3d}  "
              f"pass_rate={v['mean_pass_rate']:.2%}  pass@k={v['pass_at_k_rate']:.2%}  "
              f"var={v['variance']:.3f}")
    print()
    print("  By task_type:")
    for k, v in report["by_task_type"].items():
        print(f"    {k:15s}  cases={v['cases']:3d}  trials={v['trials_total']:3d}  "
              f"pass_rate={v['mean_pass_rate']:.2%}  pass@k={v['pass_at_k_rate']:.2%}  "
              f"var={v['variance']:.3f}")
    if args.grader_summary:
        print()
        print("  Per-grader mean (across all trials in all cases):")
        agg: dict[str, list[float]] = defaultdict(list)
        for s in summaries:
            for t in s.trials_detail:
                for g, v in t.grader_scores.items():
                    if not g.endswith("_error"):
                        agg[g].append(v)
        for g, vs in sorted(agg.items()):
            print(f"    {g:20s}  mean={sum(vs) / len(vs):.3f}  n={len(vs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
