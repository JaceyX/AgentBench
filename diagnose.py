"""
Case-level diagnostic report for AgentBench runs.
- Runs a fresh eval
- Breaks down by source (gsm8k / math23k) and task_type
- Categorizes failure modes (routing / calculator_parse / search_no_doc / keyword_miss)
- Outputs JSON report + console summary
"""
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

API = "http://127.0.0.1:8000"
OUT = Path("D:/OneDrive/桌面/AgentBench/outputs")
OUT.mkdir(exist_ok=True)


def run_eval(experiment_name: str) -> dict:
    url = f"{API}/eval/run?experiment_name={experiment_name}&prompt_version=v1&model_name=mock-agent"
    with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=120) as r:
        return json.loads(r.read())


def categorize_failure(d: dict) -> str:
    """Classify a failed case into a failure mode."""
    if d["expected_tool"] not in d["actual_tools"]:
        return "routing_mismatch"
    if d["expected_tool"] == "calculator_tool" and d["final_answer"].startswith("根据工具执行结果，计算失败"):
        return "calculator_parse_fail"
    if d["expected_tool"] == "search_tool" and "未检索到相关内容" in d["final_answer"]:
        return "search_no_doc"
    if d["expected_tool"] == "search_tool":
        return "keyword_miss_search"
    if d["expected_tool"] == "calculator_tool":
        return "keyword_miss_calc"
    if d["expected_tool"] == "weather_tool":
        return "keyword_miss_weather"
    return "other"


def stat_block(items: list) -> dict:
    if not items:
        return {"count": 0, "pass": 0, "fail": 0, "pass_rate": 0.0, "avg_latency_ms": 0, "avg_tool_accuracy": 0.0}
    n = len(items)
    p = sum(1 for x in items if x["success"])
    avg_lat = round(sum(x["latency_ms"] for x in items) / n, 2)
    avg_acc = round(sum(x["tool_accuracy"] for x in items) / n, 4)
    return {
        "count": n,
        "pass": p,
        "fail": n - p,
        "pass_rate": round(p / n, 4),
        "avg_latency_ms": avg_lat,
        "avg_tool_accuracy": avg_acc,
    }


def build_diagnostic(result: dict) -> dict:
    details = result["details"]

    by_source = defaultdict(list)
    by_task = defaultdict(list)
    by_source_task = defaultdict(list)
    for d in details:
        src = d["case_id"].split("_")[0]
        by_source[src].append(d)
        by_task[d["task_type"]].append(d)
        by_source_task[f"{src}/{d['task_type']}"].append(d)

    failure_modes = Counter()
    failure_modes_by_source = defaultdict(Counter)
    failure_modes_by_task = defaultdict(Counter)
    for d in details:
        if not d["success"]:
            mode = categorize_failure(d)
            failure_modes[mode] += 1
            failure_modes_by_source[d["case_id"].split("_")[0]][mode] += 1
            failure_modes_by_task[d["task_type"]][mode] += 1

    routing_dist_by_source = defaultdict(Counter)
    for d in details:
        actual = d["actual_tools"][0] if d["actual_tools"] else "none"
        routing_dist_by_source[d["case_id"].split("_")[0]][actual] += 1

    samples_by_mode = defaultdict(list)
    for d in details:
        if not d["success"]:
            mode = categorize_failure(d)
            if len(samples_by_mode[mode]) < 2:
                samples_by_mode[mode].append({
                    "case_id": d["case_id"],
                    "query": d["query"][:100],
                    "expected_tool": d["expected_tool"],
                    "actual_tools": d["actual_tools"],
                    "final_answer": d["final_answer"][:120],
                })

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "experiment": result["experiment"],
        "overall": result["summary"],
        "by_source": {k: stat_block(v) for k, v in by_source.items()},
        "by_task_type": {k: stat_block(v) for k, v in by_task.items()},
        "by_source_task_type": {k: stat_block(v) for k, v in by_source_task.items()},
        "failure_modes_total": dict(failure_modes),
        "failure_modes_by_source": {k: dict(v) for k, v in failure_modes_by_source.items()},
        "failure_modes_by_task_type": {k: dict(v) for k, v in failure_modes_by_task.items()},
        "routing_distribution_by_source": {k: dict(v) for k, v in routing_dist_by_source.items()},
        "samples_by_failure_mode": dict(samples_by_mode),
    }


def print_console(diag: dict) -> None:
    print("=" * 72)
    print("AgentBench Case-Level Diagnostic")
    print("=" * 72)
    o = diag["overall"]
    print(f"Experiment : {diag['experiment'].get('name')} (id={diag['experiment'].get('id')})")
    print(f"Total/Pass : {o['total_cases']} / {o['success_count']}  pass_rate={o['success_rate']}")
    print(f"Tool acc   : {o['avg_tool_accuracy']}    avg_latency={o['avg_latency_ms']}ms")
    print()

    print("--- By source ---")
    for k, v in diag["by_source"].items():
        print(f"  {k:10s}  n={v['count']:3d}  pass={v['pass']:3d}  pass_rate={v['pass_rate']:.2%}  tool_acc={v['avg_tool_accuracy']}")
    print()

    print("--- Routing distribution by source ---")
    for src, dist in diag["routing_distribution_by_source"].items():
        line = "  ".join(f"{t}={c}" for t, c in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"  {src:10s}  {line}")
    print()

    print("--- Failure modes (total) ---")
    for mode, n in sorted(diag["failure_modes_total"].items(), key=lambda x: -x[1]):
        print(f"  {mode:30s}  {n}")
    print()

    print("--- Failure modes by source ---")
    for src, modes in diag["failure_modes_by_source"].items():
        print(f"  {src}:")
        for mode, n in sorted(modes.items(), key=lambda x: -x[1]):
            print(f"     {mode:28s}  {n}")
    print()

    print("--- Sample cases per failure mode ---")
    for mode, samples in diag["samples_by_failure_mode"].items():
        print(f"  [{mode}]")
        for s in samples:
            print(f"    {s['case_id']} | expected={s['expected_tool']} actual={s['actual_tools']}")
            print(f"      Q: {s['query']}...")
            print(f"      A: {s['final_answer']}...")
    print()


def main() -> int:
    name = "diagnostic_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"Running eval: {name}")
    result = run_eval(name)
    diag = build_diagnostic(result)

    out_path = OUT / f"diagnostic_exp_{result['experiment']['id']}.json"
    out_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved: {out_path}\n")

    print_console(diag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
