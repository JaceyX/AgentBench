import json
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.evaluator.judge import llm_judge
from app.evaluator.metrics import (
    calculate_keyword_success,
    calculate_success,
    calculate_tool_accuracy,
)
from app.models import Experiment, Run, Trace
from app.runner.agent import run_agent


def load_benchmark(path: str | Path) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def create_experiment(
    db: Session,
    name: str,
    prompt_version: str,
    model_name: str,
) -> Experiment:
    experiment = Experiment(
        name=name,
        prompt_version=prompt_version,
        model_name=model_name,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def save_run_result(
    db: Session,
    experiment_id: int,
    result: dict,
) -> None:
    run = Run(
        experiment_id=experiment_id,
        case_id=result["case_id"],
        task_type=result["task_type"],
        query=result["query"],
        expected_tool=result.get("expected_tool"),
        actual_tools=json.dumps(result["actual_tools"], ensure_ascii=False),
        final_answer=result["final_answer"],
        latency_ms=result["latency_ms"],
        tool_accuracy=result["tool_accuracy"],
        keyword_success=result["keyword_success"],
        judge_score=result["judge_score"],
        success=result["success"],
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    for trace_item in result["trace"]:
        trace = Trace(
            run_id=run.id,
            step_index=trace_item["step_index"],
            node_name=trace_item["node_name"],
            input_text=trace_item.get("input", ""),
            output_text=trace_item.get("output", ""),
            tool_name=trace_item.get("tool_name"),
        )
        db.add(trace)

    db.commit()


def run_single_case(case: dict, use_llm_judge: bool = False) -> dict:
    start = time.time()

    agent_result = run_agent(case["query"])

    latency_ms = int((time.time() - start) * 1000)
    final_answer = agent_result["final_answer"]
    actual_tools = agent_result["tools_called"]

    tool_score = calculate_tool_accuracy(
        case.get("expected_tool"),
        actual_tools,
    )

    keyword_success = calculate_keyword_success(
        final_answer,
        case.get("expected_answer_keywords", []),
    )

    judge_score = None
    if use_llm_judge:
        judge_score = llm_judge(
            query=case["query"],
            answer=final_answer,
            expected_keywords=case.get("expected_answer_keywords", []),
        )

    success = calculate_success(
        tool_score=tool_score,
        keyword_success=keyword_success,
        judge_score=judge_score,
    )

    return {
        "case_id": case["id"],
        "task_type": case["task_type"],
        "query": case["query"],
        "expected_tool": case.get("expected_tool"),
        "actual_tools": actual_tools,
        "final_answer": final_answer,
        "latency_ms": latency_ms,
        "tool_accuracy": tool_score,
        "keyword_success": keyword_success,
        "judge_score": judge_score,
        "success": success,
        "trace": agent_result["trace"],
    }


def run_benchmark(
    dataset_path: str | Path,
    db: Session | None = None,
    experiment_id: int | None = None,
    use_llm_judge: bool = False,
) -> list[dict]:
    cases = load_benchmark(dataset_path)
    results = []

    for case in cases:
        result = run_single_case(case, use_llm_judge=use_llm_judge)
        results.append(result)

        if db is not None and experiment_id is not None:
            save_run_result(db, experiment_id, result)

    return results
