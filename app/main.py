import json
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import BENCHMARK_PATH, OUTPUT_DIR
from app.database import Base, engine, get_db
from app.evaluator.report import generate_report
from app.models import Experiment, Run, Trace
from app.runner.runner import create_experiment, load_benchmark, run_benchmark, run_single_case, save_run_result

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AgentBench",
    description="面向 LLM Agent 的自动化评测与回归测试平台",
    version="0.1.0",
)


@app.get("/")
def health_check():
    return {"message": "AgentBench is running"}


@app.post("/eval/run")
def run_eval(
    experiment_name: str = Query("default_experiment"),
    prompt_version: str = Query("v1"),
    model_name: str = Query("mock-agent"),
    use_llm_judge: bool = Query(False),
    db: Session = Depends(get_db),
):
    experiment = create_experiment(
        db=db,
        name=experiment_name,
        prompt_version=prompt_version,
        model_name=model_name,
    )

    results = run_benchmark(
        dataset_path=BENCHMARK_PATH,
        db=db,
        experiment_id=experiment.id,
        use_llm_judge=use_llm_judge,
    )

    report_path = OUTPUT_DIR / f"report_exp_{experiment.id}.json"
    report = generate_report(
        results=results,
        output_path=report_path,
        experiment={
            "id": experiment.id,
            "name": experiment.name,
            "prompt_version": experiment.prompt_version,
            "model_name": experiment.model_name,
        },
    )

    report["report_path"] = str(report_path)
    return report


@app.get("/eval/stream")
def stream_eval(
    experiment_name: str = Query("stream_experiment"),
    prompt_version: str = Query("v1"),
    model_name: str = Query("mock-agent"),
    use_llm_judge: bool = Query(False),
    db: Session = Depends(get_db),
):
    experiment = create_experiment(
        db=db,
        name=experiment_name,
        prompt_version=prompt_version,
        model_name=model_name,
    )

    cases = load_benchmark(BENCHMARK_PATH)

    def event_generator():
        results = []

        for case in cases:
            result = run_single_case(case, use_llm_judge=use_llm_judge)
            save_run_result(db, experiment.id, result)
            results.append(result)
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        report_path = OUTPUT_DIR / f"report_exp_{experiment.id}.json"
        report = generate_report(
            results=results,
            output_path=report_path,
            experiment={
                "id": experiment.id,
                "name": experiment.name,
                "prompt_version": experiment.prompt_version,
                "model_name": experiment.model_name,
            },
        )
        yield f"data: {json.dumps({'event': 'completed', 'report': report}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/eval/experiments")
def list_experiments(db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()

    return [
        {
            "id": exp.id,
            "name": exp.name,
            "prompt_version": exp.prompt_version,
            "model_name": exp.model_name,
            "created_at": exp.created_at,
            "run_count": len(exp.runs),
        }
        for exp in experiments
    ]


@app.get("/eval/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        return {"error": "experiment not found"}

    runs = db.query(Run).filter(Run.experiment_id == experiment_id).all()

    return {
        "experiment": {
            "id": experiment.id,
            "name": experiment.name,
            "prompt_version": experiment.prompt_version,
            "model_name": experiment.model_name,
            "created_at": experiment.created_at,
        },
        "runs": [
            {
                "id": run.id,
                "case_id": run.case_id,
                "task_type": run.task_type,
                "query": run.query,
                "expected_tool": run.expected_tool,
                "actual_tools": json.loads(run.actual_tools),
                "final_answer": run.final_answer,
                "latency_ms": run.latency_ms,
                "tool_accuracy": run.tool_accuracy,
                "keyword_success": run.keyword_success,
                "judge_score": run.judge_score,
                "success": run.success,
            }
            for run in runs
        ],
    }


@app.get("/eval/runs/{run_id}/trace")
def get_run_trace(run_id: int, db: Session = Depends(get_db)):
    traces = db.query(Trace).filter(Trace.run_id == run_id).order_by(Trace.step_index).all()

    return [
        {
            "step_index": trace.step_index,
            "node_name": trace.node_name,
            "input": trace.input_text,
            "output": trace.output_text,
            "tool_name": trace.tool_name,
            "created_at": trace.created_at,
        }
        for trace in traces
    ]
