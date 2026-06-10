import json
from pathlib import Path

from app.evaluator.metrics import summarize_results


def generate_report(
    results: list[dict],
    output_path: str | Path,
    experiment: dict | None = None,
) -> dict:
    summary = summarize_results(results)

    report = {
        "experiment": experiment or {},
        "summary": summary,
        "details": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
