def calculate_tool_accuracy(expected_tool: str | None, actual_tools: list[str]) -> float:
    if not expected_tool:
        return 1.0

    return 1.0 if expected_tool in actual_tools else 0.0


def calculate_keyword_success(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True

    hit_count = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    threshold = max(1, len(keywords) // 2)
    return hit_count >= threshold


def calculate_success(
    tool_score: float,
    keyword_success: bool,
    judge_score: float | None = None,
) -> bool:
    if judge_score is not None:
        return tool_score >= 1.0 and judge_score >= 0.7

    return tool_score >= 1.0 and keyword_success


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {
            "total_cases": 0,
            "success_count": 0,
            "success_rate": 0,
            "avg_latency_ms": 0,
            "avg_tool_accuracy": 0,
        }

    success_count = sum(1 for r in results if r["success"])
    avg_latency = sum(r["latency_ms"] for r in results) / total
    avg_tool_accuracy = sum(r["tool_accuracy"] for r in results) / total

    return {
        "total_cases": total,
        "success_count": success_count,
        "success_rate": round(success_count / total, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_tool_accuracy": round(avg_tool_accuracy, 4),
    }
