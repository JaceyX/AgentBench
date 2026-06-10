from pydantic import BaseModel


class EvalRunRequest(BaseModel):
    experiment_name: str = "default_experiment"
    prompt_version: str = "v1"
    model_name: str = "mock-agent"
    use_llm_judge: bool = False


class CaseResult(BaseModel):
    case_id: str
    task_type: str
    query: str
    expected_tool: str | None
    actual_tools: list[str]
    final_answer: str
    latency_ms: int
    tool_accuracy: float
    keyword_success: bool
    judge_score: float | None
    success: bool
