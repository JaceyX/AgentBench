def keyword_judge(query: str, answer: str, expected_keywords: list[str]) -> float:
    """
    MVP版本默认使用关键词裁判，避免必须依赖外部大模型API。
    后续可以替换为 OpenAI / Qwen / DeepSeek 的 LLM-as-a-Judge。
    """
    if not expected_keywords:
        return 1.0

    hit_count = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
    return round(hit_count / len(expected_keywords), 4)


def llm_judge(query: str, answer: str, expected_keywords: list[str]) -> float:
    """
    预留接口。
    当前项目为了可直接运行，先复用 keyword_judge。
    如果要接真实大模型，可在这里调用 OpenAI、Qwen 或 DeepSeek API。
    """
    return keyword_judge(query, answer, expected_keywords)
