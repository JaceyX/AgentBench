import ast
import operator


def weather_tool(city: str) -> str:
    weather_db = {
        "北京": "北京今天晴，气温20-28℃。",
        "上海": "上海今天多云，气温22-29℃。",
        "深圳": "深圳今天有阵雨，气温25-31℃。",
    }
    return weather_db.get(city, f"{city} 今天天气晴朗。")


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported operator")
        return _ALLOWED_OPERATORS[op_type](
            _safe_eval(node.left),
            _safe_eval(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported unary operator")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))

    raise ValueError("Unsupported expression")


def calculator_tool(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"计算失败：{exc}"


def search_tool(query: str) -> str:
    mock_docs = {
        "RAG": "RAG 是 Retrieval-Augmented Generation，即检索增强生成，通过外部知识检索增强大模型回答。",
        "Agent": "Agent 是具备规划、工具调用、记忆和执行能力的大模型应用系统。",
        "LangGraph": "LangGraph 是用于构建有状态、多步骤 Agent 工作流的框架。",
        "LoRA": "LoRA 是一种参数高效微调方法，通过低秩矩阵近似权重更新。",
    }

    for key, value in mock_docs.items():
        if key.lower() in query.lower():
            return value

    return "未检索到相关内容。"
