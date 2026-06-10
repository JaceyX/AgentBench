from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.runner.tools import calculator_tool, search_tool, weather_tool


class AgentState(TypedDict):
    query: str
    thoughts: list[str]
    tools_called: list[str]
    observations: list[str]
    final_answer: str
    trace: list[dict[str, Any]]


def _add_trace(
    state: AgentState,
    node_name: str,
    output_text: str,
    tool_name: str | None = None
) -> None:
    state["trace"].append({
        "step_index": len(state["trace"]) + 1,
        "node_name": node_name,
        "input": state["query"],
        "output": output_text,
        "tool_name": tool_name,
    })


def planner_node(state: AgentState) -> AgentState:
    query = state["query"]

    if "天气" in query:
        tool = "weather_tool"
    elif "计算" in query or any(op in query for op in ["+", "-", "*", "/", "×"]):
        tool = "calculator_tool"
    else:
        tool = "search_tool"

    thought = f"根据用户问题选择工具：{tool}"
    state["thoughts"].append(thought)
    state["tools_called"].append(tool)
    _add_trace(state, "planner", thought, tool)
    return state


def tool_node(state: AgentState) -> AgentState:
    query = state["query"]
    tool = state["tools_called"][-1]

    if tool == "weather_tool":
        city = "北京"
        for candidate in ["北京", "上海", "深圳"]:
            if candidate in query:
                city = candidate
                break
        result = weather_tool(city)

    elif tool == "calculator_tool":
        expression = (
            query.replace("计算", "")
            .replace("的结果", "")
            .replace("×", "*")
            .strip()
        )
        result = calculator_tool(expression)

    else:
        result = search_tool(query)

    state["observations"].append(result)
    _add_trace(state, "tool_executor", result, tool)
    return state


def answer_node(state: AgentState) -> AgentState:
    observation = state["observations"][-1]
    answer = f"根据工具执行结果，{observation}"
    state["final_answer"] = answer
    _add_trace(state, "answer", answer)
    return state


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")
    graph.add_edge("tool_executor", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def run_agent(query: str) -> dict[str, Any]:
    agent = build_agent()

    init_state: AgentState = {
        "query": query,
        "thoughts": [],
        "tools_called": [],
        "observations": [],
        "final_answer": "",
        "trace": [],
    }

    result = agent.invoke(init_state)
    return dict(result)
