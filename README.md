# AgentBench

> 面向 LLM Agent 的自动化评测与回归测试平台 MVP。

一个最小可运行的 Agent 评测框架：给定一份 JSONL 基准集，让 LangGraph Agent 逐条作答，
统计成功率、工具准确率、延迟等指标，并把结果持久化到 SQLite、生成 JSON 报告。

## 特性

- **基准集驱动**：JSONL 用例，按 `id / task_type / query / expected_tool / expected_answer_keywords` 描述
- **LangGraph 工作流**：`planner → tool_executor → answer` 三节点
- **三类样例任务**：Tool Calling（天气）、Calculator（AST 安全求值）、RAG QA（关键词检索）
- **多维度指标**：Success Rate、Tool Accuracy、Keyword Success、Latency、可选 LLM-as-a-Judge
- **结果持久化**：SQLite 三级表 `experiments / runs / traces`
- **报告导出**：每次实验生成一份 JSON 报告到 `outputs/`
- **HTTP 接口**：FastAPI + SSE 流式输出，可对接前端或 CI

## 技术栈

| 类别 | 选型 |
| --- | --- |
| Web 框架 | FastAPI 0.115 + Uvicorn |
| Agent 框架 | LangGraph 0.2 |
| ORM | SQLAlchemy 2.0 |
| 数据校验 | Pydantic 2.10 |
| 数据库 | SQLite |
| 配置 | python-dotenv |

## 项目结构

```
AgentBench/
├── app/
│   ├── main.py              # FastAPI 路由
│   ├── config.py            # 路径与 DB URL
│   ├── database.py          # SQLAlchemy 引擎/Session
│   ├── models.py            # ORM: Experiment / Run / Trace
│   ├── schemas.py           # Pydantic 模型
│   ├── datasets/
│   │   └── benchmark.jsonl  # 评测用例（公开基准，见 SOURCES.md）
│   ├── runner/
│   │   ├── agent.py         # LangGraph 状态机
│   │   ├── runner.py        # 加载/执行/持久化
│   │   └── tools.py         # weather / calculator / search
│   └── evaluator/
│       ├── judge.py         # keyword_judge / llm_judge 预留
│       ├── metrics.py       # 指标计算与汇总
│       └── report.py        # JSON 报告生成
├── outputs/                 # 评测报告 + 诊断报告
├── _public_data/            # 公开基准的原始下载副本（审计/复现用）
├── diagnose.py              # Case-level 失败模式诊断脚本
├── SOURCES.md               # 公开基准来源、协议、引用
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

依赖安装完成后，会在仓库根目录自动创建 `agentbench.db`（SQLite）和 `outputs/` 目录。

## 启动服务

```bash
uvicorn app.main:app --reload
```

默认监听 `http://127.0.0.1:8000`，OpenAPI 文档位于 `/docs`。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | `/` | 健康检查 |
| POST | `/eval/run` | 同步跑完整基准集，返回汇总报告 |
| GET  | `/eval/stream` | SSE 流式逐条推送执行进度 |
| GET  | `/eval/experiments` | 历史实验列表 |
| GET  | `/eval/experiments/{id}` | 单个实验的 Run 明细 |
| GET  | `/eval/runs/{id}/trace` | 单个 Run 的步骤追踪 |

### 1. 同步运行评测

```bash
curl -X POST "http://127.0.0.1:8000/eval/run?experiment_name=test_v1&prompt_version=v1&model_name=mock-agent"
```

Query 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `experiment_name` | `default_experiment` | 实验名 |
| `prompt_version` | `v1` | Prompt 版本号，用于回归对比 |
| `model_name` | `mock-agent` | 被测 Agent / 模型标识 |
| `use_llm_judge` | `false` | 是否启用 LLM-as-a-Judge（当前等价于关键词裁判） |

返回示例：

```json
{
  "experiment": {"id": 1, "name": "test_v1", "prompt_version": "v1", "model_name": "mock-agent"},
  "summary": {
    "total_cases": 30,
    "success_count": 30,
    "success_rate": 1.0,
    "avg_latency_ms": 0.49,
    "avg_tool_accuracy": 1.0
  },
  "details": [ ... ],
  "report_path": "outputs\\report_exp_1.json"
}
```

### 2. 查看历史实验

```bash
curl "http://127.0.0.1:8000/eval/experiments"
```

### 3. 查看某次实验的所有 Run

```bash
curl "http://127.0.0.1:8000/eval/experiments/1"
```

### 4. 查看某次 Run 的步骤追踪

```bash
curl "http://127.0.0.1:8000/eval/runs/1/trace"
```

### 5. SSE 流式评测

```bash
curl -N "http://127.0.0.1:8000/eval/stream?experiment_name=stream_v1"
```

每条用例完成后会推送一条 `data: {json}\n\n`，最后一条 `event: completed` 携带完整汇总报告。

## 数据集格式

`app/datasets/benchmark.jsonl` 每行一条 JSON：

```json
{"id": "case_001", "task_type": "tool_calling", "query": "查询北京今天的天气", "expected_tool": "weather_tool", "expected_answer_keywords": ["北京", "晴"]}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 用例唯一 ID（当前以 `gsm8k_` / `math23k_` 前缀区分公开基准来源） |
| `task_type` | string | 任务类型（`tool_calling` / `calculator` / `rag_qa`） |
| `query` | string | 输入给 Agent 的问题 |
| `expected_tool` | string | 期望调用的工具名 |
| `expected_answer_keywords` | string[] | 期望答案中至少出现一半的关键词 |

**当前数据集来源**：30 条 GSM8K（OpenAI 公开基准，英文）+ 30 条 Math23K（公开仓库，中文）。
**所有用例均来自公开评测基准**，无项目手写、无网络爬取。完整来源/协议/引用见 [`SOURCES.md`](./SOURCES.md)。

## 指标定义

| 指标 | 计算方式 |
| --- | --- |
| **Tool Accuracy** | `expected_tool in actual_tools` 取 1 / 0 |
| **Keyword Success** | 命中关键词数 ≥ 半数 |
| **Success** | `tool_accuracy == 1.0` 且 (有 judge 时 `judge_score ≥ 0.7`；无 judge 时 `keyword_success`) |
| **Latency (ms)** | Agent 单次 `invoke` 的耗时 |

LLM-as-a-Judge 入口已在 `app/evaluator/judge.py:llm_judge` 预留，可对接 OpenAI / Qwen / DeepSeek。

## 输出物

- **SQLite**：`agentbench.db`，包含 `experiments / runs / traces` 三张表
- **JSON 报告**：`outputs/report_exp_{id}.json`，含 `experiment` 元信息、`summary` 汇总、`details` 逐条结果

## 诊断（Case-level Diagnostic）

默认报告只给一个 `success_rate`，看不出"为什么"失败。`diagnose.py` 把每条失败用例归类到**5 种失败模式**，并按 source / task_type 双轴切分：

### 用法

```bash
# 前置：服务在跑（uvicorn app.main:app）
python diagnose.py
```

会：
1. 调一次 `POST /eval/run` 拿完整 results
2. 在 `outputs/diagnostic_exp_{id}.json` 写一份结构化报告
3. 在终端打印可读摘要

### 5 种失败模式

| 模式 | 含义 |
| --- | --- |
| `routing_mismatch` | `expected_tool` 不在 `actual_tools` 中（planner 选错工具） |
| `calculator_parse_fail` | 路由到 calculator 但 AST 求值失败（自然语言 / 全角标点） |
| `search_no_doc` | 路由到 search 但 mock_docs 未收录 |
| `keyword_miss_search` | search 返回了内容但关键词未过半 |
| `keyword_miss_calc` | calculator 算出结果但答案数字不匹配 |

### 报告字段

| 字段 | 含义 |
| --- | --- |
| `overall` | 与 `summary` 一致 |
| `by_source` | 按 `case_id` 前缀（`gsm8k` / `math23k`）切分 |
| `by_task_type` | 按 `task_type` 切分 |
| `by_source_task_type` | 交叉切分 |
| `failure_modes_total` | 失败模式分布 |
| `failure_modes_by_source` | 每个公开基准各自的失败模式分布 |
| `routing_distribution_by_source` | 每个基准被分到哪个工具 |
| `samples_by_failure_mode` | 每种失败模式 2 条样例（含 query + answer 摘要） |

### 示例输出（GSM8K + Math23K 混合）

```
Overall: 60 / 0 pass (0%), tool_acc=0.267

By source:
  gsm8k     n=30  pass_rate=0.00%  tool_acc=0.200
  math23k   n=30  pass_rate=0.00%  tool_acc=0.333

Routing:
  gsm8k:     search_tool=24  calculator_tool=6
  math23k:   search_tool=20  calculator_tool=10

Failure modes (total):
  routing_mismatch      44
  calculator_parse_fail 16
```

读这份报告就能直接判断下一步该改 router 还是改 calculator parser。

## 路线图 / 后续可扩展点

- [ ] 替换 mock 工具为真实 LLM API（OpenAI / Qwen / DeepSeek）
- [ ] 接真实 LLM-as-a-Judge
- [ ] 评测用例去重 / 模糊匹配 / 失败回放
- [ ] 前端可视化（Streamlit 或独立 React 页面）
- [ ] 接入 CI，做 Prompt 回归门禁
