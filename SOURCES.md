# 数据集来源说明

`app/datasets/benchmark.jsonl` 中所有用例均来自公开评测基准，**不包含任何项目自带的、本地手写的、网络爬取的内容**。

当前数据集组成：**30 条 GSM8K（英文） + 30 条 Math23K（中文） = 60 条**。

## GSM8K (Grade School Math 8K)

- **来源仓库**：<https://github.com/openai/grade-school-math>
- **原始文件**：`grade_school_math/data/test.jsonl`（共 1,319 条）
- **项目抽样**：使用 `random.seed(20260610)` 抽取 30 条
- **论文**：Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.
- **协议**：MIT License
- **改写内容**：仅做格式转换
  - `id` → `gsm8k_001..030`
  - `task_type` → `"calculator"`
  - `expected_tool` → `"calculator_tool"`
  - `expected_answer_keywords` → 从原 `answer` 字段 `####` 之后的最终数值提取

## Math23K (中文小学数学应用题)

- **来源仓库**：<https://github.com/SCNU203/Math23k>
- **原始文件**：`math23k_test.json`（共 1,000 条，每条为顶层 JSON 对象）
- **项目抽样**：使用 `random.seed(20260611)` 抽取 30 条
- **论文**：Wang, Y., Liu, X., & Shi, S. (2017). *Deep Neural Solver for Math Word Problems*. CIKM 2017.
- **协议**：供学术研究使用
- **改写内容**：仅做格式转换
  - `id` → `math23k_001..030`
  - `task_type` → `"calculator"`
  - `expected_tool` → `"calculator_tool"`
  - `query` ← `original_text`
  - `expected_answer_keywords` ← `ans` 字段（最终数值）

## 引用方式

```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Hyung and Kaiser, Laurent and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@inproceedings{wang2017deep,
  title={Deep Neural Solver for Math Word Problems},
  author={Wang, Yan and Liu, Xiaojiang and Shi, Shuming},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={449--458},
  year={2017}
}
```

## 本地数据来源文件

- `./_public_data/gsm8k_test.jsonl` — 从 GitHub 仓库原始下载的 GSM8K 副本
- `./_public_data/math23k_test.json` — 从 GitHub 仓库原始下载的 Math23K 副本

## 当前局限

- **GSM8K 语言不匹配**：英文题进中文 router，预期 0 通过，揭示 i18n 缺陷。
- **Mock 工具能力有限**：`calculator_tool` 只做 AST 求值（`+ - * / **`），无法分步推算应用题，GSM8K/Math23K 多数需要多步推理，单步 mock calculator 难以命中。
- **数学符号触发边界**：Math23K 中文题里 `(3/8)`、`1/2` 这类分数若被 router 误判为带 `+ - * /` 算式，会路由到 calculator 并求值错误。
