# 评估协议 v3：与 Cooperative RAG / CoRAG 的对齐和差异

本文档用于避免把“答案正确率”“证据支持率”“JSON 合法性”和“成本”混成一个分数。

## 结论先行

我们的主指标现在是：

```text
accuracy = corag_style_accuracy
         = 100 * mean(exact_presence(gold_answers, generated final_answer))
```

其中 `exact_presence` 会先做小写、去标点、去冠词、规整空白，然后判断任一 gold answer
是否作为子串出现在模型生成的 `final_answer` 中。这个定义和 2026 年 Cooperative Retrieval-Augmented Generation（CoRAG）
在 PopQA、TriviaQA、NQ、2WikiMultiHopQA 上使用 accuracy 评估“生成输出是否包含
ground-truth response”的主评估方式对齐：主表只看最终答案是否命中，不让证据审计、
引用格式或 JSON 合法性掺进主准确率。

## 我们和 CoRAG 对齐的地方

1. **主分数是答案级硬匹配**：`accuracy` / `corag_style_accuracy` 只由 final answer 与 gold answer 的 normalized sub-string match 决定。
2. **gold answer 不进入生成 prompt**：gold 只在离线 verifier / metric 里使用。
3. **证据质量不 gate 主 accuracy**：答案对但证据不支持时，主 accuracy 仍算对；但会计入 `unsupported_answer_rate`。
4. **训练 reward 采用同一硬锚点**：CoRAG 的共享 task-oriented reward 是“生成 response 是否包含 ground-truth response”；我们也保留 EM/sub-string 作为硬 reward anchor。

## 和 CoRAG 不同、需要在论文里说清楚的地方

| 维度 | CoRAG | 我们 protocol v3 | 解释 |
|---|---|---|---|
| 主指标 | PopQA/TriviaQA/NQ/2WikiMultiHopQA 使用 accuracy，判断 ground-truth response 是否包含在 generated output 中；ASQA 报 em/pre/rec | 当前 PopQA 主表报 `accuracy` / `corag_style_accuracy` | 二者主表都走答案级硬匹配，不用 LLM judge gate 主分数。 |
| 训练 reward | task-oriented reward：response 包含 ground-truth response 得 1，否则 0 | GRPO reward 里保留 normalized EM/sub-string answer anchor，并额外分解 support/citation/cost | 我们的额外 reward 分解用于机制学习，但主 eval 不混入这些项。 |
| 输出形态 | generator 直接生成 response | 大模型输出审计 JSON，其中 `final_answer` 是主评估对象 | JSON 是我们方法的一部分，但格式合法性不混进 CoRAG-style accuracy。 |
| 格式诊断 | 主表不强调 JSON，因为 CoRAG 没有审计 JSON 输出 | `schema_valid_accuracy`、`audit_json_valid_rate`、`empty_answer_rate` | 用来判断审计器是否稳定，不用于和 CoRAG 主分数直接比较。 |
| 证据诊断 | 论文主要分析 top-N、组件消融和 LLM-as-judge 辅助评估 | `evidence_support_rate`、`citation_correctness`、`evidence_quote_support_rate`、`unsupported_answer_rate` | 这些是我们解释 reranker/auditor 协同质量的附加指标。 |
| LLM-as-judge | CoRAG 在 RQ5 里做了辅助 LLM-as-a-judge | 我们不做 LLM-as-judge | 这是有意选择：保持和 pattern-based 主指标一致，避免 judge 偏差和额外成本。 |
| Top-K 设置 | CoRAG 训练时为了归因使用更小的 top-K，推理分析 top-1/top-3/top-5；PopQA 主结果常看 top-3 | 我们保留配置化 top-k，并在 v3 结果里记录 top-k/cost | 对齐比较时要注明 train/eval top-k，不要把 top-1 和 top-3 混表。 |

## 指标字段解释

- `accuracy`：官方主指标；answer-only normalized EM/sub-string。
- `corag_style_accuracy`：`accuracy` 的同义字段，提醒这是可和 CoRAG 思路对齐的答案级指标。
- `schema_valid_accuracy`：答案命中且审计 JSON 合法、`final_answer` 非空的比例；只做格式诊断。
- `evidence_support_rate`：reranker 选中的证据里是否含 gold answer。
- `citation_correctness`：大模型引用的文档是否含 gold answer。
- `evidence_quote_support_rate`：大模型给出的 quote 是否能在原文中找到且含 gold answer。
- `unsupported_answer_rate`：答案命中但选中证据不支持的比例；这类样本主 accuracy 算对，但会暴露“靠参数知识答对”的问题。
- `audit_json_valid_rate` / `audit_schema_valid_rate`：审计 JSON 是否满足必需字段和枚举约束。

## 论文汇报建议

主表：

```text
accuracy / corag_style_accuracy
```

机制分析表：

```text
evidence_support_rate
citation_correctness
unsupported_answer_rate
schema_valid_accuracy
audit_json_valid_rate
avg_selected_docs / avg_generation_candidates
```

一句话版：

```text
For fair comparison with CoRAG-style RAG evaluation, we report answer-only
normalized exact/sub-string accuracy as the primary metric, while treating JSON
validity, evidence support, citation correctness, and inference cost as auxiliary
diagnostics rather than gating conditions for correctness.
```

参考：CoRAG（Cooperative Retrieval-Augmented Generation）把 RAG 形式化为
reranker 与 generator 的协同决策问题；其共享 reward 判断 generated response 是否
包含 ground-truth response，主实验对 PopQA/TriviaQA/NQ/2WikiMultiHopQA 报 accuracy，
ASQA 报 em/pre/rec，并另设 RQ5 的 LLM-as-a-judge 辅助评估。

- Paper: https://arxiv.org/abs/2602.18734
- HTML: https://arxiv.org/html/2602.18734
