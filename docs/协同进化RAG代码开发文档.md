# EvoCo-RAG 代码开发文档

> 阅读定位：本文是工程设计与实现记录，不是复现入口。快速复现、数据下载、
> 权重下载、冒烟测试和全量实验请统一查看 [README.md](../README.md)。

## 1. 开发目标

本文档面向当前 `CoRAG-D63F` 代码，将“证据合约驱动的大小模型协同进化 RAG”方案落地为可实现的工程计划。

当前代码已经具备一个基础闭环：

```text
reranker 选择 top1_doc
generator 基于 top1_doc 生成答案
answer-only reward 判断是否命中标准答案
命中则给 top1_doc 追加正标签，否则追加负标签
下一轮 reranker 使用新标签继续训练
```

目标是把这个粗粒度闭环升级为：

```text
小模型生成证据合约
大模型生成答案并审计证据
规则验证器校验答案和证据
责任归因模块拆分失败类型
replay buffer 保存结构化经验
小模型 LoRA 学证据选择和检索策略
大模型 LoRA 学忠实生成和审计格式
```

核心要求：

1. **不全量微调大小模型**：冻结 base model，只训练 LoRA adapter 和少量额外 head。
2. **不再只用答案命中作为 reward**：reward 必须拆成答案、证据、引用、置信度和成本。
3. **不再把生成成功直接奖励给检索器**：只有证据被审计为支持答案时，才给小模型正反馈。
4. **所有模型交互都结构化存档**：每轮 contract、answer、audit、reward、failure_type 都写入 replay buffer。

## 2. 当前代码基线

当前项目根目录是：

```text
/Users/quanquan/Desktop/rag_code/CoRAG-D63F 
```

注意：`CoRAG-D63F ` 目录名末尾有一个空格。

主要文件：

| 文件 | 当前职责 | 改造方向 |
|---|---|---|
| `run_train.py` | 训练入口，包含 reranker 训练、GRPO 训练和 reward | 拆分为配置、训练循环、contract 生成、reward、replay buffer |
| `run_test.py` | 加载 adapter，rerank top-3，调用大模型生成并算 accuracy | 增加证据指标、引用指标、action 成本指标 |
| `utils.py` | 数据处理、答案标准化、exact presence、metrics | 增加 JSON 解析、证据校验、校准指标 |
| `llm_local_prompt.py` | 本地大模型 batch generation | 增加结构化 JSON 输出解析和重试 |
| `../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data_v33/Pop/train_labels_list.json` | 训练数据，labels 为文档级历史标签列表 | 后续保留为 seed labels，新增 audited labels |
| `../rag_assets/adapters/generator-CoRAG` | 大模型 LoRA adapter | 可作为已有 adapter 参考 |
| `../rag_assets/adapters/reranker-CoRAG` | 小模型 LoRA adapter | 可作为已有 adapter 参考 |

当前外部依赖模型：

```text
小模型 base: ../rag_assets/base_models/reranker/bge-reranker-v2-m3
大模型 base: ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
大模型来源: mistralai/Mistral-Nemo-Instruct-2407（12B instruct generator）
```

当前工程已按代码和资产分离。`CoRAG-D63F ` 只保留源码、配置、测试和文档，后续可作为 GitHub 仓库；数据、旧 adapter、base model、checkpoint 和输出统一放在同级 `../rag_assets/`，不进入代码仓库。

## 3. 目标工程结构

建议新增一个独立包 `evoco_rag/`，避免继续扩大 `run_train.py`：

```text
CoRAG-D63F /
├── evoco_rag/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── data.py
│   ├── small_model.py
│   ├── large_model.py
│   ├── contract.py
│   ├── auditor.py
│   ├── verifier.py
│   ├── rewards.py
│   ├── replay_buffer.py
│   ├── weights.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── small_trainer.py
│   │   ├── large_trainer.py
│   │   └── coevolution_trainer.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── evaluator.py
├── scripts/
│   ├── train_evoco.py
│   ├── eval_evoco.py
│   ├── build_seed_replay.py
│   ├── run_ablations.py
│   └── inspect_replay.py
├── configs/
│   ├── evoco_popqa.yaml
│   └── debug.yaml
├── docs/
│   ├── 协同进化RAG代码开发文档.md
│   └── 协同进化RAG论文构想.md
└── tests/
    ├── test_schemas.py
    ├── test_rewards.py
    ├── test_verifier.py
    ├── test_replay_buffer.py
    └── test_weights.py
pytest.ini
```

第一版可以不一次性拆完，但新增代码应按这个边界组织。

## 4. 数据契约

### 4.1 输入样本

内部统一样本格式：

```json
{
  "sample_id": "popqa-train-000001",
  "question": "What is Henry Feilden's occupation?",
  "answers": ["politician", "political leader"],
  "documents": [
    {
      "doc_id": 0,
      "title": "Henry Wemyss Feilden",
      "text": "Colonel Henry Wemyss Feilden...",
      "raw": "title: Henry Wemyss Feilden\ncontext: Colonel..."
    }
  ],
  "seed_labels": [["0"], ["1"], ["0"]],
  "metadata": {
    "dataset": "Pop",
    "split": "train"
  }
}
```

现有 `context` 是字符串列表，第一版可以用 `raw` 保留原格式，并用简单规则切出 `title` 和 `text`。

### 4.2 证据合约 EvidenceContract

小模型输出：

```json
{
  "sample_id": "popqa-train-000001",
  "round": 1,
  "question": "What is Henry Feilden's occupation?",
  "answerability": "high",
  "retrieval_action": "answer_now",
  "selected_evidence": [
    {
      "doc_id": 4,
      "rank": 1,
      "doc_score": 7.82,
      "relevance_confidence": 0.91,
      "evidence_confidence": 0.84,
      "span": "Henry Master Feilden was an English Conservative Party politician.",
      "span_start": null,
      "span_end": null,
      "reason": "The document mentions the target entity and occupation."
    }
  ],
  "candidate_docs": [
    {
      "doc_id": 4,
      "rank": 1,
      "doc_score": 7.82
    },
    {
      "doc_id": 1,
      "rank": 2,
      "doc_score": 5.31
    }
  ],
  "uncertainty": {
    "entity_ambiguity": false,
    "evidence_conflict": false,
    "missing_relation": false
  },
  "cost": {
    "num_ranked_docs": 5,
    "num_selected_docs": 1,
    "num_retrieval_rounds": 1
  }
}
```

字段约束：

- `retrieval_action` 只允许：`answer_now`、`retrieve_more`、`rewrite_query`、`ask_auditor`。
- `answerability` 只允许：`high`、`medium`、`low`。
- `selected_evidence` 第一阶段可只做文档级或句子级，`span_start/span_end` 可为 `null`。
- `candidate_docs` 必须保留 top-k 排序，便于训练 hard negatives。

### 4.3 大模型审计 LargeAudit

大模型输出必须是可解析 JSON：

```json
{
  "sample_id": "popqa-train-000001",
  "round": 1,
  "final_answer": "politician",
  "used_doc_ids": [4],
  "used_evidence": [
    {
      "doc_id": 4,
      "quote": "Henry Master Feilden was an English Conservative Party politician."
    }
  ],
  "answer_correctness": "correct",
  "support_level": "fully_supported",
  "failure_type": "none",
  "small_model_feedback": [
    {
      "doc_id": 4,
      "label": "positive",
      "reason": "The selected evidence directly supports the answer."
    },
    {
      "doc_id": 1,
      "label": "negative",
      "reason": "The document is about a different entity."
    }
  ],
  "suggested_action": "answer_now"
}
```

字段约束：

- `answer_correctness` 只允许：`correct`、`incorrect`、`unknown`。
- `support_level` 只允许：`fully_supported`、`partially_supported`、`unsupported`。
- `failure_type` 只允许：`none`、`retrieval_miss`、`rerank_error`、`entity_confusion`、`evidence_conflict`、`generation_error`、`unsupported_answer`、`over_retrieval`。
- `small_model_feedback.label` 只允许：`positive`、`negative`、`hard_negative`、`ignore`。

### 4.4 规则验证 RuleVerification

规则验证器独立于大模型审计，防止大模型审计噪声直接污染训练：

```json
{
  "sample_id": "popqa-train-000001",
  "answer_match": true,
  "cited_doc_contains_answer": true,
  "used_doc_in_selected_evidence": true,
  "support_rule_passed": true,
  "json_valid": true,
  "audit_trust_weight": 0.9,
  "notes": []
}
```

第一版规则：

1. `answer_match`：沿用当前 `exact_presence(answers, final_answer)`。
2. `cited_doc_contains_answer`：检查 gold answer 是否出现在 `used_doc_ids` 对应原文中。
3. `used_doc_in_selected_evidence`：大模型引用文档是否来自小模型合约。
4. `audit_trust_weight`：JSON 合法、答案匹配、引用文档包含答案时提高权重，否则降低权重。

### 4.5 ReplayExperience

每个训练样本每轮写一条 JSONL：

```json
{
  "sample_id": "popqa-train-000001",
  "round": 1,
  "question": "...",
  "answers": ["politician"],
  "documents": [{"doc_id": 0, "raw": "..."}],
  "contract": {},
  "audit": {},
  "verification": {},
  "rewards": {
    "answer_reward": 1.0,
    "support_reward": 1.0,
    "citation_reward": 1.0,
    "calibration_reward": 0.2,
    "cost_penalty": 0.1,
    "total_reward": 3.1
  },
  "training_targets": {
    "small_positive_doc_ids": [4],
    "small_negative_doc_ids": [1, 2],
    "small_action_target": "answer_now",
    "large_sft_eligible": true,
    "large_grpo_reward": 3.1
  }
}
```

## 5. 模块设计

### 5.1 `schemas.py`

职责：

- 定义 dataclass 或 Pydantic-like 简单 schema。
- 提供 `from_dict`、`to_dict`、`validate`。
- 第一版不强依赖 Pydantic，避免额外依赖；可以用标准库 `dataclasses` 和显式校验。

建议对象：

```text
RagSample
EvidenceItem
EvidenceContract
LargeAudit
RuleVerification
RewardBreakdown
ReplayExperience
```

验收标准：

- 所有 schema 能从 JSON dict 构造。
- 缺失关键字段时抛出明确错误。
- action、failure_type、support_level 等枚举值必须校验。

### 5.2 `data.py`

职责：

- 读取当前 `train_labels_list.json` 和 `test.json`。
- 转换为统一 `RagSample`。
- 支持按 `debug_size` 截断。
- 生成稳定 `sample_id`。

需要兼容两类数据：

```text
训练数据：question / answers / context / labels
测试数据：question / answers / ctxs
```

### 5.3 `small_model.py`

职责：

- 加载 `bge-reranker-v2-m3`。
- 加载或初始化 LoRA。
- 对 `(question, doc)` 批量打分。
- 生成 top-k candidate docs。
- 第一阶段用启发式 sentence selection 生成 evidence span。
- 后续阶段增加 evidence head 和 action head。

第一版接口：

```python
class SmallRagPolicy:
    def rank_documents(self, sample, top_k: int) -> list[dict]:
        ...

    def build_contract(self, sample, round_id: int, top_k: int) -> EvidenceContract:
        ...
```

后续接口：

```python
class SmallRagPolicy:
    def predict_action(self, sample, ranked_docs) -> str:
        ...

    def predict_evidence_confidence(self, sample, doc) -> float:
        ...
```

### 5.4 `contract.py`

职责：

- 将小模型 scores 转换为 EvidenceContract。
- 控制 top-k、置信度、action。
- 第一版 action 规则可以是启发式：

```text
最高分置信度 >= high_threshold 且 top1-top2 margin 足够大 → answer_now
最高分较低 → retrieve_more
top1/top2 分数接近且实体不同 → ask_auditor
```

### 5.5 `large_model.py`

职责：

- 加载 `mistralai/Mistral-Nemo-Instruct-2407`。
- 加载或初始化 LoRA。
- 批量生成。
- 支持 train mode 和 eval mode。

接口：

```python
class LargeGeneratorAuditor:
    def generate_audit(self, sample, contract) -> LargeAudit:
        ...
```

注意：

- prompt 必须强制 JSON 输出。
- 需要提取 JSON 的 robust parser，处理大模型输出多余文本。
- 解析失败时最多重试 2-3 次；仍失败则生成 `json_valid=false` 的 fallback audit。

### 5.6 `auditor.py`

职责：

- 构造大模型审计 prompt。
- 解析大模型输出为 `LargeAudit`。
- 对不合法字段做降级。

审计 prompt 必须包含：

1. 问题；
2. 标准答案仅训练时可见，测试时不可见；
3. 小模型 selected evidence；
4. top-k candidate docs；
5. JSON schema；
6. failure_type 定义。

训练阶段可以让大模型看到 gold answers 来做 teacher audit；评估阶段不能把 gold answers 放入生成 prompt。

### 5.7 `verifier.py`

职责：

- 使用规则验证答案、引用和证据。
- 输出 `RuleVerification`。
- 给大模型审计结果分配 `audit_trust_weight`。

第一版规则足够简单，但必须独立于大模型。

### 5.8 `rewards.py`

职责：

- 从 sample、contract、audit、verification 计算分解 reward。
- 构建大小模型训练 target。

建议第一版 reward：

```text
answer_reward = 1.0 if answer_match else 0.0
support_reward = 1.0 if support_rule_passed and support_level == fully_supported else 0.0
citation_reward = 1.0 if cited_doc_contains_answer else 0.0
calibration_reward = 0.2 if confidence bucket matches outcome else -0.2
cost_penalty = 0.05 * num_selected_docs + 0.1 * num_retrieval_rounds
total_reward = answer_reward + support_reward + citation_reward + calibration_reward - cost_penalty
```

责任归因规则：

| 条件 | 小模型训练 | 大模型训练 |
|---|---|---|
| answer_match=true 且 support_rule_passed=true | 正奖励 used docs，负样本为未用文档 | SFT/GRPO 正样本 |
| answer_match=true 且 support_rule_passed=false | 不奖励小模型，标记 unsupported_answer | 训练引用忠实性 |
| answer_match=false 且 support_rule_passed=true | 奖励小模型证据，训练大模型生成 | 负 reward 或 SFT 修正 |
| answer_match=false 且 support_rule_passed=false | 小模型负反馈，必要时 retrieve_more | 低权重训练或丢弃 |

### 5.9 `replay_buffer.py`

职责：

- 写入 JSONL replay。
- 按 round、dataset、failure_type 过滤。
- 采样 high-confidence positives 和 hard negatives。
- 防止低质量自训练数据无限累积。

文件建议：

```text
../rag_assets/outputs/evoco_popqa/replay/round_000.jsonl
../rag_assets/outputs/evoco_popqa/replay/round_001.jsonl
../rag_assets/outputs/evoco_popqa/replay/all.jsonl
```

### 5.10 `small_trainer.py`

职责：

- 从 replay buffer 构造 reranker 训练 batch。
- 沿用当前 `RankingLoss`。
- 增加 evidence/action 训练时，扩展多任务 loss。

第一版只实现：

```text
audited positive docs vs audited negative docs
```

后续再加：

```text
evidence confidence BCE
action policy CE
calibration loss
```

### 5.11 `large_trainer.py`

职责：

- 用 replay buffer 中高质量样本训练大模型 LoRA。
- 支持两种模式：

```text
SFT：学习结构化答案和审计格式
GRPO：使用 decomposed reward 优化生成
```

第一版可以保留当前 `GRPOTrainer`，但 `reward_funcs` 改成读取结构化 audit 和 verification。

### 5.12 `coevolution_trainer.py`

职责：

- 调度一整轮协同进化。
- 负责 round 级别的数据流、checkpoint、metrics。

伪代码：

```python
for round_id in range(num_rounds):
    samples = data_loader.load_train_samples()

    for batch in batches(samples):
        contracts = small_policy.build_contracts(batch)
        audits = large_auditor.generate_audits(batch, contracts)
        verifications = verifier.verify(batch, contracts, audits)
        experiences = rewards.build_experiences(batch, contracts, audits, verifications)
        replay_buffer.write(experiences)

    small_trainer.train(replay_buffer, round_id)
    large_trainer.train(replay_buffer, round_id)
    evaluator.evaluate(round_id)
    save_checkpoints(round_id)
```

## 6. 训练阶段规划

### 阶段 0：工程清理和可运行基线

目标：

- 保证原始 CoRAG 能以 debug 数据跑通一小轮。
- 修复明显路径问题。
- 统一配置入口。

任务：

1. 增加 `configs/debug.yaml`。
2. 增加 `scripts/train_baseline_debug.py` 或给现有 `run_train.py` 增加参数化入口。
3. 修复 `run_test.py` 中旧式本地数据路径与 `../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data/Pop/test.json` 的不一致。
4. 删除或忽略 `__pycache__`。
5. 输出 baseline 指标：accuracy、Recall@k、answer-in-context rate。

验收：

```text
debug_size=16
num_generations=2
训练和测试流程能跑完
输出 baseline metrics JSON
```

### 阶段 1：证据合约和 replay buffer

目标：

- 不改变模型训练，先让小模型输出 EvidenceContract。
- 大模型仍生成普通答案，但所有中间结果写入 replay buffer。

任务：

1. 实现 `schemas.py`。
2. 实现 `data.py`。
3. 实现 `small_model.py` 的 `rank_documents`。
4. 实现 `contract.py` 的 `build_contract`。
5. 实现 `replay_buffer.py`。
6. 新增 `scripts/build_seed_replay.py`。

验收：

```text
对 16 条样本生成合法 EvidenceContract
写入 ../rag_assets/outputs_debug/latest/replay/round_000.jsonl
schema validation 全部通过
```

### 阶段 2：大模型结构化审计

目标：

- 大模型输出 JSON 格式的答案和审计。
- 解析失败可重试和降级。

任务：

1. 实现 `auditor.py`。
2. 实现 `large_model.py` 的 `generate_audit`。
3. 实现 JSON 提取和字段校验。
4. 实现 `verifier.py`。

验收：

```text
大模型输出 JSON parse success rate >= 90%
每条样本都有 LargeAudit 和 RuleVerification
解析失败样本不会中断训练流程
```

### 阶段 3：分解 reward 和责任归因

目标：

- 替换当前 answer-only reward。
- 构造大小模型各自训练 target。

任务：

1. 实现 `rewards.py`。
2. 支持 `RewardBreakdown`。
3. 支持 `training_targets`。
4. 把当前 `myReward` 拆成：

```text
answer_reward
support_reward
citation_reward
calibration_reward
cost_penalty
```

验收：

```text
单元测试覆盖四种责任归因情况
answer=true/support=false 时 small_positive_doc_ids 为空
answer=false/support=true 时 small_positive_doc_ids 不为空
```

### 阶段 4：小模型 LoRA 自进化

目标：

- 小模型使用 audited feedback 训练，而不是只用原始 labels。

任务：

1. 实现 `small_trainer.py`。
2. 从 replay buffer 构造 positive/negative docs。
3. 沿用或迁移当前 `RankingLoss`。
4. 保存小模型 LoRA checkpoint。
5. 对比当前 CoRAG reranker label append 方案。

验收：

```text
小模型训练 loss 正常下降
Recall@1 / MRR 相比 baseline 不下降
unsupported_answer 对小模型无正奖励
```

### 阶段 5：大模型 LoRA 自进化

目标：

- 大模型学习基于证据回答，并稳定输出审计 JSON。

任务：

1. 实现 `large_trainer.py`。
2. 从 replay buffer 中筛选 `large_sft_eligible=true` 的样本。
3. 训练生成答案、used_doc_ids、support_level、failure_type。
4. 可选：接入 GRPO，用 `total_reward` 优化。

验收：

```text
JSON parse success rate 提升
supported answer rate 提升
unsupported answer rate 下降
```

### 阶段 6：完整协同进化循环

目标：

- 大小模型每轮交替更新，形成可复现实验。

任务：

1. 实现 `coevolution_trainer.py`。
2. 新增 `scripts/train_evoco.py`。
3. 每轮输出 metrics、replay、checkpoint。
4. 支持 resume。

验收：

```text
round_0 → round_1 → round_2 可连续运行
每轮 checkpoint 可加载
每轮 metrics 可比较
```

## 7. 评估设计

### 7.1 必须输出的指标

答案：

```text
accuracy
exact_match
f1 可选
```

检索：

```text
Recall@1
Recall@3
MRR
answer_in_topk_context_rate
```

证据：

```text
evidence_support_rate
citation_correctness
used_doc_precision
unsupported_answer_rate
```

策略成本：

```text
avg_selected_docs
avg_ranked_docs
audit_call_rate
cost_per_correct_answer
```

校准：

```text
confidence_success_correlation
ECE 可选
```

### 7.2 消融实验

必须保留可配置开关：

```yaml
use_evidence_audit: true
use_action_policy: true
use_decomposed_reward: true
train_small_lora: true
train_large_lora: true
```

实验矩阵：

| 配置 | 目的 |
|---|---|
| baseline_current_corag | 当前 answer-only reward |
| evoco_no_audit | 验证审计机制贡献 |
| evoco_no_action | 验证动态 action 贡献 |
| evoco_answer_only_reward | 验证 reward 拆解贡献 |
| evoco_small_only | 只训练小模型 LoRA |
| evoco_large_only | 只训练大模型 LoRA |
| evoco_full | 完整方案 |

当前 PopQAStandard 官方实验已拆成 full-data launcher spec；fast spec 只保留作调试：

- `configs/experiments/popqa_sweep_full_2gpu.yaml`：full PopQA top-k、reward、audit 开关 sweep。
- `configs/experiments/popqa_hparam_full_2gpu.yaml`：full PopQA 超参探索，用于筛 top-k、audit candidates、置信阈值和上下文长度。
- `configs/experiments/popqa_ablation_full_2gpu.yaml`：full 3-round 机制消融，用于论文主表，包括 `evoco_full`、`answer_only_reward`、`no_audit`、`no_action_policy`、`no_policy_heads`、`small_only`、`large_only`、`baseline_current_corag`。

## 8. 配置设计

建议 `configs/evoco_popqa.yaml`：

```yaml
project:
  name: evoco_rag_popqa
  seed: 42
  output_dir: ../rag_assets/outputs/evoco_popqa

data:
  train_path: ../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data_v33/Pop/train_labels_list.json
  test_path: ../rag_assets/rag_data/evoco_dataset_pack/datasets/popqa_standard/data/Pop/test.json
  dataset_name: Pop
  debug_size: null

models:
  small_base_path: ../rag_assets/base_models/reranker/bge-reranker-v2-m3
  large_base_path: ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
  small_lora_dir: ../rag_assets/checkpoints/evoco_popqa/small
  large_lora_dir: ../rag_assets/checkpoints/evoco_popqa/large

contract:
  top_k: 5
  high_conf_threshold: 0.75
  answer_now_margin: 0.15
  max_selected_docs: 5

training:
  num_rounds: 3
  batch_size: 4
  large_batch_size: 2
  num_generations: 2
  small_lr: 5.0e-5
  large_lr: 1.0e-5
  train_small_lora: true
  train_large_lora: true

reward:
  answer_weight: 1.0
  support_weight: 1.0
  citation_weight: 1.0
  calibration_weight: 0.2
  selected_doc_cost: 0.05
  retrieval_round_cost: 0.1

runtime:
  candidate_doc_char_limit: 1200
  num_audit_candidates: 3
  audit_batch_size: 2
  audit_temperature: 0.7
  max_prompt_length: 3072
  max_completion_length: 1024
  progress_interval: 50
  replay_flush_interval: 10
```

完整 PopQA 首轮包含上万条样本，并且每条样本会触发大模型审计生成。推荐先使用
`configs/evoco_popqa_fast.yaml` 做 512 条样本、单审计候选、单 round 的快速验证：

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/evoco_popqa_fast.yaml
```

训练主循环会边生成边写入 `replay/round_000.jsonl`、`contracts/round_000.jsonl`
和 `audits/round_000.jsonl`，并按 `runtime.progress_interval` 打印
`round 0: experience 500/12868 elapsed=... rate=... eta=...` 形式的进度。
如果完整训练看起来很久没有结束，先检查进度日志和 replay 文件行数。

影响训练吞吐的两个主要 batch 参数：

| 配置 | 作用 | 默认建议 |
|---|---|---|
| `runtime.audit_batch_size` | 大模型审计生成的 batch size，训练阶段和测试集 `run_inference` 都会使用 | 2 H20 先用 `2`，显存稳定后可试 `4` |
| `training.large_batch_size` | 大模型 LoRA SFT 的 batch size | 先用 `2`，如果 SFT 显存稳定再上调 |

这两个参数主要改善吞吐，不直接改变精度目标。精度相关参数仍是
`contract.top_k`、`contract.max_selected_docs`、`runtime.num_audit_candidates`
和 `runtime.candidate_doc_char_limit`。

如果训练进程在 experience generation 阶段中断，重新执行同一条训练命令时会读取
已有 `replay/round_xxx.jsonl` 中合法的样本行，跳过已经完成的 `sample_id`，并重建
`contracts/round_xxx.jsonl` 与 `audits/round_xxx.jsonl`。损坏或半截 JSON 行会被跳过。
这属于同一 round 内的 partial resume；`train_evoco.py --resume` 则用于已经完成
`round_*` checkpoint 后从下一轮继续。

每轮 `metrics/round_xxx.json` 会记录阶段耗时：

```json
"timing": {
  "experience_generation_seconds": 123.4,
  "small_training_seconds": 12.3,
  "large_training_seconds": 45.6,
  "evaluation_seconds": 1.2,
  "total_round_seconds": 182.5
}
```

## 9. 大小模型权重与 checkpoint 布局

权重文件必须分为三类，不能混用：

| 类型 | 目录 | 说明 |
|---|---|---|
| 小模型 base | `../rag_assets/base_models/reranker/bge-reranker-v2-m3` | 不放在代码仓库里，训练时只读取，不覆盖 |
| 大模型 base | `../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407` | 不放在代码仓库里，训练时只读取，不覆盖 |
| 旧版小模型 adapter | `../rag_assets/adapters/reranker-CoRAG` | 当前压缩包自带 LoRA，可作为已有结果或迁移参考 |
| 旧版大模型 adapter | `../rag_assets/adapters/generator-CoRAG` | 当前压缩包自带 LoRA，可作为已有结果或迁移参考 |
| EvoCo 小模型 checkpoint | `../rag_assets/checkpoints/evoco_popqa/small/round_000` | 新方案每轮保存的小模型 LoRA |
| EvoCo 大模型 checkpoint | `../rag_assets/checkpoints/evoco_popqa/large/round_000` | 新方案每轮保存的大模型 LoRA |
| debug 小模型 checkpoint | `../rag_assets/checkpoints/debug/small/round_000` | debug 配置使用 |
| debug 大模型 checkpoint | `../rag_assets/checkpoints/debug/large/round_000` | debug 配置使用 |

实现上已经增加 `evoco_rag/weights.py`，负责统一处理权重路径：

```text
is_lora_adapter_dir(path)
latest_round_adapter(root)
latest_checkpoint_round(root)
resolve_adapter_for_loading(path_or_root)
checkpoint_round_dir(root, round_id)
prepare_weight_layout(config)
write_weight_manifest(config)
```

关键规则：

1. `../rag_assets/checkpoints/.../small` 和 `../rag_assets/checkpoints/.../large` 是 checkpoint root，不是 adapter 本身。
2. 真正可加载的 adapter 必须是 `round_000` 这种子目录，并且包含 `adapter_config.json` 和 `adapter_model.safetensors` 或 `adapter_model.bin`。
3. `train_evoco.py` 默认新训 fresh LoRA；如果 checkpoint root 已经有 `round_*`，但没有传 `--resume`，脚本会直接退出，防止误覆盖或混用旧权重。
4. `train_evoco.py --resume` 会自动加载 checkpoint root 下最新的完整 `round_*` adapter，并从下一轮继续训练。
5. `eval_evoco.py` 可以接收具体 adapter 目录，也可以接收 checkpoint root；如果传 root，会自动解析最新 `round_*`。
6. 每次运行会写出 `weights_manifest.json`，记录 base model 路径、checkpoint root、最新 adapter 和 latest round，作为上机复现实验的权重依据。
7. 消融实验每个实验拥有独立输出目录和 checkpoint root，例如 `../rag_assets/outputs/evoco_popqa/ablations/evoco_full/`，避免不同实验覆盖同一套 LoRA。

推荐上机前检查：

```bash
python - <<'PY'
from evoco_rag.config import EvoCoConfig
from evoco_rag.weights import prepare_weight_layout
cfg = EvoCoConfig.load("configs/evoco_popqa.yaml")
print(prepare_weight_layout(cfg, create=True))
PY
```

## 10. 与现有代码的迁移关系

### 10.1 `compute_reward`

当前：

```text
normalize response
如果 final answer 中包含 gold answer，则 reward=1，否则 reward=0
```

迁移后：

```text
compute_reward → rewards.compute_decomposed_reward
```

输入从纯文本 completion 变为：

```text
sample + contract + audit + verification
```

### 10.2 `myReward`

当前：

```text
GRPO reward 函数
同时负责 reward 计算和 labels 更新
```

迁移后拆分：

```text
GRPO reward 只返回 large-model reward
ReplayBuffer 负责记录经验
SmallTrainer 负责从 replay buffer 更新小模型标签和 loss
```

原则：reward 函数不要直接写训练数据文件，避免副作用难以复现。

### 10.3 `reranker_training`

当前：

```text
根据 labels 随机采样正负文档
计算 ranking loss
保存 top1_doc 给 generator
```

迁移后：

```text
SmallTrainer 使用 audited positive/negative docs 训练
SmallRagPolicy 负责生成 EvidenceContract
```

### 10.4 `run_test`

当前：

```text
top3 文档 + generator 答案 + answer accuracy
```

迁移后：

```text
contract + audit + verification + 多维 metrics
```

测试阶段不能把 gold answers 放进大模型 prompt，只能用于离线 metrics。

## 11. 实现优先级

建议按以下顺序做，避免一开始就重写全部训练脚本：

1. **先实现 schema 和 replay buffer**：没有结构化数据，协同进化不可控。
2. **再实现 contract 生成**：复用当前 reranker scoring，不训练新 head。
3. **再实现大模型 JSON 审计**：先解决输出可解析问题。
4. **再实现 decomposed reward**：用规则验证降低审计噪声。
5. **最后接入双 LoRA 训练**：先训练小模型，再训练大模型，再做完整闭环。

第一版最小可行版本：

```text
固定小模型 LoRA
固定大模型 LoRA
生成 contract + audit + verification + decomposed reward
证明责任归因机制可运行
```

第二版：

```text
只训练小模型 LoRA
验证 reranker 指标提升和错误奖励下降
```

第三版：

```text
训练大小模型 LoRA
验证完整协同进化
```

## 12. 单元测试计划

### 12.1 Schema 测试

文件：

```text
tests/test_schemas.py
```

覆盖：

- 合法 EvidenceContract 可以通过校验。
- 非法 `retrieval_action` 抛错。
- 非法 `failure_type` 抛错。
- 缺失 `sample_id` 抛错。

### 12.2 Reward 测试

文件：

```text
tests/test_rewards.py
```

覆盖四种核心责任归因：

```text
answer=true, support=true
answer=true, support=false
answer=false, support=true
answer=false, support=false
```

重点断言：

- `answer=true/support=false` 时不能给小模型 positive doc。
- `answer=false/support=true` 时可以给小模型 positive doc，但大模型 reward 低。

### 12.3 Verifier 测试

文件：

```text
tests/test_verifier.py
```

覆盖：

- final_answer 命中 answers。
- used_doc 包含答案。
- used_doc 不在 selected_evidence 中。
- JSON 解析失败时 fallback 行为。

### 12.4 ReplayBuffer 测试

文件：

```text
tests/test_replay_buffer.py
```

覆盖：

- 写入 JSONL。
- 读取 JSONL。
- 按 `failure_type` 过滤。
- 按 `audit_trust_weight` 过滤。

## 13. 日志与可复现性

每次训练必须保存：

```text
../rag_assets/outputs/evoco_popqa/metrics/round_{round_id}.json
../rag_assets/outputs/evoco_popqa/replay/round_{round_id}.jsonl
../rag_assets/outputs/evoco_popqa/contracts/round_{round_id}.jsonl
../rag_assets/outputs/evoco_popqa/audits/round_{round_id}.jsonl
../rag_assets/checkpoints/evoco_popqa/small/round_{round_id}/
../rag_assets/checkpoints/evoco_popqa/large/round_{round_id}/
../rag_assets/outputs/evoco_popqa/used_config.yaml
../rag_assets/outputs/evoco_popqa/weights_manifest.json
```

每条 replay 必须包含：

```text
sample_id
round
contract
audit
verification
rewards
training_targets
```

禁止只保存聚合指标而不保存中间样本。否则无法分析协同进化是否真的发生。

## 14. 关键工程风险

### 14.1 大模型 JSON 输出不稳定

处理策略：

- prompt 中明确“只输出 JSON，不输出 Markdown”。
- 使用 JSON schema 示例。
- 用正则提取第一个 `{...}` 块。
- 解析失败重试。
- 最终失败写 fallback audit，不中断训练。

### 14.2 审计反馈污染小模型

处理策略：

- `audit_trust_weight < threshold` 的样本不进入小模型正样本池。
- 高置信 positive 必须同时满足：答案命中、引用文档包含答案、support_level 合格。
- 保留原始 seed labels 作为 anchor，避免自训练漂移。

### 14.3 训练脚本继续膨胀

处理策略：

- `scripts/train_evoco.py` 只负责 CLI 和 trainer 调用。
- 训练逻辑放进 `coevolution_trainer.py`。
- reward、verifier、replay buffer 不允许写在同一个函数里。

### 14.4 数据路径和模型路径混乱

处理策略：

- 所有路径只从 config 读取。
- 禁止在核心模块里硬编码 `../reranker`、`data_v33` 或仓库内 `model/`。
- debug config 和 full config 分开。
- 权重路径只通过 `evoco_rag.weights` 解析，训练脚本不能直接把 checkpoint root 传给 PEFT。

## 15. 开发里程碑

| 里程碑 | 交付物 | 判断标准 |
|---|---|---|
| M0 | baseline debug 跑通 | 16 条样本训练/评估可完成 |
| M1 | schema + replay buffer | replay JSONL 可验证、可读取 |
| M2 | evidence contract | 每条样本有 top-k、confidence、action |
| M3 | large audit | 大模型 JSON 审计 parse success rate >= 90% |
| M4 | decomposed reward | 四类责任归因单测通过 |
| M5 | small LoRA evolution | reranker 用 audited labels 训练 |
| M6 | large LoRA evolution | generator 学会结构化答案和审计 |
| M7 | full co-evolution | 多轮训练、checkpoint、metrics 完整输出 |

## 16. 最小实现路线

如果只做一版能体现创新点的原型，建议范围控制为：

1. 不增加 action head，只用启发式 action。
2. 不做 span-level 训练，只做 document-level evidence contract。
3. 大模型审计训练阶段可见 gold answers，测试阶段不可见。
4. 小模型先只训练 ranking LoRA。
5. 大模型先只保留现有 GRPO 训练，reward 换成 decomposed reward。

这样最短路径是：

```text
EvidenceContract
LargeAudit
RuleVerification
DecomposedReward
ReplayBuffer
Audited reranker training
```

这已经能体现论文核心创新：**小模型提出证据，大模型审计证据，系统按责任归因分别训练大小模型。**

## 17. 实现状态（已落地代码）

本节记录截至当前已实现的代码，对应 `CoRAG-D63F /evoco_rag/` 包。核心层（不依赖真实模型权重）已通过本机回归测试，并用 PopQA fast 配置跑通完整 no-model 消融。

### 17.1 已实现模块与对应章节

| 文件 | 对应章节 | 状态 |
|---|---|---|
| `evoco_rag/schemas.py` | §4、§5.1 | 完成：数据契约 + 枚举校验 |
| `evoco_rag/text_utils.py` | §4.4 | 完成：自包含 normalize/exact_presence + 启发式 span |
| `evoco_rag/data.py` | §5.2 | 完成：train/test 转 `RagSample` |
| `evoco_rag/contract.py` | §4 实现说明、§5.4 | 完成：打分转合约，sigmoid 置信度 + 启发式 action |
| `evoco_rag/verifier.py` | §4.4、§5.7 | 完成：规则验证 + `audit_trust_weight` |
| `evoco_rag/rewards.py` | §5.8、§7 | 完成：分解 reward + 四象限责任归因 |
| `evoco_rag/replay_buffer.py` | §5.9 | 完成：JSONL 读写、过滤、采样、降噪、`all.jsonl` 去重重建 |
| `evoco_rag/weights.py` | §9 | 完成：base/adapter/checkpoint root 解析、latest round、manifest |
| `evoco_rag/config.py` | §8 | 完成：yaml/json 配置 |
| `evoco_rag/auditor.py` | §5.6、§14.1 | 完成：prompt 构造 + robust JSON 提取、降级、兜底 |
| `evoco_rag/small_model.py` | §5.3 | 完成：rank、contract、启发式 evidence/action 接口；torch 延迟导入 |
| `evoco_rag/large_model.py` | §5.5 | 完成：bf16 默认、4bit 可选、JSON 重试、批量 audit generation；torch 延迟导入 |
| `evoco_rag/trainers/small_trainer.py` | §5.10、§10.3 | 完成：文档级 ranking LoRA 训练 |
| `evoco_rag/trainers/large_trainer.py` | §5.11、§10.1/§10.2 | 完成：批量 SFT + GRPO reward 函数，reward 无文件副作用 |
| `evoco_rag/trainers/coevolution_trainer.py` | §5.12 | 完成：单轮/多轮调度、消融开关、批量 audit、流式 contracts/audits/replay 写入、partial resume、阶段耗时、metrics/checkpoint |
| `evoco_rag/evaluation/metrics.py` | §7.1 | 完成：答案、检索、证据、成本、校准指标 |
| `evoco_rag/evaluation/evaluator.py` | §7、§10.4 | 完成：离线 evaluate + 批量测试集 run_inference，测试不见 gold |
| `scripts/build_seed_replay.py` | §6 阶段1 | 完成：纯 CPU 生成 seed replay + contracts/audits/manifest |
| `scripts/train_evoco.py` | §6 阶段6、§9 | 完成：训练入口、fresh/resume 保护、权重 manifest |
| `scripts/eval_evoco.py` | §7、§9 | 完成：测试集评估入口、adapter root/latest 解析 |
| `scripts/inspect_replay.py` | §13 | 完成：replay 分布与指标查看 |
| `scripts/run_ablations.py` | §7.2 | 完成：实验矩阵跑批、独立 checkpoint root、汇总对比表 |
| `configs/evoco_popqa.yaml`、`configs/debug.yaml`、`configs/experiments/*.yaml` | §8 | 完成：全量、debug、policy、多数据集 fast、PopQA full/ablation/hparam study specs |
| `tests/test_*.py`、`pytest.ini` | §12 | 完成：CPU 环境 80 passed, 4 skipped；GPU/torch 环境预期 84 passed |

### 17.2 与文档约定的两处对齐

1. **support_rule_passed 取独立轴语义**：定为"小模型选中证据里确实包含 gold answer"（检索/重排成功），与答案对错解耦，从而 §5.8 的"答案错/证据对"象限成立。`cited_doc_contains_answer` 单独表示大模型引用忠实性。
2. **证据合约分阶段**：第一阶段 `span/action/answerability/confidence` 由 `contract.py` 启发式封装、不引入新可训练参数；与论文构想 §4「实现说明」一致。

### 17.3 当前原型范围外的后续增强

当前版本完整实现 document-level EvoCo-RAG 闭环，并已补齐小模型 policy heads、多任务 loss、批量 audit/SFT、多数据集配置生成和官方实验矩阵。以下属于后续增强，不影响当前代码按照“证据合约 + 审计 + 规则验证 + 分解 reward + replay + 双 LoRA 训练入口”的主链路运行：

- token/span 级证据选择：当前为文档级 + 句子启发式 span，后续按 §14.2 的渐进路线升级到可验证 span。
- 偏好/奖励优化：当前提供批量 SFT、无副作用 reward 函数和 GRPO reward 接口，完整 DPO/ORPO/GRPO rollout 留到 GPU 环境联调。
- 多轮稳定性控制：当前支持 round checkpoint、trend plots 和 partial resume，后续补 drift monitor、early stop 和 rollback。
- 结果汇总工具：当前已有多 study launcher，后续补统一 `collect_results.py` 生成论文表格。

### 17.4 本机已验证 / 待 GPU 验证

- 本机 CPU 已验证：schema 校验、四象限归因、verifier、replay、JSON 解析、权重路径解析、指标、batch audit 路径、batch SFT 假模型路径、完整 no-model 消融跑批。
- 待 H20 验证：reranker/LLM 加载、bf16 训练、真实审计 JSON 成功率、双 LoRA 多轮协同进化收敛和完整 full 消融表。
  入口：先生成 `configs/local/*.yaml`，再运行 `CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/local/popqa_standard_debug.yaml`（16 条），通过后切 `configs/local/popqa_standard_fast.yaml`（512 条快速验证），最后用 `bash scripts/launch_all_experiments.sh` 跑官方实验矩阵。

## 18. 论文创新 TODO 对应的代码开发 TODO

本节与论文构想文档 §13 使用同一套编号 `ECR-*`。原则是：论文中每一个待完成创新点，都必须有明确的代码模块、方法入口、配置项、测试和验收指标；没有代码入口的创新点不能作为主贡献声称。

### 18.1 总体对应表

| 编号 | 论文创新点 | 代码开发目标 | 当前状态 | 主要文件 | 验收指标 |
|---|---|---|---|---|---|
| ECR-1 | 可训练的小模型证据-动作-置信度策略 | 在 reranker encoder 上增加 evidence/action/confidence heads，并接入多任务训练 | **代码完成，待实验成表**：head、配置、保存/加载、manifest、多任务 loss 和训练指标均已落地；生成的 dataset configs 默认启用 policy heads | `evoco_rag/small_model.py`、`evoco_rag/trainers/small_trainer.py`、`evoco_rag/config.py`、`evoco_rag/schemas.py` | 用 `configs/experiments/popqa_ablation_full_2gpu.yaml` 的 `evoco_full` vs `no_policy_heads` 和 fast debug 结果形成 H20 表 |
| ECR-2 | 责任归因式 credit assignment | 将四象限归因扩展为可统计、可消融、可追踪的训练信号 | **已完成代码落地**：归因字段、credit weight、误奖励率、metrics/replay/inspect 输出已实现 | `evoco_rag/rewards.py`、`evoco_rag/verifier.py`、`evoco_rag/replay_buffer.py`、`evoco_rag/evaluation/metrics.py` | 还需真实模型消融表验证 answer-only reward 对照 |
| ECR-3 | 可靠审计与抗噪声自训练 | 增强审计一致性、trust weight、人工抽样导出和 replay 降噪 | **已完成核心统计**：多候选摘要、`self_consistency`、`trust_components`、trust summary 已实现 | `evoco_rag/large_model.py`、`evoco_rag/auditor.py`、`evoco_rag/verifier.py`、`evoco_rag/replay_buffer.py` | 还需人工抽样导出和人工一致率评估 |
| ECR-4 | 成本感知动态检索动作 | 让 `answer_now/retrieve_more/rewrite_query/ask_auditor` 从启发式逐步变为可学习策略 | **代码完成，待实验成表**：已加入 `action_mode=heuristic/policy/hybrid`、policy action 置信覆盖、action cost penalty 和 accuracy-cost Pareto 指标 | `evoco_rag/contract.py`、`evoco_rag/small_model.py`、`evoco_rag/trainers/small_trainer.py`、`evoco_rag/evaluation/metrics.py` | 用 `configs/experiments/popqa_hparam_full_2gpu.yaml` 的 action confidence sweep 和 `popqa_ablation_full_2gpu.yaml` 的 `no_action_policy` 对比 accuracy-cost Pareto |
| ECR-5 | 多粒度证据合约 | 从 document-level 扩展到 sentence/span-level，并支持多跳证据组合 | **待做**：当前仍是文档级 + 句子启发式 span | `evoco_rag/contract.py`、`evoco_rag/schemas.py`、`evoco_rag/text_utils.py`、`evoco_rag/verifier.py` | citation correctness、span support、multi-hop coverage 可输出 |
| ECR-6 | 大模型忠实生成与审计格式协同优化 | 在 SFT 基础上接入偏好/奖励优化，降低 unsupported answer | **部分完成**：批量 SFT、批量 audit generation 和 GRPO reward 函数接口已落地，尚未完整偏好优化/GRPO rollout | `evoco_rag/trainers/large_trainer.py`、`evoco_rag/large_model.py`、`evoco_rag/auditor.py` | JSON parse rate 提升；unsupported answer rate 下降 |
| ECR-7 | 多轮协同进化稳定性 | 增加 round 级曲线、漂移监控、replay 质量统计和 early stop | **待做**：当前支持多轮 checkpoint，但缺少 drift/early-stop/plot 脚本 | `evoco_rag/trainers/coevolution_trainer.py`、`evoco_rag/replay_buffer.py`、`evoco_rag/evaluation/evaluator.py` | round-by-round accuracy/support/cost 曲线完整，退化时可停止 |
| ECR-8 | 跨数据集和强基线验证 | 扩展数据加载、baseline runner、消融矩阵和结果汇总 | **部分完成，待跑数成表**：dataset pack 与配置生成已覆盖 PopQAStandard、HotpotQA、NQ、ASQA、PopQA retrieval；`multidataset_full_2gpu.yaml` 已提供官方 full 入口 | `evoco_rag/data.py`、`configs/`、`scripts/run_ablations.py`、`evoco_rag/evaluation/metrics.py` | 还需强基线和统一结果汇总表 |

### 18.2 模块级 TODO

| 编号 | 状态 | 方法与模块 TODO | 需要新增或修改的配置 | 需要新增的测试或脚本 |
|---|---|---|---|---|
| ECR-1 | **代码完成，待实验成表** | 已新增 `SmallPolicyHeads`、`evidence_head/action_head/confidence_head`、`SmallTrainer` 多任务 loss、训练指标、head checkpoint 元数据和 policy 专用配置 | 已新增 `small_policy.use_policy_heads`、`evidence_loss_weight`、`action_loss_weight`、`calibration_loss_weight`；官方消融使用 `popqa_ablation_full_2gpu.yaml` 的 `no_policy_heads` 对照 | 已新增/扩展 `tests/test_small_policy_heads.py`、`tests/test_weights.py`；后续补真实训练回归测试 |
| ECR-2 | **已完成代码落地** | 已新增 `attribution_case`、`small_credit_weight`、`large_credit_weight`、误奖励统计；`ReplayBuffer`、`metrics`、`inspect_replay.py` 已支持输出 | 不再需要新增配置；后续根据实验需要调整 reward 权重 | 已扩展 `tests/test_rewards.py`、`tests/test_replay_buffer.py`、`tests/test_metrics.py` |
| ECR-3 | **核心统计已完成** | 已保存 audit candidate 摘要、候选得分、一致性；已拆分 `trust_components`；剩余是人工抽样导出和人工一致率评估 | 已复用 `runtime.num_audit_candidates`；后续可增加 `runtime.audit_trust_threshold`、`runtime.trust_filter_mode` | 已扩展 `tests/test_auditor.py`、`tests/test_verifier.py`；后续新增 `scripts/export_audit_review.py` |
| ECR-4 | **代码完成，待实验成表** | 已保留启发式 action 作为 teacher，并支持 `heuristic`、`policy`、`hybrid` 三种 action 模式；policy head 输出 action confidence，hybrid 模式可按置信度覆盖启发式；metrics 已输出 action cost 和 accuracy-cost Pareto 点 | 已新增 `contract.action_mode`、`contract.policy_action_min_conf`、`reward.audit_call_cost`、`reward.rewrite_cost`、`reward.retrieve_more_cost`；官方 hparam/full ablation specs 已覆盖低成本、高召回和 no-action-policy 对照 | 已新增 `tests/test_action_policy.py`，并扩展 `tests/test_rewards.py`、`tests/test_metrics.py`；后续补论文表格汇总 |
| ECR-5 | **待做** | `EvidenceItem` 增加 `sentence_id`、`span_start`、`span_end`、`hop_id`；`text_utils.py` 实现句子切分和 offset；`verifier.py` 校验 quote 与 span | `contract.evidence_granularity`、`contract.max_evidence_hops` | `tests/test_contract_spans.py`、`tests/test_verifier_spans.py` |
| ECR-6 | **部分完成** | 已完成批量 audit generation、批量 SFT、异常恢复和 eval-time blind generation；剩余是构造 unsupported correction / preference pairs，并可选接入 DPO/ORPO/GRPO | 已新增 `runtime.audit_batch_size`、`training.large_batch_size`；后续可加 `training.large_objective`、`training.preference_loss_weight` | 已新增 `tests/test_large_batching.py`；后续扩展真实 GRPO/DPO dry-run 测试 |
| ECR-7 | **待做** | `coevolution_trainer.py` 保存每轮质量摘要；`ReplayBuffer` 统计新样本、hard negative、low trust、anchor 覆盖；支持 early stop 和 rollback | `training.early_stop_metric`、`training.early_stop_patience`、`training.rollback_on_degradation` | 新增 `scripts/plot_round_metrics.py`；`tests/test_coevolution_stability.py` |
| ECR-8 | **部分完成，待跑数成表** | 数据集统一转换与 loader schema 已完成，`make_dataset_config.py` 可生成 PopQAStandard、HotpotQA、NQ、ASQA、PopQA retrieval 的 fast/full configs；强基线与结果汇总仍待补 | 已新增 dataset registry、`configs/local/*_{fast,full}.yaml` 生成入口、`configs/experiments/multidataset_full_2gpu.yaml` | 后续新增 `scripts/collect_results.py`；补强基线测试 |

### 18.3 推荐实现顺序

优先级按当前完成度和投稿价值排序：

1. **已完成代码落地：ECR-1/ECR-2/ECR-3/ECR-4**。下一步需要在真实模型输出上跑消融，形成 policy head、责任归因、审计可靠性和 accuracy-cost Pareto 表。
2. **下一步优先：ECR-7**。用 round-by-round 曲线证明“协同进化”不是单轮自训练，并增加 early stop / drift 监控降低多轮退化风险。
3. **随后做 ECR-8**：fast 多数据集入口已补齐，下一步补 full runs、强基线和统一结果表，让结论不局限于 PopQA。
4. **最后补强 ECR-5/ECR-6**：扩展证据粒度，并把当前批量 SFT 入口升级为可验证的偏好/奖励优化，支撑更完整的 A 会实验。

最小 CCF-A 冲刺版应至少完成：

```text
[代码完成] ECR-1: 可训练 small policy heads，仍需真实 GPU 实验成表
[代码完成] ECR-2: 责任归因指标和 answer-only 对照，仍需真实消融表
[代码完成] ECR-3: 审计可靠性与 trust-weight 消融，仍需人工抽样验证
[代码完成] ECR-4: 成本感知动态 action，仍需 accuracy-cost Pareto 实验表
[待做] ECR-7: round-by-round 协同进化曲线
[部分完成] ECR-8: 已有多数据集 fast 入口，仍需 full runs + 强基线 + 统一结果表
```

### 18.4 当前实现进度

截至本轮代码修正：

| 编号 | 当前代码状态 | 还未完成 |
|---|---|---|
| ECR-1 | 已增加 `small_policy` 配置段、`SmallPolicyHeads`、policy head 保存/加载元数据、`SmallTrainer` 多任务 loss、训练指标、manifest 记录和 policy 专用配置；生成的 dataset configs 默认启用 policy heads | 还需要在真实 H20 训练中用 `popqa_ablation_full_2gpu.yaml` 对比 `evoco_full` 与 `no_policy_heads`，验证 action/evidence/calibration 指标是否稳定提升 |
| ECR-2 | 已增加 `attribution_case`、`small_credit_weight`、`large_credit_weight`、`wrong_retriever_reward_if_answer_only`；`metrics`、`ReplayBuffer`、`inspect_replay.py` 均可输出责任归因统计 | 还需要在真实模型审计结果上跑 ablation，形成论文表格 |
| ECR-3 | 已增加 `LargeAudit.audit_metadata`、多候选审计摘要、`self_consistency`、`RuleVerification.trust_components`、trust summary 指标 | 还需要增加人工抽样导出脚本，并用真实审计样本估计人工一致率 |
| ECR-4 | 已增加 `contract.action_mode`、policy action confidence、hybrid action 选择逻辑、action cost reward penalty、metrics 中的 `avg_action_cost_penalty` / `accuracy_cost_pareto_point`，policy 配置默认启用 hybrid 模式 | 还需要跑 hparam/action 消融，形成 accuracy-cost Pareto 曲线 |
| ECR-5 至 ECR-8 | 文档已有 TODO，其中 ECR-6 已完成批量 SFT/audit 的工程加速，ECR-8 已有多数据集 fast 入口，ECR-7 是下一阶段最关键路径 | ECR-6 尚缺偏好优化/GRPO rollout；ECR-7 尚缺 drift/early-stop；ECR-8 尚缺 full 多数据集和强基线结果 |
