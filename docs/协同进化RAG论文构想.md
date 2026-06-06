# EvoCo-RAG：证据合约驱动的大小模型协同进化检索增强生成

## 1. 研究动机

检索增强生成（Retrieval-Augmented Generation, RAG）通过外部知识缓解大语言模型的知识过时和幻觉问题，但现有 RAG 系统通常仍存在三个核心缺陷：

1. **检索与生成责任混淆**：最终答案正确并不必然说明检索到的文档正确；大模型可能凭参数知识答对，导致错误证据被误奖励。
2. **固定检索策略缺乏自适应能力**：多数系统采用固定 top-k 文档输入大模型，无法根据问题难度、检索置信度和证据充分性动态调整检索行为。
3. **大小模型协作停留在串联层面**：小模型负责召回或重排序，大模型负责生成答案，但两者之间缺少可解释、可训练、可迭代优化的反馈机制。

本文拟提出 **EvoCo-RAG（Evidence-Contract Driven Co-Evolution RAG）**：一种证据合约驱动的大小模型协同进化 RAG 框架。其核心思想是：**小模型不只是 reranker，而是轻量级 RAG policy model；大模型不只是 generator，而是 evidence auditor 与 teacher。二者通过结构化证据合约、失败归因和参数高效更新形成闭环自进化。**

## 2. 核心问题定义

给定问题 \(q\)、候选文档集合 \(D=\{d_1,\dots,d_n\}\)、标准答案集合 \(A\)，传统 RAG 学习目标通常近似为：

\[
\max P(y \mid q, TopK(D))
\]

其中 \(TopK(D)\) 由检索器或重排序器给出，生成模型直接基于 top-k 文档生成答案 \(y\)。

但该目标无法区分以下四种情况：

| 情况 | 答案正确 | 证据支持 | 责任归因 |
|---|---:|---:|---|
| A | 是 | 是 | 检索与生成均成功 |
| B | 是 | 否 | 大模型可能凭参数知识答对，不能奖励检索器 |
| C | 否 | 是 | 检索正确，生成模型失败 |
| D | 否 | 否 | 检索与生成均需改进 |

本文的核心问题是：**如何构建一种可归因、可训练、可自进化的大小模型协同机制，使系统能够区分检索错误、证据不足、实体混淆和生成错误，并将不同失败类型转化为对应模型的训练信号。**

## 3. 方法总览

EvoCo-RAG 包含两个主要模型：

- **小模型 \(M_s\)**：轻量级 reranker / policy model，例如 `bge-reranker-v2-m3`。负责证据选择、置信度估计、检索动作决策。
- **大模型 \(M_l\)**：生成式大语言模型，例如 `mistralai/Mistral-Nemo-Instruct-2407`。负责基于证据生成答案，并审计证据是否真正支持答案。

整体流程如下：

```text
候选文档 D
  ↓
小模型 M_s 生成证据合约 C
  ↓
大模型 M_l 审计合约并生成答案
  ↓
结构化反馈 F：答案正确性、证据支持性、失败类型、动作建议
  ↓
更新经验池 B
  ↓
分别训练小模型 LoRA 与大模型 LoRA
  ↓
进入下一轮自进化
```

## 4. 证据合约机制

传统 reranker 只输出文档相关性分数：

```json
{
  "doc_id": 3,
  "score": 0.87
}
```

EvoCo-RAG 将小模型的输出封装为更丰富的 **证据合约（Evidence Contract）**：

```json
{
  "answerability": "high",
  "retrieval_action": "answer_now",
  "selected_evidence": [
    {
      "doc_id": 3,
      "span": "Henry Master Feilden was an English Conservative Party politician.",
      "relevance_score": 0.91,
      "evidence_score": 0.88,
      "reason": "The span contains the entity Henry Master Feilden and states his occupation."
    }
  ],
  "uncertainty": {
    "entity_ambiguity": false,
    "evidence_conflict": false,
    "missing_relation": false
  }
}
```

该合约使小模型从单纯排序器升级为轻量级 RAG 控制器。它不仅决定“哪个文档靠前”，还决定：

- 当前证据是否足以回答；
- 是否需要更多检索；
- 是否需要 query rewrite；
- 哪些 span 是真正证据；
- 置信度是否校准。

**实现说明（与代码现实对齐）**：`bge-reranker-v2-m3` 本质是交叉编码器，原生只输出 \((q, d)\) 的相关性分数，并不能直接生成上述 JSON。因此证据合约采用**分阶段实现**：

- **第一阶段（首版可落地）**：小模型只负责打分与 top-k 排序；`span`、`reason`、`answerability`、`retrieval_action`、`uncertainty` 等语义字段由一个确定性的“合约构造器”通过启发式规则封装（如句子级证据抽取、基于 top1–top2 分数 margin 与阈值的动作判定），`relevance_confidence` 由 logit 经 sigmoid / 温度标定得到。此阶段不引入新的可训练参数，证据合约是“打分 + 规则封装”的产物。
- **第二阶段（policy learning）**：在小模型上引入轻量级 evidence head 与 action head，把 `evidence_confidence` 与 `retrieval_action` 从启发式转为**可训练**输出，由大模型审计反馈监督（见 §6.1）。

因此，本文所称“小模型生成证据合约”指的是这套“打分骨架 + 渐进可训练 head”的整体机制，而非要求交叉编码器直接输出结构化文本。

## 5. 大模型审计机制

大模型收到问题 \(q\)、证据合约 \(C\) 和必要文档片段后，输出答案与审计反馈：

```json
{
  "final_answer": "politician",
  "used_doc_ids": [3],
  "answer_correctness": "correct",
  "support_level": "fully_supported",
  "failure_type": "none",
  "small_model_feedback": [
    {
      "doc_id": 3,
      "label": "positive",
      "reason": "The document explicitly states the occupation."
    }
  ],
  "suggested_action": "answer_now"
}
```

当出现失败时，大模型需要输出结构化失败类型：

| failure_type | 含义 | 主要训练对象 |
|---|---|---|
| `retrieval_miss` | 候选文档中缺少必要证据 | 召回模块或 query rewrite |
| `rerank_error` | 证据存在但小模型未选中 | 小模型 reranker |
| `entity_confusion` | 实体同名或语义混淆 | 小模型证据判断 + 大模型审计 |
| `evidence_conflict` | 多文档证据冲突 | 小模型置信度 + 大模型推理 |
| `generation_error` | 证据正确但答案错误 | 大模型 generator |
| `unsupported_answer` | 答案正确但证据不支持 | 大模型忠实性 + 小模型不能被奖励 |
| `over_retrieval` | 引入过多无关证据 | 小模型 action policy |

该机制的关键是：**答案对错不再是唯一训练信号，证据是否支撑答案成为独立监督信号。**

## 6. 协同进化训练目标

### 6.1 小模型训练目标

小模型 \(M_s\) 的目标由三部分组成：

#### 6.1.1 文档相关性排序

使用 margin ranking loss：

\[
\mathcal{L}_{rank}=\sum_{d^+}\sum_{d^-}\max(0, m-s(q,d^+)+s(q,d^-))
\]

其中 \(d^+\) 是被大模型审计为真正支持答案的文档，\(d^-\) 是无关或误导文档。

#### 6.1.2 证据支持预测

对文档或 span 预测是否支持答案：

\[
\mathcal{L}_{evi}=CE(\hat{z}_{q,d}, z_{q,d})
\]

其中 \(z_{q,d}\) 来自大模型审计反馈，而不是简单来自答案命中。

#### 6.1.3 检索动作策略学习

小模型预测动作：

\[
a \in \{\text{answer\_now}, \text{retrieve\_more}, \text{rewrite\_query}, \text{ask\_auditor}\}
\]

训练损失为：

\[
\mathcal{L}_{act}=CE(\hat{a}, a^*)
\]

其中 \(a^*\) 由大模型审计结果和最终奖励共同构造。

小模型总损失：

\[
\mathcal{L}_{small}=\lambda_1\mathcal{L}_{rank}+\lambda_2\mathcal{L}_{evi}+\lambda_3\mathcal{L}_{act}+\lambda_4\mathcal{L}_{calib}
\]

\(\mathcal{L}_{calib}\) 用于约束置信度校准，例如 high confidence 的样本应具有更高证据支持率。（与实现对齐：第一阶段置信度来自 reranker 分数的 sigmoid / 温度标定，\(\mathcal{L}_{calib}\) 在第二阶段引入置信度 head 后启用；ECE 等校准指标在两个阶段均可离线评估。）

### 6.2 大模型训练目标

大模型 \(M_l\) 的训练目标包括：

1. 基于证据生成正确答案；
2. 输出使用过的 doc_id / span_id；
3. 判断证据支持等级；
4. 给出小模型可学习的结构化反馈。

可采用 LoRA + GRPO 或 SFT + preference optimization 的组合方式：

\[
\mathcal{L}_{large}=\alpha \mathcal{L}_{answer}+\beta \mathcal{L}_{citation}+\gamma \mathcal{L}_{audit}
\]

其中：

- \(\mathcal{L}_{answer}\)：答案生成损失或 reward；
- \(\mathcal{L}_{citation}\)：引用证据是否与答案一致；
- \(\mathcal{L}_{audit}\)：审计标签与规则验证 / 人工标注 / 高置信自动标签的一致性。

## 7. 奖励函数设计

当前代码中的 reward 近似为：

\[
R = \mathbb{1}[\text{answer contains gold answer}]
\]

这会导致错误归因。EvoCo-RAG 将 reward 拆解为：

\[
R = R_{ans} + R_{support} + R_{cite} + R_{calib} - R_{cost}
\]

各部分定义如下：

| 奖励项 | 含义 |
|---|---|
| \(R_{ans}\) | 最终答案是否匹配 gold answer |
| \(R_{support}\) | 答案是否由选中证据支持 |
| \(R_{cite}\) | 模型引用的 doc/span 是否真实包含答案依据 |
| \(R_{calib}\) | 小模型置信度与实际成功率是否一致 |
| \(R_{cost}\) | 检索轮数、文档数量、大模型审计调用成本 |

关键原则：

- **答案正确但证据不支持**：奖励大模型答案能力，但不奖励小模型检索。
- **证据正确但答案错误**：奖励小模型检索，惩罚或训练大模型生成。
- **答案正确且证据支持**：同时奖励大小模型。
- **检索成本过高但收益有限**：惩罚小模型 action policy。

## 8. 自进化算法

```text
Algorithm: EvoCo-RAG Training

Input:
  Training questions Q
  Candidate documents D
  Small model M_s with LoRA adapter A_s
  Large model M_l with LoRA adapter A_l
  Replay buffer B

For round t = 1 ... T:
  1. For each question q:
       M_s ranks documents and produces evidence contract C_t

  2. M_l receives q and C_t:
       generates answer y_t
       audits evidence support
       outputs structured feedback F_t

  3. Rule verifier checks:
       answer match
       whether cited evidence contains answer or support span
       whether confidence agrees with outcome

  4. Store experience:
       B ← B ∪ {(q, D, C_t, y_t, F_t, rewards)}

  5. Train small model LoRA A_s:
       use audited positive / negative docs
       learn evidence score and action policy

  6. Train large model LoRA A_l:
       use high-quality supported-answer traces
       learn faithful answer generation and auditing format

  7. Filter replay buffer:
       keep high-confidence positives
       keep hard negatives
       down-weight noisy auditor feedback

Output:
  Co-evolved small model M_s + A_s
  Co-evolved large model M_l + A_l
```

## 9. 与现有工作的区别

### 9.1 与普通 RAG 的区别

普通 RAG 固定执行：

```text
retrieve → rerank → generate
```

EvoCo-RAG 执行：

```text
propose evidence contract → audit → assign responsibility → update both models
```

核心差异是：普通 RAG 关注答案，EvoCo-RAG 同时关注答案、证据、动作和失败归因。

### 9.2 与 Self-RAG 的区别

Self-RAG 通过特殊 reflection token 学习何时检索、如何批判生成内容。EvoCo-RAG 的不同点在于：

- 反思职责不完全压在大模型上；
- 小模型承担低成本 policy 与证据提案；
- 大模型审计结果被显式转化为小模型训练信号；
- 目标是大小模型共同进化，而非单一 LM 自反思。

### 9.3 与 CRAG 的区别

CRAG 使用 evaluator 判断检索质量并触发修正。EvoCo-RAG 的不同点在于：

- evaluator 不只是打质量分，而是输出证据级反馈和失败类型；
- 小模型 action policy 会被持续训练；
- 大模型同时承担生成与审计，并反哺小模型。

### 9.4 与 Adaptive-RAG 的区别

Adaptive-RAG 根据问题复杂度选择不同检索策略。EvoCo-RAG 的不同点在于：

- 策略依据不仅是问题复杂度，还包括证据充分性、置信度、引用一致性和历史反馈；
- 策略模型由大模型审计反馈持续训练；
- 训练目标包含成本约束和证据忠实性。

### 9.5 与 RankRAG 的区别

RankRAG 将 ranking 与 generation 统一到一个 LLM 中。EvoCo-RAG 的不同点在于：

- 保留小模型承担高频排序和策略决策，降低推理成本；
- 大模型主要用于复杂生成和审计；
- 通过证据合约实现模型间可解释协作，而不是把排序能力完全内化到大模型。

## 10. 预期贡献

本文可主张以下贡献：

1. **提出证据合约驱动的大小模型协作范式**：小模型生成可审计证据合约，大模型负责审计与反馈，使 RAG 协作从简单串联升级为结构化闭环。
2. **提出责任归因式协同进化训练机制**：将答案正确性、证据支持性和失败类型拆分为不同训练信号，避免把大模型参数知识导致的正确答案错误归因给检索器。
3. **提出面向 RAG 的小模型 policy learning 目标**：小模型不仅学习文档排序，还学习证据选择、置信度估计和检索动作决策。
4. **提出参数高效的双 LoRA 自进化框架**：冻结大小模型 base，通过 LoRA adapter 在多轮反馈中共同更新，降低训练成本。
5. **构建细粒度评估体系**：同时评估 answer accuracy、evidence support、citation correctness、retrieval quality、action efficiency 和 cost-quality tradeoff。

## 11. 实验设计

### 11.1 数据集

优先选择需要证据支撑且有候选文档的数据集：

- PopQA：实体属性问答，适合验证实体混淆和检索排序。
- Natural Questions：开放域问答，适合验证真实检索质量。
- HotpotQA：多跳问答，适合验证多证据合约和 retrieve_more 动作。
- ASQA：长答案问答，适合验证答案忠实性和证据覆盖。

### 11.2 Baselines

建议比较：

1. Vanilla RAG：embedding retrieval + LLM generation。
2. RAG + reranker：固定 top-k rerank 后生成。
3. Self-RAG 类方法：大模型自反思式检索增强。
4. CRAG 类方法：检索质量评估与修正。
5. Adaptive-RAG 类方法：根据问题复杂度选择检索策略。
6. 当前 CoRAG 代码版本：答案命中奖励 + top1 标签更新。
7. EvoCo-RAG：完整证据合约 + 审计反馈 + 双 LoRA 自进化。

### 11.3 指标

答案指标：

- Exact Match / Accuracy
- F1
- Long-form answer correctness

检索指标：

- Recall@k
- MRR
- nDCG
- Oracle answer-in-context rate

证据指标：

- Evidence support rate
- Citation correctness
- Used-doc precision
- Unsupported answer rate

策略指标：

- Average retrieved docs
- Large-model audit call rate
- Cost per correct answer
- Accuracy-cost Pareto frontier

校准指标：

- Expected Calibration Error (ECE)
- Confidence-success correlation

### 11.4 消融实验

必须做的消融：

| 实验 | 目的 |
|---|---|
| 去掉 evidence audit | 验证证据审计是否减少错误奖励 |
| 去掉 action policy | 验证动态检索策略是否有效 |
| 只训练小模型 LoRA | 验证小模型自进化贡献 |
| 只训练大模型 LoRA | 验证大模型自进化贡献 |
| 不使用 failure_type | 验证失败归因是否提升训练稳定性 |
| 固定 top-k vs 动态 action | 验证成本-效果权衡 |
| answer-only reward vs decomposed reward | 验证奖励拆解是否解决错误归因 |

## 12. 与当前代码的对应改造

当前代码已有雏形：

```text
reranker 选 top1_doc
generator 根据 top1_doc 生成答案
答案命中则给 top1_doc 追加正标签
```

建议改造为：

1. `reranker_training` 不只保存 `top1_doc`，而是保存 top-k 文档、分数、候选 evidence span 和 confidence。
2. generator prompt 要求输出结构化 JSON，包括 `final_answer`、`used_doc_ids`、`support_level`、`failure_type`。
3. `myReward` 从 answer-only reward 改为 decomposed reward。
4. 标签更新从 `labels[top1_index].append("1")` 改成 evidence-level audited feedback。
5. 新增 replay buffer，保存每轮合约、答案、审计和奖励。
6. reranker 增加 action head，预测 `answer_now / retrieve_more / rewrite_query / ask_auditor`。

## 13. 面向 CCF-A 的创新点 TODO 与状态

当前方案已经具备“证据合约 + 大模型审计 + 分解奖励 + 双 LoRA 闭环”的原型，但若目标是 CCF-A 级别投稿，还需要把创新从工程流程进一步强化为可验证的学习机制。下面的 TODO 使用统一编号 `ECR-*`，与代码开发文档中的同名 TODO 严格对应。

| 编号 | 创新点 | 论文中要证明的问题 | 当前状态 | 后续补强证据 | 对应代码模块 |
|---|---|---|---|---|---|
| ECR-1 | 可训练的小模型证据-动作-置信度策略 | 小模型是否真的从 reranker 升级为低成本 RAG policy model | **代码完成，待实验成表**：已加入 `SmallPolicyHeads`、policy 专用配置、head 保存/加载元数据、多任务 loss 和训练指标输出，默认主配置仍关闭 | 在 H20 上运行 `configs/debug_policy.yaml` / `configs/evoco_popqa_policy.yaml`，报告 evidence/action/calibration loss、action accuracy、ECE、Recall/MRR 消融 | `small_model.py`、`small_trainer.py`、`schemas.py`、`config.py` |
| ECR-2 | 责任归因式 credit assignment | 答案正确时如何避免错误奖励检索器，答案错误时如何识别生成失败 | **代码完成，待实验成表**：已加入 `attribution_case`、credit weight、answer-only 误奖励统计和 replay/metrics 输出 | 用真实审计结果跑 answer-only vs decomposed reward 消融，报告错误奖励率下降和四象限分布 | `rewards.py`、`verifier.py`、`replay_buffer.py`、`evaluation/metrics.py` |
| ECR-3 | 可靠审计与抗噪声自训练 | 大模型审计是否足够可靠，错误审计是否会污染小模型 | **代码完成，待人工验证**：已加入多候选审计摘要、`self_consistency`、`trust_components`、trust summary 和批量 audit generation | 抽样人工复查审计一致率，报告 trust-weight 过滤消融和低信任样本比例 | `large_model.py`、`auditor.py`、`verifier.py`、`replay_buffer.py` |
| ECR-4 | 成本感知动态检索动作 | 系统是否能在 accuracy 与调用成本之间自适应权衡 | **代码完成，待实验成表**：已加入 `action_mode=heuristic/policy/hybrid`、policy action 置信覆盖、action cost penalty 和 accuracy-cost Pareto 指标 | 用 `configs/evoco_popqa_policy.yaml` 对比固定 top-k / heuristic / hybrid policy，报告 accuracy-cost Pareto frontier | `contract.py`、`small_model.py`、`small_trainer.py`、`evaluation/metrics.py` |
| ECR-5 | 多粒度证据合约 | 证据合约是否能从文档级扩展到句子级、span 级和多跳证据组合 | 当前主要是文档级 + 句子启发式 span | sentence/span citation correctness、多跳证据覆盖率 | `contract.py`、`schemas.py`、`text_utils.py`、`verifier.py` |
| ECR-6 | 大模型忠实生成与审计格式协同优化 | 大模型是否不仅答对，还能稳定引用证据并输出可靠审计 | 当前已有批量 SFT 入口和结构化输出约束 | SFT/GRPO/DPO 对 JSON parse rate、unsupported answer rate 的贡献 | `large_trainer.py`、`large_model.py`、`auditor.py` |
| ECR-7 | 多轮协同进化稳定性 | 大小模型多轮互相学习是否带来持续收益，而不是单轮伪标签噪声 | 当前支持 round 级训练和 checkpoint | round-by-round 曲线、漂移检测、early stop、replay 质量变化 | `coevolution_trainer.py`、`replay_buffer.py`、`evaluation/evaluator.py` |
| ECR-8 | 跨数据集和强基线验证 | 方法是否只是 PopQA 适配，还是通用 RAG 机制 | 当前主要围绕 PopQA | PopQA + NQ + HotpotQA/2Wiki + ASQA，多强基线和完整消融 | `data.py`、`configs/`、`scripts/run_ablations.py`、`evaluation/metrics.py` |

### 13.1 最核心的三条投稿主线

如果篇幅和工程时间有限，应优先完成以下三条，因为它们最能支撑 CCF-A 级别的“方法贡献”：

1. **ECR-1 + ECR-2：责任归因驱动的小模型策略学习**
   重点证明小模型不是被动 reranker，而是在大模型审计反馈下学习证据选择、动作决策和置信度校准。

2. **ECR-3 + ECR-7：可靠审计约束下的自进化**
   重点证明自训练不会因为大模型审计噪声而崩坏，系统通过规则验证、trust weight、replay 过滤和多轮曲线保持稳定提升。

3. **ECR-4 + ECR-8：质量-成本权衡与跨数据集泛化**
   重点证明 EvoCo-RAG 不只是提高准确率，还能以较低大模型调用成本达到更好的 evidence support 和 citation correctness。

### 13.2 必须避免的弱表述

为了保持论文严谨性，当前阶段不宜直接声称：

- “小模型生成完整自然语言证据合约”：第一阶段小模型只输出排序分数，合约由打分骨架和规则构造器共同形成。
- “系统已经实现完全自主进化”：当前更准确的说法是“参数高效的多轮协同自训练框架”，是否稳定自进化需要由 ECR-7 的曲线证明。
- “大模型审计天然可靠”：必须强调规则验证、审计置信权重、多候选一致性和人工抽样验证。
- “方法只靠 PopQA 即可证明通用性”：PopQA 只能验证实体属性问答，A 会投稿至少需要加入开放域和多跳数据集。

## 14. 关键风险与解决方案

### 风险一：大模型审计反馈可能不可靠

解决方案：

- 对审计结果增加规则验证，例如答案是否真的出现在引用 span 附近；
- 使用多次采样或 self-consistency；
- 只将 high-confidence 审计样本写入正样本池；
- 对审计反馈设置置信权重，而不是二值硬标签。

### 风险二：小模型输出 span 能力不足

解决方案：

- 第一阶段先做文档级证据合约；
- 第二阶段加入句子级切分；
- 第三阶段再做 span-level evidence selection。

### 风险三：训练成本过高

解决方案：

- 大模型 base 冻结，只训练 LoRA；
- 小模型 base 冻结，只训练 LoRA 和轻量 action head；
- 大模型审计只用于低置信样本或训练阶段；
- 推理阶段主要依靠小模型 policy 降低调用成本。

### 风险四：自训练噪声累积

解决方案：

- replay buffer 中保留人工标注或原始 gold labels 作为 anchor；
- 对新生成标签使用时间衰减和置信度加权；
- 使用 hard negative mining，但限制低质量伪标签比例；
- 定期在验证集上 early stop。

## 15. 论文标题候选

1. **EvoCo-RAG: Evidence-Contract Driven Co-Evolution of Small and Large Models for Retrieval-Augmented Generation**
2. **Small Proposes, Large Audits: Evidence-Grounded Co-Evolution for Efficient RAG**
3. **Contract-RAG: Auditable Small-Large Model Collaboration for Faithful Retrieval-Augmented Generation**
4. **Toward Self-Evolving RAG: Responsibility-Aware Training of Retriever and Generator**

## 16. 摘要草稿

Retrieval-Augmented Generation (RAG) improves the factuality of large language models by conditioning generation on external documents. However, existing RAG systems often conflate answer correctness with retrieval quality: a model may produce the correct answer from parametric knowledge even when retrieved evidence is irrelevant, causing misleading feedback for retrievers. We propose EvoCo-RAG, an evidence-contract driven co-evolution framework for small-large model collaboration in RAG. In EvoCo-RAG, a small model acts as a lightweight RAG policy model that proposes an auditable evidence contract, including selected evidence, confidence estimates, and retrieval actions. A large model then generates the answer and audits whether the proposed evidence truly supports the answer, producing structured feedback with failure attribution. The feedback is decomposed into answer correctness, evidence support, citation correctness, confidence calibration, and retrieval cost, and is used to update both models through parameter-efficient LoRA adapters. This responsibility-aware training prevents incorrect reward assignment and enables the small model to learn adaptive retrieval policies while the large model learns faithful evidence-grounded generation. Experiments on open-domain and multi-hop QA benchmarks will evaluate answer accuracy, evidence support, citation correctness, and cost-quality trade-offs.

## 17. 相关工作链接

- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. https://arxiv.org/abs/2310.11511
- Corrective Retrieval Augmented Generation. https://arxiv.org/abs/2401.15884
- Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. https://arxiv.org/abs/2403.14403
- RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs. https://arxiv.org/abs/2407.02485
- LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- BAAI bge-reranker-v2-m3 model card. https://huggingface.co/BAAI/bge-reranker-v2-m3
