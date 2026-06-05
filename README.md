# EvoCo-RAG

证据合约驱动的大小模型协同进化 RAG 原型。

本仓库保留原始 CoRAG 基线脚本，并在 `evoco_rag/` 中新增结构化的 EvoCo-RAG 实现。后续迭代建议以 `scripts/train_evoco.py`、`scripts/eval_evoco.py` 和 `evoco_rag/` 为主，`run_train.py`、`run_test.py` 仅作为 legacy baseline 参考。

## 目录边界

当前工程已按“代码”和“资产”分离：

```text
/Users/quanquan/Desktop/rag_code/
├── CoRAG-D63F /          # 代码仓库，后续上传 GitHub
└── rag_assets/           # 本地资产，不上传 GitHub
```

注意：`CoRAG-D63F ` 目录名末尾有一个空格，进入目录时需要加引号：

```bash
cd "/Users/quanquan/Desktop/rag_code/CoRAG-D63F "
```

## 代码仓库结构

```text
CoRAG-D63F /
├── configs/
│   ├── debug.yaml
│   └── evoco_popqa.yaml
├── docs/
│   ├── 协同进化RAG代码开发文档.md
│   └── 协同进化RAG论文构想.md
├── evoco_rag/
│   ├── config.py
│   ├── schemas.py
│   ├── data.py
│   ├── contract.py
│   ├── auditor.py
│   ├── verifier.py
│   ├── rewards.py
│   ├── replay_buffer.py
│   ├── weights.py
│   ├── small_model.py
│   ├── large_model.py
│   ├── trainers/
│   └── evaluation/
├── scripts/
│   ├── build_seed_replay.py
│   ├── train_evoco.py
│   ├── eval_evoco.py
│   ├── run_ablations.py
│   └── inspect_replay.py
├── tests/
├── run_train.py          # legacy baseline
├── run_test.py           # legacy baseline
├── pytest.ini
├── .gitignore
└── README.md
```

代码仓库内不再保留 `data/`、`data_v33/`、`model/`、`outputs/`、`outputs_debug/` 或压缩包。

## 本地资产结构

本地资产目录位于：

```text
/Users/quanquan/Desktop/rag_code/rag_assets/
```

当前约定：

```text
rag_assets/
├── data/Pop/test.json
├── data_v33/Pop/train_labels_list.json
├── adapters/
│   ├── generator-CoRAG/
│   └── reranker-CoRAG/
├── base_models/
│   ├── generator/Meta-Llama-3.1-8B-Instruct/
│   └── reranker/bge-reranker-v2-m3/
├── checkpoints/
│   ├── debug/
│   └── evoco_popqa/
├── legacy/
│   ├── adapters/
│   ├── data/
│   └── outputs/
├── outputs/
├── outputs_debug/
└── archive/
```

`rag_assets/base_models/...` 是预留的 base model 位置。当前代码不会在 CPU 冒烟测试中加载这些大权重，但 GPU 训练/评测前需要把实际模型权重放到对应目录，或在 YAML 中改成真实路径。

## 方法概览

EvoCo-RAG 将原始闭环：

```text
reranker selects top1_doc
generator answers
answer hit => reward top1_doc
```

改为责任可分解的闭环：

```text
small model proposes an EvidenceContract
large model answers and audits evidence
rule verifier checks answer/evidence/citation
decomposed reward assigns responsibility
replay buffer stores structured experience
small LoRA and large LoRA are updated separately
```

核心点是避免“答案正确就奖励检索器”的错误归因。只有证据被审计为支持答案时，小模型才获得正反馈；大模型则围绕忠实生成、引用和审计格式获得反馈。

## 环境

CPU 测试和数据流检查需要：

```text
python 3.10+
pytest
numpy
pyyaml
```

GPU 训练/评测还需要：

```text
torch
transformers
peft
trl
datasets
sentence-transformers
bitsandbytes, if use_4bit=true
```

目前没有固定 `requirements.txt`，建议在目标 GPU/CUDA 环境中安装匹配版本。

## 配置

主要配置文件：

```text
configs/debug.yaml        # 16 条样本的 smoke config
configs/evoco_popqa.yaml  # PopQA 全量配置
```

`configs/debug.yaml` 的关键字段：

```yaml
project:
  output_dir: ../rag_assets/outputs_debug/latest

data:
  train_path: ../rag_assets/data_v33/Pop/train_labels_list.json
  test_path: ../rag_assets/data/Pop/test.json
  debug_size: 16

models:
  small_base_path: ../rag_assets/base_models/reranker/bge-reranker-v2-m3
  large_base_path: ../rag_assets/base_models/generator/Meta-Llama-3.1-8B-Instruct
  small_lora_dir: ../rag_assets/checkpoints/debug/small
  large_lora_dir: ../rag_assets/checkpoints/debug/large
  use_4bit: false
```

`configs/evoco_popqa.yaml` 使用同样的资产目录，但输出到：

```text
../rag_assets/outputs/evoco_popqa
../rag_assets/checkpoints/evoco_popqa/small
../rag_assets/checkpoints/evoco_popqa/large
```

## 权重规则

`evoco_rag/weights.py` 会写出每次运行的 `weights_manifest.json`，并记录：

```text
small_base_path
large_base_path
small_checkpoint_root
large_checkpoint_root
small_latest_adapter
large_latest_adapter
legacy_small_adapter
legacy_large_adapter
```

LoRA checkpoint root 不是 adapter 本身。可加载 adapter 必须是类似下面的 round 目录：

```text
../rag_assets/checkpoints/evoco_popqa/small/round_000/
../rag_assets/checkpoints/evoco_popqa/large/round_000/
```

且包含 `adapter_config.json` 和 `adapter_model.safetensors` 或 `adapter_model.bin`。

检查当前配置解析出的权重布局：

```bash
python - <<'PY'
from evoco_rag.config import EvoCoConfig
from evoco_rag.weights import prepare_weight_layout
cfg = EvoCoConfig.load("configs/debug.yaml")
print(prepare_weight_layout(cfg, create=True))
PY
```

## CPU 检查

这些命令不会加载大模型，可在本机快速确认代码和数据路径。

运行单元测试：

```bash
python -m pytest -q
```

运行语法检查：

```bash
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py
```

从真实 debug 数据构造 seed replay：

```bash
python scripts/build_seed_replay.py --config configs/debug.yaml
python scripts/inspect_replay.py --replay ../rag_assets/outputs_debug/latest/replay/round_000.jsonl
```

运行 no-model 消融布线检查：

```bash
python scripts/run_ablations.py --config configs/debug.yaml --no_models
```

预期输出位置：

```text
../rag_assets/outputs_debug/latest/replay/round_000.jsonl
../rag_assets/outputs_debug/latest/contracts/round_000.jsonl
../rag_assets/outputs_debug/latest/audits/round_000.jsonl
../rag_assets/outputs_debug/latest/weights_manifest.json
../rag_assets/outputs_debug/latest/ablations/summary.json
```

## GPU 训练

先在目标 GPU 机器上跑 debug 配置：

```bash
python scripts/train_evoco.py --config configs/debug.yaml
```

再跑全量 PopQA：

```bash
python scripts/train_evoco.py --config configs/evoco_popqa.yaml
```

从最新完整 `round_XXX` 继续训练：

```bash
python scripts/train_evoco.py --config configs/evoco_popqa.yaml --resume
```

如果需要新开一轮实验，建议改成新的输出和 checkpoint 目录，例如：

```yaml
project:
  output_dir: ../rag_assets/outputs/evoco_popqa_v2

models:
  small_lora_dir: ../rag_assets/checkpoints/evoco_popqa_v2/small
  large_lora_dir: ../rag_assets/checkpoints/evoco_popqa_v2/large
```

## 评测

使用配置中的最新 checkpoint root：

```bash
python scripts/eval_evoco.py --config configs/evoco_popqa.yaml
```

显式指定 adapter round：

```bash
python scripts/eval_evoco.py \
  --config configs/evoco_popqa.yaml \
  --small_lora ../rag_assets/checkpoints/evoco_popqa/small/round_002 \
  --large_lora ../rag_assets/checkpoints/evoco_popqa/large/round_002
```

评测阶段不会把 gold answers 放进大模型 prompt，gold answers 只用于离线指标。

## 消融实验

运行全部消融：

```bash
python scripts/run_ablations.py --config configs/evoco_popqa.yaml
```

每个消融实验写到独立输出目录：

```text
../rag_assets/outputs/evoco_popqa/ablations/baseline_current_corag/
../rag_assets/outputs/evoco_popqa/ablations/evoco_no_audit/
../rag_assets/outputs/evoco_popqa/ablations/evoco_no_action/
../rag_assets/outputs/evoco_popqa/ablations/evoco_answer_only_reward/
../rag_assets/outputs/evoco_popqa/ablations/evoco_small_only/
../rag_assets/outputs/evoco_popqa/ablations/evoco_large_only/
../rag_assets/outputs/evoco_popqa/ablations/evoco_full/
```

## GitHub 上传建议

只上传：

```text
/Users/quanquan/Desktop/rag_code/CoRAG-D63F 
```

不要上传：

```text
/Users/quanquan/Desktop/rag_code/rag_assets
```

`.gitignore` 已忽略常见 Python 缓存、输出目录、数据目录、模型目录和压缩包。由于 Git 不能忽略仓库外部的兄弟目录，关键是初始化 Git 仓库时把仓库根目录放在 `CoRAG-D63F `，不要放在 `/Users/quanquan/Desktop/rag_code`。

## 当前本地验证

本机 CPU 环境应通过：

```text
python -m pytest -q
python -m py_compile ...
python scripts/build_seed_replay.py --config configs/debug.yaml
python scripts/run_ablations.py --config configs/debug.yaml --no_models
```

仍需要 GPU 验证：

```text
loading bge-reranker-v2-m3
loading Meta-Llama-3.1-8B-Instruct
bf16 / optional 4bit generation
real JSON audit success rate
LoRA checkpoint saving and resume under real training
multi-round convergence
```

## Legacy 脚本

`run_train.py` 和 `run_test.py` 保留用于基线对照，不是 EvoCo-RAG 推荐入口。

已知差异：

- 它们现在也指向 `../rag_assets`，但训练逻辑仍是旧式 answer-only baseline。
- 它们的中间数据、GRPO 输出和 adapter 会写入 `../rag_assets/legacy/`。
- 它们不使用 `evoco_rag/weights.py`。
- 它们不写结构化 evidence contract、audit 或 replay experience。

新系统请使用 `scripts/train_evoco.py` 和 `scripts/eval_evoco.py`。
