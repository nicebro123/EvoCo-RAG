# Reproducibility and Experiment Guide

This guide is the operational contract for reproducing EvoCo-RAG and running
future ablations, hyperparameter studies, and two-GPU experiments.

## Repository Contract

The GitHub repository is code-only.

Keep these in Git:

```text
configs/
docs/
evoco_rag/
scripts/
tests/
requirements-*.txt
README.md
```

Keep these outside Git under a sibling asset root:

```text
../rag_assets/
├── data/
├── data_v33/
├── base_models/
├── checkpoints/
├── outputs/
└── outputs_debug/
```

The `.gitignore` blocks common local artifacts, model files, adapters,
checkpoints, JSONL replay files, tarballs, and run logs. Do not commit base
weights, downloaded datasets, LoRA checkpoints, replay buffers, metrics outputs,
or generated logs.

## One-Time Setup

Clone and enter the repo:

```bash
git clone https://github.com/nicebro123/EvoCo-RAG.git
cd EvoCo-RAG
mkdir -p ../rag_assets
```

Install the GPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Data Download

Download the preprocessed PopQA release:

```bash
mkdir -p ../rag_assets
gh release download data-v0 \
  --repo nicebro123/EvoCo-RAG \
  --pattern evoco_popqa_data.tar.gz \
  --dir /tmp
tar -xzf /tmp/evoco_popqa_data.tar.gz -C ../rag_assets
```

Alternative without GitHub CLI:

```bash
mkdir -p ../rag_assets
curl -L \
  https://github.com/nicebro123/EvoCo-RAG/releases/download/data-v0/evoco_popqa_data.tar.gz \
  -o /tmp/evoco_popqa_data.tar.gz
tar -xzf /tmp/evoco_popqa_data.tar.gz -C ../rag_assets
```

Verify:

```bash
test -f ../rag_assets/data_v33/Pop/train_labels_list.json
test -f ../rag_assets/data/Pop/test.json
```

## Weight Download

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

Optional mainland-China mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download the reranker:

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir ../rag_assets/base_models/reranker/bge-reranker-v2-m3
```

Download the generator:

```bash
hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

Verify:

```bash
test -d ../rag_assets/base_models/reranker/bge-reranker-v2-m3
test -d ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

## Two-H20 Run Ladder

Use `CUDA_VISIBLE_DEVICES=2,3` when the allocated physical GPUs are 2 and 3.
PyTorch will see them as logical `cuda:0` and `cuda:1`.

Start with a small smoke run:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/experiments/two_h20_smoke.yaml
```

Then run the 512-sample fast configuration:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/evoco_popqa_fast.yaml
```

Then run the main policy experiment:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/experiments/two_h20_main_policy.yaml
```

Resume an interrupted completed-round experiment:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/experiments/two_h20_main_policy.yaml \
  --resume
```

If the process stops during experience generation before checkpointing, rerun
the same command without deleting the output directory. Valid partial
`replay/round_xxx.jsonl` rows are reused, corrupted trailing rows are skipped,
and only missing sample IDs are regenerated.

## Evaluation

Evaluate the latest adapters implied by a config:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/experiments/two_h20_main_policy.yaml
```

Evaluate explicit adapter rounds:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/experiments/two_h20_main_policy.yaml \
  --small_lora ../rag_assets/checkpoints/experiments/two_h20_main_policy/small/round_002 \
  --large_lora ../rag_assets/checkpoints/experiments/two_h20_main_policy/large/round_002
```

## Ablations

List built-in ablation names:

```bash
python scripts/run_ablations.py --list
```

Run all ablations with real models:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_ablations.py \
  --config configs/experiments/two_h20_main_policy.yaml
```

Run selected ablations:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_ablations.py \
  --config configs/experiments/two_h20_main_policy.yaml \
  --only evoco_full evoco_no_audit evoco_answer_only_reward
```

CPU-safe wiring check:

```bash
python scripts/run_ablations.py \
  --config configs/evoco_popqa_fast.yaml \
  --no_models
```

Each ablation writes to:

```text
<config output_dir>/ablations/<experiment_name>/
```

The summary file is:

```text
<config output_dir>/ablations/summary.json
```

## Hyperparameter Templates

The standalone templates under `configs/experiments/` cover the first set of
planned studies:

| Config | Main variable | Why run it |
|---|---|---|
| `hparam_cost_top3.yaml` | `top_k=3`, `num_audit_candidates=1`, shorter prompt | Lower cost and faster rounds |
| `hparam_precision_top8.yaml` | `top_k=8`, `max_selected_docs=8`, longer prompt | Higher evidence recall |
| `hparam_audit_self_consistency.yaml` | `num_audit_candidates=5` | More reliable audit selection |
| `two_h20_main_policy.yaml` | policy heads + hybrid action | Main EvoCo-RAG setting |

For a new hyperparameter run, copy one template and change:

```yaml
project:
  name: evoco_rag_<new_name>
  output_dir: ../rag_assets/outputs/experiments/<new_name>
models:
  small_lora_dir: ../rag_assets/checkpoints/experiments/<new_name>/small
  large_lora_dir: ../rag_assets/checkpoints/experiments/<new_name>/large
```

Do not reuse checkpoint roots across independent experiments.

## Main Runtime Knobs

| Field | Effect | Two-H20 starting point |
|---|---|---|
| `contract.top_k` | Candidate documents passed to the evidence contract | `5` |
| `contract.max_selected_docs` | Documents included as selected evidence | `5` |
| `runtime.candidate_doc_char_limit` | Characters per candidate doc in audit prompt | `1200` |
| `runtime.num_audit_candidates` | Number of large-model audit candidates | `3` |
| `runtime.audit_batch_size` | Batch size for large-model audit generation | `2`, try `4` after stable |
| `training.batch_size` | Small reranker training batch size | `4` or `8` |
| `training.large_batch_size` | Large LoRA SFT batch size | `2` |
| `small_policy.use_policy_heads` | Enable evidence/action/confidence heads | `true` for policy experiments |

If CUDA OOM happens, reduce in this order:

```text
runtime.audit_batch_size
training.large_batch_size
runtime.max_prompt_length
runtime.candidate_doc_char_limit
contract.top_k / max_selected_docs
```

## Output Audit

During training, watch progress:

```bash
wc -l ../rag_assets/outputs/experiments/two_h20_main_policy/replay/round_000.jsonl
tail -n 20 ../rag_assets/outputs/experiments/two_h20_main_policy/metrics/round_000.json
```

Inspect a replay file:

```bash
python scripts/inspect_replay.py \
  --replay ../rag_assets/outputs/experiments/two_h20_main_policy/replay/round_000.jsonl
```

Every round records stage timing:

```json
{
  "timing": {
    "experience_generation_seconds": 0.0,
    "small_training_seconds": 0.0,
    "large_training_seconds": 0.0,
    "evaluation_seconds": 0.0,
    "total_round_seconds": 0.0
  }
}
```

## Code Checks

Run before pushing:

```bash
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py tests/*.py
python -m pytest -q
python scripts/run_ablations.py --config configs/evoco_popqa_fast.yaml --no_models
python scripts/inspect_replay.py --replay ../rag_assets/outputs/evoco_popqa_fast/ablations/evoco_full/replay/round_000.jsonl
git diff --check
git status --short --branch
```

Expected local CPU-safe result at the time of writing:

```text
57 passed, 4 skipped
```

The skipped tests are torch-dependent fake-model tests in CPU-only local
environments. On the GPU server with torch installed, they should run instead of
skipping.
