# EvoCo-RAG

Evidence-contract driven small-large model co-evolution for RAG.

This repository contains code only. Datasets, base models, LoRA adapters,
checkpoints, and generated outputs are intentionally kept outside the Git repo
under a sibling `rag_assets/` directory.

## Quick Reproduction

Clone the repository:

```bash
git clone https://github.com/nicebro123/EvoCo-RAG.git
cd EvoCo-RAG
```

Create the expected local asset layout next to the repo:

```bash
mkdir -p ../rag_assets/{data/Pop,data_v33/Pop,base_models/reranker,base_models/generator,adapters,checkpoints,outputs,outputs_debug}
```

Place the PopQA files here:

```text
../rag_assets/data_v33/Pop/train_labels_list.json
../rag_assets/data/Pop/test.json
```

For CPU-only smoke reproduction, install the minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-cpu.txt
```

Run the engineering checks:

```bash
python -m pytest -q
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py
```

Run the no-model data-flow smoke test:

```bash
python scripts/build_seed_replay.py --config configs/debug.yaml
python scripts/inspect_replay.py --replay ../rag_assets/outputs_debug/latest/replay/round_000.jsonl
python scripts/run_ablations.py --config configs/debug.yaml --no_models
```

Expected CPU smoke outputs:

```text
../rag_assets/outputs_debug/latest/replay/round_000.jsonl
../rag_assets/outputs_debug/latest/contracts/round_000.jsonl
../rag_assets/outputs_debug/latest/audits/round_000.jsonl
../rag_assets/outputs_debug/latest/weights_manifest.json
../rag_assets/outputs_debug/latest/ablations/summary.json
```

These commands do not load the reranker or generator base models. They verify
schema conversion, evidence-contract construction, replay writing, metrics, and
ablation wiring.

## Asset Layout

Full local layout:

```text
parent/
├── EvoCo-RAG/
└── rag_assets/
    ├── data/Pop/test.json
    ├── data_v33/Pop/train_labels_list.json
    ├── base_models/
    │   ├── reranker/bge-reranker-v2-m3/
    │   └── generator/Meta-Llama-3.1-8B-Instruct/
    ├── adapters/
    │   ├── reranker-CoRAG/
    │   └── generator-CoRAG/
    ├── checkpoints/
    │   ├── debug/
    │   └── evoco_popqa/
    ├── outputs/
    └── outputs_debug/
```

The repository `.gitignore` excludes data, model weights, checkpoints, outputs,
and zip archives. Do not commit `rag_assets/`.

## Configs

Main configs:

```text
configs/debug.yaml        # 16-sample smoke config
configs/evoco_popqa.yaml  # full PopQA config
```

Important default paths:

```yaml
project:
  output_dir: ../rag_assets/outputs_debug/latest

data:
  train_path: ../rag_assets/data_v33/Pop/train_labels_list.json
  test_path: ../rag_assets/data/Pop/test.json

models:
  small_base_path: ../rag_assets/base_models/reranker/bge-reranker-v2-m3
  large_base_path: ../rag_assets/base_models/generator/Meta-Llama-3.1-8B-Instruct
  small_lora_dir: ../rag_assets/checkpoints/debug/small
  large_lora_dir: ../rag_assets/checkpoints/debug/large
```

Change these YAML paths if your data or model weights live elsewhere.

## GPU Reproduction

Install GPU dependencies matching your CUDA/PyTorch environment:

```text
torch
transformers
peft
trl
datasets
sentence-transformers
bitsandbytes, only if use_4bit=true
```

Put or symlink base models to:

```text
../rag_assets/base_models/reranker/bge-reranker-v2-m3/
../rag_assets/base_models/generator/Meta-Llama-3.1-8B-Instruct/
```

Run debug training first:

```bash
python scripts/train_evoco.py --config configs/debug.yaml
```

Then run the full PopQA experiment:

```bash
python scripts/train_evoco.py --config configs/evoco_popqa.yaml
```

Resume from the latest complete `round_XXX` adapter:

```bash
python scripts/train_evoco.py --config configs/evoco_popqa.yaml --resume
```

Evaluate:

```bash
python scripts/eval_evoco.py --config configs/evoco_popqa.yaml
```

Evaluate explicit adapter rounds:

```bash
python scripts/eval_evoco.py \
  --config configs/evoco_popqa.yaml \
  --small_lora ../rag_assets/checkpoints/evoco_popqa/small/round_002 \
  --large_lora ../rag_assets/checkpoints/evoco_popqa/large/round_002
```

## Method Summary

EvoCo-RAG changes the original answer-only loop:

```text
reranker selects top1_doc
generator answers
answer hit => reward top1_doc
```

into a responsibility-aware loop:

```text
small model proposes an EvidenceContract
large model answers and audits evidence
rule verifier checks answer/evidence/citation
decomposed reward assigns responsibility
replay buffer stores structured experience
small LoRA and large LoRA are updated separately
```

The core idea is to avoid rewarding the retriever when the generator answers
correctly from parametric knowledge while the selected evidence is wrong. The
small model receives positive feedback only when audited evidence supports the
answer; the large model is optimized for faithful generation, citations, and
structured auditing.

## Repository Structure

```text
configs/              YAML experiment configs
docs/                 paper idea and engineering design documents
evoco_rag/            new EvoCo-RAG package
scripts/              train/eval/replay/ablation CLI entrypoints
tests/                CPU-safe unit and integration tests
run_train.py          legacy CoRAG baseline entrypoint
run_test.py           legacy CoRAG baseline evaluator
```

Recommended entrypoints:

```text
scripts/build_seed_replay.py
scripts/train_evoco.py
scripts/eval_evoco.py
scripts/run_ablations.py
scripts/inspect_replay.py
```

`run_train.py` and `run_test.py` are retained for baseline comparison only.

## Current Verification

Verified locally without loading GPU models:

```text
python -m pytest -q
38 passed

python -m py_compile ...
passed

python scripts/build_seed_replay.py --config configs/debug.yaml
passed

python scripts/run_ablations.py --config configs/debug.yaml --no_models
passed
```

Still required for a full paper-grade reproduction:

```text
base model download or symlink validation
GPU loading of bge-reranker-v2-m3
GPU loading of Meta-Llama-3.1-8B-Instruct
real LoRA checkpoint save/resume
multi-round convergence metrics
```
