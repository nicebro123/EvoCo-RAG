# EvoCo-RAG

Evidence-contract driven small-large model co-evolution for RAG.

This repository contains the training and evaluation code. Datasets, base
models, LoRA checkpoints, and outputs are kept outside Git history under a
sibling `rag_assets/` directory.

## Reproduction Overview

The expected workflow is:

```text
1. Clone this private repository
2. Download the preprocessed PopQA data release
3. Download the reranker and generator base weights from Hugging Face
4. Install GPU dependencies
5. Run debug training
6. Run full training
7. Run evaluation
```

## 1. Clone

This repository is private. Make sure your GitHub account has access.

```bash
gh auth login
git clone https://github.com/nicebro123/EvoCo-RAG.git
cd EvoCo-RAG
```

Create the asset root next to the repository:

```bash
mkdir -p ../rag_assets
```

Expected layout after all downloads:

```text
parent/
├── EvoCo-RAG/
└── rag_assets/
    ├── data/Pop/test.json
    ├── data_v33/Pop/train_labels_list.json
    ├── base_models/
    │   ├── reranker/bge-reranker-v2-m3/
    │   └── generator/Meta-Llama-3.1-8B-Instruct/
    ├── checkpoints/
    ├── outputs/
    └── outputs_debug/
```

## 2. Download Data

The code expects the project-preprocessed PopQA data, not raw PopQA. The
training file already contains retrieved contexts and document-level seed labels;
the test file contains retrieved contexts under `ctxs`.

Download the private release asset:

```bash
mkdir -p ../rag_assets
gh release download data-v0 \
  --repo nicebro123/EvoCo-RAG \
  --pattern evoco_popqa_data.tar.gz \
  --dir /tmp
tar -xzf /tmp/evoco_popqa_data.tar.gz -C ../rag_assets
```

Verify the required files:

```bash
test -f ../rag_assets/data_v33/Pop/train_labels_list.json
test -f ../rag_assets/data/Pop/test.json
```

Required schemas:

```text
train_labels_list.json:
  question: str
  answers: list[str]
  context: list[str]        # each item: "title: ...\ncontext: ..."
  labels: list[list[str]]   # document-level seed labels/history

test.json:
  question: str
  answers: list[str]
  ctxs: list[dict]          # each dict has title/text and optional metadata
```

## 3. Download Base Weights

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

Login to Hugging Face. This is required for Llama 3.1 because the model is gated:

```bash
hf auth login
```

Before downloading Llama, request/accept access on the model page:

```text
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

Download the small reranker:

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir ../rag_assets/base_models/reranker/bge-reranker-v2-m3
```

Download the large generator:

```bash
hf download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir ../rag_assets/base_models/generator/Meta-Llama-3.1-8B-Instruct
```

Verify the paths expected by the configs:

```bash
test -d ../rag_assets/base_models/reranker/bge-reranker-v2-m3
test -d ../rag_assets/base_models/generator/Meta-Llama-3.1-8B-Instruct
```

## 4. Install Environment

For GPU training, install PyTorch according to your CUDA environment first:

```bash
# Example only. Use the command from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies:

```bash
pip install -U transformers peft trl datasets sentence-transformers accelerate pyyaml numpy pytest
```

If using 4-bit loading, also install:

```bash
pip install -U bitsandbytes
```

For CPU-only code checks:

```bash
pip install -r requirements-cpu.txt
```

## 5. Train

Start with the debug configuration. It uses 16 training samples and writes to
`../rag_assets/outputs_debug/latest`.

```bash
python scripts/train_evoco.py --config configs/debug.yaml
```

Run the full PopQA configuration:

```bash
python scripts/train_evoco.py --config configs/evoco_popqa.yaml
```

Resume from the latest completed round:

```bash
python scripts/train_evoco.py --config configs/evoco_popqa.yaml --resume
```

Important: fresh training refuses to overwrite existing `round_*` adapters. For
a new experiment, edit these fields in the YAML:

```yaml
project:
  output_dir: ../rag_assets/outputs/evoco_popqa_v2
models:
  small_lora_dir: ../rag_assets/checkpoints/evoco_popqa_v2/small
  large_lora_dir: ../rag_assets/checkpoints/evoco_popqa_v2/large
```

## 6. Evaluate

Evaluate with the latest checkpoint roots from the config:

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

Evaluation writes:

```text
../rag_assets/outputs/evoco_popqa/metrics/test_eval.json
```

Gold answers are used only for offline metrics, not inserted into the generation
prompt.

## 7. Outputs

Each run writes:

```text
../rag_assets/outputs*/used_config.yaml
../rag_assets/outputs*/weights_manifest.json
../rag_assets/outputs*/replay/round_000.jsonl
../rag_assets/outputs*/replay/all.jsonl
../rag_assets/outputs*/contracts/round_000.jsonl
../rag_assets/outputs*/audits/round_000.jsonl
../rag_assets/outputs*/metrics/round_000.json
../rag_assets/checkpoints/*/small/round_000/
../rag_assets/checkpoints/*/large/round_000/
```

`weights_manifest.json` records the resolved base model paths, checkpoint roots,
latest adapters, and legacy adapter references.

## 8. Method Summary

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
correctly from parametric knowledge while the selected evidence is wrong.

## 9. Repository Structure

```text
configs/              YAML experiment configs
docs/                 paper idea and engineering design documents
evoco_rag/            EvoCo-RAG package
scripts/              train/eval/replay/ablation CLI entrypoints
tests/                CPU-safe tests
run_train.py          legacy CoRAG baseline entrypoint
run_test.py           legacy CoRAG baseline evaluator
```

Recommended entrypoints:

```text
scripts/train_evoco.py
scripts/eval_evoco.py
scripts/run_ablations.py
scripts/build_seed_replay.py
scripts/inspect_replay.py
```

## 10. Optional Code Checks

These checks do not reproduce the full experiment; they only verify that the code
and no-model data pipeline are wired correctly.

```bash
python -m pytest -q
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py
python scripts/build_seed_replay.py --config configs/debug.yaml
python scripts/run_ablations.py --config configs/debug.yaml --no_models
```

Current local check status:

```text
python -m pytest -q
38 passed
```

## References

- Hugging Face Hub download guide: https://huggingface.co/docs/huggingface_hub/en/guides/download
- BGE reranker model: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Llama 3.1 8B Instruct model: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
