# EvoCo-RAG

Evidence-contract driven small-large model co-evolution for RAG.

This repository contains the training and evaluation code. Datasets, base
models, LoRA checkpoints, and outputs are kept outside Git history under a
sibling `rag_assets/` directory.

## Reproduction Overview

The expected workflow is:

```text
1. Clone this repository
2. Download the Google Drive dataset pack
3. Download the reranker and generator base weights
4. Install GPU dependencies
5. Run debug training
6. Run full training
7. Run evaluation
```

For a complete reproduction and experiment guide, including two-H20 settings,
ablation commands, hyperparameter templates, and repository hygiene rules, see
[docs/REPRODUCIBILITY_AND_EXPERIMENTS.md](docs/REPRODUCIBILITY_AND_EXPERIMENTS.md).

For multi-dataset packs hosted outside GitHub, see
[docs/DATASET_PACKS.md](docs/DATASET_PACKS.md). The helper
`scripts/make_dataset_config.py` generates runnable YAML configs from a
downloaded `evoco_dataset_pack/dataset_registry.json`.

## 1. Clone

```bash
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
    ├── evoco_dataset_pack/
    │   ├── dataset_registry.json
    │   └── datasets/
    │       ├── popqa_standard/
    │       ├── hotpotqa_distractor/
    │       ├── nq_reader/
    │       ├── asqa_dpr/
    │       └── popqa_retrieval/
    ├── base_models/
    │   ├── reranker/bge-reranker-v2-m3/
    │   └── generator/Mistral-Nemo-Instruct-2407/
    ├── checkpoints/
    ├── outputs/
    └── outputs_debug/
```

## 2. Download Data

The code expects the project-preprocessed EvoCo-RAG dataset pack, not raw
PopQA/HotpotQA/NQ/ASQA files. Each dataset in the pack uses the same loader
layout:

```text
datasets/<dataset_id>/data_v33/Pop/train_labels_list.json
datasets/<dataset_id>/data/Pop/test.json
```

Google Drive folder:

```text
https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv
```

Download with `gdown`:

```bash
pip install -U gdown
mkdir -p ../rag_assets/rag_data
gdown --folder \
  "https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv" \
  -O ../rag_assets/rag_data
```

The folder contains:

```text
evoco_dataset_pack.tar.gz
evoco_dataset_pack.tar.gz.sha256
UPLOAD_README.md
```

Verify and unpack:

```bash
cd ../rag_assets/rag_data
shasum -a 256 -c evoco_dataset_pack.tar.gz.sha256
tar -xzf evoco_dataset_pack.tar.gz -C ..
cd ../../EvoCo-RAG
```

Expected checksum:

```text
803c08ec4626da3f7add8a1c0e1dfc7792bd4997fa2d26c7203a27fe56186d28  evoco_dataset_pack.tar.gz
```

List available datasets:

```bash
python scripts/verify_dataset_pack.py \
  --data-root ../rag_assets/evoco_dataset_pack

python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --list
```

Current dataset ids:

| Dataset id | Train | Test | Notes |
|---|---:|---:|---|
| `popqa_standard` | 12,868 | 1,399 | Original EvoCo-RAG PopQA layout |
| `hotpotqa_distractor` | 90,447 | 7,405 | HotpotQA distractor converted to EvoCo format |
| `nq_reader` | 50,000 | 3,119 | DPR NQ reader converted to EvoCo format |
| `asqa_dpr` | 4,353 | 948 | ASQA DPR passages converted to EvoCo format |
| `popqa_retrieval` | 11,413 | 2,854 | PopQA retrieval top-20 converted to EvoCo format |

Generate runnable configs for every dataset:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --all \
  --output-root configs/local
```

Generate full-run configs instead of 512-sample fast configs:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --all \
  --full \
  --output-root configs/local
```

`configs/local/` is ignored by Git, so locally generated dataset configs do not
pollute the repository.

## 3. Download Base Weights

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

If you are downloading from mainland China, use the Hugging Face mirror before
running `hf download`:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download the small reranker:

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir ../rag_assets/base_models/reranker/bge-reranker-v2-m3
```

Download the large generator. The default generator is
`mistralai/Mistral-Nemo-Instruct-2407`, a stronger 12B instruct model used as the
large model in EvoCo-RAG:

```bash
hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

If you do not use the mirror, the same command works against the official
Hugging Face endpoint:

```bash
unset HF_ENDPOINT
hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

Verify the paths expected by the configs:

```bash
test -d ../rag_assets/base_models/reranker/bge-reranker-v2-m3
test -d ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

## 4. Install GPU Environment

The default reproduction environment is pinned to:

```text
Python 3.10 or 3.11
CUDA 12.1 PyTorch wheels
torch==2.5.1+cu121
transformers==4.46.3
peft==0.14.0
trl==0.14.0
```

Create the environment and install the pinned GPU requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
```

Verify PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
PY
```

If your server uses another CUDA wheel target, keep the same non-torch package
versions but replace the PyTorch index/wheel lines in `requirements-gpu.txt`
with the matching command from https://pytorch.org/get-started/locally/.

If using 4-bit loading, also install:

```bash
pip install bitsandbytes==0.45.0
```

`requirements-cpu.txt` is only for CPU-only code checks. It is not sufficient
for training or evaluation with real models.

```bash
pip install -r requirements-cpu.txt
```

## 5. Train

When using the Google Drive dataset pack, run generated local configs from
`configs/local/`. Start with a 16-sample PopQA debug config:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --dataset-id popqa_standard \
  --debug-size 16 \
  --name evoco_popqa_standard_debug \
  --output configs/local/popqa_standard_debug.yaml \
  --output-dir ../rag_assets/outputs_debug/popqa_standard \
  --checkpoint-root ../rag_assets/checkpoints/debug/popqa_standard

CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_debug.yaml
```

Then run a 512-sample fast config for the dataset you want. For the standard
PopQA split:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_fast.yaml
```

For a different dataset, replace only the config name:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/hotpotqa_distractor_fast.yaml

CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/nq_reader_fast.yaml

CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/asqa_dpr_fast.yaml

CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_retrieval_fast.yaml
```

Run a full config only after the corresponding fast run finishes normally:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_full.yaml
```

For experiment batches, use the SpecFlow-style launcher. It expands a compact
study spec into per-run configs with isolated output/checkpoint paths:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml
```

After inspecting the generated `run_config.yaml` files and commands, launch
sequentially:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml \
  --launch
```

Run the same fast setting across all converted datasets:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/multidataset_fast_2gpu.yaml
```

Resume from the latest completed round:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_full.yaml \
  --resume
```

`--resume` continues from the latest completed checkpoint round. If a process is
interrupted during experience generation before checkpointing, rerun the same
command without deleting the output directory; the trainer will reuse valid
partial `replay/round_xxx.jsonl` rows, skip already completed sample IDs, and
generate only the remaining samples.

`CUDA_VISIBLE_DEVICES=2,3` exposes physical GPUs 2 and 3 to PyTorch. The large
generator uses `device_map="auto"` and can shard across the visible GPUs; the
small reranker uses logical `cuda:0`, which maps to the first visible GPU.
Change the list if you want a different subset, for example
`CUDA_VISIBLE_DEVICES=0,1`.

Generated fast configs use these cost-controlled starting settings:

```yaml
contract:
  top_k: 3
  max_selected_docs: 3
runtime:
  candidate_doc_char_limit: 800
  num_audit_candidates: 1
  audit_batch_size: 4
  audit_temperature: 0.7
training:
  batch_size: 8
  large_batch_size: 2
```

`candidate_doc_char_limit` increases evidence visibility in the audit prompt.
`num_audit_candidates` generates multiple answer/audit candidates and selects
the one with the strongest evidence-consistency score.
`runtime.audit_batch_size` batches large-model audit generation during training
and evaluation. `training.large_batch_size` batches large-model LoRA SFT steps.
Both improve throughput; they do not change the evidence selection objective.
On 2 H20 GPUs, start with `2`. If memory remains stable, try `4` for
`audit_batch_size`; reduce it if CUDA OOM appears.

Round generation is streamed. During a round, the trainer prints progress like
`round 0: experience 500/12868 elapsed=... rate=... eta=...`, and incrementally
writes:

```text
../rag_assets/outputs*/replay/round_000.jsonl
../rag_assets/outputs*/contracts/round_000.jsonl
../rag_assets/outputs*/audits/round_000.jsonl
```

If a full round looks slow, check the progress line and the replay file
line count before assuming the job is stuck:

```bash
wc -l ../rag_assets/outputs/datasets/popqa_standard_full/replay/round_000.jsonl
```

Each round also records stage timing in `metrics/round_xxx.json`:

```json
"timing": {
  "experience_generation_seconds": 123.4,
  "small_training_seconds": 12.3,
  "large_training_seconds": 45.6,
  "evaluation_seconds": 1.2,
  "total_round_seconds": 182.5
}
```

Important: fresh training refuses to overwrite existing `round_*` adapters. For
a new experiment, edit these fields in the YAML:

```yaml
project:
  output_dir: ../rag_assets/outputs/datasets/popqa_standard_v2
models:
  small_lora_dir: ../rag_assets/checkpoints/datasets/popqa_standard_v2/small
  large_lora_dir: ../rag_assets/checkpoints/datasets/popqa_standard_v2/large
```

## 6. Evaluate

Evaluate with the latest checkpoint roots from the config:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/local/popqa_standard_full.yaml
```

Evaluate explicit adapter rounds:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/local/popqa_standard_full.yaml \
  --small_lora ../rag_assets/checkpoints/datasets/popqa_standard_full/small/round_002 \
  --large_lora ../rag_assets/checkpoints/datasets/popqa_standard_full/large/round_002
```

Evaluation writes:

```text
../rag_assets/outputs/datasets/popqa_standard_full/metrics/test_eval.json
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

Experiment-ready configuration templates live under `configs/experiments/`.
Use a new `project.output_dir` and new `models.*_lora_dir` for every experiment
so checkpoints and metrics never overwrite another run.

## 10. Optional Code Checks

These checks do not reproduce the full experiment; they only verify that the code
and no-model data pipeline are wired correctly.

```bash
python -m pytest -q
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py tests/*.py
python scripts/verify_dataset_pack.py --data-root ../rag_assets/evoco_dataset_pack
python scripts/make_dataset_config.py --data-root ../rag_assets/evoco_dataset_pack --all --output-root configs/local
python scripts/launch_experiments.py --spec configs/experiments/popqa_fast_sweep_2gpu.yaml --no-gpu-scripts
python scripts/make_dataset_config.py --data-root ../rag_assets/evoco_dataset_pack --dataset-id popqa_standard --debug-size 16 --name evoco_popqa_standard_debug --output configs/local/popqa_standard_debug.yaml --output-dir ../rag_assets/outputs_debug/popqa_standard --checkpoint-root ../rag_assets/checkpoints/debug/popqa_standard
python scripts/build_seed_replay.py --config configs/local/popqa_standard_debug.yaml
python scripts/run_ablations.py --config configs/local/popqa_standard_fast.yaml --no_models
```

Current local check status:

```text
python -m pytest -q
63 passed, 4 skipped
python scripts/verify_dataset_pack.py --data-root ../rag_assets/evoco_dataset_pack
passed
python scripts/run_ablations.py --config configs/local/popqa_standard_fast.yaml --no_models
passed
python scripts/inspect_replay.py --replay ../rag_assets/outputs/datasets/popqa_standard_fast/ablations/evoco_full/replay/round_000.jsonl
passed
```

## References

- Hugging Face Hub download guide: https://huggingface.co/docs/huggingface_hub/en/guides/download
- BGE reranker model: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Mistral-Nemo-Instruct-2407 model: https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
