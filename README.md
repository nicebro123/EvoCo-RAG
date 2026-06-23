# EvoCo-RAG

Evidence-contract driven small-large model co-evolution for
Retrieval-Augmented Generation.

EvoCo-RAG keeps the GitHub repository **code-only**. Datasets, base weights,
LoRA checkpoints, replay buffers, metrics, and logs stay in a sibling
`../rag_assets/` directory.

```text
parent/
├── EvoCo-RAG/       # this repo: code, configs, scripts, tests
└── rag_assets/     # not committed: data, model weights, checkpoints, outputs
```

The normal reproduction path is:

```text
clone repo -> install env -> download weights -> download data -> smoke test -> full experiments
```

---

## 1. Clone the Code

```bash
git clone https://github.com/nicebro123/EvoCo-RAG.git
cd EvoCo-RAG
mkdir -p ../rag_assets
```

If your server rewrites GitHub URLs through a broken mirror, remove the global
rewrite and clone again:

```bash
git config --global --get-regexp 'url\..*insteadOf'
git config --global --unset-all url.https://gitclone.com/.insteadOf
```

---

## 2. Install the GPU Environment

Recommended environment:

```text
Python 3.10 or 3.11
CUDA 12.1 PyTorch wheels
torch==2.5.1+cu121
transformers==4.46.3
peft==0.14.0
trl==0.14.0
```

Create and install:

```bash
conda create -n evoco-rag python=3.10 -y
conda activate evoco-rag

pip install -U pip
pip install -r requirements-gpu.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

If your machine needs another CUDA wheel, install the matching PyTorch build
first, then keep the remaining package versions in `requirements-gpu.txt`.

---

## 3. Download Base Model Weights

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

Mainland China users can set a mirror before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download the small reranker:

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir ../rag_assets/base_models/reranker/bge-reranker-v2-m3
```

Download the default large generator / auditor. The current public default is
Meta Llama 3 8B Instruct. This is a gated Hugging Face model, so first make
sure your Hugging Face account has accepted the model license, then login on
the server:

```bash
hf auth login

hf download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir ../rag_assets/base_models/generator/Meta-Llama-3-8B-Instruct \
  --exclude 'original/*'
```

The expected paths are:

```text
../rag_assets/base_models/reranker/bge-reranker-v2-m3
../rag_assets/base_models/generator/Meta-Llama-3-8B-Instruct
```

Verify:

```bash
test -d ../rag_assets/base_models/reranker/bge-reranker-v2-m3 && echo "reranker OK"
test -d ../rag_assets/base_models/generator/Meta-Llama-3-8B-Instruct && echo "generator OK"
```

Legacy Mistral-Nemo configs are still kept for historical comparison, but new
local configs and the current PopQA full sweep use the Llama-3-8B path above.

---

## 4. Download and Unpack the Dataset

The dataset pack is distributed from Google Drive:

[https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv](https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv)

Download with `gdown`:

```bash
pip install -U gdown
mkdir -p ../rag_assets/rag_data

gdown --folder \
  "https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv" \
  -O ../rag_assets/rag_data
```

If `gdown` fails, open the Google Drive link in a browser and manually place
these files under `../rag_assets/rag_data/`:

```text
../rag_assets/rag_data/evoco_dataset_pack.tar.gz
../rag_assets/rag_data/evoco_dataset_pack.tar.gz.sha256
../rag_assets/rag_data/UPLOAD_README.md
```

Unpack:

```bash
cd ../rag_assets/rag_data
sha256sum -c evoco_dataset_pack.tar.gz.sha256
tar -xzf evoco_dataset_pack.tar.gz -C .
cd ../../EvoCo-RAG
```

On macOS, use this checksum command instead:

```bash
shasum -a 256 -c evoco_dataset_pack.tar.gz.sha256
```

Expected extracted path:

```text
../rag_assets/rag_data/evoco_dataset_pack/
├── dataset_registry.json
├── datasets.yaml
└── datasets/
    ├── popqa_standard/
    ├── hotpotqa_distractor/
    ├── nq_reader/
    ├── asqa_dpr/
    └── popqa_retrieval/
```

Validate the pack:

```bash
python scripts/verify_dataset_pack.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

---

## 5. Smoke Test

Use `run.sh` for the public workflow. It verifies data, generates local configs,
runs code checks, dry-runs the official experiment queue, then runs one
16-sample real-model PopQAStandard smoke test.

Choose your GPU ids from `nvidia-smi`. For two A800 cards shown as `0` and `1`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash run.sh test \
  --gpus 0,1 \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

If the GPU ids are `2,3`, use:

```bash
bash run.sh test --gpus 2,3
```

A successful smoke test should load both base models, print progress like
`round 0: experience k/16`, train both LoRA adapters, inspect the replay file,
and finish with `co-evolution finished`.

Useful outputs are written outside Git:

```text
../rag_assets/outputs_debug/run_sh/<run_id>/
../rag_assets/checkpoints/debug/run_sh/<run_id>/
```

---

## 6. Run Full Experiments

After the smoke test passes, start the official full-data experiment queue:

```bash
bash run.sh train \
  --gpus 0,1 \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

For an 8-GPU machine, use 2 GPUs as one experiment worker and run four workers
in parallel:

```bash
bash run.sh train \
  --gpu-pairs '0,1;2,3;4,5;6,7' \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

This keeps each run on a 2-GPU unit while distributing the official queue across
the available GPU pairs.

Or run smoke test first and then launch training automatically:

```bash
bash run.sh all \
  --gpus 0,1 \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

With `--gpus`, `run.sh train` starts one tmux session named
`evoco_all_experiments`. With `--gpu-pairs`, it starts one worker session per
GPU pair, for example:

```text
evoco_all_experiments_g0_1
evoco_all_experiments_g2_3
evoco_all_experiments_g4_5
evoco_all_experiments_g6_7
```

Attach:

```bash
tmux attach -t evoco_all_experiments
```

For an 8-GPU run, attach to the worker you want to inspect:

```bash
tmux attach -t evoco_all_experiments_g0_1
```

Detach without stopping training:

```text
Ctrl-b then d
```

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Full experiment outputs:

```text
../rag_assets/outputs/experiments/<study_name>_v2/<run_name>/
├── train.log
├── eval.log
├── replay/
├── audits/
└── metrics/
    ├── round_000.json
    ├── test_predictions_round_000.jsonl
    ├── test_eval_round_000.json
    ├── test_predictions.jsonl
    └── test_eval.json

../rag_assets/checkpoints/experiments/<study_name>_v2/<run_name>/
```

Protocol-v2 metrics enforce non-empty schema-valid answers, record actual
generation/audit execution, and persist per-example test predictions. Do not
merge older protocol-v1 results with these runs.

Every training round performs an independent test-set inference after both
models are updated. Full configs use the complete test split; fast/debug configs
use their configured `data.eval_size` or `data.debug_size` subset. Each round
writes `test_eval_round_NNN.json` and `test_predictions_round_NNN.jsonl`.
Training stops with an error if any required per-round test evaluation cannot
run, so a training-replay diagnostic is never reported as a test result. The
last round is also published as `test_eval.json` and `test_predictions.jsonl`;
the launcher reuses these files instead of running the same final evaluation
twice.

After the queue finishes, aggregate all protocol-v3 runs into one summary and
accuracy/cost ranking table:

```bash
python scripts/summarize_experiments.py \
  --root ../rag_assets/outputs/experiments
```

The command writes JSON and CSV files under
`../rag_assets/outputs/experiments/summary_v3/`. Incomplete runs and old
protocol results remain visible in the summary but are excluded from ranking.

The current default full-data study uses Llama-3-8B-Instruct on PopQA. Legacy
Mistral-Nemo sweeps, multi-dataset checks, and mechanism ablations are still
available as explicit launcher specs under `configs/experiments/`. A short
experiment matrix is available at
[docs/官方全量实验说明.md](docs/官方全量实验说明.md).

---

## 7. Common Commands

Update code from GitHub:

```bash
cd EvoCo-RAG
git pull origin main
```

Run only preflight checks, with no GPU training:

```bash
bash run.sh preflight \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

Resume the official tmux queue if it already exists:

```bash
tmux attach -t evoco_all_experiments
```

For a directly interrupted training command, rerun it with `--resume`. Resume
loads only the last round with complete test metrics, predictions, round stats,
and model adapters; an orphan checkpoint from a failed evaluation is ignored.

If a previous official queue already produced outputs and you intentionally want
to regenerate them:

```bash
bash run.sh train --gpus 0,1 --overwrite
```

---

## 8. What Not to Commit

Do not commit data, weights, checkpoints, replay buffers, generated configs, or
logs. They belong under `../rag_assets/`.

The `.gitignore` already excludes common asset paths:

```text
configs/local/
rag_assets/
data/
base_models/
outputs*/
checkpoints/
*.tar.gz
*.jsonl
*.safetensors
*.bin
*.pt
```

---

## Method Summary

EvoCo-RAG changes a static RAG pipeline into a responsibility-aware co-evolution
loop:

```text
small model proposes an EvidenceContract
large model answers and audits evidence
rule verifier checks answer / support / citation
decomposed reward assigns responsibility
replay buffer stores structured experience
small LoRA learns evidence selection and action policy
large LoRA learns evidence-grounded answer/audit behavior
```

The key point is that the retriever is not rewarded just because the final
answer is correct. It is rewarded only when the selected evidence actually
supports the answer.

## References

- BGE reranker: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- Default generator / auditor: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- Legacy generator config: [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- Hugging Face download guide: [huggingface_hub download](https://huggingface.co/docs/huggingface_hub/en/guides/download)
