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

Download the large generator / auditor:

```bash
hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

The expected paths are:

```text
../rag_assets/base_models/reranker/bge-reranker-v2-m3
../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

Verify:

```bash
test -d ../rag_assets/base_models/reranker/bge-reranker-v2-m3 && echo "reranker OK"
test -d ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407 && echo "generator OK"
```

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

Or run smoke test first and then launch training automatically:

```bash
bash run.sh all \
  --gpus 0,1 \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

`run.sh train` starts a tmux session named `evoco_all_experiments`.

Attach:

```bash
tmux attach -t evoco_all_experiments
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
../rag_assets/outputs/experiments/evoco_all_experiments/
../rag_assets/checkpoints/
```

The official queue uses full datasets and runs PopQA sweeps, PopQA
hyperparameter studies, multi-dataset checks, and mechanism ablations. Detailed
study specs live in `configs/experiments/`.

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
- Generator: [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- Hugging Face download guide: [huggingface_hub download](https://huggingface.co/docs/huggingface_hub/en/guides/download)
