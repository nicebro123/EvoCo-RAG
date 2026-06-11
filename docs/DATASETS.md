# Dataset & Weight Setup (数据集安置文档)

This document covers **everything you download and place on disk before running
anything**: the asset directory, the dataset pack, and the base model weights.

- For the full documentation map, see [README.md](README.md).
- For installing the environment and running training/eval/ablations, see
  [EXPERIMENTS.md](EXPERIMENTS.md).
- For running tests and CPU-safe code checks, see [TESTING.md](TESTING.md).

---

## 1. Asset Layout Principle

The GitHub repository is **code-only**: scripts, configs, tests, docs. Everything
heavy — datasets, model weights, LoRA checkpoints, replay buffers, metrics —
lives in a **sibling** directory `../rag_assets/`, never committed to Git.

```text
parent/
├── EvoCo-RAG/        # this Git repo (code only)
└── rag_assets/       # all assets, NOT in Git
    ├── rag_data/                 # downloaded tarball + extracted dataset pack
    │   └── evoco_dataset_pack/
    ├── base_models/              # reranker + generator weights
    ├── checkpoints/              # LoRA adapters per run
    ├── outputs/                  # full-run outputs
    └── outputs_debug/            # debug-run outputs
```

Create the asset root once:

```bash
git clone https://github.com/nicebro123/EvoCo-RAG.git
cd EvoCo-RAG
mkdir -p ../rag_assets
```

`.gitignore` already blocks weights, adapters, JSONL replay files, tarballs, and
logs. Do not force-add them.

---

## 2. Download the Dataset Pack

Datasets are distributed as a single pre-processed pack (not raw
PopQA/HotpotQA/NQ/ASQA files).

Google Drive:

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

If `gdown` cannot access the folder, open the URL in a browser and download the
three files manually into `../rag_assets/rag_data/`:

```text
evoco_dataset_pack.tar.gz
evoco_dataset_pack.tar.gz.sha256
UPLOAD_README.md
```

Verify the checksum and unpack:

```bash
cd ../rag_assets/rag_data
shasum -a 256 -c evoco_dataset_pack.tar.gz.sha256
tar -xzf evoco_dataset_pack.tar.gz -C .
cd ../../EvoCo-RAG
```

Expected SHA256:

```text
803c08ec4626da3f7add8a1c0e1dfc7792bd4997fa2d26c7203a27fe56186d28  evoco_dataset_pack.tar.gz
```

Layout after extraction:

```text
../rag_assets/
└── rag_data/
    ├── evoco_dataset_pack.tar.gz
    ├── evoco_dataset_pack.tar.gz.sha256
    └── evoco_dataset_pack/
        ├── dataset_registry.json
        ├── datasets.yaml
        └── datasets/
            └── <dataset_id>/
                ├── dataset_meta.json
                ├── data_v33/Pop/train_labels_list.json
                └── data/Pop/test.json
```

---

## 3. Included Datasets

| Dataset id | Train | Test | Source |
|---|---:|---:|---|
| `popqa_standard` | 12,868 | 1,399 | legacy EvoCo PopQA preprocessing |
| `hotpotqa_distractor` | 90,447 | 7,405 | `hotpotqa/hotpot_qa` distractor |
| `nq_reader` | 50,000 | 3,119 | `nlpconnect/dpr-nq-reader-v2` |
| `asqa_dpr` | 4,353 | 948 | `dormosol/asqa_dpr_pyserini_top100_with_text` |
| `popqa_retrieval` | 11,413 | 2,854 | `MinaGabriel/popqa-retrieval-top20` |

Every dataset uses the same loader schema:

```text
train_labels_list.json
  question: str
  answers:  list[str]
  context:  list[str]        # each item "title: ...\ncontext: ..."
  labels:   list[list[str]]  # document-level seed labels/history

test.json
  question: str
  answers:  list[str]
  ctxs:     list[dict]       # each dict has title/text (+ optional metadata)
```

Validate the pack after extraction (checks registry, files, counts, required
fields, and the actual `evoco_rag.data` loader output):

```bash
python scripts/verify_dataset_pack.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
```

The dataset tools also accept the parent directories
`../rag_assets/rag_data` or `../rag_assets`; they will resolve the extracted
`evoco_dataset_pack` automatically.

---

## 4. Generate Run Configs from the Pack

Configs are generated into `configs/local/` (Git-ignored), so they never
pollute the repo.

List dataset ids:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack --list
```

Generate **fast** (512-sample) configs for all datasets:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
  --all --output-root configs/local
```

Generate **full**-run configs for all datasets:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
  --all --full --output-root configs/local
```

Generate one **debug** config (16 samples) for a smoke run:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
  --dataset-id popqa_standard \
  --debug-size 16 \
  --name evoco_popqa_standard_debug \
  --output configs/local/popqa_standard_debug.yaml \
  --output-dir ../rag_assets/outputs_debug/popqa_standard \
  --checkpoint-root ../rag_assets/checkpoints/debug/popqa_standard
```

Generated files, e.g.:

```text
configs/local/popqa_standard_fast.yaml
configs/local/hotpotqa_distractor_fast.yaml
configs/local/nq_reader_fast.yaml
configs/local/asqa_dpr_fast.yaml
configs/local/popqa_retrieval_fast.yaml
```

---

## 5. Download Base Model Weights

EvoCo-RAG uses a small reranker (cross-encoder) and a large generator/auditor.

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

If you are in mainland China, set the mirror first:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download the small reranker:

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir ../rag_assets/base_models/reranker/bge-reranker-v2-m3
```

Download the large generator (default `mistralai/Mistral-Nemo-Instruct-2407`):

```bash
hf download mistralai/Mistral-Nemo-Instruct-2407 \
  --local-dir ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407
```

Verify the exact paths the configs expect:

```bash
test -d ../rag_assets/base_models/reranker/bge-reranker-v2-m3 && echo "reranker OK"
test -d ../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407 && echo "generator OK"
```

If a config points at a different base path, edit `models.small_base_path` /
`models.large_base_path` in that YAML.

---

## 6. Repository Hygiene

Do **not** commit any of the following — they all belong under `../rag_assets/`:

```text
raw/converted dataset JSON or parquet
model weights and tokenizers
LoRA checkpoints (checkpoints/**/round_*)
replay buffers (replay/*.jsonl)
metrics outputs and run logs
dataset tarballs
generated configs (configs/local/**)
```

Once datasets and weights are in place, continue with
[EXPERIMENTS.md](EXPERIMENTS.md).
