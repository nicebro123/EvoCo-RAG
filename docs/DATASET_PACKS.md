# Dataset Packs

EvoCo-RAG keeps datasets outside Git history. The public code repository should
contain scripts, configs, tests, and documentation only; converted datasets,
model weights, checkpoints, replay buffers, and metrics outputs belong under a
local sibling asset directory such as `../rag_assets/`.

## Google Drive Pack

The current multi-dataset pack is hosted on Google Drive:

```text
https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv
```

The folder contains:

```text
evoco_dataset_pack.tar.gz
evoco_dataset_pack.tar.gz.sha256
UPLOAD_README.md
```

Expected SHA256:

```text
803c08ec4626da3f7add8a1c0e1dfc7792bd4997fa2d26c7203a27fe56186d28  evoco_dataset_pack.tar.gz
```

Download with `gdown`:

```bash
pip install -U gdown
mkdir -p ../rag_assets/rag_data
gdown --folder \
  "https://drive.google.com/drive/folders/1FdzMIxnotAynWIWZMx5XzATNVCpm43iv" \
  -O ../rag_assets/rag_data
```

If `gdown` cannot access the folder, open the Google Drive URL in a browser,
download the three files manually, and place them in `../rag_assets/rag_data/`.

Verify and unpack:

```bash
cd ../rag_assets/rag_data
shasum -a 256 -c evoco_dataset_pack.tar.gz.sha256
tar -xzf evoco_dataset_pack.tar.gz -C ..
cd ../../EvoCo-RAG
```

Expected local layout after extraction:

```text
../rag_assets/
├── rag_data/
│   ├── evoco_dataset_pack.tar.gz
│   ├── evoco_dataset_pack.tar.gz.sha256
│   └── UPLOAD_README.md
└── evoco_dataset_pack/
    ├── dataset_registry.json
    ├── datasets.yaml
    └── datasets/
        └── <dataset_id>/
            ├── dataset_meta.json
            ├── data_v33/Pop/train_labels_list.json
            └── data/Pop/test.json
```

## Included Datasets

| Dataset id | Dataset name | Train | Test | Source |
|---|---|---:|---:|---|
| `popqa_standard` | `PopQAStandard` | 12,868 | 1,399 | legacy EvoCo PopQA preprocessing |
| `hotpotqa_distractor` | `HotpotQADistractor` | 90,447 | 7,405 | `hotpotqa/hotpot_qa` distractor |
| `nq_reader` | `NQReader` | 50,000 | 3,119 | `nlpconnect/dpr-nq-reader-v2` |
| `asqa_dpr` | `ASQADPR` | 4,353 | 948 | `dormosol/asqa_dpr_pyserini_top100_with_text` |
| `popqa_retrieval` | `PopQARetrieval` | 11,413 | 2,854 | `MinaGabriel/popqa-retrieval-top20` |

Each converted dataset follows the same schema expected by the current loader:

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

Run the strict pack validator after extraction:

```bash
python scripts/verify_dataset_pack.py \
  --data-root ../rag_assets/evoco_dataset_pack
```

The validator checks registry entries, file existence, sample counts, required
raw fields, and the actual `evoco_rag.data` loader output.

## Generate Configs

List datasets:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --list
```

Generate fast configs for all datasets:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --all \
  --output-root configs/local
```

Generated files:

```text
configs/local/popqa_standard_fast.yaml
configs/local/hotpotqa_distractor_fast.yaml
configs/local/nq_reader_fast.yaml
configs/local/asqa_dpr_fast.yaml
configs/local/popqa_retrieval_fast.yaml
```

Generate full-run configs for all datasets:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --all \
  --full \
  --output-root configs/local
```

Generate one custom debug config:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --dataset-id popqa_standard \
  --debug-size 16 \
  --name evoco_popqa_standard_debug \
  --output configs/local/popqa_standard_debug.yaml \
  --output-dir ../rag_assets/outputs_debug/popqa_standard \
  --checkpoint-root ../rag_assets/checkpoints/debug/popqa_standard
```

## Train and Evaluate

Run a fast dataset experiment:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_retrieval_fast.yaml
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_retrieval_fast.yaml \
  --resume
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/local/popqa_retrieval_fast.yaml
```

## Repository Policy

- Keep data packs on Google Drive, Hugging Face Datasets, or another data host.
- Put only scripts, registry examples, and documentation in GitHub.
- Do not commit raw parquet files, converted JSON files, model weights,
  checkpoints, replay buffers, tarballs, metrics outputs, or run logs.
- Keep generated dataset configs in `configs/local/`; this directory is ignored
  by Git.
