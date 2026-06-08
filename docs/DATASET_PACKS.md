# Dataset Packs

EvoCo-RAG keeps datasets outside Git history. A dataset pack can be hosted on
Google Drive and unpacked next to, or independent from, the code repository.

## Standard Layout

Each dataset uses the same layout as the original EvoCo PopQA release:

```text
evoco_dataset_pack/
├── dataset_registry.json
├── datasets.yaml
└── datasets/
    └── <dataset_id>/
        ├── dataset_meta.json
        ├── data_v33/Pop/train_labels_list.json
        └── data/Pop/test.json
```

The training loader only needs:

- `data_v33/Pop/train_labels_list.json`
- `data/Pop/test.json`

The `dataset_registry.json` file records dataset ids, display names, relative
paths, counts, upstream sources, and redistribution notes.

## Generate a Config

List datasets:

```bash
python scripts/make_dataset_config.py \
  --data-root /path/to/evoco_dataset_pack \
  --list
```

Generate a fast config:

```bash
python scripts/make_dataset_config.py \
  --data-root /path/to/evoco_dataset_pack \
  --dataset-id hotpotqa_distractor \
  --output configs/local/hotpotqa_distractor_fast.yaml
```

Run it:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/hotpotqa_distractor_fast.yaml
```

Generate a full-run config:

```bash
python scripts/make_dataset_config.py \
  --data-root /path/to/evoco_dataset_pack \
  --dataset-id hotpotqa_distractor \
  --full \
  --output configs/local/hotpotqa_distractor_full.yaml
```

## Recommended Policy

- Put data packs on Google Drive, Hugging Face Datasets, or another data host.
- Put only scripts, registry examples, and documentation in GitHub.
- Do not commit raw parquet files, converted JSON files, model weights,
  checkpoints, replay buffers, or run logs.
