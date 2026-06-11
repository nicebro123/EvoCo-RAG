# EvoCo-RAG Documentation Map

This directory is the navigation layer for the project docs. The repository is
code-only; datasets, model weights, checkpoints, replay buffers, metrics, and
logs live outside Git under `../rag_assets/`.

## Recommended Reading Paths

| Goal | Read in this order |
|---|---|
| Reproduce the project from scratch | [../README.md](../README.md) -> [DATASETS.md](DATASETS.md) -> [EXPERIMENTS.md](EXPERIMENTS.md) -> [TESTING.md](TESTING.md) |
| Run PopQAStandard experiments | [DATASETS.md](DATASETS.md) -> [EXPERIMENTS.md](EXPERIMENTS.md) -> [../configs/experiments/README.md](../configs/experiments/README.md) |
| Add or adapt datasets | [DATASETS.md](DATASETS.md) -> generated `configs/local/*.yaml` -> [EXPERIMENTS.md](EXPERIMENTS.md) |
| Check code health before pushing | [TESTING.md](TESTING.md) |
| Understand the research idea | [协同进化RAG论文构想.md](协同进化RAG论文构想.md) -> [协同进化RAG代码开发文档.md](协同进化RAG代码开发文档.md) |

## Document Roles

| Document | Role | Status |
|---|---|---|
| [../README.md](../README.md) | Project overview and minimal quickstart | Public-facing entrypoint |
| [DATASETS.md](DATASETS.md) | Dataset pack, asset layout, model-weight download, local config generation | Reproduction prerequisite |
| [EXPERIMENTS.md](EXPERIMENTS.md) | GPU environment, debug/fast/full ladder, evaluation, ablations, tmux launchers | Main running guide |
| [TESTING.md](TESTING.md) | CPU checks, GPU smoke checks, pre-push checklist | Code-quality guide |
| [../configs/experiments/README.md](../configs/experiments/README.md) | Study-spec format and official experiment matrix | Reference for batch runs |
| [协同进化RAG论文构想.md](协同进化RAG论文构想.md) | Paper-level motivation, method story, CCF-A risk/innovation plan | Research draft |
| [协同进化RAG代码开发文档.md](协同进化RAG代码开发文档.md) | Engineering design, implementation mapping, ECR TODO tracker | Development record |

## Minimal Reproduction Commands

Run these from the repository root after cloning.

```bash
# Dataset pack and generated local configs
python scripts/verify_dataset_pack.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
  --all --output-root configs/local

# 16-sample real-model smoke test
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_debug.yaml

# Materialize all official studies without starting tmux
bash scripts/launch_all_experiments.sh --dry-run

# CPU-safe checks before pushing
python -m pytest -q
git diff --check
```

## Official Experiment Matrix

The all-study launcher verifies the dataset pack, regenerates local dataset
configs, materializes every study, and starts one master tmux queue. The default
queue uses full-data configs and runs sequentially so a two-GPU H20 pair is not
oversubscribed. Fast specs are kept only for debugging/preflight runs.

```bash
bash scripts/launch_all_experiments.sh --dry-run
bash scripts/launch_all_experiments.sh
```

| Study | Runs | Purpose |
|---|---:|---|
| `evoco_popqa_sweep_full_2gpu` | 5 | Full PopQAStandard sweep: top-k, reward, audit switches |
| `evoco_popqa_hparam_full_2gpu` | 10 | Full hyperparameter exploration for top-k, audit count, confidence thresholds, and context length |
| `evoco_multidataset_full_2gpu` | 5 | Full checks on PopQAStandard, HotpotQA, NQ, ASQA, and PopQA retrieval |
| `evoco_popqa_full_sweep_2gpu` | 4 | Selected full PopQAStandard cost/accuracy settings |
| `evoco_popqa_ablation_full_2gpu` | 8 | Full PopQAStandard mechanism ablations for the paper table |

Total: 32 runs.

## Asset Boundary

Keep these outside Git:

```text
../rag_assets/rag_data/evoco_dataset_pack/
../rag_assets/base_models/
../rag_assets/checkpoints/
../rag_assets/outputs/
../rag_assets/outputs_debug/
```

Generated local configs under `configs/local/` are machine-specific and should
be regenerated from the dataset pack when paths change.
