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

## Minimal Run Commands

After datasets, weights, and the GPU environment are ready, run these from the
repository root:

```bash
# Full pipeline test: data/config/code checks + official dry-run + 16-sample GPU smoke.
bash run.sh test --gpus 2,3

# Start official full-data training in tmux.
bash run.sh train --gpus 2,3

# Or run test first and then train.
bash run.sh all --gpus 2,3
```

## Official Experiment Matrix

`run.sh train` launches the official **full-data** queue: 32 runs across PopQA
sweeps, PopQA hyperparameters, multi-dataset checks, selected PopQA settings,
and mechanism ablations. The queue runs in one master tmux session so a two-GPU
H20 pair is not oversubscribed.

```bash
bash run.sh train --gpus 2,3
```

For the exact study list and low-level launcher options, see
[../configs/experiments/README.md](../configs/experiments/README.md).

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
