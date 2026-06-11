# Experiment Configs

For the public quickstart, dataset setup, weight download, smoke test, and full
experiment workflow, start from [../../README.md](../../README.md). This file is
the detailed reference for experiment specs and launchers.

This directory contains two experiment styles:

- Standalone training configs: complete EvoCo-RAG configs passed directly to
  `scripts/train_evoco.py`.
- Launcher specs: compact SpecFlow-style study files with `base_config`,
  `defaults`, and `experiments`. These are expanded by
  `scripts/launch_experiments.py` into immutable per-run `run_config.yaml`
  files.

All data, weights, checkpoints, and generated outputs stay outside the Git repo
under `../rag_assets/`.

## Launcher Workflow

Generate local dataset configs first:

```bash
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
  --all \
  --output-root configs/local
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
  --all --full \
  --output-root configs/local
```

Dry-run a study. This writes per-run configs, a manifest, and per-GPU shell
scripts without starting training. By default, every generated command trains
first and then evaluates the same generated config on the test split:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_sweep_full_2gpu.yaml
```

Inspect the generated files under:

```text
../rag_assets/outputs/experiments/<study_name>/
├── launch_manifest.yaml
├── launch_tmux.sh
├── run_gpu2_3.sh
└── <run_name>/
    ├── run_config.yaml
    ├── train.log
    ├── eval.log
    └── metrics/test_eval.json
```

`metrics/test_eval.json` is the completion marker when post-training evaluation
is enabled. If a run already has that file, the generated GPU queue skips it
unless you pass `--overwrite`.
If the final training marker already exists but `metrics/test_eval.json` is
missing, the launcher runs evaluation only instead of retraining into existing
checkpoints.

Launch sequentially in the current process:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_sweep_full_2gpu.yaml \
  --launch
```

Or launch the generated per-GPU script:

```bash
bash ../rag_assets/outputs/experiments/evoco_popqa_sweep_full_2gpu/run_gpu2_3.sh
```

Start all generated GPU queues in tmux:

```bash
bash ../rag_assets/outputs/experiments/evoco_popqa_sweep_full_2gpu/launch_tmux.sh
```

Recommended bash entrypoint for all official experiment studies:

```bash
bash run.sh preflight
bash run.sh test --gpus 2,3
bash run.sh train --gpus 2,3
```

On an 8-GPU machine, keep 2 GPUs as one experiment worker and distribute runs
across four workers:

```bash
bash run.sh train --gpu-pairs '0,1;2,3;4,5;6,7'
```

This creates one worker tmux session per pair, e.g.
`evoco_all_experiments_g0_1`, `evoco_all_experiments_g2_3`,
`evoco_all_experiments_g4_5`, and `evoco_all_experiments_g6_7`.

Lower-level launcher commands, if you want to bypass `run.sh`:

```bash
bash scripts/launch_all_experiments.sh --dry-run
bash scripts/launch_all_experiments.sh
```

This verifies the dataset pack, regenerates `configs/local/*_{fast,full}.yaml`,
materializes every official full-data study, and starts tmux queues. With one
GPU pair, the master queue runs the generated per-study GPU scripts
sequentially. With `--gpu-pairs`, each pair gets its own sequential worker queue
and different pairs run in parallel.
Fast specs are kept for debugging and are not included in the default official
queue.

Official launcher specs:

| Spec | Runs | Purpose |
|---|---:|---|
| `popqa_sweep_full_2gpu.yaml` | 5 | full PopQA sweep: top-k, reward, audit switches |
| `popqa_hparam_full_2gpu.yaml` | 10 | full PopQA hyperparameter search for top-k, audit count, confidence thresholds, context length |
| `multidataset_full_2gpu.yaml` | 5 | full generalization check across converted datasets |
| `popqa_full_sweep_2gpu.yaml` | 4 | selected full PopQA settings for final cost/accuracy comparison |
| `popqa_ablation_full_2gpu.yaml` | 8 | full PopQA mechanism ablations for the main paper table |

Single-study bash entrypoint:

```bash
bash scripts/launch_tmux.sh
```

Use a different spec:

```bash
bash scripts/launch_tmux.sh configs/experiments/multidataset_full_2gpu.yaml
```

Generate configs and scripts without starting tmux:

```bash
bash scripts/launch_tmux.sh --dry-run
```

Equivalent Python entrypoint:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_sweep_full_2gpu.yaml \
  --launch-tmux
```

Set `eval_after_train: false` at the spec or experiment level only when you want
training-only runs. In that case the completion marker becomes
`metrics/round_000.json`.

The launcher automatically sets unique per-run paths:

```yaml
project.output_dir: <output_root>/<study_name>/<run_name>
models.small_lora_dir: <checkpoint_root>/<study_name>/<run_name>/small
models.large_lora_dir: <checkpoint_root>/<study_name>/<run_name>/large
```

Use dotted-key overrides in a launcher spec:

```yaml
experiments:
  - name: precision_top8
    gpu: "2,3"
    overrides:
      contract.top_k: 8
      contract.max_selected_docs: 8
      runtime.num_audit_candidates: 5
```

## Standalone Configs

| Config | Purpose | Start command |
|---|---|---|
| `two_h20_smoke.yaml` | Two-H20 smoke run on 128 PopQA samples | `CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/experiments/two_h20_smoke.yaml` |
| `two_h20_main_policy.yaml` | Main policy-head experiment for two H20 GPUs | `CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/experiments/two_h20_main_policy.yaml` |
| `hparam_cost_top3.yaml` | Lower-cost top-3 setting | `CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/experiments/hparam_cost_top3.yaml` |
| `hparam_precision_top8.yaml` | Higher-recall top-8 setting | `CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/experiments/hparam_precision_top8.yaml` |
| `hparam_audit_self_consistency.yaml` | More audit candidates for self-consistency | `CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py --config configs/experiments/hparam_audit_self_consistency.yaml` |

The standalone full PopQA settings are also represented by launcher specs:
`popqa_full_sweep_2gpu.yaml` for selected hyperparameter settings and
`popqa_ablation_full_2gpu.yaml` for mechanism ablations. Both are included in
`scripts/launch_all_experiments.sh`.

When creating a new experiment, copy one YAML and change all three roots:

```yaml
project:
  output_dir: ../rag_assets/outputs/experiments/<new_name>
models:
  small_lora_dir: ../rag_assets/checkpoints/experiments/<new_name>/small
  large_lora_dir: ../rag_assets/checkpoints/experiments/<new_name>/large
```

Fresh training refuses to overwrite existing `round_*` adapters. Use `--resume`
only when you intend to continue the same experiment.
