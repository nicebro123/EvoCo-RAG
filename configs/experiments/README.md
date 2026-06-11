# Experiment Configs

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
```

Dry-run a study. This writes per-run configs, a manifest, and per-GPU shell
scripts without starting training. By default, every generated command trains
first and then evaluates the same generated config on the test split:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml
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
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml \
  --launch
```

Or launch the generated per-GPU script:

```bash
bash ../rag_assets/outputs/experiments/evoco_popqa_fast_sweep_2gpu/run_gpu2_3.sh
```

Start all generated GPU queues in tmux:

```bash
bash ../rag_assets/outputs/experiments/evoco_popqa_fast_sweep_2gpu/launch_tmux.sh
```

Recommended bash entrypoint:

```bash
bash scripts/launch_tmux.sh
```

Use a different spec:

```bash
bash scripts/launch_tmux.sh configs/experiments/multidataset_fast_2gpu.yaml
```

Generate configs and scripts without starting tmux:

```bash
bash scripts/launch_tmux.sh --dry-run
```

Equivalent Python entrypoint:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml \
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
