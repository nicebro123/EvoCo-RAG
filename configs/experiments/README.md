# Experiment Configs

These YAML files are full standalone configs. The loader does not implement
inheritance, so each file explicitly sets data paths, model paths, output paths,
training hyperparameters, runtime knobs, and ablation flags.

All data, weights, checkpoints, and generated outputs stay outside the Git repo
under `../rag_assets/`.

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
