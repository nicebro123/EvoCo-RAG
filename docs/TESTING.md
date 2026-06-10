# Testing & Code Checks (测试文档)

This document covers **verifying the code is wired correctly without training**.
None of these checks need a GPU or model weights; they confirm the package,
schemas, reward attribution, and the no-model data pipeline.

For training/eval, see [EXPERIMENTS.md](EXPERIMENTS.md). For datasets/weights,
see [DATASETS.md](DATASETS.md).

---

## 1. CPU Environment

The core `evoco_rag` layer (schemas, data, contract, verifier, rewards,
replay_buffer, metrics) is dependency-light and imports without torch. For code
checks only:

```bash
pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` is **not** sufficient for training or evaluation with real
models — use `requirements-gpu.txt` for that (see [EXPERIMENTS.md](EXPERIMENTS.md)).

---

## 2. Unit Tests

```bash
python -m pytest -q
```

Expected on a CPU-only machine:

```text
74 passed, 4 skipped
```

The 4 skipped tests are the torch-dependent fake-model tests; on a GPU server
with torch installed they run instead of skipping.

### What the suite covers

| Test file | Covers |
|---|---|
| `tests/test_schemas.py` | schema construction + enum validation |
| `tests/test_verifier.py` | answer/citation/support rule verification |
| `tests/test_rewards.py` | four-quadrant responsibility attribution |
| `tests/test_metrics.py` | answer/retrieval/evidence/cost/calibration metrics |
| `tests/test_replay_buffer.py` | JSONL write/read/filter/sample |
| `tests/test_auditor.py` | robust JSON extraction + enum downgrade + fallback |
| `tests/test_integration.py` | no-model round: contract→verify→reward→replay, resume/streaming |
| `tests/test_ablation.py` | ablation switches (action policy / decomposed reward) |
| `tests/test_evaluator_generalization.py` | real-generalization eval path (incl. stub models) |
| `tests/test_plot_trends.py` | trend aggregation + plotting |

---

## 3. Compile Check

Confirm every module (including the torch-importing ones) parses:

```bash
python -m py_compile \
  evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py \
  scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py tests/*.py
```

---

## 4. No-Model Pipeline Checks

These exercise the full data/logic pipeline on CPU, with no model weights.

Dataset pack validation (needs the pack extracted — see [DATASETS.md](DATASETS.md)):

```bash
python scripts/verify_dataset_pack.py --data-root ../rag_assets/evoco_dataset_pack
python scripts/make_dataset_config.py --data-root ../rag_assets/evoco_dataset_pack --all --output-root configs/local
```

Experiment launcher dry run (writes configs/commands, no GPU scripts):

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml --no-gpu-scripts
```

Seed replay + ablation wiring (pure heuristic, no models). **Always use a small
debug config (16 samples) for these quick checks** so they finish in seconds —
never the 512-sample `_fast` or the full config:

```bash
# Generate a 16-sample debug config once
python scripts/make_dataset_config.py \
  --data-root ../rag_assets/evoco_dataset_pack \
  --dataset-id popqa_standard --debug-size 16 \
  --name evoco_popqa_standard_debug \
  --output configs/local/popqa_standard_debug.yaml \
  --output-dir ../rag_assets/outputs_debug/popqa_standard \
  --checkpoint-root ../rag_assets/checkpoints/debug/popqa_standard

# Run the no-model checks against the 16-sample debug config
python scripts/build_seed_replay.py --config configs/local/popqa_standard_debug.yaml
python scripts/run_ablations.py    --config configs/local/popqa_standard_debug.yaml --no_models
python scripts/inspect_replay.py \
  --replay ../rag_assets/outputs_debug/popqa_standard/replay/round_000.jsonl
```

> The committed `configs/debug.yaml` already sets `data.debug_size: 16`; any
> generated `*_debug.yaml` does the same. Quick checks should never iterate the
> full dataset.

---

## 5. Pre-Push Checklist

Run before every push:

```bash
python -m pytest -q                       # 74 passed, 4 skipped
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py
git diff --check                          # no whitespace/conflict markers
git status --short --branch               # confirm only intended files
```

Make sure no asset files (weights, checkpoints, replay JSONL, tarballs,
`configs/local/**`) are staged — they belong under `../rag_assets/`
(see [DATASETS.md §6](DATASETS.md#6-repository-hygiene)).
