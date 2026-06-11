# Testing & Code Checks (测试文档)

This document covers **verifying the code is wired correctly**. Testing has two
tiers:

- **CPU tier (no GPU, no weights):** unit tests, compile checks, and the
  no-model data pipeline — fast, run them anywhere (§1–§5).
- **GPU tier (torch + weights):** the full test suite (the 4 torch tests also
  run) plus a small-sample real-model smoke run — run these on the GPU server
  (§6).

For full training/eval, see [EXPERIMENTS.md](EXPERIMENTS.md). For
datasets/weights, see [DATASETS.md](DATASETS.md). For the full documentation
map, see [README.md](README.md).

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

Expected on a **CPU-only** machine:

```text
80 passed, 4 skipped
```

The 4 skipped tests need torch (`tests/test_large_batching.py`,
`tests/test_small_policy_heads.py`). On a **GPU server with torch installed** they
run too, so the full suite is **84 passed** — see §6. To see skip reasons:

```bash
python -m pytest -q -rs
```

### What the suite covers

| Test file | Covers | Needs torch |
|---|---|:--:|
| `tests/test_schemas.py` | schema construction + enum validation | |
| `tests/test_verifier.py` | answer/citation/support rule verification | |
| `tests/test_rewards.py` | four-quadrant responsibility attribution | |
| `tests/test_metrics.py` | answer/retrieval/evidence/cost/calibration metrics | |
| `tests/test_replay_buffer.py` | JSONL write/read/filter/sample | |
| `tests/test_auditor.py` | robust JSON extraction + enum downgrade + fallback | |
| `tests/test_integration.py` | no-model round: contract→verify→reward→replay, resume/streaming | |
| `tests/test_ablation.py` | ablation switches (action policy / decomposed reward) | |
| `tests/test_evaluator_generalization.py` | real-generalization eval path (incl. stub models) | |
| `tests/test_plot_trends.py` | trend aggregation + plotting | |
| `tests/test_large_batching.py` | large-model batched audit generation | ✓ |
| `tests/test_small_policy_heads.py` | small-model evidence/action/confidence heads | ✓ |

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
python scripts/verify_dataset_pack.py --data-root ../rag_assets/rag_data/evoco_dataset_pack
python scripts/make_dataset_config.py --data-root ../rag_assets/rag_data/evoco_dataset_pack --all --output-root configs/local
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
  --data-root ../rag_assets/rag_data/evoco_dataset_pack \
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

## 5. Testing on GPU

On the GPU server (torch + weights installed, see
[EXPERIMENTS.md §1](EXPERIMENTS.md#1-install-the-gpu-environment)), do two extra
things the CPU tier cannot.

**a) Run the full unit suite** — the 4 torch tests now activate:

```bash
python -m pytest -q          # expected: 84 passed
```

These cover large-model batched audit generation and the small-model
evidence/action/confidence heads, which only exist with torch.

**b) Real-model runs on small samples are also tests.** The 16- and 512-sample
runs are *verification*, not the final experiment — they confirm the system
behaves correctly and the numbers look sane before you pay for a full run. Only
the **full** run (in [EXPERIMENTS.md](EXPERIMENTS.md)) is the real experiment.
Treat these as a two-rung GPU test ladder:

| Test rung | Config | What it verifies |
|---|---|---|
| **16-sample smoke** | `*_debug.yaml` | models load, audits parse to valid JSON, one full round trains both LoRAs and writes a real-generalization eval — pure wiring/OOM check |
| **512-sample signal** | `*_fast.yaml` | metrics are sane (accuracy / evidence-support / `wrong_retriever_reward_rate`), losses drop, policy heads train, `ask_auditor` ratio is reasonable, cost is acceptable |

Run them with the same training entrypoint (see
[EXPERIMENTS.md run ladder](EXPERIMENTS.md#2-run-ladder-debug--fast--full)):

```bash
# rung 1 — smoke (seconds–minutes)
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_debug.yaml

# rung 2 — signal (minutes)
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_fast.yaml
```

A clean 16-sample smoke should: load both models, stream `round 0: experience
k/16`, write `replay/round_000.jsonl`, train both LoRAs, and produce
`metrics/test_eval_round_000.json` (real generalization). The 512-sample run then
tells you whether the *results* — not just the plumbing — are reasonable. Inspect
both with `scripts/inspect_replay.py` and `scripts/plot_trends.py`. Keep
`data.eval_size` small (or rely on `debug_size`) so per-round eval stays fast
during these tests. Only after rung 2 looks right do you launch the full run.

---

## 6. Pre-Push Checklist

Run before every push:

```bash
python -m pytest -q                       # 80 passed, 4 skipped
python -m py_compile evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py scripts/*.py
git diff --check                          # no whitespace/conflict markers
git status --short --branch               # confirm only intended files
```

Make sure no asset files (weights, checkpoints, replay JSONL, tarballs,
`configs/local/**`) are staged — they belong under `../rag_assets/`
(see [DATASETS.md §6](DATASETS.md#6-repository-hygiene)).
