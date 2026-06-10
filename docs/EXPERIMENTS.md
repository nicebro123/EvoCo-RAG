# Running Experiments (实验执行文档)

This document covers **installing the GPU environment and running training,
evaluation, ablations, and trend plots**.

Prerequisite: datasets and base weights already placed per
[DATASETS.md](DATASETS.md). For tests/code checks, see [TESTING.md](TESTING.md).

All commands assume the working directory is the repo root and assets are under
`../rag_assets/`.

---

## 1. Install the GPU Environment

Pinned reproduction environment:

```text
Python 3.10 or 3.11
CUDA 12.1 PyTorch wheels
torch==2.5.1+cu121   transformers==4.46.3   peft==0.14.0   trl==0.14.0
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
```

Verify the GPUs are visible:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available(), "gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

If your server uses a different CUDA wheel, keep all non-torch versions and only
replace the torch index line in `requirements-gpu.txt` with the matching command
from <https://pytorch.org/get-started/locally/>. For 4-bit loading also install
`bitsandbytes==0.45.0` and set `models.use_4bit: true` (default is bf16).

### GPU selection

`CUDA_VISIBLE_DEVICES=2,3` exposes physical GPUs 2 and 3 as logical `cuda:0/1`.
The large generator uses `device_map="auto"` and can shard across visible GPUs;
the small reranker uses `cuda:0` (the first visible GPU). Change the list to use
a different subset, e.g. `CUDA_VISIBLE_DEVICES=0,1`.

---

## 2. Run Ladder: debug → fast → full

**Always start with the smallest sample size and climb.** For any new dataset,
config change, or code change, run the 16-sample debug config first — it
exercises the full pipeline (contract → audit → verify → reward → train → eval)
end to end in minutes, so you catch wiring/OOM/JSON issues before paying for a
big run. Only move up after the smaller rung finishes cleanly.

| Rung | Samples | `data.debug_size` | Use it to… |
|---|---:|---|---|
| debug | 16 | `16` | smoke-test wiring / a code change |
| fast | 512 | `512` | get a quick signal / tune knobs / run ablations |
| full | all | `null` | final numbers |

The debug config caps **both** training and per-round generalization eval to 16
samples (`data.eval_size` falls back to `data.debug_size`), so the smoke run is
fast on both ends. Generate the configs first
(see [DATASETS.md §4](DATASETS.md#4-generate-run-configs-from-the-pack)).

**Step 1 — 16-sample debug smoke test:**

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_debug.yaml
```

**Step 2 — 512-sample fast run:**

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_fast.yaml
```

**Step 3 — full run (only after the fast run finishes cleanly):**

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_full.yaml
```

Switch dataset by changing only the config name (e.g.
`hotpotqa_distractor_fast.yaml`, `nq_reader_fast.yaml`, `asqa_dpr_fast.yaml`,
`popqa_retrieval_fast.yaml`).

### Fresh runs refuse to overwrite

Training will not overwrite existing `round_*` adapters. For a new experiment,
point the YAML at fresh paths:

```yaml
project:
  output_dir: ../rag_assets/outputs/datasets/popqa_standard_v2
models:
  small_lora_dir: ../rag_assets/checkpoints/datasets/popqa_standard_v2/small
  large_lora_dir: ../rag_assets/checkpoints/datasets/popqa_standard_v2/large
```

### Resume

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_evoco.py \
  --config configs/local/popqa_standard_full.yaml --resume
```

`--resume` continues from the latest completed checkpoint round. If a process is
killed mid-experience-generation, rerun the **same** command without deleting the
output directory: valid partial `replay/round_xxx.jsonl` rows are reused,
corrupted trailing rows are skipped, and only missing sample IDs are regenerated.

---

## 3. Per-Round Real Generalization

Each round automatically evaluates **real generalization** on the held-out test
set with `show_gold=False` (gold answers never enter the prompt). This is the
paper-facing metric and is stored as `stats["eval"]`
(`eval_source="test_generalization"`). A training-set teacher-audit diagnostic is
kept separately as `stats["train_metrics"]` for reference only.

Per-round metric files:

```text
../rag_assets/outputs*/metrics/round_000.json              # full round stats (eval = real generalization)
../rag_assets/outputs*/metrics/test_eval_round_000.json    # real generalization (paper metric)
../rag_assets/outputs*/metrics/train_eval_round_000.json   # training-set diagnostic (reference)
```

Running full test each round is expensive (every sample needs a large-model
audit). Cap the per-round test subset with `data.eval_size`; final evaluation via
`scripts/eval_evoco.py` always uses the full test set.

```yaml
data:
  eval_size: 256   # null = full test set each round
```

---

## 4. Evaluate

Evaluate the latest adapters implied by a config (full test set,
`show_gold=False`):

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/local/popqa_standard_full.yaml
```

Evaluate explicit adapter rounds:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/eval_evoco.py \
  --config configs/local/popqa_standard_full.yaml \
  --small_lora ../rag_assets/checkpoints/datasets/popqa_standard_full/small/round_002 \
  --large_lora ../rag_assets/checkpoints/datasets/popqa_standard_full/large/round_002
```

Writes `../rag_assets/outputs*/metrics/test_eval.json`. Gold answers are used
only for offline metrics, never inserted into the generation prompt.

---

## 5. Trend Summary & Plots

After a multi-round run, aggregate every round's real-generalization metrics:

```bash
python scripts/plot_trends.py --config configs/local/popqa_standard_full.yaml
```

Reads `metrics/round_*.json` and writes:

```text
metrics/trends.json
metrics/trends.csv
metrics/trend_metrics.png    # accuracy / evidence-support / unsupported-answer / ask_auditor ratio
metrics/trend_training.png   # small/large losses and policy-head accuracy
```

---

## 6. Ablations

List built-in ablation names:

```bash
python scripts/run_ablations.py --list
```

Run all ablations with real models:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_ablations.py \
  --config configs/local/popqa_standard_fast.yaml
```

Run selected ablations:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_ablations.py \
  --config configs/local/popqa_standard_fast.yaml \
  --only evoco_full evoco_no_audit evoco_answer_only_reward
```

Each ablation writes to `<output_dir>/ablations/<name>/`; the comparison summary
is `<output_dir>/ablations/summary.json`. (A CPU wiring check with `--no_models`
is documented in [TESTING.md](TESTING.md).)

---

## 7. Batch Studies (launcher + tmux)

For repeated sweeps, prefer the launcher over copying YAML files. It expands a
compact study spec into per-run configs with isolated output/checkpoint paths,
and each run trains then auto-evaluates.

```bash
# 1) dry run: write per-run configs + commands, inspect them
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml

# 2) launch sequentially
python scripts/launch_experiments.py \
  --spec configs/experiments/popqa_fast_sweep_2gpu.yaml --launch

# or start the generated tmux queues for long sweeps
bash scripts/launch_tmux.sh
bash scripts/launch_tmux.sh configs/experiments/multidataset_fast_2gpu.yaml
bash scripts/launch_tmux.sh --dry-run
```

Each generated run directory contains `run_config.yaml`, `train.log`,
`eval.log`, and `metrics/test_eval.json` (the default completion marker). If
training finished but evaluation was interrupted, the launcher runs evaluation
only instead of retraining into existing checkpoints. Spec templates and their
format live under `configs/experiments/` (see its local `README.md`).

---

## 8. Runtime Knobs

| Field | Effect | 2×H20 start |
|---|---|---|
| `contract.top_k` | candidate docs into the evidence contract | `5` |
| `contract.max_selected_docs` | docs kept as selected evidence | `5` |
| `runtime.candidate_doc_char_limit` | chars per candidate doc in audit prompt | `1200` |
| `runtime.num_audit_candidates` | large-model audit candidates (self-consistency) | `3` |
| `runtime.audit_batch_size` | batch size for audit generation | `2`, try `4` when stable |
| `training.batch_size` | small reranker batch | `4`–`8` |
| `training.large_batch_size` | large LoRA SFT batch | `2` |
| `small_policy.use_policy_heads` | enable evidence/action/confidence heads | `true` for policy runs |
| `data.eval_size` | per-round test subset size | `256` (null = full) |

On CUDA OOM, reduce in this order:

```text
runtime.audit_batch_size → training.large_batch_size → runtime.max_prompt_length
→ runtime.candidate_doc_char_limit → contract.top_k / max_selected_docs
```

---

## 9. Outputs & Monitoring

Each run writes:

```text
../rag_assets/outputs*/used_config.yaml
../rag_assets/outputs*/weights_manifest.json
../rag_assets/outputs*/replay/round_000.jsonl        # + all.jsonl
../rag_assets/outputs*/contracts/round_000.jsonl
../rag_assets/outputs*/audits/round_000.jsonl
../rag_assets/outputs*/metrics/round_000.json        # + test_eval_round_*, train_eval_round_*, trends.*
../rag_assets/checkpoints/*/small/round_000/
../rag_assets/checkpoints/*/large/round_000/
```

Watch progress during a long round:

```bash
wc -l ../rag_assets/outputs/datasets/popqa_standard_full/replay/round_000.jsonl
tail -n 20 ../rag_assets/outputs/datasets/popqa_standard_full/metrics/round_000.json
```

Inspect a replay file (failure-type distribution, credit assignment, metrics):

```bash
python scripts/inspect_replay.py \
  --replay ../rag_assets/outputs/datasets/popqa_standard_full/replay/round_000.jsonl
```

Each round records stage timing in `metrics/round_xxx.json`:

```json
"timing": {
  "experience_generation_seconds": 0.0,
  "small_training_seconds": 0.0,
  "large_training_seconds": 0.0,
  "evaluation_seconds": 0.0,
  "total_round_seconds": 0.0
}
```
