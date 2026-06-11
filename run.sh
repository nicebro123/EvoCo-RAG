#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${EVOCO_PYTHON:-python}"
DATA_ROOT="${EVOCO_DATA_ROOT:-../rag_assets/rag_data/evoco_dataset_pack}"
GPUS="${EVOCO_GPUS:-${CUDA_VISIBLE_DEVICES:-2,3}}"
RUN_ID="${EVOCO_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VERIFY_MAX_ROWS="${EVOCO_VERIFY_MAX_ROWS:-5}"

MODE="${1:-help}"
if [[ "$MODE" != -* ]]; then
  shift || true
else
  MODE="help"
fi

SKIP_VERIFY=0
GENERATE_CONFIGS=1
SKIP_CODE_CHECKS=0
SKIP_PYTEST=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash run.sh preflight [options]
  bash run.sh test [options]
  bash run.sh train [options]
  bash run.sh all [options]

Modes:
  preflight  Verify data, generate local configs, run CPU/code checks, and
             dry-run the official full-data experiment queue. No GPU training.
  test       Run preflight, then execute one unique 16-sample real-model
             PopQAStandard debug run and inspect its replay output.
  train      Start the official full-data experiment queue in tmux.
  all        Run test first, then start official full-data training.

Options:
  --data-root PATH       Dataset pack path. Default: ../rag_assets/rag_data/evoco_dataset_pack
  --gpus LIST            GPUs for the smoke test and full training. Default: 2,3
  --run-id ID            Stable id for the smoke-test output path. Default: timestamp
  --skip-verify          Skip dataset-pack verification.
  --no-generate-configs  Reuse existing configs/local/*.yaml instead of regenerating.
  --skip-code-checks     Skip bash syntax, py_compile, and pytest checks.
  --skip-pytest          Run lightweight code checks but skip pytest.
  --overwrite            Pass --overwrite to official training launcher.
  -h, --help             Show this help.

Environment:
  EVOCO_PYTHON           Python executable. Default: python
  EVOCO_DATA_ROOT        Dataset pack path.
  EVOCO_GPUS             GPU list, e.g. 2,3.
  EVOCO_RUN_ID           Smoke-test run id.
  EVOCO_VERIFY_MAX_ROWS  Rows per split checked by verification. Default: 5.

Recommended:
  bash run.sh test
  bash run.sh train
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      if (($# < 2)); then
        echo "missing value for --data-root" >&2
        exit 2
      fi
      DATA_ROOT="$2"
      shift 2
      ;;
    --gpus)
      if (($# < 2)); then
        echo "missing value for --gpus" >&2
        exit 2
      fi
      GPUS="$2"
      shift 2
      ;;
    --run-id)
      if (($# < 2)); then
        echo "missing value for --run-id" >&2
        exit 2
      fi
      RUN_ID="$2"
      shift 2
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --no-generate-configs)
      GENERATE_CONFIGS=0
      shift
      ;;
    --skip-code-checks)
      SKIP_CODE_CHECKS=1
      shift
      ;;
    --skip-pytest)
      SKIP_PYTEST=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

log() {
  printf '\n==> %s\n' "$*"
}

verify_data() {
  if ((SKIP_VERIFY)); then
    log "skip dataset verification"
    return
  fi
  log "verify dataset pack: $DATA_ROOT"
  "$PYTHON_BIN" scripts/verify_dataset_pack.py \
    --data-root "$DATA_ROOT" \
    --max-rows "$VERIFY_MAX_ROWS"
}

generate_configs() {
  if ((! GENERATE_CONFIGS)); then
    log "reuse existing configs/local"
    return
  fi
  log "generate fast configs into configs/local"
  "$PYTHON_BIN" scripts/make_dataset_config.py \
    --data-root "$DATA_ROOT" \
    --all \
    --output-root configs/local
  log "generate full configs into configs/local"
  "$PYTHON_BIN" scripts/make_dataset_config.py \
    --data-root "$DATA_ROOT" \
    --all \
    --full \
    --output-root configs/local
}

run_code_checks() {
  if ((SKIP_CODE_CHECKS)); then
    log "skip code checks"
    return
  fi
  log "bash syntax check"
  bash -n run.sh scripts/launch_all_experiments.sh scripts/launch_tmux.sh

  log "python compile check"
  "$PYTHON_BIN" -m py_compile \
    evoco_rag/*.py evoco_rag/trainers/*.py evoco_rag/evaluation/*.py \
    scripts/*.py run_train.py run_test.py utils.py llm_local_prompt.py tests/*.py

  if ((SKIP_PYTEST)); then
    log "skip pytest"
  else
    log "pytest"
    "$PYTHON_BIN" -m pytest -q
  fi
}

official_dry_run() {
  log "dry-run official full-data experiment queue"
  EVOCO_GPUS="$GPUS" bash scripts/launch_all_experiments.sh \
    --dry-run \
    --skip-verify \
    --no-generate-configs
}

preflight() {
  verify_data
  generate_configs
  run_code_checks
  official_dry_run
}

smoke_test() {
  local safe_run_id
  safe_run_id="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
  local smoke_config="configs/local/run_sh_popqa_standard_debug_${safe_run_id}.yaml"
  local smoke_output="../rag_assets/outputs_debug/run_sh/${safe_run_id}"
  local smoke_checkpoint="../rag_assets/checkpoints/debug/run_sh/${safe_run_id}"
  local replay_path="${smoke_output}/replay/round_000.jsonl"
  local metric_path="${smoke_output}/metrics/round_000.json"

  log "create isolated 16-sample smoke config: $smoke_config"
  "$PYTHON_BIN" scripts/make_dataset_config.py \
    --data-root "$DATA_ROOT" \
    --dataset-id popqa_standard \
    --debug-size 16 \
    --name "evoco_run_sh_${safe_run_id}" \
    --output "$smoke_config" \
    --output-dir "$smoke_output" \
    --checkpoint-root "$smoke_checkpoint"

  log "run 16-sample real-model smoke test on GPUs: $GPUS"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" scripts/train_evoco.py \
    --config "$smoke_config"

  log "verify smoke outputs"
  test -f "$replay_path"
  test -f "$metric_path"

  log "inspect smoke replay"
  "$PYTHON_BIN" scripts/inspect_replay.py --replay "$replay_path"

  log "smoke test output: $smoke_output"
}

train_official() {
  local extra_args=()
  if ((SKIP_VERIFY)); then
    extra_args+=("--skip-verify")
  fi
  if ((! GENERATE_CONFIGS)); then
    extra_args+=("--no-generate-configs")
  fi
  if ((OVERWRITE)); then
    extra_args+=("--overwrite")
  fi
  log "start official full-data training in tmux on GPUs: $GPUS"
  EVOCO_GPUS="$GPUS" CUDA_VISIBLE_DEVICES="$GPUS" bash scripts/launch_all_experiments.sh \
    --data-root "$DATA_ROOT" \
    "${extra_args[@]}"
}

case "$MODE" in
  preflight)
    preflight
    ;;
  test)
    preflight
    smoke_test
    ;;
  train)
    train_official
    ;;
  all)
    preflight
    smoke_test
    train_official
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
