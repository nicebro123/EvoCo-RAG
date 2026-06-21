#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${EVOCO_PYTHON:-python}"
SPEC="configs/experiments/popqa_llama8b_full_sweep_2gpu.yaml"
MODE="--launch-tmux"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_tmux.sh [SPEC] [options]
  bash scripts/launch_tmux.sh --spec configs/experiments/popqa_llama8b_full_sweep_2gpu.yaml

Defaults:
  SPEC: configs/experiments/popqa_llama8b_full_sweep_2gpu.yaml
  mode: generate configs and start tmux queues

Options:
  --spec PATH       Experiment spec YAML.
  --dry-run         Generate configs and scripts only; do not start tmux.
  --launch-tmux     Generate configs and start tmux queues. Default.
  --tmux            Alias for --launch-tmux.
  --overwrite       Forward to scripts/launch_experiments.py.
  -h, --help        Show this help.

Environment:
  EVOCO_PYTHON      Python executable to use. Default: python.
  EVOCO_GPUS        Override the spec GPU list, e.g. 0,1. Default: spec value.
EOF
}

while (($#)); do
  case "$1" in
    --spec)
      if (($# < 2)); then
        echo "missing value for --spec" >&2
        exit 2
      fi
      SPEC="$2"
      shift 2
      ;;
    --dry-run)
      MODE=""
      shift
      ;;
    --launch-tmux|--tmux)
      MODE="--launch-tmux"
      shift
      ;;
    --overwrite)
      EXTRA_ARGS+=("--overwrite")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SPEC="$1"
      shift
      ;;
  esac
done

cd "$ROOT_DIR"
echo "repo: $ROOT_DIR"
echo "spec: $SPEC"
if [[ -n "$MODE" ]]; then
  echo "mode: tmux background queues"
  exec "$PYTHON_BIN" scripts/launch_experiments.py --spec "$SPEC" "$MODE" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
  echo "mode: dry run"
  exec "$PYTHON_BIN" scripts/launch_experiments.py --spec "$SPEC" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
