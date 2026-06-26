#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${EVOCO_PYTHON:-python3}"
DATA_ROOT="${EVOCO_DATA_ROOT:-../rag_assets/rag_data/evoco_dataset_pack}"
VERIFY_MAX_ROWS="${EVOCO_VERIFY_MAX_ROWS:-5}"
GPU_PAIRS="${EVOCO_GPU_PAIRS:-}"
VERIFY_DATA=1
GENERATE_CONFIGS=1
LAUNCH_TMUX=1
EXTRA_ARGS=()
SPECS=()
DEFAULT_SPECS=(
  "configs/experiments/popqa_llama8b_full_sweep_2gpu.yaml"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_all_experiments.sh [options]

Default behavior:
  1. verify the dataset pack;
  2. regenerate configs/local fast + full configs;
  3. materialize all experiment specs;
  4. start one tmux session that runs all generated GPU queues sequentially.

The default official spec uses full-data PopQA with Meta-Llama-3-8B-Instruct.
Legacy Mistral-Nemo and multi-dataset specs remain available and can be passed
explicitly with --spec.

Options:
  --data-root PATH       Dataset pack path. Default: ../rag_assets/rag_data/evoco_dataset_pack
  --spec PATH            Add one experiment spec. Repeatable. Defaults to all official specs.
  --gpu-pairs LIST       Semicolon-separated GPU workers, e.g. '0,1;2,3;4,5;6,7'.
  --dry-run              Generate configs/scripts only; do not start tmux.
  --launch-tmux, --tmux  Start the master tmux queue. Default.
  --overwrite            Forward to scripts/launch_experiments.py.
  --skip-verify          Do not run scripts/verify_dataset_pack.py.
  --no-generate-configs  Do not regenerate configs/local.
  -h, --help             Show this help.

Environment:
  EVOCO_PYTHON           Python executable to use. Default: python.
  EVOCO_DATA_ROOT        Dataset pack path, overridden by --data-root.
  EVOCO_GPUS             Override every spec GPU list, e.g. 0,1. Default: spec value.
  EVOCO_GPU_PAIRS        Round-robin runs across GPU workers, e.g. 0,1;2,3;4,5;6,7.
                         Takes precedence over EVOCO_GPUS.
  EVOCO_VERIFY_MAX_ROWS  Rows per split checked by verification. Default: 5.
  EVOCO_MASTER_OUTPUT_DIR  Master queue directory. Default: ../rag_assets/outputs/experiments/evoco_all_experiments.
  EVOCO_TMUX_SESSION     Master tmux session name. Default: evoco_all_experiments.
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
    --spec)
      if (($# < 2)); then
        echo "missing value for --spec" >&2
        exit 2
      fi
      SPECS+=("$2")
      shift 2
      ;;
    --gpu-pairs)
      if (($# < 2)); then
        echo "missing value for --gpu-pairs" >&2
        exit 2
      fi
      GPU_PAIRS="$2"
      export EVOCO_GPU_PAIRS="$GPU_PAIRS"
      shift 2
      ;;
    --dry-run)
      LAUNCH_TMUX=0
      shift
      ;;
    --launch-tmux|--tmux)
      LAUNCH_TMUX=1
      shift
      ;;
    --overwrite)
      EXTRA_ARGS+=("--overwrite")
      shift
      ;;
    --skip-verify)
      VERIFY_DATA=0
      shift
      ;;
    --no-generate-configs)
      GENERATE_CONFIGS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        SPECS+=("$1")
        shift
      done
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SPECS+=("$1")
      shift
      ;;
  esac
done

if ((${#SPECS[@]} == 0)); then
  SPECS=("${DEFAULT_SPECS[@]}")
fi

cd "$ROOT_DIR"

echo "repo: $ROOT_DIR"
echo "data_root: $DATA_ROOT"
if ((LAUNCH_TMUX)); then
  echo "mode: master tmux queue"
else
  echo "mode: dry run"
fi
echo "specs:"
for spec in "${SPECS[@]}"; do
  echo "  - $spec"
done
if [[ -n "$GPU_PAIRS" ]]; then
  echo "gpu_pairs: $GPU_PAIRS"
fi

if ((VERIFY_DATA)); then
  "$PYTHON_BIN" scripts/verify_dataset_pack.py \
    --data-root "$DATA_ROOT" \
    --max-rows "$VERIFY_MAX_ROWS"
fi

if ((GENERATE_CONFIGS)); then
  "$PYTHON_BIN" scripts/make_dataset_config.py \
    --data-root "$DATA_ROOT" \
    --all \
    --output-root configs/local
  "$PYTHON_BIN" scripts/make_dataset_config.py \
    --data-root "$DATA_ROOT" \
    --all \
    --full \
    --output-root configs/local
fi

study_dir_for_spec() {
  "$PYTHON_BIN" - "$1" <<'PY'
from pathlib import Path
import sys
import yaml


def slug(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", "+"}:
            chars.append(char)
        elif char.isspace() or char in {",", "/"}:
            chars.append("_")
    return "".join(chars).strip("._-") or "run"


def resolve_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


spec_path = Path(sys.argv[1]).resolve()
spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
root = resolve_path(str(spec.get("output_root", "../rag_assets/outputs/experiments")))
study_name = spec.get("study_name") or spec_path.stem
print(root / slug(str(study_name)))
PY
}

STUDY_DIRS=()
for spec in "${SPECS[@]}"; do
  "$PYTHON_BIN" scripts/launch_experiments.py \
    --spec "$spec" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  STUDY_DIRS+=("$(study_dir_for_spec "$spec")")
done

if ((! LAUNCH_TMUX)); then
  echo "Dry run complete for ${#SPECS[@]} spec(s)."
  exit 0
fi

MASTER_DIR_RAW="${EVOCO_MASTER_OUTPUT_DIR:-../rag_assets/outputs/experiments/evoco_all_experiments}"
if [[ "$MASTER_DIR_RAW" = /* ]]; then
  MASTER_DIR="$MASTER_DIR_RAW"
else
  MASTER_DIR="$ROOT_DIR/$MASTER_DIR_RAW"
fi
MASTER_SCRIPT="$MASTER_DIR/run_all_gpu_queues.sh"
WORKER_DIR="$MASTER_DIR/workers"
SESSION="${EVOCO_TMUX_SESSION:-evoco_all_experiments}"
mkdir -p "$MASTER_DIR"
mkdir -p "$WORKER_DIR"
rm -f "$WORKER_DIR"/run_gpu*_queue.sh

WORKER_SCRIPTS=()
for study_dir in "${STUDY_DIRS[@]}"; do
  found=0
  for gpu_script in "$study_dir"/run_gpu*.sh; do
    if [[ ! -f "$gpu_script" ]]; then
      continue
    fi
    found=1
    worker_name="$(basename "$gpu_script" .sh)"
    worker_script="$WORKER_DIR/${worker_name}_queue.sh"
    if [[ ! -f "$worker_script" ]]; then
      {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        printf "cd %q\n" "$ROOT_DIR"
        echo
      } > "$worker_script"
      WORKER_SCRIPTS+=("$worker_script")
    fi
    {
      printf "echo %q\n" "running $gpu_script"
      printf "bash %q\n" "$gpu_script"
      echo
    } >> "$worker_script"
  done
  if ((found == 0)); then
    worker_script="$WORKER_DIR/missing_queue.sh"
    if [[ ! -f "$worker_script" ]]; then
      {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
      } > "$worker_script"
      WORKER_SCRIPTS+=("$worker_script")
    fi
    {
      printf "echo %q\n" "missing generated run_gpu*.sh under $study_dir"
      echo "exit 1"
    } >> "$worker_script"
  fi
done

if ((${#WORKER_SCRIPTS[@]} == 0)); then
  echo "no generated run_gpu*.sh scripts found" >&2
  exit 1
fi

for worker_script in "${WORKER_SCRIPTS[@]}"; do
  chmod +x "$worker_script"
done

if ((${#WORKER_SCRIPTS[@]} == 1)); then
  cp "${WORKER_SCRIPTS[0]}" "$MASTER_SCRIPT"
  chmod +x "$MASTER_SCRIPT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "attach with: tmux attach -t $SESSION"
  else
    tmux new -d -s "$SESSION" "bash $(printf "%q" "$MASTER_SCRIPT")"
    echo "Started master tmux queue: $SESSION"
    echo "attach with: tmux attach -t $SESSION"
  fi
else
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "echo 'multi-pair worker queues are launched as separate tmux sessions:'"
    for worker_script in "${WORKER_SCRIPTS[@]}"; do
      worker_name="$(basename "$worker_script" _queue.sh)"
      session_suffix="${worker_name#run_gpu}"
      worker_session="${SESSION}_g${session_suffix}"
      printf "echo %q\n" "  tmux attach -t $worker_session"
    done
  } > "$MASTER_SCRIPT"
  chmod +x "$MASTER_SCRIPT"

  for worker_script in "${WORKER_SCRIPTS[@]}"; do
    worker_name="$(basename "$worker_script" _queue.sh)"
    session_suffix="${worker_name#run_gpu}"
    worker_session="${SESSION}_g${session_suffix}"
    if tmux has-session -t "$worker_session" 2>/dev/null; then
      echo "tmux session already exists: $worker_session"
      echo "attach with: tmux attach -t $worker_session"
    else
      tmux new -d -s "$worker_session" "bash $(printf "%q" "$worker_script")"
      echo "Started worker tmux queue: $worker_session"
      echo "attach with: tmux attach -t $worker_session"
    fi
  done
  echo "Started ${#WORKER_SCRIPTS[@]} GPU-pair worker queue(s)."
fi
