#!/usr/bin/env python3
"""Generate an EvoCo-RAG training config from a dataset pack registry.

The dataset pack is expected to contain:

  dataset_registry.json
  datasets/<dataset_id>/data_v33/Pop/train_labels_list.json
  datasets/<dataset_id>/data/Pop/test.json

Example:

  python scripts/make_dataset_config.py \
    --data-root ../rag_assets/rag_data/evoco_dataset_pack \
    --dataset-id hotpotqa_distractor \
    --output configs/local/hotpotqa_distractor_fast.yaml

  python scripts/make_dataset_config.py \
    --data-root ../rag_assets/rag_data/evoco_dataset_pack \
    --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.dataset_pack import DEFAULT_DATA_ROOT, resolve_dataset_pack_root


def quote(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def load_registry(data_root: Path) -> dict:
    path = data_root / "dataset_registry.json"
    if not path.exists():
        raise SystemExit(f"dataset registry not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_dataset(registry: dict, dataset_id: str) -> dict:
    for item in registry.get("datasets", []):
        if item.get("id") == dataset_id:
            return item
    available = ", ".join(d.get("id", "") for d in registry.get("datasets", []))
    raise SystemExit(f"unknown dataset_id={dataset_id!r}. Available: {available}")


def format_run_path(template: str | None, dataset_id: str, suffix: str) -> str | None:
    if template is None:
        return None
    return template.format(dataset_id=dataset_id, suffix=suffix)


def render_config(args, dataset: dict, train_path: Path, test_path: Path) -> str:
    debug_size = None if args.full else args.debug_size
    num_rounds = args.num_rounds if args.num_rounds is not None else (3 if args.full else 1)
    suffix = "full" if args.full else "fast"
    run_name = args.name or f"evoco_{dataset['id']}_{suffix}"
    output_dir = (
        format_run_path(args.output_dir, dataset["id"], suffix)
        or f"../rag_assets/outputs/datasets/{dataset['id']}_{suffix}"
    )
    checkpoint_root = (
        format_run_path(args.checkpoint_root, dataset["id"], suffix)
        or f"../rag_assets/checkpoints/datasets/{dataset['id']}_{suffix}"
    )

    return f"""project:
  name: {quote(run_name)}
  seed: {args.seed}
  output_dir: {quote(output_dir)}

data:
  train_path: {quote(train_path)}
  test_path: {quote(test_path)}
  dataset_name: {quote(dataset.get('dataset_name') or dataset['id'])}
  debug_size: {quote(debug_size)}

models:
  small_base_path: {quote(args.small_base_path)}
  large_base_path: {quote(args.large_base_path)}
  small_lora_dir: {quote(str(Path(checkpoint_root) / "small"))}
  large_lora_dir: {quote(str(Path(checkpoint_root) / "large"))}
  use_4bit: {quote(args.use_4bit)}

contract:
  top_k: {args.top_k}
  high_conf_threshold: 0.75
  answer_now_margin: 0.15
  max_selected_docs: {args.max_selected_docs}
  action_mode: hybrid
  policy_action_min_conf: 0.45

runtime:
  candidate_doc_char_limit: {args.candidate_doc_char_limit}
  num_audit_candidates: {args.num_audit_candidates}
  audit_batch_size: {args.audit_batch_size}
  audit_temperature: 0.7
  max_prompt_length: {args.max_prompt_length}
  max_completion_length: {args.max_completion_length}
  progress_interval: 10
  replay_flush_interval: 1

small_policy:
  use_policy_heads: true
  evidence_loss_weight: 1.0
  action_loss_weight: 0.5
  calibration_loss_weight: 0.2

training:
  num_rounds: {num_rounds}
  batch_size: {args.batch_size}
  large_batch_size: {args.large_batch_size}
  small_lr: 5.0e-5
  large_lr: 1.0e-5

reward:
  answer_weight: 1.0
  support_weight: 1.0
  citation_weight: 1.0
  calibration_weight: 0.2
  selected_doc_cost: 0.05
  retrieval_round_cost: 0.1
  audit_call_cost: 0.1
  rewrite_cost: 0.1
  retrieve_more_cost: 0.1

ablation:
  use_evidence_audit: true
  use_action_policy: true
  use_decomposed_reward: true
  train_small_lora: true
  train_large_lora: true
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=(
            "Path to evoco_dataset_pack, rag_data, or rag_assets. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument("--dataset-id", help="Dataset id from dataset_registry.json.")
    parser.add_argument("--all", action="store_true", help="Generate configs for every dataset in the registry.")
    parser.add_argument("--list", action="store_true", help="List available dataset ids and exit.")
    parser.add_argument("--output", help="Output YAML path.")
    parser.add_argument("--output-root", default="configs/local", help="Directory used by --all.")
    parser.add_argument("--name", help="Override project.name.")
    parser.add_argument("--full", action="store_true", help="Generate full-run settings: debug_size=null, num_rounds=3.")
    parser.add_argument("--debug-size", type=int, default=512)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--small-base-path", default="../rag_assets/base_models/reranker/bge-reranker-v2-m3")
    parser.add_argument("--large-base-path", default="../rag_assets/base_models/generator/Mistral-Nemo-Instruct-2407")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-selected-docs", type=int, default=3)
    parser.add_argument("--candidate-doc-char-limit", type=int, default=800)
    parser.add_argument("--num-audit-candidates", type=int, default=1)
    parser.add_argument("--audit-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--large-batch-size", type=int, default=2)
    return parser.parse_args()


def validate_dataset(data_root: Path, dataset: dict) -> tuple[Path, Path]:
    train_path = data_root / dataset["train_path"]
    test_path = data_root / dataset["test_path"]
    if not train_path.exists():
        raise SystemExit(f"train file not found: {train_path}")
    if not test_path.exists():
        raise SystemExit(f"test file not found: {test_path}")
    return train_path, test_path


def write_config(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)


def main() -> None:
    args = parse_args()
    try:
        data_root = resolve_dataset_pack_root(args.data_root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    registry = load_registry(data_root)
    if args.list:
        for item in registry.get("datasets", []):
            print(
                f"{item['id']}\t"
                f"{item.get('dataset_name', item['id'])}\t"
                f"train={item.get('train_examples')}\t"
                f"test={item.get('test_examples')}"
            )
        return
    if args.all:
        if args.output:
            raise SystemExit("--output is for a single dataset. Use --output-root with --all.")
        if args.name:
            raise SystemExit("--name is for a single dataset. Omit it when using --all.")
        suffix = "full" if args.full else "fast"
        output_root = Path(args.output_root)
        for dataset in registry.get("datasets", []):
            train_path, test_path = validate_dataset(data_root, dataset)
            text = render_config(args, dataset, train_path, test_path)
            write_config(output_root / f"{dataset['id']}_{suffix}.yaml", text)
        return
    if not args.dataset_id:
        raise SystemExit("--dataset-id is required unless --list or --all is set")

    dataset = find_dataset(registry, args.dataset_id)
    train_path, test_path = validate_dataset(data_root, dataset)

    text = render_config(args, dataset, train_path, test_path)
    if args.output:
        output = Path(args.output)
    else:
        suffix = "full" if args.full else "fast"
        output = Path("configs") / "local" / f"{dataset['id']}_{suffix}.yaml"
    write_config(output, text)


if __name__ == "__main__":
    main()
