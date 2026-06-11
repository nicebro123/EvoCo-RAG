#!/usr/bin/env python3
"""Validate an EvoCo-RAG dataset pack before training.

The checker is intentionally strict: it verifies registry entries, file
existence, row counts, required raw fields, and the actual EvoCo loader output.
Use it after downloading and extracting a dataset pack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evoco_rag.data import load_test_samples, load_train_samples
from evoco_rag.dataset_pack import DEFAULT_DATA_ROOT, resolve_dataset_pack_root


class PackValidationError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise PackValidationError(f"invalid JSON: {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackValidationError(message)


def check_train_row(row: dict, dataset_id: str, index: int) -> None:
    prefix = f"{dataset_id} train[{index}]"
    require(isinstance(row.get("question"), str) and row["question"].strip(), f"{prefix}: missing question")
    require(isinstance(row.get("answers"), list) and row["answers"], f"{prefix}: answers must be non-empty list")
    require(all(isinstance(a, str) and a.strip() for a in row["answers"]), f"{prefix}: invalid answer item")
    require(isinstance(row.get("context"), list) and row["context"], f"{prefix}: context must be non-empty list")
    require(all(isinstance(c, str) and "context:" in c for c in row["context"]), f"{prefix}: invalid context item")
    require(isinstance(row.get("labels"), list), f"{prefix}: labels must be present as list")


def check_test_row(row: dict, dataset_id: str, index: int) -> None:
    prefix = f"{dataset_id} test[{index}]"
    require(isinstance(row.get("question"), str) and row["question"].strip(), f"{prefix}: missing question")
    require(isinstance(row.get("answers"), list) and row["answers"], f"{prefix}: answers must be non-empty list")
    require(all(isinstance(a, str) and a.strip() for a in row["answers"]), f"{prefix}: invalid answer item")
    require(isinstance(row.get("ctxs"), list) and row["ctxs"], f"{prefix}: ctxs must be non-empty list")
    for j, ctx in enumerate(row["ctxs"]):
        require(isinstance(ctx, dict), f"{prefix}: ctxs[{j}] must be object")
        require("title" in ctx or "text" in ctx, f"{prefix}: ctxs[{j}] missing title/text")


def iter_checked_rows(rows: list, max_rows: int | None):
    if max_rows is None:
        return enumerate(rows)
    return enumerate(rows[:max_rows])


def validate_dataset(data_root: Path, dataset: dict, max_rows: int | None, loader_debug_size: int) -> dict:
    dataset_id = dataset.get("id")
    require(isinstance(dataset_id, str) and dataset_id, "registry dataset is missing id")
    dataset_name = dataset.get("dataset_name") or dataset_id

    train_rel = dataset.get("train_path")
    test_rel = dataset.get("test_path")
    require(isinstance(train_rel, str) and train_rel, f"{dataset_id}: missing train_path")
    require(isinstance(test_rel, str) and test_rel, f"{dataset_id}: missing test_path")

    train_path = data_root / train_rel
    test_path = data_root / test_rel
    require(train_path.exists(), f"{dataset_id}: train file not found: {train_path}")
    require(test_path.exists(), f"{dataset_id}: test file not found: {test_path}")

    train_rows = load_json(train_path)
    test_rows = load_json(test_path)
    require(isinstance(train_rows, list) and train_rows, f"{dataset_id}: train JSON must be non-empty list")
    require(isinstance(test_rows, list) and test_rows, f"{dataset_id}: test JSON must be non-empty list")

    expected_train = dataset.get("train_examples")
    expected_test = dataset.get("test_examples")
    if expected_train is not None:
        require(len(train_rows) == expected_train, f"{dataset_id}: train count {len(train_rows)} != {expected_train}")
    if expected_test is not None:
        require(len(test_rows) == expected_test, f"{dataset_id}: test count {len(test_rows)} != {expected_test}")

    for i, row in iter_checked_rows(train_rows, max_rows):
        require(isinstance(row, dict), f"{dataset_id} train[{i}] must be object")
        check_train_row(row, dataset_id, i)
    for i, row in iter_checked_rows(test_rows, max_rows):
        require(isinstance(row, dict), f"{dataset_id} test[{i}] must be object")
        check_test_row(row, dataset_id, i)

    loaded_train = load_train_samples(str(train_path), dataset_name, loader_debug_size)
    loaded_test = load_test_samples(str(test_path), dataset_name, loader_debug_size)
    require(loaded_train and loaded_test, f"{dataset_id}: loader returned empty samples")
    require(all(s.question and s.documents for s in loaded_train), f"{dataset_id}: invalid loaded train sample")
    require(all(s.question and s.documents for s in loaded_test), f"{dataset_id}: invalid loaded test sample")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "checked_train_rows": len(train_rows) if max_rows is None else min(max_rows, len(train_rows)),
        "checked_test_rows": len(test_rows) if max_rows is None else min(max_rows, len(test_rows)),
        "loader_train_samples": len(loaded_train),
        "loader_test_samples": len(loaded_test),
        "first_train_docs": len(loaded_train[0].documents),
        "first_test_docs": len(loaded_test[0].documents),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=(
            "Path to extracted evoco_dataset_pack, rag_data, or rag_assets. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument("--dataset-id", action="append", help="Dataset id to validate. Repeatable; defaults to all.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum raw rows per split to inspect. 0 means inspect all rows.",
    )
    parser.add_argument(
        "--loader-debug-size",
        type=int,
        default=8,
        help="Samples per split loaded through evoco_rag.data for loader validation.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data_root = resolve_dataset_pack_root(args.data_root)
    except FileNotFoundError as exc:
        raise PackValidationError(str(exc)) from exc
    registry_path = data_root / "dataset_registry.json"
    require(registry_path.exists(), f"dataset registry not found: {registry_path}")
    registry = load_json(registry_path)
    require(registry.get("schema_version") == "evoco-dataset-pack-v1", "unsupported schema_version")
    datasets = registry.get("datasets")
    require(isinstance(datasets, list) and datasets, "registry.datasets must be non-empty list")

    requested = set(args.dataset_id or [])
    selected = [d for d in datasets if not requested or d.get("id") in requested]
    missing = requested - {d.get("id") for d in selected}
    require(not missing, f"unknown dataset_id(s): {', '.join(sorted(missing))}")

    max_rows = None if args.max_rows == 0 else args.max_rows
    summaries = [validate_dataset(data_root, d, max_rows, args.loader_debug_size) for d in selected]

    if args.json:
        print(json.dumps({"data_root": str(data_root), "datasets": summaries}, ensure_ascii=False, indent=2))
        return

    print(f"validated dataset pack: {data_root}")
    for item in summaries:
        print(
            f"{item['dataset_id']}\t"
            f"train={item['train_examples']}\t"
            f"test={item['test_examples']}\t"
            f"checked={item['checked_train_rows']}/{item['checked_test_rows']}\t"
            f"loader={item['loader_train_samples']}/{item['loader_test_samples']}\t"
            f"docs={item['first_train_docs']}/{item['first_test_docs']}"
        )


if __name__ == "__main__":
    try:
        main()
    except PackValidationError as exc:
        raise SystemExit(f"dataset pack validation failed: {exc}") from exc
