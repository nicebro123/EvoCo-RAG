#!/usr/bin/env python3
"""Aggregate protocol-v3 experiment metrics into JSON/CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = "../rag_assets/outputs/experiments"
DEFAULT_PROTOCOL_VERSION = 3
SCALAR_METRICS = (
    "accuracy",
    "corag_style_accuracy",
    "schema_valid_accuracy",
    "recall_at_k",
    "mrr",
    "answer_in_topk_context_rate",
    "evidence_support_rate",
    "citation_correctness",
    "evidence_quote_support_rate",
    "unsupported_answer_rate",
    "generator_call_rate",
    "audit_call_rate",
    "audit_nonempty_output_rate",
    "avg_generation_candidates",
    "empty_answer_rate",
    "unfulfilled_action_rate",
    "audit_schema_valid_rate",
    "audit_trust_weight_mean",
    "confidence_success_correlation",
    "ece",
    "avg_selected_docs",
    "avg_total_cost_penalty",
    "cost_per_correct_answer",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_results(root: Path, protocol_version: int = DEFAULT_PROTOCOL_VERSION) -> list[dict]:
    rows: list[dict] = []
    for manifest_path in sorted(root.glob("*/launch_manifest.yaml")):
        manifest = _read_yaml(manifest_path)
        manifest_protocol = manifest.get("evaluation_protocol_version")
        study = manifest_path.parent.name
        for run in manifest.get("runs", []) or []:
            run_dir = Path(str(run.get("run_dir") or ""))
            metrics_path = run_dir / "metrics" / "test_eval.json"
            row = {
                "study": study,
                "run_index": run.get("index"),
                "run_name": run.get("name"),
                "gpu": run.get("gpu"),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "manifest_protocol_version": manifest_protocol,
                "evaluation_protocol_version": None,
                "status": "incomplete",
            }
            if metrics_path.exists():
                metrics = _read_json(metrics_path)
                result_protocol = metrics.get("evaluation_protocol_version")
                row["evaluation_protocol_version"] = result_protocol
                if manifest_protocol != protocol_version or result_protocol != protocol_version:
                    row["status"] = "protocol_mismatch"
                else:
                    row["status"] = "complete"
                for key in SCALAR_METRICS:
                    row[key] = metrics.get(key)
            else:
                for key in SCALAR_METRICS:
                    row[key] = None
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, rows: list[dict]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "experiment_summary.json"
    summary_csv = output_dir / "experiment_summary.csv"
    ranking_csv = output_dir / "experiment_ranking.csv"
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(summary_csv, rows)
    complete = [row for row in rows if row["status"] == "complete"]
    complete.sort(key=lambda row: (
        -(float(row["accuracy"]) if row.get("accuracy") is not None else float("-inf")),
        float(row["avg_total_cost_penalty"])
        if row.get("avg_total_cost_penalty") is not None else float("inf"),
    ))
    _write_csv(ranking_csv, complete)
    return {
        "json": summary_json,
        "csv": summary_csv,
        "ranking": ranking_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--protocol-version", type=int, default=DEFAULT_PROTOCOL_VERSION)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / f"summary_v{args.protocol_version}"
    )
    rows = collect_results(root, args.protocol_version)
    paths = write_summary(output_dir, rows)
    counts = {status: sum(row["status"] == status for row in rows) for status in (
        "complete", "incomplete", "protocol_mismatch"
    )}
    print(
        f"runs={len(rows)} complete={counts['complete']} "
        f"incomplete={counts['incomplete']} protocol_mismatch={counts['protocol_mismatch']}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
