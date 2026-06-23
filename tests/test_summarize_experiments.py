import csv
import json
from pathlib import Path

import yaml

from scripts.summarize_experiments import collect_results, write_summary


def _make_run(root: Path, study: str, run_name: str, protocol: int | None, accuracy: float):
    run_dir = root / study / run_name
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy": accuracy,
        "avg_total_cost_penalty": 0.25,
    }
    if protocol is not None:
        payload["evaluation_protocol_version"] = protocol
    (metrics_dir / "test_eval.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_summary_filters_protocol_and_ranks_complete_runs(tmp_path):
    root = tmp_path / "experiments"
    study_dir = root / "study_v2"
    study_dir.mkdir(parents=True)
    good = _make_run(root, "study_v2", "good", 2, 80.0)
    better = _make_run(root, "study_v2", "better", 2, 90.0)
    old = _make_run(root, "study_v2", "old", 1, 99.0)
    missing = root / "study_v2" / "missing"
    manifest = {
        "evaluation_protocol_version": 2,
        "runs": [
            {"index": 0, "name": "good", "gpu": "0,1", "run_dir": str(good)},
            {"index": 1, "name": "better", "gpu": "2,3", "run_dir": str(better)},
            {"index": 2, "name": "old", "gpu": "4,5", "run_dir": str(old)},
            {"index": 3, "name": "missing", "gpu": "6,7", "run_dir": str(missing)},
        ],
    }
    (study_dir / "launch_manifest.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8")

    rows = collect_results(root)
    paths = write_summary(root / "summary_v2", rows)

    assert [row["status"] for row in rows] == [
        "complete", "complete", "protocol_mismatch", "incomplete"
    ]
    with paths["ranking"].open(encoding="utf-8") as f:
        ranking = list(csv.DictReader(f))
    assert [row["run_name"] for row in ranking] == ["better", "good"]
