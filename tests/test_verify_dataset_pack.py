import json
import subprocess
import sys
from pathlib import Path


def _write_pack(root: Path, *, bad: bool = False) -> None:
    train_path = Path("datasets") / "alpha" / "data_v33" / "Pop" / "train_labels_list.json"
    test_path = Path("datasets") / "alpha" / "data" / "Pop" / "test.json"
    (root / train_path).parent.mkdir(parents=True, exist_ok=True)
    (root / test_path).parent.mkdir(parents=True, exist_ok=True)
    train_row = {
        "question": "Who wrote Hamlet?",
        "answers": ["William Shakespeare"],
        "context": ["title: Hamlet\ncontext: Hamlet is a play by William Shakespeare."],
        "labels": [["positive"]],
    }
    if bad:
        train_row.pop("labels")
    test_row = {
        "question": "Who wrote Hamlet?",
        "answers": ["William Shakespeare"],
        "ctxs": [{"title": "Hamlet", "text": "Hamlet is a play by William Shakespeare."}],
    }
    (root / train_path).write_text(json.dumps([train_row]), encoding="utf-8")
    (root / test_path).write_text(json.dumps([test_row]), encoding="utf-8")
    registry = {
        "schema_version": "evoco-dataset-pack-v1",
        "datasets": [
            {
                "id": "alpha",
                "dataset_name": "Alpha",
                "train_path": str(train_path),
                "test_path": str(test_path),
                "train_examples": 1,
                "test_examples": 1,
            }
        ],
    }
    (root / "dataset_registry.json").write_text(json.dumps(registry), encoding="utf-8")


def _run_verify(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/verify_dataset_pack.py", *args],
        text=True,
        capture_output=True,
    )


def test_verify_dataset_pack_accepts_valid_pack(tmp_path):
    data_root = tmp_path / "evoco_dataset_pack"
    _write_pack(data_root)

    result = _run_verify("--data-root", str(data_root), "--json")

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["datasets"][0]["dataset_id"] == "alpha"
    assert summary["datasets"][0]["train_examples"] == 1


def test_verify_dataset_pack_resolves_rag_data_layout(tmp_path):
    asset_root = tmp_path / "rag_assets"
    data_root = asset_root / "rag_data" / "evoco_dataset_pack"
    _write_pack(data_root)

    result = _run_verify("--data-root", str(asset_root / "rag_data"), "--json")

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["data_root"] == str(data_root.resolve())
    assert summary["datasets"][0]["dataset_id"] == "alpha"


def test_verify_dataset_pack_rejects_missing_labels(tmp_path):
    data_root = tmp_path / "evoco_dataset_pack"
    _write_pack(data_root, bad=True)

    result = _run_verify("--data-root", str(data_root))

    assert result.returncode != 0
    assert "labels must be present as list" in result.stderr
