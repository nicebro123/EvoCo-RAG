import json
import subprocess
import sys
from pathlib import Path


def _write_pack(root: Path) -> None:
    datasets = []
    for dataset_id in ("alpha", "beta"):
        train_path = Path("datasets") / dataset_id / "data_v33" / "Pop" / "train_labels_list.json"
        test_path = Path("datasets") / dataset_id / "data" / "Pop" / "test.json"
        (root / train_path).parent.mkdir(parents=True, exist_ok=True)
        (root / test_path).parent.mkdir(parents=True, exist_ok=True)
        (root / train_path).write_text("[]\n", encoding="utf-8")
        (root / test_path).write_text("[]\n", encoding="utf-8")
        datasets.append(
            {
                "id": dataset_id,
                "dataset_name": dataset_id.upper(),
                "train_path": str(train_path),
                "test_path": str(test_path),
                "train_examples": 1,
                "test_examples": 1,
            }
        )
    (root / "dataset_registry.json").write_text(
        json.dumps({"schema_version": "evoco-dataset-pack-v1", "datasets": datasets}),
        encoding="utf-8",
    )


def _run_make_config(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/make_dataset_config.py", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_make_dataset_config_lists_and_generates_all(tmp_path):
    data_root = tmp_path / "evoco_dataset_pack"
    output_root = tmp_path / "configs"
    _write_pack(data_root)

    listed = _run_make_config("--data-root", str(data_root), "--list")
    assert "alpha\tALPHA\ttrain=1\ttest=1" in listed.stdout
    assert "beta\tBETA\ttrain=1\ttest=1" in listed.stdout

    _run_make_config(
        "--data-root",
        str(data_root),
        "--all",
        "--output-root",
        str(output_root),
    )
    alpha_fast = output_root / "alpha_fast.yaml"
    beta_fast = output_root / "beta_fast.yaml"
    assert alpha_fast.exists()
    assert beta_fast.exists()
    assert "project:\n  name: \"evoco_alpha_fast\"" in alpha_fast.read_text(encoding="utf-8")


def test_make_dataset_config_resolves_rag_data_layout(tmp_path):
    asset_root = tmp_path / "rag_assets"
    rag_data = asset_root / "rag_data"
    data_root = rag_data / "evoco_dataset_pack"
    output = tmp_path / "alpha.yaml"
    _write_pack(data_root)

    listed = _run_make_config("--data-root", str(rag_data), "--list")
    assert "alpha\tALPHA\ttrain=1\ttest=1" in listed.stdout

    _run_make_config(
        "--data-root",
        str(asset_root),
        "--dataset-id",
        "alpha",
        "--output",
        str(output),
    )
    text = output.read_text(encoding="utf-8")
    assert str(data_root / "datasets" / "alpha" / "data_v33" / "Pop" / "train_labels_list.json") in text


def test_make_dataset_config_resolves_asset_sibling_hint(tmp_path):
    asset_root = tmp_path / "rag_assets"
    data_root = asset_root / "rag_data" / "evoco_dataset_pack"
    _write_pack(data_root)

    listed = _run_make_config("--data-root", str(asset_root / "evoco_dataset_pack"), "--list")

    assert "alpha\tALPHA\ttrain=1\ttest=1" in listed.stdout


def test_make_dataset_config_single_full_config(tmp_path):
    data_root = tmp_path / "evoco_dataset_pack"
    output = tmp_path / "single.yaml"
    _write_pack(data_root)

    _run_make_config(
        "--data-root",
        str(data_root),
        "--dataset-id",
        "alpha",
        "--full",
        "--output",
        str(output),
        "--output-dir",
        "../rag_assets/outputs/datasets/{dataset_id}_{suffix}_custom",
        "--checkpoint-root",
        "../rag_assets/checkpoints/datasets/{dataset_id}_{suffix}_custom",
    )

    text = output.read_text(encoding="utf-8")
    assert "name: \"evoco_alpha_full\"" in text
    assert "debug_size: null" in text
    assert "num_rounds: 3" in text
    assert "../rag_assets/outputs/datasets/alpha_full_custom" in text
    assert "../rag_assets/checkpoints/datasets/alpha_full_custom/small" in text
