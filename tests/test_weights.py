import json
import os

from evoco_rag.config import EvoCoConfig
from evoco_rag.weights import (
    adapter_rounds,
    checkpoint_round_dir,
    is_lora_adapter_dir,
    latest_checkpoint_round,
    latest_round_adapter,
    prepare_weight_layout,
    resolve_adapter_for_loading,
    write_weight_manifest,
)


def _adapter_dir(root, name, model_file="adapter_model.safetensors"):
    path = root / name
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / model_file).write_text("weights", encoding="utf-8")
    return path


def test_adapter_dir_detection(tmp_path):
    adapter = _adapter_dir(tmp_path, "round_000")
    root_only = tmp_path / "empty_root"
    root_only.mkdir()
    assert is_lora_adapter_dir(str(adapter)) is True
    assert is_lora_adapter_dir(str(root_only)) is False
    assert is_lora_adapter_dir(None) is False


def test_latest_round_adapter_prefers_highest_complete_round(tmp_path):
    _adapter_dir(tmp_path, "round_000")
    latest = _adapter_dir(tmp_path, "round_002", model_file="adapter_model.bin")
    incomplete = tmp_path / "round_003"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert latest_round_adapter(str(tmp_path)) == str(latest)
    assert resolve_adapter_for_loading(str(tmp_path)) == str(latest)
    assert latest_checkpoint_round(str(tmp_path)) == 2
    assert [rid for rid, _ in adapter_rounds(str(tmp_path))] == [0, 2]


def test_resolve_direct_adapter_and_empty_root(tmp_path):
    adapter = _adapter_dir(tmp_path, "adapter")
    empty_root = tmp_path / "checkpoints"
    empty_root.mkdir()
    assert resolve_adapter_for_loading(str(adapter)) == str(adapter)
    assert resolve_adapter_for_loading(str(empty_root)) is None


def test_prepare_layout_and_manifest(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.models.small_lora_dir = str(tmp_path / "outputs" / "checkpoints" / "small")
    cfg.models.large_lora_dir = str(tmp_path / "outputs" / "checkpoints" / "large")

    layout = prepare_weight_layout(cfg, create=True)
    assert os.path.isdir(layout["small_checkpoint_root"])
    assert os.path.isdir(layout["large_checkpoint_root"])
    assert checkpoint_round_dir(cfg.models.small_lora_dir, 4).endswith("round_004")

    manifest_path = write_weight_manifest(cfg)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["weights"]["small_checkpoint_root"] == cfg.models.small_lora_dir
    assert manifest["weights"]["large_checkpoint_root"] == cfg.models.large_lora_dir
    assert manifest["runtime_config"]["candidate_doc_char_limit"] == 1200
    assert manifest["small_policy_config"]["use_policy_heads"] is False
    assert manifest["models_config"]["use_4bit"] is False
