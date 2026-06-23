from pathlib import Path

import pytest
import yaml

from evoco_rag.config import EvoCoConfig


def _is_launch_spec(path: Path) -> bool:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return isinstance(raw, dict) and isinstance(raw.get("experiments"), list)


def _all_config_paths():
    return sorted(
        path
        for path in Path("configs").rglob("*.yaml")
        if "local" not in path.parts and not _is_launch_spec(path)
    )


def test_all_yaml_configs_load_and_keep_assets_outside_repo():
    paths = _all_config_paths()
    assert paths, "expected at least one YAML config"

    output_dirs = set()
    checkpoint_dirs = set()
    for path in paths:
        cfg = EvoCoConfig.load(str(path))
        assert cfg.output_dir.startswith("../rag_assets/"), str(path)
        assert cfg.data.train_path.startswith("../rag_assets/"), str(path)
        assert cfg.data.test_path.startswith("../rag_assets/"), str(path)
        assert cfg.models.small_base_path.startswith("../rag_assets/"), str(path)
        assert cfg.models.large_base_path.startswith("../rag_assets/"), str(path)
        assert cfg.models.small_lora_dir.startswith("../rag_assets/"), str(path)
        assert cfg.models.large_lora_dir.startswith("../rag_assets/"), str(path)
        assert cfg.runtime.audit_batch_size >= 1, str(path)
        assert cfg.training.large_batch_size >= 1, str(path)
        assert cfg.contract.max_selected_docs <= max(
            cfg.contract.train_k, cfg.contract.eval_k
        ), str(path)
        output_dirs.add(cfg.output_dir)
        checkpoint_dirs.add(cfg.models.small_lora_dir)
        checkpoint_dirs.add(cfg.models.large_lora_dir)

    assert len(output_dirs) == len(paths)
    assert len(checkpoint_dirs) == len(paths) * 2


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config keys in training"):
        EvoCoConfig.from_dict({"training": {"num_generations": 2}})


def test_contract_train_eval_top_k_fallback_and_override():
    cfg = EvoCoConfig.from_dict({"contract": {"top_k": 5}})
    assert cfg.contract.train_k == 5
    assert cfg.contract.eval_k == 5

    cfg = EvoCoConfig.from_dict({
        "contract": {
            "top_k": 5,
            "train_top_k": 1,
            "eval_top_k": 3,
        }
    })
    assert cfg.contract.train_k == 1
    assert cfg.contract.eval_k == 3
