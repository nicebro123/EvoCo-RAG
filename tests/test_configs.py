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
        assert cfg.contract.max_selected_docs <= cfg.contract.top_k, str(path)
        output_dirs.add(cfg.output_dir)
        checkpoint_dirs.add(cfg.models.small_lora_dir)
        checkpoint_dirs.add(cfg.models.large_lora_dir)

    assert len(output_dirs) == len(paths)
    assert len(checkpoint_dirs) == len(paths) * 2


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config keys in training"):
        EvoCoConfig.from_dict({"training": {"policy_num_generations": 2}})


def test_training_config_accepts_large_grpo_options():
    cfg = EvoCoConfig.from_dict({
        "training": {
            "large_train_method": "grpo",
            "grpo_num_generations": 4,
            "grpo_n_per_train": 2,
            "grpo_max_steps": 3,
        }
    })
    assert cfg.training.large_train_method == "grpo"
    assert cfg.training.grpo_num_generations == 4
    assert cfg.training.grpo_n_per_train == 2
    assert cfg.training.grpo_max_steps == 3
