from pathlib import Path

from evoco_rag.config import EvoCoConfig


def _all_config_paths():
    return sorted(path for path in Path("configs").rglob("*.yaml") if "local" not in path.parts)


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
