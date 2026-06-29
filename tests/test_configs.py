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
        EvoCoConfig.from_dict({"training": {"num_generations": 2}})


def test_cabl_config_is_optional_and_loadable():
    cfg = EvoCoConfig.from_dict({
        "cabl": {
            "enabled": True,
            "loss_weight": 0.3,
            "margin": 0.8,
            "max_negatives_per_sample": 4,
            "max_prompt_length": 512,
            "use_relation_answer_pool": True,
            "use_answer_type_filter": True,
            "use_counterfactual_evidence": True,
            "hard_aware_enabled": True,
            "hard_pair_weight": 2.5,
            "skip_retrieval_absent": False,
            "relation_hint_enabled": False,
        }
    })

    assert cfg.cabl.enabled is True
    assert cfg.cabl.loss_weight == 0.3
    assert cfg.cabl.margin == 0.8
    assert cfg.cabl.max_negatives_per_sample == 4
    assert cfg.cabl.max_prompt_length == 512
    assert cfg.cabl.use_relation_answer_pool is True
    assert cfg.cabl.use_answer_type_filter is True
    assert cfg.cabl.use_counterfactual_evidence is True
    assert cfg.cabl.hard_aware_enabled is True
    assert cfg.cabl.hard_pair_weight == 2.5
    assert cfg.cabl.skip_retrieval_absent is False
    assert cfg.cabl.relation_hint_enabled is False



def test_new_attribution_modules_config_are_optional_and_loadable():
    cfg = EvoCoConfig.from_dict({
        "evidence_expansion": {
            "enabled": True,
            "backend": "sample_internal",
            "trigger_mode": "always",
            "max_expanded_docs": 5,
        },
        "evidence_hard_negative": {
            "enabled": True,
            "max_per_sample": 2,
            "weight": 2.5,
        },
        "parametric_fallback": {
            "enabled": True,
            "correct_unsupported_weight": 0.25,
        },
        "small_policy": {
            "score_pointwise_loss_weight": 0.2,
        },
    })

    assert cfg.evidence_expansion.enabled is True
    assert cfg.evidence_expansion.backend == "sample_internal"
    assert cfg.evidence_expansion.trigger_mode == "always"
    assert cfg.evidence_hard_negative.enabled is True
    assert cfg.evidence_hard_negative.max_per_sample == 2
    assert cfg.evidence_hard_negative.weight == 2.5
    assert cfg.parametric_fallback.enabled is True
    assert cfg.parametric_fallback.correct_unsupported_weight == 0.25
    assert cfg.small_policy.score_pointwise_loss_weight == 0.2
