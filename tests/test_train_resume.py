import json
from pathlib import Path

import pytest

from evoco_rag.config import EvoCoConfig
from scripts.train_evoco import resolve_training_state


def _adapter(root: Path, round_id: int) -> Path:
    path = root / f"round_{round_id:03d}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    return path


def _completed_round(output_dir: Path, round_id: int) -> None:
    metrics = output_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    stats = {
        "round": round_id,
        "eval_source": "test_generalization",
        "per_round_test_completed": True,
    }
    (metrics / f"round_{round_id:03d}.json").write_text(
        json.dumps(stats), encoding="utf-8")
    (metrics / f"test_eval_round_{round_id:03d}.json").write_text(
        json.dumps({
            "evaluation_protocol_version": 3,
            "round": round_id,
            "eval_split": "test",
            "num_examples": 1,
        }),
        encoding="utf-8",
    )
    (metrics / f"test_predictions_round_{round_id:03d}.jsonl").write_text(
        "{}\n", encoding="utf-8")


def _config(tmp_path: Path) -> EvoCoConfig:
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.models.small_lora_dir = str(tmp_path / "checkpoints" / "small")
    cfg.models.large_lora_dir = str(tmp_path / "checkpoints" / "large")
    cfg.training.num_rounds = 3
    return cfg


def test_resume_ignores_checkpoint_from_uncommitted_round(tmp_path):
    cfg = _config(tmp_path)
    _adapter(Path(cfg.models.small_lora_dir), 0)
    _adapter(Path(cfg.models.large_lora_dir), 0)

    assert resolve_training_state(cfg, resume=True) == (0, None, None)


def test_resume_uses_exact_last_completed_round(tmp_path):
    cfg = _config(tmp_path)
    small0 = _adapter(Path(cfg.models.small_lora_dir), 0)
    large0 = _adapter(Path(cfg.models.large_lora_dir), 0)
    _completed_round(Path(cfg.output_dir), 0)
    # Orphan adapters from interrupted round 1 must not be loaded.
    _adapter(Path(cfg.models.small_lora_dir), 1)
    _adapter(Path(cfg.models.large_lora_dir), 1)

    assert resolve_training_state(cfg, resume=True) == (
        1, str(small0), str(large0))


def test_resume_rejects_completed_round_with_missing_adapter(tmp_path):
    cfg = _config(tmp_path)
    _adapter(Path(cfg.models.small_lora_dir), 0)
    _completed_round(Path(cfg.output_dir), 0)

    with pytest.raises(SystemExit, match="large adapter is missing"):
        resolve_training_state(cfg, resume=True)


def test_fresh_run_rejects_existing_completed_metrics(tmp_path):
    cfg = _config(tmp_path)
    _completed_round(Path(cfg.output_dir), 0)

    with pytest.raises(SystemExit, match="completed or partial rounds"):
        resolve_training_state(cfg, resume=False)
