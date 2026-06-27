"""端到端集成测试（无模型路径）。

CoevolutionTrainer 在 small=None / large=None 时走纯逻辑分支，可在 CPU 上验证
合约 → 验证 → reward → replay → 指标 的整条链路不抛错、产物结构正确。
"""

from conftest import make_audit, make_contract, make_sample
import pytest

from evoco_rag.config import EvoCoConfig
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer


def test_make_experience_no_models(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    # 关闭审计与训练，纯走启发式合约 + 规则验证 + reward
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False

    trainer = CoevolutionTrainer(cfg, small_policy=None, large_auditor=None)
    samples = [make_sample()]
    stats = trainer.run_round(samples, round_id=0)

    assert stats["num_experiences"] == 1
    exps = trainer.replay.read(0)
    assert len(exps) == 1
    e = exps[0]
    # 结构完整性
    assert e.contract and e.verification and e.rewards and e.training_targets
    assert "total_reward" in e.rewards
    assert "small_positive_doc_ids" in e.training_targets


def test_run_multiple_rounds(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False
    cfg.training.num_rounds = 2

    trainer = CoevolutionTrainer(cfg, None, None)
    all_stats = trainer.run([make_sample()])
    assert len(all_stats) == 2
    assert {s["round"] for s in all_stats} == {0, 1}


def test_run_round_streams_outputs_and_progress(tmp_path, capsys):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False
    cfg.runtime.progress_interval = 1
    cfg.runtime.replay_flush_interval = 1

    trainer = CoevolutionTrainer(cfg, None, None)
    stats = trainer.run_round([make_sample(), make_sample()], round_id=0)

    captured = capsys.readouterr().out
    assert "round 0: experience 1/2" in captured
    assert "round 0: wrote 2 experiences" in captured
    assert stats["num_experiences"] == 2

    replay_path = tmp_path / "out" / "replay" / "round_000.jsonl"
    contracts_path = tmp_path / "out" / "contracts" / "round_000.jsonl"
    audits_path = tmp_path / "out" / "audits" / "round_000.jsonl"
    assert sum(1 for _ in replay_path.open(encoding="utf-8")) == 2
    assert sum(1 for _ in contracts_path.open(encoding="utf-8")) == 2
    assert sum(1 for _ in audits_path.open(encoding="utf-8")) == 2


def test_run_round_resumes_partial_replay_and_records_timing(tmp_path, capsys):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False
    cfg.runtime.progress_interval = 1
    cfg.runtime.replay_flush_interval = 1

    sample_a = make_sample()
    sample_a.sample_id = "sample-a"
    sample_b = make_sample()
    sample_b.sample_id = "sample-b"

    trainer = CoevolutionTrainer(cfg, None, None)
    partial = trainer.make_experience(sample_a, round_id=0)
    replay_path = tmp_path / "out" / "replay" / "round_000.jsonl"
    trainer.replay.write([partial], round_id=0)
    with replay_path.open("a", encoding="utf-8") as f:
        f.write('{"sample_id": "broken"')

    stats = trainer.run_round([sample_a, sample_b], round_id=0)

    captured = capsys.readouterr().out
    assert "skipped 1 invalid partial replay lines" in captured
    assert "resumed 1/2 existing experiences" in captured
    assert stats["num_experiences"] == 2
    assert stats["resumed_experiences"] == 1
    assert stats["generated_experiences"] == 1
    assert "experience_generation_seconds" in stats["timing"]
    assert "small_training_seconds" in stats["timing"]
    assert "large_training_seconds" in stats["timing"]
    assert "evaluation_seconds" in stats["timing"]
    assert "total_round_seconds" in stats["timing"]

    exps = trainer.replay.read(0)
    assert [e.sample_id for e in exps] == ["sample-a", "sample-b"]
    contracts_path = tmp_path / "out" / "contracts" / "round_000.jsonl"
    audits_path = tmp_path / "out" / "audits" / "round_000.jsonl"
    assert sum(1 for _ in replay_path.open(encoding="utf-8")) == 2
    assert sum(1 for _ in contracts_path.open(encoding="utf-8")) == 2
    assert sum(1 for _ in audits_path.open(encoding="utf-8")) == 2


def test_training_experience_never_exposes_gold_to_generator(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.runtime.num_audit_candidates = 3
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False

    class Small:
        def build_contract(self, sample, **kwargs):
            return make_contract(selected_doc_ids=[0], action="answer_now")

    class Large:
        def __init__(self):
            self.call = None

        def generate_audit_batch(
            self, samples, contracts, show_gold, round_id, batch_size,
            candidate_counts=None,
        ):
            self.call = {
                "show_gold": show_gold,
                "candidate_counts": candidate_counts,
            }
            return [(make_audit("politician", [0]), True) for _ in samples]

    large = Large()
    trainer = CoevolutionTrainer(cfg, Small(), large)
    trainer.make_experiences([make_sample()], round_id=0)

    assert large.call["show_gold"] is False
    assert large.call["candidate_counts"] == [3]


def test_failed_round_evaluation_does_not_commit_checkpoints(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.models.small_lora_dir = str(tmp_path / "checkpoints" / "small")
    cfg.models.large_lora_dir = str(tmp_path / "checkpoints" / "large")
    cfg.ablation.use_evidence_audit = False

    class SmallTrainer:
        def __init__(self):
            self.saved = []

        def train(self, pairs):
            return {"trained_samples": len(pairs)}

        def save(self, path):
            self.saved.append(path)

    class LargeTrainer:
        def __init__(self):
            self.saved = []

        def train_sft(self, experiences, batch_size):
            return {"trained_samples": len(experiences)}

        def save(self, path):
            self.saved.append(path)

    class FailingEvaluator:
        def evaluate(self, round_id):
            return {"num_examples": 1}

        def evaluate_generalization(self, round_id):
            raise RuntimeError("injected evaluation failure")

    small_trainer = SmallTrainer()
    large_trainer = LargeTrainer()
    trainer = CoevolutionTrainer(
        cfg,
        small_policy=None,
        large_auditor=None,
        small_trainer=small_trainer,
        large_trainer=large_trainer,
        evaluator=FailingEvaluator(),
    )

    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        trainer.run_round([make_sample()], round_id=0)

    assert small_trainer.saved == []
    assert large_trainer.saved == []
    assert not (tmp_path / "checkpoints" / "small" / "round_000").exists()
    assert not (tmp_path / "checkpoints" / "large" / "round_000").exists()

def test_run_round_trains_large_model_with_sft(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.models.large_lora_dir = str(tmp_path / "checkpoints" / "large")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = True

    class LargeTrainer:
        def __init__(self):
            self.calls = []
            self.saved = []

        def train_sft(self, experiences, batch_size):
            self.calls.append((experiences, batch_size))
            return {"method": "sft", "trained_samples": len(experiences)}

        def save(self, path):
            self.saved.append(path)

    large_trainer = LargeTrainer()
    trainer = CoevolutionTrainer(
        cfg,
        small_policy=None,
        large_auditor=None,
        large_trainer=large_trainer,
    )

    stats = trainer.run_round([make_sample()], round_id=0)

    assert stats["large"]["method"] == "sft"
    assert len(large_trainer.calls) == 1
    assert large_trainer.calls[0][1] == cfg.training.large_batch_size
    assert large_trainer.saved
