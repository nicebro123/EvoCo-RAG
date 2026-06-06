"""端到端集成测试（无模型路径）。

CoevolutionTrainer 在 small=None / large=None 时走纯逻辑分支，可在 CPU 上验证
合约 → 验证 → reward → replay → 指标 的整条链路不抛错、产物结构正确。
"""

from conftest import make_sample

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
