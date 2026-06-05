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
