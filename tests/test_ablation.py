"""验证消融开关在 CoevolutionTrainer 里真正生效。"""

from conftest import make_sample

from evoco_rag.config import EvoCoConfig
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer


def _trainer(tmp_path, **ablation):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False
    for k, v in ablation.items():
        setattr(cfg.ablation, k, v)
    return CoevolutionTrainer(cfg, None, None)


def test_no_action_policy_forces_answer_now(tmp_path):
    trainer = _trainer(tmp_path, use_action_policy=False)
    e = trainer.make_experience(make_sample(), round_id=0)
    assert e.contract["retrieval_action"] == "answer_now"


def test_answer_only_reward_has_no_support_component(tmp_path):
    trainer = _trainer(tmp_path, use_decomposed_reward=False)
    e = trainer.make_experience(make_sample(), round_id=0)
    r = e.rewards
    assert r["support_reward"] == 0.0
    assert r["citation_reward"] == 0.0
    assert r["total_reward"] == r["answer_reward"]


def test_decomposed_reward_has_components(tmp_path):
    trainer = _trainer(tmp_path, use_decomposed_reward=True)
    e = trainer.make_experience(make_sample(), round_id=0)
    # 分解 reward 至少包含 cost_penalty 这一项
    assert "cost_penalty" in e.rewards
    assert e.rewards["cost_penalty"] > 0
