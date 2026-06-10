"""验证真实泛化评估开关与回退逻辑。"""

from conftest import make_sample

from evoco_rag.config import EvoCoConfig
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer


def test_can_generalize_false_without_models(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    ev = Evaluator(cfg, small_policy=None, large_auditor=None, test_samples=[make_sample()])
    assert ev.can_generalize() is False
    assert ev.evaluate_generalization(0) is None


def test_can_generalize_false_without_test_samples(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")

    class _Stub:
        pass

    ev = Evaluator(cfg, small_policy=_Stub(), large_auditor=_Stub(), test_samples=None)
    assert ev.can_generalize() is False
    assert ev.evaluate_generalization(0) is None


class _StubSmall:
    """最小小模型桩：按文档原序打分，复用真实 build_contract 逻辑。"""

    def build_contract(self, sample, **kwargs):
        from evoco_rag.contract import build_contract
        ranked = [{"doc_id": d["doc_id"], "score": -i}
                  for i, d in enumerate(sample.documents)]
        return build_contract(
            sample, ranked,
            round_id=kwargs.get("round_id", 0),
            top_k=kwargs.get("top_k", 3),
            high_conf_threshold=kwargs.get("high_conf_threshold", 0.75),
            answer_now_margin=kwargs.get("answer_now_margin", 0.15),
            max_selected_docs=kwargs.get("max_selected_docs", 3))


class _StubLarge:
    """最小大模型桩：直接用 doc0 文本当答案，返回合法审计。"""

    def generate_audit(self, sample, contract, show_gold=False, round_id=0):
        from evoco_rag.schemas import LargeAudit
        used = contract.selected_doc_ids()[:1]
        text = (sample.doc_by_id(used[0]) or {}).get("text", "") if used else ""
        return LargeAudit(
            sample_id=sample.sample_id, round=round_id,
            final_answer=text, used_doc_ids=used,
            answer_correctness="correct", support_level="fully_supported",
            failure_type="none", suggested_action="answer_now"), True


def test_generalization_path_runs_with_stub_models(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.contract.top_k = 2
    cfg.contract.max_selected_docs = 2
    ev = Evaluator(cfg, small_policy=_StubSmall(), large_auditor=_StubLarge(),
                   test_samples=[make_sample(), make_sample()])
    assert ev.can_generalize() is True
    metrics = ev.evaluate_generalization(round_id=0)
    assert metrics is not None
    assert metrics["num_examples"] == 2
    assert "accuracy" in metrics
    # 落盘的真实泛化指标文件存在
    import os
    assert os.path.exists(os.path.join(cfg.output_dir, "metrics", "test_eval_round_000.json"))


def test_round_eval_falls_back_to_train_replay(tmp_path):
    """有 evaluator 但无法泛化时，stats['eval'] 回退到训练集诊断并标注来源。"""
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False

    evaluator = Evaluator(cfg, small_policy=None, large_auditor=None, test_samples=None)
    trainer = CoevolutionTrainer(cfg, None, None, evaluator=evaluator)
    stats = trainer.run_round([make_sample()], round_id=0)

    assert stats["eval_source"] == "train_replay"
    assert "eval" in stats
    assert stats["eval"]["num_examples"] == 1
