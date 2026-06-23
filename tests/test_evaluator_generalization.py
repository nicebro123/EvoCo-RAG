"""验证真实泛化评估开关与回退逻辑。"""

import json
import os
from pathlib import Path

import pytest

from conftest import make_sample

from evoco_rag.config import EvoCoConfig
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.trainers.coevolution_trainer import CoevolutionTrainer
from scripts.train_evoco import publish_final_round_metrics


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
    cfg.contract.top_k = 1
    cfg.contract.eval_top_k = 2
    cfg.contract.max_selected_docs = 2
    ev = Evaluator(cfg, small_policy=_StubSmall(), large_auditor=_StubLarge(),
                   test_samples=[make_sample(), make_sample()])
    assert ev.can_generalize() is True
    metrics = ev.evaluate_generalization(round_id=0)
    assert metrics is not None
    assert metrics["num_examples"] == 2
    assert "accuracy" in metrics
    # 落盘的真实泛化指标文件存在
    assert os.path.exists(os.path.join(cfg.output_dir, "metrics", "test_eval_round_000.json"))
    predictions = os.path.join(
        cfg.output_dir, "metrics", "test_predictions_round_000.jsonl")
    assert os.path.exists(predictions)
    assert sum(1 for _ in open(predictions, encoding="utf-8")) == 2
    with open(predictions, encoding="utf-8") as f:
        first = json.loads(next(f))
    assert len(first["contract"]["candidate_docs"]) == 2
    assert first["training_targets"]["evaluation_only"] is True
    assert first["training_targets"]["large_sft_target"] is None


def test_round_eval_requires_generalization_inputs(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.ablation.use_evidence_audit = False
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False

    evaluator = Evaluator(cfg, small_policy=None, large_auditor=None, test_samples=None)
    trainer = CoevolutionTrainer(cfg, None, None, evaluator=evaluator)
    with pytest.raises(RuntimeError, match="per-round test evaluation requires"):
        trainer.run_round([make_sample()], round_id=0)


def test_training_uses_train_top_k(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.contract.top_k = 5
    cfg.contract.train_top_k = 1
    cfg.contract.eval_top_k = 3
    cfg.contract.max_selected_docs = 3
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False

    trainer = CoevolutionTrainer(
        cfg,
        _StubSmall(),
        _StubLarge(),
        evaluator=Evaluator(
            cfg,
            small_policy=_StubSmall(),
            large_auditor=_StubLarge(),
            test_samples=[make_sample()],
        ),
    )
    trainer.run_round([make_sample()], round_id=0)

    replay = Path(cfg.output_dir) / "replay" / "round_000.jsonl"
    record = json.loads(replay.read_text(encoding="utf-8").splitlines()[0])
    assert len(record["contract"]["candidate_docs"]) == 1


def test_every_round_writes_test_metrics_and_predictions(tmp_path):
    cfg = EvoCoConfig()
    cfg.output_dir = str(tmp_path / "out")
    cfg.training.num_rounds = 3
    cfg.ablation.train_small_lora = False
    cfg.ablation.train_large_lora = False

    samples = [make_sample()]
    evaluator = Evaluator(
        cfg,
        small_policy=_StubSmall(),
        large_auditor=_StubLarge(),
        test_samples=samples,
    )
    trainer = CoevolutionTrainer(
        cfg,
        _StubSmall(),
        _StubLarge(),
        evaluator=evaluator,
    )

    stats = trainer.run(samples)

    assert len(stats) == 3
    assert all(item["eval_source"] == "test_generalization" for item in stats)
    assert all(item["per_round_test_completed"] is True for item in stats)
    for round_id in range(3):
        metrics_path = os.path.join(
            cfg.output_dir, "metrics", f"test_eval_round_{round_id:03d}.json")
        predictions_path = os.path.join(
            cfg.output_dir, "metrics", f"test_predictions_round_{round_id:03d}.jsonl")
        assert os.path.exists(metrics_path)
        assert os.path.exists(predictions_path)
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["round"] == round_id
        assert metrics["eval_split"] == "test"
        assert metrics["evaluation_stage"] == "per_round"
        assert metrics["num_examples"] == 1
        assert sum(1 for _ in open(predictions_path, encoding="utf-8")) == 1

    assert publish_final_round_metrics(cfg.output_dir, stats) == 2
    metrics_dir = Path(cfg.output_dir) / "metrics"
    assert (metrics_dir / "test_eval.json").read_text(encoding="utf-8") == (
        metrics_dir / "test_eval_round_002.json").read_text(encoding="utf-8")
    assert (metrics_dir / "test_predictions.jsonl").read_text(encoding="utf-8") == (
        metrics_dir / "test_predictions_round_002.jsonl").read_text(encoding="utf-8")
