from types import SimpleNamespace

import pytest

from evoco_rag.config import EvoCoConfig
from evoco_rag.trainers.small_trainer import SmallTrainer, _binary_ece


def test_small_reranker_config_parses_evidence_weight():
    cfg = EvoCoConfig.from_dict({
        "small_policy": {
            "evidence_loss_weight": 0.7,
        }
    })
    assert cfg.small_policy.evidence_loss_weight == 0.7


def test_binary_ece_is_zero_for_perfect_confidence():
    assert _binary_ece([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_binary_ece_detects_miscalibration():
    value = _binary_ece([0.9, 0.9], [0.0, 0.0])
    assert value and value > 0.8


def test_small_trainer_reports_ranking_metrics_only():
    torch = pytest.importorskip("torch")

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, pairs, **kwargs):
            return FakeBatch({"input_ids": torch.zeros(len(pairs), 2, dtype=torch.long)})

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, input_ids=None, return_dict=True, **kwargs):
            batch = input_ids.shape[0]
            feats = torch.arange(batch * 4, dtype=torch.float32).view(batch, 4) / 10.0
            return SimpleNamespace(logits=self.linear(feats))

    policy = SimpleNamespace(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    trainer = SmallTrainer(policy, batch_size=1)
    stats = trainer.train([{
        "sample_id": "s1",
        "question": "q",
        "documents": [{"doc_id": 0, "text": "positive"}, {"doc_id": 1, "text": "negative"}],
        "positive_doc_ids": [0],
        "negative_doc_ids": [1],
    }])
    assert stats["trained_samples"] == 1
    assert stats["steps"] == 1
    assert stats["avg_rank_loss"] is not None
