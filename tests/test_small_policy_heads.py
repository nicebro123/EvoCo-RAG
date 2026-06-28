import pytest
from types import SimpleNamespace

from evoco_rag.config import EvoCoConfig
from evoco_rag.schemas import RetrievalAction
from evoco_rag.small_model import ACTION_LABELS, SmallPolicyHeads, SmallRagPolicy
from evoco_rag.trainers.small_trainer import SmallTrainer, _binary_ece


def test_small_policy_config_parses():
    cfg = EvoCoConfig.from_dict({
        "small_policy": {
            "use_policy_heads": True,
            "evidence_loss_weight": 0.7,
            "action_loss_weight": 0.3,
            "calibration_loss_weight": 0.1,
        }
    })
    assert cfg.small_policy.use_policy_heads is True
    assert cfg.small_policy.evidence_loss_weight == 0.7
    assert cfg.small_policy.action_loss_weight == 0.3
    assert cfg.small_policy.calibration_loss_weight == 0.1


def test_policy_configs_enable_heads():
    debug = EvoCoConfig.load("configs/debug_policy.yaml")
    full = EvoCoConfig.load("configs/evoco_popqa_policy.yaml")
    assert debug.small_policy.use_policy_heads is True
    assert full.small_policy.use_policy_heads is True
    assert debug.contract.action_mode == "hybrid"
    assert full.contract.action_mode == "hybrid"
    assert debug.output_dir.endswith("policy_heads")
    assert full.models.small_lora_dir.endswith("evoco_popqa_policy/small")


def test_small_policy_heads_output_shapes():
    torch = pytest.importorskip("torch")
    heads = SmallPolicyHeads(hidden_size=6, num_actions=4)
    pooled = torch.zeros(3, 6)
    out = heads(pooled)
    assert out["evidence_logits"].shape == (3,)
    assert out["confidence_logits"].shape == (3,)
    assert out["action_logits"].shape == (3, 4)


def test_binary_ece_is_zero_for_perfect_confidence():
    assert _binary_ece([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_binary_ece_detects_miscalibration():
    value = _binary_ece([0.9, 0.9], [0.0, 0.0])
    assert value and value > 0.8


def test_small_trainer_reports_policy_head_metrics():
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

        def forward(self, input_ids=None, return_dict=True, output_hidden_states=False, **kwargs):
            batch = input_ids.shape[0]
            feats = torch.arange(batch * 4, dtype=torch.float32).view(batch, 1, 4) / 10.0
            return SimpleNamespace(
                logits=self.linear(feats[:, 0, :]),
                hidden_states=(feats,),
            )

    policy = SimpleNamespace(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        policy_heads=SmallPolicyHeads(hidden_size=4, num_actions=len(ACTION_LABELS)),
        action_label_to_id={label: i for i, label in enumerate(ACTION_LABELS)},
    )
    trainer = SmallTrainer(policy, batch_size=1)
    stats = trainer.train([{
        "sample_id": "s1",
        "question": "q",
        "documents": [{"doc_id": 0, "text": "positive"}, {"doc_id": 1, "text": "negative"}],
        "positive_doc_ids": [0],
        "negative_doc_ids": [1],
        "action_target": RetrievalAction.ANSWER_NOW,
    }])
    assert stats["policy_heads_enabled"] is True
    assert stats["avg_evidence_loss"] is not None
    assert stats["avg_action_loss"] is not None
    assert stats["avg_calibration_loss"] is not None
    assert stats["evidence_accuracy"] is not None
    assert stats["action_accuracy"] is not None
    assert stats["calibration_ece"] is not None
    assert policy.policy_heads_loaded is True


def test_small_trainer_learns_action_from_negative_only_failure():
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

        def forward(self, input_ids=None, return_dict=True, output_hidden_states=False, **kwargs):
            batch = input_ids.shape[0]
            feats = torch.ones(batch, 1, 4)
            return SimpleNamespace(logits=self.linear(feats[:, 0, :]), hidden_states=(feats,))

    policy = SimpleNamespace(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        policy_heads=SmallPolicyHeads(hidden_size=4, num_actions=len(ACTION_LABELS)),
        action_label_to_id={label: i for i, label in enumerate(ACTION_LABELS)},
        policy_heads_loaded=False,
    )
    stats = SmallTrainer(policy, batch_size=1).train([{
        "sample_id": "failure",
        "question": "q",
        "documents": [{"doc_id": 1, "text": "irrelevant"}],
        "positive_doc_ids": [],
        "negative_doc_ids": [1],
        "action_target": RetrievalAction.RETRIEVE_MORE,
    }])

    assert stats["trained_samples"] == 1
    assert stats["steps"] == 1
    assert stats["avg_action_loss"] is not None


def test_fresh_policy_heads_do_not_control_round_zero_contract():
    torch = pytest.importorskip("torch")

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, pairs, **kwargs):
            return FakeBatch({"input_ids": torch.zeros(len(pairs), 2, dtype=torch.long)})

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, input_ids=None, output_hidden_states=False, **kwargs):
            assert output_hidden_states is False
            return SimpleNamespace(
                logits=torch.arange(input_ids.shape[0], dtype=torch.float32).view(-1, 1),
                hidden_states=None,
            )

    policy = object.__new__(SmallRagPolicy)
    policy.model = FakeModel()
    policy.tokenizer = FakeTokenizer()
    policy.device = "cpu"
    policy.max_length = 32
    policy.policy_heads = SmallPolicyHeads(hidden_size=4, num_actions=len(ACTION_LABELS))
    policy.policy_heads_loaded = False
    policy.last_policy_prediction = {}

    ranked = policy.rank_documents(SimpleNamespace(
        question="q",
        documents=[
            {"doc_id": 0, "text": "a"},
            {"doc_id": 1, "text": "b"},
        ],
    ))

    assert policy.last_policy_prediction == {}
    assert all("policy_confidence" not in item for item in ranked)
