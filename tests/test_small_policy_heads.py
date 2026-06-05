import pytest

from evoco_rag.config import EvoCoConfig
from evoco_rag.small_model import SmallPolicyHeads


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


def test_small_policy_heads_output_shapes():
    torch = pytest.importorskip("torch")
    heads = SmallPolicyHeads(hidden_size=6, num_actions=4)
    pooled = torch.zeros(3, 6)
    out = heads(pooled)
    assert out["evidence_logits"].shape == (3,)
    assert out["confidence_logits"].shape == (3,)
    assert out["action_logits"].shape == (3, 4)
