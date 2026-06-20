import pytest

from conftest import make_sample

from evoco_rag.contract import build_contract
from evoco_rag.schemas import RetrievalAction
from evoco_rag.small_model import SmallRagPolicy


def _ranked():
    return [{"doc_id": 0, "score": 5.0}, {"doc_id": 1, "score": 4.9}]


def test_policy_action_mode_overrides_heuristic_action():
    contract = build_contract(
        make_sample(),
        _ranked(),
        action_mode="policy",
        policy_action=RetrievalAction.RETRIEVE_MORE,
        policy_action_confidence=0.2,
    )
    assert contract.retrieval_action == RetrievalAction.RETRIEVE_MORE
    assert contract.uncertainty["policy_action_used"] is True


def test_hybrid_action_mode_requires_policy_confidence():
    low = build_contract(
        make_sample(),
        _ranked(),
        action_mode="hybrid",
        policy_action=RetrievalAction.RETRIEVE_MORE,
        policy_action_confidence=0.2,
        policy_action_min_conf=0.8,
    )
    high = build_contract(
        make_sample(),
        _ranked(),
        action_mode="hybrid",
        policy_action=RetrievalAction.RETRIEVE_MORE,
        policy_action_confidence=0.9,
        policy_action_min_conf=0.8,
    )
    assert low.uncertainty["policy_action_used"] is False
    assert high.retrieval_action == RetrievalAction.RETRIEVE_MORE
    assert high.uncertainty["policy_action_used"] is True


def test_invalid_action_mode_raises():
    with pytest.raises(ValueError):
        build_contract(make_sample(), _ranked(), action_mode="teleport")


def test_contract_uses_trainable_policy_confidence_when_present():
    ranked = [
        {"doc_id": 0, "score": 9.0, "policy_confidence": 0.2},
        {"doc_id": 1, "score": 8.0, "policy_confidence": 0.1},
    ]
    contract = build_contract(make_sample(), ranked, action_mode="heuristic")
    assert contract.selected_evidence[0].relevance_confidence == 0.2


def test_retrieve_more_expands_candidate_pool_in_same_round():
    sample = make_sample()
    sample.documents.extend([
        {"doc_id": 2, "title": "d2", "text": "other", "raw": "other"},
        {"doc_id": 3, "title": "d3", "text": "other", "raw": "other"},
    ])
    policy = object.__new__(SmallRagPolicy)
    policy.policy_heads_loaded = True
    policy.last_policy_prediction = {
        "action": RetrievalAction.RETRIEVE_MORE,
        "action_confidence": 0.99,
        "action_logits": [0.0, 5.0, 0.0, 0.0],
        "action_probs": [0.01, 0.97, 0.01, 0.01],
    }
    policy.rank_documents = lambda _: [
        {"doc_id": 0, "score": 4.0},
        {"doc_id": 1, "score": 3.0},
        {"doc_id": 2, "score": 2.0},
        {"doc_id": 3, "score": 1.0},
    ]

    contract = policy.build_contract(
        sample,
        top_k=1,
        max_selected_docs=1,
        action_mode="policy",
        policy_action_min_conf=0.45,
    )

    assert contract.retrieval_action == RetrievalAction.RETRIEVE_MORE
    assert len(contract.candidate_docs) == 2
    assert len(contract.selected_evidence) == 2
    assert contract.cost["num_retrieval_rounds"] == 2
    assert contract.uncertainty["retrieval_expanded"] is True
