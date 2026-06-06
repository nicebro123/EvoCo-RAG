import pytest

from conftest import make_sample

from evoco_rag.contract import build_contract
from evoco_rag.schemas import RetrievalAction


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
