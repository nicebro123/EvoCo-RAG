from conftest import make_sample

from evoco_rag.contract import build_contract
from evoco_rag.schemas import RetrievalAction
from evoco_rag.small_model import SmallRagPolicy


def _sample_with_extra_docs():
    sample = make_sample()
    sample.documents.extend([
        {"doc_id": 2, "title": "d2", "text": "other", "raw": "other"},
        {"doc_id": 3, "title": "d3", "text": "other", "raw": "other"},
    ])
    return sample


def test_retrieve_more_expands_candidate_pool_as_evidence_budget():
    sample = _sample_with_extra_docs()
    policy = object.__new__(SmallRagPolicy)
    policy.rank_documents = lambda _: [
        {"doc_id": 0, "score": -4.0},
        {"doc_id": 1, "score": -4.01},
        {"doc_id": 2, "score": -4.02},
        {"doc_id": 3, "score": -4.03},
    ]

    contract = policy.build_contract(
        sample,
        top_k=1,
        max_selected_docs=1,
        retrieve_more_conf_threshold=0.62,
        retrieve_more_margin_threshold=0.05,
    )

    assert contract.retrieval_action == RetrievalAction.RETRIEVE_MORE
    assert len(contract.candidate_docs) == 2
    assert len(contract.selected_evidence) == 2
    assert contract.cost["num_retrieval_rounds"] == 2
    assert contract.uncertainty["retrieval_expanded"] is True


def test_ambiguous_ranking_uses_current_evidence_window_by_default():
    sample = make_sample()
    contract = build_contract(
        sample,
        [{"doc_id": 0, "score": 5.0}, {"doc_id": 1, "score": 4.99}],
        top_k=2,
        max_selected_docs=2,
    )
    assert contract.retrieval_action == RetrievalAction.ANSWER_NOW
