import pytest

from conftest import make_sample
from evoco_rag.contract import build_contract
from evoco_rag.evidence_expansion import EvidenceExpansionRuntime, maybe_expand_contract
from evoco_rag.schemas import RetrievalAction


def test_sample_internal_expansion_expands_only_existing_candidate_pool():
    sample = make_sample()
    sample.documents.append({
        "doc_id": 2,
        "title": "Henry Feilden archive",
        "text": "Another irrelevant document.",
        "raw": "Another irrelevant document.",
    })
    ranked = [
        {"doc_id": 0, "score": 0.01},
        {"doc_id": 1, "score": 0.0},
        {"doc_id": 2, "score": -0.01},
    ]
    contract = build_contract(sample, ranked, top_k=1, max_selected_docs=1)

    expanded = maybe_expand_contract(
        sample,
        ranked,
        contract,
        runtime=EvidenceExpansionRuntime(enabled=True, trigger_mode="always", max_expanded_docs=3),
        round_id=0,
        high_conf_threshold=0.75,
        answer_now_margin=0.15,
        max_selected_docs=1,
        action_mode="heuristic",
    )

    assert expanded is not contract
    assert expanded.retrieval_action == RetrievalAction.RETRIEVE_MORE
    assert len(expanded.candidate_docs) == 3
    assert expanded.uncertainty["evidence_expansion"]["triggered"] is True
    assert expanded.cost["evidence_expansion_backend"] == "sample_internal"


def test_expansion_rejects_external_backend_by_default():
    sample = make_sample()
    ranked = [{"doc_id": 0, "score": 0.0}]
    contract = build_contract(sample, ranked, top_k=1)

    with pytest.raises(ValueError, match="unsupported evidence expansion backend"):
        maybe_expand_contract(
            sample,
            ranked,
            contract,
            runtime=EvidenceExpansionRuntime(enabled=True, backend="external_index"),
            round_id=0,
            high_conf_threshold=0.75,
            answer_now_margin=0.15,
            max_selected_docs=1,
            action_mode="heuristic",
        )
