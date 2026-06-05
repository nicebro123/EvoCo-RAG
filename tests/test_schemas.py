import pytest

from evoco_rag.schemas import EvidenceContract, LargeAudit, RagSample


def test_valid_contract_passes():
    c = EvidenceContract.from_dict({
        "sample_id": "s1", "round": 1, "question": "q",
        "answerability": "high", "retrieval_action": "answer_now",
        "selected_evidence": [{"doc_id": 0, "rank": 1}],
        "candidate_docs": [{"doc_id": 0, "rank": 1, "doc_score": 5.0}],
    })
    assert c.selected_doc_ids() == [0]
    assert c.candidate_doc_ids() == [0]


def test_invalid_retrieval_action_raises():
    with pytest.raises(ValueError):
        EvidenceContract.from_dict({
            "sample_id": "s1", "answerability": "high",
            "retrieval_action": "teleport",
        })


def test_invalid_failure_type_raises():
    with pytest.raises(ValueError):
        LargeAudit.from_dict({
            "sample_id": "s1", "final_answer": "x",
            "answer_correctness": "correct", "support_level": "fully_supported",
            "failure_type": "cosmic_ray",
        })


def test_missing_sample_id_raises():
    with pytest.raises((ValueError, KeyError)):
        RagSample.from_dict({"question": "q", "answers": []})


def test_roundtrip_to_from_dict():
    c = EvidenceContract.from_dict({
        "sample_id": "s1", "answerability": "medium", "retrieval_action": "retrieve_more",
        "selected_evidence": [{"doc_id": 3, "span": "abc"}],
    })
    d = c.to_dict()
    c2 = EvidenceContract.from_dict(d)
    assert c2.selected_evidence[0].span == "abc"
