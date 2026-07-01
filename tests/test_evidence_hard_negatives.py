from types import SimpleNamespace

from conftest import make_sample
from evoco_rag.evidence_hard_negatives import HardNegativeConfig, mine_evidence_hard_negatives


def test_evidence_hard_negative_mines_same_entity_distractor_without_gold():
    sample = make_sample()
    records = mine_evidence_hard_negatives(
        sample,
        positive_doc_ids=[0],
        candidate_doc_ids=[0, 1],
        selected_doc_ids=[1],
        model_wrong_answer="officer",
        config=HardNegativeConfig(enabled=True, max_per_sample=3),
    )

    assert records
    assert records[0]["doc_id"] == 1
    assert "selected_unsupported_evidence" in records[0]["reasons"]
    assert "model_wrong_answer_source" in records[0]["reasons"]
    assert records[0]["relation"] == "occupation"


def test_evidence_hard_negative_never_returns_gold_doc():
    sample = make_sample()
    records = mine_evidence_hard_negatives(
        sample,
        positive_doc_ids=[0],
        candidate_doc_ids=[0],
        selected_doc_ids=[0],
        config=HardNegativeConfig(enabled=True),
    )

    assert all(item["doc_id"] != 0 for item in records)


def test_evidence_hard_negative_marks_same_name_relation_mismatch():
    sample = make_sample()
    sample.documents.append({
        "doc_id": 2,
        "title": "Henry Feilden",
        "text": "Henry Feilden was born in London and travelled widely.",
    })

    records = mine_evidence_hard_negatives(
        sample,
        positive_doc_ids=[0],
        candidate_doc_ids=[0, 2],
        selected_doc_ids=[2],
        config=HardNegativeConfig(enabled=True, max_per_sample=3),
    )

    target = next(item for item in records if item["doc_id"] == 2)
    assert "same_name_relation_mismatch" in target["reasons"]
    assert "weak_local_relation_support" in target["reasons"]
    assert target["entity_relation"]["same_name_distractor"] is True
