"""测试公共夹具：让 tests 能 import evoco_rag，并提供构造样本的 helper。"""

import os
import sys

import pytest

# 把项目根目录（CoRAG-D63F ）加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoco_rag.schemas import (  # noqa: E402
    Answerability,
    EvidenceContract,
    EvidenceItem,
    LargeAudit,
    RagSample,
    RetrievalAction,
)


def make_sample():
    """doc0 含答案 'politician'，doc1 不含。"""
    return RagSample(
        sample_id="pop-train-000001",
        question="What is Henry Feilden's occupation?",
        answers=["politician", "political leader"],
        documents=[
            {"doc_id": 0, "title": "Henry Master Feilden",
             "text": "Henry Master Feilden was an English Conservative Party politician.",
             "raw": "title: Henry Master Feilden\ncontext: ... politician."},
            {"doc_id": 1, "title": "Henry Wemyss Feilden",
             "text": "Colonel Henry Wemyss Feilden was a British Army officer and naturalist.",
             "raw": "title: Henry Wemyss Feilden\ncontext: ... officer."},
        ],
    )


def make_contract(selected_doc_ids, candidate_doc_ids=(0, 1),
                  action=RetrievalAction.ANSWER_NOW,
                  answerability=Answerability.HIGH):
    sample = make_sample()
    selected = [
        EvidenceItem(doc_id=did, rank=i + 1, doc_score=5.0 - i,
                     relevance_confidence=0.9 - 0.1 * i, evidence_confidence=0.8,
                     span=sample.doc_by_id(did)["text"])
        for i, did in enumerate(selected_doc_ids)
    ]
    return EvidenceContract(
        sample_id=sample.sample_id,
        round=1,
        question=sample.question,
        answerability=answerability,
        retrieval_action=action,
        selected_evidence=selected,
        candidate_docs=[{"doc_id": d, "rank": i + 1, "doc_score": 5.0 - i}
                        for i, d in enumerate(candidate_doc_ids)],
        cost={"num_selected_docs": len(selected), "num_retrieval_rounds": 1},
    )


def make_audit(final_answer, used_doc_ids, **kw):
    sample = make_sample()
    used_evidence = [
        {"doc_id": doc_id, "quote": sample.doc_by_id(doc_id)["text"]}
        for doc_id in used_doc_ids
        if sample.doc_by_id(doc_id)
    ]
    return LargeAudit(
        sample_id=sample.sample_id,
        round=1,
        final_answer=final_answer,
        used_doc_ids=list(used_doc_ids),
        used_evidence=used_evidence,
        answer_correctness=kw.get("answer_correctness", "correct"),
        support_level=kw.get("support_level", "fully_supported"),
        failure_type=kw.get("failure_type", "none"),
        suggested_action=kw.get("suggested_action", RetrievalAction.ANSWER_NOW),
    )


@pytest.fixture
def sample():
    return make_sample()
