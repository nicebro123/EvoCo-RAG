"""Evidence contract builder.

The small model is a reranker.  This module converts ranked documents into a
structured evidence contract for the large generator/auditor.  The only
adaptive behaviour kept in the mainline is evidence-budget expansion
(``retrieve_more``) when the ranking signal is weak and more candidate
documents are available.
"""

from __future__ import annotations

import math

from .schemas import Answerability, EvidenceContract, EvidenceItem, RetrievalAction
from .text_utils import best_evidence_span


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _decide_budget_status(
    top1_conf: float,
    margin: float,
    high_conf_threshold: float,
    answer_now_margin: float,
    can_retrieve_more: bool = False,
    retrieve_more_conf_threshold: float | None = None,
    retrieve_more_margin_threshold: float | None = None,
) -> tuple[str, str]:
    """Return ``(retrieval_action, answerability)`` for evidence budgeting.

    ``ANSWER_NOW`` means the current top-k evidence window is used as-is.
    ``RETRIEVE_MORE`` means the reranker signal is weak enough to expand the
    evidence window.  There is no "ask auditor" decision here; the auditor is
    part of the normal RAG pipeline.
    """
    if top1_conf >= high_conf_threshold and margin >= answer_now_margin:
        return RetrievalAction.ANSWER_NOW, Answerability.HIGH
    if (
        can_retrieve_more
        and retrieve_more_conf_threshold is not None
        and retrieve_more_margin_threshold is not None
        and top1_conf < retrieve_more_conf_threshold
        and margin < retrieve_more_margin_threshold
    ):
        return RetrievalAction.RETRIEVE_MORE, Answerability.LOW
    if can_retrieve_more and top1_conf < high_conf_threshold * 0.6:
        return RetrievalAction.RETRIEVE_MORE, Answerability.LOW
    return RetrievalAction.ANSWER_NOW, Answerability.MEDIUM


def build_contract(
    sample,
    ranked_docs: list[dict],
    round_id: int = 0,
    top_k: int = 3,
    high_conf_threshold: float = 0.75,
    answer_now_margin: float = 0.15,
    max_selected_docs: int = 3,
    num_retrieval_rounds: int = 1,
    retrieve_more_conf_threshold: float | None = None,
    retrieve_more_margin_threshold: float | None = None,
) -> EvidenceContract:
    """Package reranker scores as an evidence contract.

    The mainline system is reranker -> evidence contract -> large generator/auditor.
    """
    ranked = sorted(ranked_docs, key=lambda d: d["score"], reverse=True)
    topk = ranked[:top_k]

    confidences = [float(_sigmoid(d["score"])) for d in topk]
    top1_conf = confidences[0] if confidences else 0.0
    margin = (confidences[0] - confidences[1]) if len(confidences) >= 2 else top1_conf
    can_retrieve_more = len(ranked) > top_k
    retrieval_action, answerability = _decide_budget_status(
        top1_conf,
        margin,
        high_conf_threshold,
        answer_now_margin,
        can_retrieve_more=can_retrieve_more,
        retrieve_more_conf_threshold=retrieve_more_conf_threshold,
        retrieve_more_margin_threshold=retrieve_more_margin_threshold,
    )

    selected = []
    for rank, (doc, conf) in enumerate(zip(topk, confidences), start=1):
        if rank > max_selected_docs:
            break
        doc_obj = sample.doc_by_id(doc["doc_id"]) or {}
        text = doc_obj.get("text") or doc_obj.get("raw") or ""
        span, overlap = best_evidence_span(sample.question, text)
        selected.append(EvidenceItem(
            doc_id=doc["doc_id"],
            rank=rank,
            doc_score=round(float(doc["score"]), 4),
            relevance_confidence=round(conf, 4),
            evidence_confidence=round(float(doc.get("evidence_confidence", overlap)), 4),
            span=span,
            reason="启发式：与问题词汇重叠最高的句子" if span else "",
        ))

    candidate_docs = [
        {
            "doc_id": d["doc_id"],
            "rank": i + 1,
            "doc_score": round(float(d["score"]), 4),
        }
        for i, d in enumerate(ranked[:top_k])
    ]

    return EvidenceContract(
        sample_id=sample.sample_id,
        round=round_id,
        question=sample.question,
        answerability=answerability,
        retrieval_action=retrieval_action,
        selected_evidence=selected,
        candidate_docs=candidate_docs,
        uncertainty={
            "evidence_budget_policy": "reranker_confidence",
            "top1_confidence": round(top1_conf, 4),
            "rank_margin": round(margin, 4),
            "can_retrieve_more": can_retrieve_more,
            "evidence_conflict": False,
            "missing_relation": top1_conf < high_conf_threshold * 0.6,
            "retrieve_more_conf_threshold": retrieve_more_conf_threshold,
            "retrieve_more_margin_threshold": retrieve_more_margin_threshold,
        },
        cost={
            "num_ranked_docs": len(ranked),
            "num_selected_docs": len(selected),
            "num_retrieval_rounds": num_retrieval_rounds,
        },
    )
