"""证据合约构造器（开发文档 §4 实现说明、§5.4）。

第一阶段不引入新的可训练参数：输入小模型对候选文档的打分，用启发式规则
封装出 EvidenceContract——选 top-k、用 sigmoid 把分数转成置信度、句子级
启发式抽 span、按分数 margin 与阈值决定 retrieval_action。
"""

from __future__ import annotations

import math

from .schemas import Answerability, EvidenceContract, EvidenceItem, RetrievalAction
from .text_utils import best_evidence_span


ACTION_MODES = {"heuristic", "policy", "hybrid"}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _decide_action(top1_conf: float, margin: float, high_conf_threshold: float,
                   answer_now_margin: float, entity_ambiguity: bool) -> tuple[str, str]:
    """启发式动作策略，返回 (action, answerability)。"""
    if top1_conf >= high_conf_threshold and margin >= answer_now_margin:
        return RetrievalAction.ANSWER_NOW, Answerability.HIGH
    if entity_ambiguity and margin < answer_now_margin:
        return RetrievalAction.ASK_AUDITOR, Answerability.MEDIUM
    if top1_conf < high_conf_threshold * 0.6:
        return RetrievalAction.RETRIEVE_MORE, Answerability.LOW
    return RetrievalAction.ANSWER_NOW, Answerability.MEDIUM


def _answerability_for_action(action: str, top1_conf: float, high_conf_threshold: float) -> str:
    if action == RetrievalAction.ANSWER_NOW and top1_conf >= high_conf_threshold:
        return Answerability.HIGH
    if action == RetrievalAction.RETRIEVE_MORE:
        return Answerability.LOW
    return Answerability.MEDIUM


def _resolve_action(
    heuristic_action: str,
    heuristic_answerability: str,
    top1_conf: float,
    high_conf_threshold: float,
    action_mode: str,
    policy_action: str | None,
    policy_action_confidence: float | None,
    policy_action_min_conf: float,
) -> tuple[str, str, bool]:
    if action_mode not in ACTION_MODES:
        raise ValueError(f"非法 action_mode={action_mode!r}，允许值: {sorted(ACTION_MODES)}")
    if action_mode == "heuristic" or policy_action not in RetrievalAction.ALL:
        return heuristic_action, heuristic_answerability, False
    if action_mode == "hybrid" and (
        policy_action_confidence is None or policy_action_confidence < policy_action_min_conf
    ):
        return heuristic_action, heuristic_answerability, False
    return (
        policy_action,
        _answerability_for_action(policy_action, top1_conf, high_conf_threshold),
        True,
    )


def build_contract(
    sample,
    ranked_docs: list[dict],
    round_id: int = 0,
    top_k: int = 3,
    high_conf_threshold: float = 0.75,
    answer_now_margin: float = 0.15,
    max_selected_docs: int = 3,
    num_retrieval_rounds: int = 1,
    action_mode: str = "heuristic",
    policy_action: str | None = None,
    policy_action_confidence: float | None = None,
    policy_action_min_conf: float = 0.45,
) -> EvidenceContract:
    """把打分结果封装成证据合约。

    ranked_docs: 已按分数降序排列的 [{doc_id, score}], 通常来自 SmallRagPolicy.rank_documents。
    """
    ranked = sorted(ranked_docs, key=lambda d: d["score"], reverse=True)
    topk = ranked[:top_k]

    confidences = [_sigmoid(d["score"]) for d in topk]
    top1_conf = confidences[0] if confidences else 0.0
    margin = (confidences[0] - confidences[1]) if len(confidences) >= 2 else top1_conf

    # 简单实体歧义启发：top1/top2 置信度接近视为可能歧义
    entity_ambiguity = len(confidences) >= 2 and abs(confidences[0] - confidences[1]) < 0.05

    heuristic_action, heuristic_answerability = _decide_action(
        top1_conf, margin, high_conf_threshold, answer_now_margin, entity_ambiguity
    )
    action, answerability, policy_action_used = _resolve_action(
        heuristic_action,
        heuristic_answerability,
        top1_conf,
        high_conf_threshold,
        action_mode,
        policy_action,
        policy_action_confidence,
        policy_action_min_conf,
    )

    selected = []
    for rank, (doc, conf) in enumerate(zip(topk, confidences), start=1):
        if rank > max_selected_docs:
            break
        doc_obj = sample.doc_by_id(doc["doc_id"]) or {}
        text = doc_obj.get("text") or doc_obj.get("raw") or ""
        span, overlap = best_evidence_span(sample.question, text)
        evidence_conf = doc.get("evidence_confidence", overlap)
        selected.append(EvidenceItem(
            doc_id=doc["doc_id"],
            rank=rank,
            doc_score=round(float(doc["score"]), 4),
            relevance_confidence=round(conf, 4),
            evidence_confidence=round(float(evidence_conf), 4),
            span=span,
            reason=("policy_head" if "evidence_confidence" in doc
                    else "启发式：与问题词汇重叠最高的句子") if span else "",
        ))

    candidate_docs = [
        {
            "doc_id": d["doc_id"],
            "rank": i + 1,
            "doc_score": round(float(d["score"]), 4),
            **({"evidence_confidence": round(float(d["evidence_confidence"]), 4)}
               if "evidence_confidence" in d else {}),
            **({"policy_confidence": round(float(d["policy_confidence"]), 4)}
               if "policy_confidence" in d else {}),
        }
        for i, d in enumerate(ranked[:top_k])
    ]

    return EvidenceContract(
        sample_id=sample.sample_id,
        round=round_id,
        question=sample.question,
        answerability=answerability,
        retrieval_action=action,
        selected_evidence=selected,
        candidate_docs=candidate_docs,
        uncertainty={
            "action_mode": action_mode,
            "heuristic_action": heuristic_action,
            "heuristic_answerability": heuristic_answerability,
            "policy_action": policy_action,
            "policy_action_confidence": (
                round(float(policy_action_confidence), 4)
                if policy_action_confidence is not None else None
            ),
            "policy_action_used": policy_action_used,
            "entity_ambiguity": entity_ambiguity,
            "evidence_conflict": False,
            "missing_relation": top1_conf < high_conf_threshold * 0.6,
        },
        cost={
            "num_ranked_docs": len(ranked),
            "num_selected_docs": len(selected),
            "num_retrieval_rounds": num_retrieval_rounds,
        },
    )
