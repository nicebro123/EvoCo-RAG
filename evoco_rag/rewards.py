"""分解 reward 与责任归因（开发文档 §5.8、§7）。

把原来的 answer-only reward 拆成答案/证据/引用/校准/成本五项，并按
answer_match × support_rule_passed 的四象限，构造大小模型各自的训练 target。
核心原则：答案对错不再是唯一信号，证据是否真正支持答案成为独立监督信号。
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .schemas import (
    Answerability,
    AnswerCorrectness,
    AttributionCase,
    EvidenceContract,
    FailureType,
    LargeAudit,
    RagSample,
    RetrievalAction,
    RewardBreakdown,
    RuleVerification,
    SupportLevel,
)
from .text_utils import exact_presence


@dataclass
class RewardWeights:
    """对应 configs/evoco_popqa.yaml 的 reward 段。"""
    answer_weight: float = 1.0
    support_weight: float = 1.0
    citation_weight: float = 1.0
    calibration_weight: float = 0.2
    selected_doc_cost: float = 0.05
    retrieval_round_cost: float = 0.1
    audit_call_cost: float = 0.1
    retrieve_more_cost: float = 0.1


def classify_attribution_case(answer_match: bool, support_rule_passed: bool) -> str:
    """Map answer/support outcomes to the responsibility attribution quadrant."""
    if answer_match and support_rule_passed:
        return AttributionCase.BOTH_SUCCESS
    if answer_match and not support_rule_passed:
        return AttributionCase.PARAMETRIC_ANSWER_WITHOUT_SUPPORT
    if (not answer_match) and support_rule_passed:
        return AttributionCase.RETRIEVER_SUCCESS_GENERATOR_FAIL
    return AttributionCase.BOTH_FAIL


def _predicted_confidence_bucket(contract: EvidenceContract) -> str:
    """把小模型的置信度粗分为 high / medium / low，用于校准奖励。"""
    if contract.answerability == Answerability.HIGH:
        return "high"
    if contract.answerability == Answerability.LOW:
        return "low"
    # answerability=medium 时，看 top 证据的相关性置信度
    if contract.selected_evidence:
        top_conf = max(e.relevance_confidence for e in contract.selected_evidence)
        if top_conf >= 0.75:
            return "high"
        if top_conf <= 0.35:
            return "low"
    return "medium"


def compute_decomposed_reward(
    sample: RagSample,
    contract: EvidenceContract,
    audit: LargeAudit,
    verification: RuleVerification,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    w = weights or RewardWeights()

    answer_reward = w.answer_weight * (1.0 if verification.answer_match else 0.0)

    support_reward = w.support_weight * (
        1.0
        if (verification.support_rule_passed and audit.support_level == SupportLevel.FULLY)
        else 0.0
    )

    citation_reward = w.citation_weight * (
        1.0 if verification.cited_doc_contains_answer else 0.0
    )

    # 校准奖励：high 置信度应对应命中、low 置信度应对应未命中。
    bucket = _predicted_confidence_bucket(contract)
    if bucket == "medium":
        calibration_reward = 0.0
    else:
        aligned = (bucket == "high" and verification.answer_match) or (
            bucket == "low" and not verification.answer_match
        )
        calibration_reward = w.calibration_weight * (1.0 if aligned else -1.0)

    num_selected = contract.cost.get("num_selected_docs", len(contract.selected_evidence))
    num_rounds = contract.cost.get("num_retrieval_rounds", 1)
    action_cost_penalty = 0.0
    audit_metadata = audit.audit_metadata or {}
    try:
        candidate_count = max(
            1, int(audit_metadata.get("generation_candidate_count", 1)))
    except (TypeError, ValueError):
        candidate_count = 1
    extra_audit_candidates = max(0, candidate_count - 1)
    if extra_audit_candidates:
        action_cost_penalty += w.audit_call_cost * extra_audit_candidates
    elif contract.retrieval_action == RetrievalAction.RETRIEVE_MORE:
        action_cost_penalty += w.retrieve_more_cost
    cost_penalty = (
        w.selected_doc_cost * num_selected
        + w.retrieval_round_cost * num_rounds
        + action_cost_penalty
    )

    total = answer_reward + support_reward + citation_reward + calibration_reward - cost_penalty
    attribution_case = classify_attribution_case(
        verification.answer_match, verification.support_rule_passed)

    return RewardBreakdown(
        answer_reward=round(answer_reward, 4),
        support_reward=round(support_reward, 4),
        citation_reward=round(citation_reward, 4),
        calibration_reward=round(calibration_reward, 4),
        action_cost_penalty=round(action_cost_penalty, 4),
        cost_penalty=round(cost_penalty, 4),
        total_reward=round(total, 4),
        attribution_case=attribution_case,
    )


def _docs_containing_answer(sample: RagSample, doc_ids: list[int]) -> list[int]:
    out = []
    for did in doc_ids:
        doc = sample.doc_by_id(did)
        text = (doc.get("text") or doc.get("raw") or "") if doc else ""
        if exact_presence(sample.answers, text):
            out.append(did)
    return out


def _supported_answer_target(sample: RagSample, doc_ids: list[int]) -> dict | None:
    """Build a compact supervised generator target from gold-backed evidence."""
    for did in doc_ids:
        doc = sample.doc_by_id(did) or {}
        text = doc.get("text") or doc.get("raw") or ""
        for answer in sample.answers:
            answer = str(answer or "").strip()
            if not answer or not exact_presence([answer], text):
                continue
            match = re.search(re.escape(answer), text, flags=re.IGNORECASE)
            if match:
                quote = text[max(0, match.start() - 80):min(len(text), match.end() + 120)].strip()
            else:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                quote = next(
                    (sentence.strip() for sentence in sentences if exact_presence([answer], sentence)),
                    text[:240].strip(),
                )
            return {
                "final_answer": answer,
                "used_doc_ids": [did],
                "used_evidence": [{"doc_id": did, "quote": quote}],
                "answer_correctness": AnswerCorrectness.CORRECT,
                "support_level": SupportLevel.FULLY,
                "failure_type": FailureType.NONE,
                "small_model_feedback": [],
                "suggested_action": RetrievalAction.ANSWER_NOW,
            }
    return None


def build_training_targets(
    sample: RagSample,
    contract: EvidenceContract,
    audit: LargeAudit,
    verification: RuleVerification,
    reward: RewardBreakdown,
    include_supervised_targets: bool = True,
) -> dict:
    """按四象限构造大小模型训练 target（开发文档 §5.8 责任归因表）。

    | answer | support | 小模型               | 大模型                |
    |--------|---------|----------------------|-----------------------|
    | T      | T       | 正:含答案选中文档    | SFT 正样本            |
    | T      | F       | 真文档正例/误选负例   | 不训练无证据答案      |
    | F      | T       | 正:含答案选中文档    | 低 reward / SFT 修正  |
    | F      | F       | 漏排正例/误选负例     | 无支持则丢弃          |
    """
    answer_match = verification.answer_match
    support = verification.support_rule_passed
    attribution_case = classify_attribution_case(answer_match, support)

    selected_ids = contract.selected_doc_ids()
    candidate_ids = contract.candidate_doc_ids()
    all_doc_ids = [
        doc.get("doc_id") for doc in sample.documents if doc.get("doc_id") is not None
    ]
    relevant_pool_ids = _docs_containing_answer(sample, all_doc_ids)

    small_positive_doc_ids: list[int] = []
    small_negative_doc_ids: list[int] = []
    failure_type = audit.failure_type
    do_not_reward_retriever_reason = ""

    if support:
        # 检索/重排成功：所有可验证相关文档均可作为正样本，top-k 中
        # 未含答案的文档作为 hard negatives。
        small_positive_doc_ids = relevant_pool_ids
        small_negative_doc_ids = [
            did for did in candidate_ids
            if did not in small_positive_doc_ids
            and not exact_presence(
                sample.answers,
                (sample.doc_by_id(did) or {}).get("text")
                or (sample.doc_by_id(did) or {}).get("raw")
                or "",
            )
        ]
        if not answer_match:
            # 证据对但答案错 → 生成错误
            failure_type = FailureType.GENERATION_ERROR
    else:
        # top-k 未命中时，不能奖励错误选中文档；但候选池里真正含答案的
        # missed positives 必须作为监督，否则 reranker 永远无法纠正漏排。
        small_positive_doc_ids = [
            did for did in relevant_pool_ids if did not in selected_ids
        ]
        small_negative_doc_ids = list(selected_ids)
        do_not_reward_retriever_reason = "selected_evidence_not_supporting_answer"
        if answer_match:
            # 答案对但证据不支持 → 大模型凭参数知识答对，标记 unsupported_answer
            failure_type = FailureType.UNSUPPORTED_ANSWER
            do_not_reward_retriever_reason = "parametric_answer_without_support"

    # 大模型学习由规则验证过的证据监督目标，而不是复制自己的审计输出。
    # 这同时为“检索正确、生成错误”的样本提供明确纠错信号。
    large_sft_target = (
        _supported_answer_target(sample, selected_ids)
        if support and include_supervised_targets
        else None
    )
    large_sft_eligible = large_sft_target is not None

    if attribution_case == AttributionCase.BOTH_SUCCESS:
        small_credit_weight = 1.0
        large_credit_weight = 1.0
    elif attribution_case == AttributionCase.PARAMETRIC_ANSWER_WITHOUT_SUPPORT:
        small_credit_weight = 0.0
        large_credit_weight = 0.5
    elif attribution_case == AttributionCase.RETRIEVER_SUCCESS_GENERATOR_FAIL:
        small_credit_weight = 1.0
        large_credit_weight = 0.5
    else:
        small_credit_weight = 0.0
        large_credit_weight = 0.0

    return {
        "small_positive_doc_ids": small_positive_doc_ids,
        "small_negative_doc_ids": small_negative_doc_ids,
        "small_target_source": "gold_rule_verifier",
        "large_sft_eligible": large_sft_eligible,
        "large_sft_target": large_sft_target,
        "large_sft_target_source": "gold_supported_evidence" if large_sft_target else None,
        "evaluation_only": not include_supervised_targets,
        "failure_type": failure_type,
        "attribution_case": attribution_case,
        "small_credit_weight": small_credit_weight,
        "large_credit_weight": large_credit_weight,
        "do_not_reward_retriever_reason": do_not_reward_retriever_reason,
        "wrong_retriever_reward_if_answer_only": (
            attribution_case == AttributionCase.PARAMETRIC_ANSWER_WITHOUT_SUPPORT
        ),
    }
