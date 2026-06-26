"""CoRAG-style 规则验证器（开发文档 §4.4、§5.7）。

本模块刻意不使用 LLM-as-judge。答案正确性采用与 CoRAG/OpenQA 常用评测
一致的 normalized EM/sub-string hard anchor：标准答案归一化后作为子串出现在
final_answer 中即视为命中。引用与证据支持也只用确定性规则校验，避免
LLM 审计噪声直接污染训练信号。
"""

from __future__ import annotations

from .schemas import LargeAudit, RagSample, EvidenceContract, RuleVerification, SupportLevel
from .text_utils import exact_presence


def _doc_text(sample: RagSample, doc_id: int) -> str:
    doc = sample.doc_by_id(doc_id)
    if not doc:
        return ""
    # 优先用切分后的 text，没有则退回 raw
    return doc.get("text") or doc.get("raw") or ""


def _quote_supports_answer(sample: RagSample, evidence: dict) -> bool:
    try:
        doc_id = int(evidence.get("doc_id"))
    except (TypeError, ValueError):
        return False
    quote = str(evidence.get("quote") or "").strip()
    if not quote:
        return False
    document = _doc_text(sample, doc_id)
    return quote.lower() in document.lower() and exact_presence(sample.answers, quote)


def verify(
    sample: RagSample,
    contract: EvidenceContract,
    audit: LargeAudit,
    json_valid: bool = True,
    high_trust: float = 0.9,
    low_trust: float = 0.3,
) -> RuleVerification:
    """对单条样本做规则验证。

    规则（开发文档 §4.4 第一版）：
      1. answer_match: gold answer 是否出现在 final_answer 中
         （normalized EM/sub-string，即 exact_presence）。
      2. cited_doc_contains_answer: gold answer 是否出现在 used_doc_ids 对应原文中。
      3. used_doc_in_selected_evidence: 大模型引用文档是否来自小模型合约候选。
      4. support_rule_passed: 小模型选中的证据是否含 gold answer（规则级"支持"，
         不直接采信大模型自报的 support_level）。
      5. audit_trust_weight: JSON 合法 + 答案匹配 + 引用文档含答案 → 高权重，否则低权重。
    """
    notes: list[str] = []

    answer_match = exact_presence(sample.answers, audit.final_answer)

    cited_doc_contains_answer = False
    for doc_id in audit.used_doc_ids:
        if exact_presence(sample.answers, _doc_text(sample, doc_id)):
            cited_doc_contains_answer = True
            break

    evidence_quote_support = any(
        _quote_supports_answer(sample, item)
        for item in (audit.used_evidence or [])
        if isinstance(item, dict)
    )
    if not evidence_quote_support:
        notes.append("audit 未提供可验证且包含答案的原文 quote")

    contract_doc_ids = set(contract.selected_doc_ids()) | set(contract.candidate_doc_ids())
    if audit.used_doc_ids:
        used_doc_in_selected_evidence = all(
            did in contract_doc_ids for did in audit.used_doc_ids
        )
    else:
        used_doc_in_selected_evidence = False
        notes.append("audit 未给出 used_doc_ids")

    # 规则级"证据支持"独立于答案对错：小模型选中的证据里是否真的包含 gold answer。
    # 这是判定检索/重排是否成功、能否给小模型正反馈的依据（开发文档 §5.8 四象限）。
    support_rule_passed = any(
        exact_presence(sample.answers, _doc_text(sample, did))
        for did in contract.selected_doc_ids()
    )

    # 与大模型自报 support_level 的一致性提示（仅记录，不作为硬判定）
    if audit.support_level == SupportLevel.FULLY and not support_rule_passed:
        notes.append("大模型自报 fully_supported 但规则未通过，疑似 unsupported_answer")

    audit_metadata = audit.audit_metadata or {}
    try:
        self_consistency_score = float(audit_metadata.get("self_consistency", 1.0))
    except (TypeError, ValueError):
        self_consistency_score = 1.0
    self_consistency_score = max(0.0, min(1.0, self_consistency_score))

    # 信任权重
    if json_valid and answer_match and cited_doc_contains_answer and evidence_quote_support:
        audit_trust_weight = high_trust
    elif json_valid and (answer_match or used_doc_in_selected_evidence):
        audit_trust_weight = (high_trust + low_trust) / 2
    else:
        audit_trust_weight = low_trust
    if not json_valid:
        audit_trust_weight = min(audit_trust_weight, low_trust)
        notes.append("audit JSON 非法或解析失败，降低信任权重")
    if self_consistency_score < 0.5:
        audit_trust_weight = min(audit_trust_weight, (high_trust + low_trust) / 2)
        notes.append("audit 多候选一致性较低，降低信任权重")

    trust_components = {
        "json_valid_score": 1.0 if json_valid else 0.0,
        "answer_match_score": 1.0 if answer_match else 0.0,
        "citation_score": 1.0 if cited_doc_contains_answer else 0.0,
        "evidence_quote_support_score": 1.0 if evidence_quote_support else 0.0,
        "selected_doc_score": 1.0 if used_doc_in_selected_evidence else 0.0,
        "support_rule_score": 1.0 if support_rule_passed else 0.0,
        "self_consistency_score": round(self_consistency_score, 4),
    }

    return RuleVerification(
        sample_id=sample.sample_id,
        answer_match=answer_match,
        cited_doc_contains_answer=cited_doc_contains_answer,
        used_doc_in_selected_evidence=used_doc_in_selected_evidence,
        support_rule_passed=support_rule_passed,
        json_valid=json_valid,
        audit_trust_weight=round(audit_trust_weight, 4),
        trust_components=trust_components,
        notes=notes,
    )
