"""大模型审计 prompt 构造与输出解析（开发文档 §5.6、§13.1）。

本模块不依赖 torch：只负责
  - 构造强制 JSON 输出的审计 prompt；
  - 从大模型自由文本里 robust 地抽取 JSON；
  - 把 JSON 解析为 LargeAudit，非法字段降级，解析失败给 fallback audit。
真正的生成在 large_model.LargeGeneratorAuditor 里完成。
"""

from __future__ import annotations

import json
from typing import Optional

from .schemas import (
    AnswerCorrectness,
    FeedbackLabel,
    FailureType,
    LargeAudit,
    RetrievalAction,
    SupportLevel,
)

AUDIT_PROMPT_STYLE_AUDIT_JSON = "audit_json"
AUDIT_PROMPT_STYLE_HYBRID_ANALYSIS_JSON = "hybrid_analysis_json"
AUDIT_PROMPT_STYLE_CORAG_ANALYSIS_PLUS_AUDIT = "corag_analysis_plus_audit"
AUDIT_PROMPT_STYLES = {
    AUDIT_PROMPT_STYLE_AUDIT_JSON,
    AUDIT_PROMPT_STYLE_HYBRID_ANALYSIS_JSON,
    AUDIT_PROMPT_STYLE_CORAG_ANALYSIS_PLUS_AUDIT,
}


AUDIT_JSON_SCHEMA_HINT = """You MUST output ONLY a single JSON object, no Markdown, no extra text.
JSON schema:
{
  "final_answer": "<short entity-style answer string; no explanation>",
  "used_doc_ids": [<int doc_id>, ...],
  "used_evidence": [{"doc_id": <int>, "quote": "<verbatim span>"}],
  "answer_correctness": "correct|incorrect|unknown",
  "support_level": "fully_supported|partially_supported|unsupported",
  "failure_type": "none|retrieval_miss|rerank_error|entity_confusion|evidence_conflict|generation_error|unsupported_answer|over_retrieval",
  "small_model_feedback": [{"doc_id": <int>, "label": "positive|negative|hard_negative|ignore", "reason": "<why>"}],
  "suggested_action": "answer_now|retrieve_more|rewrite_query|ask_auditor"
}"""

HYBRID_AUDIT_JSON_SCHEMA_HINT = """You MUST first analyze the evidence, then output ONLY a single JSON object, no Markdown, no extra text.
JSON schema:
{
  "analysis": [
    {
      "doc_id": <int doc_id>,
      "extraction": "<verbatim or near-verbatim evidence span, or empty if irrelevant>",
      "reason": "<why this document supports, contradicts, or distracts from the question>",
      "support": "supporting|partial|irrelevant|distractor|conflicting"
    }
  ],
  "final_answer": "<short entity-style answer string; no explanation>",
  "used_doc_ids": [<int doc_id>, ...],
  "used_evidence": [{"doc_id": <int>, "quote": "<verbatim span>"}],
  "answer_correctness": "correct|incorrect|unknown",
  "support_level": "fully_supported|partially_supported|unsupported",
  "failure_type": "none|retrieval_miss|rerank_error|entity_confusion|evidence_conflict|generation_error|unsupported_answer|over_retrieval",
  "small_model_feedback": [{"doc_id": <int>, "label": "positive|negative|hard_negative|ignore", "reason": "<why>"}],
  "suggested_action": "answer_now|retrieve_more|rewrite_query|ask_auditor"
}"""

CORAG_ANALYSIS_PLUS_AUDIT_JSON_SCHEMA_HINT = """You MUST follow CoRAG's two-step document analysis and final-answer discipline, but the output MUST still be ONLY a single valid JSON object, no Markdown, no code fence, no extra text.
Put final_answer near the top and do not leave it empty.
JSON schema:
{
  "final_answer": "<short entity-style answer string; no explanation>",
  "answer_reasoning": "<one concise sentence explaining why final_answer follows from the supporting documents>",
  "document_analysis": [
    {
      "doc_id": <int doc_id>,
      "extraction": "<evidence span copied or closely paraphrased from this document>",
      "explanation": "<how this document addresses the question, or why it is a distractor>",
      "support": "supporting|partial|irrelevant|distractor|conflicting"
    }
  ],
  "used_doc_ids": [<int doc_id>, ...],
  "used_evidence": [{"doc_id": <int>, "quote": "<verbatim span>"}],
  "answer_correctness": "correct|incorrect|unknown",
  "support_level": "fully_supported|partially_supported|unsupported",
  "failure_type": "none|retrieval_miss|rerank_error|entity_confusion|evidence_conflict|generation_error|unsupported_answer|over_retrieval",
  "small_model_feedback": [{"doc_id": <int>, "label": "positive|negative|hard_negative|ignore", "reason": "<why>"}],
  "suggested_action": "answer_now|retrieve_more|rewrite_query|ask_auditor"
}"""

FAILURE_TYPE_DEFS = """failure_type definitions:
- none: answer correct and evidence supports it.
- retrieval_miss: required evidence is absent from candidate docs.
- rerank_error: evidence exists in candidates but was not selected.
- entity_confusion: same-name or semantically confused entities.
- evidence_conflict: multiple docs give conflicting evidence.
- generation_error: evidence is correct but the answer is wrong.
- unsupported_answer: answer correct but not supported by the cited evidence.
- over_retrieval: too many irrelevant docs were selected."""


def build_audit_prompt(
    sample,
    contract,
    show_gold: bool = False,
    candidate_doc_char_limit: int = 1200,
    prompt_style: str = AUDIT_PROMPT_STYLE_AUDIT_JSON,
) -> list[dict]:
    """构造 (system, user) chat messages。

    运行时训练和评估都必须保持 show_gold=False。参数仅保留给离线 prompt
    诊断，不能用于生成 replay 或测试预测。
    """
    if prompt_style not in AUDIT_PROMPT_STYLES:
        raise ValueError(
            f"unknown audit prompt_style={prompt_style!r}; "
            f"expected one of {sorted(AUDIT_PROMPT_STYLES)}"
        )
    hybrid = prompt_style == AUDIT_PROMPT_STYLE_HYBRID_ANALYSIS_JSON
    corag_analysis = prompt_style == AUDIT_PROMPT_STYLE_CORAG_ANALYSIS_PLUS_AUDIT
    if corag_analysis:
        schema_hint = CORAG_ANALYSIS_PLUS_AUDIT_JSON_SCHEMA_HINT
    elif hybrid:
        schema_hint = HYBRID_AUDIT_JSON_SCHEMA_HINT
    else:
        schema_hint = AUDIT_JSON_SCHEMA_HINT

    system = (
        "You are an evidence auditor for a retrieval-augmented QA system. "
        "Given a question, a small model's selected evidence and candidate documents, "
        "you must (1) answer the question, (2) cite which documents you used, and "
        "(3) audit whether the evidence truly supports your answer. "
        + (
            "Before deciding the final answer, explicitly analyze every candidate "
            "document inside the JSON analysis array. The analysis is not optional: "
            "it is the reasoning trace that will be converted into retriever feedback. "
            if hybrid else ""
        )
        + (
            "Use CoRAG's two-step discipline inside JSON: first fill document_analysis "
            "for each document with extraction and explanation, then write a concise "
            "final_answer and answer_reasoning. The answer should appear in final_answer, "
            "answer_reasoning, and at least one supporting extraction when evidence exists. "
            if corag_analysis else ""
        )
        + "For factoid QA, final_answer must be the shortest correct entity or phrase, "
        "not a full sentence and not an explanation. Prefer an answer phrase that "
        "appears verbatim in the quoted evidence. Be careful with same-name or "
        "near-name distractors: the cited document must describe the entity asked "
        "in the question, not another entity. For occupation questions, output the "
        "profession/category phrase explicitly associated with the question entity; "
        "do not output nationalities, dates, titles, or an unrelated role from a "
        "different same-name person. If several occupations are supported, choose "
        "the concise common category rather than a long biographical sentence.\n\n"
        + schema_hint + "\n\n" + FAILURE_TYPE_DEFS
    )

    limit = max(1, int(candidate_doc_char_limit or 1200))
    lines = [f"Question: {sample.question}", "", "Small model selected evidence:"]
    for ev in contract.selected_evidence:
        lines.append(f"  - doc_id={ev.doc_id} (conf={ev.relevance_confidence}): {ev.span}")
    lines.append("")
    lines.append("Candidate documents:")
    for cand in contract.candidate_docs:
        doc = sample.doc_by_id(cand.get("doc_id")) or {}
        text = doc.get("text") or doc.get("raw") or ""
        truncated = text[:limit]
        suffix = " ..." if len(text) > limit else ""
        lines.append(f"  [doc_id={cand.get('doc_id')}] {doc.get('title', '')}: {truncated}{suffix}")
    lines.append("")
    if corag_analysis:
        lines.append("CoRAG-style JSON analysis rules:")
        lines.append("  - Output JSON only, but internally follow Step 1 Document Analysis and Step 2 Final Answer.")
        lines.append("  - document_analysis must contain one item per candidate document when possible.")
        lines.append("  - Each document_analysis item must include extraction, explanation, and support.")
        lines.append("  - Put the answer phrase in final_answer; do not leave final_answer empty.")
        lines.append("  - answer_reasoning should briefly restate the answer and cite the supporting document.")
        lines.append("  - If a same-name document is about the wrong entity, mark it as distractor.")
        lines.append("")
    elif hybrid:
        lines.append("Evidence analysis rules:")
        lines.append("  - Fill the JSON analysis array before final_answer.")
        lines.append("  - For each candidate document, extract the most relevant span if any.")
        lines.append("  - Mark support as supporting, partial, irrelevant, distractor, or conflicting.")
        lines.append("  - Treat same-name or near-name documents about the wrong entity as distractor.")
        lines.append("  - The final JSON decision must be consistent with the analysis array.")
        lines.append("")
    lines.append("Answer selection rules:")
    lines.append("  - final_answer should be copied from the evidence whenever possible.")
    lines.append("  - cited quotes must contain the answer phrase or directly justify it.")
    lines.append("  - reject same-name distractors whose title/content does not match the question entity.")
    lines.append("  - for 'What is X\'s occupation?' choose X's occupation/profession, not a birthplace, nationality, date, honorific, or another person's role.")
    lines.append("  - if selected evidence is about the wrong entity, use a better candidate document or mark the answer unsupported.")
    if show_gold:
        lines.append("")
        lines.append(f"(Training-only) Gold answers: {sample.answers}")
    lines.append("")
    if corag_analysis:
        lines.append("Now output ONLY the valid JSON object with final_answer, answer_reasoning, document_analysis, and audit fields.")
    elif hybrid:
        lines.append("Now perform the evidence analysis inside the JSON and output ONLY that JSON object.")
    else:
        lines.append("Now output ONLY the JSON object described in the schema.")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def extract_json_block(text: str) -> Optional[dict]:
    """从自由文本中抽取第一个平衡的 {...} JSON 块。"""
    if not text:
        return None
    # 先尝试直接解析
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        block = text[start:i + 1]
                        try:
                            return json.loads(block)
                        except ValueError:
                            break  # 这个起点不行，找下一个 {
        start = text.find("{", start + 1)
    return None


def _coerce_enum(value, allowed: set, default):
    return value if value in allowed else default


def _sanitize(d: dict) -> dict:
    """Normalize already validated payload fields for LargeAudit."""
    out = dict(d)
    analysis = []
    for item in (d.get("analysis", []) or []) + (d.get("document_analysis", []) or []):
        if not isinstance(item, dict):
            continue
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            continue
        support = str(item.get("support") or "irrelevant")
        if support not in {
            "supporting",
            "partial",
            "irrelevant",
            "distractor",
            "conflicting",
        }:
            support = "irrelevant"
        analysis.append({
            "doc_id": doc_id,
            "extraction": str(item.get("extraction") or ""),
            "reason": str(item.get("reason") or item.get("explanation") or ""),
            "support": support,
        })
    out["_analysis"] = analysis
    out["answer_correctness"] = _coerce_enum(
        d.get("answer_correctness"), AnswerCorrectness.ALL, AnswerCorrectness.UNKNOWN)
    out["support_level"] = _coerce_enum(
        d.get("support_level"), SupportLevel.ALL, SupportLevel.UNSUPPORTED)
    out["failure_type"] = _coerce_enum(
        d.get("failure_type"), FailureType.ALL, FailureType.NONE)
    out["suggested_action"] = _coerce_enum(
        d.get("suggested_action"), RetrievalAction.ALL, RetrievalAction.ANSWER_NOW)
    # used_doc_ids 必须是 int 列表
    ids = []
    for x in d.get("used_doc_ids", []) or []:
        try:
            ids.append(int(x))
        except (ValueError, TypeError):
            continue
    out["used_doc_ids"] = ids
    evidence = []
    for item in d.get("used_evidence", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            continue
        evidence.append({"doc_id": doc_id, "quote": str(item.get("quote") or "")})
    out["used_evidence"] = evidence
    fb = []
    for item in d.get("small_model_feedback", []) or []:
        if isinstance(item, dict) and "label" in item:
            item = dict(item)
            from .schemas import FeedbackLabel
            item["label"] = _coerce_enum(item["label"], FeedbackLabel.ALL, FeedbackLabel.IGNORE)
        fb.append(item)
    out["small_model_feedback"] = fb
    return out


def _audit_schema_error(d: object) -> str | None:
    """Return a compact reason when a parsed JSON value violates the audit schema."""
    if not isinstance(d, dict):
        return "top_level_not_object"
    required = {
        "final_answer",
        "used_doc_ids",
        "used_evidence",
        "answer_correctness",
        "support_level",
        "failure_type",
        "small_model_feedback",
        "suggested_action",
    }
    missing = sorted(required - set(d))
    if missing:
        return "missing_fields:" + ",".join(missing)
    if not isinstance(d.get("final_answer"), str) or not d["final_answer"].strip():
        return "empty_final_answer"
    if not isinstance(d.get("used_doc_ids"), list):
        return "used_doc_ids_not_list"
    if not isinstance(d.get("used_evidence"), list):
        return "used_evidence_not_list"
    if not isinstance(d.get("small_model_feedback"), list):
        return "small_model_feedback_not_list"
    if d.get("answer_correctness") not in AnswerCorrectness.ALL:
        return "invalid_answer_correctness"
    if d.get("support_level") not in SupportLevel.ALL:
        return "invalid_support_level"
    if d.get("failure_type") not in FailureType.ALL:
        return "invalid_failure_type"
    if d.get("suggested_action") not in RetrievalAction.ALL:
        return "invalid_suggested_action"
    for value in d["used_doc_ids"]:
        try:
            int(value)
        except (TypeError, ValueError):
            return "invalid_used_doc_id"
    for item in d["used_evidence"]:
        if not isinstance(item, dict) or "doc_id" not in item or "quote" not in item:
            return "invalid_used_evidence"
        try:
            int(item["doc_id"])
        except (TypeError, ValueError):
            return "invalid_used_evidence_doc_id"
        if not isinstance(item["quote"], str):
            return "invalid_used_evidence_quote"
    for item in d["small_model_feedback"]:
        if not isinstance(item, dict):
            return "invalid_small_model_feedback"
        if item.get("label") not in FeedbackLabel.ALL:
            return "invalid_small_model_feedback_label"
    analysis_fields = []
    if "analysis" in d:
        if not isinstance(d.get("analysis"), list):
            return "analysis_not_list"
        analysis_fields.append("analysis")
    if "document_analysis" in d:
        if not isinstance(d.get("document_analysis"), list):
            return "document_analysis_not_list"
        analysis_fields.append("document_analysis")
    if analysis_fields:
        allowed_support = {
            "supporting",
            "partial",
            "irrelevant",
            "distractor",
            "conflicting",
        }
        for field in analysis_fields:
            for item in d[field]:
                if not isinstance(item, dict):
                    return f"invalid_{field}"
                if "doc_id" not in item:
                    return f"invalid_{field}_doc_id"
                try:
                    int(item["doc_id"])
                except (TypeError, ValueError):
                    return f"invalid_{field}_doc_id"
                if not isinstance(item.get("extraction", ""), str):
                    return f"invalid_{field}_extraction"
                reason_value = item.get("reason", item.get("explanation", ""))
                if not isinstance(reason_value, str):
                    return f"invalid_{field}_reason"
                if item.get("support") not in allowed_support:
                    return f"invalid_{field}_support"
    return None


def fallback_audit(
    sample_id: str,
    round_id: int,
    raw_text: str = "",
    reason: str = "parse_failed",
    raw_json: object | None = None,
) -> LargeAudit:
    """Return an explicitly empty, untrusted audit after parsing/schema failure."""
    return LargeAudit(
        sample_id=sample_id,
        round=round_id,
        final_answer="",
        used_doc_ids=[],
        answer_correctness=AnswerCorrectness.UNKNOWN,
        support_level=SupportLevel.UNSUPPORTED,
        failure_type=FailureType.GENERATION_ERROR,
        suggested_action=RetrievalAction.ASK_AUDITOR,
        audit_metadata={
            "parse_status": "fallback",
            "schema_error": reason,
            "raw_text": (raw_text or "")[:2000],
            **({"raw_json": raw_json} if raw_json is not None else {}),
        },
    )


def parse_audit(text: str, sample_id: str, round_id: int) -> tuple[LargeAudit, bool]:
    """解析大模型输出。返回 (LargeAudit, json_valid)。"""
    block = extract_json_block(text)
    if block is None:
        return fallback_audit(sample_id, round_id, text, reason="json_not_found"), False
    schema_error = _audit_schema_error(block)
    if schema_error:
        return fallback_audit(
            sample_id,
            round_id,
            text,
            reason=schema_error,
            raw_json=block,
        ), False
    block.setdefault("sample_id", sample_id)
    block.setdefault("round", round_id)
    try:
        sanitized = _sanitize(block)
        analysis = sanitized.pop("_analysis", [])
        audit = LargeAudit.from_dict(sanitized)
        audit.audit_metadata = {
            **(audit.audit_metadata or {}),
            "parse_status": "parsed",
            # Generated completion only; prompts/documents are intentionally
            # excluded so CoRAG-style metrics do not count input evidence.
            "raw_text": (text or "")[:8000],
            "raw_json": block,
            **({"evidence_analysis": analysis} if analysis else {}),
        }
        return audit, True
    except (TypeError, ValueError):
        return fallback_audit(
            sample_id,
            round_id,
            text,
            reason="large_audit_validation_failed",
            raw_json=block,
        ), False
