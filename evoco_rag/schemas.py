"""结构化数据契约（开发文档 §4、§5.1）。

只用标准库 dataclasses + 显式校验，不引入 Pydantic，避免额外依赖。
所有 schema 都提供：
    - from_dict(d): 从 JSON dict 构造，缺关键字段或枚举非法时抛 ValueError；
    - to_dict():    转回可 json.dumps 的 dict；
    - validate():   单独触发一次校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 枚举（开发文档 §4.2 / §4.3 的字段约束）
# ---------------------------------------------------------------------------
class Answerability:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ALL = {HIGH, MEDIUM, LOW}


class RetrievalAction:
    ANSWER_NOW = "answer_now"
    RETRIEVE_MORE = "retrieve_more"
    REWRITE_QUERY = "rewrite_query"
    ASK_AUDITOR = "ask_auditor"
    ALL = {ANSWER_NOW, RETRIEVE_MORE, REWRITE_QUERY, ASK_AUDITOR}


class AnswerCorrectness:
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"
    ALL = {CORRECT, INCORRECT, UNKNOWN}


class SupportLevel:
    FULLY = "fully_supported"
    PARTIALLY = "partially_supported"
    UNSUPPORTED = "unsupported"
    ALL = {FULLY, PARTIALLY, UNSUPPORTED}


class FailureType:
    NONE = "none"
    RETRIEVAL_MISS = "retrieval_miss"
    RERANK_ERROR = "rerank_error"
    ENTITY_CONFUSION = "entity_confusion"
    EVIDENCE_CONFLICT = "evidence_conflict"
    GENERATION_ERROR = "generation_error"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    OVER_RETRIEVAL = "over_retrieval"
    ALL = {
        NONE,
        RETRIEVAL_MISS,
        RERANK_ERROR,
        ENTITY_CONFUSION,
        EVIDENCE_CONFLICT,
        GENERATION_ERROR,
        UNSUPPORTED_ANSWER,
        OVER_RETRIEVAL,
    }


class FeedbackLabel:
    POSITIVE = "positive"
    NEGATIVE = "negative"
    HARD_NEGATIVE = "hard_negative"
    IGNORE = "ignore"
    ALL = {POSITIVE, NEGATIVE, HARD_NEGATIVE, IGNORE}


class AttributionCase:
    BOTH_SUCCESS = "both_success"
    PARAMETRIC_ANSWER_WITHOUT_SUPPORT = "parametric_answer_without_support"
    RETRIEVER_SUCCESS_GENERATOR_FAIL = "retriever_success_generator_fail"
    BOTH_FAIL = "both_fail"
    ALL = {
        BOTH_SUCCESS,
        PARAMETRIC_ANSWER_WITHOUT_SUPPORT,
        RETRIEVER_SUCCESS_GENERATOR_FAIL,
        BOTH_FAIL,
    }


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _check_enum(value: Any, allowed: set, field_name: str) -> None:
    _require(value in allowed, f"非法 {field_name}={value!r}，允许值: {sorted(allowed)}")


# ---------------------------------------------------------------------------
# 输入样本
# ---------------------------------------------------------------------------
@dataclass
class RagSample:
    sample_id: str
    question: str
    answers: list[str]
    documents: list[dict]            # 每个 doc: {doc_id, title, text, raw}
    seed_labels: Optional[list] = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> "RagSample":
        _require(bool(self.sample_id), "RagSample 缺少 sample_id")
        _require(isinstance(self.answers, list), "answers 必须是 list")
        _require(isinstance(self.documents, list), "documents 必须是 list")
        for i, doc in enumerate(self.documents):
            _require("doc_id" in doc, f"documents[{i}] 缺少 doc_id")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "RagSample":
        return cls(
            sample_id=d["sample_id"],
            question=d.get("question", ""),
            answers=d.get("answers", []),
            documents=d.get("documents", []),
            seed_labels=d.get("seed_labels"),
            metadata=d.get("metadata", {}),
        ).validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def doc_by_id(self, doc_id: int) -> Optional[dict]:
        for doc in self.documents:
            if doc.get("doc_id") == doc_id:
                return doc
        return None


# ---------------------------------------------------------------------------
# 证据合约
# ---------------------------------------------------------------------------
@dataclass
class EvidenceItem:
    doc_id: int
    rank: int = 0
    doc_score: float = 0.0
    relevance_confidence: float = 0.0
    evidence_confidence: float = 0.0
    span: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    reason: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceItem":
        _require("doc_id" in d, "EvidenceItem 缺少 doc_id")
        return cls(
            doc_id=d["doc_id"],
            rank=d.get("rank", 0),
            doc_score=d.get("doc_score", 0.0),
            relevance_confidence=d.get("relevance_confidence", 0.0),
            evidence_confidence=d.get("evidence_confidence", 0.0),
            span=d.get("span"),
            span_start=d.get("span_start"),
            span_end=d.get("span_end"),
            reason=d.get("reason", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceContract:
    sample_id: str
    round: int
    question: str
    answerability: str
    retrieval_action: str
    selected_evidence: list[EvidenceItem] = field(default_factory=list)
    candidate_docs: list[dict] = field(default_factory=list)
    uncertainty: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)

    def validate(self) -> "EvidenceContract":
        _require(bool(self.sample_id), "EvidenceContract 缺少 sample_id")
        _check_enum(self.answerability, Answerability.ALL, "answerability")
        _check_enum(self.retrieval_action, RetrievalAction.ALL, "retrieval_action")
        _require(isinstance(self.selected_evidence, list), "selected_evidence 必须是 list")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceContract":
        evidence = [
            e if isinstance(e, EvidenceItem) else EvidenceItem.from_dict(e)
            for e in d.get("selected_evidence", [])
        ]
        return cls(
            sample_id=d["sample_id"],
            round=d.get("round", 0),
            question=d.get("question", ""),
            answerability=d.get("answerability", Answerability.MEDIUM),
            retrieval_action=d.get("retrieval_action", RetrievalAction.ANSWER_NOW),
            selected_evidence=evidence,
            candidate_docs=d.get("candidate_docs", []),
            uncertainty=d.get("uncertainty", {}),
            cost=d.get("cost", {}),
        ).validate()

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "round": self.round,
            "question": self.question,
            "answerability": self.answerability,
            "retrieval_action": self.retrieval_action,
            "selected_evidence": [e.to_dict() for e in self.selected_evidence],
            "candidate_docs": self.candidate_docs,
            "uncertainty": self.uncertainty,
            "cost": self.cost,
        }

    def selected_doc_ids(self) -> list[int]:
        return [e.doc_id for e in self.selected_evidence]

    def candidate_doc_ids(self) -> list[int]:
        return [c.get("doc_id") for c in self.candidate_docs]


# ---------------------------------------------------------------------------
# 大模型审计
# ---------------------------------------------------------------------------
@dataclass
class LargeAudit:
    sample_id: str
    round: int
    final_answer: str
    used_doc_ids: list[int] = field(default_factory=list)
    used_evidence: list[dict] = field(default_factory=list)
    answer_correctness: str = AnswerCorrectness.UNKNOWN
    support_level: str = SupportLevel.UNSUPPORTED
    failure_type: str = FailureType.NONE
    small_model_feedback: list[dict] = field(default_factory=list)
    suggested_action: str = RetrievalAction.ANSWER_NOW
    audit_metadata: dict = field(default_factory=dict)

    def validate(self) -> "LargeAudit":
        _require(bool(self.sample_id), "LargeAudit 缺少 sample_id")
        _check_enum(self.answer_correctness, AnswerCorrectness.ALL, "answer_correctness")
        _check_enum(self.support_level, SupportLevel.ALL, "support_level")
        _check_enum(self.failure_type, FailureType.ALL, "failure_type")
        _check_enum(self.suggested_action, RetrievalAction.ALL, "suggested_action")
        for fb in self.small_model_feedback:
            if "label" in fb:
                _check_enum(fb["label"], FeedbackLabel.ALL, "small_model_feedback.label")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "LargeAudit":
        return cls(
            sample_id=d["sample_id"],
            round=d.get("round", 0),
            final_answer=d.get("final_answer", ""),
            used_doc_ids=d.get("used_doc_ids", []),
            used_evidence=d.get("used_evidence", []),
            answer_correctness=d.get("answer_correctness", AnswerCorrectness.UNKNOWN),
            support_level=d.get("support_level", SupportLevel.UNSUPPORTED),
            failure_type=d.get("failure_type", FailureType.NONE),
            small_model_feedback=d.get("small_model_feedback", []),
            suggested_action=d.get("suggested_action", RetrievalAction.ANSWER_NOW),
            audit_metadata=d.get("audit_metadata", {}),
        ).validate()

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 规则验证
# ---------------------------------------------------------------------------
@dataclass
class RuleVerification:
    sample_id: str
    answer_match: bool = False
    cited_doc_contains_answer: bool = False
    used_doc_in_selected_evidence: bool = False
    support_rule_passed: bool = False
    json_valid: bool = True
    audit_trust_weight: float = 0.0
    trust_components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "RuleVerification":
        return cls(
            sample_id=d["sample_id"],
            answer_match=d.get("answer_match", False),
            cited_doc_contains_answer=d.get("cited_doc_contains_answer", False),
            used_doc_in_selected_evidence=d.get("used_doc_in_selected_evidence", False),
            support_rule_passed=d.get("support_rule_passed", False),
            json_valid=d.get("json_valid", True),
            audit_trust_weight=d.get("audit_trust_weight", 0.0),
            trust_components=d.get("trust_components", {}),
            notes=d.get("notes", []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 奖励与经验
# ---------------------------------------------------------------------------
@dataclass
class RewardBreakdown:
    answer_reward: float = 0.0
    support_reward: float = 0.0
    citation_reward: float = 0.0
    calibration_reward: float = 0.0
    action_cost_penalty: float = 0.0
    cost_penalty: float = 0.0
    total_reward: float = 0.0
    attribution_case: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardBreakdown":
        values = {k: d.get(k, 0.0) for k in (
            "answer_reward", "support_reward", "citation_reward",
            "calibration_reward", "action_cost_penalty", "cost_penalty", "total_reward",
        )}
        values["attribution_case"] = d.get("attribution_case", "")
        return cls(**values)


@dataclass
class ReplayExperience:
    sample_id: str
    round: int
    question: str
    answers: list[str]
    documents: list[dict]
    contract: dict
    audit: dict
    verification: dict
    rewards: dict
    training_targets: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayExperience":
        _require("sample_id" in d, "ReplayExperience 缺少 sample_id")
        return cls(
            sample_id=d["sample_id"],
            round=d.get("round", 0),
            question=d.get("question", ""),
            answers=d.get("answers", []),
            documents=d.get("documents", []),
            contract=d.get("contract", {}),
            audit=d.get("audit", {}),
            verification=d.get("verification", {}),
            rewards=d.get("rewards", {}),
            training_targets=d.get("training_targets", {}),
        )
