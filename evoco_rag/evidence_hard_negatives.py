"""Evidence-side hard-negative mining for EvoCo-RAG.

The existing CABL module mines answer counterfactuals for the generator. This
module mines document/evidence hard negatives for the reranker: documents that
do not contain a gold answer but look deceptively close to the question entity
or to the model's wrong answer. The implementation is deterministic and uses
only the candidate documents already present in a sample, so it does not change
the CoRAG-style candidate-pool protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .cabl import relation_key_for_question
from .entity_consistency import (
    entity_consistency_features,
    question_entity_hint,
    relation_keywords,
)
from .text_utils import exact_presence, normalize_answer, split_sentences


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "why", "how", "did", "does",
    "do", "has", "have", "had", "occupation", "profession", "job", "work",
    "birth", "place", "date", "nationality", "country", "located",
}


@dataclass(frozen=True)
class HardNegativeConfig:
    enabled: bool = True
    max_per_sample: int = 3
    min_title_overlap: float = 0.34
    min_question_overlap: float = 0.18
    wrong_answer_bonus: float = 0.8
    title_overlap_weight: float = 1.0
    question_overlap_weight: float = 0.6
    entity_confusion_bonus: float = 0.7
    relation_mismatch_bonus: float = 0.5
    selected_unsupported_bonus: float = 0.3


def _tokens(text: str) -> set[str]:
    return {
        tok for tok in normalize_answer(text).split()
        if tok and tok not in _STOPWORDS and not tok.isdigit()
    }


def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    if not left:
        return 0.0
    return len(left & set(b)) / len(left)


def _doc_text(doc: dict) -> str:
    return str(doc.get("text") or doc.get("raw") or "")


def _relation_window_score(text: str, relation: str) -> float:
    if relation == "generic":
        return 0.0
    terms = relation_keywords(relation)
    if not terms:
        return 0.0
    best = 0.0
    for sent in split_sentences(text[:2200]):
        toks = _tokens(sent)
        if not toks:
            continue
        best = max(best, min(1.0, len(toks & terms) / 2.0))
    return best


def _entity_relation_hardness(
    sample,
    doc: dict,
    relation: str,
    cfg: HardNegativeConfig,
) -> tuple[list[str], float, dict]:
    features = entity_consistency_features(sample, doc)
    title_overlap = float(features.get("title_entity_overlap", features.get("entity_overlap", 0.0)))
    relation_score = float(features.get("relation_score", 0.0))
    local_support = float(features.get("local_support_score", 0.0))
    same_name_distractor = bool(features.get("same_name_distractor"))
    reasons = []
    bonus = 0.0
    if same_name_distractor:
        reasons.append("same_name_entity_distractor")
        bonus += cfg.entity_confusion_bonus
    if relation != "generic" and title_overlap >= cfg.min_title_overlap and relation_score <= 0.0:
        reasons.append("same_name_relation_mismatch")
        bonus += cfg.relation_mismatch_bonus
    if relation != "generic" and title_overlap >= 0.34 and local_support < 0.35:
        reasons.append("weak_local_relation_support")
        bonus += 0.25
    return reasons, bonus, features


def _candidate_rank_map(candidate_doc_ids: Iterable[int] | None) -> dict[int, int]:
    out = {}
    for i, raw in enumerate(candidate_doc_ids or []):
        try:
            out[int(raw)] = i
        except (TypeError, ValueError):
            continue
    return out


def mine_evidence_hard_negatives(
    sample,
    *,
    positive_doc_ids: Iterable[int] | None = None,
    candidate_doc_ids: Iterable[int] | None = None,
    selected_doc_ids: Iterable[int] | None = None,
    model_wrong_answer: str = "",
    config: HardNegativeConfig | None = None,
) -> list[dict]:
    """Return ranked hard-negative document records for one sample.

    A hard negative is still a negative: it must not contain any gold alias and
    must not be listed as a positive document. Its hardness comes from surface
    and semantic proximity to the question: title/entity overlap, question-text
    overlap, and whether it appears to be the source of the model's wrong
    answer. This targets same-name / same-title-family failures without adding
    an external corpus.
    """

    cfg = config or HardNegativeConfig()
    if not cfg.enabled or cfg.max_per_sample <= 0:
        return []

    positive = {int(x) for x in (positive_doc_ids or []) if x is not None}
    selected = {int(x) for x in (selected_doc_ids or []) if x is not None}
    candidate_rank = _candidate_rank_map(candidate_doc_ids)
    entity_tokens = _tokens(question_entity_hint(getattr(sample, "question", "")))
    question_tokens = _tokens(getattr(sample, "question", ""))
    wrong_answer = str(model_wrong_answer or "").strip()
    relation = relation_key_for_question(getattr(sample, "question", ""))

    records: list[dict] = []
    for doc in getattr(sample, "documents", []) or []:
        try:
            doc_id = int(doc.get("doc_id"))
        except (TypeError, ValueError):
            continue
        if doc_id in positive:
            continue
        text = _doc_text(doc)
        if exact_presence(getattr(sample, "answers", []), text):
            continue

        title = str(doc.get("title") or "")
        title_overlap = _overlap(entity_tokens or question_tokens, _tokens(title))
        question_overlap = _overlap(question_tokens, _tokens(title + " " + text[:1200]))
        relation_reasons, entity_relation_bonus, entity_features = _entity_relation_hardness(
            sample, doc, relation, cfg)
        relation_window = _relation_window_score(title + "\n" + text, relation)
        wrong_answer_hit = bool(
            wrong_answer
            and not exact_presence(getattr(sample, "answers", []), wrong_answer)
            and exact_presence([wrong_answer], title + "\n" + text)
        )

        reasons = []
        if title_overlap >= cfg.min_title_overlap:
            reasons.append("title_entity_overlap")
        if question_overlap >= cfg.min_question_overlap:
            reasons.append("question_text_overlap")
        reasons.extend(reason for reason in relation_reasons if reason not in reasons)
        if wrong_answer_hit:
            reasons.append("model_wrong_answer_source")
        if doc_id in selected:
            reasons.append("selected_unsupported_evidence")
        if not reasons:
            continue

        score = (
            cfg.title_overlap_weight * title_overlap
            + cfg.question_overlap_weight * question_overlap
            + (cfg.wrong_answer_bonus if wrong_answer_hit else 0.0)
            + entity_relation_bonus
            + (cfg.selected_unsupported_bonus if doc_id in selected else 0.0)
            + (0.08 if doc_id in candidate_rank else 0.0)
        )
        records.append({
            "doc_id": doc_id,
            "hardness": round(float(score), 4),
            "reasons": reasons,
            "title_overlap": round(float(title_overlap), 4),
            "question_overlap": round(float(question_overlap), 4),
            "relation": relation,
            "relation_window_score": round(float(relation_window), 4),
            "entity_relation": {
                "entity_overlap": entity_features.get("entity_overlap"),
                "relation_score": entity_features.get("relation_score"),
                "local_support_score": entity_features.get("local_support_score"),
                "same_name_distractor": entity_features.get("same_name_distractor"),
            },
            "candidate_rank": candidate_rank.get(doc_id),
        })

    records.sort(
        key=lambda item: (
            item.get("hardness", 0.0),
            -1 if item.get("candidate_rank") is None else -int(item.get("candidate_rank") or 0),
        ),
        reverse=True,
    )
    return records[: cfg.max_per_sample]
