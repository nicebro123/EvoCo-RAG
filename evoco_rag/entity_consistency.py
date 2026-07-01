"""Entity-relation consistency reranking for same-name RAG evidence attribution.

This module is deliberately lightweight and protocol-safe: it only reorders the
candidate documents already present inside one sample. It does not query an
external corpus and does not inspect gold answers. The goal is to reduce the
PopQA-style failure mode where the reranker selects a plausible same-name or
near-name distractor instead of the document whose local evidence matches both
the question entity and the question relation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .cabl import relation_key_for_question
from .text_utils import normalize_answer, split_sentences


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "how",
    "did", "does", "do", "has", "have", "had", "occupation", "profession",
    "job", "work", "birth", "place", "date", "nationality", "country",
    "located", "born",
}

_RELATION_KEYWORDS = {
    "occupation": {
        "occupation", "profession", "job", "work", "worked", "career",
        "actor", "actress", "artist", "author", "banker", "barrister",
        "businessman", "businessperson", "cartoonist", "composer", "director",
        "economist", "entrepreneur", "footballer", "journalist", "judge",
        "lawyer", "manager", "mathematician", "musician", "naturalist",
        "officer", "painter", "philanthropist", "politician", "producer",
        "professor", "scientist", "singer", "soldier", "writer",
    },
    "date": {
        "date", "year", "born", "birth", "died", "death", "founded", "established",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    },
    "location": {
        "where", "birthplace", "birth", "born", "located", "location", "city",
        "country", "county", "state", "province", "district", "region", "town",
        "village", "municipality", "island", "capital",
    },
    "nationality": {
        "nationality", "citizen", "country", "american", "australian", "british",
        "canadian", "chinese", "dutch", "english", "french", "german", "indian",
        "irish", "italian", "japanese", "russian", "scottish", "spanish", "swedish",
    },
    "person": {
        "spouse", "wife", "husband", "father", "mother", "child", "children", "son",
        "daughter", "brother", "sister", "parent", "partner",
    },
}


@dataclass(frozen=True)
class EntityConsistencyConfig:
    enabled: bool = False
    weight: float = 0.35
    min_entity_overlap: float = 0.34
    exact_title_bonus: float = 0.35
    missing_entity_penalty: float = 0.15
    parenthetical_penalty: float = 0.08
    max_boost: float = 0.75
    # EAEV-lite additions: the score is still bounded by max_boost and uses
    # only sample-internal evidence, but it distinguishes identity, relation,
    # and local support instead of trusting title overlap alone.
    relation_weight: float = 0.25
    local_support_weight: float = 0.25
    same_name_distractor_penalty: float = 0.18


def question_entity_hint(question: str) -> str:
    """Extract a conservative surface entity hint from common PopQA questions."""

    q = str(question or "").strip()
    patterns = [
        r"(?:what|who)\s+(?:is|was|are|were)\s+(.+?)(?:'s|’s)\s+",
        r"(?:what|who)\s+(?:is|was|are|were)\s+(.+?)\?",
        r"(?:when|where)\s+(?:was|is|were|are)\s+(.+?)(?:\s+born|\s+located|\?)",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.I)
        if m:
            return m.group(1).strip(" ?.,;:\"'")
    m = re.search(
        r"(?:is|was|are|were)\s+(.+?)(?:'s|’s|\s+born|\s+located|\s+occupation|\s+profession|\?)",
        q,
        flags=re.I,
    )
    if m:
        return m.group(1).strip(" ?.,;:\"'")
    return q


def _tokens(text: str) -> set[str]:
    return {
        tok for tok in normalize_answer(text).split()
        if tok and tok not in _STOPWORDS and not tok.isdigit()
    }


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    if not left_set:
        return 0.0
    return len(left_set & set(right)) / len(left_set)


def _strip_parenthetical(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", str(title or "")).strip()


def _doc_text(doc: dict) -> str:
    return str(doc.get("text") or doc.get("raw") or "")


def relation_keywords(relation: str) -> set[str]:
    return set(_RELATION_KEYWORDS.get(relation, set()))


def _relation_score(question: str, doc_text: str, relation: str) -> float:
    if relation == "generic":
        return 0.0
    keywords = relation_keywords(relation)
    if not keywords:
        return 0.0
    doc_tokens = _tokens(doc_text)
    question_tokens = _tokens(question)
    hits = keywords & doc_tokens
    if not hits:
        return 0.0
    q_hits = keywords & question_tokens
    denom = max(1, min(4, len(q_hits) or 2))
    return min(1.0, len(hits) / denom)


def _best_local_support(question: str, entity_tokens: set[str], doc_text: str, relation: str) -> tuple[float, str]:
    relation_terms = relation_keywords(relation)
    question_tokens = _tokens(question)
    best_score = 0.0
    best_sentence = ""
    for sent in split_sentences(doc_text):
        sent_tokens = _tokens(sent)
        if not sent_tokens:
            continue
        entity_hit = _overlap(entity_tokens, sent_tokens) if entity_tokens else 0.0
        relation_hit = 0.0 if relation == "generic" else min(1.0, len(sent_tokens & relation_terms) / 2.0)
        question_hit = _overlap(question_tokens, sent_tokens) if question_tokens else 0.0
        # Treat local support as joint support. Mentioning the entity alone is
        # not enough for relation-specific questions, because same-name pages
        # often mention the right name but answer a different relation.
        if relation != "generic" and relation_hit <= 0.0:
            score = 0.10 * question_hit
        else:
            score = 0.45 * entity_hit + 0.45 * relation_hit + 0.10 * question_hit
        if score > best_score:
            best_score = score
            best_sentence = sent
    return min(1.0, best_score), best_sentence[:240]


def entity_consistency_features(sample, doc: dict) -> dict:
    question = getattr(sample, "question", "")
    entity = question_entity_hint(question)
    entity_tokens = _tokens(entity)
    title = str(doc.get("title") or "")
    clean_title = _strip_parenthetical(title)
    title_tokens = _tokens(clean_title or title)
    full_title_tokens = _tokens(title)
    doc_text = _doc_text(doc)
    relation = relation_key_for_question(question)

    title_overlap = _overlap(entity_tokens, title_tokens or full_title_tokens)
    body_entity_overlap = _overlap(entity_tokens, _tokens(doc_text[:1200]))
    entity_overlap = max(title_overlap, body_entity_overlap)
    exact_title = bool(entity_tokens) and normalize_answer(entity) == normalize_answer(clean_title)
    has_parenthetical = "(" in title and ")" in title
    missing_entity = bool(entity_tokens) and entity_overlap < 1e-9
    rel_score = _relation_score(question, title + "\n" + doc_text[:1600], relation)
    local_score, local_window = _best_local_support(question, entity_tokens, doc_text[:2200], relation)

    same_name_distractor = bool(
        entity_tokens
        and title_overlap >= 0.5
        and (
            has_parenthetical
            or (not exact_title and body_entity_overlap < 0.5)
            or (relation != "generic" and rel_score == 0.0 and local_score < 0.35)
        )
    )
    entity_relation_consistent = bool(entity_overlap >= 0.5 and (relation == "generic" or rel_score > 0.0 or local_score >= 0.45))

    return {
        "question_entity": entity,
        "relation": relation,
        "entity_overlap": round(float(entity_overlap), 4),
        "title_entity_overlap": round(float(title_overlap), 4),
        "body_entity_overlap": round(float(body_entity_overlap), 4),
        "relation_score": round(float(rel_score), 4),
        "local_support_score": round(float(local_score), 4),
        "local_support_window": local_window,
        "exact_title_match": exact_title,
        "missing_entity": missing_entity,
        "has_parenthetical_title": has_parenthetical,
        "same_name_distractor": same_name_distractor,
        "entity_relation_consistent": entity_relation_consistent,
    }


def entity_consistency_score(sample, doc: dict, cfg: EntityConsistencyConfig) -> tuple[float, dict]:
    features = entity_consistency_features(sample, doc)
    overlap = float(features["entity_overlap"])
    relation = str(features.get("relation") or "generic")
    rel_score = float(features.get("relation_score", 0.0))
    local_score = float(features.get("local_support_score", 0.0))

    score = cfg.weight * overlap
    if overlap >= cfg.min_entity_overlap:
        score += cfg.weight * 0.25
    if features["exact_title_match"]:
        score += cfg.exact_title_bonus
    if relation != "generic":
        score += cfg.relation_weight * rel_score
        score += cfg.local_support_weight * local_score
    if features["missing_entity"]:
        score -= cfg.missing_entity_penalty
    if features["has_parenthetical_title"] and not features["exact_title_match"]:
        score -= cfg.parenthetical_penalty
    if features["same_name_distractor"]:
        score -= cfg.same_name_distractor_penalty

    score = max(-cfg.max_boost, min(cfg.max_boost, score))
    features["entity_consistency_score"] = round(float(score), 4)
    return float(score), features


def apply_entity_consistency_rerank(
    sample,
    ranked_docs: list[dict],
    config: EntityConsistencyConfig | None = None,
) -> list[dict]:
    """Return ranked docs after adding an entity-relation consistency boost.

    The input order is preserved as a deterministic tie breaker. Each returned
    item keeps the original score in ``pre_entity_score`` and adds transparent
    metadata for analysis.
    """

    cfg = config or EntityConsistencyConfig()
    if not cfg.enabled or not ranked_docs:
        return ranked_docs
    out = []
    for order, item in enumerate(ranked_docs):
        doc = getattr(sample, "doc_by_id", lambda _id: None)(item.get("doc_id")) or {}
        boost, features = entity_consistency_score(sample, doc, cfg)
        new_item = dict(item)
        old_score = float(new_item.get("score", 0.0))
        new_item["pre_entity_score"] = old_score
        new_item["entity_consistency_boost"] = boost
        new_item["entity_consistency"] = features
        new_item["score"] = old_score + boost
        new_item["_entity_original_order"] = order
        out.append(new_item)
    out.sort(
        key=lambda d: (
            float(d.get("score", 0.0)),
            float(d.get("pre_entity_score", d.get("base_score", 0.0))),
            -int(d.get("_entity_original_order", 0)),
        ),
        reverse=True,
    )
    for item in out:
        item.pop("_entity_original_order", None)
    return out
