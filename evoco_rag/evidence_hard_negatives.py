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
from .text_utils import exact_presence, normalize_answer


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


def _question_entity_hint(question: str) -> str:
    """Extract a conservative surface entity hint from common PopQA questions."""

    q = str(question or "").strip()
    # PopQA templates often look like: "What is Ada Lovelace's occupation?".
    m = re.search(
        r"(?:is|was|are|were)\s+(.+?)(?:'s|’s|\s+born|\s+located|\s+occupation|\s+profession|\?)",
        q,
        flags=re.I,
    )
    if m:
        return m.group(1).strip(" ?.,;:\"'")
    return q


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
    entity_tokens = _tokens(_question_entity_hint(getattr(sample, "question", "")))
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
            + (0.15 if doc_id in selected else 0.0)
            + (0.08 if doc_id in candidate_rank else 0.0)
        )
        records.append({
            "doc_id": doc_id,
            "hardness": round(float(score), 4),
            "reasons": reasons,
            "title_overlap": round(float(title_overlap), 4),
            "question_overlap": round(float(question_overlap), 4),
            "relation": relation,
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
