"""Counterfactual Answer Boundary Learning data utilities.

CABL turns a normal QA/RAG replay item into answer-boundary supervision:

    gold answer > plausible counterfactual answer + margin

The module intentionally avoids external NER libraries. It mines conservative
hard negatives from three places already available in EvoCo-RAG replay:

1. the model's own wrong final answer;
2. entities/answer-like spans in selected or candidate evidence;
3. high-scoring distractor documents.

The output is a list of pairwise boundary records consumed by LargeTrainer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .text_utils import exact_presence, normalize_answer


_ENTITY_LIKE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'.-]*|[0-9]{2,4})(?:\s+(?:[A-Z][A-Za-z0-9'.-]*|of|the|and|&|[0-9]{2,4})){0,5}\b"
)


_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

_OCCUPATION_TERMS = {
    "actor", "actress", "artist", "author", "banker", "barrister", "businessman",
    "businessperson", "businesswoman", "cartoonist", "composer", "director", "economist",
    "entrepreneur", "footballer", "illustrator", "journalist", "judge", "lawyer",
    "manager", "mathematician", "musician", "naturalist", "officer", "painter",
    "philanthropist", "politician", "producer", "professor", "scientist", "singer",
    "soldier", "writer",
}

_NATIONALITY_TERMS = {
    "american", "australian", "belgian", "british", "canadian", "chinese", "dutch",
    "english", "french", "german", "icelandic", "indian", "irish", "italian",
    "japanese", "russian", "scottish", "spanish", "swedish", "welsh",
}

_LOCATION_TERMS = {
    "city", "country", "county", "district", "province", "state", "town", "village",
}


def relation_key_for_question(question: str) -> str:
    """Map common RAG question templates to coarse answer relations.

    The mapping is intentionally simple and deterministic so each switch can be
    ablated without introducing another model or dependency.
    """

    q = normalize_answer(question)
    if any(x in q for x in ("occupation", "profession", "work as", "job")):
        return "occupation"
    if q.startswith("when ") or " date" in q or " year" in q:
        return "date"
    if q.startswith("where ") or any(x in q for x in ("birthplace", "born in", "located")):
        return "location"
    if "nationality" in q or "citizen" in q or "country" in q:
        return "nationality"
    if any(x in q for x in ("spouse", "wife", "husband", "father", "mother", "child")):
        return "person"
    return "generic"


def _looks_like_date(answer: str) -> bool:
    norm = normalize_answer(answer)
    if re.search(r"\b\d{3,4}\b", norm):
        return True
    return any(month in norm.split() for month in _MONTHS)


def _looks_like_occupation(answer: str) -> bool:
    answer = _clean_answer(answer)
    norm = normalize_answer(answer)
    words = set(norm.split())
    if words & _OCCUPATION_TERMS:
        return True
    if _looks_like_date(answer):
        return False
    # Many PopQA occupation aliases are lowercase noun phrases (e.g. talent
    # manager). Proper-name-like spans are usually entities, not occupations.
    has_alpha = any(ch.isalpha() for ch in answer)
    return bool(has_alpha and answer == answer.lower() and 1 <= len(norm.split()) <= 4)


def _looks_like_nationality(answer: str) -> bool:
    norm = normalize_answer(answer)
    return bool(set(norm.split()) & _NATIONALITY_TERMS)


def _looks_like_location(answer: str) -> bool:
    answer = _clean_answer(answer)
    norm = normalize_answer(answer)
    if _looks_like_date(answer):
        return False
    if set(norm.split()) & _LOCATION_TERMS:
        return True
    return any(ch.isupper() for ch in answer[:1]) and len(norm.split()) <= 5


def _matches_relation_type(answer: str, relation: str) -> bool:
    if relation == "generic":
        return True
    if relation == "occupation":
        return _looks_like_occupation(answer)
    if relation == "date":
        return _looks_like_date(answer)
    if relation == "nationality":
        return _looks_like_nationality(answer)
    if relation in {"location", "person"}:
        return _looks_like_location(answer)
    return True


def build_relation_answer_pool(experiences: Iterable) -> dict[str, list[str]]:
    """Build relation-aware answer pools from gold answers in the training set."""

    pools: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for exp in experiences or []:
        relation = relation_key_for_question(getattr(exp, "question", ""))
        for answer in getattr(exp, "answers", []) or []:
            clean = _clean_answer(answer)
            norm = normalize_answer(clean)
            if not clean or not norm or norm in seen[relation]:
                continue
            if not _matches_relation_type(clean, relation):
                continue
            pools[relation].append(clean)
            seen[relation].add(norm)
    return dict(pools)


def _make_counterfactual_evidence(evidence_text: str, positive: str, negative: str) -> str:
    """Create a short corrupted-evidence distractor by replacing gold with negative.

    Prefer answer-defining copular contexts ("is/was a/an <answer>") over the
    first surface mention, so we corrupt the answer boundary rather than an
    incidental phrase such as "fellow cartoonist".
    """

    evidence = _short_context(evidence_text, limit=360)
    positive = _clean_answer(positive)
    negative = _clean_answer(negative)
    if not evidence or not positive or not negative:
        return ""
    answer = re.escape(positive)
    copular = re.compile(
        rf"(\b(?:is|was|are|were|be|been|became|become)\s+(?:an?\s+)?)(?:{answer})\b",
        flags=re.IGNORECASE,
    )
    if copular.search(evidence):
        return copular.sub(lambda m: m.group(1) + negative, evidence, count=1)
    pattern = re.compile(answer, flags=re.IGNORECASE)
    if pattern.search(evidence):
        return pattern.sub(negative, evidence, count=1)
    return ""


def _clean_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return text


def _canonical_gold(answers: Iterable[str]) -> str:
    for answer in answers or []:
        answer = _clean_answer(answer)
        if answer:
            return answer
    return ""


def _same_answer(candidate: str, answers: list[str]) -> bool:
    cand = normalize_answer(candidate)
    if not cand:
        return True
    for answer in answers or []:
        gold = normalize_answer(answer)
        if not gold:
            continue
        if cand == gold or cand in gold or gold in cand:
            return True
    return False


def _doc_text(doc: dict) -> str:
    return str(doc.get("text") or doc.get("raw") or doc.get("title") or "")


def _doc_by_id(documents: list[dict], doc_id: int | None) -> dict | None:
    for doc in documents:
        if doc.get("doc_id") == doc_id:
            return doc
    return None


def _short_context(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 4].rstrip() + " ..."


def _question_overlap(question: str, text: str) -> float:
    q_tokens = set(normalize_answer(question).split())
    t_tokens = set(normalize_answer(text).split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _add_candidate(
    candidates: list[dict],
    seen: set[str],
    answer: str,
    *,
    source: str,
    negative_type: str,
    question: str,
    answers: list[str],
    evidence_doc_id: int | None = None,
    evidence_text: str = "",
    priority: float = 0.0,
    min_chars: int = 2,
) -> None:
    answer = _clean_answer(answer)
    norm = normalize_answer(answer)
    if len(answer) < min_chars or not norm or norm in seen:
        return
    if _same_answer(answer, answers):
        return
    # Avoid sentence fragments masquerading as answers.
    if len(answer.split()) > 8:
        return
    seen.add(norm)
    hardness = priority + _question_overlap(question, evidence_text or answer)
    candidates.append({
        "answer": answer,
        "type": negative_type,
        "source": source,
        "hardness": round(float(hardness), 4),
        "evidence_doc_id": evidence_doc_id,
        "evidence": _short_context(evidence_text),
    })


def _candidate_doc_ids(exp) -> list[int]:
    ids: list[int] = []
    contract = exp.contract or {}
    for item in contract.get("selected_evidence", []) or []:
        if item.get("doc_id") is not None:
            ids.append(item["doc_id"])
    for item in contract.get("candidate_docs", []) or []:
        if item.get("doc_id") is not None:
            ids.append(item["doc_id"])
    # Preserve order while removing duplicates.
    out, seen = [], set()
    for doc_id in ids:
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
    return out


def mine_counterfactual_answers(
    exp,
    *,
    max_negatives: int = 3,
    min_negative_chars: int = 2,
    answer_pool: dict[str, list[str]] | None = None,
    use_model_self_error: bool = True,
    use_relation_answer_pool: bool = False,
    use_answer_type_filter: bool = False,
    use_retrieved_distractors: bool = True,
) -> list[dict]:
    """Mine plausible wrong answers for one replay experience.

    The function is conservative by design: it never returns gold aliases or
    empty strings, and it prefers negatives that the current model or retrieved
    context made plausible.
    """

    answers = list(exp.answers or [])
    question = str(exp.question or "")
    relation = relation_key_for_question(question)
    candidates: list[dict] = []
    seen: set[str] = set()

    def type_ok(answer: str) -> bool:
        return (not use_answer_type_filter) or _matches_relation_type(answer, relation)

    # 1) Model self-error: the most valuable negative because it is a real
    # confusion made by the current generator. Keep it first by priority, but
    # still let answer-type filtering remove obvious schema noise.
    verification = exp.verification or {}
    audit = exp.audit or {}
    self_error = audit.get("final_answer", "")
    if (
        use_model_self_error
        and not verification.get("answer_match", False)
        and type_ok(self_error)
    ):
        _add_candidate(
            candidates,
            seen,
            self_error,
            source="model_self_error",
            negative_type="self_error_answer",
            question=question,
            answers=answers,
            priority=1.2,
            min_chars=min_negative_chars,
        )

    # 2) Relation-aware answer pool: same relation and same answer type, but
    # from other samples' gold labels. This creates genuinely confusable
    # negatives such as politician vs actor/lawyer/journalist.
    if use_relation_answer_pool and answer_pool:
        for pooled in answer_pool.get(relation, []) or []:
            if not type_ok(pooled):
                continue
            _add_candidate(
                candidates,
                seen,
                pooled,
                source="relation_answer_pool",
                negative_type=f"same_relation_{relation}",
                question=question,
                answers=answers,
                priority=1.0,
                min_chars=min_negative_chars,
            )
            if len(candidates) >= max_negatives * 3:
                break

    # 3) Evidence distractors: entity-like spans in selected/candidate docs. This
    # remains useful as a fallback, but answer-type filtering prevents trivial
    # negatives like dates/places for occupation questions.
    if use_retrieved_distractors:
        for doc_id in _candidate_doc_ids(exp):
            doc = _doc_by_id(exp.documents, doc_id) or {}
            text = _doc_text(doc)
            if not text:
                continue
            contains_gold = exact_presence(answers, text)
            for match in _ENTITY_LIKE.finditer(text[:1600]):
                span = _clean_answer(match.group(0))
                if not span or not type_ok(span):
                    continue
                _add_candidate(
                    candidates,
                    seen,
                    span,
                    source="retrieved_distractor",
                    negative_type=(
                        "same_context_distractor"
                        if contains_gold else "unsupported_retrieved_entity"
                    ),
                    question=question,
                    answers=answers,
                    evidence_doc_id=doc_id,
                    evidence_text=text,
                    priority=0.7 if contains_gold else 0.45,
                    min_chars=min_negative_chars,
                )
                if len(candidates) >= max_negatives * 3:
                    break
            if len(candidates) >= max_negatives * 3:
                break

    candidates.sort(key=lambda item: item.get("hardness", 0.0), reverse=True)
    return candidates[: max(0, int(max_negatives))]


def build_boundary_pairs(
    exp,
    *,
    max_negatives: int = 3,
    margin: float = 0.5,
    min_negative_chars: int = 2,
    answer_pool: dict[str, list[str]] | None = None,
    use_model_self_error: bool = True,
    use_relation_answer_pool: bool = False,
    use_answer_type_filter: bool = False,
    use_retrieved_distractors: bool = True,
    use_counterfactual_evidence: bool = False,
) -> list[dict]:
    """Convert a replay experience into gold-vs-negative boundary pairs."""

    positive = _canonical_gold(exp.answers)
    if not positive:
        return []
    pairs = []
    for neg in mine_counterfactual_answers(
        exp,
        max_negatives=max_negatives,
        min_negative_chars=min_negative_chars,
        answer_pool=answer_pool,
        use_model_self_error=use_model_self_error,
        use_relation_answer_pool=use_relation_answer_pool,
        use_answer_type_filter=use_answer_type_filter,
        use_retrieved_distractors=use_retrieved_distractors,
    ):
        evidence = neg.get("evidence", "")
        if not evidence and exp.documents:
            first_doc_id = _candidate_doc_ids(exp)[0] if _candidate_doc_ids(exp) else None
            doc = _doc_by_id(exp.documents, first_doc_id) if first_doc_id is not None else None
            evidence = _short_context(_doc_text(doc or {}))
        pair = {
            "sample_id": exp.sample_id,
            "question": exp.question,
            "positive": positive,
            "negative": neg["answer"],
            "negative_type": neg["type"],
            "source": neg["source"],
            "hardness": neg["hardness"],
            "margin": float(margin),
            "evidence_doc_id": neg.get("evidence_doc_id"),
            "evidence": evidence,
        }
        if use_counterfactual_evidence:
            cf_evidence = _make_counterfactual_evidence(evidence, positive, neg["answer"])
            if cf_evidence:
                pair["counterfactual_evidence"] = cf_evidence
        pairs.append(pair)
    return pairs
