"""文本归一化与证据匹配工具。

与原 utils.py 的 normalize_answer / exact_presence 行为保持一致，但在此独立
实现，避免 evoco_rag 核心层依赖 jsonlines 等重模块（开发文档 §13.4：禁止核心
模块硬依赖外部环境）。
"""

from __future__ import annotations

import re
import string


def normalize_answer(s: str) -> str:
    """小写化、去标点、去冠词、规整空白。与 utils.normalize_answer 等价。"""
    s = s or ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(answers: list[str], context: str) -> bool:
    """任一标准答案（归一化后）作为子串出现在 context 中即视为命中。"""
    norm_context = normalize_answer(context)
    for ans in answers:
        normalized = normalize_answer(ans)
        if normalized and normalized in norm_context:
            return True
    return False


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """轻量句子切分，用于第一阶段启发式 span 抽取。"""
    text = (text or "").strip()
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]


_WORD = re.compile(r"[a-z0-9]+")


def _content_tokens(s: str) -> set:
    return set(_WORD.findall(normalize_answer(s)))


def best_evidence_span(question: str, text: str) -> tuple[str, float]:
    """从文档文本中选与问题词汇重叠最高的句子作为启发式证据 span。

    返回 (span_text, overlap_ratio)。overlap_ratio 可用作粗粒度 evidence 置信度。
    """
    sents = split_sentences(text)
    if not sents:
        return (text or "")[:300], 0.0

    q_tokens = _content_tokens(question)
    if not q_tokens:
        return sents[0], 0.0

    best_sent, best_score = sents[0], -1.0
    for sent in sents:
        s_tokens = _content_tokens(sent)
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens) / len(q_tokens)
        if overlap > best_score:
            best_sent, best_score = sent, overlap
    return best_sent, max(best_score, 0.0)
