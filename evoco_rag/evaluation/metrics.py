"""细粒度指标（开发文档 §7.1）。

从一批 ReplayExperience（或等价 dict）离线计算答案 / 检索 / 证据 / 策略成本 /
校准指标。纯 Python + numpy，可单测。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..text_utils import exact_presence


def _mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else 0.0


def _doc_text(documents, doc_id):
    for d in documents:
        if d.get("doc_id") == doc_id:
            return d.get("text") or d.get("raw") or ""
    return ""


def _relevant_ids(documents, answers):
    return {d.get("doc_id") for d in documents
            if exact_presence(answers, d.get("text") or d.get("raw") or "")}


def compute_metrics(experiences: Iterable) -> dict:
    exps = [e if isinstance(e, dict) else e.to_dict() for e in experiences]
    if not exps:
        return {}

    answer_match, support, citation, unsupported = [], [], [], []
    recall_at_k, mrr, answer_in_topk = [], [], []
    selected_counts, audit_calls, cost_penalties = [], [], []
    used_precisions = []
    confidences, outcomes = [], []

    for e in exps:
        v = e.get("verification", {})
        c = e.get("contract", {})
        a = e.get("audit", {})
        r = e.get("rewards", {})
        docs = e.get("documents", [])
        answers = e.get("answers", [])

        am = bool(v.get("answer_match"))
        sp = bool(v.get("support_rule_passed"))
        answer_match.append(am)
        support.append(sp)
        citation.append(bool(v.get("cited_doc_contains_answer")))
        unsupported.append(1.0 if (am and not sp) else 0.0)

        # 检索：candidate_docs 已按分数降序（top-k）
        ranked_ids = [cd.get("doc_id") for cd in c.get("candidate_docs", [])]
        relevant = _relevant_ids(docs, answers)
        hit = any(rid in relevant for rid in ranked_ids)
        answer_in_topk.append(1.0 if hit else 0.0)
        recall_at_k.append(1.0 if hit else 0.0)
        rr = 0.0
        for rank, rid in enumerate(ranked_ids, start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        mrr.append(rr)

        # 证据/引用：used_doc_precision = 引用文档中真正含答案的比例
        used = a.get("used_doc_ids", []) or []
        if used:
            prec = _mean([1.0 if exact_presence(answers, _doc_text(docs, u)) else 0.0 for u in used])
            used_precisions.append(prec)

        # 成本
        cost = c.get("cost", {})
        selected_counts.append(cost.get("num_selected_docs", 0))
        audit_calls.append(1.0 if (a.get("final_answer") or used) else 0.0)
        cost_penalties.append(r.get("cost_penalty", 0.0))

        # 校准：top 选中证据的 relevance_confidence vs 命中
        sel = c.get("selected_evidence", [])
        if sel:
            confidences.append(max(s.get("relevance_confidence", 0.0) for s in sel))
            outcomes.append(1.0 if am else 0.0)

    num = len(exps)
    num_correct = sum(answer_match)

    metrics = {
        "num_examples": num,
        # 答案
        "accuracy": 100.0 * _mean(answer_match),
        # 检索
        "recall_at_k": _mean(recall_at_k),
        "mrr": _mean(mrr),
        "answer_in_topk_context_rate": _mean(answer_in_topk),
        # 证据
        "evidence_support_rate": _mean(support),
        "citation_correctness": _mean(citation),
        "used_doc_precision": _mean(used_precisions) if used_precisions else 0.0,
        "unsupported_answer_rate": _mean(unsupported),
        # 策略成本
        "avg_selected_docs": _mean(selected_counts),
        "audit_call_rate": _mean(audit_calls),
        "cost_per_correct_answer": (sum(cost_penalties) / num_correct) if num_correct else None,
        # 校准
        "confidence_success_correlation": _corr(confidences, outcomes),
        "ece": _ece(confidences, outcomes),
    }
    return metrics


def _corr(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _ece(confidences, outcomes, n_bins: int = 10):
    if not confidences:
        return None
    conf = np.asarray(confidences)
    acc = np.asarray(outcomes)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = acc[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
