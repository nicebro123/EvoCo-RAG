"""Protocol-safe evidence expansion utilities.

The default backend stays inside each sample's existing candidate documents. It
can expand top-k to a larger window when the first contract looks risky, but it
does not query an external corpus/index. This preserves comparability with
CoRAG-style candidate-pool evaluation while leaving a clean hook for future
corpus-backed retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import build_contract
from .schemas import Answerability, RetrievalAction


SUPPORTED_BACKENDS = {"none", "sample_internal"}


@dataclass(frozen=True)
class EvidenceExpansionRuntime:
    enabled: bool = False
    backend: str = "sample_internal"
    trigger_mode: str = "risk"  # risk | always
    max_expanded_docs: int = 5
    min_top_confidence: float = 0.55
    min_margin: float = 0.08
    expand_on_entity_ambiguity: bool = True
    expand_on_missing_relation: bool = True


def _runtime_from_config(runtime: EvidenceExpansionRuntime | Any) -> EvidenceExpansionRuntime:
    if isinstance(runtime, EvidenceExpansionRuntime):
        return runtime
    values = {
        field: getattr(runtime, field)
        for field in EvidenceExpansionRuntime.__dataclass_fields__
        if hasattr(runtime, field)
    }
    return EvidenceExpansionRuntime(**values)


def _contract_confidences(contract) -> list[float]:
    return [
        float(item.relevance_confidence)
        for item in getattr(contract, "selected_evidence", []) or []
    ]


def expansion_risk_reasons(contract, runtime: EvidenceExpansionRuntime | Any) -> list[str]:
    rt = _runtime_from_config(runtime)
    if not rt.enabled or rt.backend == "none":
        return []
    if rt.backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported evidence expansion backend={rt.backend!r}; "
            f"allowed={sorted(SUPPORTED_BACKENDS)}"
        )
    if rt.trigger_mode == "always":
        return ["always"]
    if rt.trigger_mode != "risk":
        raise ValueError("evidence expansion trigger_mode must be 'risk' or 'always'")

    uncertainty = getattr(contract, "uncertainty", {}) or {}
    confidences = _contract_confidences(contract)
    top_conf = confidences[0] if confidences else 0.0
    margin = abs(confidences[0] - confidences[1]) if len(confidences) >= 2 else top_conf
    reasons = []
    if getattr(contract, "answerability", "") == Answerability.LOW:
        reasons.append("low_answerability")
    if top_conf < rt.min_top_confidence:
        reasons.append("low_top_confidence")
    if len(confidences) >= 2 and margin < rt.min_margin:
        reasons.append("small_top_margin")
    if rt.expand_on_entity_ambiguity and uncertainty.get("entity_ambiguity"):
        reasons.append("entity_ambiguity")
    if rt.expand_on_missing_relation and uncertainty.get("missing_relation"):
        reasons.append("missing_relation")
    return reasons


def maybe_expand_contract(
    sample,
    ranked_docs: list[dict],
    contract,
    *,
    runtime: EvidenceExpansionRuntime | Any,
    round_id: int,
    high_conf_threshold: float,
    answer_now_margin: float,
    max_selected_docs: int,
    action_mode: str,
    policy_action: str | None = None,
    policy_action_confidence: float | None = None,
    policy_action_min_conf: float = 0.45,
):
    """Expand an evidence contract within the existing sample candidate pool."""

    rt = _runtime_from_config(runtime)
    reasons = expansion_risk_reasons(contract, rt)
    if not reasons:
        return contract

    current_top_k = len(getattr(contract, "candidate_docs", []) or [])
    expanded_top_k = min(len(ranked_docs), max(current_top_k + 1, int(rt.max_expanded_docs)))
    if expanded_top_k <= current_top_k:
        contract.uncertainty.setdefault("evidence_expansion", {
            "triggered": False,
            "backend": rt.backend,
            "reason": reasons,
            "blocked": "no_additional_sample_documents",
            "initial_top_k": current_top_k,
            "expanded_top_k": current_top_k,
        })
        return contract

    expanded = build_contract(
        sample,
        ranked_docs,
        round_id=round_id,
        top_k=expanded_top_k,
        high_conf_threshold=high_conf_threshold,
        answer_now_margin=answer_now_margin,
        max_selected_docs=max(max_selected_docs, expanded_top_k),
        num_retrieval_rounds=2,
        action_mode=action_mode,
        policy_action=policy_action,
        policy_action_confidence=policy_action_confidence,
        policy_action_min_conf=policy_action_min_conf,
    )
    prior_uncertainty = getattr(contract, "uncertainty", {}) or {}
    expanded.uncertainty.update(prior_uncertainty)
    expanded.uncertainty["evidence_expansion"] = {
        "triggered": True,
        "backend": rt.backend,
        "reason": reasons,
        "initial_top_k": current_top_k,
        "expanded_top_k": expanded_top_k,
    }
    expanded.retrieval_action = RetrievalAction.RETRIEVE_MORE
    expanded.cost["num_retrieval_rounds"] = max(2, int(expanded.cost.get("num_retrieval_rounds", 1)))
    expanded.cost["evidence_expansion_backend"] = rt.backend
    return expanded
