from conftest import make_audit, make_contract, make_sample

from evoco_rag.evaluation.metrics import compute_metrics
from evoco_rag.rewards import build_training_targets, compute_decomposed_reward
from evoco_rag.schemas import ReplayExperience
from evoco_rag.verifier import verify


def _exp(selected_ids, final_answer):
    sample = make_sample()
    contract = make_contract(selected_doc_ids=selected_ids)
    audit = make_audit(final_answer=final_answer, used_doc_ids=selected_ids)
    v = verify(sample, contract, audit)
    r = compute_decomposed_reward(sample, contract, audit, v)
    t = build_training_targets(sample, contract, audit, v, r)
    return ReplayExperience(
        sample_id=sample.sample_id, round=0, question=sample.question,
        answers=sample.answers, documents=sample.documents,
        contract=contract.to_dict(), audit=audit.to_dict(),
        verification=v.to_dict(), rewards=r.to_dict(), training_targets=t)


def test_metrics_keys_and_accuracy():
    exps = [_exp([0], "politician"), _exp([1], "banker")]
    m = compute_metrics(exps)
    for key in ("strict_accuracy", "accuracy", "corag_style_accuracy",
                "strict_vs_corag_style_accuracy_gap",
                "recall_at_k", "mrr", "evidence_support_rate",
                "citation_correctness", "unsupported_answer_rate",
                "evidence_quote_support_rate",
                "avg_selected_docs", "generator_call_rate", "audit_call_rate",
                "audit_nonempty_output_rate", "avg_generation_candidates",
                "empty_answer_rate", "unfulfilled_action_rate",
                "attribution_case_distribution", "wrong_retriever_reward_rate",
                "audit_json_valid_rate", "audit_trust_weight_mean",
                "evolution_failure_mode_distribution", "evolution_relation_distribution",
                "evolution_target_module_distribution",
                "audit_schema_valid_rate", "audit_parse_status_distribution",
                "audit_schema_error_distribution",
                "avg_action_cost_penalty", "avg_total_cost_penalty",
                "accuracy_cost_pareto_point"):
        assert key in m
    assert m["num_examples"] == 2
    assert m["evaluation_protocol_version"] == 3
    assert m["primary_accuracy_metric"] == "strict_accuracy"
    assert m["strict_accuracy"] == 50.0  # 一对一错
    assert m["accuracy"] == 50.0  # backward-compatible alias
    assert m["corag_style_accuracy"] == 50.0
    assert "avg_total_cost_penalty" in m["accuracy_cost_pareto_point"]
    assert "strict_accuracy" in m["accuracy_cost_pareto_point"]
    assert "corag_style_accuracy" in m["accuracy_cost_pareto_point"]
    assert m["evolution_failure_mode_distribution"] == {"success": 1, "rerank_miss": 1}
    assert m["evolution_relation_distribution"] == {"occupation": 2}
    assert m["evolution_target_module_distribution"] == {"none": 1, "small": 1}


def test_corag_style_accuracy_uses_generated_raw_text_not_documents():
    exp = _exp([0], "architect")
    exp.audit["audit_metadata"] = {
        "raw_text": "Document analysis says the person was a politician, but final answer is architect."
    }
    m = compute_metrics([exp])
    assert m["accuracy"] == 0.0
    assert m["corag_style_accuracy"] == 100.0
    assert m["strict_vs_corag_style_accuracy_gap"] == 100.0


def test_corag_style_accuracy_does_not_count_input_documents():
    exp = _exp([0], "architect")
    exp.audit["audit_metadata"] = {"raw_text": "final answer is architect"}
    m = compute_metrics([exp])
    assert m["accuracy"] == 0.0
    assert m["corag_style_accuracy"] == 0.0



def test_unsupported_answer_rate():
    # 答案对但证据不支持 → unsupported
    exps = [_exp([1], "politician")]
    m = compute_metrics(exps)
    assert m["unsupported_answer_rate"] == 1.0
    assert m["evidence_support_rate"] == 0.0
    assert m["wrong_retriever_reward_rate"] == 1.0
    assert m["attribution_case_distribution"]["parametric_answer_without_support"] == 1


def test_empty_metrics():
    assert compute_metrics([]) == {}


def test_execution_metrics_use_audit_metadata_not_nonempty_answer():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    audit.audit_metadata = {
        "generator_called": True,
        "generation_candidate_count": 3,
        "extra_audit_called": True,
        "action_fallback": False,
    }
    v = verify(sample, contract, audit)
    r = compute_decomposed_reward(sample, contract, audit, v)
    t = build_training_targets(sample, contract, audit, v, r)
    exp = ReplayExperience(
        sample_id=sample.sample_id,
        round=0,
        question=sample.question,
        answers=sample.answers,
        documents=sample.documents,
        contract=contract.to_dict(),
        audit=audit.to_dict(),
        verification=v.to_dict(),
        rewards=r.to_dict(),
        training_targets=t,
    )

    metrics = compute_metrics([exp])
    assert metrics["generator_call_rate"] == 1.0
    assert metrics["audit_call_rate"] == 1.0
    assert metrics["avg_generation_candidates"] == 3.0
    assert metrics["audit_nonempty_output_rate"] == 1.0
    assert metrics["empty_answer_rate"] == 0.0
