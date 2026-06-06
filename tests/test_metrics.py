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
    for key in ("accuracy", "recall_at_k", "mrr", "evidence_support_rate",
                "citation_correctness", "unsupported_answer_rate",
                "avg_selected_docs", "audit_call_rate",
                "attribution_case_distribution", "wrong_retriever_reward_rate",
                "audit_json_valid_rate", "audit_trust_weight_mean",
                "avg_action_cost_penalty", "avg_total_cost_penalty",
                "accuracy_cost_pareto_point"):
        assert key in m
    assert m["num_examples"] == 2
    assert m["accuracy"] == 50.0  # 一对一错
    assert "avg_total_cost_penalty" in m["accuracy_cost_pareto_point"]


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
