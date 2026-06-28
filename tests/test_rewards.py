"""四类责任归因测试（开发文档 §11.2）。

doc0 含答案 'politician'，doc1 不含。通过控制：
  - selected_evidence 是否含 doc0  → 决定 support_rule_passed
  - final_answer 是否为 'politician' → 决定 answer_match
覆盖 (answer, support) 的四个象限。
"""

from conftest import make_audit, make_contract, make_sample

from evoco_rag.rewards import build_training_targets, compute_decomposed_reward
from evoco_rag.schemas import AttributionCase, FailureType, RetrievalAction
from evoco_rag.verifier import verify


def _pipeline(selected_ids, final_answer):
    sample = make_sample()
    contract = make_contract(selected_doc_ids=selected_ids)
    audit = make_audit(final_answer=final_answer, used_doc_ids=selected_ids)
    verification = verify(sample, contract, audit, json_valid=True)
    reward = compute_decomposed_reward(sample, contract, audit, verification)
    targets = build_training_targets(sample, contract, audit, verification, reward)
    return verification, reward, targets


def test_answer_true_support_true():
    v, r, t = _pipeline(selected_ids=[0], final_answer="politician")
    assert v.answer_match is True
    assert v.support_rule_passed is True
    assert 0 in t["small_positive_doc_ids"]
    assert t["large_sft_eligible"] is True
    assert r.answer_reward == 1.0 and r.support_reward == 1.0
    assert r.attribution_case == AttributionCase.BOTH_SUCCESS
    assert t["attribution_case"] == AttributionCase.BOTH_SUCCESS
    assert t["small_credit_weight"] == 1.0
    assert t["large_credit_weight"] == 1.0


def test_answer_true_support_false():
    # 答案对，但选中证据(doc1)不含答案 → 大模型凭参数知识答对
    v, r, t = _pipeline(selected_ids=[1], final_answer="politician")
    assert v.answer_match is True
    assert v.support_rule_passed is False
    # 不能奖励错误选中的 doc1，但应使用候选池中漏排的真实 doc0 纠正 reranker。
    assert 1 not in t["small_positive_doc_ids"]
    assert 0 in t["small_positive_doc_ids"]
    assert t["failure_type"] == FailureType.UNSUPPORTED_ANSWER
    assert t["large_sft_eligible"] is False
    assert t["large_sft_target"] is None
    assert r.support_reward == 0.0
    assert r.attribution_case == AttributionCase.PARAMETRIC_ANSWER_WITHOUT_SUPPORT
    assert t["wrong_retriever_reward_if_answer_only"] is True
    assert t["small_credit_weight"] == 0.0
    assert t["do_not_reward_retriever_reason"] == "parametric_answer_without_support"


def test_answer_false_support_true():
    # 选中证据(doc0)含答案，但生成答错 → 奖励小模型，归因 generation_error
    v, r, t = _pipeline(selected_ids=[0], final_answer="banker")
    assert v.answer_match is False
    assert v.support_rule_passed is True
    # 关键断言：可以给小模型 positive doc
    assert 0 in t["small_positive_doc_ids"]
    assert t["failure_type"] == FailureType.GENERATION_ERROR
    # 大模型 reward 较低（answer_reward=0）
    assert r.answer_reward == 0.0
    assert t["large_sft_eligible"] is True
    assert t["large_sft_target"]["final_answer"] == "politician"
    assert t["large_sft_target"]["used_doc_ids"] == [0]
    assert t["small_action_target"] == RetrievalAction.ASK_AUDITOR
    assert t["attribution_case"] == AttributionCase.RETRIEVER_SUCCESS_GENERATOR_FAIL
    assert t["small_credit_weight"] == 1.0


def test_answer_false_support_false():
    v, r, t = _pipeline(selected_ids=[1], final_answer="banker")
    assert v.answer_match is False
    assert v.support_rule_passed is False
    assert 0 in t["small_positive_doc_ids"]
    assert 1 in t["small_negative_doc_ids"]
    # With no extra candidate documents, retrieve_more would be a no-op; teach
    # the small policy to ask for extra audit/generation instead.
    assert t["small_action_target"] == RetrievalAction.ASK_AUDITOR
    assert t["attribution_case"] == AttributionCase.BOTH_FAIL


def test_cost_penalty_grows_with_selected_docs():
    _, r1, _ = _pipeline(selected_ids=[0], final_answer="politician")
    _, r2, _ = _pipeline(selected_ids=[0, 1], final_answer="politician")
    assert r2.cost_penalty > r1.cost_penalty


def test_action_cost_penalty_for_retrieve_more():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[1], action=RetrievalAction.RETRIEVE_MORE)
    audit = make_audit(final_answer="banker", used_doc_ids=[1])
    verification = verify(sample, contract, audit, json_valid=True)
    reward = compute_decomposed_reward(sample, contract, audit, verification)
    assert reward.action_cost_penalty > 0.0
    assert reward.cost_penalty > 0.0


def test_audit_cost_uses_actual_extra_candidate_count():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0], action=RetrievalAction.ANSWER_NOW)
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    audit.audit_metadata = {"generation_candidate_count": 3}
    verification = verify(sample, contract, audit, json_valid=True)
    reward = compute_decomposed_reward(sample, contract, audit, verification)
    assert reward.action_cost_penalty == 0.2


def test_action_target_retrieve_more_only_when_extra_docs_exist():
    sample = make_sample()
    sample.documents.append({
        "doc_id": 2,
        "title": "Henry Feilden extra",
        "text": "A third irrelevant candidate.",
        "raw": "A third irrelevant candidate.",
    })
    contract = make_contract(
        selected_doc_ids=[1],
        candidate_doc_ids=(0, 1),
        action=RetrievalAction.RETRIEVE_MORE,
    )
    audit = make_audit(final_answer="banker", used_doc_ids=[1])
    verification = verify(sample, contract, audit, json_valid=True)
    reward = compute_decomposed_reward(sample, contract, audit, verification)
    targets = build_training_targets(sample, contract, audit, verification, reward)
    assert targets["small_action_target"] == RetrievalAction.RETRIEVE_MORE


def test_evolution_signal_routes_generator_fail_to_large_boundary():
    _, _, t = _pipeline(selected_ids=[0], final_answer="banker")
    signal = t["evolution_signal"]

    assert signal["relation"] == "occupation"
    assert signal["failure_mode"] == "generation_error"
    assert signal["target_module"] == "large"
    assert signal["hard_sample"] is True
    assert signal["should_train_generator_boundary"] is True
    assert signal["should_train_retriever"] is False


def test_evolution_signal_routes_rerank_miss_to_small_model():
    _, _, t = _pipeline(selected_ids=[1], final_answer="banker")
    signal = t["evolution_signal"]

    assert signal["failure_mode"] == "rerank_miss"
    assert signal["target_module"] == "small"
    assert signal["answer_in_any_doc"] is True
    assert signal["answer_in_candidate_docs"] is True
    assert signal["answer_in_selected_evidence"] is False
    assert signal["should_train_retriever"] is True
    assert signal["should_train_generator_boundary"] is False


def test_evolution_signal_detects_relation_confusion():
    # selected evidence contains the gold occupation, but the model copies a
    # plausible occupation from a same-name distractor in the candidate set.
    _, _, t = _pipeline(selected_ids=[0, 1], final_answer="officer")
    signal = t["evolution_signal"]

    assert signal["relation"] == "occupation"
    assert signal["failure_mode"] == "relation_confusion"
    assert signal["target_module"] == "large"
    assert signal["wrong_answer_in_candidate_docs"] is True
    assert signal["should_train_generator_boundary"] is True
