from evoco_rag.replay_buffer import ReplayBuffer
from evoco_rag.schemas import ReplayExperience


def _exp(sample_id, failure_type, trust, pos=None, neg=None):
    attribution_case = "parametric_answer_without_support" if failure_type == "unsupported_answer" else "both_success"
    return ReplayExperience(
        sample_id=sample_id,
        round=0,
        question="q",
        answers=["a"],
        documents=[{"doc_id": 0}],
        contract={},
        audit={"failure_type": failure_type, "audit_metadata": {"self_consistency": 0.75}},
        verification={"audit_trust_weight": trust, "json_valid": trust >= 0.5},
        rewards={"total_reward": 1.0, "attribution_case": attribution_case},
        training_targets={
            "failure_type": failure_type,
            "attribution_case": attribution_case,
            "small_positive_doc_ids": pos or [],
            "small_negative_doc_ids": neg or [],
            "large_sft_eligible": failure_type == "none",
            "wrong_retriever_reward_if_answer_only": failure_type == "unsupported_answer",
            "do_not_reward_retriever_reason": (
                "parametric_answer_without_support" if failure_type == "unsupported_answer" else ""
            ),
        },
    )


def test_write_and_read_roundtrip(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9, pos=[0]), _exp("s2", "retrieval_miss", 0.3)]
    n = rb.write(exps, round_id=0)
    assert n == 2
    loaded = rb.read(round_id=0)
    assert len(loaded) == 2
    assert loaded[0].sample_id == "s1"
    # 重写同一 round 不应在 all.jsonl 里追加重复记录。
    rb.write(exps, round_id=0)
    assert len(rb.read()) == 2


def test_filter_by_failure_type(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9), _exp("s2", "retrieval_miss", 0.3),
            _exp("s3", "retrieval_miss", 0.8)]
    rb.write(exps, round_id=0)
    loaded = rb.read(round_id=0)
    miss = rb.filter_by_failure_type(loaded, "retrieval_miss")
    assert {e.sample_id for e in miss} == {"s2", "s3"}


def test_filter_by_trust(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9), _exp("s2", "none", 0.3)]
    rb.write(exps, round_id=0)
    loaded = rb.read(round_id=0)
    high = rb.filter_by_trust(loaded, min_weight=0.5)
    assert {e.sample_id for e in high} == {"s1"}


def test_sample_small_training_pairs_respects_trust(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9, pos=[0]), _exp("s2", "none", 0.2, pos=[1])]
    rb.write(exps, round_id=0)
    loaded = rb.read(round_id=0)
    pairs = rb.sample_small_training_pairs(loaded, min_trust=0.5)
    assert len(pairs) == 1
    assert pairs[0]["sample_id"] == "s1"


def test_rule_verified_small_target_bypasses_audit_trust_filter(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exp = _exp("failure", "retrieval_miss", 0.2, neg=[0])
    exp.training_targets["small_target_source"] = "gold_rule_verifier"
    exp.training_targets["small_action_target"] = "retrieve_more"
    rb.write([exp], round_id=0)

    pairs = rb.sample_small_training_pairs(rb.read(0), min_trust=0.5)

    assert len(pairs) == 1
    assert pairs[0]["negative_doc_ids"] == [0]
    assert pairs[0]["action_target"] == "retrieve_more"


def test_large_sft_selection(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9), _exp("s2", "generation_error", 0.9)]
    rb.write(exps, round_id=0)
    loaded = rb.read(round_id=0)
    sft = rb.sample_large_sft(loaded)
    assert {e.sample_id for e in sft} == {"s1"}


def test_credit_assignment_summary_counts_wrong_retriever_reward(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9), _exp("s2", "unsupported_answer", 0.9)]
    rb.write(exps, round_id=0)
    summary = rb.credit_assignment_summary(rb.read(0))
    assert summary["wrong_retriever_reward_count"] == 1
    assert summary["wrong_retriever_reward_rate"] == 0.5
    assert summary["attribution_case_distribution"]["parametric_answer_without_support"] == 1


def test_trust_summary_reports_json_and_consistency(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9), _exp("s2", "retrieval_miss", 0.3)]
    rb.write(exps, round_id=0)
    summary = rb.trust_summary(rb.read(0))
    assert summary["audit_json_valid_rate"] == 0.5
    assert summary["low_trust_rate"] == 0.5
    assert summary["audit_self_consistency_mean"] == 0.75



def test_sample_small_training_pairs_carries_doc_weights(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exp = _exp("weighted", "retrieval_miss", 0.2, pos=[0], neg=[1])
    exp.documents = [{"doc_id": 0}, {"doc_id": 1}]
    exp.training_targets["small_target_source"] = "gold_rule_verifier"
    exp.training_targets["small_positive_doc_weights"] = {"0": 1.0}
    exp.training_targets["small_negative_doc_weights"] = {"1": 2.5}
    exp.training_targets["small_hard_negative_doc_ids"] = [1]
    rb.write([exp], round_id=0)

    pairs = rb.sample_small_training_pairs(rb.read(0), min_trust=0.5)

    assert pairs[0]["positive_doc_weights"] == {"0": 1.0}
    assert pairs[0]["negative_doc_weights"] == {"1": 2.5}
    assert pairs[0]["hard_negative_doc_ids"] == [1]
