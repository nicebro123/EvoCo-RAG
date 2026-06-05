from evoco_rag.replay_buffer import ReplayBuffer
from evoco_rag.schemas import ReplayExperience


def _exp(sample_id, failure_type, trust, pos=None, neg=None):
    return ReplayExperience(
        sample_id=sample_id,
        round=0,
        question="q",
        answers=["a"],
        documents=[{"doc_id": 0}],
        contract={},
        audit={"failure_type": failure_type},
        verification={"audit_trust_weight": trust},
        rewards={"total_reward": 1.0},
        training_targets={
            "failure_type": failure_type,
            "small_positive_doc_ids": pos or [],
            "small_negative_doc_ids": neg or [],
            "large_sft_eligible": failure_type == "none",
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


def test_large_sft_selection(tmp_path):
    rb = ReplayBuffer(root=str(tmp_path / "replay"))
    exps = [_exp("s1", "none", 0.9), _exp("s2", "generation_error", 0.9)]
    rb.write(exps, round_id=0)
    loaded = rb.read(round_id=0)
    sft = rb.sample_large_sft(loaded)
    assert {e.sample_id for e in sft} == {"s1"}
