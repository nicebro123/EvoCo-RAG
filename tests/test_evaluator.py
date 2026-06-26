from conftest import make_audit, make_contract, make_sample

from evoco_rag.config import EvoCoConfig
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.schemas import RetrievalAction


class RecordingSmall:
    def __init__(self, action=RetrievalAction.ANSWER_NOW):
        self.action = action
        self.calls = []

    def build_contract(self, sample, **kwargs):
        self.calls.append(kwargs)
        return make_contract(selected_doc_ids=[0], action=self.action)


class RecordingLarge:
    def __init__(self):
        self.calls = []

    def generate_audit_batch(
        self, samples, contracts, show_gold, round_id, batch_size,
        candidate_counts=None,
    ):
        self.calls.append({
            "contracts": contracts,
            "contract": contracts[0] if contracts else None,
            "show_gold": show_gold,
            "round_id": round_id,
            "batch_size": batch_size,
            "candidate_counts": candidate_counts,
        })
        return [
            (make_audit(final_answer="politician", used_doc_ids=[0]), True)
            for _ in samples
        ]

    def generate_audit(self, sample, contract, show_gold, round_id):
        self.calls.append({
            "contract": contract,
            "show_gold": show_gold,
            "round_id": round_id,
        })
        return make_audit(final_answer="politician", used_doc_ids=[0]), True


def test_run_inference_passes_evidence_budget_config():
    cfg = EvoCoConfig()
    cfg.contract.retrieve_more_conf_threshold = 0.62
    cfg.contract.retrieve_more_margin_threshold = 0.05
    cfg.runtime.audit_batch_size = 3
    small = RecordingSmall()
    large = RecordingLarge()

    Evaluator(cfg, small, large).run_inference([make_sample()], round_id=2)

    assert small.calls[0]["retrieve_more_conf_threshold"] == 0.62
    assert small.calls[0]["retrieve_more_margin_threshold"] == 0.05
    assert large.calls[0]["show_gold"] is False
    assert large.calls[0]["round_id"] == 2
    assert large.calls[0]["batch_size"] == 3
    assert large.calls[0]["candidate_counts"] == [cfg.runtime.num_audit_candidates]
def test_run_inference_streams_prediction_chunks(tmp_path):
    cfg = EvoCoConfig()
    cfg.runtime.audit_batch_size = 1
    cfg.runtime.progress_interval = 2
    small = RecordingSmall(action=RetrievalAction.ANSWER_NOW)
    large = RecordingLarge()
    path = tmp_path / "predictions.jsonl"

    metrics = Evaluator(cfg, small, large).run_inference(
        [make_sample(), make_sample(), make_sample()],
        predictions_path=str(path),
    )

    assert metrics["num_examples"] == 3
    assert len(large.calls) == 2
    assert sum(1 for _ in path.open(encoding="utf-8")) == 3
