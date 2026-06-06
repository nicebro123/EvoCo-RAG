from conftest import make_audit, make_contract, make_sample

from evoco_rag.config import EvoCoConfig
from evoco_rag.evaluation.evaluator import Evaluator
from evoco_rag.schemas import RetrievalAction


class RecordingSmall:
    def __init__(self, action=RetrievalAction.ASK_AUDITOR):
        self.action = action
        self.calls = []

    def build_contract(self, sample, **kwargs):
        self.calls.append(kwargs)
        return make_contract(selected_doc_ids=[0], action=self.action)


class RecordingLarge:
    def __init__(self):
        self.calls = []

    def generate_audit(self, sample, contract, show_gold, round_id):
        self.calls.append({
            "contract": contract,
            "show_gold": show_gold,
            "round_id": round_id,
        })
        return make_audit(final_answer="politician", used_doc_ids=[0]), True


def test_run_inference_passes_action_policy_config():
    cfg = EvoCoConfig()
    cfg.contract.action_mode = "hybrid"
    cfg.contract.policy_action_min_conf = 0.77
    small = RecordingSmall()
    large = RecordingLarge()

    Evaluator(cfg, small, large).run_inference([make_sample()], round_id=2)

    assert small.calls[0]["action_mode"] == "hybrid"
    assert small.calls[0]["policy_action_min_conf"] == 0.77
    assert large.calls[0]["show_gold"] is False
    assert large.calls[0]["round_id"] == 2


def test_run_inference_no_action_ablation_forces_answer_now():
    cfg = EvoCoConfig()
    cfg.ablation.use_action_policy = False
    small = RecordingSmall(action=RetrievalAction.ASK_AUDITOR)
    large = RecordingLarge()

    Evaluator(cfg, small, large).run_inference([make_sample()])

    assert large.calls[0]["contract"].retrieval_action == RetrievalAction.ANSWER_NOW
