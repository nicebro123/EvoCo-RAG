import json
from types import SimpleNamespace

import pytest

from conftest import make_audit, make_contract, make_sample
from evoco_rag.large_model import LargeGeneratorAuditor
from evoco_rag.schemas import ReplayExperience
from evoco_rag.trainers.large_trainer import LargeTrainer


def _audit_json(answer="politician", used_doc_ids=None):
    return json.dumps({
        "final_answer": answer,
        "used_doc_ids": used_doc_ids or [0],
        "used_evidence": [{
            "doc_id": 0,
            "quote": "Henry Master Feilden was an English Conservative Party politician.",
        }],
        "answer_correctness": "correct",
        "support_level": "fully_supported",
        "failure_type": "none",
        "small_model_feedback": [{
            "doc_id": 0,
            "label": "positive",
            "reason": "contains the answer",
        }],
        "suggested_action": "answer_now",
    })


def test_large_auditor_batches_generation_and_retries_only_invalid_outputs():
    auditor = object.__new__(LargeGeneratorAuditor)
    auditor.candidate_doc_char_limit = 1200
    auditor.num_audit_candidates = 1
    auditor.audit_batch_size = 2
    auditor.audit_temperature = 0.7
    auditor.json_retry = 2

    sample_a = make_sample()
    sample_a.sample_id = "sample-a"
    sample_b = make_sample()
    sample_b.sample_id = "sample-b"
    contract_a = make_contract(selected_doc_ids=[0])
    contract_a.sample_id = sample_a.sample_id
    contract_b = make_contract(selected_doc_ids=[0])
    contract_b.sample_id = sample_b.sample_id

    queued_outputs = [
        ["not json", _audit_json()],
        [_audit_json()],
    ]
    calls = []

    def fake_generate_many(messages_batch, temperature):
        calls.append((len(messages_batch), temperature))
        return queued_outputs.pop(0)

    auditor._generate_many = fake_generate_many

    results = auditor.generate_audit_batch(
        [sample_a, sample_b],
        [contract_a, contract_b],
        show_gold=False,
        round_id=4,
        batch_size=2,
    )

    assert calls == [(2, 0.0), (1, 0.7)]
    assert [ok for _, ok in results] == [True, True]
    assert [audit.sample_id for audit, _ in results] == ["sample-a", "sample-b"]
    assert results[0][0].audit_metadata["num_candidates"] == 2
    assert results[1][0].audit_metadata["num_candidates"] == 1


def test_large_auditor_uses_per_sample_candidate_counts():
    auditor = object.__new__(LargeGeneratorAuditor)
    auditor.candidate_doc_char_limit = 1200
    auditor.num_audit_candidates = 3
    auditor.audit_batch_size = 2
    auditor.audit_temperature = 0.7
    auditor.json_retry = 3

    sample_a = make_sample()
    sample_a.sample_id = "sample-a"
    sample_b = make_sample()
    sample_b.sample_id = "sample-b"
    contract_a = make_contract(selected_doc_ids=[0])
    contract_a.sample_id = sample_a.sample_id
    contract_b = make_contract(selected_doc_ids=[0])
    contract_b.sample_id = sample_b.sample_id

    queued_outputs = [
        [_audit_json(), _audit_json()],
        [_audit_json()],
        [_audit_json()],
    ]
    calls = []

    def fake_generate_many(messages_batch, temperature):
        calls.append((len(messages_batch), temperature))
        return queued_outputs.pop(0)

    auditor._generate_many = fake_generate_many
    results = auditor.generate_audit_batch(
        [sample_a, sample_b],
        [contract_a, contract_b],
        candidate_counts=[1, 3],
        batch_size=2,
    )

    assert calls == [(2, 0.0), (1, 0.7), (1, 0.7)]
    assert results[0][0].audit_metadata["generation_candidate_count"] == 1
    assert results[0][0].audit_metadata["extra_audit_called"] is False
    assert results[1][0].audit_metadata["generation_candidate_count"] == 3
    assert results[1][0].audit_metadata["extra_audit_called"] is True


def test_invalid_candidates_do_not_create_fake_self_consistency():
    auditor = object.__new__(LargeGeneratorAuditor)
    auditor.candidate_doc_char_limit = 1200
    auditor.num_audit_candidates = 1
    auditor.audit_batch_size = 1
    auditor.audit_temperature = 0.7
    auditor.json_retry = 1
    auditor._generate_many = lambda messages, temperature: ["{}"]

    audit, valid = auditor.generate_audit_batch(
        [make_sample()], [make_contract([0])], candidate_counts=[1]
    )[0]

    assert valid is False
    assert audit.final_answer == ""
    assert audit.audit_metadata["self_consistency"] == 0.0


def _experience(sample_id: str) -> ReplayExperience:
    sample = make_sample()
    sample.sample_id = sample_id
    contract = make_contract(selected_doc_ids=[0])
    contract.sample_id = sample_id
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    audit.sample_id = sample_id
    return ReplayExperience(
        sample_id=sample_id,
        round=0,
        question=sample.question,
        answers=sample.answers,
        documents=sample.documents,
        contract=contract.to_dict(),
        audit=audit.to_dict(),
        verification={},
        rewards={},
        training_targets={},
    )


class FakeTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages) + "\nassistant:"

    def _encode(self, text, max_length=None):
        torch = pytest.importorskip("torch")
        length = max(1, len(text))
        if max_length is not None:
            length = min(length, max_length)
        return torch.arange(1, length + 1, dtype=torch.long)

    def __call__(
        self,
        texts,
        padding=False,
        return_tensors=None,
        truncation=False,
        max_length=None,
    ):
        torch = pytest.importorskip("torch")
        is_single = isinstance(texts, str)
        items = [texts] if is_single else list(texts)
        encoded = [self._encode(text, max_length=max_length) for text in items]
        width = max(len(row) for row in encoded)
        input_ids = []
        attention = []
        for row in encoded:
            pad = width - len(row)
            if self.padding_side == "left":
                ids = torch.cat([torch.zeros(pad, dtype=torch.long), row])
                mask = torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(row), dtype=torch.long)])
            else:
                ids = torch.cat([row, torch.zeros(pad, dtype=torch.long)])
                mask = torch.cat([torch.ones(len(row), dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
            input_ids.append(ids)
            attention.append(mask)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention),
        }


class FakeModel:
    def __init__(self, fail=False):
        torch = pytest.importorskip("torch")
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.device = torch.device("cpu")
        self.batch_sizes = []
        self.fail = fail

    def parameters(self):
        return [self.weight]

    def train(self):
        return self

    def __call__(self, input_ids, attention_mask, labels):
        if self.fail:
            raise RuntimeError("intentional failure")
        self.batch_sizes.append(input_ids.shape[0])
        loss = self.weight * 0 + input_ids.float().sum() * 0 + 0.25
        return SimpleNamespace(loss=loss)


def test_large_trainer_sft_uses_configured_batch_size_and_restores_padding():
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    model = FakeModel()
    trainer = LargeTrainer(
        SimpleNamespace(model=model, tokenizer=tokenizer),
        batch_size=2,
        max_prompt_length=4096,
        max_completion_length=256,
    )

    stats = trainer.train_sft([_experience("a"), _experience("b"), _experience("c")])

    assert torch.is_tensor(model.weight)
    assert stats["batch_size"] == 2
    assert stats["steps"] == 2
    assert model.batch_sizes == [2, 1]
    assert tokenizer.padding_side == "left"


def test_large_trainer_restores_padding_after_training_error():
    tokenizer = FakeTokenizer()
    model = FakeModel(fail=True)
    trainer = LargeTrainer(
        SimpleNamespace(model=model, tokenizer=tokenizer),
        batch_size=2,
        max_prompt_length=4096,
        max_completion_length=256,
    )

    with pytest.raises(RuntimeError, match="intentional failure"):
        trainer.train_sft([_experience("a")])

    assert tokenizer.padding_side == "left"


def test_large_trainer_uses_compact_verifier_target():
    exp = _experience("a")
    exp.audit["audit_metadata"] = {"raw_text": "x" * 5000}
    exp.training_targets["large_sft_target"] = {
        "final_answer": "politician",
        "used_doc_ids": [0],
        "used_evidence": [{"doc_id": 0, "quote": "politician"}],
        "answer_correctness": "correct",
        "support_level": "fully_supported",
        "failure_type": "none",
        "small_model_feedback": [],
        "suggested_action": "answer_now",
    }
    trainer = LargeTrainer(SimpleNamespace())

    example = trainer._build_sft_example(exp)
    payload = json.loads(example["target"])

    assert payload["final_answer"] == "politician"
    assert "audit_metadata" not in payload
    assert "raw_text" not in example["target"]


def test_large_trainer_grpo_reward_uses_verifier_signal():
    exp = _experience("a")
    sample_json = json.dumps({
        "sample_id": exp.sample_id,
        "question": exp.question,
        "answers": exp.answers,
        "documents": exp.documents,
    }, ensure_ascii=False)
    contract_json = json.dumps(exp.contract, ensure_ascii=False)
    rewards = []
    reward_func = LargeTrainer.make_grpo_reward_func(reward_log=rewards)

    values = reward_func(
        completions=[
            _audit_json(answer="politician", used_doc_ids=[0]),
            _audit_json(answer="banker", used_doc_ids=[1]),
        ],
        sample_json=[sample_json, sample_json],
        contract_json=[contract_json, contract_json],
        round_id=[0, 0],
    )

    assert values[0] > values[1]
    assert rewards == values
