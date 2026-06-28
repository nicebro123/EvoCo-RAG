from conftest import make_audit, make_contract, make_sample

from evoco_rag.auditor import build_audit_prompt, extract_json_block, parse_audit
from evoco_rag.large_model import LargeGeneratorAuditor


def test_extract_clean_json():
    d = extract_json_block('{"a": 1, "b": [2, 3]}')
    assert d == {"a": 1, "b": [2, 3]}


def test_extract_json_with_surrounding_text():
    text = 'Sure! Here is the answer:\n```json\n{"final_answer": "politician"}\n```\nDone.'
    d = extract_json_block(text)
    assert d["final_answer"] == "politician"


def test_extract_json_with_nested_braces_and_strings():
    text = 'noise {"x": {"y": "a}b"}, "z": 1} trailing'
    d = extract_json_block(text)
    assert d["x"]["y"] == "a}b"
    assert d["z"] == 1


def test_parse_audit_valid():
    text = '{"final_answer": "politician", "used_doc_ids": [0], ' \
           '"used_evidence": [{"doc_id": 0, "quote": "politician"}], ' \
           '"answer_correctness": "correct", "support_level": "fully_supported", ' \
           '"failure_type": "none", "small_model_feedback": [], ' \
           '"suggested_action": "answer_now"}'
    audit, ok = parse_audit(text, "s1", 1)
    assert ok is True
    assert audit.final_answer == "politician"
    assert audit.used_doc_ids == [0]
    assert audit.audit_metadata["parse_status"] == "parsed"
    assert audit.audit_metadata["raw_json"]["final_answer"] == "politician"
    assert audit.audit_metadata["raw_text"] == text


def test_parse_audit_rejects_illegal_enum_and_missing_fields():
    text = '{"final_answer": "x", "failure_type": "cosmic_ray", ' \
           '"answer_correctness": "weird", "support_level": "nope", ' \
           '"suggested_action": "teleport", "used_doc_ids": ["1", 2, "bad"]}'
    audit, ok = parse_audit(text, "s1", 1)
    assert ok is False
    assert audit.final_answer == ""
    assert audit.audit_metadata["parse_status"] == "fallback"
    assert audit.audit_metadata["schema_error"].startswith("missing_fields:")


def test_parse_audit_rejects_parseable_empty_object():
    audit, ok = parse_audit("{}", "s1", 1)
    assert ok is False
    assert audit.final_answer == ""
    assert "missing_fields" in audit.audit_metadata["schema_error"]


def test_parse_audit_fallback_on_garbage():
    audit, ok = parse_audit("no json here at all", "s1", 1)
    assert ok is False
    assert audit.sample_id == "s1"
    assert audit.final_answer == ""
    assert audit.audit_metadata["parse_status"] == "fallback"
    assert "no json" in audit.audit_metadata["raw_text"]


def test_build_audit_prompt_uses_configurable_doc_limit():
    sample = make_sample()
    sample.documents[0]["text"] = "x" * 50
    contract = make_contract([0])

    messages = build_audit_prompt(sample, contract, candidate_doc_char_limit=12)
    text = messages[-1]["content"]

    assert "x" * 12 in text
    assert "x" * 13 not in text
    assert "..." in text


def test_audit_candidate_score_prefers_supported_cited_answer():
    sample = make_sample()
    contract = make_contract([0])
    good = make_audit("politician", [0])
    good.used_evidence = [{"doc_id": 0, "quote": "Conservative Party politician"}]
    bad = make_audit(
        "British Army officer and naturalist",
        [],
        support_level="unsupported",
        failure_type="unsupported_answer",
    )

    good_score = LargeGeneratorAuditor.score_audit_candidate(sample, contract, good, True)
    bad_score = LargeGeneratorAuditor.score_audit_candidate(sample, contract, bad, True)

    assert good_score > bad_score
