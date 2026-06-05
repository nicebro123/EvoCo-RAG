from evoco_rag.auditor import extract_json_block, parse_audit


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
           '"answer_correctness": "correct", "support_level": "fully_supported", ' \
           '"failure_type": "none", "suggested_action": "answer_now"}'
    audit, ok = parse_audit(text, "s1", 1)
    assert ok is True
    assert audit.final_answer == "politician"
    assert audit.used_doc_ids == [0]


def test_parse_audit_downgrades_illegal_enum():
    text = '{"final_answer": "x", "failure_type": "cosmic_ray", ' \
           '"answer_correctness": "weird", "support_level": "nope", ' \
           '"suggested_action": "teleport", "used_doc_ids": ["1", 2, "bad"]}'
    audit, ok = parse_audit(text, "s1", 1)
    assert ok is True  # 降级后仍可构造
    assert audit.failure_type == "none"
    assert audit.answer_correctness == "unknown"
    assert audit.used_doc_ids == [1, 2]


def test_parse_audit_fallback_on_garbage():
    audit, ok = parse_audit("no json here at all", "s1", 1)
    assert ok is False
    assert audit.sample_id == "s1"
