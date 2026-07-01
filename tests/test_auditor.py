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


def test_build_hybrid_audit_prompt_requests_analysis_json():
    sample = make_sample()
    contract = make_contract([0])

    messages = build_audit_prompt(
        sample,
        contract,
        prompt_style="hybrid_analysis_json",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert '"analysis"' in system
    assert '"supporting|partial|irrelevant|distractor|conflicting"' in system
    assert "Evidence analysis rules:" in user
    assert "output ONLY that JSON object" in user


def test_build_corag_analysis_plus_audit_prompt_stays_json_only():
    sample = make_sample()
    contract = make_contract([0])

    messages = build_audit_prompt(
        sample,
        contract,
        prompt_style="corag_analysis_plus_audit",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert '"final_answer"' in system
    assert '"answer_reasoning"' in system
    assert '"document_analysis"' in system
    assert "CoRAG's two-step" in system
    assert "CoRAG-style JSON analysis rules:" in user
    assert "Output JSON only" in user
    assert "output ONLY the valid JSON object" in user



def test_build_hybrid_json_repair_prompt_keeps_structured_analysis_and_repair_fields():
    sample = make_sample()
    contract = make_contract([0])

    messages = build_audit_prompt(
        sample,
        contract,
        prompt_style="hybrid_json_repair",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert '"final_answer"' in system
    assert '"answer_summary"' in system
    assert '"analysis"' in system
    assert "repair-oriented hybrid JSON" in system
    assert "Hybrid JSON repair rules:" in user
    assert "Put final_answer and answer_summary before the analysis array" in user
    assert "valid repair-oriented JSON object" in user


def test_parse_hybrid_json_repair_preserves_answer_summary_and_analysis():
    text = '{"final_answer":"politician",' \
           '"answer_summary":"The final answer is politician, supported by doc_id=0.",' \
           '"analysis":[{"doc_id":0,"extraction":"Conservative Party politician",' \
           '"reason":"matches Henry Master Feilden","support":"supporting"}],' \
           '"used_doc_ids":[0], "used_evidence":[{"doc_id":0,"quote":"Conservative Party politician"}],' \
           '"answer_correctness":"correct", "support_level":"fully_supported",' \
           '"failure_type":"none", "small_model_feedback":[],' \
           '"suggested_action":"answer_now"}'

    audit, ok = parse_audit(text, "s1", 1)

    assert ok is True
    assert audit.final_answer == "politician"
    assert audit.audit_metadata["answer_summary"].startswith("The final answer")
    assert audit.audit_metadata["evidence_analysis"][0]["support"] == "supporting"

def test_parse_hybrid_audit_preserves_evidence_analysis_metadata():
    text = '{"analysis":[{"doc_id":0,"extraction":"Conservative Party politician",' \
           '"reason":"matches Henry Master Feilden","support":"supporting"},' \
           '{"doc_id":1,"extraction":"British Army officer",' \
           '"reason":"same-name distractor","support":"distractor"}],' \
           '"final_answer": "politician", "used_doc_ids": [0], ' \
           '"used_evidence": [{"doc_id": 0, "quote": "Conservative Party politician"}], ' \
           '"answer_correctness": "correct", "support_level": "fully_supported", ' \
           '"failure_type": "none", "small_model_feedback": [], ' \
           '"suggested_action": "answer_now"}'

    audit, ok = parse_audit(text, "s1", 1)

    assert ok is True
    assert audit.final_answer == "politician"
    analysis = audit.audit_metadata["evidence_analysis"]
    assert analysis[0]["doc_id"] == 0
    assert analysis[0]["support"] == "supporting"
    assert analysis[1]["support"] == "distractor"



def test_build_relaxed_corag_json_prompt_uses_flat_analysis_text():
    sample = make_sample()
    contract = make_contract([0])

    messages = build_audit_prompt(
        sample,
        contract,
        prompt_style="relaxed_corag_json",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert '"analysis_text"' in system
    assert '"document_analysis"' not in system
    assert "top-level JSON flat" in system
    assert "Relaxed CoRAG JSON rules:" in user
    assert "analysis_text is a single natural-language string" in user
    assert "output ONLY the valid flat JSON object" in user


def test_parse_relaxed_corag_json_preserves_analysis_text_metadata():
    text = '{"analysis_text":"Document 0 says Henry Master Feilden was a Conservative Party politician. Therefore the final answer is politician.",' \
           '"final_answer": "politician", "used_doc_ids": [0], ' \
           '"used_evidence": [{"doc_id": 0, "quote": "Conservative Party politician"}], ' \
           '"answer_correctness": "correct", "support_level": "fully_supported", ' \
           '"failure_type": "none", "small_model_feedback": [], ' \
           '"suggested_action": "answer_now"}'

    audit, ok = parse_audit(text, "s1", 1)

    assert ok is True
    assert audit.final_answer == "politician"
    assert "final answer is politician" in audit.audit_metadata["analysis_text"]
    assert audit.audit_metadata["raw_json"]["analysis_text"].startswith("Document 0")

def test_parse_corag_analysis_plus_audit_preserves_document_analysis_metadata():
    text = '{"final_answer": "politician",' \
           '"answer_reasoning": "Document 0 states that Henry Master Feilden was a politician.",' \
           '"document_analysis":[{"doc_id":0,"extraction":"Conservative Party politician",' \
           '"explanation":"This directly answers the occupation question.","support":"supporting"},' \
           '{"doc_id":1,"extraction":"British Army officer",' \
           '"explanation":"This is a same-name distractor.","support":"distractor"}],' \
           '"used_doc_ids": [0], ' \
           '"used_evidence": [{"doc_id": 0, "quote": "Conservative Party politician"}], ' \
           '"answer_correctness": "correct", "support_level": "fully_supported", ' \
           '"failure_type": "none", "small_model_feedback": [], ' \
           '"suggested_action": "answer_now"}'

    audit, ok = parse_audit(text, "s1", 1)

    assert ok is True
    assert audit.final_answer == "politician"
    raw = audit.audit_metadata["raw_json"]
    assert raw["answer_reasoning"].startswith("Document 0")
    analysis = audit.audit_metadata["evidence_analysis"]
    assert analysis[0]["reason"] == "This directly answers the occupation question."
    assert analysis[1]["support"] == "distractor"


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
