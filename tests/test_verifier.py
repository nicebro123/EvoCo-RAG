from conftest import make_audit, make_contract, make_sample

from evoco_rag.verifier import verify


def test_answer_match_hits_answers():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="He was a politician.", used_doc_ids=[0])
    v = verify(sample, contract, audit)
    assert v.answer_match is True


def test_cited_doc_contains_answer():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    v = verify(sample, contract, audit)
    assert v.cited_doc_contains_answer is True


def test_used_doc_not_in_selected_evidence():
    sample = make_sample()
    # 合约候选只有 doc0，但大模型引用了 doc1
    contract = make_contract(selected_doc_ids=[0], candidate_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[1])
    v = verify(sample, contract, audit)
    assert v.used_doc_in_selected_evidence is False


def test_json_invalid_lowers_trust():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    v_ok = verify(sample, contract, audit, json_valid=True)
    v_bad = verify(sample, contract, audit, json_valid=False)
    assert v_bad.audit_trust_weight <= v_ok.audit_trust_weight
    assert v_bad.json_valid is False


def test_high_trust_when_all_rules_pass():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    v = verify(sample, contract, audit)
    assert v.audit_trust_weight >= 0.9


def test_trust_components_are_explainable():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    v = verify(sample, contract, audit)
    assert v.trust_components["json_valid_score"] == 1.0
    assert v.trust_components["citation_score"] == 1.0
    assert v.trust_components["support_rule_score"] == 1.0


def test_low_self_consistency_lowers_trust():
    sample = make_sample()
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer="politician", used_doc_ids=[0])
    audit.audit_metadata["self_consistency"] = 0.2
    v = verify(sample, contract, audit)
    assert v.audit_trust_weight < 0.9
    assert v.trust_components["self_consistency_score"] == 0.2
