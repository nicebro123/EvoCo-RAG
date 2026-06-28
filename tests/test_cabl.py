from conftest import make_audit, make_contract, make_sample
from evoco_rag.cabl import build_boundary_pairs, build_relation_answer_pool, mine_counterfactual_answers
from evoco_rag.schemas import ReplayExperience


def _exp(answer_match=False, final_answer="banker"):
    sample = make_sample()
    sample.documents[0]["text"] = (
        "Henry Master Feilden was an English Conservative Party politician. "
        "John Smith was a banker mentioned as a distractor."
    )
    contract = make_contract(selected_doc_ids=[0])
    audit = make_audit(final_answer=final_answer, used_doc_ids=[0])
    return ReplayExperience(
        sample_id=sample.sample_id,
        round=0,
        question=sample.question,
        answers=sample.answers,
        documents=sample.documents,
        contract=contract.to_dict(),
        audit=audit.to_dict(),
        verification={"answer_match": answer_match},
        rewards={},
        training_targets={},
    )


def test_cabl_mines_model_self_error_first():
    negatives = mine_counterfactual_answers(_exp(final_answer="banker"), max_negatives=2)

    assert negatives
    assert negatives[0]["answer"] == "banker"
    assert negatives[0]["source"] == "model_self_error"


def test_cabl_filters_gold_aliases_from_negative_answers():
    exp = _exp(final_answer="politician")

    pairs = build_boundary_pairs(exp, max_negatives=5, margin=0.7)

    assert pairs
    assert all(pair["positive"] == "politician" for pair in pairs)
    assert all(pair["negative"].lower() != "politician" for pair in pairs)
    assert all(pair["margin"] == 0.7 for pair in pairs)



def test_cabl_relation_pool_prefers_same_type_negatives():
    base = _exp(answer_match=True, final_answer="politician")
    actor_exp = _exp(answer_match=True, final_answer="actor")
    actor_exp.sample_id = "actor-sample"
    actor_exp.question = "What is Ada Actor's occupation?"
    actor_exp.answers = ["actor"]
    lawyer_exp = _exp(answer_match=True, final_answer="lawyer")
    lawyer_exp.sample_id = "lawyer-sample"
    lawyer_exp.question = "What is Laura Lawyer's occupation?"
    lawyer_exp.answers = ["lawyer"]

    pool = build_relation_answer_pool([base, actor_exp, lawyer_exp])
    negatives = mine_counterfactual_answers(
        base,
        max_negatives=3,
        answer_pool=pool,
        use_model_self_error=False,
        use_relation_answer_pool=True,
        use_answer_type_filter=True,
        use_retrieved_distractors=False,
    )

    assert {item["answer"] for item in negatives} == {"actor", "lawyer"}
    assert all(item["source"] == "relation_answer_pool" for item in negatives)
    assert all(item["type"] == "same_relation_occupation" for item in negatives)


def test_cabl_type_filter_removes_trivial_occupation_distractors():
    exp = _exp(answer_match=True, final_answer="politician")
    exp.documents[0]["text"] = (
        "Henry Master Feilden was an English Conservative Party politician. "
        "He was born in June 1955 near Georgia."
    )

    negatives = mine_counterfactual_answers(
        exp,
        max_negatives=5,
        use_model_self_error=False,
        use_relation_answer_pool=False,
        use_answer_type_filter=True,
        use_retrieved_distractors=True,
    )

    assert all(item["answer"] not in {"June", "1955", "Georgia"} for item in negatives)


def test_cabl_mines_lowercase_occupation_distractors():
    exp = _exp(answer_match=True, final_answer="politician")
    exp.documents[1]["text"] = (
        "Colonel Henry Wemyss Feilden was a British Army officer and naturalist."
    )

    negatives = mine_counterfactual_answers(
        exp,
        max_negatives=5,
        use_model_self_error=False,
        use_relation_answer_pool=False,
        use_answer_type_filter=True,
        use_retrieved_distractors=True,
    )

    assert {item["answer"] for item in negatives} & {"officer", "naturalist"}


def test_cabl_counterfactual_evidence_switch_adds_corrupted_evidence():
    base = _exp(answer_match=True, final_answer="politician")
    actor_exp = _exp(answer_match=True, final_answer="actor")
    actor_exp.sample_id = "actor-sample"
    actor_exp.answers = ["actor"]
    pool = build_relation_answer_pool([base, actor_exp])

    pairs = build_boundary_pairs(
        base,
        max_negatives=1,
        answer_pool=pool,
        use_model_self_error=False,
        use_relation_answer_pool=True,
        use_answer_type_filter=True,
        use_retrieved_distractors=False,
        use_counterfactual_evidence=True,
    )

    assert pairs
    assert pairs[0]["negative"] == "actor"
    assert "counterfactual_evidence" in pairs[0]
    assert "actor" in pairs[0]["counterfactual_evidence"]


def test_relation_key_detects_place_of_birth_questions():
    from evoco_rag.cabl import relation_key_for_question

    assert relation_key_for_question("What is Ada Lovelace's place of birth?") == "location"


def test_hard_aware_cabl_weights_self_evolution_pairs():
    exp = _exp(answer_match=False, final_answer="banker")
    exp.training_targets = {
        "evolution_signal": {
            "source": "attribution_verifier",
            "relation": "occupation",
            "failure_mode": "generation_error",
        }
    }

    pairs = build_boundary_pairs(
        exp,
        hard_aware=True,
        hard_pair_weight=2.5,
        max_negatives=1,
    )

    assert pairs
    assert pairs[0]["source"] == "self_evolution_error"
    assert pairs[0]["negative_type"] == "generation_error_self_error"
    assert pairs[0]["weight"] == 2.5
    assert pairs[0]["relation"] == "occupation"
    assert "occupation" in pairs[0]["relation_hint"]


def test_hard_aware_cabl_skips_retrieval_absent_samples():
    exp = _exp(answer_match=False, final_answer="banker")
    exp.training_targets = {
        "evolution_signal": {
            "source": "attribution_verifier",
            "relation": "occupation",
            "failure_mode": "retrieval_absent",
        }
    }

    pairs = build_boundary_pairs(exp, hard_aware=True, skip_retrieval_absent=True)

    assert pairs == []
