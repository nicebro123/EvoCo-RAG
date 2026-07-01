from conftest import make_sample

from evoco_rag.config import EvoCoConfig
from evoco_rag.entity_consistency import (
    EntityConsistencyConfig,
    apply_entity_consistency_rerank,
    entity_consistency_features,
    question_entity_hint,
)


def test_question_entity_hint_extracts_possessive_popqa_entity():
    assert question_entity_hint("What is Henry Feilden's occupation?") == "Henry Feilden"


def test_entity_consistency_features_detect_exact_title():
    sample = make_sample()
    features = entity_consistency_features(sample, sample.documents[0])
    assert features["question_entity"] == "Henry Feilden"
    assert features["entity_overlap"] == 1.0
    assert features["relation"] == "occupation"
    assert features["relation_score"] > 0.0
    assert features["local_support_score"] > 0.0
    assert features["exact_title_match"] is False


def test_entity_consistency_rerank_promotes_entity_matching_doc():
    sample = make_sample()
    sample.question = "What is Walter Köbel's occupation?"
    sample.documents = [
        {"doc_id": 0, "title": "Walter Köbel", "text": "Walter Köbel was a politician."},
        {"doc_id": 1, "title": "Walter Köberle", "text": "Walter Köberle was an ice hockey player."},
    ]
    ranked = [
        {"doc_id": 1, "score": 10.0},
        {"doc_id": 0, "score": 9.9},
    ]
    out = apply_entity_consistency_rerank(
        sample,
        ranked,
        EntityConsistencyConfig(enabled=True, weight=0.35, max_boost=0.75),
    )
    assert out[0]["doc_id"] == 0
    assert out[0]["pre_entity_score"] == 9.9
    assert out[0]["entity_consistency_boost"] > out[1]["entity_consistency_boost"]
    assert "entity_consistency" in out[0]


def test_entity_consistency_config_loads_from_yaml_section():
    cfg = EvoCoConfig.from_dict({
        "entity_consistency": {
            "enabled": True,
            "weight": 0.42,
        }
    })
    assert cfg.entity_consistency.enabled is True
    assert cfg.entity_consistency.weight == 0.42


def test_entity_relation_features_penalize_same_name_relation_distractor():
    sample = make_sample()
    doc = {
        "doc_id": 2,
        "title": "Henry Feilden (soldier)",
        "text": "Henry Feilden served in the army and later explored natural history.",
    }

    features = entity_consistency_features(sample, doc)

    assert features["title_entity_overlap"] == 1.0
    assert features["same_name_distractor"] is True
    assert features["relation"] == "occupation"


def test_entity_relation_rerank_uses_relation_window_not_title_only():
    sample = make_sample()
    sample.question = "What is Henry Feilden's occupation?"
    sample.documents = [
        {"doc_id": 0, "title": "Henry Feilden", "text": "Henry Feilden was a Conservative Party politician."},
        {"doc_id": 1, "title": "Henry Feilden", "text": "Henry Feilden was born in London and served overseas."},
    ]
    ranked = [
        {"doc_id": 1, "score": 5.0},
        {"doc_id": 0, "score": 4.95},
    ]

    out = apply_entity_consistency_rerank(
        sample,
        ranked,
        EntityConsistencyConfig(enabled=True, weight=0.2, relation_weight=0.35, local_support_weight=0.35),
    )

    assert out[0]["doc_id"] == 0
    assert out[0]["entity_consistency"]["relation_score"] > out[1]["entity_consistency"]["relation_score"]
