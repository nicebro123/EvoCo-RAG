from evoco_rag.text_utils import answer_contains_any_gold, exact_presence


def test_exact_presence_ignores_empty_answers():
    assert exact_presence(["", "   ", "the"], "Any non-empty context") is False


def test_exact_presence_matches_normalized_nonempty_answer():
    assert exact_presence(["The Eiffel Tower"], "Paris contains the Eiffel Tower.") is True


def test_answer_contains_any_gold_matches_corag_style_containment():
    assert answer_contains_any_gold(
        ["composer"],
        "Bruce McDaniel was an American composer and musician.",
    ) is True


def test_answer_contains_any_gold_uses_normalized_substring():
    # CoRAG/InstructRAG-style containment is intentionally permissive: after
    # normalization, any gold answer included in the generated output is correct.
    assert answer_contains_any_gold(["us"], "He discussed US foreign policy.") is True
    assert answer_contains_any_gold(["U.S."], "He was born in the US.") is True
