from evoco_rag.text_utils import exact_presence


def test_exact_presence_ignores_empty_answers():
    assert exact_presence(["", "   ", "the"], "Any non-empty context") is False


def test_exact_presence_matches_normalized_nonempty_answer():
    assert exact_presence(["The Eiffel Tower"], "Paris contains the Eiffel Tower.") is True
