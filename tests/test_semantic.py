import pytest


@pytest.fixture(scope="module")
def semantic(run_sweep):
    return run_sweep("--strategies", "semantic", "--hints", "keys", "--budget", "8000")


def test_semantic_strategy_embeds_through_the_local_gemini_endpoint_and_answers(semantic):
    row = semantic.row()

    assert (row["end"], row["success"], row["recalls"], row["gc_runs"], row["embed_calls"]) == (
        "answered", True, 1, 3, 3)


def test_semantic_strategy_keeps_only_the_messages_closest_to_the_root_texts(semantic):
    assert semantic.collected(semantic.row()) == [[15, 5, 4, 6, 14, 3, 8, 10, 12, 2], [19, 17, 18, 16],
                                                  [21, 23, 22, 20]]
