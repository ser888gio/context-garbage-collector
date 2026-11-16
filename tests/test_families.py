import pytest

FOUR = "mark_sweep,mark_sweep_chain,write_barrier,oracle"


@pytest.fixture(scope="module")
def links_and_paraphrases(run_sweep):
    return run_sweep("--family", "multihop,paraphrase", "--strategies", FOUR, "--hints", "keys")


@pytest.fixture(scope="module")
def upfront_links(run_sweep):
    return run_sweep("--family", "multihop", "--upfront", "true", "--strategies", FOUR, "--hints", "keys")


@pytest.fixture(scope="module")
def corrections(run_sweep):
    return run_sweep("--family", "poison", "--seeds", "2", "--strategies", FOUR, "--hints", "keys")


@pytest.mark.parametrize("family, strategy, expected", [
    ("multihop", "mark_sweep", ("answered", True, "7890", 1, 4, 14)),
    ("multihop", "mark_sweep_chain", ("answered", True, "7890", 1, 4, 14)),
    ("multihop", "write_barrier", ("answered", True, "7890", 1, 4, 14)),
    ("multihop", "oracle", ("answered", True, "7890", 1, 4, 14)),
    ("paraphrase", "mark_sweep", ("answered", True, "5969", 2, 5, 15)),
    ("paraphrase", "mark_sweep_chain", ("answered", True, "5969", 2, 5, 15)),
    ("paraphrase", "write_barrier", ("answered", True, "5969", 2, 5, 15)),
    ("paraphrase", "oracle", ("answered", True, "5969", 0, 5, 13)),
])
def test_agent_resolves_links_and_descriptions_through_recalls(links_and_paraphrases, family, strategy, expected):
    row = links_and_paraphrases.row(task_family=family, strategy=strategy)

    assert (row["end"], row["success"], row["answer"], row["recalls"], row["gc_runs"], row["turns"]) == expected


@pytest.mark.parametrize("family, strategy, expected", [
    ("multihop", "mark_sweep", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("multihop", "oracle", [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("paraphrase", "mark_sweep", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21],
                                  [3, 9, 22, 23, 26, 27]]),
    ("paraphrase", "mark_sweep_chain", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21],
                                        [9, 22, 23, 26, 27]]),
    ("paraphrase", "oracle", [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 16, 17], [18, 19], [20, 21]]),
])
def test_link_and_paraphrase_runs_collect_the_pinned_message_ids(links_and_paraphrases, family, strategy, expected):
    row = links_and_paraphrases.row(task_family=family, strategy=strategy)

    assert links_and_paraphrases.collected(row) == expected


@pytest.mark.parametrize("strategy, expected", [
    ("mark_sweep", ("answered", True, "7890", 0, 5, 12)),
    ("mark_sweep_chain", ("answered", True, "7890", 0, 5, 12)),
    ("write_barrier", ("answered", True, "7890", 0, 5, 12)),
    ("oracle", ("answered", True, "7890", 1, 4, 13)),
])
def test_upfront_question_lets_the_agent_follow_the_link_without_recalls(upfront_links, strategy, expected):
    row = upfront_links.row(strategy=strategy)

    assert (row["end"], row["success"], row["answer"], row["recalls"], row["gc_runs"], row["turns"]) == expected


@pytest.mark.parametrize("strategy, expected", [
    ("mark_sweep", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 16, 17], [18, 19], [20, 21]]),
    ("mark_sweep_chain", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [16, 17], [18, 19], [20, 21]]),
])
def test_chain_marking_keeps_the_link_file_that_plain_mark_sweep_collects(upfront_links, strategy, expected):
    assert upfront_links.collected(upfront_links.row(strategy=strategy)) == expected


@pytest.mark.parametrize("strategy, expected", [
    ("mark_sweep", ("answered", True, "2390", 2, 5, 15)),
    ("mark_sweep_chain", ("answered", True, "2390", 2, 5, 15)),
    ("write_barrier", ("answered", True, "2390", 2, 5, 15)),
    ("oracle", ("answered", True, "2390", 0, 4, 13)),
])
def test_agent_prefers_the_correction_over_the_revoked_code(corrections, strategy, expected):
    row = corrections.row(strategy=strategy)

    assert (row["end"], row["success"], row["answer"], row["recalls"], row["gc_runs"], row["turns"]) == expected


@pytest.mark.parametrize("strategy, expected", [
    ("mark_sweep", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21], [3, 22, 23, 26, 27]]),
    ("mark_sweep_chain", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21],
                          [22, 23, 26, 27]]),
    ("write_barrier", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21],
                       [15, 3, 22, 23, 26, 27]]),
    ("oracle", [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20]]),
])
def test_poison_runs_collect_the_pinned_message_ids(corrections, strategy, expected):
    assert corrections.collected(corrections.row(strategy=strategy)) == expected
