import pytest

ALL = "none,truncate,lru_mask,mask,hybrid,refcount,mark_sweep,mark_sweep_chain,write_barrier,generational,h2o,oracle,manual"


@pytest.fixture(scope="module")
def matrix(run_sweep):
    return run_sweep("--strategies", ALL, "--hints", "keys")


@pytest.mark.parametrize("strategy, expected", [
    ("none", ("oom", False, 0, 2, 7)),
    ("truncate", ("answered", True, 1, 4, 14)),
    ("lru_mask", ("answered", True, 1, 9, 14)),
    ("mask", ("answered", True, 1, 9, 14)),
    ("hybrid", ("answered", True, 1, 5, 14)),
    ("refcount", ("answered", True, 1, 4, 14)),
    ("mark_sweep", ("answered", True, 1, 4, 14)),
    ("mark_sweep_chain", ("answered", True, 1, 4, 14)),
    ("write_barrier", ("answered", True, 1, 4, 14)),
    ("generational", ("answered", True, 1, 4, 14)),
    ("h2o", ("answered", True, 1, 9, 14)),
    ("oracle", ("answered", True, 0, 5, 13)),
    ("manual", ("answered", True, 1, 5, 19)),
])
def test_needle_run_ends_as_pinned_per_strategy(matrix, strategy, expected):
    row = matrix.row(strategy=strategy)

    assert (row["end"], row["success"], row["recalls"], row["gc_runs"], row["turns"]) == expected


@pytest.mark.parametrize("strategy, expected", [
    ("none", [[], []]),
    ("truncate", [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("lru_mask", [[5], [7], [9], [11], [13], [15], [17], [19], [21]]),
    ("mask", [[3, 5], [7], [9], [11], [13], [15], [17], [19], [21]]),
    ("hybrid", [[3, 5], [7, 4, 6, 8, 9, 10, 11], [13, 14, 15], [17, 18, 19], [21]]),
    ("refcount", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("mark_sweep", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("mark_sweep_chain", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("write_barrier", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("generational", [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]),
    ("h2o", [[4, 6], [5, 7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]]),
    ("oracle", [[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 16, 17], [18, 19], [20, 21]]),
    ("manual", [[], [], [], [], []]),
])
def test_each_gc_run_collects_the_pinned_message_ids(matrix, strategy, expected):
    assert matrix.collected(matrix.row(strategy=strategy)) == expected


def test_hybrid_summarizes_through_the_chat_endpoint(matrix):
    row = matrix.row(strategy="hybrid")

    assert (row["summary_calls"], row["success"]) == (3, True)


def test_hybrid_summary_replaces_the_content_of_the_oldest_collected_message(matrix):
    content = matrix.content(matrix.row(strategy="hybrid"), 2)

    assert content == "[summary of msgs 2-11] Summary: The access code for project ORION is 7634."


def test_manual_agent_forgets_old_results_when_the_memory_note_shows_high_usage(matrix):
    row = matrix.row(strategy="manual")

    assert (row["forgets"], row["recalls"], row["success"]) == (13, 1, True)


def test_manual_results_carry_their_message_id(matrix):
    row = matrix.row(strategy="manual")

    assert matrix.content(row, 19).startswith("[msg 19] ")


def test_manual_always_sends_the_memory_note_even_without_the_pressure_flag(matrix):
    assert matrix.notes[0] == "[memory] context 2% of budget (95/4000 tokens), 0 messages collected."


def test_manual_note_never_enters_the_stored_trace(matrix):
    assert "[memory]" not in matrix.trace_text(matrix.row(strategy="manual"))
