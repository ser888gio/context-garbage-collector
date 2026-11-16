import pytest


@pytest.fixture(scope="module")
def coalescing(run_sweep):
    return run_sweep("--family", "needle,poison", "--seeds", "2", "--strategies", "mark_sweep", "--hints", "pointer",
                     "--coalesce", "true,false")


@pytest.fixture(scope="module")
def pressure(run_sweep):
    return run_sweep("--strategies", "mark_sweep", "--hints", "keys", "--pressure", "true")


@pytest.fixture(scope="module")
def triggers(run_sweep):
    return run_sweep("--strategies", "mark_sweep", "--hints", "keys", "--triggers", "every_turn,threshold:0.7:0.4")


@pytest.fixture(scope="module")
def upfront(run_sweep):
    return run_sweep("--strategies", "mark_sweep", "--hints", "keys", "--upfront", "true")


@pytest.mark.parametrize("family, expected", [
    ("needle", ("answered", True, "3770", 1)),
    ("poison", ("answered", True, "2390", 1)),
])
def test_agent_recalls_from_a_coalesced_message_and_still_answers(coalescing, family, expected):
    row = coalescing.row(task_family=family, coalesce=True)

    assert (row["end"], row["success"], row["answer"], row["recalls"]) == expected


def test_coalescing_shrinks_the_prompts_sent_to_the_model(coalescing):
    coalesced = coalescing.row(task_family="needle", coalesce=True)
    plain = coalescing.row(task_family="needle", coalesce=False)

    assert (coalesced["input_tokens"], plain["input_tokens"]) == (19584, 22939)


def test_coalescing_changes_what_the_model_is_sent_not_what_is_collected(coalescing):
    coalesced = coalescing.row(task_family="needle", coalesce=True)
    plain = coalescing.row(task_family="needle", coalesce=False)

    assert coalescing.collected(coalesced) == coalescing.collected(plain)


def test_pressure_note_is_sent_with_every_model_call(pressure):
    assert (len(pressure.notes), pressure.notes[0], pressure.notes[3]) == (
        14, "[memory] context 2% of budget (95/4000 tokens), 0 messages collected.",
        "[memory] context 46% of budget (1866/4000 tokens), 0 messages collected.")


def test_pressure_note_is_never_stored_in_the_trace(pressure):
    assert "[memory]" not in pressure.trace_text(pressure.row())


def test_pressure_note_does_not_change_what_is_collected(pressure):
    assert pressure.collected(pressure.row()) == [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17],
                                                  [18, 19, 20, 21]]


def test_every_turn_trigger_runs_the_collector_before_every_model_call(triggers):
    row = triggers.row(trigger="every_turn")

    assert (row["gc_runs"], triggers.collected(row)) == (14, [
        [], [], [2], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [3, 22, 23],
        []])


def test_threshold_trigger_with_a_target_also_collects_reachable_results_down_to_it(triggers):
    row = triggers.row(trigger="threshold:0.7:0.4")

    assert (row["gc_runs"], triggers.collected(row)) == (4, [
        [2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21, 3]])


def test_upfront_question_makes_the_agent_answer_without_a_recall(upfront):
    row = upfront.row()

    assert (row["end"], row["success"], row["recalls"], row["gc_runs"], row["turns"]) == ("answered", True, 0, 5, 12)


def test_upfront_question_keeps_the_needle_out_of_the_collected_set(upfront):
    assert upfront.collected(upfront.row()) == [[2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13], [14, 16, 17], [18, 19],
                                                [20, 21]]
