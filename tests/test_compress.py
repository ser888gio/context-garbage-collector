import pytest

COMPRESS = ("--strategies", "compress", "--hints", "keys", "--budget", "8000")
BLOCK = ("[compressed msgs 3-21] The access code for project ORPHEUS is 9376. "
         "The access code for project ORBITAL is 5242. (earlier: The access code for project ORION is 7634.)")


@pytest.fixture(scope="module")
def twice(run_sweep):
    return run_sweep(*COMPRESS)


@pytest.fixture(scope="module")
def lossy(run_sweep):
    return run_sweep("--models", "openai/scripted-lossy", *COMPRESS)


@pytest.fixture(scope="module")
def greedy(run_sweep):
    return run_sweep("--models", "openai/scripted-greedy", *COMPRESS)


@pytest.fixture(scope="module")
def backwards(run_sweep):
    return run_sweep("--models", "openai/scripted-backwards", *COMPRESS)


@pytest.fixture(scope="module")
def rootsonly(run_sweep):
    return run_sweep("--models", "openai/scripted-rootsonly", *COMPRESS)


@pytest.fixture(scope="module")
def coalesced(run_sweep):
    return run_sweep(*COMPRESS, "--coalesce", "true")


@pytest.fixture(scope="module")
def pressured(run_sweep):
    return run_sweep(*COMPRESS, "--pressure", "true")


@pytest.fixture(scope="module")
def manual(run_sweep):
    return run_sweep(*COMPRESS, "--strategies", "manual")


@pytest.fixture(scope="module")
def other(run_sweep):
    return run_sweep(*COMPRESS, "--strategies", "mark_sweep")


def entry(sweep, msg_id):
    return next(o for o in sweep.trace(sweep.rows[0])["heap"] if o["id"] == msg_id)


def texts(messages):
    return [m.get("content") or "" for m in messages]


def markers(messages):
    return [t for t in texts(messages) if t.startswith(("[collected", "[compressed"))]


def test_a_compress_run_answers_correctly(twice):
    row = twice.rows[0]

    assert (row["end"], row["success"], row["answer"]) == ("answered", True, "5242")


def test_the_row_counts_compress_calls_and_compressed_messages(twice):
    row = twice.rows[0]

    assert (row["compressions"], row["compressed_msgs"]) == (2, 20)


def test_the_summary_replaces_the_first_compressed_message_with_a_block(backwards):
    assert backwards.content(backwards.rows[0], 3) == (
        "[compressed msgs 3-21] The access code for project ORION is 7634. "
        "The access code for project ORPHEUS is 9376. The access code for project ORBITAL is 5242.")


def test_compressing_over_an_earlier_block_keeps_its_summary_inside_the_new_one(twice):
    assert twice.content(twice.rows[0], 3) == BLOCK


def test_the_summary_block_is_shown_to_the_model_as_it_is_stored(twice):
    assert [t for t in texts(twice.seen[-1]) if t.startswith("[compressed")] == [BLOCK]


def test_the_original_of_the_block_message_stays_in_the_trace(twice):
    assert entry(twice, 3)["saved"].startswith("[msg 3] notes/00.txt\nnotes/01.txt")


def test_the_block_stays_collected_rather_than_live(twice):
    assert entry(twice, 3)["dead"] is True


def test_the_other_compressed_messages_become_recallable_stubs_with_hints(twice):
    assert twice.content(twice.rows[0], 9) == (
        "[collected: msg 9 read_file notes/02.txt — keys: 7634, ORION; call recall(9) to restore]")


def test_a_recall_restores_a_message_that_a_compress_collected(twice):
    row = twice.rows[0]

    assert (row["recalls"], twice.content(row, 17).startswith("[msg 17] jacket copper quiet tunnel")) == (1, True)


def test_a_compressed_message_nobody_recalled_stays_a_stub(twice):
    assert twice.content(twice.rows[0], 19) == (
        "[collected: msg 19 read_file notes/06.txt; call recall(19) to restore]")


def test_an_agent_whose_summary_dropped_the_codes_recalls_the_original_and_still_answers(lossy):
    row = lossy.rows[0]

    assert (row["success"], row["recalls"], lossy.content(row, 3)) == (
        True, 1, "[compressed msgs 3-21] Notes read, nothing important. (earlier: Notes read, nothing important.)")


def test_compressing_removes_no_message_so_every_tool_call_keeps_its_result(twice):
    assert "".join(m["role"][0] for m in twice.seen[-1]) == "suatatatatatatatatatatatatatauat"


def test_the_compress_result_carries_its_message_id_and_the_ids_it_compressed(twice):
    assert twice.content(twice.rows[0], 25) == (
        "[msg 25] compressed [3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] into msg 3")


def test_a_range_over_every_id_skips_the_roots(greedy):
    assert (entry(greedy, 0)["dead"], entry(greedy, 1)["dead"], greedy.content(greedy.rows[0], 1)[:24]) == (
        False, False, "The notes/ folder has 10")


def test_a_range_over_every_id_skips_the_newest_turn(greedy):
    assert (entry(greedy, 14)["dead"], entry(greedy, 14)["msg"]["tool_calls"][0]["function"]["name"]) == (
        False, "compress")


def test_a_range_over_every_id_compresses_all_the_others(greedy):
    assert (greedy.rows[0]["compressed_msgs"], greedy.content(greedy.rows[0], 2)) == (
        12, "[compressed msgs 2-13] The access code for project ORION is 7634.")


def test_a_range_of_only_roots_is_refused_with_the_reason(rootsonly):
    assert rootsonly.seen[7][-1]["content"] == (
        "[msg 15] refused: nothing to compress: ids are roots, newest turn, unknown or already collected")


def test_a_refused_range_adds_nothing_to_the_counters(rootsonly):
    assert (rootsonly.rows[0]["compressions"], rootsonly.rows[0]["compressed_msgs"]) == (1, 19)


def test_a_refused_range_leaves_the_context_untouched(rootsonly):
    assert markers(rootsonly.seen[7]) == []


def test_a_range_that_runs_backwards_is_refused_with_the_reason(backwards):
    assert backwards.seen[7][-1]["content"] == "[msg 15] refused: from_id is after to_id"


def test_a_backwards_range_leaves_the_context_untouched(backwards):
    assert markers(backwards.seen[7]) == []


def test_no_note_is_sent_while_context_is_under_half_the_budget(twice):
    assert twice.noted[:6] == [False] * 6


def test_the_note_is_sent_at_the_first_call_above_half_the_budget(twice):
    assert twice.notes[0] == "[memory] context 55% of budget (4436/8000 tokens), 0 messages collected."


def test_after_crossing_half_the_budget_the_note_is_sent_every_fifth_call(twice):
    assert twice.noted == [False] * 6 + [True] + [False] * 4 + [True] + [False] * 4


def test_the_note_stays_on_every_fifth_call_while_context_stays_above_half_the_budget(backwards):
    assert backwards.noted == [False] * 6 + [True] + [False] * 4 + [True] + [False] * 4


def test_manual_still_gets_the_note_on_every_call(manual):
    assert manual.noted == [True] * 14


def test_pressure_true_still_sends_the_note_on_every_call_for_compress(pressured):
    assert pressured.noted == [True] * 16


def test_the_note_never_enters_the_stored_trace(twice):
    assert "[memory]" not in twice.trace_text(twice.rows[0])


def test_compress_offers_the_compress_tool_and_not_forget(twice):
    assert twice.offered[0][1] == ["compress", "list_files", "read_file", "recall"]


def test_manual_offers_forget_and_not_compress(manual):
    assert manual.offered[0][1] == ["forget", "list_files", "read_file", "recall"]


def test_other_strategies_offer_neither_tool(other):
    assert other.offered[0][1] == ["list_files", "read_file", "recall"]


def test_other_strategies_record_zero_compress_counters(other):
    assert (other.rows[0]["compressions"], other.rows[0]["compressed_msgs"]) == (0, 0)


def test_other_strategies_do_not_prefix_results_with_message_ids(other):
    assert other.seen[1][3]["content"].startswith("notes/00.txt")


def test_with_coalesce_the_summary_block_stays_its_own_message(coalesced):
    assert [(m["role"], m["content"]) for m in coalesced.seen[-1] if (m.get("content") or "").startswith("[compressed")] == [
        ("tool", BLOCK)]


def test_with_coalesce_the_other_stubs_are_still_grouped_into_runs(coalesced):
    grouped = [t[:33] for t in texts(coalesced.seen[-1]) if t.startswith("[collected ")]

    assert grouped == [
        "[collected 6 earlier tool results", "[collected 2 earlier tool results"]
