import json

import pytest


@pytest.mark.parametrize("search, success, answer, searches", [
    ("true", True, "5242", 1),
    ("false", False, "I don't know", 0),
])
def test_search_lets_the_agent_find_a_collected_needle_without_hints(run_sweep, search, success, answer, searches):
    row = run_sweep("--strategies", "mark_sweep", "--hints", "none", "--search", search).rows[0]

    assert (row["end"], row["success"], row["answer"], row["recalls"], row["searches"]) == (
        "answered", success, answer, 0, searches)


def test_search_returns_the_collected_sentence_and_never_a_live_message(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--hints", "none", "--search", "true")

    assert sweep.content(sweep.rows[0], 27) == "msg 15: The access code for project ORBITAL is 5242"


def test_on_poison_a_live_correction_is_answered_without_searching_the_collected_revoked_code(run_sweep):
    row = run_sweep("--family", "poison", "--strategies", "mark_sweep", "--hints", "none", "--search", "true").rows[0]

    assert (row["success"], row["answer"], row["recalls"], row["searches"]) == (True, "1663", 0, 0)


def test_search_finds_text_that_only_survives_in_a_summarized_original(run_sweep):
    sweep = run_sweep("--family", "paraphrase", "--strategies", "hybrid", "--hints", "none", "--upfront", "true",
                      "--search", "true")
    row = sweep.rows[0]

    assert sweep.content(row, 2).startswith("[summary of msgs 2-9] Summary:")
    assert sweep.content(row, 25) == ("msg 2: Listing the notes to find the central forest ranger post\n"
                                      "msg 15: Project ORBITAL is the central forest ranger post")


@pytest.mark.parametrize("search, searches", [("true", 1), ("false", 0)])
def test_the_row_records_whether_search_was_offered_and_how_often_it_was_used(run_sweep, search, searches):
    row = run_sweep("--strategies", "mark_sweep", "--hints", "none", "--search", search).rows[0]

    assert (row["search"], row["searches"]) == (search == "true", searches)


def test_an_unknown_search_value_is_rejected_by_the_cli(run_sweep):
    proc = run_sweep("--search", "banana", check=False).proc

    assert proc.returncode == 2
    assert "unknown --search value" in proc.stderr


def test_without_search_the_model_gets_the_original_system_prompt_and_tools(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--search", "false")

    assert {tools == ["list_files", "read_file", "recall"] for _, tools in sweep.offered} == {True}
    assert {system for system, _ in sweep.offered} == {
        "You are a careful agent working with tools. To save space, older messages may be replaced by "
        "'[collected: msg N ...]' stubs. If you need what a stub held, call recall with its id."}


def test_with_search_the_model_gets_one_more_tool_and_one_more_sentence(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--search", "true")

    assert {tools == ["list_files", "read_file", "recall", "search"] for _, tools in sweep.offered} == {True}
    assert {system.endswith(" You can call search(query) to find text in collected messages without restoring them.")
            for system, _ in sweep.offered} == {True}


def test_resuming_skips_cells_recorded_before_the_search_field_existed(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--search", "false")
    runs = sweep.cwd / "results" / "runs.jsonl"
    legacy = [{k: v for k, v in json.loads(line).items() if k not in ("search", "searches")}
              for line in runs.read_text().splitlines()]
    runs.write_text("".join(json.dumps(r) + "\n" for r in legacy))
    before = sweep.agent.requests

    sweep.rerun()

    assert (sweep.agent.requests - before, len(sweep.rows)) == (0, 1)


def test_a_search_cell_is_not_skipped_because_the_same_cell_ran_without_search(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--hints", "none", "--search", "false")

    sweep.cli(*sweep.argv, "--search", "true")

    assert sorted(r["search"] for r in sweep.rows) == [False, True]
