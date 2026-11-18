import json

import pytest

REREADER = ("--models", "openai/scripted-rereader", "--triggers", "every_turn", "--dedup", "true,false")


@pytest.fixture(scope="module")
def reread(run_sweep):
    return run_sweep(*REREADER, "--strategies", "none", "--budget", "40000")


@pytest.fixture(scope="module")
def recalled_twice(run_sweep):
    return run_sweep(*REREADER, "--strategies", "mask", "--hints", "keys")


@pytest.fixture(scope="module")
def searched_twice(run_sweep):
    return run_sweep(*REREADER, "--strategies", "mark_sweep", "--hints", "none", "--search", "true")


def live(sweep, row):
    return {o["id"]: not o["dead"] for o in sweep.trace(row)["heap"]}


def test_dedup_collects_the_older_read_of_a_file_that_was_read_again(reread):
    row = reread.row(dedup=True)

    assert reread.content(row, 5) == "[collected: msg 5 read_file notes/00.txt; call recall(5) to restore]"
    assert [(e["turn"], e["collected"]) for e in reread.trace(row)["events"] if e["collected"]] == [(13, [5])]


def test_dedup_keeps_the_newer_read_live(reread):
    row = reread.row(dedup=True)

    assert (live(reread, row)[5], live(reread, row)[25]) == (False, True)


def test_dedup_keeps_the_first_read_of_every_other_file_and_the_listing(reread):
    row = reread.row(dedup=True)

    assert sorted(i for i, alive in live(reread, row).items() if not alive) == [5]


def test_without_dedup_both_reads_of_the_file_stay_live(reread):
    row = reread.row(dedup=False)

    assert (live(reread, row)[5], live(reread, row)[25], reread.collected(row)) == (True, True, [[]] * 14)


@pytest.mark.parametrize("dedup, dedups, repeated_calls, success", [(True, 1, 1, True), (False, 0, 1, True)])
def test_the_row_counts_the_messages_dedup_collected(reread, dedup, dedups, repeated_calls, success):
    row = reread.row(dedup=dedup)

    assert (row["dedups"], row["repeated_calls"], row["success"]) == (dedups, repeated_calls, success)


def test_two_identical_recall_calls_both_keep_their_results(recalled_twice):
    row = recalled_twice.row(dedup=True)

    assert (row["recalls"], row["dedups"], live(recalled_twice, row)[29], live(recalled_twice, row)[31]) == (
        2, 0, True, True)


def test_two_identical_search_calls_both_keep_their_results(searched_twice):
    row = searched_twice.row(dedup=True)

    assert (row["searches"], row["dedups"], live(searched_twice, row)[29], live(searched_twice, row)[31]) == (
        2, 0, True, True)


def test_an_unknown_dedup_value_is_rejected_by_the_cli(run_sweep):
    proc = run_sweep("--dedup", "banana", check=False).proc

    assert proc.returncode == 2
    assert "unknown --dedup value" in proc.stderr


def test_resuming_skips_cells_recorded_before_the_dedup_field_existed(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--dedup", "false")
    runs = sweep.cwd / "results" / "runs.jsonl"
    legacy = [{k: v for k, v in json.loads(line).items() if k not in ("dedup", "dedups")}
              for line in runs.read_text().splitlines()]
    runs.write_text("".join(json.dumps(r) + "\n" for r in legacy))
    before = sweep.agent.requests

    sweep.rerun()

    assert (sweep.agent.requests - before, len(sweep.rows)) == (0, 1)


def test_a_dedup_cell_is_not_skipped_because_the_same_cell_ran_without_dedup(run_sweep):
    sweep = run_sweep("--strategies", "mark_sweep", "--hints", "none", "--dedup", "false")

    sweep.cli(*sweep.argv, "--dedup", "true")

    assert sorted(r["dedup"] for r in sweep.rows) == [False, True]
