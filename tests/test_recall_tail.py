import json

import pytest

from cgc import collectors as C
from cgc.agent import call_tool

ARGS = ("--strategies", "mask", "--hints", "keys", "--triggers", "every_turn")


@pytest.fixture(scope="module")
def both(run_sweep):
    return run_sweep(*ARGS, "--recall", "in_place,tail")


def tool_results(sweep, row):
    return [o for o in sweep.trace(row)["heap"] if o["msg"]["role"] == "tool"]


def test_tail_recall_is_the_newest_tool_result_and_carries_the_original(both):
    row = both.row(recall="tail")

    newest = tool_results(both, row)[-1]["msg"]["content"]

    assert newest.startswith("[recalled msg 15] jacket copper quiet tunnel")
    assert "ORBITAL is 5242" in newest


def test_tail_recall_leaves_the_stub_where_it_was(both):
    row = both.row(recall="tail")

    assert both.content(row, 15).startswith("[collected: msg 15 read_file notes/")
    assert next(o["dead"] for o in both.trace(row)["heap"] if o["id"] == 15) is True


def test_in_place_recall_still_restores_the_message_and_answers_restored(both):
    row = both.row(recall="in_place")

    assert tool_results(both, row)[-1]["msg"]["content"] == "restored"
    assert "ORBITAL is 5242" in both.content(row, 15)


def test_the_row_records_the_recall_mode_and_both_modes_answer(both):
    assert sorted((r["recall"], r["recalls"], r["success"]) for r in both.rows) == [
        ("in_place", 1, True), ("tail", 1, True)]


def test_tail_recall_keeps_a_longer_stable_prefix_than_restoring_in_place(both):
    assert both.row(recall="tail")["stable_prefix_share"] > both.row(recall="in_place")["stable_prefix_share"]


@pytest.fixture
def heap():
    stub = C.Obj(5, {"role": "tool", "content": "[collected: msg 5]"}, 1, saved="the original", dead=True)
    live = C.Obj(6, {"role": "tool", "content": "still here"}, 1)
    return [stub, live]


@pytest.mark.parametrize("obj_id", [99, 6, -1])
def test_tail_recall_of_an_unknown_or_live_id_is_the_same_error_as_in_place(heap, obj_id):
    tail = call_tool("recall", {"id": obj_id}, {}, heap, 3, "tail")
    in_place = call_tool("recall", {"id": obj_id}, {}, heap, 3, "in_place")

    assert (tail, in_place) == ("error: nothing to restore", "error: nothing to restore")


def test_tail_recall_does_not_touch_the_original(heap):
    result = call_tool("recall", {"id": 5}, {}, heap, 9, "tail")

    assert (result, heap[0].dead, heap[0].saved, heap[0].touched, heap[0].last_used) == (
        "[recalled msg 5] the original", True, "the original", 0, 0)


def test_a_rerun_with_the_other_recall_mode_is_not_skipped(run_sweep):
    sweep = run_sweep(*ARGS, "--recall", "in_place")

    sweep.cli(*sweep.argv, "--recall", "tail")

    assert sorted(r["recall"] for r in sweep.rows) == ["in_place", "tail"]


def test_a_rerun_with_the_same_recall_mode_is_skipped(run_sweep):
    sweep = run_sweep(*ARGS, "--recall", "tail")
    before = sweep.agent.requests

    sweep.rerun()

    assert (sweep.agent.requests - before, len(sweep.rows)) == (0, 1)


def test_rows_from_before_the_recall_field_existed_count_as_in_place_when_resuming(run_sweep):
    sweep = run_sweep(*ARGS)
    runs = sweep.cwd / "results" / "runs.jsonl"
    runs.write_text("".join(json.dumps({k: v for k, v in json.loads(line).items() if k != "recall"}) + "\n"
                            for line in runs.read_text().splitlines()))
    before = sweep.agent.requests

    sweep.rerun()

    assert (sweep.agent.requests - before, len(sweep.rows)) == (0, 1)


def test_an_unknown_recall_value_is_rejected_by_the_cli(run_sweep):
    proc = run_sweep("--recall", "banana", check=False).proc

    assert proc.returncode == 2
    assert "unknown --recall value" in proc.stderr


def test_report_load_backfills_recall_for_old_rows(run_sweep):
    from analysis.report import load
    sweep = run_sweep(*ARGS)
    runs = sweep.cwd / "results" / "runs.jsonl"
    runs.write_text("".join(json.dumps({k: v for k, v in json.loads(line).items() if k != "recall"}) + "\n"
                            for line in runs.read_text().splitlines()))

    assert list(load(runs).recall) == ["in_place"]
