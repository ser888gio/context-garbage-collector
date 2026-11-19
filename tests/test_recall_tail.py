import pytest

from conftest import converse

from cgc.context import Context
from cgc.tasks import needle_task

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


def conversation(recall):
    ctx = Context(strategy="mask", trigger="every_turn", budget=4000, hints="keys", recall=recall)
    _, history, _ = converse(ctx, needle_task(0, 10, upfront=True))
    return ctx, history


@pytest.mark.parametrize("obj_id", [99, 1, -1])
def test_recall_of_an_unknown_or_live_id_is_the_same_error_in_both_modes(obj_id):
    results = [conversation(mode)[0].call("recall", {"id": obj_id}) for mode in ("tail", "in_place")]

    assert results == ["error: nothing to restore", "error: nothing to restore"]


def test_tail_recall_does_not_touch_the_stub():
    ctx, history = conversation("tail")

    ctx.call("recall", {"id": 15})

    assert ctx.prepare(history)[15]["content"].startswith("[collected: msg 15 read_file notes/05.txt")


def test_in_place_recall_puts_the_original_back_where_it_was():
    ctx, history = conversation("in_place")

    result = ctx.call("recall", {"id": 5})
    history.append({"role": "user", "content": "go on"})

    assert (result, ctx.prepare(history)[5]["content"]) == ("restored", history[5]["content"])


def test_a_rerun_with_the_other_recall_mode_is_not_skipped(run_sweep):
    sweep = run_sweep(*ARGS, "--recall", "in_place")

    sweep.cli(*sweep.argv, "--recall", "tail")

    assert sorted(r["recall"] for r in sweep.rows) == ["in_place", "tail"]


def test_a_rerun_with_the_same_recall_mode_is_skipped(run_sweep):
    sweep = run_sweep(*ARGS, "--recall", "tail")
    before = sweep.agent.requests

    sweep.rerun()

    assert (sweep.agent.requests - before, len(sweep.rows)) == (0, 1)


def test_an_unknown_recall_value_is_rejected_by_the_cli(run_sweep):
    proc = run_sweep("--recall", "banana", check=False).proc

    assert proc.returncode == 2
    assert "unknown --recall value" in proc.stderr
