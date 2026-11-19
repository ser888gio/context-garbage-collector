import pytest
from conftest import converse

from cgc.context import Context
from cgc.tasks import needle_task

STUB_15 = "[collected: msg 15 read_file notes/05.txt — keys: 5242, ORBITAL; call recall(15) to restore]"


def collecting_context(**options):
    return Context(strategy="mask", trigger="every_turn", budget=4000, hints="keys", **options)


@pytest.fixture
def conversation():
    ctx = collecting_context()
    answer, history, sent = converse(ctx, needle_task(0, 10, upfront=True))
    return ctx, answer, history, sent


def test_old_tool_results_reach_the_model_as_stubs(conversation):
    _, _, _, sent = conversation

    assert STUB_15 in [m["content"] for m in sent[-1]]


def test_the_agent_answers_correctly_through_the_library(conversation):
    assert conversation[1] == "5242"


def test_stats_count_every_collection_recall_and_search(conversation):
    assert conversation[0].stats == {"gc_runs": 14, "tokens_freed": 6951, "recalls": 1, "searches": 1, "dedups": 0}


def test_recall_returns_the_original_message_content(conversation):
    ctx, _, history, _ = conversation

    assert ctx.call("recall", {"id": 15}) == f"[recalled msg 15] {history[15]['content']}"


def test_search_returns_the_matching_sentence_from_a_collected_message(conversation):
    assert conversation[0].call("search", {"query": "ORBITAL"}) == "msg 15: The access code for project ORBITAL is 5242"


def test_the_callers_history_is_never_modified(conversation):
    _, _, history, _ = conversation

    assert history[15]["content"].startswith("jacket copper quiet tunnel")


def test_preparing_an_unchanged_history_again_adds_nothing(conversation):
    ctx, _, history, _ = conversation

    assert ctx.prepare(history) == ctx.prepare(history)


def test_an_edited_history_is_rebuilt_from_scratch(conversation):
    ctx, _, history, _ = conversation
    history[1] = {"role": "user", "content": "a different task"}

    prepared = ctx.prepare(history)

    assert [m["content"] for m in prepared if "call recall(" in (m["content"] or "")] == []
    assert ctx.call("recall", {"id": 15}) == "error: nothing to restore"


def test_a_truncated_history_is_rebuilt_from_scratch(conversation):
    ctx, _, history, _ = conversation

    assert ctx.prepare(history[:4]) == history[:4]


def test_tools_offered_are_recall_and_search_only_when_search_is_on():
    assert ([t["function"]["name"] for t in Context(search=True).tools],
            [t["function"]["name"] for t in Context(search=False).tools]) == (["recall", "search"], ["recall"])


@pytest.mark.parametrize("name,options", [("list_files", {}), ("search", {"search": False})])
def test_calls_that_are_not_ours_return_none_so_callers_can_chain(name, options):
    assert Context(**options).call(name, {}) is None


@pytest.mark.parametrize("options", [{"strategy": "banana"}, {"strategy": "semantic"}, {"strategy": "oracle"}])
def test_strategies_that_cannot_run_are_refused_up_front(options):
    with pytest.raises(ValueError):
        Context(**options)
