import json

import pytest
from hypothesis import given, settings, strategies as hs

import collectors as C
from collectors import Obj


def sys_msg(i=0, content="You are an agent."):
    return Obj(i, {"role": "system", "content": content}, 0, root=True)


def user(i, content, turn=0):
    return Obj(i, {"role": "user", "content": content}, turn, root=True, touched=turn, last_used=turn)


def reads(i, path, turn):
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]}
    return Obj(i, msg, turn, touched=turn, last_used=turn)


def says(i, content, turn):
    return Obj(i, {"role": "assistant", "content": content}, turn, touched=turn, last_used=turn)


def result(i, path, content, turn):
    return Obj(i, {"role": "tool", "tool_call_id": f"c{i - 1}", "content": content}, turn,
               label=f"read_file {path}", touched=turn, last_used=turn)


def test_mark_sweep_keeps_file_mentioning_what_the_task_asks_about():
    heap = [sys_msg(), user(1, "Find the code for ORION."),
            reads(2, "a.txt", 1), result(3, "a.txt", "ORION code is 4821", 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "nothing here", 2),
            reads(6, "c.txt", 3), result(7, "c.txt", "latest", 3)]

    C.mark_sweep(heap, {"budget": 100})

    assert [heap[3].dead, heap[5].dead] == [False, True]


def test_newest_turn_is_never_collected_even_when_unreferenced():
    heap = [sys_msg(), user(1, "Read files."), reads(2, "a.txt", 1), result(3, "a.txt", "fresh text", 1)]

    C.mark_sweep(heap, {"budget": 100})

    assert heap[3].dead is False


def test_refcount_leaks_a_cycle_that_mark_sweep_collects():
    def cycle_heap():
        return [sys_msg(), user(1, "Read files."),
                says(2, "Noted TAURUS.", 1), says(3, "TAURUS again.", 2),
                reads(4, "z.txt", 3), result(5, "z.txt", "latest", 3)]
    rc, ms = cycle_heap(), cycle_heap()

    C.refcount(rc, {"budget": 100})
    C.mark_sweep(ms, {"budget": 100})

    assert [rc[2].dead, rc[3].dead] == [False, False]
    assert [ms[2].dead, ms[3].dead] == [True, True]


def test_refcount_cascades_once_the_only_referencer_dies():
    heap = [sys_msg(), user(1, "Read files."),
            reads(2, "a.txt", 1), result(3, "a.txt", "old data", 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "latest", 2)]

    C.refcount(heap, {"budget": 100})

    assert [heap[2].dead, heap[3].dead] == [True, True]


def test_stub_keeps_tool_call_id_so_the_api_still_pairs_calls_and_results():
    heap = [sys_msg(), user(1, "Read."), reads(2, "a.txt", 1), result(3, "a.txt", "x" * 100, 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "latest", 2)]

    C.mark_sweep(heap, {"budget": 100})

    assert heap[3].msg == {"role": "tool", "tool_call_id": "c2",
                           "content": "[collected: msg 3 read_file a.txt; call recall(3) to restore]"}


def test_recalled_message_survives_the_next_collection():
    heap = [sys_msg(), user(1, "Read."), reads(2, "a.txt", 1), result(3, "a.txt", "old data " * 20, 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "latest", 2)]
    C.mark_sweep(heap, {"budget": 100})

    assert C.restore(heap, 3, 2) is True
    C.mark_sweep(heap, {"budget": 100})

    assert heap[3].dead is False


def test_restore_of_a_live_message_reports_nothing_to_restore():
    heap = [sys_msg(), user(1, "Read.")]

    assert C.restore(heap, 1, 0) is False


def test_truncate_collects_oldest_first_and_stops_at_target():
    heap = [sys_msg(), user(1, "Read."),
            reads(2, "a.txt", 1), result(3, "a.txt", "a" * 400, 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "b" * 400, 2),
            reads(6, "c.txt", 3), result(7, "c.txt", "c" * 400, 3)]

    C.truncate(heap, {"budget": 800, "target": 0.5})

    assert [o.dead for o in heap] == [False, False, True, True, False, False, False, False]


def test_lru_masks_tool_output_not_mentioned_for_three_turns():
    heap = [sys_msg(), user(1, "Read."),
            reads(2, "a.txt", 1), result(3, "a.txt", "alpha", 1),
            reads(4, "b.txt", 1), result(5, "b.txt", "beta", 1),
            says(6, "Back to b.txt for a moment.", 3),
            reads(8, "c.txt", 4), result(9, "c.txt", "latest", 4)]

    C.lru_mask(heap, {"budget": 100, "lru_turns": 3})

    assert [heap[3].dead, heap[5].dead] == [True, False]


def test_generational_minor_gc_spares_promoted_objects():
    heap = [sys_msg(), user(1, "Read."),
            says(2, "old note", 1), reads(3, "b.txt", 2), result(4, "b.txt", "latest", 2)]
    heap[2].survived = 2

    C.generational(heap, {"budget": 10_000, "threshold": 0.7, "promote_after": 2})

    assert heap[2].dead is False


def test_generational_major_gc_collects_promoted_garbage_under_pressure():
    heap = [sys_msg(), user(1, "Read."),
            says(2, "old note " * 50, 1), reads(3, "b.txt", 2), result(4, "b.txt", "latest", 2)]
    heap[2].survived = 2

    C.generational(heap, {"budget": 100, "threshold": 0.7, "promote_after": 2})

    assert heap[2].dead is True


def test_generational_promotes_reachable_young_objects():
    heap = [sys_msg(), user(1, "Find ORION."), result(2, "a.txt", "ORION 4821", 1),
            reads(3, "b.txt", 2), result(4, "b.txt", "latest", 2)]

    C.generational(heap, {"budget": 10_000, "threshold": 0.7, "promote_after": 2})

    assert heap[2].survived == 1


# Any heap: ordered turns, mixed roles, random key-bearing content.
contents_st = hs.sampled_from(["plain words", "ORION 4821", "see a.txt", "TAURUS", "b.txt and ORION", "x" * 300])


@hs.composite
def heaps(draw):
    heap = [sys_msg(), user(1, draw(contents_st))]
    turn = 0
    for i in range(2, draw(hs.integers(2, 14)) * 2, 2):
        turn += draw(hs.integers(0, 1))
        path = draw(hs.sampled_from(["a.txt", "b.txt", "c.txt"]))
        heap += [reads(i, path, turn), result(i + 1, path, draw(contents_st), turn)]
    return heap


@pytest.mark.parametrize("name", list(C.STRATEGIES))
@settings(max_examples=60, deadline=None)
@given(heap=heaps(), budget=hs.integers(50, 3000))
def test_no_strategy_touches_roots_newest_turn_or_message_structure(name, heap, budget):
    newest = max(o.turn for o in heap)
    protected_before = [(o.id, o.msg["content"]) for o in heap if o.root or o.turn == newest]
    shape_before = [(o.msg["role"], o.msg.get("tool_call_id"), o.msg.get("tool_calls")) for o in heap]

    C.STRATEGIES[name](heap, {"budget": budget})

    assert [(o.id, o.msg["content"]) for o in heap if o.root or o.turn == newest] == protected_before
    assert [(o.msg["role"], o.msg.get("tool_call_id"), o.msg.get("tool_calls")) for o in heap] == shape_before


@pytest.mark.parametrize("name", list(C.STRATEGIES))
@settings(max_examples=60, deadline=None)
@given(heap=heaps(), budget=hs.integers(50, 3000))
def test_every_collected_message_can_be_restored_verbatim(name, heap, budget):
    original = {o.id: o.msg["content"] for o in heap}

    C.STRATEGIES[name](heap, {"budget": budget})
    for o in heap:
        if o.saved is not None:
            C.restore(heap, o.id, 99)

    assert {o.id: o.msg["content"] for o in heap} == original


def test_hint_stub_lists_sorted_keys_of_the_collected_message():
    o = result(12, "notes/12.txt", "ORION code is 4821. " + "plain filler words " * 10, 1)

    C.kill(o, hints=True)

    assert o.msg["content"] == "[collected: msg 12 read_file notes/12.txt — keys: 4821, ORION; call recall(12) to restore]"


def test_hint_stub_has_no_keys_part_when_content_has_no_keys():
    o = result(12, "notes/12.txt", "just some plain lowercase words " * 5, 1)
    o.label = "read_file"

    C.kill(o, hints=True)

    assert o.msg["content"] == "[collected: msg 12 read_file; call recall(12) to restore]"


def test_oracle_keeps_the_message_with_the_key_and_collects_an_unrelated_old_one():
    heap = [sys_msg(), user(1, "go"), result(2, "a.txt", "code is 4242 here", 1),
            result(3, "b.txt", "nothing relevant " * 20, 1), says(4, "ok", 5)]

    out = C.oracle(heap, {"oracle_keys": {"4242"}})

    assert out == [3]
    assert heap[2].msg["content"] == "code is 4242 here"


def test_forget_kills_an_old_message_and_refuses_a_root():
    heap = [sys_msg(), user(1, "Read files."), reads(2, "a.txt", 1),
            result(3, "a.txt", "old text " * 50, 1), reads(4, "b.txt", 2), result(5, "b.txt", "fresh", 2)]

    done, refused = C.forget(heap, [3, 1, 5, 99])

    assert (done, refused, heap[3].dead, heap[1].dead) == ([3], [1, 5, 99], True, False)
