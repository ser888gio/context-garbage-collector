import json

import pytest
from hypothesis import given, settings, strategies as hs

from cgc import collectors as C
from cgc.collectors import Obj


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


def aged_heap():
    return [sys_msg(), user(1, "Read files."),
            reads(2, "a.txt", 1), result(3, "a.txt", "x" * 400, 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "y" * 400, 2),
            reads(6, "c.txt", 4), result(7, "c.txt", "z" * 400, 4)]


def test_mask_stubs_only_tool_results_older_than_mask_turns():
    heap = aged_heap()

    C.mask(heap, {"budget": 100, "mask_turns": 3})

    assert [o.dead for o in heap[2:]] == [False, True, False, False, False, False]


def test_hybrid_summarizes_what_mask_left_into_the_oldest_message_and_keeps_structure():
    heap = aged_heap()

    C.hybrid(heap, {"budget": 100, "mask_turns": 10, "summarize": lambda parts: "SUM"})

    assert heap[2].msg["content"] == "[summary of msgs 2-5] SUM"
    assert [c["id"] for c in heap[2].msg["tool_calls"]] == ["c2"]
    assert [o.dead for o in heap[2:]] == [False, True, True, True, False, False]
    assert heap[3].msg["tool_call_id"] == "c2"


def test_hybrid_does_not_summarize_its_own_summary_again():
    heap = aged_heap()
    calls = []

    def summarize(parts):
        calls.append(parts)
        return "SUM"

    C.hybrid(heap, {"budget": 100, "mask_turns": 10, "summarize": summarize})
    C.hybrid(heap, {"budget": 100, "mask_turns": 10, "summarize": summarize})

    assert len(calls) == 1


def test_hybrid_without_summarize_only_masks():
    heap = aged_heap()

    C.hybrid(heap, {"budget": 100, "mask_turns": 3})

    assert [o.dead for o in heap[2:]] == [False, True, False, False, False, False]


def test_forget_kills_an_old_message_and_refuses_a_root():
    heap = [sys_msg(), user(1, "Read files."), reads(2, "a.txt", 1),
            result(3, "a.txt", "old text " * 50, 1), reads(4, "b.txt", 2), result(5, "b.txt", "fresh", 2)]

    done, refused = C.forget(heap, [3, 1, 5, 99])

    assert (done, refused, heap[3].dead, heap[1].dead) == ([3], [1, 5, 99], True, False)


def test_write_barrier_collects_the_superseded_message_and_keeps_the_newer_one():
    heap = [sys_msg(), user(1, "What is the code for ORION?"),
            reads(2, "a.txt", 1), result(3, "a.txt", "The access code for project ORION is 4821.", 1),
            reads(4, "b.txt", 2),
            result(5, "b.txt", "Correction: the access code for project ORION is now 7310; the old code 4821 is revoked.", 2),
            reads(6, "c.txt", 3), result(7, "c.txt", "latest", 3)]

    collected = C.write_barrier(heap, {"budget": 100})

    assert (3 in collected, heap[3].dead, heap[5].dead) == (True, True, False)


def test_mark_sweep_keeps_the_stale_message_that_write_barrier_collects():
    heap = [sys_msg(), user(1, "What is the code for ORION?"),
            reads(2, "a.txt", 1), result(3, "a.txt", "The access code for project ORION is 4821.", 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "The access code for project ORION is 7310.", 2),
            reads(6, "c.txt", 3), result(7, "c.txt", "latest", 3)]

    C.mark_sweep(heap, {"budget": 100})

    assert heap[3].dead is False


def test_write_barrier_keeps_facts_about_different_keys():
    heap = [sys_msg(), user(1, "What is the code for ORION or VEGA?"),
            reads(2, "a.txt", 1), result(3, "a.txt", "The access code for project ORION is 4821.", 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "The access code for project VEGA is 7310.", 2),
            reads(6, "c.txt", 3), result(7, "c.txt", "latest", 3)]

    C.write_barrier(heap, {"budget": 100})

    assert [heap[3].dead, heap[5].dead] == [False, False]


def note():
    return result(12, "notes/12.txt", "ORION code is 4821 and the spare is 90210. " + "plain filler words " * 10, 1)


@pytest.mark.parametrize("hints,stub", [
    ("none", "[collected: msg 12 read_file notes/12.txt; call recall(12) to restore]"),
    ("keys", "[collected: msg 12 read_file notes/12.txt — keys: 4821, 90210, ORION; call recall(12) to restore]"),
    ("preview", '[collected: msg 12 read_file notes/12.txt — keys: 4821, 90210, ORION — preview: "ORION code is 4821 and the spare is 90210. plain filler words"; call recall(12) to restore]'),
    ("numbers", "[collected: msg 12 read_file notes/12.txt — keys: 4821, 90210, ORION — numbers: 4821, 90210; call recall(12) to restore]"),
])
def test_stub_mode_decides_what_the_stub_carries(hints, stub):
    o = note()

    C.kill(o, hints=hints)

    assert o.msg["content"] == stub


def test_false_and_true_still_mean_none_and_keys():
    a, b = note(), note()

    C.kill(a, hints=False)
    C.kill(b, hints=True)

    assert (a.msg["content"], b.msg["content"]) == (
        "[collected: msg 12 read_file notes/12.txt; call recall(12) to restore]",
        "[collected: msg 12 read_file notes/12.txt — keys: 4821, 90210, ORION; call recall(12) to restore]")


def test_preview_stub_is_skipped_when_it_would_not_be_shorter():
    o = result(12, "notes/12.txt", "ORION is 4821", 1)

    C.kill(o, hints="preview")

    assert (o.msg["content"], o.saved) == ("ORION is 4821", None)


def fat_heap():
    h = [sys_msg(), user(1, "Find the ORION code in a.txt b.txt c.txt d.txt e.txt")]
    for i, p in zip(range(2, 11, 2), "abcde"):
        h += [reads(i, p + ".txt", i // 2), result(i + 1, p + ".txt", "word " * 200, i // 2)]
    return h + [reads(12, "f.txt", 6), result(13, "f.txt", "latest", 6)]


def test_hysteresis_collects_more_than_plain_threshold_on_the_same_heap():
    plain, low = fat_heap(), fat_heap()

    a = C.STRATEGIES["mark_sweep"](plain, {"budget": 1500})
    b = C.STRATEGIES["mark_sweep"](low, {"budget": 1500, "target": 0.4})

    assert (a, b) == ([], [3, 5, 7, 9, 11])


def test_hysteresis_stops_as_soon_as_size_is_at_or_below_target():
    heap = fat_heap()

    out = C.STRATEGIES["refcount"](heap, {"budget": 1500, "target": 0.7})

    assert (out, C.size(heap) <= 1050) == ([3, 5, 7], True)


def test_h2o_keeps_sinks_recent_turns_and_the_most_referenced_result():
    heap = [sys_msg(), user(1, "Read the notes."),
            reads(2, "a.txt", 1), result(3, "a.txt", "alpha " * 50, 1),
            reads(4, "b.txt", 2), result(5, "b.txt", "bravo " * 50, 2),
            reads(6, "c.txt", 3), result(7, "c.txt", "charlie " * 50, 3),
            says(8, "c.txt looks useful", 3), says(9, "back to c.txt", 4),
            reads(10, "d.txt", 4), result(11, "d.txt", "delta " * 50, 4),
            reads(12, "e.txt", 5), result(13, "e.txt", "echo " * 50, 5),
            reads(14, "f.txt", 6), result(15, "f.txt", "foxtrot " * 50, 6)]

    out = C.h2o(heap, {"budget": 100, "h2o_heavy": 1})

    assert out == [4, 5, 6, 8, 9, 10, 11]


def chain_heap():
    files = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
    heap = [sys_msg(), user(1, "Find the code for ORION."),
            reads(2, "list", 1),
            Obj(3, {"role": "tool", "tool_call_id": "c2", "content": "\n".join(files)}, 1,
                label="list_files", touched=1, last_used=1)]
    bodies = {"a.txt": "filler one", "b.txt": "ORION is the same as the code for project VEGA",
              "c.txt": "project VEGA is 7468", "d.txt": "filler two", "e.txt": "filler three"}
    for n, f in enumerate(files):
        heap += [reads(4 + 2 * n, f, 2 + n), result(5 + 2 * n, f, bodies[f], 2 + n)]
    heap += [says(14, "Checking.", 8), reads(15, "z.txt", 8), result(16, "z.txt", "latest", 8)]
    return heap


def dead_ids(heap):
    return [o.id for o in heap if o.dead]


def test_mark_sweep_collects_the_value_file_behind_a_link_file():
    heap = chain_heap()

    C.mark_sweep(heap, {"budget": 100})

    assert dead_ids(heap) == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]


def test_mark_sweep_chain_keeps_the_value_file_and_still_collects_the_filler_files():
    heap = chain_heap()

    C.mark_sweep_chain(heap, {"budget": 100})

    assert dead_ids(heap) == [2, 4, 5, 10, 11, 12, 13]
