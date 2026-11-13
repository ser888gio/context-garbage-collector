from analysis.timeline import states


def msg(id, turn, role, **kw):
    return {"id": id, "turn": turn, "root": False, "dead": False, "label": "", "saved": None,
            "msg": {"role": role, **kw}}


def recall(id, turn, target):
    call = {"id": "c", "type": "function", "function": {"name": "recall", "arguments": '{"id": %d}' % target}}
    return msg(id, turn, "assistant", content=None, tool_calls=[call])


def test_collect_recall_collect_grid():
    trace = {
        "row": {"turns": 4},
        "events": [{"turn": 1, "before": 9, "after": 5, "collected": [0]},
                   {"turn": 4, "before": 9, "after": 5, "collected": [0]}],
        "heap": [msg(0, 0, "tool", content="x"), msg(1, 2, "user", content="q"), recall(2, 3, 0)],
    }
    assert states(trace) == [
        [1, 2, 2, 3, 2],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]


def test_recall_of_live_message_changes_nothing():
    trace = {
        "row": {"turns": 2},
        "events": [],
        "heap": [msg(0, 0, "tool", content="x"), recall(1, 1, 0)],
    }
    assert states(trace) == [[1, 1, 1], [0, 1, 1]]
