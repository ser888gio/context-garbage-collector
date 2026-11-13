import json

import audit


def msg(i, role, content=None, saved=None, calls=None):
    m = {"role": role, "content": content}
    if calls:
        m["tool_calls"] = [{"function": {"name": "recall", "arguments": json.dumps({"id": c})}} for c in calls]
    return {"id": i, "turn": i, "root": False, "dead": False, "label": "", "msg": m, "saved": saved}


TRACE = {
    "row": {"run_id": "r1", "model": "m", "strategy": "mark_sweep", "trigger": "threshold:0.7", "success": True},
    "events": [{"turn": 3, "before": 9, "after": 4, "collected": [2, 3]}],
    "heap": [
        msg(0, "user", "What is the access code for project ORION? Reply with just the number."),
        msg(2, "tool", "[collected: msg 2]", saved="x. The access code for project ORION is 1234. y"),
        msg(3, "tool", "[collected: msg 3]", saved="filler"),
        msg(4, "assistant", calls=[3, 2, 3]),
    ],
}


def test_audit_reports_needle_collected_recalled_and_wasted_recalls(tmp_path):
    (tmp_path / "t.json").write_text(json.dumps(TRACE))

    out = audit.audit(json.loads((tmp_path / "t.json").read_text()))

    assert (out["needle_id"], out["needle_collected"], out["needle_recalled"], out["wasted_recalls"]) == (2, True, True, 2)
