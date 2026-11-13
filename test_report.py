import json

import report

ROWS = [
    dict(model="m/a", strategy="mark_sweep", trigger="every_turn", success=True, end="answered",
         input_tokens=1000, cached_input_tokens=500, recalls=2, max_est_tokens=300),
    dict(model="m/a", strategy="mark_sweep", trigger="every_turn", success=False, end="oom",
         input_tokens=3000, cached_input_tokens=0, recalls=0, max_est_tokens=500),
    dict(model="m/a", strategy="none", trigger="every_turn", success=True, end="answered",
         input_tokens=2000, cached_input_tokens=1000, recalls=0, max_est_tokens=900),
]


def test_table_aggregates_per_model_strategy_trigger(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS))

    t = report.table(path).set_index(["strategy"])

    assert t.loc["mark_sweep"].to_dict() == {
        "model": "m/a", "trigger": "every_turn", "runs": 2, "success_rate": 0.5, "oom": 1,
        "input_tokens": 2000.0, "cached_share": 0.25, "recalls": 1.0, "max_est_tokens": 400.0,
    }
    assert t.loc["none"]["success_rate"] == 1.0
    assert t.loc["none"]["oom"] == 0
