import json

from analysis import report

ROWS = [
    dict(model="m/a", strategy="mark_sweep", trigger="every_turn", success=True, end="answered", turns=10, task_id="t1", hints=False,
         input_tokens=1000, cached_input_tokens=500, recalls=2, max_est_tokens=300),
    dict(model="m/a", strategy="mark_sweep", trigger="every_turn", success=False, end="oom", turns=10, task_id="t1", hints=False,
         input_tokens=3000, cached_input_tokens=0, recalls=0, max_est_tokens=500),
    dict(model="m/a", strategy="none", trigger="every_turn", success=True, end="answered", turns=10, task_id="t1", hints=False,
         input_tokens=2000, cached_input_tokens=1000, recalls=0, max_est_tokens=900),
]


def test_table_aggregates_per_model_strategy_trigger(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS))

    t = report.table(path).drop(columns="excess_tokens").set_index(["strategy"])

    assert t.loc["mark_sweep"].to_dict() == {
        "model": "m/a", "trigger": "every_turn", "runs": 2, "success_rate": 0.5, "oom": 1,
        "input_tokens": 2000.0, "cached_share": 0.25, "recalls": 1.0, "max_est_tokens": 400.0,
    }
    assert t.loc["none"]["success_rate"] == 1.0
    assert t.loc["none"]["oom"] == 0


def test_excess_tokens_is_prompt_tokens_per_turn_over_the_oracle_run(tmp_path):
    base = dict(model="m/a", trigger="every_turn", success=True, end="answered", turns=10, hints=False,
                cached_input_tokens=0, recalls=0, max_est_tokens=0)
    rows = [dict(base, strategy="oracle", task_id="t1", input_tokens=1000),
            dict(base, strategy="mark_sweep", task_id="t1", input_tokens=3000),
            dict(base, strategy="mark_sweep", task_id="t1", input_tokens=2000),
            dict(base, strategy="refcount", task_id="t2", input_tokens=3000)]
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    t = report.table(path).set_index("strategy")

    assert t.loc["oracle", "excess_tokens"] == 0.0
    assert t.loc["mark_sweep", "excess_tokens"] == 150.0
    assert t.loc["refcount", "excess_tokens"] != t.loc["refcount", "excess_tokens"]
