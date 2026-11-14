import json

import pytest

from analysis.tradeoff import points

M = "groq/openai/gpt-oss-120b"


def row(strategy, success, tokens, hints="keys", coalesce=None, family="needle", **kw):
    r = dict(model=M, strategy=strategy, trigger="threshold:0.7", pressure=False, hints=hints,
             task_family=family, task_id=f"{family}-p0.5-late-n30-s0", success=success,
             input_tokens=tokens, cached_input_tokens=0, turns=10, end="done", error="")
    if coalesce is not None:
        r["coalesce"] = coalesce
    return {**r, **kw}


def write(tmp_path, rows):
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def cell(pts, strategy, variant, family="needle"):
    return pts[(pts.strategy == strategy) & (pts.variant == variant) & (pts.family == family)].iloc[0]


@pytest.fixture
def pts(tmp_path):
    rows = (
        [row("mark_sweep", i < 6, 1000) for i in range(10)]
        + [row("mark_sweep", True, 500, hints="pointer", coalesce=True) for i in range(10)]
        + [row("oracle", True, 2000) for _ in range(8)]
        + [row("truncate", True, 100) for _ in range(7)]
        + [row("refcount", True, 100, hints="pointer", coalesce=False) for _ in range(10)]
        + [row("mark_sweep", True, 100, family="poison") for _ in range(10)]
        + [row("mark_sweep", True, 100, pressure=True) for _ in range(10)]
    )
    return points(write(tmp_path, rows))


def test_missing_coalesce_counts_as_keys_baseline(pts):
    c = cell(pts, "mark_sweep", "keys")
    assert (c.n, c.success, c.tokens) == (10, 0.6, 1000)


def test_pointer_with_coalesce_is_its_own_variant(pts):
    c = cell(pts, "mark_sweep", "pointer+coalesce")
    assert (c.n, c.success, c.tokens) == (10, 1.0, 500)


def test_cells_under_eight_runs_are_dropped(pts):
    assert "truncate" not in set(pts.strategy)
    assert "oracle" in set(pts.strategy)


def test_other_hint_coalesce_combinations_and_pressure_runs_are_excluded(pts):
    assert "refcount" not in set(pts.strategy)
    assert cell(pts, "mark_sweep", "keys").n == 10
    assert set(pts.variant) == {"keys", "pointer+coalesce"}
