import pytest

from analysis.audit import audit


@pytest.fixture(scope="module")
def sweep(run_sweep):
    return run_sweep("--family", "needle,multihop,poison,paraphrase", "--strategies", "mark_sweep", "--budget", "100000")


def file_read(sweep, family, path):
    trace = sweep.trace(sweep.row(task_family=family))
    return next(o["msg"]["content"] for o in trace["heap"] if o["label"] == f"read_file {path}")


@pytest.mark.parametrize("family, key, text", [
    ("needle", "needle_path", "The access code for project ORBITAL is 5242."),
    ("paraphrase", "needle_path", "The access code for project ORBITAL is 5969."),
    ("multihop", "needle_path", "The access code for project ORION is the same as the code for project VEGA."),
    ("multihop", "value_path", "The access code for project VEGA is 7890."),
    ("poison", "needle_path", "Correction: the access code for project ORION is now 1663; the old code 7890 is revoked."),
    ("poison", "old_path", "The access code for project ORION is 7890."),
])
def test_trace_records_the_path_of_the_file_holding_each_fact(sweep, family, key, text):
    path = sweep.trace(sweep.row(task_family=family))["fact_paths"][key]

    assert text in file_read(sweep, family, path)


def test_audit_finds_the_needle_by_recorded_path_even_when_other_files_look_alike(sweep):
    trace = sweep.trace(sweep.row(task_family="needle"))

    needle = audit(trace)["needle_id"]

    assert "ORBITAL is 5242" in next(o["msg"]["content"] for o in trace["heap"] if o["id"] == needle)
