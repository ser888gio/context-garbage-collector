from types import SimpleNamespace

import pytest

from cgc.agent import run
from cgc.tasks import needle_task


@pytest.mark.parametrize("flag", ["pressure", "coalesce", "upfront"])
def test_a_typo_in_a_true_false_flag_exits_2_naming_the_flag(run_sweep, flag):
    proc = run_sweep(f"--{flag}", "ture", check=False).proc

    assert (proc.returncode, f"unknown --{flag} value in 'ture'; choose from true, false" in proc.stderr) == (2, True)


def test_one_typo_in_a_comma_list_rejects_the_whole_list(run_sweep):
    proc = run_sweep("--pressure", "true,flase", check=False).proc

    assert (proc.returncode, "unknown --pressure value in 'true,flase'" in proc.stderr) == (2, True)


def test_library_callers_may_pass_hints_as_a_bool_and_the_row_records_the_mode():
    def complete(messages, tools):
        message = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1))

    row, _ = run(needle_task(0, n_files=10), "m/x", "mark_sweep", "threshold:0.7", 4000, complete, hints=True)

    assert row["hints"] == "keys"
