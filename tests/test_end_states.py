import pytest


@pytest.fixture(scope="module")
def ends(run_sweep):
    return run_sweep("--models", "openai/scripted-agent:free,openai/scripted-api-error:free,openai/scripted-agent",
                     "--strategies", "mark_sweep", "--hints", "keys", "--max-turns", "3", "--max-cost", "0")


@pytest.fixture(scope="module")
def bad_hints(run_sweep):
    return run_sweep("--hints", "bogus", check=False)


def test_a_run_that_does_not_answer_in_time_ends_as_max_turns(ends):
    row = ends.row(model="openai/scripted-agent:free")

    assert (row["end"], row["success"], row["turns"]) == ("max_turns", False, 3)


def test_an_endpoint_error_ends_the_run_as_api_error_on_the_first_turn(ends):
    row = ends.row(model="openai/scripted-api-error:free")

    assert (row["end"], row["success"], row["turns"], row["input_tokens"]) == ("api_error", False, 1, 0)


def test_api_error_text_names_the_failure_but_redacts_the_key(ends):
    error = ends.row(model="openai/scripted-api-error:free")["error"]

    assert (error.startswith("BadRequestError: "), "key=REDACTED&alt=json" in error, "SECRET-KEY-123" in error) == (
        True, True, False)


def test_cost_cap_skips_the_paid_model_and_says_so(ends):
    assert "skip openai/scripted-agent: spent $0.0000 >= cap $0.0" in ends.proc.stdout


def test_cost_cap_lets_free_models_run_and_leaves_no_row_for_the_skipped_one(ends):
    assert sorted(r["model"] for r in ends.rows) == ["openai/scripted-agent:free", "openai/scripted-api-error:free"]


def test_unknown_hints_value_exits_with_code_two_and_names_the_valid_values(bad_hints):
    assert (bad_hints.proc.returncode, "unknown --hints value 'bogus'; choose from false, none, true, keys, preview, "
            "numbers, fresh, pointer" in bad_hints.proc.stderr) == (2, True)


def test_unknown_hints_value_runs_no_cell(bad_hints):
    assert bad_hints.rows == []
