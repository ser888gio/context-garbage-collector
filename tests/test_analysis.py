import re

import pytest

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def reported(run_sweep):
    return run_sweep("--strategies", "none,mark_sweep,oracle", "--hints", "keys")


@pytest.fixture(scope="module")
def tradeoff(run_sweep):
    gpt = ("--models", "openai/gpt-oss-120b", "--seeds", "0,1,2,3,4,5,6,7")
    keys = run_sweep(*gpt, "--strategies", "mark_sweep,oracle", "--hints", "keys")
    return run_sweep(*gpt, "--strategies", "mark_sweep", "--hints", "pointer", "--coalesce", "true", into=keys)


def cells(line):
    return re.split(r"\s*\|\s*", line.strip())[1:-1]


def test_report_prints_one_row_per_strategy_with_success_oom_and_tokens(reported):
    out = reported.cli("analysis.report").stdout

    assert [cells(line) for line in out.splitlines()[2:]] == [
        ["openai/scripted-agent", "mark_sweep", "threshold:0.7", "1", "1.00", "0", "22762.00", "0.00", "1.00",
         "-108.14", "2722.00"],
        ["openai/scripted-agent", "none", "threshold:0.7", "1", "0.00", "1", "9414.00", "0.00", "0.00",
         "-389.14", "4425.00"],
        ["openai/scripted-agent", "oracle", "threshold:0.7", "1", "1.00", "0", "22542.00", "0.00", "0.00",
         "0.00", "2722.00"]]


def test_report_writes_the_success_vs_tokens_plot(reported):
    reported.cli("analysis.report")

    assert (reported.cwd / "results" / "success_vs_tokens.png").read_bytes()[:8] == PNG


def test_tradeoff_lists_each_point_that_has_at_least_eight_runs(tradeoff):
    out = tradeoff.cli("analysis.tradeoff", "results/runs.jsonl").stdout

    assert [line.split() for line in out.splitlines()] == []


def test_tradeoff_writes_its_plot(tradeoff):
    tradeoff.cli("analysis.tradeoff", "results/runs.jsonl")

    assert (tradeoff.cwd / "docs" / "img" / "tradeoff.png").read_bytes()[:8] == PNG
