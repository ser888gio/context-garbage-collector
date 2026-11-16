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
        ["openai/scripted-agent", "mark_sweep", "threshold:0.7", "1", "1.00", "0", "22738.00", "0.00", "1.00",
         "-108.16", "2720.00"],
        ["openai/scripted-agent", "none", "threshold:0.7", "1", "0.00", "1", "9406.00", "0.00", "0.00",
         "-388.59", "4423.00"],
        ["openai/scripted-agent", "oracle", "threshold:0.7", "1", "1.00", "0", "22520.00", "0.00", "0.00",
         "0.00", "2720.00"]]


def test_report_writes_the_success_vs_tokens_plot(reported):
    reported.cli("analysis.report")

    assert (reported.cwd / "results" / "success_vs_tokens.png").read_bytes()[:8] == PNG


def test_tradeoff_lists_keys_and_pointer_coalesce_points_with_eight_runs(tradeoff):
    out = tradeoff.cli("analysis.tradeoff", "results/runs.jsonl").stdout

    assert [line.split()[1:] for line in out.splitlines()[1:]] == [
        ["needle", "mark_sweep", "keys", "8", "1.0", "22808.25", "0"],
        ["needle", "mark_sweep", "pointer+coalesce", "8", "1.0", "19487.00", "0"],
        ["needle", "oracle", "keys", "8", "1.0", "22580.62", "0"]]


def test_tradeoff_writes_its_plot(tradeoff):
    tradeoff.cli("analysis.tradeoff", "results/runs.jsonl")

    assert (tradeoff.cwd / "docs" / "img" / "tradeoff.png").read_bytes()[:8] == PNG
