import re

import pytest


@pytest.fixture(scope="module")
def sweep(run_sweep):
    return run_sweep("--strategies", "mark_sweep", "--hints", "none,keys")


def test_keys_hints_let_the_agent_recall_the_collected_needle_and_answer(sweep):
    row = sweep.row(hints="keys")

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, 1)


def test_without_hints_the_collected_needle_is_lost(sweep):
    row = sweep.row(hints="none")

    assert (row["end"], row["success"], row["answer"]) == ("answered", False, "I don't know")


def test_audit_reports_the_needle_collected_in_both_runs_and_recalled_only_with_keys(sweep):
    out = sweep.cli("analysis.audit").stdout

    summary = out.strip().splitlines()[-1]
    assert re.split(r"\s*\|\s*", summary) == ["", "mark_sweep", "0.5", "1", "0.5", "0", ""]


def test_results_report_lists_success_rate_per_hints_mode(sweep):
    sweep.cli("analysis.results")

    report = (sweep.cwd / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert re.search(r"\| openai/scripted-agent \| needle\s+\| mark_sweep \| keys\s+\|.*?\|\s+1\.00 \|", report)
    assert re.search(r"\| openai/scripted-agent \| needle\s+\| mark_sweep \| none\s+\|.*?\|\s+0\.00 \|", report)


def test_timeline_renders_a_png_from_a_trace(sweep):
    trace = sweep.cwd / "results" / "traces" / f"{sweep.row(hints='keys')['run_id']}.json"

    sweep.cli("analysis.timeline", str(trace), "timeline.png")

    assert (sweep.cwd / "timeline.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_rerunning_the_sweep_skips_finished_cells_without_calling_the_model(sweep):
    before = sweep.agent.requests

    sweep.rerun()

    assert (sweep.agent.requests - before, len(sweep.rows)) == (0, 2)
