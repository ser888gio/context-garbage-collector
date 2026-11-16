import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


class ScriptedAgent:
    """Offline stand-in for a model behind an OpenAI-compatible endpoint: lists the notes, reads each one,
    says DONE, then answers from what it can see or recalls the one stub whose keys name the project."""

    def __init__(self):
        self.files, self.requests = [], 0

    def reply(self, messages):
        self.requests += 1
        calls = [c["function"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        if not calls:
            self.files = []
            return self.call("list_files", {})
        if calls[-1]["name"] == "list_files" and not self.files:
            self.files = messages[-1]["content"].split("\n")
        read = {json.loads(c["arguments"]).get("path") for c in calls if c["name"] == "read_file"}
        unread = [p for p in self.files if p not in read]
        if unread:
            return self.call("read_file", {"path": unread[0]})
        question = next((m["content"] for m in reversed(messages)
                         if m["role"] == "user" and "access code" in m["content"]), None)
        if question is None:
            return self.text("DONE")
        name = re.search(r"project (\w+)\?", question).group(1)
        for m in messages:
            if found := re.search(rf"project {name} is (\d+)", m.get("content") or ""):
                return self.text(found.group(1))
        recalled = {json.loads(c["arguments"])["id"] for c in calls if c["name"] == "recall"}
        for m in messages:
            stub = re.match(rf"\[collected: msg (\d+) .*\b{name}\b", m.get("content") or "")
            if stub and int(stub.group(1)) not in recalled:
                return self.call("recall", {"id": int(stub.group(1))})
        return self.text("I don't know")

    def call(self, name, args):
        tool_call = {"id": f"call_{self.requests}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
        return {"role": "assistant", "content": None, "tool_calls": [tool_call]}, "tool_calls"

    def text(self, content):
        return {"role": "assistant", "content": content}, "stop"


def serve(agent):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            message, finish = agent.reply(body["messages"])
            prompt = len(json.dumps(body["messages"])) // 4
            out = json.dumps({"id": "chatcmpl-e2e", "object": "chat.completion", "created": 0, "model": body["model"],
                              "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                              "usage": {"prompt_tokens": prompt, "completion_tokens": 5,
                                        "total_tokens": prompt + 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def cli(cwd, env, *args):
    return subprocess.run([sys.executable, "-m", *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=600, check=True)


SWEEP = ["cgc.agent", "--models", "openai/scripted-agent", "--strategies", "mark_sweep", "--hints", "none,keys",
         "--seeds", "0", "--n-files", "10", "--budget", "4000", "--max-cost", "1"]


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    agent = ScriptedAgent()
    server = serve(agent)
    cwd = tmp_path_factory.mktemp("sweep")
    (cwd / "docs").mkdir()
    # The scripted agent is a real HTTP endpoint, so litellm, the CLI and the result files are all exercised.
    env = {**os.environ, "PYTHONPATH": str(REPO), "OPENAI_API_KEY": "e2e",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    cli(cwd, env, *SWEEP)
    yield {"agent": agent, "cwd": cwd, "env": env}
    server.shutdown()


def rows(cwd):
    return {r["hints"]: r for r in map(json.loads, (cwd / "results" / "runs.jsonl").read_text().splitlines())}


def test_keys_hints_let_the_agent_recall_the_collected_needle_and_answer(sweep):
    row = rows(sweep["cwd"])["keys"]

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, 1)


def test_without_hints_the_collected_needle_is_lost(sweep):
    row = rows(sweep["cwd"])["none"]

    assert (row["end"], row["success"], row["answer"]) == ("answered", False, "I don't know")


def test_audit_reports_the_needle_collected_in_both_runs_and_recalled_only_with_keys(sweep):
    out = cli(sweep["cwd"], sweep["env"], "analysis.audit").stdout

    summary = out.strip().splitlines()[-1]
    assert re.split(r"\s*\|\s*", summary) == ["", "mark_sweep", "0.5", "1", "0.5", "0", ""]


def test_results_report_lists_success_rate_per_hints_mode(sweep):
    cli(sweep["cwd"], sweep["env"], "analysis.results")

    report = (sweep["cwd"] / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert re.search(r"\| openai/scripted-agent \| needle\s+\| mark_sweep \| keys\s+\|.*?\|\s+1\.00 \|", report)
    assert re.search(r"\| openai/scripted-agent \| needle\s+\| mark_sweep \| none\s+\|.*?\|\s+0\.00 \|", report)


def test_timeline_renders_a_png_from_a_trace(sweep):
    trace = sweep["cwd"] / "results" / "traces" / f"{rows(sweep['cwd'])['keys']['run_id']}.json"

    cli(sweep["cwd"], sweep["env"], "analysis.timeline", str(trace), "timeline.png")

    assert (sweep["cwd"] / "timeline.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_rerunning_the_sweep_skips_finished_cells_without_calling_the_model(sweep):
    before = sweep["agent"].requests

    cli(sweep["cwd"], sweep["env"], *SWEEP)

    assert (sweep["agent"].requests - before, len(rows(sweep["cwd"]))) == (0, 2)
