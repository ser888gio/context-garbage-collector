import json
import os
import re
import subprocess
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

QUESTION = re.compile(r"What is the access code for (?:project (\w+)|the (.+?) project)\?")
STUB_ENTRY = re.compile(r"msg (\d+) read_file (\S+)([^;\]]*)")
SUMMARIZE = "Summarize the conversation excerpt"
LEAKY_URL = "https://upstream.invalid/v1/models/m:generate?key=SECRET-KEY-123&alt=json"


class ScriptedAgent:
    """Offline stand-in for a model behind an OpenAI-compatible endpoint. It lists the notes, reads every one,
    says DONE, then answers the question from what is visible: it follows "same as the code for project X"
    links, prefers a correction over the code it revokes, resolves a description to a project name, and
    recalls collected messages whose stubs name the project it is chasing (skipping stubs that point at a
    superseding message). With a `forget` tool it frees old tool results while reading if the `[memory]` note
    shows high usage."""

    FORGET_AT_PERCENT = 60

    def __init__(self):
        self.requests, self.notes = 0, []

    def reply(self, messages, tools):
        self.requests += 1
        if messages[0]["content"].startswith(SUMMARIZE):
            facts = re.findall(r"(?:The|Correction: the) access code for project[^.]*\.", messages[1]["content"])
            return self.text("Summary: " + " ".join(facts))

        calls = [c["function"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        stubs = [(int(i), path, rest) for m in messages if (m.get("content") or "").startswith("[collected")
                 for i, path, rest in STUB_ENTRY.findall(m["content"])]
        if not calls and not stubs:
            return self.call("list_files", {})

        last = messages[-1]
        usage = re.match(r"\[memory\] context (\d+)%", last["content"] if last["role"] == "user" else "")
        if usage:
            self.notes.append(last["content"])

        read = {json.loads(c["arguments"]).get("path") for c in calls if c["name"] == "read_file"}
        read |= {path for _, path, _ in stubs}
        n_files = int(re.search(r"has (\d+) note files", messages[1]["content"]).group(1))
        unread = [p for p in (f"notes/{i:02d}.txt" for i in range(n_files)) if p not in read]
        if unread:
            live = [int(m.group(1)) for msg in messages if msg["role"] == "tool"
                    if (m := re.match(r"\[msg (\d+)\] ", msg["content"]))]
            tool_names = {t["function"]["name"] for t in tools or []}
            if "forget" in tool_names and usage and int(usage.group(1)) >= self.FORGET_AT_PERCENT and live[:-1]:
                return self.call("forget", {"ids": live[:-1]})
            return self.call("read_file", {"path": unread[0]})

        asked = next((QUESTION.search(m["content"]) for m in reversed(messages)
                      if m["role"] == "user" and QUESTION.search(m["content"])), None)
        if asked is None:
            return self.text("DONE")

        seen = "\n".join(m.get("content") or "" for m in messages if m["role"] != "system")
        code = dict(re.findall(r"project (\w+) is (\d+)", seen))
        corrected = dict(re.findall(r"project (\w+) is now (\d+)", seen))
        alias = dict(re.findall(r"project (\w+) is the same as the code for project (\w+)", seen))
        names = {desc: name for name, desc in re.findall(r"Project (\w+) is the ([^.]+)\.", seen)}
        wanted = asked.group(1) or names.get(asked.group(2))
        while wanted in alias:
            wanted = alias[wanted]

        recalled = {json.loads(c["arguments"])["id"] for c in calls if c["name"] == "recall"}
        if wanted is None:
            targets = [i for i, _, rest in stubs
                       if re.search(r"\b[A-Z][A-Z0-9_-]{2,}\b", rest) and re.search(r"\b\d{4,}\b", rest)]
        else:
            targets = [int(p.group(1)) if (p := re.search(rf"{wanted} \(superseded by msg (\d+)\)", rest)) else i
                       for i, _, rest in stubs if re.search(rf"\b{wanted}\b", rest)]
        stub_ids = {i for i, _, _ in stubs}
        pending = [i for i in dict.fromkeys(targets) if i in stub_ids and i not in recalled]
        if pending:
            return self.call("recall", {"id": pending[0]})
        answer = corrected.get(wanted) or code.get(wanted)
        return self.text(answer or "I don't know")

    def call(self, name, args):
        tool_call = {"id": f"call_{self.requests}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
        return {"role": "assistant", "content": None, "tool_calls": [tool_call]}, "tool_calls"

    def text(self, content):
        return {"role": "assistant", "content": content}, "stop"


def embed(text):
    vec = [0.0] * 64
    for word in re.findall(r"\w+", text.lower()):
        vec[zlib.crc32(word.encode()) % 64] += 1.0
    return vec


def serve(agent):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, payload = 200, None
            if self.path.endswith(":batchEmbedContents"):
                payload = {"embeddings": [{"values": embed(r["content"]["parts"][0]["text"])}
                                          for r in body["requests"]]}
            elif "api-error" in body["model"]:
                status = 400
                payload = {"error": {"message": f"bad request {LEAKY_URL}", "type": "invalid_request_error",
                                     "code": None}}
            else:
                message, finish = agent.reply(body["messages"], body.get("tools"))
                prompt = len(json.dumps(body["messages"])) // 4
                payload = {"id": "chatcmpl-e2e", "object": "chat.completion", "created": 0, "model": body["model"],
                           "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                           "usage": {"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5}}
            out = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def cli(cwd, env, *args, check=True):
    proc = subprocess.run([sys.executable, "-m", *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=600)
    assert not check or proc.returncode == 0, proc.stderr
    return proc


class Sweep:
    """The files and output of one `python -m cgc.agent` invocation run in its own temp directory."""

    def __init__(self, cwd, env, agent, proc, notes, argv):
        self.cwd, self.env, self.agent, self.proc, self.notes, self.argv = cwd, env, agent, proc, notes, argv

    @property
    def rows(self):
        path = self.cwd / "results" / "runs.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def row(self, **match):
        found = [r for r in self.rows if all(r[k] == v for k, v in match.items())]
        assert len(found) == 1, f"{len(found)} rows match {match}"
        return found[0]

    def trace(self, row):
        return json.loads((self.cwd / "results" / "traces" / f"{row['run_id']}.json").read_text(encoding="utf-8"))

    def content(self, row, msg_id):
        return next(o["msg"]["content"] for o in self.trace(row)["heap"] if o["id"] == msg_id)

    def collected(self, row):
        return [e["collected"] for e in self.trace(row)["events"]]

    def cli(self, *args, check=True):
        return cli(self.cwd, self.env, *args, check=check)

    def rerun(self):
        return self.cli(*self.argv)


@pytest.fixture(scope="session")
def scripted(tmp_path_factory):
    agent = ScriptedAgent()
    server = serve(agent)
    base = f"http://127.0.0.1:{server.server_port}"
    # The scripted agent is a real HTTP endpoint, so litellm, the CLI and the result files are all exercised.
    env = {**os.environ, "PYTHONPATH": str(REPO), "OPENAI_API_KEY": "e2e", "OPENAI_BASE_URL": f"{base}/v1",
           "GEMINI_API_KEY": "e2e", "GEMINI_API_BASE": f"{base}/v1beta", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    yield agent, env, tmp_path_factory
    server.shutdown()


@pytest.fixture(scope="session")
def run_sweep(scripted):
    """run_sweep(*cli_args) -> Sweep. Defaults: scripted model, seed 0, 10 files, budget 4000; later flags win."""
    agent, env, tmp_path_factory = scripted

    def run(*args, check=True):
        cwd = tmp_path_factory.mktemp("sweep")
        (cwd / "docs" / "img").mkdir(parents=True)
        start = len(agent.notes)
        argv = ["cgc.agent", "--models", "openai/scripted-agent", "--seeds", "0", "--n-files", "10",
                "--budget", "4000", "--max-cost", "1", *args]
        proc = cli(cwd, env, *argv, check=check)
        return Sweep(cwd, env, agent, proc, agent.notes[start:], argv)

    return run
