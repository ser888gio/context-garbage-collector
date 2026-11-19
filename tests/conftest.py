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
STUB_ENTRY = re.compile(r"msg (\d+) read_file (notes/\d+\.txt)([^;\]]*)")
SUMMARIZE = "Summarize the conversation excerpt"
LEAKY_URL = "https://upstream.invalid/v1/models/m:generate?key=SECRET-KEY-123&alt=json"


class ScriptedAgent:
    """Offline stand-in for a model behind an OpenAI-compatible endpoint. It lists the notes, reads every one,
    says DONE, then answers the question from what is visible: it follows "same as the code for project X"
    links, prefers a correction over the code it revokes, resolves a description to a project name, and
    recalls collected messages whose stubs name the project it is chasing (skipping stubs that point at a
    superseding message). With a `forget` tool it frees old tool results while reading if the `[memory]` note
    shows high usage. With a `compress` tool it replaces the old tool results it has read, from the oldest visible
    summary block or result up to the one before the newest, with a summary of the access codes in them (the
    `scripted-lossy` model leaves them out). Its first compress call is deliberately
    odd for the `greedy` (every id), `backwards` (from_id after to_id) and `rootsonly` models. The `scripted-rereader` model also re-reads the first note once after reading them all,
    and repeats every recall and search call once."""

    FORGET_AT_PERCENT = 60
    COMPRESS_AT_PERCENT = 55
    FIRST_COMPRESS_RANGES = {"greedy": (0, 9999), "backwards": (9, 3), "rootsonly": (0, 1)}

    def __init__(self):
        self.requests, self.notes, self.context_size, self.offered, self.noted = 0, [], 0, [], []
        self.seen = []
        # ADK pairs calls with responses by id, so it needs ids that never repeat.
        self.unique_ids = False

    def reply(self, messages, tools, model=""):
        self.requests += 1
        self.seen.append(messages)
        self.context_size = len(messages)
        if messages[0]["content"].startswith(SUMMARIZE):
            facts = re.findall(r"(?:The|Correction: the) access code for project[^.]*\.", messages[1]["content"])
            return self.text("Summary: " + " ".join(facts))

        self.offered.append((messages[0]["content"], sorted(t["function"]["name"] for t in tools or [])))
        calls = [c["function"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        stubs = [(int(i), path, rest) for m in messages if (m.get("content") or "").startswith("[collected")
                 for i, path, rest in STUB_ENTRY.findall(m["content"])]
        last = messages[-1]
        usage = re.match(r"\[memory\] context (\d+)%", last["content"] if last["role"] == "user" else "")
        self.noted.append(bool(usage))
        if usage:
            self.notes.append(last["content"])
        if not calls and not stubs:
            call, finish = self.call("list_files", {})
            upfront = QUESTION.search(messages[1]["content"])
            if upfront and "search" in {t["function"]["name"] for t in tools or []}:
                call["content"] = f"Listing the notes to find {upfront.group(1) and 'project ' + upfront.group(1) or 'the ' + upfront.group(2)}"
            return call, finish

        rereader = "rereader" in model
        if rereader and calls and calls[-1]["name"] in ("recall", "search") and calls.count(calls[-1]) == 1:
            return self.call(calls[-1]["name"], json.loads(calls[-1]["arguments"]))

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
            if "compress" in tool_names and usage and int(usage.group(1)) >= self.COMPRESS_AT_PERCENT and live[:-1]:
                starts = [int(i) for m in messages for i in re.findall(r"\[compressed msgs (\d+)-", m.get("content") or "")]
                old = " ".join(m["content"] for m in messages if m["role"] == "tool"
                               and (n := re.match(r"\[msg (\d+)\] ", m["content"])) and int(n.group(1)) < live[-1])
                facts = re.findall(r"(?:The|Correction: the) access code for project[^.]*\.", old)
                summary = "Notes read, nothing important." if "lossy" in model else " ".join(facts) or "Notes read."
                odd = next((r for k, r in self.FIRST_COMPRESS_RANGES.items() if k in model), None)
                if odd and not any(c["name"] == "compress" for c in calls):
                    return self.call("compress", {"from_id": odd[0], "to_id": odd[1], "summary": summary})
                return self.call("compress", {"from_id": min(starts + live[:-1]), "to_id": live[-2], "summary": summary})
            return self.call("read_file", {"path": unread[0]})

        again = {"name": "read_file", "arguments": json.dumps({"path": "notes/00.txt"})}
        if rereader and calls.count(again) == 1:
            return self.call("read_file", {"path": "notes/00.txt"})

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

        if wanted in corrected:
            return self.text(corrected[wanted])

        searched = {json.loads(c["arguments"])["query"] for c in calls if c["name"] == "search"}
        query = wanted or asked.group(2)
        offers_search = "search" in {t["function"]["name"] for t in tools or []}
        if offers_search and query not in searched and (wanted is None or wanted not in code):
            return self.call("search", {"query": query})

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
        tool_call = {"id": f"call_{self.requests if self.unique_ids else self.context_size}", "type": "function",
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
                message, finish = agent.reply(body["messages"], body.get("tools"), body["model"])
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
        self.offered, self.noted, self.seen = [], [], []

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

    def trace_text(self, row):
        return (self.cwd / "results" / "traces" / f"{row['run_id']}.json").read_text(encoding="utf-8")

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
    """run_sweep(*cli_args, into=None) -> Sweep. Defaults: scripted model, seed 0, 10 files, budget 4000; later
    flags win. `into=<Sweep>` appends to that sweep's results directory instead of starting a new one."""
    agent, env, tmp_path_factory = scripted

    def run(*args, check=True, into=None):
        cwd = into.cwd if into else tmp_path_factory.mktemp("sweep")
        (cwd / "docs" / "img").mkdir(parents=True, exist_ok=True)
        start, start_offered, start_noted, start_seen = len(agent.notes), len(agent.offered), len(agent.noted), len(agent.seen)
        argv = ["cgc.agent", "--models", "openai/scripted-agent", "--seeds", "0", "--n-files", "10",
                "--budget", "4000", "--max-cost", "1", *args]
        proc = cli(cwd, env, *argv, check=check)
        sweep = Sweep(cwd, env, agent, proc, agent.notes[start:], argv)
        sweep.offered = agent.offered[start_offered:]
        sweep.noted = agent.noted[start_noted:]
        sweep.seen = agent.seen[start_seen:]
        return sweep

    return run


def converse(ctx, task):
    """Drive the ScriptedAgent through `ctx` the way an agent loop would. Returns (answer, history, sent)."""
    from cgc.agent import FILE_TOOLS, SYSTEM

    agent = ScriptedAgent()
    history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    sent = []
    while True:
        sent.append(ctx.prepare(history))
        message, finish = agent.reply(sent[-1], FILE_TOOLS + ctx.tools)
        history.append(message)
        if finish == "stop":
            return message["content"], history, sent
        for c in message["tool_calls"]:
            name, args = c["function"]["name"], json.loads(c["function"]["arguments"])
            result = ctx.call(name, args)
            if result is None:
                result = "\n".join(task["files"]) if name == "list_files" else task["files"][args["path"]]
            history.append({"role": "tool", "tool_call_id": c["id"], "content": result})
