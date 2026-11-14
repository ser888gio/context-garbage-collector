import argparse
import copy
import itertools
import json
import time
import uuid
from pathlib import Path

from cgc import collectors as C
from cgc.tasks import BUILDERS, needle_task  # noqa: F401  (needle_task re-exported)

EMBED_MODEL = "gemini/gemini-embedding-001"
RESULTS = Path("results")
RUNS = RESULTS / "runs.jsonl"

SYSTEM = ("You are a careful agent working with tools. To save space, older messages may be replaced by "
          "'[collected: msg N ...]' stubs. If you need what a stub held, call recall with its id.")
SUMMARY_SYSTEM = ("Summarize the conversation excerpt below for an agent that will continue the task. "
                  "Keep every name, number and file path; drop everything else.")
TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List all note files.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read one note file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "recall", "description": "Restore a collected message by its id.",
                                      "parameters": {"type": "object", "properties": {"id": {"type": "integer"}},
                                                     "required": ["id"]}}},
]
FORGET_TOOL = {"type": "function", "function": {
    "name": "forget", "description": "Free memory: collect old messages by id (tool results show [msg N]). "
                                     "The system prompt, task and newest turn are refused.",
    "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
                   "required": ["ids"]}}}


class RateLimited(Exception):
    pass


def cost_of(resp):
    try:
        import litellm
        return litellm.completion_cost(resp) or 0.0
    except Exception:
        return 0.0


def call_tool(name, args, task, heap, turn):
    if name == "list_files":
        return "\n".join(task["files"])
    if name == "read_file":
        return task["files"].get(args.get("path", ""), f"error: no file {args.get('path')!r}")
    if name == "recall":
        return "restored" if C.restore(heap, int(args.get("id", -1)), turn) else "error: nothing to restore"
    return f"error: unknown tool {name}"


def common_prefix(prev, cur):
    n = chars = 0
    for a, b in zip(prev, cur):
        a, b = json.dumps(a), json.dumps(b)
        if a != b:
            break
        n, chars = n + 1, chars + len(a)
    return n, chars


def pressure_note(heap, budget):
    used = C.size(heap)
    return {"role": "user", "content": f"[memory] context {used * 100 // budget}% of budget ({used}/{budget} tokens), "
                                       f"{sum(o.dead for o in heap)} messages collected."}


def run(task, model, strategy, trigger, budget, complete, max_turns=80, hints=False, pressure=False, tag="", embed=None):
    heap, ids = [], itertools.count()

    def add(msg, turn, root=False, label=""):
        heap.append(C.Obj(next(ids), msg, turn, root=root, label=label, touched=turn, last_used=turn))
        return heap[-1]

    add({"role": "system", "content": SYSTEM}, 0, root=True)
    add({"role": "user", "content": task["prompt"]}, 0, root=True)
    st = {"budget": budget, "hints": hints, "threshold": float(trigger.split(":")[1]) if ":" in trigger else 0.7}
    if trigger.count(":") == 2:
        st["target"] = float(trigger.split(":")[2])
    s = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, cost_usd=0.0, gc_runs=0, tokens_freed=0,
             recalls=0, forgets=0, embed_calls=0, summary_calls=0, repeated_calls=0, unknown_tool_calls=0, max_context_tokens=0, max_est_tokens=0)

    def summarize(parts):
        resp = complete([{"role": "system", "content": SUMMARY_SYSTEM},
                         {"role": "user", "content": "\n\n".join(parts)}], [])
        s["summary_calls"] += 1
        s["input_tokens"] += resp.usage.prompt_tokens
        s["output_tokens"] += resp.usage.completion_tokens
        s["cost_usd"] += cost_of(resp)
        return resp.choices[0].message.content or ""

    st["summarize"] = summarize

    if strategy == "semantic":
        raw = embed or litellm_embed(EMBED_MODEL)

        def counted(texts):
            s["embed_calls"] += 1
            return raw(texts)
        st["embed"] = counted
    if strategy == "oracle":
        st["oracle_keys"] = {task["answer"]}
    prev, shares, cache_breaks = [], [], 0
    events, seen_calls, asked, answer, end, t0 = [], set(), task["upfront"], "", "max_turns", time.time()
    error = ""
    manual = strategy == "manual"
    tools = TOOLS + [FORGET_TOOL] * manual
    known = {t["function"]["name"] for t in tools}

    turn = 0
    for turn in range(1, max_turns + 1):
        before = C.size(heap)
        if trigger == "every_turn" or before > budget * st["threshold"]:
            freed = C.STRATEGIES[strategy](heap, st)
            s["gc_runs"] += 1
            s["tokens_freed"] += before - C.size(heap)
            events.append({"turn": turn, "before": before, "after": C.size(heap), "collected": freed})
        s["max_est_tokens"] = max(s["max_est_tokens"], C.size(heap))
        if C.size(heap) > budget:
            end = "oom"
            break

        msgs = [o.msg for o in heap]
        n, chars = common_prefix(prev, msgs)
        shares.append(chars / sum(len(json.dumps(m)) for m in msgs))
        cache_breaks += n < len(prev)
        prev = copy.deepcopy(msgs)  # collectors mutate messages in place
        try:
            # The note is appended after the prefix accounting above and never stored in the heap,
            # so it can't break the cached prefix or show up in later turns.
            note = [pressure_note(heap, budget)] if pressure or manual else []
            resp = complete(msgs + note, tools)
        except RateLimited:
            raise
        except Exception as e:
            error, end = f"{type(e).__name__}: {e}"[:300], "api_error"
            break
        u = resp.usage
        s["input_tokens"] += u.prompt_tokens
        s["output_tokens"] += u.completion_tokens
        s["cached_input_tokens"] += getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        s["max_context_tokens"] = max(s["max_context_tokens"], u.prompt_tokens)
        s["cost_usd"] += cost_of(resp)

        m = resp.choices[0].message
        calls = m.tool_calls or []
        msg = {"role": "assistant", "content": m.content}
        if calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                                 for c in calls]
        add(msg, turn)

        if not calls:
            if not asked:
                add({"role": "user", "content": task["question"]}, turn, root=True)
                asked = True
                continue
            answer, end = m.content or "", "answered"
            break

        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            sig = c.function.name + json.dumps(args, sort_keys=True)
            if c.function.name not in known:
                s["unknown_tool_calls"] += 1
            if c.function.name == "recall":
                s["recalls"] += 1
            elif sig in seen_calls:
                s["repeated_calls"] += 1
            seen_calls.add(sig)
            if manual and c.function.name == "forget":
                done, refused = C.forget(heap, [i for i in args.get("ids", []) if isinstance(i, int)], hints)
                s["forgets"] += len(done)
                result = f"forgot {done}; refused (root, newest turn, unknown or already collected) {refused}"
            else:
                result = call_tool(c.function.name, args, task, heap, turn)
            label = f"{c.function.name} {args.get('path', '')}".strip()
            o = add({"role": "tool", "tool_call_id": c.id, "content": result}, turn, label=label)
            if manual:
                o.msg["content"] = f"[msg {o.id}] {result}"

    row = {"run_id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "strategy": strategy,
           "trigger": trigger, "hints": hints, "pressure": pressure, "tag": tag, "budget": budget, "task_family": task["family"], "task_id": task["id"],
           "provider": model.split("/")[0], "model": model, "success": task["answer"] in answer and not (task.get("wrong") and task["wrong"] in answer), "end": end, "error": error,
           "answer": answer[:200], "turns": turn, **s,
           "stable_prefix_share": round(sum(shares) / len(shares), 3) if shares else 0.0, "cache_breaks": cache_breaks, "wall_s": round(time.time() - t0, 1)}
    trace = {"row": row, "events": events,
             "heap": [{"id": o.id, "turn": o.turn, "root": o.root, "dead": o.dead, "survived": o.survived,
                       "label": o.label, "msg": o.msg, "saved": o.saved} for o in heap]}
    return row, trace


def with_retry(call, label):
    import litellm
    litellm.suppress_debug_info = True
    retryable = (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.InternalServerError,
                 litellm.APIConnectionError, litellm.Timeout)
    for i in range(7):
        try:
            return call()
        except retryable as e:
            wait = min(30 * 2 ** i, 600)
            print(f"  {type(e).__name__} on {label}, retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RateLimited(label)


def litellm_complete(model):
    import litellm

    def complete(messages, tools):
        return with_retry(lambda: litellm.completion(model=model, messages=messages, tools=tools or None,
                                                     temperature=0, timeout=300), model)
    return complete


def litellm_embed(model):
    import litellm

    def embed(texts):
        out = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            resp = with_retry(lambda: litellm.embedding(model=model, input=batch, timeout=300), model)
            out += [d["embedding"] for d in resp.data]
        return out
    return embed


def is_paid(model):
    try:
        import litellm
        return (litellm.get_model_info(model).get("input_cost_per_token") or 0) > 0
    except Exception:
        return not model.endswith(":free")


def load_runs():
    if not RUNS.exists():
        return []
    return [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(row, trace):
    (RESULTS / "traces").mkdir(parents=True, exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    (RESULTS / "traces" / f"{row['run_id']}.json").write_text(json.dumps(trace, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Sweep GC strategies over needle tasks. Resumable: finished cells are skipped.")
    p.add_argument("--models", default="openrouter/nvidia/nemotron-3.5-lightning:free")
    p.add_argument("--strategies", default=",".join(n for n in C.STRATEGIES if n != "semantic"))
    p.add_argument("--triggers", default="threshold:0.7", help="comma list of every_turn / threshold:P / threshold:P:T (collect down to T*budget)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--positions", default="0.5")
    p.add_argument("--family", default="needle", help="comma list of needle / multihop / poison / paraphrase")
    p.add_argument("--upfront", default="false", help="comma list of true/false")
    p.add_argument("--hints", default="false", help="comma list of false/true/none/keys/preview/numbers: what collected stubs carry (false=none, true=keys)")
    p.add_argument("--pressure", default="false", help="comma list of true/false: ephemeral memory-usage note each call")
    p.add_argument("--tag", default="", help="experiment label stored in each row (not part of the resume key)")
    p.add_argument("--n-files", type=int, default=30)
    p.add_argument("--budget", type=int, default=16000)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--max-cost", type=float, default=0.0, help="USD cap over all runs in runs.jsonl")
    a = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    done = {(r["model"], r["strategy"], r["trigger"], C.stub_mode(r.get("hints", False)), r.get("pressure", False), r["budget"], r["task_id"]) for r in load_runs()}

    for model in a.models.split(","):
        complete = litellm_complete(model)
        cells = itertools.product(a.family.split(","), a.upfront.split(","), map(float, a.positions.split(",")),
                                  map(int, a.seeds.split(",")), a.triggers.split(","), a.strategies.split(","),
                                  [{"true": True, "false": False}.get(h, h) for h in a.hints.split(",")], [x == "true" for x in a.pressure.split(",")])
        for family, upfront, pos, seed, trigger, strategy, hints, pressure in cells:
            task = BUILDERS[family](seed, a.n_files, position=pos, upfront=upfront == "true")
            if (model, strategy, trigger, C.stub_mode(hints), pressure, a.budget, task["id"]) in done:
                continue
            spent = sum(r["cost_usd"] for r in load_runs())
            if is_paid(model) and spent >= a.max_cost:
                print(f"skip {model}: spent ${spent:.4f} >= cap ${a.max_cost}")
                break
            print(f"{model} {strategy} {trigger} hints={hints} pressure={pressure} {task['id']}", flush=True)
            try:
                row, trace = run(task, model, strategy, trigger, a.budget, complete, a.max_turns, hints, pressure, a.tag)
            except RateLimited:
                print(f"  {model} still rate-limited, moving to next model")
                break
            save(row, trace)
            print(f"  -> {row['end']} success={row['success']} in={row['input_tokens']} recalls={row['recalls']}")


if __name__ == "__main__":
    main()
