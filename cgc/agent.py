import argparse
import copy
import itertools
import json
import re
import time
import uuid
from pathlib import Path

from cgc import collectors as C
from cgc.tasks import BUILDERS

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
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the text of collected messages without restoring them. "
                                     "Returns matching sentences with their message ids.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
SEARCH_SENTENCE = " You can call search(query) to find text in collected messages without restoring them."
FORGET_TOOL = {"type": "function", "function": {
    "name": "forget", "description": "Free memory: collect old messages by id (tool results show [msg N]). "
                                     "The system prompt, task and newest turn are refused.",
    "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
                   "required": ["ids"]}}}
COMPRESS_TOOL = {"type": "function", "function": {
    "name": "compress", "description": "Replace messages from_id..to_id (inclusive) that you no longer need verbatim "
                                       "with your summary. Each original stays recallable by id. The system prompt, "
                                       "task and newest turn are never compressed.",
    "parameters": {"type": "object", "properties": {"from_id": {"type": "integer"}, "to_id": {"type": "integer"},
                                                    "summary": {"type": "string"}},
                   "required": ["from_id", "to_id", "summary"]}}}


class RateLimited(Exception):
    pass


QUOTA_HINT = re.compile(r"429|RESOURCE_EXHAUSTED|quota|rate.?limit", re.I)


def error_text(e):
    # Gemini error messages embed the request URL, including the API key.
    return re.sub(r"key=[^&\s'\"]+", "key=REDACTED", f"{type(e).__name__}: {e}")[:300]


def cost_of(resp):
    try:
        import litellm
        return litellm.completion_cost(resp) or 0.0
    except Exception:
        return 0.0


TAIL_RECALL_TOOL = {"type": "function", "function": {
    "name": "recall", "description": "Return a collected message's content as this call's result; the stub stays in place.",
    "parameters": TOOLS[2]["function"]["parameters"]}}


def call_tool(name, args, task, heap, turn, recall="in_place"):
    if name == "list_files":
        return "\n".join(task["files"])
    if name == "read_file":
        return task["files"].get(args.get("path", ""), f"error: no file {args.get('path')!r}")
    if name == "recall" and recall == "tail":
        obj_id = int(args.get("id", -1))
        o = next((o for o in heap if o.id == obj_id), None)
        return f"[recalled msg {obj_id}] {o.saved}" if o and o.saved is not None else "error: nothing to restore"
    if name == "recall":
        return "restored" if C.restore(heap, int(args.get("id", -1)), turn) else "error: nothing to restore"
    if name == "search":
        return C.search(heap, str(args.get("query", "")))
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


def parse_trigger(trigger):
    parts = trigger.split(":")
    threshold = float(parts[1]) if len(parts) > 1 else 0.7
    target = float(parts[2]) if len(parts) == 3 else None
    return threshold, target


def add_usage(s, resp):
    s["input_tokens"] += resp.usage.prompt_tokens
    s["output_tokens"] += resp.usage.completion_tokens
    s["cost_usd"] += cost_of(resp)


def build_tools(recall, manual, compress, search):
    return (TOOLS[:2] + [TAIL_RECALL_TOOL] if recall == "tail" else TOOLS) + [FORGET_TOOL] * manual + [COMPRESS_TOOL] * compress + [SEARCH_TOOL] * search


def memory_notes(heap, budget, st, model_calls, nudge, pressure, manual, compress):
    send = pressure or manual
    if compress and not send and C.size(heap) > budget * st["nudge_from"]:
        nudge.setdefault("first", model_calls)
        send = (model_calls - nudge["first"]) % st["nudge_every"] == 0
    return [pressure_note(heap, budget)] if send else []


def track_usage(s, resp):
    add_usage(s, resp)
    u = resp.usage
    s["cached_input_tokens"] += getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    s["max_context_tokens"] = max(s["max_context_tokens"], u.prompt_tokens)


def assistant_message(m):
    msg = {"role": "assistant", "content": m.content}
    if m.tool_calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function",
                              "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                             for c in m.tool_calls]
    return msg


def guarded(fn, *args):
    try:
        return fn(*args), ""
    except RateLimited:
        raise
    except Exception as e:
        return None, error_text(e)


def is_success(task, answer):
    wrong = task.get("wrong")
    return task["answer"] in answer and not (wrong and wrong in answer)


def build_result(task, model, cfg, heap, events, s, shares, cache_breaks, end, error, answer, turn, t0):
    row = {"run_id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **cfg,
           "task_family": task["family"], "task_id": task["id"], "provider": model.split("/")[0], "model": model,
           "success": is_success(task, answer), "end": end, "error": error, "answer": answer[:200], "turns": turn, **s,
           "stable_prefix_share": round(sum(shares) / len(shares), 3) if shares else 0.0,
           "cache_breaks": cache_breaks, "wall_s": round(time.time() - t0, 1)}
    fact_paths = {k: v for k, v in task.items() if k.endswith("_path")}
    trace = {"row": row, "events": events, "fact_paths": fact_paths,
             "heap": [{"id": o.id, "turn": o.turn, "root": o.root, "dead": o.dead, "survived": o.survived,
                       "label": o.label, "msg": o.msg, "saved": o.saved,
                       "archived": o.archived} for o in heap]}
    return row, trace


def run(task, model, strategy, trigger, budget, complete, max_turns=80, hints=False, pressure=False, tag="",
        embed=None, coalesce=False, search=False, dedup=False, recall="in_place"):
    hints = C.stub_mode(hints)
    heap, ids = [], itertools.count()

    def add(msg, turn, root=False, label="", call=""):
        heap.append(C.Obj(next(ids), msg, turn, root=root, label=label, call=call, touched=turn, last_used=turn))
        return heap[-1]

    add({"role": "system", "content": SYSTEM + SEARCH_SENTENCE * search}, 0, root=True)
    add({"role": "user", "content": task["prompt"]}, 0, root=True)
    threshold, target = parse_trigger(trigger)
    st = {"budget": budget, "hints": hints, "threshold": threshold, "dedup": dedup, "nudge_from": 0.5,
          "nudge_every": 5}
    if target is not None:
        st["target"] = target
    s = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, cost_usd=0.0, gc_runs=0, tokens_freed=0,
             recalls=0, searches=0, forgets=0, dedups=0, compressions=0, compressed_msgs=0, embed_calls=0, summary_calls=0, repeated_calls=0, unknown_tool_calls=0,
             max_context_tokens=0, max_est_tokens=0)

    def summarize(parts):
        resp = complete([{"role": "system", "content": SUMMARY_SYSTEM},
                         {"role": "user", "content": "\n\n".join(parts)}], [])
        s["summary_calls"] += 1
        add_usage(s, resp)
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
    compress = strategy == "compress"
    tools = build_tools(recall, manual, compress, search)
    nudge, model_calls = {}, 0
    known = {t["function"]["name"] for t in tools}

    def handle_call(c, turn):
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        sig = c.function.name + json.dumps(args, sort_keys=True)
        if c.function.name not in known:
            s["unknown_tool_calls"] += 1
        if c.function.name == "recall":
            s["recalls"] += 1
        elif c.function.name == "search" and search:
            s["searches"] += 1
        elif sig in seen_calls:
            s["repeated_calls"] += 1
        seen_calls.add(sig)
        if manual and c.function.name == "forget":
            done, refused = C.forget(heap, [i for i in args.get("ids", []) if isinstance(i, int)], hints)
            s["forgets"] += len(done)
            result = f"forgot {done}; refused (root, newest turn, unknown or already collected) {refused}"
        elif compress and c.function.name == "compress":
            done, refused = C.compress(heap, args.get("from_id"), args.get("to_id"), args.get("summary"), hints)
            s["compressions"] += bool(done)
            s["compressed_msgs"] += len(done)
            result = f"compressed {done} into msg {done[0]}" if done else f"refused: {refused}"
        else:
            result = call_tool(c.function.name, args, task, heap, turn, recall)
        label = f"{c.function.name} {args.get('path', '')}".strip()
        o = add({"role": "tool", "tool_call_id": c.id, "content": result}, turn, label=label, call=sig)
        if manual or compress:
            o.msg["content"] = f"[msg {o.id}] {result}"

    turn = 0
    for turn in range(1, max_turns + 1):
        before = C.size(heap)
        if trigger == "every_turn" or before > budget * threshold:
            freed, error = guarded(C.collect, strategy, heap, st)
            if error:
                end = "api_error"
                break
            s["gc_runs"] += 1
            s["dedups"] = st.get("dedups", 0)
            s["tokens_freed"] += before - C.size(heap)
            events.append({"turn": turn, "before": before, "after": C.size(heap), "collected": freed})
        s["max_est_tokens"] = max(s["max_est_tokens"], C.size(heap))
        if C.size(heap) > budget:
            end = "oom"
            break

        msgs = C.render(heap, coalesce)
        n, chars = common_prefix(prev, msgs)
        shares.append(chars / sum(len(json.dumps(m)) for m in msgs))
        cache_breaks += n < len(prev)
        prev = copy.deepcopy(msgs)  # collectors mutate messages in place
        # The note is appended after the prefix accounting above and never stored in the heap,
        # so it can't break the cached prefix or show up in later turns.
        model_calls += 1
        note = memory_notes(heap, budget, st, model_calls, nudge, pressure, manual, compress)
        resp, error = guarded(complete, msgs + note, tools)
        if error:
            end = "api_error"
            break
        track_usage(s, resp)

        m = resp.choices[0].message
        calls = m.tool_calls or []
        add(assistant_message(m), turn)

        if not calls:
            if not asked:
                add({"role": "user", "content": task["question"]}, turn, root=True)
                asked = True
                continue
            answer, end = m.content or "", "answered"
            break

        for c in calls:
            handle_call(c, turn)

    cfg = dict(strategy=strategy, trigger=trigger, hints=hints, pressure=pressure, coalesce=coalesce, search=search, dedup=dedup, recall=recall, tag=tag,
               budget=budget)
    return build_result(task, model, cfg, heap, events, s, shares, cache_breaks, end, error, answer, turn, t0)


def with_retry(call, label, sleep=time.sleep):
    import litellm
    litellm.suppress_debug_info = True
    retryable = (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.InternalServerError,
                 litellm.APIConnectionError, litellm.Timeout)
    for i in range(7):
        try:
            return call()
        except Exception as e:
            if not (isinstance(e, retryable) or QUOTA_HINT.search(str(e))):
                raise
            wait = min(30 * 2 ** i, 600)
            print(f"  {type(e).__name__} on {label}, retry in {wait}s", flush=True)
            sleep(wait)
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
        # Gemini free tier allows 100 embedded texts/min and each text counts as a request.
        for i in range(0, len(texts), 20):
            batch = texts[i:i + 20]
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


def cell_key(model, strategy, trigger, hints, pressure, coalesce, search, dedup, recall, budget, task_id):
    return (model, strategy, trigger, C.stub_mode(hints), pressure, coalesce, search, dedup, recall, budget, task_id)


def flag_list(p, name, raw, allowed):
    values = raw.split(",")
    if not set(values) <= set(allowed):
        p.error(f"unknown --{name} value in {raw!r}; choose from {', '.join(allowed)}")
    return values


def bool_list(p, name, raw):
    return [v == "true" for v in flag_list(p, name, raw, ("true", "false"))]


def main():
    p = argparse.ArgumentParser(description="Sweep GC strategies over needle tasks. Resumable: finished cells are skipped.")
    p.add_argument("--models", default="openrouter/nvidia/nemotron-3.5-lightning:free")
    p.add_argument("--strategies", default=",".join(n for n in C.STRATEGIES if n != "semantic"))
    p.add_argument("--triggers", default="threshold:0.7", help="comma list of every_turn / threshold:P / threshold:P:T (collect down to T*budget)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--positions", default="0.5")
    p.add_argument("--family", default="needle", help="comma list of needle / multihop / poison / paraphrase")
    p.add_argument("--upfront", default="false", help="comma list of true/false")
    p.add_argument("--hints", default="false", help="comma list of false/true/none/keys/preview/numbers/fresh/pointer: what collected stubs carry (false=none, true=keys)")
    p.add_argument("--pressure", default="false", help="comma list of true/false: ephemeral memory-usage note each call")
    p.add_argument("--coalesce", default="false", help="comma list of true/false: render runs of collected tool results as one message")
    p.add_argument("--search", default="false", help="comma list of true/false: offer a search tool over collected messages")
    p.add_argument("--dedup", default="false", help="comma list of true/false: collect older results of repeated list_files/read_file calls")
    p.add_argument("--recall", default="in_place", help="comma list of in_place/tail: restore a recalled message where it was, or return its content as the tool result at the end")
    p.add_argument("--tag", default="", help="experiment label stored in each row (not part of the resume key)")
    p.add_argument("--n-files", type=int, default=30)
    p.add_argument("--budget", type=int, default=16000)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--max-cost", type=float, default=0.0, help="USD cap over all runs in runs.jsonl")
    a = p.parse_args()
    try:
        hint_modes = [C.stub_mode(h) for h in a.hints.split(",")]
    except KeyError as e:
        p.error(f"unknown --hints value {e.args[0]!r}; choose from {', '.join(k for k in C.MODES if isinstance(k, str))}")

    upfront_modes = bool_list(p, "upfront", a.upfront)
    pressure_modes = bool_list(p, "pressure", a.pressure)
    coalesce_modes = bool_list(p, "coalesce", a.coalesce)
    search_modes = bool_list(p, "search", a.search)
    dedup_modes = bool_list(p, "dedup", a.dedup)
    recall_modes = flag_list(p, "recall", a.recall, ("in_place", "tail"))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    done = {cell_key(r["model"], r["strategy"], r["trigger"], r["hints"], r["pressure"],
                     r["coalesce"], r["search"], r["dedup"], r["recall"], r["budget"], r["task_id"]) for r in load_runs()}

    for model in a.models.split(","):
        complete = litellm_complete(model)
        cells = itertools.product(a.family.split(","), upfront_modes, map(float, a.positions.split(",")),
                                  map(int, a.seeds.split(",")), a.triggers.split(","), a.strategies.split(","),
                                  hint_modes, pressure_modes, coalesce_modes, search_modes, dedup_modes,
                                  recall_modes)
        for family, upfront, pos, seed, trigger, strategy, hints, pressure, coalesce, search, dedup, recall in cells:
            task = BUILDERS[family](seed, a.n_files, position=pos, upfront=upfront)
            if cell_key(model, strategy, trigger, hints, pressure, coalesce, search, dedup, recall, a.budget, task["id"]) in done:
                continue
            spent = sum(r["cost_usd"] for r in load_runs())  # re-read: parallel sweeps share runs.jsonl and the cap
            if is_paid(model) and spent >= a.max_cost:
                print(f"skip {model}: spent ${spent:.4f} >= cap ${a.max_cost}")
                break
            print(f"{model} {strategy} {trigger} hints={hints} pressure={pressure} coalesce={coalesce} search={search} dedup={dedup} recall={recall} {task['id']}", flush=True)
            try:
                row, trace = run(task, model, strategy, trigger, a.budget, complete, a.max_turns, hints, pressure, a.tag, coalesce=coalesce, search=search, dedup=dedup, recall=recall)
            except RateLimited:
                print(f"  {model} still rate-limited, moving to next model")
                break
            save(row, trace)
            print(f"  -> {row['end']} success={row['success']} in={row['input_tokens']} recalls={row['recalls']} searches={row['searches']}")


if __name__ == "__main__":
    main()
