import argparse
import itertools
import json
import random
import time
import uuid
from pathlib import Path

import collectors as C

RESULTS = Path("results")
RUNS = RESULTS / "runs.jsonl"

WORDS = """river stone quiet window garden paper light morning table bridge forest cloud simple
green market letter yellow honest silver planet engine summer winter careful bottle mountain
little thunder pocket golden distant candle music silent harbor velvet lantern meadow copper
narrow gentle orchard ribbon shadow travel wooden village mirror island autumn basket feather
kettle ladder marble needle pillow saddle tunnel valley wander anchor blanket circle dinner
garage hollow jacket lemon mitten orange parcel rocket sugar tomato violet whistle""".split()
NAMES = ["ORION", "ORIOLE", "ORINOCO", "OREGANO", "ORACLE", "ORCHID", "ORBITAL", "ORPHEUS"]

SYSTEM = ("You are a careful agent working with tools. To save space, older messages may be replaced by "
          "'[collected: msg N ...]' stubs. If you need what a stub held, call recall with its id.")
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


class RateLimited(Exception):
    pass


def needle_task(seed, n_files=30, words_per_file=450, position=0.5, upfront=False):
    rng = random.Random(seed)
    names = rng.sample(NAMES, 4)
    codes = rng.sample(range(1000, 10000), 4)
    needle_at = min(n_files - 1, int(position * n_files))
    fact_files = [needle_at] + rng.sample([i for i in range(n_files) if i != needle_at], 3)
    files = {}
    for i in range(n_files):
        body = [rng.choice(WORDS) for _ in range(words_per_file)]
        if i in fact_files:
            j = fact_files.index(i)
            body.insert(words_per_file // 2, f". The access code for project {names[j]} is {codes[j]}. ")
        files[f"notes/{i:02d}.txt"] = " ".join(body)
    question = f"What is the access code for project {names[0]}? Reply with just the number."
    then = f"Then answer: {question}" if upfront else "When you have read all of them, reply with exactly DONE."
    return {
        "family": "needle",
        "id": f"needle-p{position}-{'up' if upfront else 'late'}-n{n_files}-s{seed}",
        "files": files,
        "prompt": f"The notes/ folder has {n_files} note files. Read every one with read_file, one per call. {then}",
        "question": question,
        "answer": str(codes[0]),
        "upfront": upfront,
    }


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


def run(task, model, strategy, trigger, budget, complete, max_turns=80, hints=False):
    heap, ids = [], itertools.count()

    def add(msg, turn, root=False, label=""):
        heap.append(C.Obj(next(ids), msg, turn, root=root, label=label, touched=turn, last_used=turn))

    add({"role": "system", "content": SYSTEM}, 0, root=True)
    add({"role": "user", "content": task["prompt"]}, 0, root=True)
    st = {"budget": budget, "hints": hints, "threshold": float(trigger.split(":")[1]) if ":" in trigger else 0.7}
    s = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, cost_usd=0.0, gc_runs=0, tokens_freed=0,
             recalls=0, repeated_calls=0, max_context_tokens=0, max_est_tokens=0)
    if strategy == "oracle":
        st["oracle_keys"] = {task["answer"]}
    events, seen_calls, asked, answer, end, t0 = [], set(), task["upfront"], "", "max_turns", time.time()

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

        resp = complete([o.msg for o in heap], TOOLS)
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
            if c.function.name == "recall":
                s["recalls"] += 1
            elif sig in seen_calls:
                s["repeated_calls"] += 1
            seen_calls.add(sig)
            result = call_tool(c.function.name, args, task, heap, turn)
            label = f"{c.function.name} {args.get('path', '')}".strip()
            add({"role": "tool", "tool_call_id": c.id, "content": result}, turn, label=label)

    row = {"run_id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "strategy": strategy,
           "trigger": trigger, "hints": hints, "budget": budget, "task_family": task["family"], "task_id": task["id"],
           "provider": model.split("/")[0], "model": model, "success": task["answer"] in answer, "end": end,
           "answer": answer[:200], "turns": turn, **s, "wall_s": round(time.time() - t0, 1)}
    trace = {"row": row, "events": events,
             "heap": [{"id": o.id, "turn": o.turn, "root": o.root, "dead": o.dead, "survived": o.survived,
                       "label": o.label, "msg": o.msg, "saved": o.saved} for o in heap]}
    return row, trace


def litellm_complete(model):
    import litellm
    litellm.suppress_debug_info = True
    retryable = (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.InternalServerError,
                 litellm.APIConnectionError, litellm.Timeout)

    def complete(messages, tools):
        for i in range(7):
            try:
                return litellm.completion(model=model, messages=messages, tools=tools, temperature=0, timeout=300)
            except retryable as e:
                wait = min(30 * 2 ** i, 600)
                print(f"  {type(e).__name__} on {model}, retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RateLimited(model)
    return complete


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
    p.add_argument("--strategies", default=",".join(C.STRATEGIES))
    p.add_argument("--triggers", default="threshold:0.7", help="comma list of every_turn / threshold:P")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--positions", default="0.5")
    p.add_argument("--upfront", default="false", help="comma list of true/false")
    p.add_argument("--hints", default="false", help="comma list of true/false: keys listed in collected stubs")
    p.add_argument("--n-files", type=int, default=30)
    p.add_argument("--budget", type=int, default=16000)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--max-cost", type=float, default=0.0, help="USD cap over all runs in runs.jsonl")
    a = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(".env")
    done = {(r["model"], r["strategy"], r["trigger"], r.get("hints", False), r["budget"], r["task_id"]) for r in load_runs()}

    for model in a.models.split(","):
        complete = litellm_complete(model)
        cells = itertools.product(a.upfront.split(","), map(float, a.positions.split(",")),
                                  map(int, a.seeds.split(",")), a.triggers.split(","), a.strategies.split(","),
                                  [h == "true" for h in a.hints.split(",")])
        for upfront, pos, seed, trigger, strategy, hints in cells:
            task = needle_task(seed, a.n_files, position=pos, upfront=upfront == "true")
            if (model, strategy, trigger, hints, a.budget, task["id"]) in done:
                continue
            spent = sum(r["cost_usd"] for r in load_runs())
            if is_paid(model) and spent >= a.max_cost:
                print(f"skip {model}: spent ${spent:.4f} >= cap ${a.max_cost}")
                break
            print(f"{model} {strategy} {trigger} hints={hints} {task['id']}", flush=True)
            try:
                row, trace = run(task, model, strategy, trigger, a.budget, complete, a.max_turns, hints)
            except RateLimited:
                print(f"  {model} still rate-limited, moving to next model")
                break
            save(row, trace)
            print(f"  -> {row['end']} success={row['success']} in={row['input_tokens']} recalls={row['recalls']}")


if __name__ == "__main__":
    main()
