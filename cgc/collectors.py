import json
import math
import re
from dataclasses import dataclass

# ponytail: regex keys (paths, URLs, CAPS names, 4+ digit numbers, short quotes); swap for an LLM/embedding linker as its own strategy
KEY_RE = re.compile(r'https?://\S+|[\w./-]+\.\w{1,5}\b|\b[A-Z][A-Z0-9_-]{2,}\b|\b\d{4,}\b|"[^"]{3,40}"')


@dataclass
class Obj:
    id: int
    msg: dict
    turn: int
    root: bool = False
    label: str = ""
    saved: str | None = None
    dead: bool = False
    touched: int = 0
    last_used: int = 0
    survived: int = 0
    # Without this, the summary message (still live and collectable) would be re-summarized every GC run.
    summarized: bool = False

    @property
    def live(self):
        return not self.dead

    @property
    def is_referencer(self):
        # Tool results are data: they hold references only as targets. Otherwise one list_files
        # result links every file and becomes a hub that keeps the whole heap reachable.
        return self.root or self.msg["role"] in ("assistant", "user", "system")


def arg_values(arguments):
    # Raw JSON would make '"path"' a shared key between every read_file call.
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments or ""
    return " ".join(str(v) for v in args.values()) if isinstance(args, dict) else str(args)


def text(o):
    calls = " ".join(c["function"]["name"] + " " + arg_values(c["function"]["arguments"])
                     for c in o.msg.get("tool_calls") or [])
    return f"{o.msg.get('content') or ''} {calls} {o.label}"


def keys(o):
    return set(KEY_RE.findall(text(o)))


def size(heap):
    # ponytail: chars/4 estimate, same for every model so the artificial budget is comparable
    return sum(len(json.dumps(o.msg)) for o in heap) // 4


def now(heap):
    return max(o.turn for o in heap)


def protected(o, t):
    # The newest turn's messages haven't been seen by the model yet; a just-recalled message
    # would otherwise be collected again before the model reads it.
    return o.root or o.turn == t or o.touched == t


def collectable(heap):
    t = now(heap)
    return [o for o in heap if o.live and not protected(o, t)]

NUM_RE = re.compile(r"\b\d{4,}\b")
MODES = {False: "none", "false": "none", "none": "none", True: "keys", "true": "keys", "keys": "keys",
         "preview": "preview", "numbers": "numbers"}


def stub_mode(hints):
    return MODES[hints]


def kill(o, hints=False):
    mode = stub_mode(hints)
    o.dead = True
    content = o.msg.get("content")
    what = f"msg {o.id} {o.label}".strip()
    found = sorted(k for k in keys(o) if k not in o.label)[:6] if mode != "none" else []
    what += f" — keys: {', '.join(found)}" if found else ""
    if mode == "numbers" and content:
        nums = list(dict.fromkeys(NUM_RE.findall(content)))
        what += f" — numbers: {', '.join(nums)}" if nums else ""
    if mode == "preview" and content:
        what += f' — preview: "{" ".join(content.split()[:12])}"'
    stub = f"[collected: {what}; call recall({o.id}) to restore]"
    if content and o.saved is None and len(stub) < len(content):
        o.saved = content
        o.msg["content"] = stub


def restore(heap, obj_id, t):
    o = next((o for o in heap if o.id == obj_id), None)
    if o is None or o.saved is None:
        return False
    o.msg["content"], o.saved, o.dead = o.saved, None, False
    o.touched = o.last_used = t
    return True


def reachable(heap, extra_roots=(), chain_max_keys=None):
    t = now(heap)
    live = [o for o in heap if o.live]
    k = {o.id: keys(o) for o in live}
    seen = {o.id for o in live if protected(o, t)} | {o.id for o in extra_roots}
    stack = [o for o in live if o.id in seen]
    while stack:
        src = stack.pop()
        if not (src.is_referencer or (chain_max_keys is not None and len(k[src.id]) <= chain_max_keys)):
            continue
        for o in live:
            if o.id not in seen and k[src.id] & k[o.id]:
                seen.add(o.id)
                stack.append(o)
    return seen


def none(heap, st):
    return []


def truncate(heap, st):
    target = st["budget"] * st.get("target", 0.5)
    out = []
    for o in collectable(heap):
        if size(heap) <= target:
            break
        kill(o, st.get("hints", False))
        out.append(o.id)
    return out


def lru_mask(heap, st):
    t = now(heap)
    users = [o for o in heap if o.live and o.msg["role"] == "assistant"]
    for o in heap:
        ko = keys(o)
        for u in users:
            if u.turn > o.last_used and ko & keys(u):
                o.last_used = u.turn
    out = [o for o in collectable(heap) if o.msg["role"] == "tool" and t - o.last_used >= st.get("lru_turns", 3)]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


def mask(heap, st):
    t = now(heap)
    out = [o for o in collectable(heap) if o.msg["role"] == "tool" and t - o.turn >= st.get("mask_turns", 3)]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


def hybrid(heap, st):
    out = mask(heap, st)
    if "summarize" not in st or size(heap) <= st["budget"] * st.get("threshold", 0.7):
        return out
    old = [o for o in collectable(heap) if not o.summarized]
    if not old:
        return out
    summary = st["summarize"]([f"{o.msg['role']}: {text(o)}" for o in old])
    for o in old[1:]:
        kill(o, st.get("hints", False))
    old[0].msg["content"] = f"[summary of msgs {old[0].id}-{old[-1].id}] {summary}"
    old[0].summarized = True
    return out + [o.id for o in old[1:]]


def refcount(heap, st):
    t = now(heap)
    out = []
    changed = True
    while changed:
        changed = False
        live = [o for o in heap if o.live]
        k = {o.id: keys(o) for o in live}
        for o in live:
            if protected(o, t):
                continue
            if not any(r.id != o.id and r.is_referencer and k[r.id] & k[o.id] for r in live):
                kill(o, st.get("hints", False))
                out.append(o.id)
                changed = True
                break
    return out


def mark_sweep(heap, st):
    seen = reachable(heap)
    out = [o for o in collectable(heap) if o.id not in seen]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


def mark_sweep_chain(heap, st):
    # A reached tool result also propagates, but only if it has at most chain_max_refs keys. A link or value
    # file carries ~3 (name, name, path); a list_files result carries one path per file, so it stays a leaf.
    # A per-key fanout guard cannot do this: every path appears in exactly 3 messages (listing, call, result).
    seen = reachable(heap, chain_max_keys=st.get("chain_max_refs", 3))
    out = [o for o in collectable(heap) if o.id not in seen]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


CAPS_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")


def facts(o):
    # A clause (split on . ; newline) with one CAPS key asserts its LAST 4+ digit number. Splitting on ";"
    # keeps "ORION is now 7310; the old code 4821 is revoked" as ORION = 7310, not the revoked 4821.
    out = {}
    for sentence in re.split(r"[.!?](?:\s|$)|[;\n]", o.msg.get("content") or ""):
        caps, nums = set(CAPS_RE.findall(sentence)), NUM_RE.findall(sentence)
        if len(caps) == 1 and nums:
            out[caps.pop()] = nums[-1]
    return out


def write_barrier(heap, st):
    live = [o for o in heap if o.live]
    f = {o.id: facts(o) for o in live}
    stale = [o for o in collectable(heap)
             if any(f[p.id].get(k, v) != v for p in live if p.id > o.id for k, v in f[o.id].items())]
    for o in stale:
        kill(o, st.get("hints", False))
    return [o.id for o in stale] + mark_sweep(heap, st)


def generational(heap, st):
    k_promote = st.get("promote_after", 2)
    old = [o for o in heap if o.live and o.survived >= k_promote]
    seen = reachable(heap, extra_roots=old)
    young = [o for o in collectable(heap) if o.survived < k_promote]
    out = []
    for o in young:
        if o.id in seen:
            o.survived += 1
        else:
            kill(o, st.get("hints", False))
            out.append(o.id)
    st["major"] = size(heap) > st["budget"] * st.get("threshold", 0.7)
    if st["major"]:
        out += mark_sweep(heap, st)
    return out


def h2o(heap, st):
    t = now(heap)
    live = [o for o in heap if o.live and not o.root]
    pinned = {o.id for o in live[:st.get("h2o_sink", 2)]} | {o.id for o in live if o.turn > t - st.get("h2o_recent", 2)}
    pool = [o for o in collectable(heap) if o.id not in pinned]
    k = {o.id: keys(o) for o in heap if o.live}
    refs = {o.id: sum(r.id != o.id and r.is_referencer and bool(k[r.id] & k[o.id]) for r in heap if r.live) for o in pool}
    # Zero-reference results are never "heavy"; without this the top-M slots would keep arbitrary garbage.
    heavy = sorted((o for o in pool if o.msg["role"] == "tool" and refs[o.id]), key=lambda o: (-refs[o.id], o.id))
    keep = {o.id for o in heavy[:st.get("h2o_heavy", 4)]}
    out = [o for o in pool if o.id not in keep]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


def forget(heap, ids, hints=False):
    t = now(heap)
    by_id = {o.id: o for o in heap}
    done, refused = [], []
    for i in ids:
        o = by_id.get(i)
        if o is None or o.dead or protected(o, t):
            refused.append(i)
        else:
            kill(o, hints)
            done.append(i)
    return done, refused


def oracle(heap, st):
    out = [o for o in collectable(heap)
           if not any(k in (o.saved if o.saved is not None else o.msg.get("content") or "") for k in st.get("oracle_keys", ()))]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def semantic(heap, st):
    if "embed" not in st:
        raise ValueError("semantic strategy needs st['embed'], a callable list[str] -> list[vector]")
    cache = st.setdefault("_emb", {})
    pool = collectable(heap)
    objs = [o for o in heap if o.live and o.root] + pool
    want = {(o.id, hash(text(o))): text(o) for o in objs if (o.id, hash(text(o))) not in cache}
    if want:
        cache.update(zip(want, st["embed"](list(want.values()))))
    vec = {o.id: cache[(o.id, hash(text(o)))] for o in objs}
    roots = [vec[o.id] for o in objs if o.root]
    centre = [sum(col) for col in zip(*roots)] if roots else []
    ranked = sorted(pool, key=lambda o: (-cosine(vec[o.id], centre), o.id))
    out = ranked[st.get("semantic_k", 4):]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


def with_target(fn):
    # Hysteresis for reachability collectors: after the normal pass, also collect the oldest
    # reachable-but-not-protected tool results until size <= target * budget.
    def wrapped(heap, st):
        out = fn(heap, st)
        for o in collectable(heap):
            if "target" not in st or size(heap) <= st["budget"] * st["target"]:
                break
            if o.msg["role"] == "tool":
                kill(o, st.get("hints", False))
                out.append(o.id)
        return out
    wrapped.__name__ = fn.__name__
    return wrapped


STRATEGIES = {f.__name__: f for f in (none, truncate, lru_mask, mask, hybrid, refcount, mark_sweep, mark_sweep_chain, write_barrier,
                                             generational, h2o, oracle, semantic)}
for _n in ("refcount", "mark_sweep", "mark_sweep_chain", "write_barrier", "generational", "semantic"):
    STRATEGIES[_n] = with_target(STRATEGIES[_n])
STRATEGIES["manual"] = none
