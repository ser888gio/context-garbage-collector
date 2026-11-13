import json
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


def kill(o, hints=False):
    o.dead = True
    content = o.msg.get("content")
    what = f"msg {o.id} {o.label}".strip()
    found = sorted(k for k in keys(o) if k not in o.label)[:6] if hints else []
    what += f" — keys: {', '.join(found)}" if found else ""
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


def reachable(heap, extra_roots=()):
    t = now(heap)
    live = [o for o in heap if o.live]
    k = {o.id: keys(o) for o in live}
    seen = {o.id for o in live if protected(o, t)} | {o.id for o in extra_roots}
    stack = [o for o in live if o.id in seen]
    while stack:
        src = stack.pop()
        if not src.is_referencer:
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


def oracle(heap, st):
    out = [o for o in collectable(heap)
           if not any(k in (o.saved if o.saved is not None else o.msg.get("content") or "") for k in st.get("oracle_keys", ()))]
    for o in out:
        kill(o, st.get("hints", False))
    return [o.id for o in out]


STRATEGIES = {f.__name__: f for f in (none, truncate, lru_mask, refcount, mark_sweep, generational, oracle)}
