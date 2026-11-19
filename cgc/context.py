import copy
import json

from cgc import collectors as C

# The sweep results will decide the final values.
DEFAULTS = dict(strategy="mark_sweep", trigger="threshold:0.7", budget=16000, hints="pointer", coalesce=True,
                search=True, dedup=False, recall="tail")

RECALL_PARAMS = {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}
RECALL_TOOL = {"type": "function", "function": {
    "name": "recall", "description": "Restore a collected message by its id.", "parameters": RECALL_PARAMS}}
TAIL_RECALL_TOOL = {"type": "function", "function": {
    "name": "recall", "description": "Return a collected message's content as this call's result; the stub stays in place.",
    "parameters": RECALL_PARAMS}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the text of collected messages without restoring them. "
                                     "Returns matching sentences with their message ids.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}


def parse_trigger(trigger):
    parts = trigger.split(":")
    threshold = float(parts[1]) if len(parts) > 1 else 0.7
    target = float(parts[2]) if len(parts) == 3 else None
    return threshold, target


def describe_call(c):
    try:
        args = json.loads(c["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    name = c["function"]["name"]
    return f"{name} {args.get('path', '')}".strip(), name + json.dumps(args, sort_keys=True)


class Context:
    def __init__(self, strategy=DEFAULTS["strategy"], trigger=DEFAULTS["trigger"], budget=DEFAULTS["budget"],
                 hints=DEFAULTS["hints"], coalesce=DEFAULTS["coalesce"], search=DEFAULTS["search"],
                 dedup=DEFAULTS["dedup"], recall=DEFAULTS["recall"], embed=None, summarize=None, oracle_keys=None):
        if strategy not in C.STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(C.STRATEGIES)}")
        if strategy == "semantic" and embed is None:
            raise ValueError("the semantic strategy needs embed, a callable list[str] -> list[vector]")
        if strategy == "oracle" and not oracle_keys:
            raise ValueError("the oracle strategy is an experiment baseline and needs oracle_keys")
        if recall not in ("in_place", "tail"):
            raise ValueError(f"recall must be in_place or tail, not {recall!r}")
        self.strategy, self.trigger, self.budget = strategy, trigger, budget
        self.coalesce, self.search, self.recall = coalesce, search, recall
        self.hints = C.stub_mode(hints)
        threshold, target = parse_trigger(trigger)
        self.st = {"budget": budget, "hints": self.hints, "threshold": threshold, "dedup": dedup}
        if target is not None:
            self.st["target"] = target
        if embed:
            self.st["embed"] = embed
        if summarize:
            self.st["summarize"] = summarize
        if oracle_keys:
            self.st["oracle_keys"] = set(oracle_keys)
        self.tools = [TAIL_RECALL_TOOL if recall == "tail" else RECALL_TOOL] + [SEARCH_TOOL] * search
        self.stats = dict(gc_runs=0, tokens_freed=0, recalls=0, searches=0, dedups=0)
        self.events = []
        self.turn = 0
        self._reset()

    def _reset(self):
        self.heap, self._seen, self._calls, self._has_user = [], [], {}, False

    def sync(self, messages, pinned=()):
        n = len(self._seen)
        if messages[:n] != self._seen:
            self._reset()
            n = 0
        for i, m in enumerate(messages[n:], n):
            self._add(i, m, i in pinned)

    def _add(self, i, m, pinned):
        label = call = ""
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                self._calls[c["id"]] = describe_call(c)
        elif m["role"] == "tool":
            label, call = self._calls.get(m.get("tool_call_id"), ("", ""))
        root = pinned or (i == 0 and m["role"] == "system") or (m["role"] == "user" and not self._has_user)
        self._has_user |= m["role"] == "user"
        self._seen.append(copy.deepcopy(m))
        self.heap.append(C.Obj(i, dict(m), self.turn, root=root, label=label, call=call, touched=self.turn,
                               last_used=self.turn))

    def prepare(self, messages, pinned=()):
        self.sync(messages, pinned)
        self.turn += 1
        before = C.size(self.heap)
        if self.trigger == "every_turn" or before > self.budget * self.st["threshold"]:
            freed = C.collect(self.strategy, self.heap, self.st)
            self.stats["gc_runs"] += 1
            self.stats["dedups"] = self.st.get("dedups", 0)
            self.stats["tokens_freed"] += before - C.size(self.heap)
            self.events.append({"turn": self.turn, "before": before, "after": C.size(self.heap), "collected": freed})
        return [dict(m) for m in C.render(self.heap, self.coalesce)]

    def call(self, name, args):
        if name == "recall":
            self.stats["recalls"] += 1
            obj_id = int(args.get("id", -1))
            if self.recall == "in_place":
                return "restored" if C.restore(self.heap, obj_id, self.turn) else "error: nothing to restore"
            o = next((o for o in self.heap if o.id == obj_id), None)
            return f"[recalled msg {obj_id}] {o.saved}" if o and o.saved is not None else "error: nothing to restore"
        if name == "search" and self.search:
            self.stats["searches"] += 1
            return C.search(self.heap, str(args.get("query", "")))
        return None
