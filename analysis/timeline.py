import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

NONE, LIVE, COLLECTED, RESTORED = 0, 1, 2, 3
COLORS = ["#f2f2f2", "#4c9f70", "#d9534f", "#f0ad4e"]
NEEDLE = "The access code for project"


def recalled_ids(msg):
    for c in msg.get("tool_calls") or []:
        if c["function"]["name"] == "recall":
            yield json.loads(c["function"]["arguments"] or "{}").get("id")


def states(trace):
    heap, turns = trace["heap"], trace["row"]["turns"]
    by_id = {o["id"]: o for o in heap}
    grid = {o["id"]: [LIVE if t >= o["turn"] else NONE for t in range(turns + 1)] for o in heap}
    for t in range(turns + 1):
        for e in trace["events"]:
            if e["turn"] == t:
                for i in e["collected"]:
                    grid[i][t:] = [COLLECTED] * (turns + 1 - t)
        for o in heap:
            if o["turn"] == t and o["msg"]["role"] == "assistant":
                for i in recalled_ids(o["msg"]):
                    if i in grid and grid[i][t] == COLLECTED:
                        grid[i][t:] = [RESTORED] * (turns + 1 - t)
    return [grid[o["id"]] for o in heap]


def original(o):
    return o["saved"] if o["saved"] is not None else o["msg"].get("content") or ""


def needle_rows(trace):
    heap = trace["heap"]
    asked = [re.search(r"code for project (\w+)\?", o["msg"].get("content") or "") for o in heap if o["msg"]["role"] == "user"]
    target = next((m.group(1) for m in asked if m), None)
    rows = [i for i, o in enumerate(heap) if o["msg"]["role"] == "tool" and NEEDLE in original(o)]
    return rows, [i for i in rows if f"{NEEDLE} {target} is" in original(heap[i])]


def draw(trace, out):
    row, heap = trace["row"], trace["heap"]
    grid, (needles, target) = states(trace), needle_rows(trace)
    fig, (top, ax) = plt.subplots(2, 1, figsize=(11, 2 + len(heap) * 0.16), sharex=True,
                                  gridspec_kw={"height_ratios": [1, max(3, len(heap) / 6)]})
    ev = trace["events"]
    top.plot([e["turn"] for e in ev], [e["before"] for e in ev], "o", color="#d9534f", ms=3, label="before GC")
    top.plot([e["turn"] for e in ev], [e["after"] for e in ev], "o", color="#4c9f70", ms=3, label="after GC")
    top.set_ylabel("est. tokens")
    top.legend(fontsize=7, loc="upper left")
    ax.imshow(grid, cmap=ListedColormap(COLORS), vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_xlabel("turn")
    ax.set_ylabel("message id")
    ax.set_yticks(range(len(heap)))
    ax.set_yticklabels([o["id"] for o in heap], fontsize=6)
    for i, lab in enumerate(ax.get_yticklabels()):
        if i in needles:
            lab.set_color("#1f4fd8")
        if i in target:
            lab.set_fontweight("bold")
            lab.set_fontsize(8)
            ax.plot(-0.9, i, marker=">", color="#1f4fd8", clip_on=False)
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, ["not created", "live", "collected", "restored"])]
              + [Patch(color="#1f4fd8", label="needle row (bold = asked project)")],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=5)
    fig.suptitle(f"{row['model']} | {row['strategy']} | {row['trigger']} | success={row['success']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    draw(json.loads(src.read_text(encoding="utf-8")), sys.argv[2] if len(sys.argv) > 2 else src.with_suffix(".png"))
