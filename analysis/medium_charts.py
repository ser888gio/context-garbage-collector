"""Charts for docs/medium-draft.md. Usage: python -m analysis.medium_charts [runs.jsonl] [out_dir]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

MODEL, TRIGGER, BUDGET = "groq/openai/gpt-oss-120b", "threshold:0.7", 16000
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e3e2de"
AUTO, MODEL_MANAGED, ORACLE, BEFORE = "#2a78d6", "#eb6834", "#9a9992", "#9ec5f4"

STRATEGIES = ["oracle", "truncate", "lru_mask", "hybrid", "mark_sweep", "refcount",
              "write_barrier", "manual", "compress"]
FAMILIES = ["needle", "multihop", "poison"]


def load(path):
    cells = defaultdict(lambda: [0, 0])
    for line in open(path):
        r = json.loads(line)
        if (r["model"], r["trigger"], r["budget"]) != (MODEL, TRIGGER, BUDGET):
            continue
        if r["pressure"] or r["dedup"] or r["recall"] != "in_place":
            continue
        key = (r["task_family"], r["strategy"], r["hints"], r["coalesce"], r["search"])
        cells[key][0] += bool(r["success"])
        cells[key][1] += 1
    return cells


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def strategy_chart(cells, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, fam in zip(axes, FAMILIES):
        style(ax)
        for i, s in enumerate(STRATEGIES):
            ok, n = cells[(fam, s, "keys", False, False)]
            assert n == 10, (fam, s, n)
            color = ORACLE if s == "oracle" else MODEL_MANAGED if s in ("manual", "compress") else AUTO
            ax.barh(i, ok, height=0.6, color=color)
            ax.text(ok + 0.2, i, f"{ok}/{n}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 11.5)
        ax.set_xticks([0, 5, 10])
        ax.set_title(fam, color=INK, fontsize=11, loc="left")
    axes[0].set_yticks(range(len(STRATEGIES)), STRATEGIES, color=INK)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORACLE, AUTO, MODEL_MANAGED)]
    fig.legend(handles, ["oracle (upper bound)", "automatic collector", "model manages its own memory"],
               loc="lower center", ncol=3, frameon=False, labelcolor=MUTED)
    fig.suptitle("Successful runs out of 10, by collector and task (gpt-oss-120b, keys stubs)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def fixes_chart(cells, out):
    c = lambda *k: cells[k]
    rows = [
        ("needle, mark_sweep\nbare stubs -> key stubs", c("needle", "mark_sweep", "none", False, False), c("needle", "mark_sweep", "keys", False, False)),
        ("needle, mark_sweep, bare stubs\nno search -> search", c("needle", "mark_sweep", "none", False, False), c("needle", "mark_sweep", "none", False, True)),
        ("poison, mark_sweep\nkey stubs -> pointer + coalesce", c("poison", "mark_sweep", "keys", False, False), c("poison", "mark_sweep", "pointer", True, False)),
        ("poison, hybrid\nno search -> search", c("poison", "hybrid", "keys", False, False), c("poison", "hybrid", "keys", False, True)),
        ("multihop, hybrid\nno search -> search", c("multihop", "hybrid", "keys", False, False), c("multihop", "hybrid", "keys", False, True)),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    style(ax)
    for i, (label, (b, bn), (a, an)) in enumerate(rows):
        assert bn == an == 10, label
        ax.barh(i - 0.18, b, height=0.32, color=BEFORE)
        ax.barh(i + 0.18, a, height=0.32, color=AUTO)
        ax.text(b + 0.2, i - 0.18, f"{b}/10", va="center", color=INK, fontsize=9)
        ax.text(a + 0.2, i + 0.18, f"{a}/10", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 11.5)
    ax.set_xticks([0, 5, 10])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BEFORE, AUTO)]
    ax.legend(handles, ["before", "after"], loc="center right", bbox_to_anchor=(1, 0.62), frameon=False, labelcolor=MUTED)
    ax.set_title("What moved the numbers: successful runs out of 10", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    runs = sys.argv[1] if len(sys.argv) > 1 else "results/runs.jsonl"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/medium")
    out.mkdir(parents=True, exist_ok=True)
    cells = load(runs)
    strategy_chart(cells, out / "strategies.png")
    fixes_chart(cells, out / "fixes.png")
