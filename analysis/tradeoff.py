import sys

import matplotlib
import pandas as pd

from analysis.report import load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ["mark_sweep", "refcount", "lru_mask", "truncate", "write_barrier", "oracle", "generational", "h2o"]
STYLE = {"needle": ("#1f77b4", "o"), "multihop": ("#d95f02", "s"), "poison": ("#7b3294", "^")}


def points(path):
    df = load(path)
    df = df[df.model.str.contains("gpt-oss-120b") & (df.trigger == "threshold:0.7") & ~df.pressure
            & (df.hints == "keys") & df.strategy.isin(STRATEGIES)]
    return df.groupby(["task_family", "strategy"]).agg(
        n=("success", "size"), success=("success", "mean"), tokens=("input_tokens", "mean")).reset_index()


def plot(pts, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fam, (color, marker) in STYLE.items():
        p = pts[pts.task_family == fam]
        ax.scatter(p.tokens / 1000, p.success, c=color, marker=marker, s=70, label=fam, alpha=0.85)
        for _, r in p.iterrows():
            ax.annotate(f"{r.strategy} ({r.n})", (r.tokens / 1000, r.success), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="mean input tokens (k)", ylabel="success rate", ylim=(-0.05, 1.1),
           title="Cost vs accuracy, gpt-oss-120b, keys hints (n in brackets)")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    pts = points(sys.argv[1])
    print(pts.round(2).to_string())
    plot(pts, "docs/img/tradeoff.png")
