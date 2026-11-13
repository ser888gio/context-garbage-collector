import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["model", "strategy", "trigger"]


def load(path):
    df = pd.read_json(path, lines=True)
    for col, default in {"hints": False, "pressure": False, "tag": "", "error": "", "cache_breaks": float("nan"),
                         "stable_prefix_share": float("nan"), "max_est_tokens": float("nan"),
                         "max_context_tokens": float("nan"), "recalls": float("nan"), "cost_usd": 0.0}.items():
        if col not in df:
            df[col] = default
    df[["hints", "pressure"]] = df[["hints", "pressure"]].fillna(False).astype(bool)
    df["tag"] = df.tag.fillna("")
    df["cached_share"] = df.cached_input_tokens / df.input_tokens
    df["oom"] = df.end == "oom"
    oracle = df[df.strategy == "oracle"].assign(ppt=lambda d: d.input_tokens / d.turns)
    ref = oracle.groupby(["model", "task_id", "hints"]).ppt.mean().rename("oracle_ppt")
    df = df.join(ref, on=["model", "task_id", "hints"])
    df["excess_tokens"] = df.input_tokens / df.turns - df.oracle_ppt
    return df


def table(path):
    df = load(path)
    return df.groupby(KEYS).agg(
        runs=("success", "size"), success_rate=("success", "mean"), oom=("oom", "sum"),
        input_tokens=("input_tokens", "mean"), cached_share=("cached_share", "mean"),
        recalls=("recalls", "mean"), excess_tokens=("excess_tokens", "mean"), max_est_tokens=("max_est_tokens", "mean"),
    ).reset_index()


def plot(t, out):
    fig, ax = plt.subplots()
    ax.scatter(t.input_tokens, t.success_rate)
    for _, r in t.iterrows():
        ax.annotate(f"{r.model.split('/')[-1]} {r.strategy} {r.trigger}", (r.input_tokens, r.success_rate), fontsize=6)
    ax.set_xlabel("mean input_tokens")
    ax.set_ylabel("success rate")
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    t = table(sys.argv[1] if len(sys.argv) > 1 else "results/runs.jsonl")
    print(t.to_markdown(index=False, floatfmt=".2f"))
    plot(t, "results/success_vs_tokens.png")
