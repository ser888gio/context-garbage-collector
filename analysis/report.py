import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["model", "strategy", "trigger"]
RUNS = "results/runs.jsonl"


def cli(out):
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=RUNS)
    ap.add_argument("--out", default=out)
    args = ap.parse_args()
    return args.path, args.out


def load(path):
    df = pd.read_json(path, lines=True)
    df["cached_share"] = df.cached_input_tokens / df.input_tokens
    df["oom"] = df.end == "oom"
    oracle = df[df.strategy == "oracle"].assign(ppt=lambda d: d.input_tokens / d.turns)
    ref = oracle.groupby(["model", "task_id", "hints"]).ppt.mean().rename("oracle_ppt")
    df = df.join(ref, on=["model", "task_id", "hints"])
    df["excess_tokens"] = df.input_tokens / df.turns - df.oracle_ppt
    return df


def summarize(df, keys, sums, means):
    named = {c: (c, "sum") for c in sums} | {c: (c, "mean") for c in means}
    return df.groupby(keys).agg(runs=("success", "size"), success_rate=("success", "mean"), **named).reset_index()


def table(path):
    return summarize(load(path), KEYS, ["oom"],
                     ["input_tokens", "cached_share", "recalls", "excess_tokens", "max_est_tokens"])


def plot(t, out):
    fig, ax = plt.subplots()
    ax.scatter(t.input_tokens, t.success_rate)
    for _, r in t.iterrows():
        ax.annotate(f"{r.model.split('/')[-1]} {r.strategy} {r.trigger}", (r.input_tokens, r.success_rate), fontsize=6)
    ax.set_xlabel("mean input_tokens")
    ax.set_ylabel("success rate")
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    path, out = cli("results/success_vs_tokens.png")
    t = table(path)
    print(t.to_markdown(index=False, floatfmt=".2f"))
    plot(t, out)
