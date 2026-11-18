import matplotlib

from analysis.report import cli, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ["needle", "multihop", "poison"]
MOVERS = {"mark_sweep": "#1f77b4", "refcount": "#d95f02", "write_barrier": "#2ca02c"}
REFS = ["oracle", "truncate", "lru_mask"]
MIN_N = 8
MODEL = "gpt-oss-120b"


def points(path):
    df = load(path)
    df = df[df.model.str.contains(MODEL) & (df.trigger == "threshold:0.7") & ~df.pressure
            & df.strategy.isin(list(MOVERS) + REFS)]
    df = df.assign(api_errors=df.error.ne(""))
    keys = (df.hints == "keys") & ~df.coalesce
    ptr = (df.hints == "pointer") & df.coalesce
    df = df[keys | ptr].assign(variant=lambda d: (d.hints == "pointer").map({True: "pointer+coalesce", False: "keys"}))
    df = df[df.strategy.isin(MOVERS) | (df.variant == "keys")]
    out = df.groupby(["task_family", "strategy", "variant"]).agg(
        n=("success", "size"), success=("success", "mean"), tokens=("input_tokens", "mean"),
        api_errors=("api_errors", "sum")).reset_index()
    out = out[out.n >= MIN_N].rename(columns={"task_family": "family"})
    return out.reset_index(drop=True)


def plot(pts, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        p = pts[pts.family == fam]
        for _, r in p[p.strategy.isin(REFS)].iterrows():
            ax.scatter(r.tokens / 1000, r.success, c="#888888", marker="D", s=40, zorder=2)
            ax.annotate(r.strategy, (r.tokens / 1000, r.success), fontsize=8, color="#555555",
                        xytext=(6, 6), textcoords="offset points")
        for s, color in MOVERS.items():
            a = p[(p.strategy == s) & (p.variant == "keys")]
            b = p[(p.strategy == s) & (p.variant == "pointer+coalesce")]
            if len(a):
                ax.scatter(a.tokens / 1000, a.success, facecolors="white", edgecolors=color, s=70, linewidths=1.8, zorder=3)
            if len(b):
                ax.scatter(b.tokens / 1000, b.success, c=color, s=70, zorder=3, label=s)
            if len(a) and len(b):
                ax.annotate("", (b.tokens.iloc[0] / 1000, b.success.iloc[0]), (a.tokens.iloc[0] / 1000, a.success.iloc[0]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.4, shrinkA=5, shrinkB=5), zorder=2)
        ax.set(title=fam, xlabel="mean input tokens (k)")
        ax.grid(alpha=0.3)
        ax.margins(x=0.15)
    axes[0].set(ylabel="success rate", ylim=(-0.05, 1.1))
    handles, labels = axes[2].get_legend_handles_labels()
    ref = plt.Line2D([], [], marker="D", ls="", c="#888888")
    hollow = plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black")
    fig.legend(handles + [ref, hollow], labels + ["reference (keys hints)", "hollow = keys, filled = pointer+coalesce"],
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    n = pts.n
    fig.suptitle(f"Cost vs accuracy: keys hints → pointer + coalesce ({MODEL}, {n.max()} seeds)", fontsize=11)
    note = f"n per point: {n.min()}-{n.max()} runs; points with fewer than {MIN_N} runs omitted"
    if pts.api_errors.sum():
        note += f"; {int(pts.api_errors.sum())} API-error runs counted as failures"
    fig.text(0.5, 0.905, note, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    path, out = cli("docs/img/tradeoff.png")
    pts = points(path)
    print(pts.round(2).to_string())
    plot(pts, out)
