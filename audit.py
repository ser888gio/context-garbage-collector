import glob
import json
import re
import sys

import pandas as pd


def original(m):
    return m["saved"] if m["saved"] is not None else m["msg"].get("content") or ""


def audit(trace):
    row, heap = trace["row"], trace["heap"]
    prompts = " ".join(m["msg"].get("content") or "" for m in heap if m["msg"]["role"] == "user")
    asked = re.search(r"access code for project (\w+)\?", prompts)
    fact = asked and f"The access code for project {asked.group(1)} is"
    # runs that hit oom never asked the question, so they have no needle
    needle = next((m["id"] for m in heap if fact and m["msg"]["role"] == "tool" and fact in original(m)), None)
    recalls = [json.loads(c["function"]["arguments"] or "{}").get("id")
               for m in heap for c in m["msg"].get("tool_calls") or [] if c["function"]["name"] == "recall"]
    return {"run_id": row["run_id"], "model": row["model"], "strategy": row["strategy"], "trigger": row["trigger"],
            "success": row["success"], "needle_id": needle,
            "needle_collected": any(needle in e["collected"] for e in trace["events"]),
            "needle_recalled": needle in recalls, "wasted_recalls": sum(r != needle for r in recalls)}


def main(pattern="results/traces/*.json"):
    df = pd.DataFrame([audit(json.load(open(f, encoding="utf-8"))) for f in sorted(glob.glob(pattern))])
    print(df.to_markdown(index=False))
    print()
    print(df.groupby("strategy")[["success", "needle_collected", "needle_recalled", "wasted_recalls"]].mean().to_markdown())


if __name__ == "__main__":
    main(*sys.argv[1:])
