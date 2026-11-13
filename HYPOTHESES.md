# Hypotheses and Roadmap

## Hypotheses

Each one is checked against `results/runs.jsonl`. Status: open / supported / rejected / unclear.

| # | Hypothesis | How we check it | Status |
|---|---|---|---|
| H1 | Without GC the agent fails once context exceeds the budget. | `none` ends with `oom` on 30-file needle tasks. | open |
| H2 | Any GC beats no GC on long tasks. | success rate of each strategy vs `none`. | open |
| H3 | Mark-sweep keeps the needle when the question is in the task, and loses it when the question comes late. | `upfront=true` vs `false`: success and recalls. | open |
| H4 | When the GC drops something needed, models use `recall` to get it back. | recalls > 0 on runs that still succeed. | open |
| H5 | Refcount keeps more garbage than mark-sweep (it leaks cycles). | tokens_freed and max_est_tokens: refcount vs mark_sweep. | open |
| H6 | Generational GC keeps a stable prefix, so it gets more cache hits. | cached_input_tokens / input_tokens per strategy; provider-independent: `stable_prefix_share` and `cache_breaks`. | open |
| H7 | Collecting every turn costs fewer tokens but breaks the cache more than threshold collection. | `every_turn` vs `threshold:0.7`: input and cached tokens, plus `stable_prefix_share` and `cache_breaks`. | open |
| H8 | Simple LRU masking is as good as the smarter collectors (as in "The Complexity Trap"). | lru_mask vs mark_sweep/generational: success and tokens. | open |
| H9 | GC can make the agent thrash: recall, re-collect, recall again. | repeated recalls of the same id in traces. | open |
| H10 | The oracle sets the upper bound; real collectors reach X% of its success at Y% of its tokens. | `oracle` vs each strategy: success and input_tokens; `audit.py` for what each lost. | open |

## Roadmap

Done:
- Collectors: none, truncate, lru_mask, refcount, mark_sweep, generational
- Oracle collector (upper bound) and `audit.py` (needle collected / recalled / wasted recalls per run)
- Needle task, agent loop, resumable sweep CLI, offline tests
- `report.py`: results table and success-vs-tokens plot

Next:
1. First sweep on free models; write findings to `JOURNAL.md`.
2. Task families: poisoning (wrong fact corrected later), long chat, research.
3. Vary needle position (lost-in-the-middle) and budget.
4. New strategies: LLM-based reference finder, summarize instead of stub.
5. Repeat the best strategies on a stronger model.

## Known limits

- References are found by regex (paths, CAPS names, numbers). Plain-word links are missed.
- Token sizes are estimated as chars/4, the same for every model.
- Free models are rate-limited; sweeps run slowly and resume where they stopped.
