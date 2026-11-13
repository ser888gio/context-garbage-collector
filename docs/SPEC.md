# Context Garbage Collector — Experiment Spec

Goal: explore how classic garbage-collection strategies behave when applied to an LLM agent's context. Learning experiment, no fixed hypothesis.

## Prior art
- MemGPT (2023): context as RAM, paging to external storage — https://arxiv.org/abs/2310.08560
- Self-GC (2026): LLM-driven collector over indexed context objects — https://arxiv.org/abs/2607.00692
- The Complexity Trap (NeurIPS 2025): observation masking ≈ LLM summarization at half the cost — https://arxiv.org/abs/2508.21433
- CliffCompaction (2026): compaction pipelines on Terminal-Bench — https://arxiv.org/abs/2609.26779
- JAATO gc_hybrid / gc_budget strategies — https://jaato-framework-and-examples.github.io/jaato/web/enterprise/gc-system.html

Gap: no side-by-side comparison of classic GC algorithms on the same agent, tasks, and metrics.

## Problems the GC should address (and how we see them)
| Problem | What it is | Measured by |
|---|---|---|
| Context rot | accuracy drops as input grows, even on trivial tasks; semantically similar distractors make it worse (Chroma 2025) | success vs. max_context_tokens; needle tasks include similar distractors |
| Lost in the middle | facts in the middle are recalled worse than at start/end (Liu et al. 2023) | needle position as a task parameter |
| Context poisoning | an early wrong fact/hallucination keeps being referenced | tasks with an injected wrong tool output that is later corrected; did the agent use the correction? |
| Context distraction | agent repeats past actions instead of reasoning | repeated identical tool calls per run |
| Context confusion | too many tools / irrelevant output steers decisions | fixed small toolset; tool-output bloat share = tool tokens / total |
| Cache invalidation | editing earlier context breaks the prompt/KV cache, so "fewer tokens" can still cost more | cached_input_tokens and real cost per run |
| Quadratic cost | every turn re-sends the whole history | cumulative input tokens per turn |

Key tension: every GC that rewrites the middle of the context breaks the cache. Hypothesis worth checking: generational GC fits caching naturally — old generation is a stable cached prefix, only the young tail churns.

## Model of the heap
- Object = one message / tool result, with id, turn, token count, type.
- Roots = system prompt, task goal, latest user turn.
- References = deterministic extraction (ids, file paths, URLs, quoted strings, entity names) linking a message to earlier ones.
- Collected objects become stubs `[collected: msg 12, read_file a.py]`; agent has a `recall(id)` tool. Recalls are logged — a recall means the GC was wrong.

## Strategies
| Name | Rule |
|---|---|
| none | baseline, never collect (fails when window overflows) |
| truncate | baseline, drop oldest |
| lru_mask | stub tool outputs not touched in the last N turns |
| refcount | stub objects with zero incoming refs from live objects |
| mark_sweep | stub objects unreachable from roots |
| generational | young gen collected often (mark-sweep); survivors of K collections promoted to old gen, collected rarely |

## Variables
- trigger: `every_turn` | `threshold(p)` with p ∈ {0.5, 0.7, 0.9} of a fixed budget
- budget: fixed artificial context budget (e.g. 16k tokens) so pressure appears on small tasks

## Tasks (in this order)
1. Needle: agent reads many generated files/tool outputs, must answer facts planted early. Exact scoring.
2. Long chat: scripted multi-turn user with facts introduced and asked about later.
3. Research: doc search over a local corpus with large outputs; scored against a key.

## Stack
Python, plain loop, no agent framework. LLM calls through LiteLLM (one interface for OpenAI, Gemini, Anthropic, etc., plus normalized usage incl. cached tokens and per-call cost). Keys from `.env` (`GEMINI_API_KEY`, `OPENROUTER_API_KEY`).
Model is a variable, always pinned to an exact slug — never `openrouter/free` or `openrouter/auto`, which route to random models and would confound every comparison.
Initial models: `gemini/gemini-2.5-flash` (Gemini API) and two free OpenRouter models with tool calling, e.g. `openrouter/google/gemma-4-31b-it:free` and `openrouter/nvidia/nemotron-3.5-lightning:free`.
Free tiers are rate-limited: the runner retries with backoff and records the provider, so sweeps can run slowly but unattended. Temperature 0, 3 seeds per cell. Context rot differs per model family, so compare strategies within a model, not across.

## Results
`results/runs.jsonl`, one row per run:
`strategy, trigger, budget, task_family, task_id, seed, provider, model, success, turns, input_tokens, cached_input_tokens, output_tokens, cost_usd, gc_runs, tokens_freed, recalls, repeated_calls, max_context_tokens, wall_s`

Full transcripts: `results/traces/<run_id>.json`.
Analysis: `analysis.ipynb` — success vs cost scatter (Pareto), recalls per strategy, context-size-over-time plots.
Observations: `JOURNAL.md`, dated entries.
