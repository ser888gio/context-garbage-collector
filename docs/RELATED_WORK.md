# Related Work and New Experiment Ideas

All arXiv IDs below were checked against the arXiv abstract page. Summaries are one-line paraphrases of the abstracts, not of the full texts.

## Papers

| Paper | Year | Idea | Mapping to GC / this project |
|---|---|---|---|
| [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | 2023 | OS-style virtual context: the model pages information between memory tiers. | Paging / memory hierarchy. Our `recall` is page-in; `manual` is MemGPT's self-managed memory. |
| [Self-GC: Self-Governing Context for Long-Horizon LLM Agents](https://arxiv.org/abs/2607.00692) | 2026 | Context as indexed objects with lifecycles; a planner proposes fold/mask/prune and the harness keeps recoverable storage. | Closest prior art: same heap view. A planner-driven collector vs our regex mark-sweep. |
| [The Complexity Trap](https://arxiv.org/abs/2508.21433) | 2025 | Observation masking matches LLM summarization on SWE agents at about half the cost; a hybrid is cheaper still. | `mask`, `lru_mask`, `hybrid` strategies; H8, H15. |
| [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/abs/2510.00615) | 2025 | Compression guidelines for observations/history are refined from failure analysis. | Learned collector policy. Failure analysis is what `audit.py` already produces. |
| [Agentic Context Engineering (ACE)](https://arxiv.org/abs/2510.04618) | 2025 | Context evolves as a playbook that is incrementally edited instead of rewritten. | Incremental edits instead of wholesale rewrites; relevant to `hybrid`. |
| [MEM1: Learning to Synergize Memory and Reasoning](https://arxiv.org/abs/2506.15841) | 2025 | RL trains the agent to keep a constant-size internal state across turns. | Copying/compacting collector with a single survivor object. |
| [Sleep-time Compute](https://arxiv.org/abs/2504.13171) | 2025 | Do reasoning on a context offline before the query arrives. | Idle-time / concurrent GC: build indexes or summaries between turns. |
| [Mem0](https://arxiv.org/abs/2504.19413) | 2025 | Extract, consolidate and retrieve facts from conversations as an external memory. | Old generation as a fact store instead of raw messages. |
| [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) | 2025 | Memories are linked notes that update older notes when new ones arrive. | Explicit reference graph that is maintained, not regex-derived; write-barrier analogue. |
| [Don't Break the Cache: Prompt Caching for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2601.06007) | 2026 | Compares full-context, system-prompt-only and exclude-tool-results caching on agent runs; reports 41-80% cost savings. | Direct evidence for H6/H7: where a collector cuts determines cache hits. |
| [Prompt Cache: Modular Attention Reuse](https://arxiv.org/abs/2311.04934) | 2023 | Reuse precomputed attention states for recurring prompt segments. | Stable prefix = what `stable_prefix_share` approximates. |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | 2024 | 500 questions on recall, temporal reasoning, knowledge updates and abstention in long chats. | Source for a long-chat task family and a knowledge-update task (extends `poison`). |
| [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo)](https://arxiv.org/abs/2402.17753) | 2024 | Very long multi-session dialogues with QA; models struggle with recall and reasoning. | Template for a "long chat" family (roadmap item 2). |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 2023 | Accuracy drops when the relevant fact is in the middle of a long input. | Needle position sweep (roadmap item 3); also an argument for what to keep at the edges. |
| [RULER](https://arxiv.org/abs/2404.06654) | 2024 | Synthetic tasks beyond simple retrieval show models degrade well before the stated context size. | Context rot: `none` may fail before OOM. Multi-needle and distractor variants. |
| [LLMLingua](https://arxiv.org/abs/2310.05736) | 2023 | Token-level prompt compression, up to about 20x. | Alternative to stubs: partial compaction instead of full collection. |
| [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) | 2023 | KV eviction keeps recent tokens plus the few with the highest accumulated attention. | Frequency + recency. Token-level analogue of `lru_mask`. |
| [Scissorhands](https://arxiv.org/abs/2305.17118) | 2023 | Tokens that mattered earlier keep mattering ("persistence of importance"), so keep them under a budget. | Justifies using past references as a liveness predictor. |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | 2023 | Keep the first tokens (attention sinks) plus a recent window; drop the middle. | `truncate` with a pinned prefix: roots plus a sliding window. |
| [SnapKV](https://arxiv.org/abs/2404.14469) | 2024 | Select per-head prompt positions using the attention of the last few prompt tokens. | Query-aware selection: score old messages against the newest turn. |
| [Quest: Query-Aware Sparsity](https://arxiv.org/abs/2406.10774) | 2024 | Load only the KV pages most relevant to the current query. | Prefetch: bring back stubs that match the current query before the model asks. |

Count: 21 papers.

KV-cache papers act on tokens inside the model; this project acts on messages outside it. The shared idea is eviction under a budget with importance estimated from recency, frequency or the current query.

## New experiment ideas

Ranked by expected insight per effort. Top 3 are marked.

1. **[TOP] Batched collection with hysteresis (cache-aware trigger).** Add a trigger that, once over 70%, collects down to about 40% instead of just under the threshold, so later collections are rare and the prefix stays stable. GC analogy: collect to a low-water mark. Inspired by Don't Break the Cache and Prompt Cache. Hypothesis: fewer `cache_breaks` and lower input cost at equal success; one new trigger function, metrics already exist.

2. **[TOP] Sink + recent + heavy-hitter collector (`h2o`).** Keep the first K messages, the last N turns, and the tool results whose keys the model referenced most often; stub the rest. Analogy: LRU with frequency counts (LFU/ARC). Inspired by H2O, Scissorhands, StreamingLLM. Hypothesis: beats `lru_mask` on multihop where an early file is reused, at similar token cost; one collector function.

3. **[TOP] Stub content ablation.** Since hints turn 0% into ~100%, vary what a stub carries: nothing, keys only (current), keys + first 80 chars, keys + extracted numbers, one-line LLM gloss. Analogy: how much a tombstone or forwarding pointer needs to hold. Inspired by LLMLingua, ACON, Mem0. Hypothesis: keys carry most of the gain; previews add little but cost tokens; one stub formatter and a sweep flag.

4. **Key index message instead of per-stub hints.** Keep one append-only "[index] key -> msg id" message and leave stubs bare. Analogy: symbol table / remembered set. Inspired by Sleep-time Compute (build it between turns). Hypothesis: same success as per-stub hints, fewer tokens, and fewer cache breaks if the index is appended rather than rewritten.

5. **Query-aware prefetch.** After each assistant turn, score stubs by key overlap with the model's last text and restore the top-k automatically. Analogy: hardware prefetch. Inspired by Quest, SnapKV. Hypothesis: lowers `recalls` and turns to answer, but inflates tokens if scoring is loose; one pre-call hook.

6. **Needle position and distractor sweep.** Needle at 10/50/90% of the file list, plus near-duplicate distractor keys (RULER-style). Analogy: none (workload variation). Inspired by Lost in the Middle, RULER. Hypothesis: `none` degrades before OOM, and mark-sweep is sensitive to key collisions (false references); mostly a task-builder change.

7. **Knowledge-update task.** A fact changes during the session and the question asks for the latest value, with the stale one still reachable. Analogy: write barrier / invalidation. Inspired by LongMemEval, A-MEM. Hypothesis: `write_barrier` beats `mark_sweep` only when contradictions are detectable by key; extend the `poison` builder with a numeric-update variant.

8. **Planner collector (LLM picks ids).** One LLM call lists message ids to fold or mask, given the stub index. Analogy: programmer GC hints vs automatic. Inspired by Self-GC, ACON. Hypothesis: better than regex on plain-word links, but costs a call per collection; compare with `manual` and `hybrid` on cost per success.

9. **Single-state fold (MEM1-style, no RL).** Replace all collected messages with one rolling "state" message rewritten by an LLM each time. Analogy: copying collector with one to-space object. Inspired by MEM1, ACE. Hypothesis: constant context size but exact values (numbers) decay over repeated rewrites vs stubs. Close to `hybrid`, so mostly a variant.

10. **Archive search tool instead of id recall.** Collected messages go to an archive and the model calls `search(query)` rather than `recall(id)`. Analogy: swap space with a content-addressable lookup. Inspired by MemGPT, Mem0. Hypothesis: works when keys are unknown (no hints), reducing the need for hints; needs a tool and a naive keyword index.

11. **Long-chat task family.** Multi-session dialogue with facts scattered across sessions and questions after a long gap. Analogy: old-generation promotion of facts. Inspired by LoCoMo, LongMemEval. Hypothesis: regex references work poorly on prose (few paths or CAPS names), so `lru_mask` and a summary tier gain relative to mark-sweep. Largest effort of the list (new builder, key extraction for prose).
