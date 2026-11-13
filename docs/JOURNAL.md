# Journal

## 2026-11-13 — first real run
Nemotron 3.5 Lightning (free), mark_sweep, needle in the middle, question asked late.
- GC ran twice: context 11.5k → 2.2k tokens. Pipeline works end to end.
- Wrong answer. After the question the model recalled only 2 of ~25 stubs, then said "not found".
- Stubs only say `read_file notes/12.txt`, so the model can't tell which stub holds the fact.
- Idea: add a short hint to each stub (first words, or key names like ORION) and test it as a variant.

## 2026-11-13 — sweep 2, Groq gpt-oss-120b: stub hints decide everything
Needle task, question asked late, 3 seeds, budget 16k, threshold 0.7.
- Without hints every real collector failed (0/3 each); the model recalled blindly (12–25 recalls) and lost the needle.
- With hints (stub lists the keys it held) mark_sweep, refcount, lru_mask, truncate went 3/3; generational 2/3. Oracle 3/3 either way.
- Hints also halved tokens (mark_sweep 181k → 107k) and cut cache breaks from 14–27 to 2–6.
- Cheapest successful collector: refcount + hints (93k input tokens, below the oracle's ~126k — the oracle keeps the needle but the agent still reads every file).
- Caveat: n=3 per cell, one model. Need Gemini/Nemotron and more seeds.
