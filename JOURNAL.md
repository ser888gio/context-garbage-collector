# Journal

## 2026-11-13 — first real run
Nemotron 3.5 Lightning (free), mark_sweep, needle in the middle, question asked late.
- GC ran twice: context 11.5k → 2.2k tokens. Pipeline works end to end.
- Wrong answer. After the question the model recalled only 2 of ~25 stubs, then said "not found".
- Stubs only say `read_file notes/12.txt`, so the model can't tell which stub holds the fact.
- Idea: add a short hint to each stub (first words, or key names like ORION) and test it as a variant.
