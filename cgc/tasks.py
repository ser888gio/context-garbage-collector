import random

WORDS = """river stone quiet window garden paper light morning table bridge forest cloud simple
green market letter yellow honest silver planet engine summer winter careful bottle mountain
little thunder pocket golden distant candle music silent harbor velvet lantern meadow copper
narrow gentle orchard ribbon shadow travel wooden village mirror island autumn basket feather
kettle ladder marble needle pillow saddle tunnel valley wander anchor blanket circle dinner
garage hollow jacket lemon mitten orange parcel rocket sugar tomato violet whistle""".split()
NAMES = ["ORION", "ORIOLE", "ORINOCO", "OREGANO", "ORACLE", "ORCHID", "ORBITAL", "ORPHEUS"]


FACT = "The access code for project {} is {}."


def build(rng, n_files, words_per_file, facts):
    files = {}
    for i in range(n_files):
        body = [rng.choice(WORDS) for _ in range(words_per_file)]
        if i in facts:
            body.insert(words_per_file // 2, f". {facts[i]} ")
        files[f"notes/{i:02d}.txt"] = " ".join(body)
    return files


def task(family, seed, n_files, position, upfront, files, answer, name="ORION", question=None, **extra):
    question = question or f"What is the access code for project {name}? Reply with just the number."
    then = f"Then answer: {question}" if upfront else "When you have read all of them, reply with exactly DONE."
    return {
        "family": family,
        "id": f"{family}-p{position}-{'up' if upfront else 'late'}-n{n_files}-s{seed}",
        "files": files,
        "prompt": f"The notes/ folder has {n_files} note files. Read every one with read_file, one per call. {then}",
        "question": question,
        "answer": answer,
        "upfront": upfront,
        **extra,
    }


def needle_task(seed, n_files=30, words_per_file=450, position=0.5, upfront=False):
    rng = random.Random(seed)
    names = rng.sample(NAMES, 4)
    codes = rng.sample(range(1000, 10000), 4)
    needle_at = min(n_files - 1, int(position * n_files))
    fact_files = [needle_at] + rng.sample([i for i in range(n_files) if i != needle_at], 3)
    facts = {f: FACT.format(names[j], codes[j]) for j, f in enumerate(fact_files)}
    files = build(rng, n_files, words_per_file, facts)
    return task("needle", seed, n_files, position, upfront, files, str(codes[0]), names[0])


def multihop_task(seed, n_files=30, words_per_file=450, position=0.5, upfront=False):
    rng = random.Random(seed)
    others = rng.sample(NAMES[1:], 2)
    codes = rng.sample(range(1000, 10000), 3)
    link_at = min(n_files - 1, int(position * n_files))
    value_at, *distract_at = rng.sample([i for i in range(n_files) if i != link_at], 3)
    facts = {link_at: "The access code for project ORION is the same as the code for project VEGA.",
             value_at: FACT.format("VEGA", codes[0])}
    facts.update({f: FACT.format(others[j], codes[j + 1]) for j, f in enumerate(distract_at)})
    files = build(rng, n_files, words_per_file, facts)
    return task("multihop", seed, n_files, position, upfront, files, str(codes[0]))


def poison_task(seed, n_files=30, words_per_file=450, position=0.2, upfront=False):
    rng = random.Random(seed)
    others = rng.sample(NAMES[1:], 2)
    old, new, *distract_codes = rng.sample(range(1000, 10000), 4)
    old_at = min(n_files - 2, int(position * n_files))
    new_at = rng.randint(old_at + 1, n_files - 1)
    distract_at = rng.sample([i for i in range(n_files) if i not in (old_at, new_at)], 2)
    facts = {old_at: FACT.format("ORION", old),
             new_at: f"Correction: the access code for project ORION is now {new}; the old code {old} is revoked."}
    facts.update({f: FACT.format(others[j], distract_codes[j]) for j, f in enumerate(distract_at)})
    files = build(rng, n_files, words_per_file, facts)
    return task("poison", seed, n_files, position, upfront, files, str(new), wrong=str(old))


DESCRIPTIONS = ["northern mountain observatory", "southern coastal lighthouse", "eastern river dam",
                "western desert telescope", "central forest ranger post", "island weather station",
                "highland radio relay", "harbor customs depot"]


def paraphrase_task(seed, n_files=30, words_per_file=450, position=0.5, upfront=False):
    rng = random.Random(seed)
    names = rng.sample(NAMES, 4)
    descs = rng.sample(DESCRIPTIONS, 4)
    codes = rng.sample(range(1000, 10000), 4)
    needle_at = min(n_files - 1, int(position * n_files))
    fact_files = [needle_at] + rng.sample([i for i in range(n_files) if i != needle_at], 3)
    facts = {f: f"Project {names[j]} is the {descs[j]}. " + FACT.format(names[j], codes[j]) for j, f in enumerate(fact_files)}
    files = build(rng, n_files, words_per_file, facts)
    question = f"What is the access code for the {descs[0]} project? Reply with just the number."
    return task("paraphrase", seed, n_files, position, upfront, files, str(codes[0]), names[0], question=question)


BUILDERS = {"paraphrase": paraphrase_task, "needle": needle_task, "multihop": multihop_task, "poison": poison_task}
