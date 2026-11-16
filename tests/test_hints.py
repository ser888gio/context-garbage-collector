import pytest


@pytest.fixture(scope="module")
def previews(run_sweep):
    return run_sweep("--hints", "preview,numbers", "--strategies", "mark_sweep")


@pytest.fixture(scope="module")
def corrections(run_sweep):
    return run_sweep("--family", "poison", "--seeds", "0,2", "--strategies", "mark_sweep",
                     "--hints", "keys,fresh,pointer")


def test_preview_stub_carries_keys_and_the_first_twelve_words(previews):
    stub = previews.content(previews.row(hints="preview"), 9)

    assert stub == ('[collected: msg 9 read_file notes/02.txt — keys: 7634, ORION — preview: '
                    '"tunnel circle pillow light kettle circle tunnel pillow letter forest forest distant"; '
                    'call recall(9) to restore]')


def test_numbers_stub_lists_every_four_digit_number_of_the_message(previews):
    stub = previews.content(previews.row(hints="numbers"), 9)

    assert stub == ("[collected: msg 9 read_file notes/02.txt — keys: 7634, ORION — numbers: 7634; "
                    "call recall(9) to restore]")


@pytest.mark.parametrize("hints, stub", [
    ("keys", "[collected: msg 15 read_file notes/05.txt — keys: 7890, ORION; call recall(15) to restore]"),
    ("fresh", "[collected: msg 15 read_file notes/05.txt — keys: ORION; call recall(15) to restore]"),
])
def test_fresh_stub_drops_the_revoked_code_that_keys_stub_still_shows(corrections, hints, stub):
    row = corrections.row(task_id="poison-p0.5-late-n10-s0", hints=hints)

    assert corrections.content(row, 15) == stub


@pytest.mark.parametrize("seed, superseding", [("s0", 23), ("s2", 21)])
def test_poison_pointer_stub_points_at_the_superseding_message(corrections, seed, superseding):
    row = corrections.row(task_id=f"poison-p0.5-late-n10-{seed}", hints="pointer")

    assert corrections.content(row, 15) == (
        f"[collected: msg 15 read_file notes/05.txt — keys: ORION (superseded by msg {superseding}); "
        "call recall(15) to restore]")


@pytest.mark.parametrize("hints, recalls", [("keys", 2), ("fresh", 2), ("pointer", 1)])
def test_pointer_hint_saves_the_recall_of_the_superseded_message(corrections, hints, recalls):
    row = corrections.row(task_id="poison-p0.5-late-n10-s2", hints=hints)

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, recalls)
